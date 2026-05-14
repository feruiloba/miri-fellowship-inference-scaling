"""
Scrape token usage + benchmark score from artificialanalysis.ai for every evaluation.

Strategy:
  1. Discover all eval slugs from the RSC endpoint of the AA Intelligence Index
     evaluations page. Fall back to scraping /evaluations with playwright.
  2. For each eval slug, visit /evaluations/{slug}?eval-token-usage=token-usage and
     scrape two charts:
       * "{slug}-benchmark-leaderboard-token-usage" — stacked bar with
         input / reasoning / answer tokens per model.
       * "{slug}-benchmark-leaderboard-results"      — bar with benchmark score
         per model (percentage).
  3. Join on the model display name.
  4. Write one CSV per benchmark + one combined CSV.

Usage:
    python src/data_processing/fetch/fetch_artificial_analysis_evaluations.py
"""

import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from playwright.async_api import async_playwright

BASE = "https://artificialanalysis.ai"
RSC_INDEX_URL = f"{BASE}/evaluations/artificial-analysis-intelligence-index?_rsc=1ctfu"
EVALS_PAGE_URL = f"{BASE}/evaluations"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

OUTPUT_DIR = Path("data/artificial_analysis/evaluations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMBINED_CSV = Path("data/artificial_analysis/aa_evaluations_combined.csv")

# Evals on /evaluations that aren't actually individual benchmarks (no per-eval chart).
SKIP_SLUGS = {
    "artificial-analysis-intelligence-index",
    "artificial-analysis-openness-index",
}

_JS_DIR = Path(__file__).parent
_GET_SVG_CHART_DATA_JS = (_JS_DIR / "get_svg_chart_data.js").read_text()


# ---------------------------------------------------------------------------
# Eval slug discovery
# ---------------------------------------------------------------------------

def discover_eval_slugs_via_rsc() -> list[str]:
    """Fetch the RSC payload of the Intelligence Index eval page and pull eval slugs."""
    headers = {"User-Agent": USER_AGENT, "RSC": "1"}
    r = requests.get(RSC_INDEX_URL, headers=headers, timeout=30)
    r.raise_for_status()
    slugs = sorted(set(re.findall(r"/evaluations/([a-z0-9-]+)", r.text)))
    return [s for s in slugs if s not in SKIP_SLUGS]


def _name_key(name: str) -> str:
    """Lowercase + strip all whitespace — used to match chart axis labels (which
    drop spaces around parens when wrapping) against RSC display names."""
    return re.sub(r"\s+", "", name).lower()


def _iter_balanced_json_objects(text: str):
    """Yield every balanced `{...}` substring in `text` (nested objects included).

    Tracks string state + escapes so braces inside string literals don't fool us.
    """
    starts: list[int] = []
    in_str = False
    esc = False
    for i, c in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            starts.append(i)
        elif c == "}" and starts:
            start = starts.pop()
            yield text[start : i + 1]


def _extract_slug_aliases(rsc_text: str) -> list[tuple[str, str]]:
    """Return (display_alias, slug) pairs from an RSC payload.

    Walks every balanced JSON object embedded in the RSC stream. For each
    object that decodes successfully and has top-level `slug` + `name`/
    `shortName`, emit (name, slug) and (shortName, slug). Because we read the
    name strictly from the same object that owns the slug, neighbor objects
    can't contaminate the mapping.
    """
    out: list[tuple[str, str]] = []
    for obj_str in _iter_balanced_json_objects(rsc_text):
        # Cheap filter: skip anything that obviously isn't a model entry.
        if '"slug":' not in obj_str or '"name":' not in obj_str:
            continue
        try:
            obj = json.loads(obj_str)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        slug = obj.get("slug")
        if not isinstance(slug, str):
            continue
        for key in ("name", "shortName"):
            val = obj.get(key)
            if isinstance(val, str) and val:
                out.append((val, slug))
    return out


def fetch_slug_map_for_eval(eval_slug: str) -> dict[str, str]:
    """Per-eval normalized-name → slug map from that eval's RSC payload.

    `name` is treated as authoritative; `shortName` is only used as a fallback
    when no `name` claimed the same normalized key. This stops a sibling
    model's shortName from clobbering the canonical name → slug pair (e.g.
    'Claude 4 Sonnet (Thinking)' has shortName 'Claude 4 Sonnet', which
    otherwise overwrites the real Claude 4 Sonnet's mapping)."""
    url = f"{BASE}/evaluations/{eval_slug}?_rsc=1ctfu"
    headers = {"User-Agent": USER_AGENT, "RSC": "1"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    name_map: dict[str, str] = {}
    short_map: dict[str, str] = {}
    for obj_str in _iter_balanced_json_objects(r.text):
        if '"slug":' not in obj_str or '"name":' not in obj_str:
            continue
        try:
            obj = json.loads(obj_str)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        slug = obj.get("slug")
        if not isinstance(slug, str):
            continue
        name_val = obj.get("name")
        if isinstance(name_val, str) and name_val:
            name_map[_name_key(name_val)] = slug
        short_val = obj.get("shortName")
        if isinstance(short_val, str) and short_val:
            short_map.setdefault(_name_key(short_val), slug)
    # `name` wins over `shortName`.
    return {**short_map, **name_map}


def fetch_slug_map_combined(eval_slugs: list[str]) -> dict[str, str]:
    """Union slug maps across all evals — each eval's RSC carries that eval's
    full model roster, which is broader than the Intelligence Index RSC.
    Falls back to the AA stats CSV for models that don't appear in any RSC."""
    combined: dict[str, str] = {}
    for s in eval_slugs:
        try:
            combined.update(fetch_slug_map_for_eval(s))
        except Exception as e:
            print(f"[slug-map] {s} failed: {e}")
    csv = Path("data/artificial_analysis/artificial_analysis_llm_stats.csv")
    if csv.exists():
        df = pd.read_csv(csv)
        for _, row in df.dropna(subset=["slug", "name"]).iterrows():
            key = _name_key(row["name"])
            combined.setdefault(key, row["slug"])
    return combined


async def discover_eval_slugs_via_playwright() -> list[str]:
    """Fallback: render /evaluations and harvest links to /evaluations/<slug>."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=USER_AGENT)
        page = await ctx.new_page()
        await page.goto(EVALS_PAGE_URL, wait_until="networkidle", timeout=60_000)
        hrefs = await page.evaluate(
            "() => [...document.querySelectorAll('a[href*=\"/evaluations/\"]')]"
            ".map(a => a.getAttribute('href'))"
        )
        await browser.close()
    slugs = set()
    for h in hrefs:
        m = re.match(r"^/evaluations/([a-z0-9-]+)", h or "")
        if m:
            slugs.add(m.group(1))
    return sorted(s for s in slugs if s not in SKIP_SLUGS)


# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------

def parse_token_count(value: str) -> float | None:
    """Convert strings like '240M', '78M', '1.2B', '500K' to raw numbers."""
    if value is None:
        return None
    value = value.strip()
    mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    m = re.match(r"^([\d.]+)([KMB]?)$", value, re.IGNORECASE)
    if m:
        return float(m.group(1)) * mult.get(m.group(2).upper(), 1)
    try:
        return float(value)
    except ValueError:
        return None


def parse_score(value: str) -> float | None:
    """Convert '37.7%' / '0.45' to a float in [0, 100] (percentage) or as-is if no %."""
    if value is None:
        return None
    value = value.strip()
    m = re.match(r"^([\d.]+)\s*%$", value)
    if m:
        return float(m.group(1))
    try:
        return float(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Chart scraping
# ---------------------------------------------------------------------------

async def get_svg_chart_data(page, section_id: str) -> dict | None:
    """Run the JS extractor. Returns None when the section is missing."""
    data = await page.evaluate(_GET_SVG_CHART_DATA_JS, section_id)
    if not data or (not data["axis_labels"] and not data["bar_rects"]):
        return None
    return data


def _normalize_model(name: str) -> str:
    """Normalize multi-line / whitespace-collapsed model display names for joining."""
    return re.sub(r"\s+", " ", name).strip()


def _group_value_labels_by_bar(value_labels: list[dict]) -> list[list[tuple[float, str]]]:
    """
    Group SVG <text> labels by x-position (= which bar they belong to).
    Returns a list aligned with the chart x-order: each entry is the list of
    (y, text) labels at that bar, sorted top-to-bottom (y ascending).
    """
    groups: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for v in value_labels:
        groups[round(v["x"])].append((v["y"], v["text"]))
    sorted_keys = sorted(groups)
    return [sorted(groups[k], key=lambda yt: yt[0]) for k in sorted_keys]


def parse_token_chart(data: dict) -> pd.DataFrame:
    """
    Parse a stacked input/reasoning/answer token-usage chart.

    For each chart index i, we have up to three rect heights (input, reasoning,
    answer) and a set of text labels at that bar's x-position. The top label
    (smallest y) is the TOTAL; remaining labels annotate individual segments
    when they're large enough to fit text. Heights are used to split the total
    proportionally when a segment has no visible label.
    """
    axis_labels = data["axis_labels"]
    value_labels = data["value_labels"]
    bar_rects = data["bar_rects"]

    heights: dict[str, dict[int, float]] = {
        "input": {}, "reasoning": {}, "answer": {},
    }
    for r in bar_rects:
        t = r["testid"] or ""
        m = re.match(r"bar\.item\.tokenCounts_(\w+)Tokens\.(\d+)$", t)
        if not m:
            continue
        series = m.group(1).lower()
        if series in heights:
            heights[series][int(m.group(2))] = r["height"]

    bar_label_groups = _group_value_labels_by_bar(value_labels)

    rows = []
    for i, model in enumerate(axis_labels):
        inp_h = heights["input"].get(i, 0.0)
        rsn_h = heights["reasoning"].get(i, 0.0)
        ans_h = heights["answer"].get(i, 0.0)
        total_h = inp_h + rsn_h + ans_h

        labels = bar_label_groups[i] if i < len(bar_label_groups) else []
        total_raw = labels[0][1] if labels else None
        total_val = parse_token_count(total_raw)

        if total_h > 0 and total_val is not None:
            # Always derive from proportions for consistency; segment text labels
            # are rounded/abbreviated, so heights give the cleanest split.
            input_val = total_val * (inp_h / total_h) if inp_h > 0 else 0.0
            reasoning_val = total_val * (rsn_h / total_h) if rsn_h > 0 else 0.0
            answer_val = total_val * (ans_h / total_h) if ans_h > 0 else 0.0
        else:
            input_val = reasoning_val = answer_val = None

        rows.append({
            "model": _normalize_model(model),
            "input_tokens": input_val,
            "reasoning_tokens": reasoning_val,
            "answer_tokens": answer_val,
        })
    return pd.DataFrame(rows)


def parse_score_chart(data: dict) -> pd.DataFrame:
    """Parse the results bar chart. One value per model (percentage or raw)."""
    axis_labels = data["axis_labels"]
    bar_label_groups = _group_value_labels_by_bar(data["value_labels"])

    rows = []
    for i, model in enumerate(axis_labels):
        labels = bar_label_groups[i] if i < len(bar_label_groups) else []
        score_raw = labels[0][1] if labels else None
        rows.append({
            "model": _normalize_model(model),
            "score": parse_score(score_raw),
            "score_raw": score_raw,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-eval orchestration
# ---------------------------------------------------------------------------

async def _new_page(p):
    browser = await p.chromium.launch(
        headless=True,
        args=["--disable-blink-features=AutomationControlled"],
    )
    ctx = await browser.new_context(user_agent=USER_AGENT)
    page = await ctx.new_page()
    await page.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
    )
    return browser, page


async def scrape_eval(page, slug: str, models_param: str | None, slug_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Scrape one evaluation page → DataFrame keyed by model."""
    url = f"{BASE}/evaluations/{slug}?eval-token-usage=token-usage"
    if models_param:
        url += f"&models={quote(models_param, safe='')}"

    # Use domcontentloaded — `networkidle` can hang for several minutes on AA pages
    # with the giant `models=` param because of constant background fetches.
    await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
    # Trigger lazy-loaded charts below the fold
    for _ in range(20):
        await page.evaluate("window.scrollBy(0, 800)")
        await page.wait_for_timeout(200)
    # Wait for the token-usage chart to actually exist (with a bounded timeout).
    try:
        await page.wait_for_selector(
            '[id$="-benchmark-leaderboard-token-usage"] rect[data-testid]',
            timeout=30_000,
        )
    except Exception:
        pass  # let downstream chart-resolver decide there's nothing to scrape
    await page.wait_for_timeout(1_000)

    # Resolve actual section ids — sub-benchmarks (e.g. tau2-bench → "bench-telecom-...")
    # use prefixes other than the eval slug, so match by suffix instead.
    section_ids = await page.evaluate("""
        () => {
            const find = (suffix) => {
                const tokenParent = document.getElementById(suffix === 'token-usage' ? 'token-usage' : 'results');
                const scope = tokenParent || document;
                const el = scope.querySelector(`[id$="-benchmark-leaderboard-${suffix}"]`);
                return el ? el.id : null;
            };
            return { token: find('token-usage'), score: find('results') };
        }
    """)
    token_section = section_ids.get("token")
    score_section = section_ids.get("score")

    token_data = await get_svg_chart_data(page, token_section) if token_section else None
    score_data = await get_svg_chart_data(page, score_section) if score_section else None

    if token_data is None and score_data is None:
        print(f"[{slug}] no token-usage or results chart found, skipping.")
        return pd.DataFrame()

    token_df = parse_token_chart(token_data) if token_data else pd.DataFrame(
        columns=["model", "input_tokens", "reasoning_tokens", "answer_tokens"]
    )
    score_df = parse_score_chart(score_data) if score_data else pd.DataFrame(
        columns=["model", "score", "score_raw"]
    )

    # Some eval pages render multiple bars with the same display name (different
    # configs not reflected in the visible label). Keep the first occurrence so
    # the join doesn't produce a cross product.
    token_df = token_df.drop_duplicates(subset="model", keep="first")
    score_df = score_df.drop_duplicates(subset="model", keep="first")
    merged = pd.merge(token_df, score_df, on="model", how="outer")
    merged.insert(0, "benchmark", slug)
    # Resolve model slug by normalized display-name match against the RSC map.
    if slug_map:
        merged.insert(2, "model_slug", merged["model"].map(lambda m: slug_map.get(_name_key(m))))
    # score on 0-1 scale (raw label may be a percentage or already a fraction)
    merged["score_raw"] = merged["score"].apply(
        lambda v: None if pd.isna(v) else (v / 100.0 if v > 1.0 else v)
    )
    merged["total_output_tokens"] = (
        merged["reasoning_tokens"].fillna(0) + merged["answer_tokens"].fillna(0)
    ).where(merged[["reasoning_tokens", "answer_tokens"]].notna().any(axis=1))
    print(f"[{slug}] models: {len(merged)} (tokens: {len(token_df)}, scores: {len(score_df)})")
    return merged


async def scrape_all(slugs: list[str], models_param: str | None, slug_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Run each eval in a fresh browser context to avoid renderer state leaking
    across pages (the big `models=` URL can leave chromium in a bad state)."""
    combined = []
    async with async_playwright() as p:
        for slug in slugs:
            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"],
            )
            ctx = await browser.new_context(user_agent=USER_AGENT)
            page = await ctx.new_page()
            await page.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            )
            try:
                df = await asyncio.wait_for(
                    scrape_eval(page, slug, models_param, slug_map),
                    timeout=600,  # 10-minute hard ceiling per eval
                )
            except asyncio.TimeoutError:
                print(f"[{slug}] TIMEOUT (>10min), skipping")
                await browser.close()
                continue
            except Exception as e:
                print(f"[{slug}] ERROR: {e}")
                await browser.close()
                continue
            finally:
                # browser.close() handled in error paths; close here for success
                pass
            if df.empty:
                await browser.close()
                continue
            out_path = OUTPUT_DIR / f"{slug}.csv"
            df.to_csv(out_path, index=False)
            combined.append(df)
            await browser.close()
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_models_param() -> str | None:
    """
    Build a comma-separated `models=` query value from the AA stats CSV.

    Passing every known slug forces AA to render bars for every model that has
    data on the given eval (instead of the default ~15-30 visible). Chromium
    happily handles URLs of ~15 KB.
    """
    csv = Path("data/artificial_analysis/artificial_analysis_llm_stats.csv")
    if not csv.exists():
        print(f"[warn] {csv} missing — pages will use AA's default model set.")
        return None
    df = pd.read_csv(csv)
    slugs = df["slug"].dropna().unique().tolist()
    print(f"Passing models= with {len(slugs)} slugs (~{sum(len(s) for s in slugs) + len(slugs)*3} chars).")
    return ",".join(slugs)


async def main():
    try:
        slugs = discover_eval_slugs_via_rsc()
        print(f"Discovered {len(slugs)} eval slugs via RSC.")
    except Exception as e:
        print(f"RSC discovery failed ({e}); falling back to playwright.")
        slugs = await discover_eval_slugs_via_playwright()
        print(f"Discovered {len(slugs)} eval slugs via playwright.")

    if not slugs:
        raise SystemExit("No eval slugs discovered.")

    models_param = load_models_param()
    slug_map = fetch_slug_map_combined(slugs)
    print(f"Slug map: {len(slug_map)} display-name → slug entries (unioned across {len(slugs)} evals).")
    print(f"Scraping {len(slugs)} evaluations: {slugs}\n")

    combined = await scrape_all(slugs, models_param, slug_map)
    if combined.empty:
        print("Nothing scraped.")
        return

    COMBINED_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED_CSV, index=False)
    print(f"\nSaved combined → {COMBINED_CSV}  ({len(combined)} rows)")
    print(f"Per-benchmark CSVs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
