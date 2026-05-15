"""
Build a unified per-run dataset across AA evaluations and ECI log_viewer
summaries for the two benchmarks where both sources have coverage.

Canonical benchmarks:
  GPQA Diamond:
    AA evaluations  → benchmark="gpqa-diamond"
    log_viewer task → "GPQA Diamond"
  AIME (approximate cross-source — different exams, same style):
    AA evaluations  → benchmark="aime-2025" (the real 2025 AIME)
    log_viewer task → "OTIS Mock AIME 2024-2025" (OTIS mock test of similar
                       difficulty; combined here as the closest available pair)

Each output row is one (model, run) measurement and carries a `source` column
so downstream plots can colour by source if desired.

Token semantics: AA's `total_output_tokens` already includes both reasoning
and answer tokens. For log_viewer rows we follow the same convention as
eci_from_log_viewer.py — when a provider reports reasoning tokens in parallel
with rather than as a subset of `total_output_tokens` (Google / xAI /
OpenAI reasoning models), the recorded `total_output_tokens` is
output+reasoning. So the unified `total_inference_tokens` column is comparable
across both sources.

Output:
  data/combined_eci_aa/aa_and_log_viewer.csv
"""

import json
from glob import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AA_EVAL_CSV = ROOT / "data" / "artificial_analysis" / "aa_evaluations_combined.csv"
AA_STATS_CSV = ROOT / "data" / "artificial_analysis" / "artificial_analysis_llm_stats.csv"
LOG_VIEWER_DIR = ROOT / "data" / "eci" / "log_viewer_summary"
OUT_DIR = ROOT / "data" / "combined_eci_aa"

# Canonical benchmark -> (AA benchmark slug, log_viewer task name)
BENCHMARK_MAP = {
    "GPQA Diamond": ("gpqa-diamond", "GPQA Diamond"),
    "AIME": ("aime-2025", "OTIS Mock AIME 2024-2025"),
}

OUT_COLUMNS = [
    "canonical_benchmark", "source", "model", "model_slug",
    "company_slug", "release_date",
    "score_raw", "reasoning_tokens", "answer_tokens", "total_inference_tokens",
]


def _slug_release_lookup() -> dict[str, str]:
    stats = pd.read_csv(AA_STATS_CSV, usecols=["slug", "release_date"])
    stats = stats.dropna(subset=["slug", "release_date"])
    return dict(zip(stats["slug"], stats["release_date"]))


def _slug_company_lookup() -> dict[str, str]:
    stats = pd.read_csv(AA_STATS_CSV, usecols=["slug", "company_slug"])
    stats = stats.dropna(subset=["slug"])
    return dict(zip(stats["slug"], stats["company_slug"].fillna("")))


def _load_aa_rows(slug_to_date: dict[str, str], slug_to_company: dict[str, str]) -> pd.DataFrame:
    ev = pd.read_csv(AA_EVAL_CSV)
    aa_to_canonical = {aa: c for c, (aa, _lv) in BENCHMARK_MAP.items()}
    ev = ev[ev["benchmark"].isin(aa_to_canonical)].copy()
    ev = ev.dropna(subset=["score_raw", "total_output_tokens", "model_slug"])
    ev = ev[ev["total_output_tokens"] > 0]

    ev["canonical_benchmark"] = ev["benchmark"].map(aa_to_canonical)
    ev["source"] = "aa"
    ev["release_date"] = ev["model_slug"].map(slug_to_date)
    ev["company_slug"] = ev["model_slug"].map(slug_to_company)
    ev["reasoning_tokens"] = pd.to_numeric(ev["reasoning_tokens"], errors="coerce")
    ev["answer_tokens"] = pd.to_numeric(ev["answer_tokens"], errors="coerce")
    ev["total_inference_tokens"] = pd.to_numeric(ev["total_output_tokens"], errors="coerce")

    return ev[OUT_COLUMNS]


def _log_viewer_tokens(totals: dict) -> tuple[float | None, float | None, float | None]:
    """(reasoning, answer, total_inference) following the parallel/subset
    convention from eci_from_log_viewer.py."""
    raw_output = totals.get("total_output_tokens")
    raw_reasoning = totals.get("total_reasoning_tokens")
    if raw_output is None:
        return None, None, None
    if raw_reasoning is not None and raw_output < raw_reasoning:
        # parallel counters — output is answer-only
        return raw_reasoning, raw_output, raw_output + raw_reasoning
    return raw_reasoning, raw_output - (raw_reasoning or 0), raw_output


def _load_log_viewer_rows() -> pd.DataFrame:
    lv_to_canonical = {lv: c for c, (_aa, lv) in BENCHMARK_MAP.items()}
    rows = []
    for p in sorted(glob(str(LOG_VIEWER_DIR / "*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        task = d.get("task")
        canonical = lv_to_canonical.get(task)
        if canonical is None:
            continue
        score = d.get("accuracy")
        if score is None:
            continue
        reasoning_tokens, answer_tokens, total_tokens = _log_viewer_tokens(d.get("totals") or {})
        if total_tokens is None or total_tokens <= 0:
            continue

        model_id = d.get("model") or ""
        fallback_name = model_id.rsplit("/", 1)[-1] if model_id else ""
        rows.append({
            "canonical_benchmark": canonical,
            "source": "log_viewer",
            "model": d.get("aa_name") or fallback_name,
            "model_slug": d.get("aa_model_slug") or "",
            "company_slug": d.get("company_slug") or "",
            "release_date": d.get("release_date"),
            "score_raw": float(score),
            "reasoning_tokens": reasoning_tokens,
            "answer_tokens": answer_tokens,
            "total_inference_tokens": total_tokens,
        })
    return pd.DataFrame(rows, columns=OUT_COLUMNS)


def main() -> None:
    slug_to_date = _slug_release_lookup()
    slug_to_company = _slug_company_lookup()

    aa = _load_aa_rows(slug_to_date, slug_to_company)
    lv = _load_log_viewer_rows()
    combined = pd.concat([aa, lv], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "aa_and_log_viewer.csv"
    combined.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")
    print()
    print(
        combined.groupby(["canonical_benchmark", "source"])
        .size()
        .rename("n")
        .to_string()
    )


if __name__ == "__main__":
    main()
