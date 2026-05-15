"""
Annotate data/derived_eci/eci_from_benchmarks.csv with a `parameters` column
sourced from data/eci/epoch_all_ai_models.csv (Parameters column).

Matching strategy, in priority order:
  1. Exact slug match (eci row slug ↔ slugified epoch Model)
  2. Slugified model-name match after stripping trailing parenthetical
     config tags like "(Non-reasoning)", "(high)", "(xhigh)", etc.
  3. Fuzzy match (rapidfuzz WRatio, threshold 92) on the cleaned name

Overwrites the CSV in place. Unmatched models are printed at the end.
"""

import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[3]
ECI_CSV = ROOT / "data" / "derived_eci" / "eci_from_benchmarks.csv"
EPOCH_CSV = ROOT / "data" / "eci" / "epoch_all_ai_models.csv"

FUZZY_THRESHOLD = 92


def slugify(name: str) -> str:
    s = str(name).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def strip_config_suffix(name: str) -> str:
    """Drop trailing "(...)" tags like "(Non-reasoning)", "(high)", "(xhigh)"."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(name)).strip()


def build_epoch_lookup() -> tuple[dict[str, float], dict[str, str]]:
    """Return (slug → parameters, slug → epoch model name) maps.

    When multiple Epoch rows share the same slug, keep the one with the highest
    parameter count (the rest are usually null or partial entries).
    """
    ep = pd.read_csv(EPOCH_CSV, usecols=["Model", "Parameters"])
    ep = ep.dropna(subset=["Model"])
    ep["slug"] = ep["Model"].map(slugify)
    ep = ep.sort_values("Parameters", ascending=False, na_position="last")
    ep = ep.drop_duplicates(subset=["slug"], keep="first")
    return (
        dict(zip(ep["slug"], ep["Parameters"])),
        dict(zip(ep["slug"], ep["Model"])),
    )


def resolve_parameters(
    row_slug: str,
    row_model: str,
    slug_to_params: dict[str, float],
    epoch_slugs: list[str],
) -> tuple[float | None, str]:
    """Return (parameter_count, match_method).

    Method values:
      slug / name / fuzzy:<x> — Epoch row found with a numeric Parameters value
      no_params:<method>     — Epoch row found but Parameters is NaN
      unmatched              — no Epoch row found at all
    """
    candidates: list[tuple[str, str]] = []
    if row_slug and row_slug in slug_to_params:
        candidates.append((row_slug, "slug"))

    cleaned = strip_config_suffix(row_model)
    name_slug = slugify(cleaned)
    if name_slug and name_slug in slug_to_params and name_slug != row_slug:
        candidates.append((name_slug, "name"))

    if not candidates and cleaned:
        match = process.extractOne(
            cleaned,
            epoch_slugs,
            scorer=fuzz.WRatio,
            processor=lambda s: s.replace("-", " "),
            score_cutoff=FUZZY_THRESHOLD,
        )
        if match is not None:
            candidates.append((match[0], f"fuzzy:{match[0]}"))

    for matched_slug, method in candidates:
        p = slug_to_params[matched_slug]
        if pd.notna(p):
            return float(p), method

    if candidates:
        return None, f"no_params:{candidates[0][1]}"
    return None, "unmatched"


def main() -> None:
    df = pd.read_csv(ECI_CSV)
    slug_to_params, slug_to_name = build_epoch_lookup()
    epoch_slugs = list(slug_to_params)

    # Cache per unique (slug, model) to avoid recomputing fuzzy matches
    cache: dict[tuple[str, str], tuple[float | None, str]] = {}
    params: list[float | None] = []
    for _, row in df.iterrows():
        key = (str(row.get("slug") or ""), str(row.get("model") or ""))
        if key not in cache:
            cache[key] = resolve_parameters(key[0], key[1], slug_to_params, epoch_slugs)
        params.append(cache[key][0])

    df["parameters"] = params
    df.to_csv(ECI_CSV, index=False)

    # Summary
    matched = sum(1 for v in params if v is not None)
    total = len(params)
    print(f"Annotated {ECI_CSV} — {matched}/{total} rows matched ({matched / total:.1%})")

    by_method: dict[str, int] = {}
    for _, m in cache.values():
        bucket = m.split(":", 1)[0]
        by_method[bucket] = by_method.get(bucket, 0) + 1
    print("Match methods (unique models):")
    for m, n in sorted(by_method.items(), key=lambda x: -x[1]):
        print(f"  {m}: {n}")

    truly_unmatched = sorted({
        key[1] for key, (_, m) in cache.items() if m == "unmatched"
    })
    if truly_unmatched:
        print(f"\nNot found in Epoch ({len(truly_unmatched)}):")
        for name in truly_unmatched[:30]:
            print(f"  - {name}")
        if len(truly_unmatched) > 30:
            print(f"  ... and {len(truly_unmatched) - 30} more")


if __name__ == "__main__":
    main()
