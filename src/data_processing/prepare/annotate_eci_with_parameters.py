"""
Annotate data/derived_eci/eci_from_benchmarks.csv with extra Epoch-derived columns:

  parameters                    — actual parameter count from epoch_all_ai_models.csv
  chinchilla_optimal_parameters — N_opt computed from training compute via the
                                  Chinchilla scaling law (only populated when
                                  Epoch has a `Training compute (FLOP)` value)

Matching strategy, in priority order:
  1. Exact slug match (eci row slug ↔ slugified epoch Model)
  2. Slugified model-name match after stripping trailing parenthetical
     config tags like "(Non-reasoning)", "(high)", "(xhigh)", etc.
  3. Fuzzy match (rapidfuzz WRatio, threshold 92) on the cleaned name

Overwrites the CSV in place. Unmatched models are printed at the end.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[3]
ECI_CSV = ROOT / "data" / "derived_eci" / "eci_from_benchmarks.csv"
EPOCH_CSV = ROOT / "data" / "eci" / "epoch_all_ai_models.csv"

# Reuse the canonical Chinchilla constants/formula instead of duplicating them
sys.path.insert(0, str(ROOT / "src" / "chinchilla_analysis"))
from _chinchilla import chinchilla_optimal  # noqa: E402

FUZZY_THRESHOLD = 92


def slugify(name: str) -> str:
    s = str(name).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def strip_config_suffix(name: str) -> str:
    """Drop trailing "(...)" tags like "(Non-reasoning)", "(high)", "(xhigh)"."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(name)).strip()


def build_epoch_lookup() -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """Return (slug → parameters, slug → training_compute, slug → epoch model name).

    Rows with the same slugified name are merged field-by-field — the first
    non-null value wins for each of `Parameters` and `Training compute (FLOP)`.
    This way a slug that has parameters in one Epoch row and training compute
    in another still ends up with both populated.
    """
    ep = pd.read_csv(
        EPOCH_CSV,
        usecols=["Model", "Parameters", "Training compute (FLOP)"],
    )
    ep = ep.dropna(subset=["Model"])
    ep["slug"] = ep["Model"].map(slugify)

    def first_non_null(s: pd.Series) -> float:
        non_null = s.dropna()
        return float(non_null.iloc[0]) if not non_null.empty else float("nan")

    grouped = ep.groupby("slug", sort=False).agg(
        parameters=("Parameters", first_non_null),
        training_compute=("Training compute (FLOP)", first_non_null),
        model_name=("Model", "first"),
    )
    return (
        grouped["parameters"].to_dict(),
        grouped["training_compute"].to_dict(),
        grouped["model_name"].to_dict(),
    )


def resolve_match(
    row_slug: str,
    row_model: str,
    epoch_slugs: list[str],
) -> tuple[str | None, str]:
    """Return (matched epoch slug, match method).

    Method values:
      slug / name / fuzzy:<x> — Epoch row located by that strategy
      unmatched               — no Epoch row located
    """
    slug_set = set(epoch_slugs)

    if row_slug and row_slug in slug_set:
        return row_slug, "slug"

    cleaned = strip_config_suffix(row_model)
    name_slug = slugify(cleaned)
    if name_slug and name_slug in slug_set and name_slug != row_slug:
        return name_slug, "name"

    if cleaned:
        match = process.extractOne(
            cleaned,
            epoch_slugs,
            scorer=fuzz.WRatio,
            processor=lambda s: s.replace("-", " "),
            score_cutoff=FUZZY_THRESHOLD,
        )
        if match is not None:
            return match[0], f"fuzzy:{match[0]}"

    return None, "unmatched"


def compute_chinchilla_optimal_n(training_compute: float | None) -> float | None:
    """N_opt under the Chinchilla scaling law, or None if compute is unknown."""
    if training_compute is None or not np.isfinite(training_compute) or training_compute <= 0:
        return None
    n_opt, _ = chinchilla_optimal(np.asarray([training_compute], dtype=float))
    return float(n_opt[0])


def main() -> None:
    df = pd.read_csv(ECI_CSV)
    slug_to_params, slug_to_compute, _ = build_epoch_lookup()
    epoch_slugs = list(slug_to_params)

    # Cache the match per unique (slug, model) so fuzzy is only called once.
    cache: dict[tuple[str, str], tuple[str | None, str]] = {}
    params_col: list[float | None] = []
    n_opt_col: list[float | None] = []

    for _, row in df.iterrows():
        key = (str(row.get("slug") or ""), str(row.get("model") or ""))
        if key not in cache:
            cache[key] = resolve_match(key[0], key[1], epoch_slugs)

        matched_slug, _ = cache[key]
        if matched_slug is None:
            params_col.append(None)
            n_opt_col.append(None)
            continue

        p = slug_to_params.get(matched_slug)
        params_col.append(float(p) if p is not None and np.isfinite(p) else None)

        c = slug_to_compute.get(matched_slug)
        n_opt_col.append(compute_chinchilla_optimal_n(c))

    df["parameters"] = params_col
    df["chinchilla_optimal_parameters"] = n_opt_col
    df.to_csv(ECI_CSV, index=False)

    # ---------------- Summary ----------------
    total = len(df)
    n_params = sum(1 for v in params_col if v is not None)
    n_chin = sum(1 for v in n_opt_col if v is not None)
    print(f"Annotated {ECI_CSV}")
    print(f"  rows with parameters:                   {n_params}/{total} ({n_params / total:.1%})")
    print(f"  rows with chinchilla_optimal_parameters: {n_chin}/{total} ({n_chin / total:.1%})")

    # Per-unique-model breakdown of match method
    by_method: dict[str, int] = {}
    for _, m in cache.values():
        bucket = m.split(":", 1)[0]
        by_method[bucket] = by_method.get(bucket, 0) + 1
    print("\nMatch methods (unique (slug, model) pairs):")
    for m, n in sorted(by_method.items(), key=lambda x: -x[1]):
        print(f"  {m}: {n}")

    truly_unmatched = sorted({
        key[1] for key, (matched, _) in cache.items() if matched is None
    })
    if truly_unmatched:
        print(f"\nNot found in Epoch ({len(truly_unmatched)}):")
        for name in truly_unmatched[:30]:
            print(f"  - {name}")
        if len(truly_unmatched) > 30:
            print(f"  ... and {len(truly_unmatched) - 30} more")


if __name__ == "__main__":
    main()
