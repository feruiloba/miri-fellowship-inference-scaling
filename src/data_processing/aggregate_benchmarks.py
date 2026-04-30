"""
Aggregate benchmark CSVs from Epoch AI into a single CSV with one row per
(base model, reasoning level) pair.

Reasoning models are identified by suffixes like _low, _medium, _high, _xhigh, _max
in their "Model version" field. Each reasoning level gets its own row, so a model
like o3 can appear as o3 (low), o3 (medium), o3 (high), etc.

If a model has multiple scores for the same benchmark at the same reasoning level,
the best score is kept.
"""

import re
import pandas as pd
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "benchmark_data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "aggregated_reasoning_models.csv"

# Reasoning-level suffixes to detect and strip
REASONING_SUFFIXES = re.compile(
    r"_(low|medium|high|xhigh|max|unknown|\d+K)$", re.IGNORECASE
)

# Columns that are never scores
NON_SCORE_COLUMNS = {
    "model version", "release date", "organization", "country",
    "training compute (flop)", "training compute notes", "stderr",
    "log viewer", "logs", "started at", "id", "name", "notes",
    "source", "source link", "shots", "confidence", "model name",
    "description", "display name", "model accessibility",
    "scaffold", "agent", "tools", "agent org", "model org",
    "run date", "created", "evaluation date",
}

# Files to skip (meta-files, not individual benchmarks)
SKIP_FILES = {"epoch_capabilities_index.csv"}


def detect_score_column(df: pd.DataFrame) -> str | None:
    """Find the primary score column in a benchmark CSV."""
    for col in df.columns:
        if col.lower() in NON_SCORE_COLUMNS:
            continue
        numeric_vals = pd.to_numeric(df[col], errors="coerce")
        if numeric_vals.notna().sum() > 0:
            return col
    return None


def extract_reasoning_level(model_version: str) -> str | None:
    """Extract the reasoning level suffix from a model version string."""
    if not isinstance(model_version, str):
        return None
    match = REASONING_SUFFIXES.search(model_version)
    return match.group(1) if match else None


def get_base_model(model_version: str) -> str:
    """Strip the reasoning suffix to get the base model identifier."""
    if not isinstance(model_version, str):
        return str(model_version)
    return REASONING_SUFFIXES.sub("", model_version)


def get_benchmark_name(filename: str) -> str:
    """Derive a clean benchmark name from the filename."""
    name = filename.replace(".csv", "")
    name = name.replace("_external", "")
    return name


def load_model_metadata() -> pd.DataFrame:
    """Load model metadata from the ECI file for display names, dates, etc."""
    eci_path = BENCHMARK_DIR / "epoch_capabilities_index.csv"
    if not eci_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(eci_path, on_bad_lines="skip")
    cols_to_keep = [
        col for col in ["Model version", "Release date", "Organization", "Country",
                        "Model name", "Display name", "Training compute (FLOP)"]
        if col in df.columns
    ]
    return df[cols_to_keep]


def process_benchmark_file(filepath: Path) -> pd.DataFrame | None:
    """
    Process a single benchmark CSV. Returns a DataFrame with columns:
    [base_model, reasoning_level, score, release_date, organization, country,
     training_compute_flop, model_version_raw, score_column]
    Keeps every (base_model, reasoning_level) pair, picking the best score
    when there are duplicates within the same pair.
    """
    try:
        df = pd.read_csv(filepath, on_bad_lines="skip")
    except Exception as e:
        print(f"  Warning: Could not read {filepath.name}: {e}")
        return None

    if "Model version" not in df.columns:
        print(f"  Warning: No 'Model version' column in {filepath.name}")
        return None

    score_col = detect_score_column(df)
    if score_col is None:
        print(f"  Warning: No score column found in {filepath.name}")
        return None

    df["_score"] = pd.to_numeric(df[score_col], errors="coerce")
    df["_reasoning_level"] = df["Model version"].astype(str).apply(extract_reasoning_level)

    reasoning_df = df[df["_reasoning_level"].notna()].copy()
    if reasoning_df.empty:
        return None

    reasoning_df["_base_model"] = reasoning_df["Model version"].astype(str).apply(get_base_model)

    # Drop rows with no valid score
    reasoning_df = reasoning_df.dropna(subset=["_score"])
    if reasoning_df.empty:
        return None

    # For each (base_model, reasoning_level), pick the best score
    best_idx = reasoning_df.groupby(["_base_model", "_reasoning_level"])["_score"].idxmax()
    best_df = reasoning_df.loc[best_idx.dropna()].copy()

    result = pd.DataFrame({
        "base_model": best_df["_base_model"].values,
        "reasoning_level": best_df["_reasoning_level"].values,
        "score": best_df["_score"].values,
        "model_version_raw": best_df["Model version"].values,
    })

    for col, out_col in [
        ("Release date", "release_date"),
        ("Organization", "organization"),
        ("Country", "country"),
        ("Training compute (FLOP)", "training_compute_flop"),
    ]:
        if col in best_df.columns:
            result[out_col] = best_df[col].values

    result["score_column"] = score_col
    return result


def main():
    print(f"Reading benchmarks from: {BENCHMARK_DIR}")

    meta_df = load_model_metadata()

    # Process all benchmark CSVs
    benchmark_scores = {}  # benchmark_name -> DataFrame
    csv_files = sorted(BENCHMARK_DIR.glob("*.csv"))

    for filepath in csv_files:
        if filepath.name in SKIP_FILES:
            continue

        benchmark_name = get_benchmark_name(filepath.name)
        result = process_benchmark_file(filepath)

        if result is not None:
            benchmark_scores[benchmark_name] = result
            n_pairs = len(result)
            print(f"  {benchmark_name}: {n_pairs} (model, level) pairs, score column = '{result['score_column'].iloc[0]}'")
        else:
            print(f"  {benchmark_name}: no reasoning models found")

    if not benchmark_scores:
        print("No benchmark data found for reasoning models.")
        return

    # Collect all unique (base_model, reasoning_level) pairs across benchmarks
    all_pairs = set()
    for bdf in benchmark_scores.values():
        for _, row in bdf.iterrows():
            all_pairs.add((row["base_model"], row["reasoning_level"]))

    # Build metadata for each (base_model, reasoning_level)
    rows = []
    for base_model, reasoning_level in all_pairs:
        meta = {"base_model": base_model, "reasoning_level": reasoning_level}
        for bdf in benchmark_scores.values():
            match = bdf[(bdf["base_model"] == base_model) & (bdf["reasoning_level"] == reasoning_level)]
            if match.empty:
                continue
            r = match.iloc[0]
            for field in ["release_date", "organization", "country", "training_compute_flop"]:
                if field in r.index and pd.notna(r[field]) and r[field] != "":
                    if field not in meta or meta[field] == "" or pd.isna(meta.get(field)):
                        meta[field] = r[field]
        rows.append(meta)

    meta_rows = pd.DataFrame(rows)

    # Add display names from ECI
    if not meta_df.empty:
        display_cache = {}
        eci_meta_cache = {}
        for _, eci_row in meta_df.iterrows():
            bm = get_base_model(str(eci_row["Model version"]))
            if bm in display_cache:
                continue
            display = None
            if "Display name" in eci_row.index and pd.notna(eci_row["Display name"]) and eci_row["Display name"] != "":
                display = eci_row["Display name"]
            elif "Model name" in eci_row.index and pd.notna(eci_row["Model name"]) and eci_row["Model name"] != "":
                display = eci_row["Model name"]
            if display:
                display = re.sub(r"\s*\((low|medium|high|xhigh|max|\d+[kK]\s*thinking)\)$", "", display)
                display_cache[bm] = display
            eci_meta_cache[bm] = {
                "release_date": eci_row.get("Release date"),
                "organization": eci_row.get("Organization"),
                "country": eci_row.get("Country"),
                "training_compute_flop": eci_row.get("Training compute (FLOP)"),
            }

        meta_rows["display_name"] = meta_rows["base_model"].map(display_cache)

        # Fill missing metadata from ECI
        for idx, row in meta_rows.iterrows():
            bm = row["base_model"]
            if bm in eci_meta_cache:
                for field, val in eci_meta_cache[bm].items():
                    if (field not in row or pd.isna(row.get(field)) or row.get(field) == "") and pd.notna(val) and val != "":
                        meta_rows.at[idx, field] = val

    # Create score columns: one per benchmark
    benchmark_names = sorted(benchmark_scores.keys())
    score_units = {}

    for bname in benchmark_names:
        bdf = benchmark_scores[bname]
        score_units[bname] = bdf["score_column"].iloc[0]
        # Build a lookup keyed on (base_model, reasoning_level)
        score_map = {}
        for _, r in bdf.iterrows():
            score_map[(r["base_model"], r["reasoning_level"])] = r["score"]
        meta_rows[bname] = meta_rows.apply(
            lambda row: score_map.get((row["base_model"], row["reasoning_level"])), axis=1
        )

    # Count how many benchmarks each row has scores for
    meta_rows["num_benchmarks"] = meta_rows[benchmark_names].notna().sum(axis=1)

    # Build final column order
    id_cols = ["display_name", "base_model", "reasoning_level", "release_date",
               "organization", "country", "training_compute_flop", "num_benchmarks"]
    id_cols = [c for c in id_cols if c in meta_rows.columns]

    final_cols = id_cols + benchmark_names
    output_df = meta_rows[final_cols].copy()

    # Sort by release date descending, then base_model, then reasoning level
    level_order = {"low": 0, "medium": 1, "high": 2, "xhigh": 3, "max": 4, "unknown": 5}
    output_df["_sort_date"] = pd.to_datetime(output_df["release_date"], errors="coerce")
    output_df["_sort_level"] = output_df["reasoning_level"].map(
        lambda x: level_order.get(x, 99) if isinstance(x, str) else 99
    )
    output_df = output_df.sort_values(
        ["_sort_date", "base_model", "_sort_level"],
        ascending=[False, True, True],
        na_position="last",
    )
    output_df = output_df.drop(columns=["_sort_date", "_sort_level"])

    # Rename benchmark columns to include the unit
    rename_map = {bname: f"{bname} ({score_units[bname]})" for bname in benchmark_names}
    output_df = output_df.rename(columns=rename_map)

    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nOutput written to: {OUTPUT_FILE}")
    print(f"Total rows (model x reasoning level): {len(output_df)}")
    print(f"Total benchmarks: {len(benchmark_names)}")


if __name__ == "__main__":
    main()
