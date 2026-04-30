"""
Aggregate benchmark CSVs from Epoch AI into a single CSV with one row per reasoning model.

Reasoning models are identified by suffixes like _low, _medium, _high, _xhigh, _max
in their "Model version" field. For each base model + benchmark combination,
the best score across reasoning variants is kept.
"""

import os
import re
import pandas as pd
import numpy as np
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

# Files to skip (not individual model benchmarks or meta-files)
SKIP_FILES = {"epoch_capabilities_index.csv"}


def detect_score_column(df: pd.DataFrame) -> str | None:
    """Find the primary score column in a benchmark CSV."""
    for col in df.columns:
        if col.lower() in NON_SCORE_COLUMNS:
            continue
        # Check if the column has numeric data
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
    # Keep relevant columns
    cols_to_keep = []
    for col in ["Model version", "Release date", "Organization", "Country",
                 "Model name", "Display name", "Training compute (FLOP)"]:
        if col in df.columns:
            cols_to_keep.append(col)
    return df[cols_to_keep]


def process_benchmark_file(filepath: Path) -> pd.DataFrame | None:
    """
    Process a single benchmark CSV. Returns a DataFrame with columns:
    [base_model, reasoning_level, score, release_date, organization, country,
     training_compute, model_version_raw]
    or None if no reasoning models found.
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

    # Convert score to numeric
    df["_score"] = pd.to_numeric(df[score_col], errors="coerce")

    # Filter for reasoning models only
    df["_reasoning_level"] = df["Model version"].astype(str).apply(extract_reasoning_level)
    reasoning_df = df[df["_reasoning_level"].notna()].copy()

    if reasoning_df.empty:
        return None

    reasoning_df["_base_model"] = reasoning_df["Model version"].astype(str).apply(get_base_model)

    # Drop rows with no valid score
    reasoning_df = reasoning_df.dropna(subset=["_score"])
    if reasoning_df.empty:
        return None

    # For each base model, pick the best score
    best_idx = reasoning_df.groupby("_base_model")["_score"].idxmax()
    best_df = reasoning_df.loc[best_idx.dropna()].copy()

    result = pd.DataFrame({
        "base_model": best_df["_base_model"].values,
        "reasoning_level": best_df["_reasoning_level"].values,
        "score": best_df["_score"].values,
        "model_version_raw": best_df["Model version"].values,
    })

    # Carry over metadata columns if available
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

    # Load model metadata for display names
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
            print(f"  {benchmark_name}: {len(result)} reasoning models, score column = '{result['score_column'].iloc[0]}'")
        else:
            print(f"  {benchmark_name}: no reasoning models found")

    if not benchmark_scores:
        print("No benchmark data found for reasoning models.")
        return

    # Build the unified metadata: for each base_model, collect release_date,
    # organization, country, training_compute from whichever benchmark has it
    all_models = set()
    for bname, bdf in benchmark_scores.items():
        all_models.update(bdf["base_model"].tolist())

    # Collect metadata across all benchmarks
    model_meta = {}
    for base_model in all_models:
        meta = {"base_model": base_model}
        for bdf in benchmark_scores.values():
            row = bdf[bdf["base_model"] == base_model]
            if row.empty:
                continue
            row = row.iloc[0]
            for field in ["release_date", "organization", "country", "training_compute_flop"]:
                if field in row.index and pd.notna(row[field]) and row[field] != "":
                    if field not in meta or meta[field] == "" or pd.isna(meta.get(field)):
                        meta[field] = row[field]
            if "reasoning_level" not in meta:
                meta["best_reasoning_level"] = row["reasoning_level"]
        model_meta[base_model] = meta

    # Try to get display names from ECI
    if not meta_df.empty:
        for base_model in all_models:
            eci_rows = meta_df[meta_df["Model version"].astype(str).apply(get_base_model) == base_model]
            if not eci_rows.empty:
                row = eci_rows.iloc[0]
                display = None
                if "Display name" in row.index and pd.notna(row["Display name"]) and row["Display name"] != "":
                    display = row["Display name"]
                elif "Model name" in row.index and pd.notna(row["Model name"]) and row["Model name"] != "":
                    display = row["Model name"]
                if display:
                    # Strip any reasoning suffix from display name
                    display = re.sub(r"\s*\((low|medium|high|xhigh|max|\d+[kK]\s*thinking)\)$", "", display)
                    model_meta[base_model]["display_name"] = display
                # Also fill metadata from ECI
                for eci_col, meta_key in [
                    ("Release date", "release_date"),
                    ("Organization", "organization"),
                    ("Country", "country"),
                    ("Training compute (FLOP)", "training_compute_flop"),
                ]:
                    if eci_col in row.index and pd.notna(row[eci_col]) and row[eci_col] != "":
                        if meta_key not in model_meta[base_model] or pd.isna(model_meta[base_model].get(meta_key)):
                            model_meta[base_model][meta_key] = row[eci_col]

    # Build the pivot table
    meta_rows = pd.DataFrame(model_meta.values())

    # Create score columns: one per benchmark
    benchmark_names = sorted(benchmark_scores.keys())
    score_units = {}  # benchmark_name -> score column name (unit)

    for bname in benchmark_names:
        bdf = benchmark_scores[bname]
        score_units[bname] = bdf["score_column"].iloc[0]
        score_map = dict(zip(bdf["base_model"], bdf["score"]))
        meta_rows[bname] = meta_rows["base_model"].map(score_map)

    # Count how many benchmarks each model has scores for
    meta_rows["num_benchmarks"] = meta_rows[benchmark_names].notna().sum(axis=1)

    # Build final column order
    id_cols = ["display_name", "base_model", "release_date", "organization",
               "country", "training_compute_flop", "best_reasoning_level", "num_benchmarks"]
    # Only include columns that exist
    id_cols = [c for c in id_cols if c in meta_rows.columns]

    final_cols = id_cols + benchmark_names
    output_df = meta_rows[final_cols].copy()

    # Sort by release date (most recent first), then by name
    if "release_date" in output_df.columns:
        output_df["_sort_date"] = pd.to_datetime(output_df["release_date"], errors="coerce")
        output_df = output_df.sort_values("_sort_date", ascending=False, na_position="last")
        output_df = output_df.drop(columns=["_sort_date"])

    # Rename benchmark columns to include the unit
    rename_map = {bname: f"{bname} ({score_units[bname]})" for bname in benchmark_names}
    output_df = output_df.rename(columns=rename_map)

    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nOutput written to: {OUTPUT_FILE}")
    print(f"Total reasoning models: {len(output_df)}")
    print(f"Total benchmarks: {len(benchmark_names)}")


if __name__ == "__main__":
    main()
