"""
Compute ECI scores for models in inference_scaling_tokens_only.csv.

Each (model, x_value) pair is treated as a distinct "model" and projected onto
the ECI scale using pre-fitted benchmark parameters from the full ECI model.

Steps:
1. Fit the full ECI model on eci_benchmarks.csv to get benchmark parameters.
2. Map inference scaling benchmark names to ECI benchmark names.
3. Apply random baseline correction to inference scaling scores.
4. For each (model, x_value), use fit_capabilities_given_benchmarks to estimate
   capability, then convert to ECI scale.

Note: Only benchmarks present in both datasets contribute to the ECI estimate.
Currently this is GPQA diamond and OTIS Mock AIME 2024-2025.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add eci-public to path
ECI_SRC = Path(__file__).resolve().parent.parent.parent / "eci-public" / "src"
sys.path.insert(0, str(ECI_SRC))

from eci.fitting import (
    fit_eci_model,
    fit_capabilities_given_benchmarks,
    compute_eci_scores,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
ECI_BENCHMARKS_FILE = DATA_DIR / "eci_benchmarks.csv"
ISC_FILE = DATA_DIR / "inference_scaling_tokens_only.csv"
OUTPUT_FILE = DATA_DIR / "inference_scaling_eci_scores.csv"

# Map ISC benchmark names -> ECI benchmark names
BENCHMARK_NAME_MAP = {
    "GPQA Diamond": "GPQA diamond",
    "OTIS Mock AIME 2024-2025": "OTIS Mock AIME 2024-2025",
}

# Random baselines from the ECI dataloader (for baseline correction)
RANDOM_BASELINES = {
    "GPQA diamond": 0.25,
    "OTIS Mock AIME 2024-2025": 0.001,
}


def apply_baseline_correction(performance: float, benchmark: str) -> float:
    """Transform raw score from [baseline, 1] to [0, 1]."""
    baseline = RANDOM_BASELINES.get(benchmark, 0.0)
    return (performance - baseline) / (1.0 - baseline)


def main():
    # --- Step 1: Fit the full ECI model on eci_benchmarks.csv ---
    print("Loading ECI benchmark data...")
    eci_df = pd.read_csv(ECI_BENCHMARKS_FILE)

    print(f"Fitting ECI model on {eci_df['model_id'].nunique()} models, "
          f"{eci_df['benchmark_id'].nunique()} benchmarks...")
    model_df, bench_df = fit_eci_model(
        eci_df,
        anchor_benchmark="Winogrande",
        bootstrap_samples=0,  # skip bootstrap for speed
    )

    # Convert to ECI scale
    eci_model_df, eci_bench_df = compute_eci_scores(model_df, bench_df)
    print(f"ECI model fit complete. Score range: "
          f"{eci_model_df['eci'].min():.1f} - {eci_model_df['eci'].max():.1f}")

    # Show the benchmark parameters for the overlapping benchmarks
    overlap_benchmarks = set(BENCHMARK_NAME_MAP.values())
    overlap_bench = eci_bench_df[eci_bench_df["benchmark"].isin(overlap_benchmarks)]
    print(f"\nOverlapping benchmark parameters:")
    for _, row in overlap_bench.iterrows():
        print(f"  {row['benchmark']}: difficulty={row['difficulty']:.3f}, "
              f"discriminability={row['discriminability']:.3f}, "
              f"EDI={row['edi']:.1f}")

    # --- Step 2: Prepare inference scaling data ---
    print(f"\nLoading inference scaling data from {ISC_FILE}...")
    isc_df = pd.read_csv(ISC_FILE)

    # Map benchmark names and filter to overlapping benchmarks
    isc_df["eci_benchmark"] = isc_df["benchmark"].map(BENCHMARK_NAME_MAP)
    isc_matched = isc_df.dropna(subset=["eci_benchmark"]).copy()

    dropped = len(isc_df) - len(isc_matched)
    if dropped > 0:
        unmapped = isc_df[isc_df["eci_benchmark"].isna()]["benchmark"].unique()
        print(f"Dropped {dropped} rows with unmapped benchmarks: {list(unmapped)}")

    print(f"Matched {len(isc_matched)} data points across "
          f"{isc_matched['eci_benchmark'].nunique()} benchmarks")

    # Apply random baseline correction (same as ECI pipeline)
    isc_matched["performance_corrected"] = isc_matched.apply(
        lambda row: apply_baseline_correction(row["performance"], row["eci_benchmark"]),
        axis=1,
    )

    # Clip to valid range
    isc_matched["performance_corrected"] = isc_matched["performance_corrected"].clip(0.0, 1.0)

    # --- Step 3: Create synthetic models for each (model, x_value) ---
    # Group by (model, x_value) and build a DataFrame for fit_capabilities_given_benchmarks
    isc_matched["synthetic_model"] = (
        isc_matched["model"] + " @ " + isc_matched["x_value"].astype(str) + " tokens"
    )

    # Check how many benchmarks each synthetic model has
    bench_counts = isc_matched.groupby("synthetic_model")["eci_benchmark"].nunique()
    print(f"\nBenchmarks per (model, x_value):")
    print(bench_counts.value_counts().sort_index().to_string())

    # Build the input DataFrame for fit_capabilities_given_benchmarks
    # It needs: model_id, benchmark_id, performance, benchmark, Model
    synthetic_models = isc_matched["synthetic_model"].unique()
    syn_model_ids = {m: f"syn_{i}" for i, m in enumerate(synthetic_models)}

    fit_df = pd.DataFrame({
        "model_id": isc_matched["synthetic_model"].map(syn_model_ids),
        "benchmark_id": isc_matched["eci_benchmark"],  # not used by the function
        "performance": isc_matched["performance_corrected"],
        "benchmark": isc_matched["eci_benchmark"],
        "Model": isc_matched["synthetic_model"],
    })

    # Aggregate duplicates (same model+token+benchmark): take max
    fit_df = fit_df.groupby(["model_id", "benchmark"], as_index=False).agg({
        "performance": "max",
        "Model": "first",
    })
    # Re-add benchmark_id (not actually used by the function, but needed as column)
    fit_df["benchmark_id"] = fit_df["benchmark"]

    # --- Step 4: Project onto ECI space ---
    print("\nFitting capabilities for inference scaling models...")
    projected_df = fit_capabilities_given_benchmarks(
        fit_df,
        bench_df,  # raw benchmark params (not ECI-scaled)
        bootstrap_samples=0,
        regularization_strength=0.01,  # lighter reg since we have few data points
    )

    # Convert to ECI scale using the same anchors
    # We need the scaling parameters (a, b) from compute_eci_scores
    anchor_low_cap = model_df.loc[model_df["Model"] == "Claude 3.5 Sonnet", "capability"].iloc[0]
    anchor_high_cap = model_df.loc[model_df["Model"] == "GPT-5", "capability"].iloc[0]
    b_scale = (150.0 - 130.0) / (anchor_high_cap - anchor_low_cap)
    a_scale = 130.0 - b_scale * anchor_low_cap

    projected_df["eci"] = a_scale + b_scale * projected_df["capability"]

    # --- Step 5: Build output ---
    # Parse back the original model name and x_value from the synthetic model name
    projected_df["original_model"] = projected_df["Model"].str.replace(
        r" @ [\d.]+ tokens$", "", regex=True
    )
    projected_df["x_value"] = projected_df["Model"].str.extract(
        r" @ ([\d.]+) tokens$"
    ).astype(float)

    # Merge with original ISC metadata
    isc_meta = isc_df[["model_id", "model", "x_value", "x_unit"]].drop_duplicates(
        subset=["model", "x_value"]
    )
    output = projected_df.merge(
        isc_meta,
        left_on=["original_model", "x_value"],
        right_on=["model", "x_value"],
        how="left",
        suffixes=("_eci", "_isc"),
    )

    # Count benchmarks used per row
    bench_used = fit_df.groupby("model_id")["benchmark"].nunique().reset_index()
    bench_used.columns = ["model_id_eci", "n_benchmarks_used"]
    output = output.merge(bench_used, on="model_id_eci", how="left")

    # Select and order output columns
    output_cols = [
        "model_id_isc", "original_model", "x_value", "x_unit",
        "capability", "eci", "n_benchmarks_used",
    ]
    output_cols = [c for c in output_cols if c in output.columns]
    output = output[output_cols].copy()
    output = output.rename(columns={
        "model_id_isc": "model_id",
        "original_model": "model",
    })
    output = output.sort_values(["model", "x_value"])

    output.to_csv(OUTPUT_FILE, index=False)
    print(f"\nOutput written to: {OUTPUT_FILE}")
    print(f"Total rows: {len(output)}")
    print(f"\nECI score summary by model:")
    for model in output["model"].unique():
        sub = output[output["model"] == model]
        print(f"  {model}: ECI {sub['eci'].min():.1f} - {sub['eci'].max():.1f} "
              f"({len(sub)} token levels, {sub['n_benchmarks_used'].max()} benchmarks)")


if __name__ == "__main__":
    main()
