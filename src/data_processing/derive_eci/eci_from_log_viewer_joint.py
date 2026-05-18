"""
Estimate one ECI per model from all log_viewer benchmarks (joint fit)
=====================================================================

Like eci_from_log_viewer.py, but produces a *single* ECI estimate per model
by jointly fitting all of that model's benchmark scores against the fixed
benchmark difficulty/discriminability params from the public ECI fit.

Pipeline:
  1. Collect (model, benchmark, accuracy) rows from log_viewer JSONs.
  2. Apply random-baseline correction and (Model, benchmark) max-aggregation
     to match eci-public's preprocessing.
  3. Call eci-public's `fit_capabilities_given_benchmarks` to estimate one
     capability per model.
  4. Scale capability -> ECI via cached (a, b) from the public anchors.

Output: data/derived_eci/eci_per_model_joint.csv (one row per model_id).
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "data_processing" / "derive_eci"))
sys.path.insert(0, str(ROOT / "eci-public" / "src"))

from eci.dataloader import RANDOM_BASELINES  # noqa: E402
from eci.fitting import fit_capabilities_given_benchmarks  # noqa: E402

from eci_from_log_viewer import (  # noqa: E402
    OUT_DIR,
    collect_log_viewer_rows,
    get_eci_params,
)


OUT_CSV = OUT_DIR / "eci_per_model_joint.csv"


def prepare_fit_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Shape log_viewer rows into the columns eci-public's fitter expects.

    Mirrors eci-public's preprocessing: baseline-correct, clip, then
    aggregate duplicate (model_id, benchmark) rows by max performance.
    """
    df = raw.copy()
    df = df.dropna(subset=["accuracy", "model_id"])

    baselines = df["eci_benchmark"].map(RANDOM_BASELINES).fillna(0.0)
    df["performance"] = (df["accuracy"] - baselines) / (1.0 - baselines)
    df["performance"] = df["performance"].clip(0.0, 1.0)

    # eci-public aggregates duplicate (Model, benchmark) by max
    df = (
        df.sort_values("performance", ascending=False)
        .drop_duplicates(subset=["model_id", "eci_benchmark"], keep="first")
        .reset_index(drop=True)
    )

    df = df.rename(columns={"eci_benchmark": "benchmark"})
    # fit_capabilities_given_benchmarks needs model_id, benchmark_id,
    # performance, benchmark, Model. We synthesize benchmark_id = benchmark
    # and use aa_name (fallback to model_id) as the human-readable Model.
    df["benchmark_id"] = df["benchmark"]
    df["Model"] = df["aa_name"].where(df["aa_name"].astype(bool), df["model_id"])
    return df


def main():
    bench_df, a, b = get_eci_params()

    raw = collect_log_viewer_rows()
    print(
        f"Collected {len(raw)} log_viewer rows across "
        f"{raw['task'].nunique()} tasks, {raw['model_id'].nunique()} models"
    )

    fit_df = prepare_fit_frame(raw)
    print(
        f"After baseline correction + max-aggregation: {len(fit_df)} rows, "
        f"{fit_df['model_id'].nunique()} models, "
        f"{fit_df['benchmark'].nunique()} benchmarks"
    )

    cap_df = fit_capabilities_given_benchmarks(
        fit_df[["model_id", "benchmark_id", "performance", "benchmark", "Model"]],
        bench_df,
        bootstrap_samples=0,
    )
    cap_df["eci_estimated"] = a + b * cap_df["capability"]

    # Per-(model, benchmark) mean token usage across runs, then sum across
    # the benchmarks that contributed to the joint fit.
    per_bench_tokens = (
        raw.groupby(["model_id", "eci_benchmark"], as_index=False)
        .agg(
            reasoning_tokens=("reasoning_tokens", "mean"),
            answer_tokens=("answer_tokens", "mean"),
            total_output_tokens=("total_output_tokens", "mean"),
        )
    )
    tokens = (
        per_bench_tokens.groupby("model_id", as_index=False)
        .agg(
            total_reasoning_tokens=("reasoning_tokens", "sum"),
            total_answer_tokens=("answer_tokens", "sum"),
            total_inference_tokens=("total_output_tokens", "sum"),
        )
    )

    # Per-model benchmark inventory for transparency
    summary = (
        fit_df.groupby("model_id")
        .agg(
            n_benchmarks=("benchmark", "nunique"),
            benchmarks_used=("benchmark", lambda s: ", ".join(sorted(set(s)))),
            mean_accuracy=("accuracy", "mean"),
            aa_name=("aa_name", "first"),
            aa_model_slug=("aa_model_slug", "first"),
            company_slug=("company_slug", "first"),
        )
        .reset_index()
    )

    out = cap_df.merge(summary, on="model_id", how="left").merge(
        tokens, on="model_id", how="left"
    )
    out = out[[
        "model_id",
        "aa_name",
        "aa_model_slug",
        "company_slug",
        "eci_estimated",
        "capability",
        "n_benchmarks",
        "benchmarks_used",
        "mean_accuracy",
        "total_reasoning_tokens",
        "total_answer_tokens",
        "total_inference_tokens",
    ]].sort_values("eci_estimated", ascending=False).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(out)} models)")
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
