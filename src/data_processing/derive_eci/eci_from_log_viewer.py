"""
Estimate ECI from log_viewer benchmark accuracy
================================================

Method 2: For each (model, benchmark) accuracy in data/log_viewer_summary/,
invert the IRT model from eci-public to produce a per-benchmark ECI estimate,
and append rows to data/eci_from_benchmarks/eci_from_benchmarks.csv.

Inversion: given benchmark difficulty D and discriminability α from the
fitted ECI model,
    performance = sigmoid(α · (capability - D))
    => capability = D + logit(perf) / α
    => ECI       = a + b · capability    (linear scaling from compute_eci_scores)

The benchmark params + scaling come from a one-time fit of the public ECI
data (cached to data/eci_from_benchmarks/_cache/).
"""

import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

# Make eci-public importable without installing
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eci-public" / "src"))

from eci import load_benchmark_data, fit_eci_model, compute_eci_scores  # noqa: E402
from eci.dataloader import RANDOM_BASELINES  # noqa: E402
from eci.fitting import (  # noqa: E402
    DEFAULT_ANCHOR_MODEL_LOW, DEFAULT_ANCHOR_ECI_LOW,
    DEFAULT_ANCHOR_MODEL_HIGH, DEFAULT_ANCHOR_ECI_HIGH,
)


SUMMARY_DIR = ROOT / "data" / "log_viewer_summary"
OUT_DIR = ROOT / "data" / "eci_from_benchmarks"
OUT_CSV = OUT_DIR / "eci_from_benchmarks.csv"
CACHE_DIR = OUT_DIR / "_cache"
BENCH_CACHE = CACHE_DIR / "bench_params.csv"
SCALE_CACHE = CACHE_DIR / "eci_scaling.json"

# Map log_viewer task names → eci-public benchmark names
TASK_TO_ECI_BENCH = {
    "GPQA Diamond": "GPQA diamond",
    "OTIS Mock AIME 2024-2025": "OTIS Mock AIME 2024-2025",
    "Chess Puzzles": "Chess Puzzles",
    "frontiermath_2025_02_28_public": "FrontierMath-2025-02-28-Private",
    "frontiermath_tier_4_2025_07_01_public": "FrontierMath-Tier-4-2025-07-01-Private",
    "simpleqa_verified": "SimpleQA Verified",
    "swe_bench_verified": "SWE-Bench verified",
}


def get_eci_params():
    """Fit (or load cached) benchmark params and ECI scaling from public ECI data."""
    if BENCH_CACHE.exists() and SCALE_CACHE.exists():
        bench_df = pd.read_csv(BENCH_CACHE)
        with open(SCALE_CACHE) as f:
            scale = json.load(f)
        return bench_df, scale["a"], scale["b"]

    print("Fitting ECI model on public benchmark data (one-time, cached)...")
    df = load_benchmark_data()
    model_df, bench_df = fit_eci_model(df, bootstrap_samples=0)
    eci_df, edi_df = compute_eci_scores(model_df, bench_df)

    # Recover linear scaling: eci = a + b * capability
    low = model_df.loc[model_df["Model"] == DEFAULT_ANCHOR_MODEL_LOW, "capability"].iloc[0]
    high = model_df.loc[model_df["Model"] == DEFAULT_ANCHOR_MODEL_HIGH, "capability"].iloc[0]
    b = (DEFAULT_ANCHOR_ECI_HIGH - DEFAULT_ANCHOR_ECI_LOW) / (high - low)
    a = DEFAULT_ANCHOR_ECI_LOW - b * low

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bench_df.to_csv(BENCH_CACHE, index=False)
    with open(SCALE_CACHE, "w") as f:
        json.dump({"a": a, "b": b}, f)
    return bench_df, a, b


def collect_log_viewer_rows():
    """One row per JSON file: model, task, accuracy, tokens."""
    rows = []
    for p in sorted(glob(str(SUMMARY_DIR / "*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        task = d.get("task")
        if task not in TASK_TO_ECI_BENCH:
            continue
        acc = d.get("accuracy")
        if acc is None:
            continue
        totals = d.get("totals") or {}
        rows.append({
            "file": Path(p).name,
            "task": task,
            "eci_benchmark": TASK_TO_ECI_BENCH[task],
            "model_id": d.get("model"),
            "accuracy": float(acc),
            "reasoning_tokens": totals.get("total_reasoning_tokens"),
            "answer_tokens": (totals.get("total_output_tokens") or 0)
                             - (totals.get("total_reasoning_tokens") or 0)
                             if totals.get("total_output_tokens") is not None
                             else None,
            "total_output_tokens": totals.get("total_output_tokens"),
        })
    return pd.DataFrame(rows)


def estimate_eci(df, bench_df, a, b):
    """Per-row ECI from single-benchmark accuracy via inverse IRT."""
    bench_lookup = bench_df.set_index("benchmark")[["difficulty", "discriminability"]].to_dict("index")

    # Apply random baseline correction (same as eci-public's pipeline)
    baselines = df["eci_benchmark"].map(RANDOM_BASELINES).fillna(0.0)
    perf = (df["accuracy"] - baselines) / (1.0 - baselines)
    # Match eci-public's performance_clip_eps=1e-3
    eps = 1e-3
    perf = perf.clip(eps, 1 - eps)

    diff = df["eci_benchmark"].map(lambda b_: bench_lookup[b_]["difficulty"])
    disc = df["eci_benchmark"].map(lambda b_: bench_lookup[b_]["discriminability"])

    logit = np.log(perf / (1 - perf))
    capability = diff + logit / disc
    df = df.copy()
    df["eci_estimated"] = a + b * capability
    return df


def update_output_csv(new_rows):
    """Append new rows to the master CSV, dropping any prior rows from these sources."""
    new_sources = set(new_rows["benchmark_source"].unique())
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        existing = existing[~existing["benchmark_source"].isin(new_sources)]
        out = pd.concat([existing, new_rows], ignore_index=True)
    else:
        out = new_rows
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    return out


if __name__ == "__main__":
    bench_df, a, b = get_eci_params()

    raw = collect_log_viewer_rows()
    print(f"Collected {len(raw)} log_viewer rows across {raw['task'].nunique()} tasks")

    scored = estimate_eci(raw, bench_df, a, b)

    out_rows = pd.DataFrame({
        "model": scored["model_id"],
        "slug": "",
        "company": "",
        "benchmark_source": "log_viewer_" + scored["task"].map(
            lambda t: t.lower().replace(" ", "_").replace("-", "_")
        ),
        "benchmark_score": scored["accuracy"],
        "eci_estimated": scored["eci_estimated"],
        "reasoning_tokens": scored["reasoning_tokens"],
        "answer_tokens": scored["answer_tokens"],
        "total_output_tokens": scored["total_output_tokens"],
    })

    full = update_output_csv(out_rows)
    print(f"Wrote {OUT_CSV} ({len(full)} total rows; {len(out_rows)} new from log_viewer)")
    print(out_rows.head(10).to_string(index=False))
