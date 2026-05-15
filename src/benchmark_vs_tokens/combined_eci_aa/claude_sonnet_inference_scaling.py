"""
Per-benchmark inference-scaling curves for Claude 3.7 Sonnet and Claude 4 Sonnet.

Pools data from AA evaluations (aa_evaluations_combined.csv) and the ECI
log_viewer summaries. Each panel is one benchmark; per panel, both models
get a line of score vs total inference tokens, with marker shape encoding
the data source (circle = AA, triangle = log_viewer).

Two name pairs are merged into a canonical benchmark so we can pool sources:
  AA "gpqa-diamond"  +  log_viewer "GPQA Diamond"           → "GPQA Diamond"
  AA "aime-2025"     +  log_viewer "OTIS Mock AIME 2024-2025" → "AIME"

A benchmark is shown only when *both* target models have ≥2 plottable rows.

Output:
  output/benchmark_vs_tokens/combined_eci_aa/claude_sonnet_inference_scaling.{png,csv}
"""

import json
import math
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AA_EVAL_CSV = ROOT / "data" / "artificial_analysis" / "aa_evaluations_combined.csv"
LOG_VIEWER_DIR = ROOT / "data" / "eci" / "log_viewer_summary"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens" / "combined_eci_aa"

TARGET_MODELS = {
    "claude-3-7-sonnet": "Claude 3.7 Sonnet",
    "claude-4-sonnet":   "Claude 4 Sonnet",
}
MODEL_COLORS = {
    "claude-3-7-sonnet": "#1f77b4",
    "claude-4-sonnet":   "#d62728",
}
SOURCE_MARKERS = {"aa": "o", "log_viewer": "^"}

# Map raw benchmark / task name -> canonical name when we want to pool sources
BENCHMARK_MERGE = {
    "gpqa-diamond": "GPQA Diamond",
    "GPQA Diamond": "GPQA Diamond",
    "aime-2025": "AIME",
    "OTIS Mock AIME 2024-2025": "AIME",
}

MIN_POINTS_PER_MODEL = 2


def _log_viewer_total_tokens(totals: dict) -> float | None:
    raw_out = totals.get("total_output_tokens")
    raw_r = totals.get("total_reasoning_tokens")
    if raw_out is None or raw_out <= 0:
        return None
    if raw_r is not None and raw_out < raw_r:
        return float(raw_out) + float(raw_r)
    return float(raw_out)


def _load() -> pd.DataFrame:
    rows = []

    aa = pd.read_csv(AA_EVAL_CSV)
    aa = aa.dropna(subset=["model_slug", "total_output_tokens", "score_raw"])
    aa = aa[(aa["total_output_tokens"] > 0) & aa["model_slug"].isin(TARGET_MODELS)]
    for _, r in aa.iterrows():
        rows.append({
            "source": "aa",
            "raw_benchmark": r["benchmark"],
            "model_slug": r["model_slug"],
            "model_name": r.get("model"),
            "tokens": float(r["total_output_tokens"]),
            "score": float(r["score_raw"]),
        })

    for p in sorted(glob(str(LOG_VIEWER_DIR / "*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        slug = d.get("aa_model_slug")
        if slug not in TARGET_MODELS:
            continue
        score = d.get("accuracy")
        tokens = _log_viewer_total_tokens(d.get("totals") or {})
        if score is None or tokens is None:
            continue
        rows.append({
            "source": "log_viewer",
            "raw_benchmark": d.get("task"),
            "model_slug": slug,
            "model_name": d.get("aa_name") or d.get("model", "").rsplit("/", 1)[-1],
            "tokens": tokens,
            "score": float(score),
        })

    df = pd.DataFrame(rows)
    df["benchmark"] = df["raw_benchmark"].map(lambda b: BENCHMARK_MERGE.get(b, b))
    return df


def _eligible_benchmarks(df: pd.DataFrame) -> list[str]:
    """Benchmarks where every target model has ≥ MIN_POINTS_PER_MODEL rows."""
    counts = df.groupby(["benchmark", "model_slug"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(TARGET_MODELS), fill_value=0)
    keep = counts[(counts >= MIN_POINTS_PER_MODEL).all(axis=1)].index.tolist()
    return sorted(keep)


def _draw_panel(ax, df: pd.DataFrame, benchmark: str) -> None:
    sub = df[df["benchmark"] == benchmark]
    for slug, name in TARGET_MODELS.items():
        family = sub[sub["model_slug"] == slug].sort_values("tokens")
        if family.empty:
            continue
        color = MODEL_COLORS[slug]
        # One connecting line through all points of this model
        ax.plot(
            family["tokens"], family["score"],
            color=color, linewidth=1.5, alpha=0.6, zorder=2,
        )
        # Scatter by source so marker shape distinguishes provenance
        for src, marker in SOURCE_MARKERS.items():
            pts = family[family["source"] == src]
            if pts.empty:
                continue
            ax.scatter(
                pts["tokens"], pts["score"],
                color=color, marker=marker, s=55, zorder=3,
                edgecolors="white", linewidths=0.6,
                label=f"{name} ({src})",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Total inference tokens")
    ax.set_ylabel("Score (fraction correct)")
    ax.set_title(benchmark, fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)


def main() -> None:
    df = _load()
    if df.empty:
        print("No rows for target models.")
        return

    benchmarks = _eligible_benchmarks(df)
    if not benchmarks:
        print(f"No benchmark has ≥{MIN_POINTS_PER_MODEL} points for every target model.")
        return

    cols = 3
    rows = math.ceil(len(benchmarks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 4 * rows), squeeze=False)
    for idx, bench in enumerate(benchmarks):
        _draw_panel(axes[idx // cols][idx % cols], df, bench)
    for j in range(len(benchmarks), rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        f"Inference-token scaling per benchmark — "
        f"{', '.join(TARGET_MODELS.values())}  "
        f"(AA = ●, log_viewer = ▲)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "claude_sonnet_inference_scaling.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    out_csv = OUT_DIR / "claude_sonnet_inference_scaling.csv"
    df[df["benchmark"].isin(benchmarks)].sort_values(
        ["benchmark", "model_slug", "tokens"]
    ).to_csv(out_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    print(f"\nBenchmarks plotted ({len(benchmarks)}):")
    for b in benchmarks:
        per = df[df["benchmark"] == b].groupby("model_slug").size()
        print("  " + b + " — " + ", ".join(f"{TARGET_MODELS[s]}: {n}" for s, n in per.items()))


if __name__ == "__main__":
    main()
