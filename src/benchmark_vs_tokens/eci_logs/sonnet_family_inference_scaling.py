"""
Inference-token scaling for Claude 3.7 Sonnet, Claude 4 Sonnet, and Claude 4.5
Sonnet on GPQA Diamond and OTIS Mock AIME 2024-2025, using only ECI log_viewer
summaries.

Each panel is one benchmark; per panel, each model gets a curve of
total_inference_tokens (log x) vs accuracy. Points within a model are
connected by a line in token order.

Output:
  output/benchmark_vs_tokens/eci_logs/sonnet_family_inference_scaling.{png,csv}
"""

import json
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LOG_VIEWER_DIR = ROOT / "data" / "eci" / "log_viewer_summary"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens" / "eci_logs"

TARGETS = {
    "claude-3-7-sonnet": "Claude 3.7 Sonnet",
    "claude-4-sonnet":   "Claude 4 Sonnet",
    "claude-4-5-sonnet": "Claude 4.5 Sonnet",
}
MODEL_COLORS = {
    "claude-3-7-sonnet": "#1f77b4",
    "claude-4-sonnet":   "#2ca02c",
    "claude-4-5-sonnet": "#d62728",
}
BENCHMARKS = ["GPQA Diamond", "OTIS Mock AIME 2024-2025"]


def _total_inference_tokens(totals: dict) -> float | None:
    """Match eci_from_log_viewer.py: handle the parallel/subset reasoning-token
    conventions so all providers are comparable."""
    out = totals.get("total_output_tokens")
    r = totals.get("total_reasoning_tokens")
    if out is None or out <= 0:
        return None
    if r is not None and out < r:
        return float(out) + float(r)
    return float(out)


def _load() -> pd.DataFrame:
    rows = []
    for p in sorted(glob(str(LOG_VIEWER_DIR / "*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        slug = d.get("aa_model_slug")
        task = d.get("task")
        if slug not in TARGETS or task not in BENCHMARKS:
            continue
        score = d.get("accuracy")
        tokens = _total_inference_tokens(d.get("totals") or {})
        if score is None or tokens is None:
            continue
        rows.append({
            "task": task,
            "model_slug": slug,
            "model_name": TARGETS[slug],
            "file": Path(p).name,
            "tokens": tokens,
            "score": float(score),
        })
    return pd.DataFrame(rows)


def _draw_panel(ax, df: pd.DataFrame, benchmark: str) -> None:
    sub = df[df["task"] == benchmark]
    for slug, name in TARGETS.items():
        fam = sub[sub["model_slug"] == slug].sort_values("tokens")
        if fam.empty:
            continue
        color = MODEL_COLORS[slug]
        ax.plot(
            fam["tokens"], fam["score"],
            marker="o", color=color, linewidth=1.6, markersize=7,
            markeredgecolor="white", markeredgewidth=0.5,
            label=f"{name} (n={len(fam)})",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total inference tokens")
    ax.set_ylabel("Accuracy")
    ax.set_title(benchmark, fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def main() -> None:
    df = _load()
    if df.empty:
        print("No log_viewer rows for target Sonnet models on the chosen benchmarks.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, benchmark in zip(axes, BENCHMARKS):
        _draw_panel(ax, df, benchmark)
    fig.suptitle(
        "Inference-token scaling on ECI log_viewer benchmarks — "
        "Claude 3.7 / 4 / 4.5 Sonnet",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "sonnet_family_inference_scaling.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    out_csv = OUT_DIR / "sonnet_family_inference_scaling.csv"
    df.sort_values(["task", "model_slug", "tokens"]).to_csv(out_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    print()
    summary = (
        df.groupby(["task", "model_slug"])
        .agg(n=("score", "size"),
             token_min=("tokens", "min"), token_max=("tokens", "max"),
             score_min=("score", "min"), score_max=("score", "max"))
    )
    print(summary.to_string(float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
