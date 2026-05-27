"""
gpt-oss-20b vs gpt-oss-120b: inference tokens vs benchmark score.

Both models were released on 2025-08-05 with training compute estimates
in data/eci/epoch_all_ai_models.csv (5.49e23 vs 4.94e24 FLOP, ~9× ratio).
Each has two AA effort variants (low / high) on 11 benchmarks, so we can
draw a 2-point segment per model per benchmark and compare inference-token
scaling at fixed training compute.

Output:
  output/benchmark_vs_tokens/aa_evaluations/gpt_oss_effort_comparison.png
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EVAL_CSV = ROOT / "data" / "artificial_analysis" / "aa_evaluations_combined.csv"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens" / "aa_evaluations"

MODEL_SLUGS = {
    "gpt-oss-20b":  ["gpt-oss-20b",  "gpt-oss-20b-low"],     # high, low
    "gpt-oss-120b": ["gpt-oss-120b", "gpt-oss-120b-low"],
}

MODEL_COLORS = {
    "gpt-oss-20b":  "#1f77b4",
    "gpt-oss-120b": "#d62728",
}

MODEL_COMPUTE = {
    "gpt-oss-20b":  5.49e23,
    "gpt-oss-120b": 4.94e24,
}


def _effort_from_slug(slug: str) -> str:
    return "low" if slug.endswith("-low") else "high"


def load() -> pd.DataFrame:
    all_slugs = sum(MODEL_SLUGS.values(), [])
    ev = pd.read_csv(EVAL_CSV)
    df = ev[ev["model_slug"].isin(all_slugs)].copy()
    df = df.dropna(subset=["total_output_tokens", "score_raw"])
    df = df[df["total_output_tokens"] > 0]
    df["effort"] = df["model_slug"].map(_effort_from_slug)
    df["model"] = df["model_slug"].map(
        lambda s: next(m for m, slugs in MODEL_SLUGS.items() if s in slugs)
    )
    return df


def _draw_panel(ax, benchmark: str, sub: pd.DataFrame) -> None:
    for model, slugs in MODEL_SLUGS.items():
        m = sub[sub["model"] == model].sort_values("total_output_tokens")
        if m.empty:
            continue
        color = MODEL_COLORS[model]
        ax.plot(
            m["total_output_tokens"], m["score_raw"],
            color=color, linewidth=1.5, alpha=0.7, zorder=2, linestyle="-",
        )
        ax.scatter(
            m["total_output_tokens"], m["score_raw"],
            color=color, s=70, zorder=3, edgecolor="white", linewidth=0.6,
            label=f"{model} ({MODEL_COMPUTE[model]:.1e} FLOP)" if model not in ax.get_legend_handles_labels()[1] else None,
        )
        for _, row in m.iterrows():
            ax.annotate(
                row["effort"],
                (row["total_output_tokens"], row["score_raw"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=7, color=color, va="bottom",
            )

    ax.set_xscale("log")
    ax.set_title(benchmark, fontsize=10)
    ax.set_xlabel("Inference tokens")
    ax.set_ylabel("Score (raw)")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)


def main() -> None:
    df = load()
    # Skip benchmarks where either model has only 1 effort level
    counts = df.groupby(["benchmark", "model"])["effort"].nunique().unstack(fill_value=0)
    eligible = counts[(counts.get("gpt-oss-20b", 0) >= 2) & (counts.get("gpt-oss-120b", 0) >= 2)].index.tolist()
    eligible = sorted(eligible)
    print(f"Plotting {len(eligible)} benchmarks with both effort levels for both models")

    cols = 3
    rows = math.ceil(len(eligible) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.0 * rows), squeeze=False)

    legend_handles = []
    for idx, b in enumerate(eligible):
        ax = axes[idx // cols][idx % cols]
        _draw_panel(ax, b, df[df["benchmark"] == b])
        if idx == 0:
            legend_handles = ax.get_legend_handles_labels()
    # hide unused
    for j in range(len(eligible), rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    # Single shared legend
    if legend_handles[0]:
        fig.legend(
            handles=legend_handles[0], labels=legend_handles[1],
            loc="lower center", ncol=2, fontsize=10,
            bbox_to_anchor=(0.5, -0.01),
        )

    fig.suptitle(
        "gpt-oss-20b vs gpt-oss-120b — inference tokens vs benchmark score\n"
        "(both released 2025-08-05; training compute ratio ~9×)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "gpt_oss_effort_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
