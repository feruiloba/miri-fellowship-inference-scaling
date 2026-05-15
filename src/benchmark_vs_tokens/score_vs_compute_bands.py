"""
Bucket models by test-time compute (total inference tokens) and plot the
average benchmark score per band.

Idea: a per-row scatter of (tokens, score) is dominated by cross-model
variation. Averaging within log-spaced compute bands smooths that out and
shows whether more inference compute correlates with higher scores once
model identity is collapsed.

Defaults to AA's gpqa-diamond evaluation; pass another benchmark slug as the
first CLI argument (e.g. `aime-2025`) to retarget.

Output:
  output/benchmark_vs_tokens/score_vs_compute_bands__<benchmark>.png
  output/benchmark_vs_tokens/score_vs_compute_bands__<benchmark>.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVAL_CSV = ROOT / "data" / "artificial_analysis" / "aa_evaluations_combined.csv"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens" / "aa_evaluations"

DEFAULT_BENCHMARK = "gpqa-diamond"
N_BANDS = 8


def main(benchmark: str = DEFAULT_BENCHMARK) -> None:
    df = pd.read_csv(EVAL_CSV)
    df = df[df["benchmark"] == benchmark].copy()
    df = df.dropna(subset=["total_output_tokens", "score_raw"])
    df = df[df["total_output_tokens"] > 0]
    if df.empty:
        print(f"No usable rows for benchmark={benchmark!r}")
        return

    log_tokens = np.log10(df["total_output_tokens"].to_numpy(dtype=float))
    edges = np.linspace(log_tokens.min(), log_tokens.max(), N_BANDS + 1)
    df["band"] = pd.cut(log_tokens, bins=edges, include_lowest=True, labels=False)

    centers = 10 ** ((edges[:-1] + edges[1:]) / 2)
    summary = (
        df.groupby("band", observed=True)
        .agg(
            n=("score_raw", "size"),
            mean_score=("score_raw", "mean"),
            std_score=("score_raw", "std"),
        )
        .reindex(range(N_BANDS))
    )
    summary["band_center_tokens"] = centers
    summary["band_lo_tokens"] = 10 ** edges[:-1]
    summary["band_hi_tokens"] = 10 ** edges[1:]
    # Standard error of the mean (std / sqrt(n)), guarded for n<2
    summary["sem"] = summary["std_score"] / np.sqrt(summary["n"].clip(lower=1))
    summary.loc[summary["n"] < 2, "sem"] = np.nan

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bench_slug = benchmark.replace("-", "_")
    out_csv = OUT_DIR / f"score_vs_compute_bands__{bench_slug}.csv"
    summary.to_csv(out_csv, index_label="band_idx")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Background: every model's (tokens, score_raw) as a faint dot
    ax.scatter(
        df["total_output_tokens"], df["score_raw"],
        s=10, alpha=0.18, color="#888888", edgecolor="none",
        label=f"individual runs (n={len(df)})",
    )

    have_data = summary["n"].fillna(0) > 0
    ax.errorbar(
        summary.loc[have_data, "band_center_tokens"],
        summary.loc[have_data, "mean_score"],
        yerr=summary.loc[have_data, "sem"],
        fmt="o-", color="#1f77b4", linewidth=1.8, markersize=7,
        capsize=4, capthick=1.1,
        label="band mean ± SEM",
    )

    # n labels above each band point
    for _, row in summary[have_data].iterrows():
        ax.annotate(
            f"n={int(row['n'])}",
            (row["band_center_tokens"], row["mean_score"]),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=8, color="#1f77b4",
        )

    ax.set_xscale("log")
    ax.set_xlabel(f"Total inference tokens (AA {benchmark} run)")
    ax.set_ylabel("Score (fraction correct)")
    ax.set_title(
        f"Mean {benchmark} score per compute band  "
        f"({N_BANDS} log-spaced bands, AA evaluations)"
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9)

    out_png = OUT_DIR / f"score_vs_compute_bands__{bench_slug}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    print()
    print(summary[["band_lo_tokens", "band_hi_tokens", "n", "mean_score", "sem"]]
          .to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    benchmark = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BENCHMARK
    main(benchmark)
