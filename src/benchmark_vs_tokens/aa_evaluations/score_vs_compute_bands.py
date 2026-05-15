"""
Bucket models by test-time compute (total inference tokens) and plot a
per-band central-tendency statistic (mean, median, or top-10 mean) of the
benchmark score, across several important AA benchmarks.

Idea: a per-row scatter of (tokens, score) is dominated by cross-model
variation. Reducing within log-spaced compute bands smooths that out and
shows whether more inference compute correlates with higher scores once
model identity is collapsed.

One figure with four panels; each panel is one benchmark. The combined CSV
has a `benchmark` column distinguishing rows.

Usage:
    python src/benchmark_vs_tokens/score_vs_compute_bands.py [mean|median|top10]

Default stat = mean.

  mean   — average ± SEM of every run in the band
  median — median + IQR whiskers of every run in the band
  top10  — average ± SEM of the top-10 scoring runs in the band (frontier view)

Output (stat suffix only added for non-mean variants):
    output/benchmark_vs_tokens/aa_evaluations/score_vs_compute_bands.{png,csv}
    output/benchmark_vs_tokens/aa_evaluations/score_vs_compute_bands_median.{png,csv}
    output/benchmark_vs_tokens/aa_evaluations/score_vs_compute_bands_top10.{png,csv}
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EVAL_CSV = ROOT / "data" / "artificial_analysis" / "aa_evaluations_combined.csv"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens" / "aa_evaluations"

BENCHMARKS = [
    "gpqa-diamond",
    "aime-2025",
    "humanitys-last-exam",
    "artificial-analysis-long-context-reasoning",
]
DEFAULT_STAT = "mean"
N_BANDS = 8
TOP_K = 10


def _summarize(df: pd.DataFrame, stat: str, centers: np.ndarray) -> pd.DataFrame:
    """Per-band aggregates."""
    grouped = df.groupby("band", observed=True)["score_raw"]
    summary = grouped.size().rename("n").to_frame().reindex(range(N_BANDS))

    if stat == "mean":
        summary["center"] = grouped.mean().reindex(range(N_BANDS))
        sem = grouped.std().reindex(range(N_BANDS)) / np.sqrt(
            summary["n"].clip(lower=1)
        )
        summary["spread_lo"] = summary["center"] - sem
        summary["spread_hi"] = summary["center"] + sem
        summary["spread_label"] = "mean ± SEM"
    elif stat == "median":
        summary["center"] = grouped.median().reindex(range(N_BANDS))
        summary["spread_lo"] = grouped.quantile(0.25).reindex(range(N_BANDS))
        summary["spread_hi"] = grouped.quantile(0.75).reindex(range(N_BANDS))
        summary["spread_label"] = "median (IQR whiskers)"
    elif stat == "top10":
        def _top_stats(s: pd.Series) -> pd.Series:
            top = s.nlargest(TOP_K)
            return pd.Series({
                "n_top": len(top),
                "mean": top.mean(),
                "sem": top.std() / np.sqrt(len(top)) if len(top) >= 2 else np.nan,
            })
        agg = grouped.apply(_top_stats).unstack().reindex(range(N_BANDS))
        summary["n_top"] = agg["n_top"]
        summary["center"] = agg["mean"]
        summary["spread_lo"] = agg["mean"] - agg["sem"]
        summary["spread_hi"] = agg["mean"] + agg["sem"]
        summary["spread_label"] = f"top-{TOP_K} mean ± SEM"
    else:
        raise ValueError(f"unknown stat {stat!r}")

    summary.loc[summary["n"] < 2, ["spread_lo", "spread_hi"]] = np.nan
    summary["band_center_tokens"] = centers
    return summary


def _build_panel(benchmark: str, stat: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    df = pd.read_csv(EVAL_CSV)
    df = df[df["benchmark"] == benchmark].copy()
    df = df.dropna(subset=["total_output_tokens", "score_raw"])
    df = df[df["total_output_tokens"] > 0]
    if df.empty:
        return None

    log_tokens = np.log10(df["total_output_tokens"].to_numpy(dtype=float))
    edges = np.linspace(log_tokens.min(), log_tokens.max(), N_BANDS + 1)
    df["band"] = pd.cut(log_tokens, bins=edges, include_lowest=True, labels=False)

    centers = 10 ** ((edges[:-1] + edges[1:]) / 2)
    summary = _summarize(df, stat, centers)
    summary["band_lo_tokens"] = 10 ** edges[:-1]
    summary["band_hi_tokens"] = 10 ** edges[1:]
    return df, summary


def _draw_panel(ax, benchmark: str, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    ax.scatter(
        df["total_output_tokens"], df["score_raw"],
        s=10, alpha=0.18, color="#888888", edgecolor="none",
        label=f"individual runs (n={len(df)})",
    )

    have_data = summary["n"].fillna(0) > 0
    centers_have = summary.loc[have_data, "band_center_tokens"].to_numpy()
    y_have = summary.loc[have_data, "center"].to_numpy()
    lo = summary.loc[have_data, "spread_lo"].to_numpy()
    hi = summary.loc[have_data, "spread_hi"].to_numpy()
    yerr = np.vstack([y_have - lo, hi - y_have])
    yerr = np.where(np.isnan(yerr), 0.0, yerr)

    ax.errorbar(
        centers_have, y_have, yerr=yerr,
        fmt="o-", color="#1f77b4", linewidth=1.8, markersize=6,
        capsize=4, capthick=1.0,
        label=f"band {summary['spread_label'].iloc[0]}",
    )

    for _, row in summary[have_data].iterrows():
        ax.annotate(
            f"{int(row['n'])}",
            (row["band_center_tokens"], row["center"]),
            textcoords="offset points", xytext=(0, 7),
            ha="center", fontsize=7, color="#1f77b4",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total inference tokens")
    ax.set_ylabel("Score (fraction correct)")
    ax.set_title(benchmark, fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)


def main(stat: str = DEFAULT_STAT, benchmarks: list[str] = BENCHMARKS) -> None:
    if stat not in {"mean", "median", "top10"}:
        raise SystemExit(f"stat must be 'mean', 'median', or 'top10', got {stat!r}")

    panels: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for b in benchmarks:
        result = _build_panel(b, stat)
        if result is None:
            print(f"skipping {b!r} (no rows)")
            continue
        df, summary = result
        panels.append((b, df, summary))

    if not panels:
        print("Nothing to plot.")
        return

    cols = 2
    rows_n = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 5 * rows_n), squeeze=False)
    for idx, (b, df, summary) in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        _draw_panel(ax, b, df, summary)
    for j in range(len(panels), rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        f"{stat.title()} score per compute band  "
        f"({N_BANDS} log-spaced bands, AA evaluations)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if stat == "mean" else f"_{stat}"
    out_png = OUT_DIR / f"score_vs_compute_bands{suffix}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    combined = pd.concat(
        [s.assign(benchmark=b) for b, _, s in panels],
        ignore_index=False,
    ).rename_axis("band_idx").reset_index()
    out_csv = OUT_DIR / f"score_vs_compute_bands{suffix}.csv"
    combined.to_csv(out_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    for b, df, summary in panels:
        plottable = int(summary["n"].fillna(0).gt(0).sum())
        print(f"  {b}: {len(df)} rows, {plottable} plotted bands")


if __name__ == "__main__":
    args = sys.argv[1:]
    stat = args[0] if len(args) >= 1 else DEFAULT_STAT
    main(stat)
