"""
Per-period frontier curves of score vs test-time compute, across several
important AA benchmarks.

For each benchmark, bin every run into log-spaced compute bands; then,
within each (release_period, band) cell take the top-10 scores and average
them. Releases are grouped into 6-month periods (H1 = Jan-Jun, H2 = Jul-Dec).
One curve per period in each panel shows how the compute frontier shifts
over time on that benchmark.

Joins:
  data/artificial_analysis/aa_evaluations_combined.csv  (benchmark, model_slug, tokens, score)
  data/artificial_analysis/artificial_analysis_llm_stats.csv  (slug → release_date)

Output:
  output/benchmark_vs_tokens/aa_evaluations/top10_grouped_by_release_period.{png,csv}

Each panel is one benchmark; the CSV has a `benchmark` column distinguishing
the rows.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EVAL_CSV = ROOT / "data" / "artificial_analysis" / "aa_evaluations_combined.csv"
STATS_CSV = ROOT / "data" / "artificial_analysis" / "artificial_analysis_llm_stats.csv"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens" / "aa_evaluations"

BENCHMARKS = [
    "gpqa-diamond",
    "aime-2025",
    "humanitys-last-exam",
    "artificial-analysis-long-context-reasoning",
]
N_BANDS = 8
TOP_K = 10
MIN_ROWS_PER_PERIOD = 5  # drop periods too sparse for a meaningful curve
MIN_BAND_N = 4           # drop (period, band) points with fewer runs than this


def _period_label(dt: pd.Series) -> pd.Series:
    """e.g. 2024-04-12 → '2024-H1', 2025-09-30 → '2025-H2'."""
    year = dt.dt.year.astype("Int64")
    half = ((dt.dt.month - 1) // 6 + 1).astype("Int64")
    return year.astype(str) + "-H" + half.astype(str)


def _period_sort_key(label: str) -> tuple[int, int]:
    year_part, half_part = label.split("-H")
    return int(year_part), int(half_part)


def _load(benchmark: str) -> pd.DataFrame:
    ev = pd.read_csv(EVAL_CSV)
    ev = ev[ev["benchmark"] == benchmark].copy()
    ev = ev.dropna(subset=["total_output_tokens", "score_raw", "model_slug"])
    ev = ev[ev["total_output_tokens"] > 0]

    stats = pd.read_csv(STATS_CSV, usecols=["slug", "release_date"])
    stats = stats.dropna(subset=["release_date"]).rename(columns={"slug": "model_slug"})
    ev = ev.merge(stats, on="model_slug", how="left")
    dt = pd.to_datetime(ev["release_date"], errors="coerce")
    ev = ev[dt.notna()].copy()
    ev["period"] = _period_label(dt[dt.notna()])
    return ev


def _build_panel(benchmark: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series] | None:
    df = _load(benchmark)
    if df.empty:
        return None

    log_tokens = np.log10(df["total_output_tokens"].to_numpy(dtype=float))
    edges = np.linspace(log_tokens.min(), log_tokens.max(), N_BANDS + 1)
    centers = 10 ** ((edges[:-1] + edges[1:]) / 2)
    df["band"] = pd.cut(log_tokens, bins=edges, include_lowest=True, labels=False)

    rows = []
    for (period, band), sub in df.groupby(["period", "band"], observed=True):
        top = sub["score_raw"].nlargest(TOP_K)
        rows.append({
            "period": period,
            "band_idx": int(band),
            "band_center_tokens": centers[int(band)],
            "n_band": len(sub),
            "n_top": len(top),
            "top_mean": float(top.mean()),
            "top_max": float(top.max()),
        })
    summary = pd.DataFrame(rows)
    summary = summary[summary["n_band"] >= MIN_BAND_N]

    period_counts = df.groupby("period").size()
    keep_periods = period_counts[period_counts >= MIN_ROWS_PER_PERIOD].index.tolist()
    summary = summary[summary["period"].isin(keep_periods)]
    summary = summary.sort_values(
        by=["period", "band_idx"],
        key=lambda s: s.map(_period_sort_key) if s.name == "period" else s,
    )
    return df, summary, period_counts


def _draw_panel(ax, benchmark: str, df: pd.DataFrame,
                summary: pd.DataFrame, period_counts: pd.Series) -> None:
    ax.scatter(
        df["total_output_tokens"], df["score_raw"],
        s=8, alpha=0.10, color="#888888", edgecolor="none",
    )

    periods_sorted = sorted(summary["period"].unique(), key=_period_sort_key)
    cmap = plt.get_cmap("viridis")
    for i, period in enumerate(periods_sorted):
        sub = summary[summary["period"] == period].sort_values("band_idx")
        color = cmap(i / max(len(periods_sorted) - 1, 1))
        ax.plot(
            sub["band_center_tokens"], sub["top_mean"],
            marker="o", color=color, linewidth=1.8, markersize=6,
            label=f"{period} (n={int(period_counts[period])})",
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{int(row['n_top'])}",
                (row["band_center_tokens"], row["top_mean"]),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=6, color=color,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Total inference tokens")
    ax.set_ylabel(f"Top-{TOP_K} mean score per band")
    ax.set_title(benchmark, fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(title="release period", loc="lower right", fontsize=7, framealpha=0.9)


def main(benchmarks: list[str] = BENCHMARKS) -> None:
    panels: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.Series]] = []
    for b in benchmarks:
        result = _build_panel(b)
        if result is None:
            print(f"skipping {b!r} (no rows)")
            continue
        df, summary, pc = result
        panels.append((b, df, summary, pc))

    if not panels:
        print("Nothing to plot.")
        return

    cols = 2
    rows_n = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 5 * rows_n), squeeze=False)
    for idx, (b, df, summary, pc) in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        _draw_panel(ax, b, df, summary, pc)
    for j in range(len(panels), rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        f"Per-half-year compute-frontier curves  "
        f"(top-{TOP_K} mean, {N_BANDS} log-spaced bands, n≥{MIN_BAND_N} per cell)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "top10_grouped_by_release_period.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    combined = pd.concat(
        [s.assign(benchmark=b) for b, _, s, _ in panels], ignore_index=True
    )
    out_csv = OUT_DIR / "top10_grouped_by_release_period.csv"
    combined.to_csv(out_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    for b, df, summary, _ in panels:
        print(f"  {b}: {len(df)} rows, {len(summary)} plotted points")


if __name__ == "__main__":
    main()
