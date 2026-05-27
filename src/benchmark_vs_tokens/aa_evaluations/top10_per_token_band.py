"""
Bucket runs into log-spaced inference-token bands and plot the per-band
top-10 mean score, across several important AA benchmarks.

Idea: a per-row scatter of (tokens, score) is dominated by cross-model
variation. Aggregating within log-spaced inference-token bands smooths
that out, and the top-10 mean per band traces the frontier of what's
achievable at each compute level.

One figure with four panels; each panel is one benchmark. The combined CSV
has a `benchmark` column distinguishing rows.

Output:
    output/benchmark_vs_tokens/aa_evaluations/top10_per_token_band.{png,csv}
"""

import math
from pathlib import Path

import matplotlib.dates as mdates
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


def _summarize(df: pd.DataFrame, centers: np.ndarray) -> pd.DataFrame:
    """Per-band top-K mean ± SEM."""
    grouped = df.groupby("band", observed=True)["score_raw"]
    summary = grouped.size().rename("n").to_frame().reindex(range(N_BANDS))

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

    summary.loc[summary["n"] < 2, ["spread_lo", "spread_hi"]] = np.nan
    summary["band_center_tokens"] = centers
    return summary


def _load_release_dates() -> dict[str, str]:
    stats = pd.read_csv(STATS_CSV, usecols=["slug", "release_date"])
    stats = stats.dropna(subset=["slug", "release_date"])
    return dict(zip(stats["slug"], stats["release_date"]))


def _build_panel(benchmark: str, slug_to_date: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    df = pd.read_csv(EVAL_CSV)
    df = df[df["benchmark"] == benchmark].copy()
    df = df.dropna(subset=["total_output_tokens", "score_raw"])
    df = df[df["total_output_tokens"] > 0]
    if df.empty:
        return None
    df["release_date"] = pd.to_datetime(
        df["model_slug"].map(slug_to_date), errors="coerce"
    )

    log_tokens = np.log10(df["total_output_tokens"].to_numpy(dtype=float))
    edges = np.linspace(log_tokens.min(), log_tokens.max(), N_BANDS + 1)
    df["band"] = pd.cut(log_tokens, bins=edges, include_lowest=True, labels=False)

    centers = 10 ** ((edges[:-1] + edges[1:]) / 2)
    summary = _summarize(df, centers)
    summary["band_lo_tokens"] = 10 ** edges[:-1]
    summary["band_hi_tokens"] = 10 ** edges[1:]
    return df, summary


def _draw_panel(ax, benchmark: str, df: pd.DataFrame, summary: pd.DataFrame,
                date_norm, cmap):
    # Points with known release date: coloured by date. Points without a date
    # fall back to a faint gray dot so they're not silently dropped.
    has_date = df["release_date"].notna()
    if has_date.any():
        sub = df[has_date]
        ax.scatter(
            sub["total_output_tokens"], sub["score_raw"],
            s=14, alpha=0.7, edgecolor="none",
            c=mdates.date2num(sub["release_date"]),
            cmap=cmap, norm=date_norm,
        )
    if (~has_date).any():
        sub_n = df[~has_date]
        ax.scatter(
            sub_n["total_output_tokens"], sub_n["score_raw"],
            s=8, alpha=0.25, color="#888888", edgecolor="none",
        )
    n_total = len(df)
    n_dated = int(has_date.sum())
    ax.scatter([], [], s=14, c="#888888",
               label=f"runs (n={n_total}; {n_dated} dated)")

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


def main(benchmarks: list[str] = BENCHMARKS) -> None:
    slug_to_date = _load_release_dates()

    panels: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for b in benchmarks:
        result = _build_panel(b, slug_to_date)
        if result is None:
            print(f"skipping {b!r} (no rows)")
            continue
        df, summary = result
        panels.append((b, df, summary))

    if not panels:
        print("Nothing to plot.")
        return

    # Build a shared date colour scale across all panels
    all_dates = pd.concat([p[1]["release_date"].dropna() for p in panels])
    if all_dates.empty:
        date_min = date_max = pd.Timestamp("2024-01-01")
    else:
        date_min, date_max = all_dates.min(), all_dates.max()
    date_norm = plt.Normalize(
        vmin=mdates.date2num(date_min), vmax=mdates.date2num(date_max)
    )
    cmap = plt.get_cmap("viridis")

    cols = 2
    rows_n = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 5 * rows_n), squeeze=False)
    for idx, (b, df, summary) in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        _draw_panel(ax, b, df, summary, date_norm, cmap)
    for j in range(len(panels), rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        f"Top-{TOP_K} mean score per inference-token band  "
        f"({N_BANDS} log-spaced bands, AA evaluations)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))

    # Shared colorbar on the right
    sm = plt.cm.ScalarMappable(norm=date_norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.95, 0.10, 0.015, 0.78])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.yaxis.set_major_locator(mdates.AutoDateLocator())
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    cbar.set_label("Model release date", fontsize=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "top10_per_token_band.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    combined = pd.concat(
        [s.assign(benchmark=b) for b, _, s in panels],
        ignore_index=False,
    ).rename_axis("band_idx").reset_index()
    out_csv = OUT_DIR / "top10_per_token_band.csv"
    combined.to_csv(out_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    for b, df, summary in panels:
        plottable = int(summary["n"].fillna(0).gt(0).sum())
        print(f"  {b}: {len(df)} rows, {plottable} plotted bands")


if __name__ == "__main__":
    main()
