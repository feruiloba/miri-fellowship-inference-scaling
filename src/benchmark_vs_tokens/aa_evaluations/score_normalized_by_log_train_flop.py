"""
Scatter of score / log10(training FLOP) vs. inference tokens, with each
point colored by model release date. No per-period aggregation.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EVAL_CSV = ROOT / "data" / "artificial_analysis" / "aa_evaluations_combined.csv"
STATS_CSV = ROOT / "data" / "artificial_analysis" / "artificial_analysis_llm_stats.csv"
MERGED_CSV = ROOT / "data" / "merged_datasets.csv"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens" / "aa_evaluations"

BENCHMARKS = [
    "gpqa-diamond",
    "aime-2025",
    "humanitys-last-exam",
    "artificial-analysis-long-context-reasoning",
]


def _train_flop_by_slug() -> pd.DataFrame:
    m = pd.read_csv(MERGED_CSV, usecols=["AA_slug", "Training compute (FLOP)"])
    m = m.dropna(subset=["AA_slug", "Training compute (FLOP)"])
    m = m.rename(columns={
        "AA_slug": "model_slug",
        "Training compute (FLOP)": "train_flop",
    })
    return m.drop_duplicates(subset=["model_slug"])


def _load(benchmark: str, train: pd.DataFrame) -> pd.DataFrame:
    ev = pd.read_csv(EVAL_CSV)
    ev = ev[ev["benchmark"] == benchmark].copy()
    ev = ev.dropna(subset=["total_output_tokens", "score_raw", "model_slug"])
    ev = ev[ev["total_output_tokens"] > 0]

    stats = pd.read_csv(STATS_CSV, usecols=["slug", "release_date"])
    stats = stats.rename(columns={"slug": "model_slug"})
    ev = ev.merge(stats, on="model_slug", how="left")
    ev = ev.merge(train, on="model_slug", how="inner")
    ev = ev[ev["train_flop"] > 1]
    ev["release_date"] = pd.to_datetime(ev["release_date"], errors="coerce")
    ev["score_per_log_train"] = ev["score_raw"] / np.log10(ev["train_flop"])
    return ev


def _draw_panel(ax, benchmark, df, vmin, vmax):
    dated = df[df["release_date"].notna()]
    undated = df[df["release_date"].isna()]

    if not undated.empty:
        ax.scatter(
            undated["total_output_tokens"], undated["score_per_log_train"],
            s=18, alpha=0.35, color="#bbbbbb", edgecolor="none",
            label="(no release date)",
        )
    sc = None
    if not dated.empty:
        sc = ax.scatter(
            dated["total_output_tokens"], dated["score_per_log_train"],
            c=mdates.date2num(dated["release_date"]),
            cmap="viridis", vmin=vmin, vmax=vmax,
            s=22, alpha=0.85, edgecolor="none",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Total inference tokens")
    ax.set_ylabel("score / log10(train FLOP)")
    ax.set_title(benchmark, fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    if not undated.empty:
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    return sc


def main(benchmarks=BENCHMARKS):
    train = _train_flop_by_slug()
    panels = []
    all_dates = []
    for b in benchmarks:
        df = _load(b, train)
        if df.empty:
            print(f"skipping {b!r} (no rows)")
            continue
        panels.append((b, df))
        all_dates.append(df["release_date"].dropna())

    if not panels:
        print("Nothing to plot.")
        return

    dates_concat = pd.concat(all_dates) if all_dates else pd.Series(dtype="datetime64[ns]")
    vmin = mdates.date2num(dates_concat.min()) if len(dates_concat) else 0
    vmax = mdates.date2num(dates_concat.max()) if len(dates_concat) else 1

    cols = 2
    rows_n = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 5 * rows_n), squeeze=False)
    last_sc = None
    for idx, (b, df) in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        sc = _draw_panel(ax, b, df, vmin, vmax)
        if sc is not None:
            last_sc = sc
    for j in range(len(panels), rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
        cbar.set_label("Model release date")
        loc = mdates.AutoDateLocator()
        cbar.ax.yaxis.set_major_locator(loc)
        cbar.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    fig.suptitle(
        "Score per log10(train FLOP) vs. inference tokens — colored by model release date",
        fontsize=12,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "score_normalized_by_log_train_flop.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    combined = pd.concat([df.assign(benchmark=b) for b, df in panels], ignore_index=True)
    out_csv = OUT_DIR / "score_normalized_by_log_train_flop.csv"
    combined[[
        "benchmark", "model_slug", "release_date", "train_flop",
        "total_output_tokens", "score_raw", "score_per_log_train",
    ]].to_csv(out_csv, index=False)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    for b, df in panels:
        n_dated = df["release_date"].notna().sum()
        print(f"  {b}: {len(df)} rows ({n_dated} with release date)")


if __name__ == "__main__":
    main()
