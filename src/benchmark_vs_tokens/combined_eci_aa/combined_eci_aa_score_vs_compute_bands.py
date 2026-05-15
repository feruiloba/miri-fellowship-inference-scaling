"""
Score vs test-time compute bands on the combined AA + ECI log_viewer dataset.

Two panels — GPQA Diamond and AIME — the only benchmarks where both AA
evaluations and the ECI log_viewer have measurements (AIME is approximate:
AA's aime-2025 + log_viewer's OTIS Mock AIME 2024-2025). Each panel pools
both sources; coloured background scatter shows individual runs, the
foreground line is the per-band statistic.

Usage:
    python src/benchmark_vs_tokens/combined_eci_aa_score_vs_compute_bands.py [mean|median|top10]

Output:
  output/benchmark_vs_tokens/combined_eci_aa/score_vs_compute_bands{_<stat>}.{png,csv}
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _combined_eci_aa import BENCHMARKS, OUT_DIR, load_combined  # noqa: E402

DEFAULT_STAT = "mean"
N_BANDS = 8
TOP_K = 10
SOURCE_COLORS = {"aa": "#1f77b4", "log_viewer": "#ff7f0e"}


def _summarize(df: pd.DataFrame, stat: str, centers: np.ndarray) -> pd.DataFrame:
    grouped = df.groupby("band", observed=True)["score_raw"]
    summary = grouped.size().rename("n").to_frame().reindex(range(N_BANDS))
    if stat == "mean":
        summary["center"] = grouped.mean().reindex(range(N_BANDS))
        sem = grouped.std().reindex(range(N_BANDS)) / np.sqrt(summary["n"].clip(lower=1))
        summary["spread_lo"] = summary["center"] - sem
        summary["spread_hi"] = summary["center"] + sem
        summary["spread_label"] = "mean ± SEM"
    elif stat == "median":
        summary["center"] = grouped.median().reindex(range(N_BANDS))
        summary["spread_lo"] = grouped.quantile(0.25).reindex(range(N_BANDS))
        summary["spread_hi"] = grouped.quantile(0.75).reindex(range(N_BANDS))
        summary["spread_label"] = "median (IQR whiskers)"
    elif stat == "top10":
        def _top(s: pd.Series) -> pd.Series:
            top = s.nlargest(TOP_K)
            return pd.Series({
                "n_top": len(top),
                "mean": top.mean(),
                "sem": top.std() / np.sqrt(len(top)) if len(top) >= 2 else np.nan,
            })
        agg = grouped.apply(_top).unstack().reindex(range(N_BANDS))
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
    df = load_combined(benchmark)
    if df.empty:
        return None
    log_tokens = np.log10(df["total_inference_tokens"].to_numpy(dtype=float))
    edges = np.linspace(log_tokens.min(), log_tokens.max(), N_BANDS + 1)
    df["band"] = pd.cut(log_tokens, bins=edges, include_lowest=True, labels=False)
    centers = 10 ** ((edges[:-1] + edges[1:]) / 2)
    summary = _summarize(df, stat, centers)
    summary["band_lo_tokens"] = 10 ** edges[:-1]
    summary["band_hi_tokens"] = 10 ** edges[1:]
    return df, summary


def _draw_panel(ax, benchmark: str, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    for src, color in SOURCE_COLORS.items():
        sub = df[df["source"] == src]
        if sub.empty:
            continue
        ax.scatter(
            sub["total_inference_tokens"], sub["score_raw"],
            s=10, alpha=0.25, color=color, edgecolor="none",
            label=f"{src} (n={len(sub)})",
        )

    have = summary["n"].fillna(0) > 0
    cx = summary.loc[have, "band_center_tokens"].to_numpy()
    cy = summary.loc[have, "center"].to_numpy()
    lo = summary.loc[have, "spread_lo"].to_numpy()
    hi = summary.loc[have, "spread_hi"].to_numpy()
    yerr = np.where(np.isnan(np.vstack([cy - lo, hi - cy])), 0.0,
                    np.vstack([cy - lo, hi - cy]))
    ax.errorbar(
        cx, cy, yerr=yerr,
        fmt="o-", color="black", linewidth=1.8, markersize=6,
        capsize=3, capthick=1.0,
        label=f"band {summary['spread_label'].iloc[0]} (both sources pooled)",
    )
    for _, row in summary[have].iterrows():
        ax.annotate(
            f"{int(row['n'])}", (row["band_center_tokens"], row["center"]),
            textcoords="offset points", xytext=(0, 7),
            ha="center", fontsize=7, color="black",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total inference tokens")
    ax.set_ylabel("Score (fraction correct)")
    ax.set_title(benchmark, fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)


def main(stat: str = DEFAULT_STAT) -> None:
    if stat not in {"mean", "median", "top10"}:
        raise SystemExit(f"stat must be 'mean', 'median', or 'top10', got {stat!r}")

    panels = []
    for b in BENCHMARKS:
        result = _build_panel(b, stat)
        if result is None:
            print(f"skipping {b!r}")
            continue
        df, summary = result
        panels.append((b, df, summary))
    if not panels:
        return

    cols = 2
    rows_n = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 5 * rows_n), squeeze=False)
    for idx, (b, df, summary) in enumerate(panels):
        _draw_panel(axes[idx // cols][idx % cols], b, df, summary)
    for j in range(len(panels), rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        f"{stat.title()} score per compute band  "
        f"({N_BANDS} log-spaced bands, AA + log_viewer pooled)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if stat == "mean" else f"_{stat}"
    out_png = OUT_DIR / f"score_vs_compute_bands{suffix}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    combined = pd.concat(
        [s.assign(benchmark=b) for b, _, s in panels], ignore_index=False
    ).rename_axis("band_idx").reset_index()
    out_csv = OUT_DIR / f"score_vs_compute_bands{suffix}.csv"
    combined.to_csv(out_csv, index=False)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    for b, df, summary in panels:
        plottable = int(summary["n"].fillna(0).gt(0).sum())
        n_aa = (df["source"] == "aa").sum()
        n_lv = (df["source"] == "log_viewer").sum()
        print(f"  {b}: {len(df)} rows ({n_aa} aa + {n_lv} log_viewer), {plottable} plotted bands")


if __name__ == "__main__":
    stat = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_STAT
    main(stat)
