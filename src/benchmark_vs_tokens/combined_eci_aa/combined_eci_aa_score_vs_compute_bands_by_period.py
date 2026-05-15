"""
Per-release-period frontier curves of score vs test-time compute on the
combined AA + ECI log_viewer dataset (2 panels: GPQA Diamond and AIME).

Releases are grouped into 6-month periods (H1 = Jan-Jun, H2 = Jul-Dec).
For each (period, compute band) cell, the top-10 scores are averaged.

Output:
  output/benchmark_vs_tokens/combined_eci_aa/score_vs_compute_bands_by_period.{png,csv}
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _combined_eci_aa import BENCHMARKS, OUT_DIR, load_combined  # noqa: E402

N_BANDS = 8
TOP_K = 10
MIN_ROWS_PER_PERIOD = 5
MIN_BAND_N = 4


def _period_label(dt: pd.Series) -> pd.Series:
    year = dt.dt.year.astype("Int64")
    half = ((dt.dt.month - 1) // 6 + 1).astype("Int64")
    return year.astype(str) + "-H" + half.astype(str)


def _period_sort_key(label: str) -> tuple[int, int]:
    y, h = label.split("-H")
    return int(y), int(h)


def _build_panel(benchmark: str) -> tuple | None:
    df = load_combined(benchmark)
    if df.empty:
        return None
    dt = pd.to_datetime(df["release_date"], errors="coerce")
    df = df[dt.notna()].copy()
    df["period"] = _period_label(dt[dt.notna()])

    log_tokens = np.log10(df["total_inference_tokens"].to_numpy(dtype=float))
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
    pc = df.groupby("period").size()
    keep = pc[pc >= MIN_ROWS_PER_PERIOD].index.tolist()
    summary = summary[summary["period"].isin(keep)]
    summary = summary.sort_values(
        by=["period", "band_idx"],
        key=lambda s: s.map(_period_sort_key) if s.name == "period" else s,
    )
    return df, summary, pc


def _draw_panel(ax, benchmark: str, df: pd.DataFrame,
                summary: pd.DataFrame, pc: pd.Series) -> None:
    ax.scatter(df["total_inference_tokens"], df["score_raw"],
               s=8, alpha=0.10, color="#888888", edgecolor="none")
    periods_sorted = sorted(summary["period"].unique(), key=_period_sort_key)
    cmap = plt.get_cmap("viridis")
    for i, period in enumerate(periods_sorted):
        sub = summary[summary["period"] == period].sort_values("band_idx")
        color = cmap(i / max(len(periods_sorted) - 1, 1))
        ax.plot(sub["band_center_tokens"], sub["top_mean"],
                marker="o", color=color, linewidth=1.8, markersize=6,
                label=f"{period} (n={int(pc[period])})")
        for _, row in sub.iterrows():
            ax.annotate(f"{int(row['n_top'])}",
                        (row["band_center_tokens"], row["top_mean"]),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=6, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Total inference tokens")
    ax.set_ylabel(f"Top-{TOP_K} mean score per band")
    ax.set_title(benchmark, fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(title="release period", loc="lower right", fontsize=7, framealpha=0.9)


def main() -> None:
    panels = []
    for b in BENCHMARKS:
        result = _build_panel(b)
        if result is None:
            print(f"skipping {b!r}")
            continue
        panels.append((b, *result))
    if not panels:
        return

    cols = 2
    rows_n = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 5 * rows_n), squeeze=False)
    for idx, (b, df, summary, pc) in enumerate(panels):
        _draw_panel(axes[idx // cols][idx % cols], b, df, summary, pc)
    for j in range(len(panels), rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        f"Per-half-year compute-frontier curves  "
        f"(top-{TOP_K} mean, {N_BANDS} bands, n≥{MIN_BAND_N}, AA + log_viewer)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "score_vs_compute_bands_by_period.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    combined = pd.concat(
        [s.assign(benchmark=b) for b, _, s, _ in panels], ignore_index=True
    )
    out_csv = OUT_DIR / "score_vs_compute_bands_by_period.csv"
    combined.to_csv(out_csv, index=False)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    for b, df, summary, _ in panels:
        print(f"  {b}: {len(df)} rows, {len(summary)} plotted points")


if __name__ == "__main__":
    main()
