"""
Derived ECI vs total inference tokens, from data/derived_eci/eci_from_benchmarks.csv.

One panel per `benchmark_source` (e.g. `aa_gpqa_diamond`, `log_viewer_swe_bench_verified`,
`aa_index`) so token semantics stay consistent within each panel: an `aa_index`
row's tokens cover all AA benchmarks summed, while `aa_gpqa_diamond` tokens
cover only that one benchmark. Plotting them on the same axes would distort
the relationship.

Output:
  output/benchmark_vs_tokens/derived_eci_vs_tokens.png
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ECI_CSV = ROOT / "data" / "derived_eci" / "eci_from_benchmarks.csv"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens" / "derived_eci"

MIN_POINTS_PER_PANEL = 5  # drop sources too sparse to be meaningful


def _panel_title(src: str) -> str:
    if src == "aa_index":
        return "AA Index (aggregated)"
    if src.startswith("aa_"):
        return f"AA · {src[len('aa_'):].replace('_', ' ')}"
    if src.startswith("log_viewer_"):
        return f"Log viewer · {src[len('log_viewer_'):].replace('_', ' ')}"
    return src


def main() -> None:
    df = pd.read_csv(ECI_CSV)
    df = df.dropna(subset=["eci_estimated", "total_inference_tokens"])
    df = df[df["total_inference_tokens"] > 0]

    sources = (
        df["benchmark_source"]
        .value_counts()
        .loc[lambda s: s >= MIN_POINTS_PER_PANEL]
        .index.tolist()
    )
    if not sources:
        print("No benchmark_source with enough points to plot.")
        return

    n = len(sources)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4 * rows), squeeze=False)

    # Shared y-range so panels are visually comparable
    y_lo, y_hi = df["eci_estimated"].min(), df["eci_estimated"].max()
    y_pad = 0.04 * (y_hi - y_lo)

    for i, src in enumerate(sources):
        ax = axes[i // cols][i % cols]
        sub = df[df["benchmark_source"] == src]
        x = sub["total_inference_tokens"].to_numpy(dtype=float)
        y = sub["eci_estimated"].to_numpy(dtype=float)

        ax.scatter(x, y, s=18, alpha=0.55, color="#1f77b4", edgecolor="none")

        # log-linear fit for orientation; skip if tokens have no spread
        if x.min() > 0 and x.max() / x.min() > 2:
            lx = np.log10(x)
            slope, intercept = np.polyfit(lx, y, 1)
            r = np.corrcoef(lx, y)[0, 1]
            xs = np.linspace(lx.min(), lx.max(), 60)
            ax.plot(
                10 ** xs, intercept + slope * xs,
                color="black", linewidth=1.1, linestyle="--",
                label=f"slope={slope:.2f}  r={r:.2f}",
            )
            ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

        ax.set_xscale("log")
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.set_title(f"{_panel_title(src)}  (n={len(sub)})", fontsize=10)
        ax.set_xlabel("Total inference tokens")
        ax.set_ylabel("ECI (estimated)")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # Hide unused subplots
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("Estimated ECI vs total inference tokens, by benchmark source", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "derived_eci_vs_tokens.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path} ({len(df)} points across {n} panels)")


if __name__ == "__main__":
    main()
