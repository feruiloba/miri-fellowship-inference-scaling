"""
Plot the Box-Cox concavity `c` per release-quantile bucket, over time.

Reads:
  output/benchmark_vs_tokens/aa_evaluations/boxcox_fits_by_release_period.csv

Each row is a per-(benchmark, bucket) fit, where the bucket label encodes the
release-date span (e.g. "2024-04 → 2024-11  (12 models)"). We parse that span
and plot c at its midpoint, with a horizontal bar marking the full range.

Output:
  output/test_time_scaling_experiments/boxcox_c_over_release_buckets.png
"""

import math
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "output" / "benchmark_vs_tokens" / "aa_evaluations" / "boxcox_fits_by_release_period.csv"
OUT_PATH = ROOT / "output" / "test_time_scaling_experiments" / "boxcox_c_over_release_buckets.png"

LABEL_RE = re.compile(r"(\d{4}-\d{2})\s*→\s*(\d{4}-\d{2})")


def _parse_span(label: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    m = LABEL_RE.search(label)
    if not m:
        return None
    lo = pd.to_datetime(m.group(1), format="%Y-%m")
    hi = pd.to_datetime(m.group(2), format="%Y-%m")
    return lo, hi


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    spans = df["label"].map(_parse_span)
    df = df.assign(
        lo=spans.map(lambda s: s[0] if s else pd.NaT),
        hi=spans.map(lambda s: s[1] if s else pd.NaT),
    ).dropna(subset=["lo", "hi", "c"])
    df["mid"] = df["lo"] + (df["hi"] - df["lo"]) / 2

    PANEL_ORDER = [
        "gpqa-diamond",
        "aime-2025",
        "humanitys-last-exam",
        "artificial-analysis-long-context-reasoning",
    ]
    present = set(df["benchmark"].unique())
    benchmarks = [b for b in PANEL_ORDER if b in present]
    benchmarks += sorted(b for b in present if b not in PANEL_ORDER)
    cols = 2
    rows = math.ceil(len(benchmarks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.6 * rows), squeeze=False)

    MARKER_SIZE = 55

    for idx, bench in enumerate(benchmarks):
        ax = axes[idx // cols][idx % cols]
        sub = df[df["benchmark"] == bench].sort_values("mid")

        ax.plot(
            sub["mid"], sub["c"],
            color="#1f77b4", alpha=0.6, linewidth=1.5, zorder=2,
        )
        ax.scatter(
            sub["mid"], sub["c"],
            s=MARKER_SIZE, alpha=0.9, color="#1f77b4",
            edgecolor="white", linewidths=0.6, zorder=3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"c={row['c']:.2f}\nR²={row['r2']:.2f}\nn={int(row['n_models'])}",
                (row["mid"], row["c"]),
                xytext=(0, 8), textcoords="offset points",
                fontsize=7, ha="center", color="#1f77b4",
            )

        ax.set_ylim(-0.05, 1.1)
        ax.axhline(1.0, color="#d62728", linewidth=1, linestyle="--",
                   alpha=0.5, label="c = 1 (linear in log-tokens)")
        ax.axhline(0.0, color="#999999", linewidth=1, linestyle=":",
                   alpha=0.5, label="c → 0 (logarithmic limit)")

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")

        ax.set_title(bench, fontsize=11)
        ax.set_xlabel("Release date (bucket midpoint)")
        ax.set_ylabel("Box-Cox c")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.9)

    for j in range(len(benchmarks), rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        "Box-Cox concavity c per release-quantile bucket, per benchmark",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
