"""
Plot the Box-Cox concavity parameter `c` vs model release date,
one panel per benchmark we've fit a curve for.

`c` is the diminishing-returns exponent in
    f(x) = m·((1 + x − h)^c − 1)/c + C
with x = log₁₀ tokens. Bounds: c ∈ [1e-3, 1]. c = 1 means score climbs
linearly with log-tokens (no diminishing returns); c → 0 means logarithmic
(strong diminishing returns).

Reads:
  output/test_time_scaling_experiments/fit_<bench>_boxcox_params.csv

Output:
  output/test_time_scaling_experiments/boxcox_c_vs_release_date.png
"""

import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIT_DIR = ROOT / "output" / "test_time_scaling_experiments"

# (csv stem → (panel title, identifier-column))
SOURCES = [
    ("fit_aa_index_boxcox_params",                              "AA Index",                  "base"),
    ("fit_gpqa_diamond_boxcox_params",                          "GPQA Diamond",              "family_id"),
    ("fit_aime_2025_boxcox_params",                             "AIME 2025",                 "family_id"),
    ("fit_artificial_analysis_long_context_reasoning_boxcox_params",
     "AA long-context reasoning", "family_id"),
]


def _load(stem: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(FIT_DIR / f"{stem}.csv")
    df = df.dropna(subset=["c", "release_date"])
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])
    df = df.rename(columns={label_col: "family"})
    return df


def main() -> None:
    panels = [(title, _load(stem, col)) for stem, title, col in SOURCES]
    panels = [(t, d) for t, d in panels if not d.empty]
    if not panels:
        print("No fit CSVs found.")
        return

    cols = 2
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.6 * rows), squeeze=False)

    # Marker size encodes n (number of effort variants in the fit)
    def _size(n: float) -> float:
        return 30 + 30 * max(n - 2, 0)  # n=2 → 30, n=3 → 60, n=4 → 90

    for idx, (title, df) in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        # Trustworthy (n ≥ 3) plotted on top with stronger color
        weak = df[df["n"] < 3]
        strong = df[df["n"] >= 3]
        if not weak.empty:
            ax.scatter(
                weak["release_date"], weak["c"],
                s=weak["n"].apply(_size), alpha=0.25, color="#888888",
                edgecolor="none", label=f"n=2 (degenerate fit, {len(weak)})",
            )
        if not strong.empty:
            ax.scatter(
                strong["release_date"], strong["c"],
                s=strong["n"].apply(_size), alpha=0.85, color="#1f77b4",
                edgecolor="white", linewidths=0.6,
                label=f"n≥3 ({len(strong)})",
            )
            for _, row in strong.iterrows():
                ax.annotate(
                    row["family"], (row["release_date"], row["c"]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=7, va="center", color="#1f77b4",
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

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Model release date")
        ax.set_ylabel("Box-Cox c")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.9)

    for j in range(len(panels), rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("Box-Cox concavity c vs release date, per benchmark", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = FIT_DIR / "boxcox_c_vs_release_date.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
