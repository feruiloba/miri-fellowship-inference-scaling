"""
Diminishing-Returns Ratio
==========================

For each model family with >= 3 effort-level variants (sorted by effort: low →
medium → high), computes:

    R = (y_3 - y_2) / (y_2 - y_1)

where (x_1, y_1), (x_2, y_2), (x_3, y_3) are consecutive (tokens, AA Index)
points with x_3 > x_2 > x_1.

Interpretation:
- R < 1  → diminishing returns (each effort bump adds less than the previous)
- R = 1  → constant returns
- R > 1  → accelerating returns

For families with 4 effort levels, every consecutive triple gets its own R.
"""

import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


AA_FILE = "data/artificial_analysis/artificial_analysis_llm_stats.csv"
TOKENS_FILE = "data/artificial_analysis/aa_output_tokens.csv"

LEVEL_ORDER = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4}


def load_effort_models():
    """Find effort-level variants and join AA Index with output tokens (mirrors 13a)."""
    aa = pd.read_csv(AA_FILE)
    tokens = pd.read_csv(TOKENS_FILE)

    effort_pattern = r'\((?:high|medium|low|xhigh|minimal)\)\s*$'
    aa_effort = aa[aa["name"].str.contains(effort_pattern, case=False, na=False)].copy()
    aa_effort["level"] = (
        aa_effort["name"]
        .str.extract(r'\((high|medium|low|xhigh|minimal)\)', flags=re.IGNORECASE)[0]
        .str.lower()
    )
    aa_effort["base"] = (
        aa_effort["name"]
        .str.replace(r'\s*\((?:high|medium|low|xhigh|minimal)\)\s*$', '',
                     regex=True, flags=re.IGNORECASE)
        .str.strip()
    )

    level_counts = aa_effort.groupby("base")["level"].nunique()
    multi_bases = level_counts[level_counts >= 3].index
    aa_effort = aa_effort[aa_effort["base"].isin(multi_bases)].copy()

    tokens["reasoning_tokens"] = pd.to_numeric(tokens["reasoning_tokens"], errors="coerce")
    tokens["answer_tokens"] = pd.to_numeric(tokens["answer_tokens"], errors="coerce")
    tokens["total_output_tokens"] = (
        tokens["reasoning_tokens"].fillna(0) + tokens["answer_tokens"].fillna(0)
    )

    def _norm(s):
        return re.sub(r'[\s\-_]+', '', str(s)).lower()

    token_lookup = {_norm(r["model"]): r["total_output_tokens"] for _, r in tokens.iterrows()}
    aa_effort["total_output_tokens"] = aa_effort["name"].apply(lambda n: token_lookup.get(_norm(n)))
    aa_effort["aa_index"] = pd.to_numeric(aa_effort["artificial_analysis_index"], errors="coerce")
    aa_effort["release_date"] = pd.to_datetime(aa_effort["release_date"], errors="coerce")

    valid = aa_effort[
        aa_effort["aa_index"].notna()
        & aa_effort["total_output_tokens"].notna()
        & (aa_effort["total_output_tokens"] > 0)
    ].copy()
    return valid


def compute_ratios(df):
    """For each family with >= 3 points, compute R comparing the last increment to the first.

    - 3 points: R = (y3 - y2) / (y2 - y1)
    - 4+ points: R = (y_n - y_{n-1}) / (y_2 - y_1)   (last jump vs first jump)
    """
    rows = []
    for base in sorted(df["base"].unique()):
        family = df[df["base"] == base].sort_values(
            "level", key=lambda s: s.map(LEVEL_ORDER)
        )
        n = len(family)
        if n < 3:
            continue

        x = family["total_output_tokens"].values.astype(float)
        y = family["aa_index"].values.astype(float)
        levels = family["level"].tolist()

        release_date = family["release_date"].min()

        # First jump: y[1] - y[0]; last jump: y[n-1] - y[n-2]
        dy_first = y[1] - y[0]
        dy_last = y[n - 1] - y[n - 2]
        R = dy_last / dy_first if dy_first != 0 else float("nan")

        rows.append({
            "base": base,
            "release_date": release_date,
            "n_points": n,
            "first_jump": f"{levels[0]} → {levels[1]}",
            "last_jump": f"{levels[n - 2]} → {levels[n - 1]}",
            "x_first_lo": x[0], "x_first_hi": x[1],
            "x_last_lo": x[n - 2], "x_last_hi": x[n - 1],
            "y_first_lo": y[0], "y_first_hi": y[1],
            "y_last_lo": y[n - 2], "y_last_hi": y[n - 1],
            "dy_first": dy_first,
            "dy_last": dy_last,
            "R": R,
        })
    return pd.DataFrame(rows)


def plot_ratios_over_time(ratios, save_prefix):
    """Plot the Diminishing-Returns Ratio (R) against release date."""
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_df = ratios.dropna(subset=["release_date", "R"]).copy()

    if len(plot_df) == 0:
        print("No valid data to plot R vs time.")
        return

    ax.scatter(plot_df["release_date"], plot_df["R"], color="#1f77b4", s=80, zorder=3, edgecolors="white")

    # Horizontal line for constant returns
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7, zorder=1, label="Constant returns (R=1)")

    # Annotate points
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["base"],
            (row["release_date"], row["R"]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=9, va="center", zorder=4
        )

    ax.set_xlabel("Release Date", fontsize=11)
    ax.set_ylabel("Diminishing-Returns Ratio (R)", fontsize=11)
    ax.set_title("Diminishing-Returns Ratio Over Time", fontsize=14, fontweight="bold")

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    ax.grid(True, alpha=0.2, which="both")
    ax.set_axisbelow(True)
    ax.legend()

    plt.savefig(f"{save_prefix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_prefix}.png")


if __name__ == "__main__":
    OUT_DIR = "output/test_time_scaling_experiments"
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_effort_models()
    ratios = compute_ratios(df)

    print("=" * 78)
    print("DIMINISHING-RETURNS RATIO: R = (last jump in y) / (first jump in y)")
    print("=" * 78)
    print("R < 1: diminishing  |  R = 1: constant  |  R > 1: accelerating\n")

    for _, row in ratios.iterrows():
        print(f"{row['base']} ({row['n_points']} points):")
        print(f"  First jump [{row['first_jump']}]: "
              f"y {row['y_first_lo']:.2f} → {row['y_first_hi']:.2f}  "
              f"(Δy = {row['dy_first']:+.2f})")
        print(f"  Last jump  [{row['last_jump']}]: "
              f"y {row['y_last_lo']:.2f} → {row['y_last_hi']:.2f}  "
              f"(Δy = {row['dy_last']:+.2f})")
        print(f"  R = {row['R']:.3f}")
        print()

    out_csv = f"{OUT_DIR}/diminishing_returns_ratio.csv"
    ratios.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    plot_prefix = f"{OUT_DIR}/diminishing_returns_ratio"
    plot_ratios_over_time(ratios, plot_prefix)
