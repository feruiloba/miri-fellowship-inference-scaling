"""
Test-Time Compute Scaling vs. AA Index
=======================================

For models that have low/medium/high effort-level variants, plots the
relationship between test-time compute (or output tokens as proxy) and
AA Index score. Each model family is connected by a line to show how
capability scales with increased inference compute.

Test-time compute is estimated as:  2 × parameters × total_output_tokens
For models without known parameter counts, total output tokens alone is
used (proportional to test-time compute since params are constant within
a model family).

Data Sources:
- data/artificial_analysis/artificial_analysis_llm_stats.csv
    - 'artificial_analysis_index': AA capability score
    - effort-level variants: (high), (medium), (low)
- data/artificial_analysis/aa_output_tokens.csv
    - 'reasoning_tokens' + 'answer_tokens': total output per variant
- data/merged_datasets.csv
    - 'Parameters': model parameter count (where available)

Author: Analysis conducted with Claude (Anthropic)
"""

import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

AA_FILE = "data/artificial_analysis/artificial_analysis_llm_stats.csv"
TOKENS_FILE = "data/artificial_analysis/aa_output_tokens.csv"
MERGED_FILE = "data/merged_datasets.csv"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_effort_models():
    """
    Find models with multiple effort levels (low/medium/high) and join
    their AA Index scores with output token counts.
    """
    aa = pd.read_csv(AA_FILE)
    tokens = pd.read_csv(TOKENS_FILE)
    merged = pd.read_csv(MERGED_FILE)
    merged["Parameters"] = pd.to_numeric(merged["Parameters"], errors="coerce")

    # Parse effort levels from AA model names
    effort_pattern = r'\((?:high|medium|low|xhigh|minimal)\)\s*$'
    aa_effort = aa[aa["name"].str.contains(effort_pattern, case=False, na=False)].copy()
    aa_effort["level"] = (
        aa_effort["name"]
        .str.extract(r'\((high|medium|low|xhigh|minimal)\)', flags=re.IGNORECASE)[0]
        .str.lower()
    )
    aa_effort["base"] = (
        aa_effort["name"]
        .str.replace(r'\s*\((?:high|medium|low|xhigh|minimal)\)\s*$', '', regex=True, flags=re.IGNORECASE)
        .str.strip()
    )

    # Keep only base models with >= 2 effort levels
    level_counts = aa_effort.groupby("base")["level"].nunique()
    multi_bases = level_counts[level_counts >= 2].index
    aa_effort = aa_effort[aa_effort["base"].isin(multi_bases)].copy()

    # Parse token data — normalize names for matching
    tokens["reasoning_tokens"] = pd.to_numeric(tokens["reasoning_tokens"], errors="coerce")
    tokens["answer_tokens"] = pd.to_numeric(tokens["answer_tokens"], errors="coerce")
    tokens["total_output_tokens"] = (
        tokens["reasoning_tokens"].fillna(0) + tokens["answer_tokens"].fillna(0)
    )

    def _norm(s):
        return re.sub(r'[\s\-_]+', '', str(s)).lower()

    # Build token lookup
    token_lookup = {}
    for _, row in tokens.iterrows():
        token_lookup[_norm(row["model"])] = row["total_output_tokens"]

    # Match each AA effort variant to its token count
    aa_effort["total_output_tokens"] = aa_effort["name"].apply(
        lambda n: token_lookup.get(_norm(n))
    )

    # Get parameter counts from merged dataset
    param_lookup = {}
    for base in multi_bases:
        hits = merged[
            merged["Model"].str.contains(re.escape(base), case=False, na=False)
            & merged["Parameters"].notna()
        ]
        if len(hits) > 0:
            param_lookup[base] = hits["Parameters"].iloc[0]

    aa_effort["parameters"] = aa_effort["base"].map(param_lookup)

    # Compute test-time compute where possible
    aa_effort["test_time_flops"] = (
        2 * aa_effort["parameters"] * aa_effort["total_output_tokens"]
    )

    # AA Index
    aa_effort["aa_index"] = pd.to_numeric(
        aa_effort["artificial_analysis_index"], errors="coerce"
    )

    # Filter to rows with both AA index and tokens
    valid = aa_effort[
        aa_effort["aa_index"].notna()
        & aa_effort["total_output_tokens"].notna()
        & (aa_effort["total_output_tokens"] > 0)
    ].copy()

    return valid


# =============================================================================
# PLOTTING
# =============================================================================

LEVEL_ORDER = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4}
LEVEL_MARKERS = {"minimal": "d", "low": "v", "medium": "s", "high": "^", "xhigh": "^"}

# Color palette for model families
FAMILY_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
]


def plot_scaling(df, save_prefix):
    """
    Two-panel figure:
      Left: output tokens vs AA Index (all models)
      Right: test-time FLOPs vs AA Index (only models with known params)
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    bases = sorted(df["base"].unique())
    color_map = {b: FAMILY_COLORS[i % len(FAMILY_COLORS)] for i, b in enumerate(bases)}

    _plot_panel(ax, df, "total_output_tokens", color_map,
                xlabel="Total output tokens",
                title="Output tokens vs. AA Index by effort level")

    fig.suptitle(
        "Test-time compute scaling: how capability changes with inference budget",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plt.savefig(f"{save_prefix}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{save_prefix}.svg", format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_prefix}.png / .svg")


def _plot_panel(ax, df, x_col, color_map, xlabel, title, x_unit="tokens"):
    """Plot one panel: x_col vs AA Index, lines connecting effort levels."""
    bases = sorted(df["base"].unique())

    for base in bases:
        family = df[df["base"] == base].sort_values(
            "level", key=lambda s: s.map(LEVEL_ORDER)
        )
        color = color_map[base]

        # Draw connecting line
        ax.plot(
            family[x_col], family["aa_index"],
            color=color, linewidth=2, alpha=0.7, zorder=2,
        )

        # Draw points with markers per level
        for _, row in family.iterrows():
            marker = LEVEL_MARKERS.get(row["level"], "o")
            ax.scatter(
                row[x_col], row["aa_index"],
                color=color, marker=marker, s=80, zorder=3,
                edgecolors="white", linewidths=0.5,
            )

        # Label the line at the high-effort end
        high = family[family["level"].isin(["high", "xhigh"])]
        if len(high) == 0:
            high = family.iloc[[-1]]
        label_row = high.iloc[-1]

        # Compute slope for annotation
        if len(family) >= 2:
            log_x = np.log10(family[x_col].values)
            aa_vals = family["aa_index"].values
            if log_x[-1] != log_x[0]:
                slope = (aa_vals[-1] - aa_vals[0]) / (log_x[-1] - log_x[0])
                label = f"{base}\n(+{slope:.1f} AA/10× {x_unit})"
            else:
                label = base
        else:
            label = base

        ax.annotate(
            label,
            (label_row[x_col], label_row["aa_index"]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold",
            va="center",
        )

    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("AA Index", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.2, which="both")
    ax.set_axisbelow(True)

    # Legend for effort markers
    from matplotlib.lines import Line2D
    has_high = "high" in df["level"].values or "xhigh" in df["level"].values
    legend_levels = [
        ("minimal", "Minimal"),
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
    ]
    marker_legend = [
        Line2D([0], [0], marker=LEVEL_MARKERS[key], color="gray", linestyle="None",
               markersize=8, label=label)
        for key, label in legend_levels
        if (key == "high" and has_high) or (key != "high" and key in df["level"].values)
    ]
    ax.legend(handles=marker_legend, loc="lower right", fontsize=9,
              framealpha=0.9, title="Effort level")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    OUT_DIR = "output/13_inference_scaling"
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_effort_models()

    print("=" * 70)
    print("TEST-TIME COMPUTE SCALING VS. AA INDEX")
    print("=" * 70)
    print(f"Total variants: {len(df)}")
    print(f"Model families: {df['base'].nunique()}")
    print(f"With known params: {df['parameters'].notna().sum()}")
    print()

    # Print per-family summary
    for base in sorted(df["base"].unique()):
        family = df[df["base"] == base].sort_values(
            "level", key=lambda s: s.map(LEVEL_ORDER)
        )
        levels = family["level"].tolist()
        aa_scores = family["aa_index"].tolist()
        tokens = family["total_output_tokens"].tolist()
        params = family["parameters"].iloc[0] if family["parameters"].notna().any() else None

        # Compute scaling slope
        if len(family) >= 2:
            log_t = np.log10(np.array(tokens, dtype=float))
            aa = np.array(aa_scores, dtype=float)
            slope = (aa[-1] - aa[0]) / (log_t[-1] - log_t[0]) if log_t[-1] != log_t[0] else float("nan")
        else:
            slope = float("nan")

        params_str = f"{params:.0e}" if params else "unknown"
        print(f"{base}:")
        print(f"  Params: {params_str}")
        print(f"  Levels: {', '.join(levels)}")
        print(f"  AA Index: {' → '.join(f'{s:.1f}' for s in aa_scores)}")
        print(f"  Tokens: {' → '.join(f'{t:.1e}' for t in tokens)}")
        print(f"  Slope: {slope:+.1f} AA per 10× tokens")
        print()

    save_prefix = f"{OUT_DIR}/test_time_compute_vs_aa_index"
    plot_scaling(df, save_prefix)

    # Save table
    table = df[["base", "name", "level", "aa_index", "total_output_tokens",
                 "parameters", "test_time_flops"]].sort_values(["base", "level"])
    table.to_csv(f"{save_prefix}_table.csv", index=False)
    print(f"Saved: {save_prefix}_table.csv")

    print("\nDone.")
