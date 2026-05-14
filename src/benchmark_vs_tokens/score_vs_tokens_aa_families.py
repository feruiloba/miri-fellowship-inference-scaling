"""Plot benchmark score vs total output tokens for AA model families.

Reads per-benchmark CSVs from data/artificial_analysis/evaluation_families/
(produced by src/data_processing/prepare/find_aa_evaluation_families.py),
plots each variant as a point, and connects variants within the same family
with a line — so you can see how a family's score scales with token spend
across reasoning-effort levels.

Usage:
    python src/benchmark_vs_tokens/score_vs_tokens_aa_families.py <benchmark>
    python src/benchmark_vs_tokens/score_vs_tokens_aa_families.py --list
    python src/benchmark_vs_tokens/score_vs_tokens_aa_families.py --all
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FAMILY_DIR = ROOT / "data" / "artificial_analysis" / "evaluation_families"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens"

FAMILY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "unknown"


def _list_benchmarks() -> list[Path]:
    return sorted(FAMILY_DIR.glob("*.csv"))


def plot_benchmark(csv_path: Path) -> Path | None:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["total_output_tokens", "score_raw"])
    df = df[df["total_output_tokens"] > 0]
    if df.empty:
        print(f"  no plottable rows for {csv_path.stem}")
        return None

    families = sorted(df["family_id"].unique())
    color_for = {f: FAMILY_COLORS[i % len(FAMILY_COLORS)] for i, f in enumerate(families)}

    fig, ax = plt.subplots(figsize=(12, 8))
    for fam in families:
        sub = df[df["family_id"] == fam].sort_values("total_output_tokens")
        xs = sub["total_output_tokens"].tolist()
        ys = sub["score_raw"].tolist()
        color = color_for[fam]

        if len(sub) >= 2:
            ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.6, zorder=2)
        ax.scatter(
            xs, ys, color=color, s=50, alpha=0.9, zorder=3,
            edgecolors="white", linewidths=0.5,
        )
        # label rightmost point with the family id
        ax.annotate(
            fam,
            (xs[-1], ys[-1]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=7, color=color, va="center",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total output tokens (reasoning + answer, per model)", fontsize=11)
    ax.set_ylabel("Benchmark score (0-1)", fontsize=11)
    ax.set_title(
        f"Score vs. tokens — {csv_path.stem}  "
        f"(n={len(df)} variants, {len(families)} families)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.2, which="both")
    ax.set_axisbelow(True)
    ax.set_ylim(-0.02, 1.02)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUT_DIR / f"score_vs_tokens_aa_families__{_slug(csv_path.stem)}"
    fig.savefig(f"{out_base}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_base}.png ({len(df)} variants, {len(families)} families)")
    return out_base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", nargs="?", help="eval slug (e.g. 'aime-2025')")
    parser.add_argument("--list", action="store_true", help="list available benchmarks and exit")
    parser.add_argument("--all", action="store_true", help="plot all benchmarks")
    args = parser.parse_args()

    csvs = _list_benchmarks()
    by_slug = {p.stem: p for p in csvs}

    if args.list or (not args.benchmark and not args.all):
        print("Available benchmarks (variant rows):")
        for p in csvs:
            n = sum(1 for _ in p.open()) - 1
            print(f"  {n:4d}  {p.stem}")
        return

    targets = list(by_slug) if args.all else [args.benchmark]
    for t in targets:
        if t not in by_slug:
            print(f"unknown benchmark: {t!r} (use --list to see options)")
            continue
        print(f"plotting: {t}")
        plot_benchmark(by_slug[t])


if __name__ == "__main__":
    main()
