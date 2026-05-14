"""Plot model score vs total output tokens for a given benchmark, from ECI logs.

Data source: per-run summaries in data/log_viewer_summary/, which are
distilled from the ECI (eci-public) log-viewer JSON exports. Each summary
gives one model × benchmark run with its accuracy and total_avg_output_tokens.

Usage:
    python src/benchmark_vs_tokens/score_vs_tokens_eci.py <benchmark>
    python src/benchmark_vs_tokens/score_vs_tokens_eci.py --list
    python src/benchmark_vs_tokens/score_vs_tokens_eci.py --all

Multiple ECI runs of the same model on the same benchmark are connected with
a line so it's easy to see how a single model's score changes with token
spend (e.g. across reasoning-effort levels or repeat runs).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = ROOT / "data" / "log_viewer_summary"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens"

FAMILY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def _load_all() -> list[dict]:
    rows = []
    for p in sorted(SUMMARY_DIR.glob("*.json")):
        d = json.load(p.open())
        rows.append(d)
    return rows


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "unknown"


def _short_model_name(model: str) -> str:
    """Drop the provider prefix; keep model id."""
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def plot_benchmark(task: str, rows: list[dict], min_runs: int = 1, svg: bool = False) -> Path | None:
    df = [
        r for r in rows
        if r.get("task") == task
        and r.get("accuracy") is not None
        and (r.get("totals") or {}).get("total_avg_output_tokens")
    ]
    if not df:
        print(f"  no plottable rows for {task!r}")
        return None

    # group by model so multi-run models share a color and get connected
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in df:
        by_model[r["model"]].append(r)

    if min_runs > 1:
        by_model = {m: rs for m, rs in by_model.items() if len(rs) >= min_runs}
        if not by_model:
            print(f"  no models with >= {min_runs} runs for {task!r}")
            return None
        df = [r for rs in by_model.values() for r in rs]

    models = sorted(by_model.keys())
    color_for = {m: FAMILY_COLORS[i % len(FAMILY_COLORS)] for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(12, 8))

    for model in models:
        group = sorted(
            by_model[model],
            key=lambda r: r["totals"]["total_avg_output_tokens"],
        )
        xs = [r["totals"]["total_avg_output_tokens"] for r in group]
        ys = [r["accuracy"] for r in group]
        color = color_for[model]

        if len(group) >= 2:
            ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.6, zorder=2)

        ax.scatter(
            xs, ys,
            color=color, s=50, alpha=0.9, zorder=3,
            edgecolors="white", linewidths=0.5,
        )

        # label the rightmost point per model
        ax.annotate(
            _short_model_name(model),
            (xs[-1], ys[-1]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=7, color=color, va="center",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total avg output tokens (per run, summed across samples)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(
        f"Score vs. tokens (ECI logs) — {task}  (n={len(df)} runs, {len(models)} models)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.2, which="both")
    ax.set_axisbelow(True)
    ax.set_ylim(-0.02, 1.02)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"__min{min_runs}runs" if min_runs > 1 else ""
    out_base = OUT_DIR / f"score_vs_tokens_eci__{_slug(task)}{suffix}"
    fig.savefig(f"{out_base}.png", dpi=150, bbox_inches="tight")
    if svg:
        fig.savefig(f"{out_base}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_base}.png ({len(df)} runs, {len(models)} models)")
    return out_base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", nargs="?", help="benchmark/task name (e.g. 'GPQA Diamond')")
    parser.add_argument("--list", action="store_true", help="list available benchmarks and exit")
    parser.add_argument("--all", action="store_true", help="generate plots for all benchmarks")
    parser.add_argument("--min-runs", type=int, default=1,
                        help="only include models with at least this many runs on the benchmark")
    parser.add_argument("--svg", action="store_true", help="also save an SVG alongside the PNG")
    args = parser.parse_args()

    rows = _load_all()
    tasks = Counter(r.get("task") for r in rows if r.get("task"))

    if args.list or (not args.benchmark and not args.all):
        print("Available benchmarks (count of runs):")
        for t, c in tasks.most_common():
            print(f"  {c:4d}  {t}")
        return

    targets = list(tasks) if args.all else [args.benchmark]
    for t in targets:
        if t not in tasks:
            print(f"unknown benchmark: {t!r} (use --list to see options)")
            continue
        print(f"plotting: {t}")
        plot_benchmark(t, rows, min_runs=args.min_runs, svg=args.svg)


if __name__ == "__main__":
    main()
