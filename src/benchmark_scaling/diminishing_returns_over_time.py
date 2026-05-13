"""
Diminishing-Returns Ratio on log_viewer_summary data
=====================================================

For each (benchmark, model) with >= 3 runs (sorted by total_avg_output_tokens
ascending, treating token spend as effort proxy), computes:

    R = (y_n - y_{n-1}) / (y_2 - y_1)

where y is accuracy and x is total_avg_output_tokens. This mirrors
src/test_time_compute/diminishing_returns_ratio.py but uses
the per-benchmark summaries under data/log_viewer_summary/ instead of AA Index.

R < 1: diminishing | R = 1: constant | R > 1: accelerating

Usage:
    python src/benchmark_scaling/diminishing_returns_over_time.py
    python src/benchmark_scaling/diminishing_returns_over_time.py --benchmark "GPQA Diamond"
    python src/benchmark_scaling/diminishing_returns_over_time.py --list
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = ROOT / "data" / "log_viewer_summary"
OUT_DIR = ROOT / "output" / "benchmark_scaling"

MIN_RUNS = 3


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "unknown"


def _short_model(m: str) -> str:
    return m.split("/", 1)[1] if "/" in m else m


def _load_all() -> list[dict]:
    return [json.load(p.open()) for p in sorted(SUMMARY_DIR.glob("*.json"))]


def compute_ratios(rows: list[dict], task: str) -> pd.DataFrame:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("task") != task:
            continue
        if r.get("accuracy") is None:
            continue
        if not (r.get("totals") or {}).get("total_avg_output_tokens"):
            continue
        by_model[r["model"]].append(r)

    out = []
    for model, runs in by_model.items():
        if len(runs) < MIN_RUNS:
            continue
        runs_sorted = sorted(runs, key=lambda r: r["totals"]["total_avg_output_tokens"])
        x = [r["totals"]["total_avg_output_tokens"] for r in runs_sorted]
        y = [r["accuracy"] for r in runs_sorted]
        n = len(runs_sorted)

        dy_first = y[1] - y[0]
        dy_last = y[n - 1] - y[n - 2]
        R = dy_last / dy_first if dy_first != 0 else float("nan")

        # release_date / aa_name from the matched AA row (set by match_models_to_aa.py)
        release_dates = [r.get("release_date") for r in runs_sorted if r.get("release_date")]
        aa_names = [r.get("aa_name") for r in runs_sorted if r.get("aa_name")]
        release_date = release_dates[0] if release_dates else None
        aa_name = aa_names[0] if aa_names else None

        out.append({
            "task": task,
            "model": model,
            "aa_name": aa_name,
            "release_date": release_date,
            "n_points": n,
            "x_first_lo": x[0], "x_first_hi": x[1],
            "x_last_lo": x[n - 2], "x_last_hi": x[n - 1],
            "y_first_lo": y[0], "y_first_hi": y[1],
            "y_last_lo": y[n - 2], "y_last_hi": y[n - 1],
            "dy_first": dy_first,
            "dy_last": dy_last,
            "R": R,
        })
    return pd.DataFrame(out)


def plot_ratios_over_time(ratios: pd.DataFrame, title: str, save_prefix: Path) -> None:
    """Scatter R against release_date, mirroring 13b_diminishing_returns_ratio.py."""
    df = ratios.dropna(subset=["R"]).copy()
    df["release_date_parsed"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date_parsed"])
    if df.empty:
        print(f"  no R-vs-time points for {title!r} (need release_date on summaries)")
        return

    fig, ax = plt.subplots(figsize=(11, 6.5))
    if "task" in df.columns and df["task"].nunique() > 1:
        tasks = sorted(df["task"].unique())
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        for i, t in enumerate(tasks):
            sub = df[df["task"] == t]
            ax.scatter(sub["release_date_parsed"], sub["R"],
                       color=palette[i % len(palette)], s=60, alpha=0.85,
                       edgecolors="white", linewidths=0.5, label=t, zorder=3)
    else:
        ax.scatter(df["release_date_parsed"], df["R"],
                   color="#1f77b4", s=70, alpha=0.85,
                   edgecolors="white", linewidths=0.5, zorder=3)

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7, zorder=1, label="Constant returns (R=1)")

    for _, row in df.iterrows():
        label = row.get("aa_name") or _short_model(row["model"])
        ax.annotate(label, (row["release_date_parsed"], row["R"]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=7, va="center", zorder=4)

    ax.set_xlabel("Release Date", fontsize=11)
    ax.set_ylabel("Diminishing-Returns Ratio (R)", fontsize=11)
    ax.set_title(f"Diminishing-Returns Ratio Over Time — {title}",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=8)

    fig.savefig(f"{save_prefix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_prefix}.png ({len(df)} points)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", help="run for a single benchmark (default: all)")
    parser.add_argument("--list", action="store_true", help="list benchmarks with >=1 model that has >=3 runs")
    args = parser.parse_args()

    rows = _load_all()
    tasks = sorted({r["task"] for r in rows if r.get("task")})

    if args.list:
        print("Benchmarks with >= 1 model having >= 3 runs:")
        for t in tasks:
            df = compute_ratios(rows, t)
            if not df.empty:
                print(f"  {len(df):3d} models  {t}")
        return

    targets = [args.benchmark] if args.benchmark else tasks
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_ratios = []
    for t in targets:
        if t not in tasks:
            print(f"unknown benchmark: {t!r}")
            continue
        df = compute_ratios(rows, t)
        if df.empty:
            print(f"skipping {t!r}: no models with >= {MIN_RUNS} runs")
            continue
        print(f"benchmark: {t}  ({len(df)} qualifying models)")
        for _, row in df.iterrows():
            print(f"  {_short_model(row['model'])}: n={row['n_points']}  "
                  f"Δy_first={row['dy_first']:+.3f}  Δy_last={row['dy_last']:+.3f}  R={row['R']:.3f}")
        out_csv = OUT_DIR / f"diminishing_returns__{_slug(t)}.csv"
        df.to_csv(out_csv, index=False)
        print(f"  saved {out_csv}")
        plot_ratios_over_time(df, t, OUT_DIR / f"diminishing_returns_over_time__{_slug(t)}")
        all_ratios.append(df)

    if all_ratios:
        combined = pd.concat(all_ratios, ignore_index=True)
        combined_csv = OUT_DIR / "diminishing_returns__all.csv"
        combined.to_csv(combined_csv, index=False)
        print(f"saved combined: {combined_csv}")
        plot_ratios_over_time(combined, "all benchmarks",
                              OUT_DIR / "diminishing_returns_over_time__all")


if __name__ == "__main__":
    main()
