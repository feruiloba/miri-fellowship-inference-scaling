"""
Score vs. inference tokens with one OLS line per half-year (semester).

Procedure (per benchmark):
  1. For each semester, find the peak = the run with the highest score that
     semester. Drop every run with more inference tokens than the peak run
     (keep only runs up to and including the peak's token budget).
  2. Fit a straight line score = a + b * log10(tokens) per semester across the
     surviving runs.

Output:
  output/benchmark_vs_tokens/aa_evaluations/linear_fits_by_release_semester.{png,csv}
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EVAL_CSV = ROOT / "data" / "artificial_analysis" / "aa_evaluations_combined.csv"
STATS_CSV = ROOT / "data" / "artificial_analysis" / "artificial_analysis_llm_stats.csv"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens" / "aa_evaluations"

BENCHMARKS = [
    "gpqa-diamond",
    "aime-2025",
    "humanitys-last-exam",
    "artificial-analysis-long-context-reasoning",
]
MIN_POINTS_FOR_FIT = 2       # need at least this many surviving runs to fit a line


def _period_label(dt: pd.Series) -> pd.Series:
    year = dt.dt.year.astype("Int64")
    half = ((dt.dt.month - 1) // 6 + 1).astype("Int64")
    return year.astype(str) + "-H" + half.astype(str)


def _period_sort_key(label: str) -> tuple[int, int]:
    y, h = label.split("-H")
    return int(y), int(h)


def _load(benchmark: str) -> pd.DataFrame:
    ev = pd.read_csv(EVAL_CSV)
    ev = ev[ev["benchmark"] == benchmark].copy()
    ev = ev.dropna(subset=["total_output_tokens", "score_raw", "model_slug"])
    ev = ev[ev["total_output_tokens"] > 0]

    stats = pd.read_csv(STATS_CSV, usecols=["slug", "release_date"])
    stats = stats.rename(columns={"slug": "model_slug"}).dropna(subset=["release_date"])
    ev = ev.merge(stats, on="model_slug", how="left")
    dt = pd.to_datetime(ev["release_date"], errors="coerce")
    ev = ev[dt.notna()].copy()
    ev["period"] = _period_label(dt[dt.notna()])
    ev["log_tokens"] = np.log10(ev["total_output_tokens"].astype(float))
    return ev


def _truncate_after_peak(sub: pd.DataFrame) -> pd.DataFrame:
    """Keep only runs with at most as many tokens as the highest-scoring run
    that semester (drop everything past the peak)."""
    peak_log_tokens = sub.loc[sub["score_raw"].idxmax(), "log_tokens"]
    return sub[sub["log_tokens"] <= peak_log_tokens]


def _fit_per_period(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit OLS per semester on the runs up to that semester's peak-score budget."""
    fits = []
    points: dict[str, pd.DataFrame] = {}
    for period, sub in df.groupby("period"):
        kept = _truncate_after_peak(sub)
        n = len(kept)
        if n < MIN_POINTS_FOR_FIT:
            continue
        x = kept["log_tokens"].to_numpy()
        y = kept["score_raw"].to_numpy()
        if x.max() - x.min() < 1e-9:
            continue
        b, a = np.polyfit(x, y, 1)
        y_hat = a + b * x
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        fits.append({
            "period": period,
            "n_fit_points": n,
            "n_models": int(kept["model_slug"].nunique()),
            "intercept": a,
            "slope_per_decade": b,
            "r2": r2,
            "x_min": float(x.min()),
            "x_max": float(x.max()),
        })
        points[period] = kept
    return pd.DataFrame(fits), points


def _draw_panel(ax, benchmark: str, df: pd.DataFrame,
                fits: pd.DataFrame, points: dict[str, pd.DataFrame]) -> None:
    ax.scatter(
        df["total_output_tokens"], df["score_raw"],
        s=8, alpha=0.10, color="#888888", edgecolor="none",
    )

    periods = sorted(fits["period"].tolist(), key=_period_sort_key)
    cmap = plt.get_cmap("viridis")
    for i, period in enumerate(periods):
        color = cmap(i / max(len(periods) - 1, 1))
        row = fits[fits["period"] == period].iloc[0]
        pts = points[period]
        ax.scatter(
            pts["total_output_tokens"], pts["score_raw"],
            s=18, color=color, alpha=0.75, edgecolor="white", linewidth=0.3,
        )
        xs = np.linspace(row["x_min"], row["x_max"], 50)
        ys = row["intercept"] + row["slope_per_decade"] * xs
        ax.plot(
            10 ** xs, ys, color=color, linewidth=2.0,
            label=(
                f"{period}: slope={row['slope_per_decade']:+.3f}/dec, "
                f"R²={row['r2']:.2f}, n={int(row['n_fit_points'])}"
            ),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total inference tokens")
    ax.set_ylabel("Score (raw)")
    ax.set_title(benchmark, fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(title="semester fit (runs up to peak)",
              loc="lower right", fontsize=7, framealpha=0.9)


def main(benchmarks: list[str] = BENCHMARKS) -> None:
    panels = []
    all_fits = []
    for b in benchmarks:
        df = _load(b)
        if df.empty:
            print(f"skipping {b!r} (no rows)")
            continue
        fits, points = _fit_per_period(df)
        if fits.empty:
            print(f"skipping {b!r} (no period had enough points)")
            continue
        panels.append((b, df, fits, points))
        all_fits.append(fits.assign(benchmark=b))

    if not panels:
        print("Nothing to plot.")
        return

    cols = 2
    rows_n = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 5 * rows_n),
                             squeeze=False)
    for idx, (b, df, fits, points) in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        _draw_panel(ax, b, df, fits, points)
    for j in range(len(panels), rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        "Per-semester OLS fit of score vs log10(tokens)  "
        "(runs truncated after each semester's peak-score budget)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "linear_fits_by_release_semester.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    combined = pd.concat(all_fits, ignore_index=True)
    combined = combined.sort_values(
        by=["benchmark", "period"],
        key=lambda s: s.map(_period_sort_key) if s.name == "period" else s,
    )
    out_csv = OUT_DIR / "linear_fits_by_release_semester.csv"
    combined.to_csv(out_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    for b, df, fits, _ in panels:
        print(f"  {b}: {len(df)} runs, {len(fits)} semester fits")


if __name__ == "__main__":
    main()
