"""
Plot how the per-semester linear fits evolve over time.

Reads:
  output/benchmark_vs_tokens/aa_evaluations/linear_fits_by_release_semester.csv
  data/artificial_analysis/aa_evaluations_combined.csv
  data/artificial_analysis/artificial_analysis_llm_stats.csv

Outputs three panels vs semester, one line per benchmark:
  1. slope (per decade of log10 tokens)
  2. intercept
  3. argmax tokens — the inference-token count of the single run that
     achieved the highest observed score for that (semester, benchmark).

Output:
  output/test_time_scaling_experiments/semester_fit_trends.{png,csv}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "output" / "benchmark_vs_tokens" / "aa_evaluations" / "linear_fits_by_release_semester.csv"
EVAL_CSV = ROOT / "data" / "artificial_analysis" / "aa_evaluations_combined.csv"
STATS_CSV = ROOT / "data" / "artificial_analysis" / "artificial_analysis_llm_stats.csv"
OUT_DIR = ROOT / "output" / "test_time_scaling_experiments"


def _period_sort_key(label: str) -> tuple[int, int]:
    y, h = label.split("-H")
    return int(y), int(h)


def _period_to_decimal(label: str) -> float:
    y, h = _period_sort_key(label)
    return y + (0.0 if h == 1 else 0.5)


def _period_label(dt: pd.Series) -> pd.Series:
    year = dt.dt.year.astype("Int64")
    half = ((dt.dt.month - 1) // 6 + 1).astype("Int64")
    return year.astype(str) + "-H" + half.astype(str)


def _argmax_tokens_per_cell() -> pd.DataFrame:
    """For each (benchmark, semester), find the single run with the highest
    score_raw and return its total_output_tokens."""
    ev = pd.read_csv(EVAL_CSV)
    ev = ev.dropna(subset=["total_output_tokens", "score_raw", "model_slug"])
    ev = ev[ev["total_output_tokens"] > 0]

    stats = pd.read_csv(STATS_CSV, usecols=["slug", "release_date"])
    stats = stats.rename(columns={"slug": "model_slug"}).dropna(subset=["release_date"])
    ev = ev.merge(stats, on="model_slug", how="left")
    dt = pd.to_datetime(ev["release_date"], errors="coerce")
    ev = ev[dt.notna()].copy()
    ev["period"] = _period_label(dt[dt.notna()])

    idx = ev.groupby(["benchmark", "period"])["score_raw"].idxmax()
    out = ev.loc[idx, ["benchmark", "period", "total_output_tokens",
                       "score_raw", "model_slug"]].rename(
        columns={"total_output_tokens": "argmax_tokens",
                 "score_raw": "argmax_score",
                 "model_slug": "argmax_model"}
    ).reset_index(drop=True)

    # Max inference tokens used by any model in each (benchmark, semester).
    max_tok = (
        ev.groupby(["benchmark", "period"])["total_output_tokens"]
          .max().reset_index(name="max_tokens")
    )
    return out.merge(max_tok, on=["benchmark", "period"], how="left")


def main() -> None:
    df = pd.read_csv(IN_CSV)
    argmax_df = _argmax_tokens_per_cell()
    df = df.merge(argmax_df, on=["benchmark", "period"], how="left")
    df["period_decimal"] = df["period"].map(_period_to_decimal)
    df = df.sort_values(["benchmark", "period_decimal"])

    # Deltas vs immediately prior semester within the same benchmark.
    grp = df.groupby("benchmark", sort=False)
    df["delta_slope"] = grp["slope_per_decade"].diff()
    df["delta_intercept"] = grp["intercept"].diff()
    df["argmax_token_ratio"] = df["argmax_tokens"] / grp["argmax_tokens"].shift(1)

    benchmarks = sorted(df["benchmark"].unique())
    cmap = plt.get_cmap("tab10")
    bench_color = {b: cmap(i) for i, b in enumerate(benchmarks)}

    # 4 rows (one per benchmark) × 2 cols (slope, argmax_tokens).
    n_b = len(benchmarks)
    fig, axes = plt.subplots(n_b, 2, figsize=(12, 3.0 * n_b), sharex=True,
                             squeeze=False)

    def _label(d: float) -> str:
        y = int(d)
        return f"{y}-H{1 if (d - y) < 0.25 else 2}"

    semester_ticks = sorted(df["period_decimal"].unique())
    semester_labels = [_label(d) for d in semester_ticks]

    for i, b in enumerate(benchmarks):
        sub = df[df["benchmark"] == b].sort_values("period_decimal")
        color = bench_color[b]
        ax_s, ax_a = axes[i]

        ax_s.plot(sub["period_decimal"], sub["slope_per_decade"],
                  "o-", color=color, linewidth=1.8, markersize=6)
        ax_s.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax_s.set_ylabel("Slope / decade")
        ax_s.set_title(f"{b} — slope")
        ax_s.grid(True, alpha=0.3)

        am = sub.dropna(subset=["argmax_tokens"])
        ax_a.plot(am["period_decimal"], am["argmax_tokens"],
                  "o-", color=color, linewidth=1.8, markersize=6,
                  label="argmax tokens")
        mam = sub.dropna(subset=["max_tokens"])
        ax_a.scatter(mam["period_decimal"], mam["max_tokens"],
                     marker="*", color=color, s=130, edgecolor="white",
                     linewidth=0.5, zorder=4, label="max tokens (any model)")
        ax_a.set_yscale("log")
        ax_a.set_ylabel("Tokens")
        ax_a.set_title(f"{b} — tokens")
        ax_a.grid(True, which="both", alpha=0.3)
        ax_a.legend(fontsize=7, loc="lower right", framealpha=0.9)

    for ax in axes[-1]:
        ax.set_xticks(semester_ticks)
        ax.set_xticklabels(semester_labels, rotation=30, ha="right")
        ax.set_xlabel("Semester")

    fig.suptitle("Per-benchmark linear-fit trends over time", fontsize=12)

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "semester_fit_trends.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")

    out_csv = OUT_DIR / "semester_fit_trends.csv"
    df[["benchmark", "period", "slope_per_decade", "intercept",
        "argmax_tokens", "argmax_score", "argmax_model", "max_tokens",
        "delta_slope", "delta_intercept", "argmax_token_ratio",
        "r2", "n_fit_points"]].to_csv(out_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
