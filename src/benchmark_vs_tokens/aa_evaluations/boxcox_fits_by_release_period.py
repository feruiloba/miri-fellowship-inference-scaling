"""
Per-release-period Box-Cox curves of inference-token scaling.

Groups unique models into PERIOD_MONTHS-wide release-date buckets,
applies a monotone-upper-envelope filter (keep points within 1σ of
the running max as tokens increase), then fits Box-Cox per bucket
with a multi-start grid to avoid local minima.

Output:
  output/benchmark_vs_tokens/aa_evaluations/boxcox_fits_by_release_period.{png,csv}
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

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

PARAM_NAMES = ["m", "c", "h", "C"]
PERIOD_MONTHS = 9  # width of each release-date bucket, in months
C_ANCHOR = 0.5     # stage-1 c value used to pin down m

# Multi-start grid: each (c0, h0_offset) combo seeds one curve_fit call.
# We keep the result with the lowest SSE across all starts.
_C0_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
_H0_OFFSETS = (0.25, 1.0, 3.0)  # h0 = x.min() − offset


def model_boxcox(x, m, c, h, C):
    base = 1.0 + x - h
    base = np.where(base > 1e-12, base, 1e-12)
    return m * (np.power(base, c) - 1.0) / c + C


def r_squared(y, y_hat):
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _fit_stage1_fixed_c(x, y) -> float | None:
    """Fit Box-Cox with c held at C_ANCHOR; return best-SSE m."""
    def _model(xv, m, h, C):
        return model_boxcox(xv, m, C_ANCHOR, h, C)

    m0 = float((y.max() - y.min()) if y.max() > y.min() else 1.0)
    C0 = float(max(y.min(), 0.0))
    lower = [0.0,    x.min() - 5.0, 0.0]
    upper = [np.inf, x.min() - 1e-6, 1.0]

    best_m: float | None = None
    best_sse = np.inf
    for h_off in _H0_OFFSETS:
        p0 = [m0, float(x.min()) - h_off, C0]
        try:
            popt, _ = curve_fit(_model, x, y, p0=p0,
                                bounds=(lower, upper), maxfev=50000)
        except Exception:  # noqa: BLE001
            continue
        m_fit, h_fit, C_fit = popt
        sse = float(np.sum((y - model_boxcox(x, m_fit, C_ANCHOR, h_fit, C_fit)) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_m = float(m_fit)
    return best_m


def _fit_stage2_fixed_m(x, y, m_fixed: float):
    """Fit Box-Cox with m held at m_fixed; sweep c0 and h0 for multi-start."""
    def _model(xv, c, h, C):
        return model_boxcox(xv, m_fixed, c, h, C)

    C0 = float(max(y.min(), 0.0))
    lower = [1e-3, x.min() - 5.0, 0.0]
    upper = [1.0,  x.min() - 1e-6, 1.0]

    best: dict | None = None
    last_err: str | None = None
    for c0 in _C0_GRID:
        for h_off in _H0_OFFSETS:
            p0 = [c0, float(x.min()) - h_off, C0]
            try:
                popt, _ = curve_fit(_model, x, y, p0=p0,
                                    bounds=(lower, upper), maxfev=50000)
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                continue
            full = np.array([m_fixed, popt[0], popt[1], popt[2]])
            sse = float(np.sum((y - model_boxcox(x, *full)) ** 2))
            if best is None or sse < best["sse"]:
                best = {"params": full, "sse": sse}
    if best is None:
        return None, last_err
    return best, None


def fit_bucket(x: np.ndarray, y: np.ndarray):
    """Two-stage Box-Cox fit:
      stage 1 — fix c=C_ANCHOR (0.5), fit (m, h, C); record m_anchor.
      stage 2 — fix m=m_anchor, fit (c, h, C) with multi-start."""
    if len(x) < 4 or x.max() == x.min():
        return None
    m_anchor = _fit_stage1_fixed_c(x, y)
    if m_anchor is None:
        return {"params": np.full(4, np.nan), "r2": np.nan,
                "success": False, "msg": "stage 1 (fixed-c) failed"}
    best, err = _fit_stage2_fixed_m(x, y, m_anchor)
    if best is None:
        return {"params": np.full(4, np.nan), "r2": np.nan,
                "success": False, "msg": err or "stage 2 (fixed-m) failed"}
    return {"params": best["params"],
            "r2": r_squared(y, model_boxcox(x, *best["params"])),
            "success": True}


def _load(benchmark: str) -> pd.DataFrame:
    ev = pd.read_csv(EVAL_CSV)
    ev = ev[ev["benchmark"] == benchmark].copy()
    ev = ev.dropna(subset=["total_output_tokens", "score_raw", "model_slug"])
    ev = ev[ev["total_output_tokens"] > 0]

    stats = pd.read_csv(STATS_CSV, usecols=["slug", "release_date"])
    stats = stats.dropna(subset=["release_date"]).rename(columns={"slug": "model_slug"})
    ev = ev.merge(stats, on="model_slug", how="left")
    ev["release_date"] = pd.to_datetime(ev["release_date"], errors="coerce")
    ev = ev.dropna(subset=["release_date"]).copy()
    return ev


def _period_key(dt: pd.Timestamp) -> int:
    """Map a date to a PERIOD_MONTHS-wide bucket index (month-aligned)."""
    return (int(dt.year) * 12 + int(dt.month) - 1) // PERIOD_MONTHS


def _period_span(period_idx: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ((lo_year, lo_month), (hi_year, hi_month)) inclusive."""
    start_m = period_idx * PERIOD_MONTHS
    end_m = start_m + PERIOD_MONTHS - 1
    return (start_m // 12, start_m % 12 + 1), (end_m // 12, end_m % 12 + 1)


def _bucket_unique_models(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each model_slug to a PERIOD_MONTHS-wide release-date bucket
    by its first release date, then propagate the bucket id to every run
    of that slug. Bucket ids are contiguous integers ordered chronologically."""
    per_model = df.groupby("model_slug")["release_date"].min()
    period_per_model = per_model.map(_period_key)
    ordered_periods = sorted(set(period_per_model))
    period_to_id = {p: i for i, p in enumerate(ordered_periods)}
    bucket_map = period_per_model.map(period_to_id)
    df = df.copy()
    df["bucket"] = df["model_slug"].map(bucket_map).astype("Int64")
    df.attrs["period_to_id"] = period_to_id
    return df


def _bucket_label(sub: pd.DataFrame) -> str:
    """e.g. '2024-01 → 2024-09  (12 models)' for a 9-month period."""
    per_slug = sub.groupby("model_slug")["release_date"].min()
    period_idx = _period_key(per_slug.iloc[0])
    (lo_y, lo_m), (hi_y, hi_m) = _period_span(period_idx)
    return (
        f"{lo_y:04d}-{lo_m:02d} → {hi_y:04d}-{hi_m:02d}  "
        f"({len(per_slug)} models)"
    )


ENVELOPE_SIGMA_MULT = 1.0  # keep points within this many σ of the running max


def _monotone_upper_envelope(kept: pd.DataFrame) -> pd.DataFrame:
    """Sort by tokens ascending; keep a point iff its score is at least
    (running max so far − ENVELOPE_SIGMA_MULT · σ of the bucket's scores).
    New running-max points always pass; nearby points within that band
    are also kept; points further below the running max are dropped."""
    sorted_kept = kept.sort_values("total_output_tokens", kind="stable")
    scores = sorted_kept["score_raw"].to_numpy(dtype=float)
    running_max = np.maximum.accumulate(scores)
    sigma = float(np.std(scores)) if len(scores) > 1 else 0.0
    keep_mask = scores >= (running_max - ENVELOPE_SIGMA_MULT * sigma)
    return sorted_kept.loc[keep_mask]


def _fit_per_bucket(df: pd.DataFrame) -> list[dict]:
    """Per-bucket two-stage fit: each bucket gets its own m_anchor from
    stage 1 (c=C_ANCHOR), then stage 2 refits (c, h, C) with that m."""
    df = df.dropna(subset=["bucket"]).copy()

    out = []
    for bucket, sub in df.groupby("bucket", observed=True):
        kept = _monotone_upper_envelope(sub)
        x = np.log10(kept["total_output_tokens"].to_numpy(dtype=float))
        y = kept["score_raw"].to_numpy(dtype=float)
        info = fit_bucket(x, y)
        if info is None or not info["success"]:
            continue
        out.append({
            "bucket": int(bucket),
            "label": _bucket_label(kept),
            "n_runs": int(len(kept)),
            "n_models": int(kept["model_slug"].nunique()),
            "x": x, "y": y,
            "params": info["params"], "r2": info["r2"],
        })
    out.sort(key=lambda r: r["bucket"])
    return out


def _draw_panel(ax, benchmark: str, df: pd.DataFrame, fits: list[dict]) -> None:
    # Background: all runs faded grey
    ax.scatter(
        df["total_output_tokens"], df["score_raw"],
        s=8, alpha=0.10, color="#888888", edgecolor="none",
    )

    cmap = plt.get_cmap("viridis")
    n_buckets = max(len(fits), 1)
    for i, info in enumerate(fits):
        color = cmap(i / max(n_buckets - 1, 1))
        ax.scatter(
            10 ** info["x"], info["y"],
            color=color, s=18, alpha=0.55, edgecolor="white", linewidth=0.3,
            zorder=3,
        )
        x_min, x_max = info["x"].min(), info["x"].max()
        x_dense = np.linspace(x_min, x_max, 200)
        y_dense = model_boxcox(x_dense, *info["params"])
        ax.plot(
            10 ** x_dense, y_dense,
            color=color, linewidth=2.0, alpha=0.95, zorder=4,
            label=f"{info['label']}  R²={info['r2']:.2f}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total inference tokens")
    ax.set_ylabel("Score (raw)")
    ax.set_title(benchmark, fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    if fits:
        ax.legend(title=f"{PERIOD_MONTHS}-month release period (Box-Cox fit)",
                  loc="lower right", fontsize=7, framealpha=0.9)


def main(benchmarks: list[str] = BENCHMARKS) -> None:
    panels: list[tuple[str, pd.DataFrame, list[dict]]] = []
    for b in benchmarks:
        df = _load(b)
        if df.empty:
            print(f"skipping {b!r} (no rows)")
            continue
        df = _bucket_unique_models(df)
        fits = _fit_per_bucket(df)
        mean_r2 = float(np.mean([f["r2"] for f in fits])) if fits else float("nan")
        print(f"  {b}: mean R²={mean_r2:.3f}")
        panels.append((b, df, fits))

    if not panels:
        print("Nothing to plot.")
        return

    cols = 2
    rows_n = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 5 * rows_n), squeeze=False)
    for idx, (b, df, fits) in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        _draw_panel(ax, b, df, fits)
    for j in range(len(panels), rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        f"Box-Cox fits per {PERIOD_MONTHS}-month release period  "
        f"(keep points within {ENVELOPE_SIGMA_MULT}σ of the running max as tokens increase)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "boxcox_fits_by_release_period.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_png}")

    csv_rows = []
    for b, _, fits in panels:
        for info in fits:
            csv_rows.append({
                "benchmark": b,
                "bucket": info["bucket"],
                "label": info["label"],
                "n_runs": info["n_runs"],
                "n_models": info["n_models"],
                "r2": info["r2"],
                **dict(zip(PARAM_NAMES, info["params"])),
            })
    out_csv = OUT_DIR / "boxcox_fits_by_release_period.csv"
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    for b, df, fits in panels:
        n_models_total = df["model_slug"].nunique()
        print(f"  {b}: {len(df)} rows, {n_models_total} models → "
              f"{len(fits)} buckets fit")


if __name__ == "__main__":
    main()
