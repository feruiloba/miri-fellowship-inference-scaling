"""
Fit a Box-Cox-style scaling curve to each AA evaluation family for one benchmark.

Same functional form as fit_aa_index_to_inference_compute.py, but the y-axis
is the per-benchmark score (`score_raw`, 0-1) instead of AA Index, and one
fit is produced per `family_id` in
data/artificial_analysis/aa_evaluation_families_combined.csv. Families are
sets of effort variants for the same base model (e.g. the non-reasoning and
reasoning rows for Claude 3.7 Sonnet).

Usage:
    python src/test_time_scaling_experiments/fit_aa_evaluation_to_inference_compute.py <benchmark>
e.g.
    python src/test_time_scaling_experiments/fit_aa_evaluation_to_inference_compute.py gpqa-diamond

Output (slug derived from the benchmark, "-" → "_"):
    output/test_time_scaling_experiments/fit_<benchmark>_boxcox.png
    output/test_time_scaling_experiments/fit_<benchmark>_boxcox_params.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
FAMILY_CSV = ROOT / "data" / "artificial_analysis" / "aa_evaluation_families_combined.csv"
STATS_CSV = ROOT / "data" / "artificial_analysis" / "artificial_analysis_llm_stats.csv"
OUT_DIR = ROOT / "output" / "test_time_scaling_experiments"


# Same Box-Cox form as fit_aa_index_to_inference_compute.py: kept inline so
# this script is self-contained rather than coupled to the AA-Index variant.
def model_boxcox(x, m, c, h, C):
    """f(x) = m·((1 + x − h)^c − 1)/c + C   (requires 1 + x − h > 0)."""
    base = 1.0 + x - h
    base = np.where(base > 1e-12, base, 1e-12)
    return m * (np.power(base, c) - 1.0) / c + C


PARAM_NAMES = ["m", "c", "h", "C"]


def r_squared(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_family(x: np.ndarray, y: np.ndarray):
    """Fit Box-Cox to one family's (x, y); same bounds as the AA-Index variant."""
    m0 = (y.max() - y.min()) if y.max() > y.min() else 1.0
    c0 = 0.5
    h0 = float(x.min()) - 0.5
    C0 = float(y.min())
    p0 = [m0, c0, h0, C0]
    lower = [0.0, 1e-3, x.min() - 5.0, 0.0]
    upper = [np.inf, 1.0, x.min() - 1e-6, np.inf]
    try:
        popt, _ = curve_fit(
            model_boxcox, x, y, p0=p0, bounds=(lower, upper), maxfev=50000,
        )
        return {"params": popt, "r2": r_squared(y, model_boxcox(x, *popt)), "success": True}
    except Exception as e:  # noqa: BLE001
        return {"params": np.full(4, np.nan), "r2": np.nan, "success": False, "msg": str(e)}


_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def plot_fits(df: pd.DataFrame, fits: dict, benchmark: str, save_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 9))

    families = sorted(df["family_id"].unique())
    color_map = {f: _PALETTE[i % len(_PALETTE)] for i, f in enumerate(families)}

    for fam in families:
        sub = df[df["family_id"] == fam].sort_values("total_output_tokens")
        color = color_map[fam]
        x = np.log10(sub["total_output_tokens"].to_numpy(dtype=float))
        y = sub["score_raw"].to_numpy(dtype=float)

        ax.scatter(
            sub["total_output_tokens"], y,
            color=color, s=55, zorder=3,
            edgecolors="white", linewidths=0.5,
        )
        ax.plot(
            sub["total_output_tokens"], y,
            color=color, linewidth=1, alpha=0.35, zorder=2, linestyle=":",
        )

        info = fits.get(fam)
        if info and info["success"] and len(x) >= 2 and x.max() > x.min():
            x_dense = np.linspace(x.min(), x.max(), 200)
            ax.plot(
                10 ** x_dense, model_boxcox(x_dense, *info["params"]),
                color=color, linewidth=2.0, alpha=0.9, zorder=4,
            )
            label = f"{fam}  (R²={info['r2']:.3f})"
        else:
            label = fam
        last = sub.iloc[-1]
        ax.annotate(
            label, (last["total_output_tokens"], last["score_raw"]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold", va="center",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total inference tokens", fontsize=11)
    ax.set_ylabel(f"Score (fraction correct) — {benchmark}", fontsize=11)
    ax.set_title(
        f"Per-family Box-Cox fit on {benchmark}   "
        f"(x = log₁₀ tokens, y = score)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.2, which="both")
    ax.set_axisbelow(True)

    out = save_prefix.with_suffix(".png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def _load_slug_release_dates() -> dict[str, str]:
    stats = pd.read_csv(STATS_CSV, usecols=["slug", "release_date"])
    stats = stats.dropna(subset=["slug", "release_date"])
    return dict(zip(stats["slug"], stats["release_date"]))


def _family_release_date(
    sub: pd.DataFrame, family_id: str, slug_to_date: dict[str, str]
) -> str | None:
    """Earliest known release_date across the family's variants.

    Tries each variant's model_slug, then the family_id itself as a fallback.
    """
    dates: list[str] = []
    for slug in sub["model_slug"].dropna().unique():
        d = slug_to_date.get(slug)
        if d:
            dates.append(d)
    if not dates:
        d = slug_to_date.get(family_id)
        if d:
            dates.append(d)
    return min(dates) if dates else None


def main(benchmark: str) -> None:
    df = pd.read_csv(FAMILY_CSV)
    df = df[df["benchmark"] == benchmark].copy()
    df = df.dropna(subset=["total_output_tokens", "score_raw", "family_id"])
    df = df[df["total_output_tokens"] > 0]

    if df.empty:
        print(f"No usable rows for benchmark={benchmark!r}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = benchmark.replace("-", "_")
    save_prefix = OUT_DIR / f"fit_{slug}_boxcox"

    slug_to_date = _load_slug_release_dates()

    fits: dict = {}
    rows = []
    skipped: list[tuple[str, int]] = []
    for fam, sub in df.groupby("family_id"):
        release = _family_release_date(sub, fam, slug_to_date)
        x = np.log10(sub["total_output_tokens"].to_numpy(dtype=float))
        y = sub["score_raw"].to_numpy(dtype=float)
        if len(x) < 2:
            skipped.append((fam, len(x)))
            rows.append({"family_id": fam, "release_date": release,
                         "n": len(x), "r2": np.nan,
                         **{p: np.nan for p in PARAM_NAMES}})
            continue
        info = fit_family(x, y)
        fits[fam] = info
        v = info["params"]
        rows.append({"family_id": fam, "release_date": release,
                     "n": len(x), "r2": info["r2"],
                     **dict(zip(PARAM_NAMES, v))})

    table = pd.DataFrame(rows).sort_values(["n", "r2"], ascending=[False, False])
    csv_path = save_prefix.with_name(save_prefix.name + "_params.csv")
    table.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    fitted = sum(1 for r in rows if not np.isnan(r["r2"]))
    print(
        f"benchmark={benchmark}  families={len(rows)}  "
        f"fitted={fitted}  skipped(n<2)={len(skipped)}  rows={len(df)}"
    )

    plot_fits(df, fits, benchmark, save_prefix)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: fit_aa_evaluation_to_inference_compute.py <benchmark-slug>"
        )
    main(sys.argv[1])
