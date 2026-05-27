"""
MoE-aware variant of chinchilla_param_token_vs_optimal: dense rows are
plotted as-is, and MoE rows use active parameters (N_active = C / (6*D),
the identity Epoch's "Training compute" already assumes for MoE).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _chinchilla import load_models_with_chinchilla, filter_for_plotting

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "chinchilla_analysis"
OUT.mkdir(parents=True, exist_ok=True)

C_DENSE_N = "#4C72B0"
C_DENSE_D = "#DD8452"


def _fit_log_y_vs_date(dates, y):
    """Fit log10(y) = m * years + b, where years is float-year (e.g. 2024.5).
    Returns (slope_per_year, intercept_at_year_0, r2, n)."""
    if hasattr(dates, "to_numpy"):
        dates = dates.to_numpy()
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    dt = pd.to_datetime(pd.Series(dates))
    yrs = (dt.dt.year + (dt.dt.dayofyear - 1) / 365.25).to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(yrs) & np.isfinite(y) & (y > 0)
    if mask.sum() < 3:
        return None
    ly = np.log10(y[mask])
    res = sp_stats.linregress(yrs[mask], ly)
    return res.slope, res.intercept, res.rvalue ** 2, int(mask.sum())


def _draw_fit_vs_date(ax, dates, y, color):
    fit = _fit_log_y_vs_date(dates, y)
    if fit is None:
        return
    slope, intercept, r2, n = fit
    dt = pd.to_datetime(pd.Series(dates))
    yrs = (dt.dt.year + (dt.dt.dayofyear - 1) / 365.25).to_numpy(dtype=float)
    xs = np.linspace(yrs.min(), yrs.max(), 50)
    ys = 10 ** (slope * xs + intercept)
    xs_dt = pd.to_datetime(
        [pd.Timestamp(year=int(yr), month=1, day=1)
         + pd.Timedelta(days=(yr - int(yr)) * 365.25) for yr in xs]
    )
    factor = 10 ** slope
    ax.plot(xs_dt, ys, color=color, linewidth=2, linestyle="--", alpha=0.9,
            label=(f"fit: ×{factor:.2f}/yr  ({slope:+.2f} OOM/yr),  "
                   f"R²={r2:.2f},  n={n}"))


def _plot(dense, moe, title_suffix, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    (ax1, ax2), (ax3, ax4) = axes

    all_pts = pd.concat([dense, moe], ignore_index=True)
    ax1.scatter(all_pts["Training compute (FLOP)"], all_pts["N_ratio"],
                alpha=0.7, color=C_DENSE_N)
    ax2.scatter(all_pts["Training compute (FLOP)"], all_pts["D_ratio"],
                alpha=0.7, color=C_DENSE_D)
    ax3.scatter(all_pts["Publication date"], all_pts["N_ratio"],
                alpha=0.7, color=C_DENSE_N)
    ax4.scatter(all_pts["Publication date"], all_pts["D_ratio"],
                alpha=0.7, color=C_DENSE_D)
    _draw_fit_vs_date(ax3, all_pts["Publication date"],
                      all_pts["N_ratio"], "#222222")
    _draw_fit_vs_date(ax4, all_pts["Publication date"],
                      all_pts["D_ratio"], "#222222")

    for ax in (ax1, ax2, ax3, ax4):
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1,
                   label="Chinchilla-optimal")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax1.set_xscale("log")
    ax1.set_xlabel("Training compute C (FLOP)")
    ax1.set_ylabel("N_actual / N_opt")
    ax1.set_title(f"Parameters vs. Chinchilla-optimal{title_suffix}")

    ax2.set_xscale("log")
    ax2.set_xlabel("Training compute C (FLOP)")
    ax2.set_ylabel("D_actual / D_opt")
    ax2.set_title(f"Training tokens vs. Chinchilla-optimal{title_suffix}")

    ax3.set_xlabel("Publication date")
    ax3.set_ylabel("N_actual / N_opt")
    ax3.set_title(f"Parameters vs. Chinchilla-optimal (by date){title_suffix}")
    ax3.tick_params(axis="x", rotation=30)

    ax4.set_xlabel("Publication date")
    ax4.set_ylabel("D_actual / D_opt")
    ax4.set_title(f"Training tokens vs. Chinchilla-optimal (by date){title_suffix}")
    ax4.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"saved {out_png}")


def main():
    d = load_models_with_chinchilla()
    plot_df = filter_for_plotting(d)
    dense = plot_df[~plot_df["is_moe"]].copy()
    moe = plot_df[plot_df["is_moe"]].copy()

    # MoE rows use N_active = C / (6*D)
    moe["N_active"] = moe["Training compute (FLOP)"] / (6.0 * moe["D_actual"])
    moe["N_actual"] = moe["N_active"]
    moe["N_ratio"] = moe["N_actual"] / moe["N_opt"]
    _plot(
        dense=dense,
        moe=moe,
        title_suffix=" (MoE: active params)",
        out_png=OUT / "chinchilla_param_token_vs_optimal_moe_active.png",
    )
    cols = ["Model", "Training compute (FLOP)", "N_actual", "N_opt",
            "N_ratio", "D_actual", "D_opt", "D_ratio"]
    combined = pd.concat([dense.assign(arch="dense"),
                          moe.assign(arch="moe_active")], ignore_index=True)
    combined.sort_values("Training compute (FLOP)")[
        cols + ["arch"]
    ].to_csv(
        OUT / "chinchilla_param_token_vs_optimal_moe_active.csv", index=False
    )


if __name__ == "__main__":
    main()
