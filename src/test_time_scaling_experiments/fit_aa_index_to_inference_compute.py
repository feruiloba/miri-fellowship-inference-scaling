"""
Fit a functional form to each model-family scaling curve from 13a.

Currently fitting:  f(x) = A * tanh(k * (x - h)) + C
where x = log10(total_output_tokens) and f(x) = AA Index.

To try a different functional form, change MODEL_FUNC, MODEL_NAME,
INITIAL_GUESS, and PARAM_NAMES below.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark_vs_tokens", "aa_evaluations"))
from importlib import import_module
mod = import_module("aa_index_per_family_effort_lines")
load_effort_models = mod.load_effort_models
LEVEL_ORDER = mod.LEVEL_ORDER
LEVEL_MARKERS = mod.LEVEL_MARKERS
FAMILY_COLORS = mod.FAMILY_COLORS


# =============================================================================
# FUNCTIONAL FORM — swap this block to try a different fit
# =============================================================================

def model_tanh(x, A, k, h, C):
    """f(x) = A * tanh(k * (x - h)) + C"""
    return A * np.tanh(k * (x - h)) + C

def model_sigmoid(x, L, k, h, C):
    """f(x) = L / (1 + exp(-k*(x-h))) + C
    L = full span (top - bottom), C = lower asymptote, h = midpoint x, k = steepness."""
    return L / (1.0 + np.exp(-k * (x - h))) + C

def model_boxcox(x, m, c, h, C):
    """f(x) = m * ((1 + x - h)^c - 1) / c + C
    Box-Cox style power transform. Requires 1 + x - h > 0."""
    base = 1.0 + x - h
    base = np.where(base > 1e-12, base, 1e-12)
    return m * (np.power(base, c) - 1.0) / c + C

# Per-form metadata + bounds. Pick which one is active via CLI arg
# (defaults to "boxcox").
FORM_CONFIGS = {
    "boxcox": {
        "func": model_boxcox,
        "name": "m·((1+x−h)^c − 1)/c + C",
        "param_names": ["m", "c", "h", "C"],
        # bounds order: [m, c, h, C]; h ≤ x.min() keeps it on the
        # diminishing-returns branch.
        "bounds": lambda x, y: (
            [0.0,    1e-3, x.min() - 5.0, 0.0],
            [np.inf, 1.0,  x.min() - 1e-6, np.inf],
        ),
        "initial_guess": lambda x, y: [
            (y.max() - y.min()) if y.max() > y.min() else 1.0,
            0.5,
            float(x.min()) - 0.5,
            float(y.min()),
        ],
    },
    "sigmoid": {
        "func": model_sigmoid,
        "name": "L / (1 + exp(−k·(x−h))) + C",
        "param_names": ["L", "k", "h", "C"],
        # AA index is roughly 0–60; cap L generously. Forcing h ≤ x.min()
        # keeps the visible curve on the concave-down (saturating) branch.
        "bounds": lambda x, y: (
            [0.0,   1e-2, x.min() - 5.0, 0.0],
            [100.0, 10.0, x.min() - 1e-6, np.inf],
        ),
        "initial_guess": lambda x, y: [
            max(y.max() - y.min(), 1e-3),
            1.0,
            float(x.min()) - 0.5,
            float(max(y.min(), 0.0)),
        ],
    },
}

# Active config — overridden in main() from CLI arg.
FORM = "boxcox"
MODEL_FUNC = FORM_CONFIGS[FORM]["func"]
MODEL_NAME = FORM_CONFIGS[FORM]["name"]
PARAM_NAMES = FORM_CONFIGS[FORM]["param_names"]


def _select_form(name: str) -> None:
    """Switch MODEL_FUNC/MODEL_NAME/PARAM_NAMES to the named form."""
    global FORM, MODEL_FUNC, MODEL_NAME, PARAM_NAMES
    if name not in FORM_CONFIGS:
        raise ValueError(f"Unknown form {name!r}; choose from {list(FORM_CONFIGS)}.")
    FORM = name
    MODEL_FUNC = FORM_CONFIGS[FORM]["func"]
    MODEL_NAME = FORM_CONFIGS[FORM]["name"]
    PARAM_NAMES = FORM_CONFIGS[FORM]["param_names"]


# =============================================================================
# FITTING — joint fit with shared k across families
# =============================================================================
# Per-family params: A_i, h_i, C_i.   Globally shared: k.
# Total params = 1 + 3 * n_families.   Total data points = sum(n_i).
# Need sum(n_i) >= 1 + 3 * n_families  (each family contributes; k is pooled).
# Families with n_i == 1 are dropped (no info on shape).

JOINT_PER_FAMILY = ["A", "h", "C"]


def _unpack(theta, n_fam):
    k = theta[0]
    rest = theta[1:].reshape(n_fam, 3)  # columns: A, h, C
    return k, rest


def joint_residuals(theta, families):
    """families: list of (x_array, y_array). theta packs [k, P1_a,h1,C1, P2_a,h2,C2, ...].
    P_a is the per-family amplitude param (A for tanh, L for sigmoid)."""
    k, per = _unpack(theta, len(families))
    res = []
    for (x, y), (P_a, h, C) in zip(families, per):
        res.append(MODEL_FUNC(x, P_a, k, h, C) - y)
    return np.concatenate(res)


def fit_joint(family_data):
    """
    Per-family independent fit (k is free per family).
    Same bounds as before: h <= x.min() (diminishing returns only),
    C >= 0, k > 0.

    family_data: dict base -> (x, y)
    Returns: dict base -> {"params": (amp, k, h, C), "r2": ..., "success": True}
    """
    cfg = FORM_CONFIGS[FORM]
    fits = {}
    for base, (x, y) in family_data.items():
        p0 = cfg["initial_guess"](x, y)
        lower, upper = cfg["bounds"](x, y)

        try:
            popt, _ = curve_fit(MODEL_FUNC, x, y, p0=p0,
                                bounds=(lower, upper), maxfev=50000)
            y_hat = MODEL_FUNC(x, *popt)
            r2 = r_squared(y, y_hat)
            fits[base] = {"params": popt, "r2": r2,
                          "success": True, "msg": "per-family"}
        except Exception as e:
            fits[base] = {"success": False, "msg": str(e),
                          "params": np.full(4, np.nan), "r2": np.nan}

    fits["_shared_k"] = None  # no longer shared
    return fits


def r_squared(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# =============================================================================
# PLOTTING
# =============================================================================

def plot_fits(df, fits, save_prefix):
    fig, ax = plt.subplots(figsize=(13, 9))

    bases = sorted(df["base"].unique())
    color_map = {b: FAMILY_COLORS[i % len(FAMILY_COLORS)] for i, b in enumerate(bases)}

    for base in bases:
        family = df[df["base"] == base].sort_values(
            "level", key=lambda s: s.map(LEVEL_ORDER)
        )
        color = color_map[base]
        x = np.log10(family["total_output_tokens"].values.astype(float))
        y = family["aa_index"].values.astype(float)

        # Data points
        for _, row in family.iterrows():
            marker = LEVEL_MARKERS.get(row["level"], "o")
            ax.scatter(
                10 ** np.log10(row["total_output_tokens"]),
                row["aa_index"],
                color=color, marker=marker, s=80, zorder=3,
                edgecolors="white", linewidths=0.5,
            )

        # Connect raw data
        ax.plot(family["total_output_tokens"], y,
                color=color, linewidth=1, alpha=0.35, zorder=2, linestyle=":")

        # Fitted curve
        info = fits.get(base)
        if info and info["success"]:
            x_dense = np.linspace(x.min(), x.max(), 200)
            y_dense = MODEL_FUNC(x_dense, *info["params"])
            ax.plot(10 ** x_dense, y_dense,
                    color=color, linewidth=2.2, alpha=0.9, zorder=4)
            r2 = info["r2"]
            label = f"{base}  (R²={r2:.3f})"
        else:
            label = f"{base}  (no fit)"

        # Annotate at high end
        last = family.iloc[-1]
        ax.annotate(
            label,
            (last["total_output_tokens"], last["aa_index"]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold", va="center",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total output tokens", fontsize=11)
    ax.set_ylabel("AA Index", fontsize=11)
    ax.set_title(f"Per-family fit: {MODEL_NAME}   (x = log₁₀ tokens)",
                 fontsize=12)
    ax.grid(True, alpha=0.2, which="both")
    ax.set_axisbelow(True)

    plt.savefig(f"{save_prefix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_prefix}.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    OUT_DIR = "output/test_time_scaling_experiments"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Optional CLI arg: "boxcox" (default) or "sigmoid".
    if len(sys.argv) > 1:
        _select_form(sys.argv[1])

    df = load_effort_models()

    print("=" * 70)
    print(f"FITTING {MODEL_NAME} TO EACH MODEL FAMILY")
    print("=" * 70)
    print(f"x = log10(total_output_tokens),  y = AA Index")
    print()

    # Build per-family (x, y) arrays. Drop n=1 families (no shape info).
    family_data = {}
    skipped = []
    for base in sorted(df["base"].unique()):
        family = df[df["base"] == base].sort_values(
            "level", key=lambda s: s.map(LEVEL_ORDER)
        )
        x = np.log10(family["total_output_tokens"].values.astype(float))
        y = family["aa_index"].values.astype(float)
        if len(x) < 2:
            skipped.append((base, len(x)))
            continue
        family_data[base] = (x, y)

    print(f"Fitting {len(family_data)} families independently "
          f"(per-family params: {', '.join(PARAM_NAMES)}). Skipped n<2: "
          f"{[b for b, _ in skipped]}")
    print()

    fits = fit_joint(family_data)

    # Earliest release date per base (variants share the AA model's release_date)
    base_release = (
        df.dropna(subset=["release_date"])
          .groupby("base")["release_date"]
          .min()
          .to_dict()
    )

    p1, p2, p3, p4 = PARAM_NAMES
    rows = []
    for base, (x, y) in family_data.items():
        info = fits[base]
        v1, v2, v3, v4 = info["params"]
        param_str = f"{p1}={v1:+.3f}, {p2}={v2:+.3f}, {p3}={v3:+.3f}, {p4}={v4:+.3f}"
        print(f"{base}: n={len(x)}  R²={info['r2']:.3f}  {param_str}")
        rows.append({"base": base, "release_date": base_release.get(base),
                     "n": len(x), "r2": info["r2"],
                     p1: v1, p2: v2, p3: v3, p4: v4})

    for base, n in skipped:
        rows.append({"base": base, "release_date": base_release.get(base),
                     "n": n, "r2": np.nan,
                     p1: np.nan, p2: np.nan, p3: np.nan, p4: np.nan})

    save_prefix = f"{OUT_DIR}/fit_aa_index_{MODEL_FUNC.__name__.removeprefix('model_')}"
    plot_fits(df, fits, save_prefix)

    table = pd.DataFrame(rows)
    table.to_csv(f"{save_prefix}_params.csv", index=False)
    print(f"Saved: {save_prefix}_params.csv")
    print("\nDone.")
