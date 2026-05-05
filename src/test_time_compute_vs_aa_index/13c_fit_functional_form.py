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

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
mod = import_module("13a_test_time_compute_vs_aa_index")
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

MODEL_FUNC = model_sigmoid
MODEL_NAME = "L/(1+exp(−k·(x−h))) + C"
PARAM_NAMES = ["L", "k", "h", "C"]
# Initial guess given (x, y) arrays (x = log10 tokens, y = AA index)
def initial_guess(x, y):
    A0 = (y.max() - y.min()) / 2 if y.max() > y.min() else 1.0
    C0 = (y.max() + y.min()) / 2
    h0 = float(np.median(x))
    k0 = 1.0
    return [A0, k0, h0, C0]


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
    fits = {}
    for base, (x, y) in family_data.items():
        # Per-family bounds, order: [amp, k, h, C]
        amp0 = (y.max() - y.min()) if y.max() > y.min() else 1.0
        k0   = 3.0
        h0   = float(x.min()) - 0.5
        C0   = float(y.min())
        p0    = [amp0, k0, h0, C0]
        lower = [-np.inf, 1e-3, x.min() - 2.0, 0.0]
        upper = [ np.inf, 50.0, x.min(),       np.inf]

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
    plt.savefig(f"{save_prefix}.svg", format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_prefix}.png / .svg")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    OUT_DIR = "output/13_inference_scaling"
    os.makedirs(OUT_DIR, exist_ok=True)

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

    amp_name = PARAM_NAMES[0]  # "A" for tanh, "L" for sigmoid
    rows = []
    for base, (x, y) in family_data.items():
        info = fits[base]
        P_a, k, h, C = info["params"]
        param_str = f"{amp_name}={P_a:+.3f}, k={k:+.3f}, h={h:+.3f}, C={C:+.3f}"
        print(f"{base}: n={len(x)}  R²={info['r2']:.3f}  {param_str}")
        rows.append({"base": base, "n": len(x), "r2": info["r2"],
                     amp_name: P_a, "k": k, "h": h, "C": C})

    for base, n in skipped:
        rows.append({"base": base, "n": n, "r2": np.nan,
                     amp_name: np.nan, "k": np.nan, "h": np.nan, "C": np.nan})

    save_prefix = f"{OUT_DIR}/fit_{MODEL_FUNC.__name__}"
    plot_fits(df, fits, save_prefix)

    table = pd.DataFrame(rows)
    table.to_csv(f"{save_prefix}_params.csv", index=False)
    print(f"Saved: {save_prefix}_params.csv")
    print("\nDone.")
