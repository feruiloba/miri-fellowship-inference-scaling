"""
AA Index vs. ECI
================

Scatter of Artificial Analysis Index vs. Epoch Capability Index (ECI) for
models that have both, with a linear fit.

Data source: data/merged_datasets.csv
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MERGED_FILE = "data/merged_datasets.csv"
TOKENS_FILE = "data/artificial_analysis/aa_output_tokens.csv"
OUT_DIR = "output/capability_indices"


def load_points():
    df = pd.read_csv(MERGED_FILE)
    df["aa_index"] = pd.to_numeric(df["AA_artificial_analysis_index"], errors="coerce")
    df["eci"] = pd.to_numeric(df["eci"], errors="coerce")
    df = df[df["aa_index"].notna() & df["eci"].notna()].copy()

    tokens = pd.read_csv(TOKENS_FILE)
    tokens["aa_output_tokens"] = (
        tokens["reasoning_tokens"].fillna(0) + tokens["answer_tokens"].fillna(0)
    )
    df = df.merge(
        tokens[["model", "reasoning_tokens", "answer_tokens", "aa_output_tokens"]],
        left_on="AA_name", right_on="model", how="left",
    ).drop(columns=["model"])

    return df[["Model", "AA_name", "aa_index", "eci", "eci_ci_low", "eci_ci_high",
               "reasoning_tokens", "answer_tokens", "aa_output_tokens"]]


def plot(df, save_prefix):
    x = df["eci"].to_numpy()
    y = df["aa_index"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    r = np.corrcoef(x, y)[0, 1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x, y, s=36, alpha=0.7, color="#1f77b4", edgecolor="white", linewidth=0.6)

    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#d62728", linewidth=2,
            label=f"y = {slope:.3f}·x + {intercept:.2f}\n$R^2$ = {r2:.3f}, r = {r:.3f}")

    ax.set_xlabel("ECI (Epoch Capability Index)", fontsize=12)
    ax.set_ylabel("AA Index (Artificial Analysis)", fontsize=12)
    ax.set_title(f"AA Index vs. ECI ({len(df)} models)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_prefix}.png")

    return {"slope": slope, "intercept": intercept, "r2": r2, "pearson_r": r, "n": len(df)}


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_points()
    save_prefix = f"{OUT_DIR}/aa_index_vs_eci"
    fit = plot(df, save_prefix)

    df.to_csv(f"{save_prefix}_points.csv", index=False)
    print(f"Saved: {save_prefix}_points.csv ({len(df)} rows)")

    pd.DataFrame([fit]).to_csv(f"{save_prefix}_fit.csv", index=False)
    print(f"Saved: {save_prefix}_fit.csv")
    print(f"Fit: y = {fit['slope']:.4f}·x + {fit['intercept']:.4f}  "
          f"(R² = {fit['r2']:.4f}, r = {fit['pearson_r']:.4f}, n = {fit['n']})")
