"""
Plot actual (N, D) vs. Chinchilla-optimal (N_opt, D_opt) per model.

Produces a 2x2 figure: N_ratio and D_ratio on the y-axis, with training
compute and publication date on the x-axes.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from _chinchilla import (
    load_models_with_chinchilla,
    filter_for_plotting,
    G,
    ALPHA,
    BETA,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "chinchilla_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    d = load_models_with_chinchilla()

    print(f"G = {G:.4f}")
    print(f"alpha/(alpha+beta) = {ALPHA / (ALPHA + BETA):.4f}  "
          f"beta/(alpha+beta) = {BETA / (ALPHA + BETA):.4f}")
    print()

    cols = ["Model", "Training compute (FLOP)", "N_actual", "N_opt", "N_ratio",
            "D_actual", "D_opt", "D_ratio"]
    pd.options.display.float_format = "{:.3e}".format
    summary = d[cols].sort_values("Training compute (FLOP)")
    print(summary.to_string(index=False))
    summary.to_csv(OUT / "chinchilla_param_token_vs_optimal.csv", index=False)

    plot_df = filter_for_plotting(d)
    dense = plot_df[~plot_df["is_moe"]]
    moe = plot_df[plot_df["is_moe"]]
    c_dense = "#4C72B0"
    c_moe = "#C44E52"
    c_dense_d = "#DD8452"
    c_moe_d = "#8172B2"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    (ax1, ax2), (ax3, ax4) = axes

    ax1.scatter(dense["Training compute (FLOP)"], dense["N_ratio"], alpha=0.7,
                color=c_dense, label="Dense")
    ax1.scatter(moe["Training compute (FLOP)"], moe["N_ratio"], alpha=0.8,
                color=c_moe, marker="^", label="MoE (N_total)")
    ax1.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Chinchilla-optimal")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Training compute C (FLOP)")
    ax1.set_ylabel("N_actual / N_opt")
    ax1.set_title("Parameters vs. Chinchilla-optimal")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.scatter(dense["Training compute (FLOP)"], dense["D_ratio"],
                alpha=0.7, color=c_dense_d, label="Dense")
    ax2.scatter(moe["Training compute (FLOP)"], moe["D_ratio"],
                alpha=0.8, color=c_moe_d, marker="^", label="MoE")
    ax2.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Chinchilla-optimal")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Training compute C (FLOP)")
    ax2.set_ylabel("D_actual / D_opt")
    ax2.set_title("Training tokens vs. Chinchilla-optimal")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.scatter(dense["Publication date"], dense["N_ratio"], alpha=0.7,
                color=c_dense, label="Dense")
    ax3.scatter(moe["Publication date"], moe["N_ratio"], alpha=0.8,
                color=c_moe, marker="^", label="MoE (N_total)")
    ax3.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Chinchilla-optimal")
    ax3.set_yscale("log")
    ax3.set_xlabel("Publication date")
    ax3.set_ylabel("N_actual / N_opt")
    ax3.set_title("Parameters vs. Chinchilla-optimal (by date)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis="x", rotation=30)

    ax4.scatter(dense["Publication date"], dense["D_ratio"],
                alpha=0.7, color=c_dense_d, label="Dense")
    ax4.scatter(moe["Publication date"], moe["D_ratio"],
                alpha=0.8, color=c_moe_d, marker="^", label="MoE")
    ax4.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Chinchilla-optimal")
    ax4.set_yscale("log")
    ax4.set_xlabel("Publication date")
    ax4.set_ylabel("D_actual / D_opt")
    ax4.set_title("Training tokens vs. Chinchilla-optimal (by date)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    out_png = OUT / "chinchilla_param_token_vs_optimal.png"
    fig.savefig(out_png, dpi=160)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
