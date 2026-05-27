"""
Plot excess loss (L_actual − L_opt) per model under the Chinchilla scaling law.

Produces a 1x2 figure: ΔL vs. training compute and ΔL vs. publication date.
"""

from pathlib import Path
import matplotlib.pyplot as plt

from _chinchilla import (
    load_models_with_chinchilla,
    filter_for_plotting,
    chinchilla_loss,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "chinchilla_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    d = load_models_with_chinchilla()
    plot_df = filter_for_plotting(d).copy()

    # For MoE rows, substitute N_active = C / (6*D) and recompute delta_L.
    moe_mask = plot_df["is_moe"]
    n_active = (
        plot_df.loc[moe_mask, "Training compute (FLOP)"]
        / (6.0 * plot_df.loc[moe_mask, "D_actual"])
    )
    plot_df.loc[moe_mask, "N_actual"] = n_active
    l_actual = chinchilla_loss(
        plot_df["N_actual"].to_numpy(), plot_df["D_actual"].to_numpy()
    )
    plot_df["delta_L"] = l_actual - plot_df["L_opt"]

    color = "#4C72B0"
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(14, 6))

    axa.scatter(plot_df["Training compute (FLOP)"], plot_df["delta_L"],
                alpha=0.7, color=color)
    axa.axhline(0.0, color="red", linestyle="--", linewidth=1,
                label="Chinchilla-optimal")
    axa.set_xscale("log")
    axa.set_yscale("symlog", linthresh=0.01)
    axa.set_xlabel("Training compute C (FLOP)")
    axa.set_ylabel("L_actual − L_opt (nats/token)")
    axa.set_title("Excess loss vs. Chinchilla-optimal")
    axa.legend()
    axa.grid(True, alpha=0.3)

    axb.scatter(plot_df["Publication date"], plot_df["delta_L"],
                alpha=0.7, color=color)
    axb.axhline(0.0, color="red", linestyle="--", linewidth=1,
                label="Chinchilla-optimal")
    axb.set_yscale("symlog", linthresh=0.01)
    axb.set_xlabel("Publication date")
    axb.set_ylabel("L_actual − L_opt (nats/token)")
    axb.set_title("Excess loss vs. Chinchilla-optimal (by date)")
    axb.legend()
    axb.grid(True, alpha=0.3)
    axb.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    out_png = OUT / "chinchilla_excess_loss.png"
    fig.savefig(out_png, dpi=160)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
