"""
Stacked bar: training compute + assumed lifetime inference compute per model,
with ECI overlay.

Inference compute is approximated as 2 * N_params * LIFETIME_TOKENS, where
LIFETIME_TOKENS is an assumed lifetime inference demand (default 2e12 tokens).
N_params uses AA active parameter count when available, otherwise the Epoch
total parameter count (overestimates inference FLOPs for MoE models).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "chinchilla_analysis"
OUT.mkdir(parents=True, exist_ok=True)

LIFETIME_TOKENS = 2e12


def main():
    merged = pd.read_csv(ROOT / "data" / "merged_datasets.csv")

    mask = (
        merged["Parameters"].notna()
        & merged["Training compute (FLOP)"].notna()
        & merged["eci"].notna()
        & (merged["Training compute (FLOP)"] > 1e24)
        & (~merged["Model"].str.contains("Grok", case=False, na=False))
    )
    df = merged[mask].copy()

    df["n_params"] = df["AA_active_parameter_count"].fillna(df["Parameters"])
    df["train_flops"] = df["Training compute (FLOP)"]
    df["infer_flops"] = 2 * df["n_params"] * LIFETIME_TOKENS
    df["total_flops"] = df["train_flops"] + df["infer_flops"]

    df = df.drop_duplicates(subset=["Model"], keep="first")
    df["Publication date"] = pd.to_datetime(df["Publication date"], errors="coerce")
    df = df.sort_values("Publication date", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(max(11, 0.28 * len(df)), 7))
    x = range(len(df))
    train_color = "#4C72B0"
    infer_color = "#DD8452"

    ax.bar(x, df["train_flops"], color=train_color, label="Training compute")
    ax.bar(x, df["infer_flops"], bottom=df["train_flops"], color=infer_color,
           label=f"Inference compute ({LIFETIME_TOKENS:.0e} tokens lifetime)")

    ax.set_ylabel("Compute (FLOP)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"], rotation=75, ha="right", fontsize=8)

    ax2 = ax.twinx()
    ax2.scatter(x, df["eci"], marker="D", s=45, color="black",
                zorder=5, label="ECI")
    ax2.set_ylabel("ECI")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax.set_title(
        f"Training + assumed lifetime inference compute per model "
        f"(inference = 2·N·{LIFETIME_TOKENS:.0e} tokens)"
    )
    fig.tight_layout()

    png = OUT / "training_plus_inference_compute_bar.png"
    fig.savefig(png, dpi=160)
    print(f"saved {png}")

    df[["Model", "n_params", "train_flops", "infer_flops",
        "total_flops", "eci"]].to_csv(
        OUT / "training_plus_inference_compute_bar.csv", index=False)


if __name__ == "__main__":
    main()
