"""
Shared Chinchilla scaling-law formulas and per-model dataset prep.

Formulas from https://www.harmdevries.com/post/model-size-vs-compute-overhead/

    L(N, D) = E + A / N^alpha + B / D^beta
    G       = (alpha * A / (beta * B)) ^ (1 / (alpha + beta))
    N_opt   = G       * (C / 6) ^ (beta  / (alpha + beta))
    D_opt   = (1 / G) * (C / 6) ^ (alpha / (alpha + beta))
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

E = 1.69
A = 406.4
B = 410.7
ALPHA = 0.32
BETA = 0.28
G = (ALPHA * A / (BETA * B)) ** (1.0 / (ALPHA + BETA))


def chinchilla_optimal(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_opt = G * (C / 6.0) ** (BETA / (ALPHA + BETA))
    d_opt = (1.0 / G) * (C / 6.0) ** (ALPHA / (ALPHA + BETA))
    return n_opt, d_opt


def chinchilla_loss(N: np.ndarray, D: np.ndarray) -> np.ndarray:
    return E + A / np.power(N, ALPHA) + B / np.power(D, BETA)


def load_models_with_chinchilla() -> pd.DataFrame:
    m = pd.read_csv(ROOT / "data" / "merged_datasets.csv")
    m["Training dataset size (total)"] = pd.to_numeric(
        m["Training dataset size (total)"], errors="coerce"
    )
    mask = m["Training compute (FLOP)"].notna() & m["Parameters"].notna()
    d = m[mask].drop_duplicates(subset=["Model"]).copy()

    C = d["Training compute (FLOP)"].to_numpy()
    n_opt, d_opt = chinchilla_optimal(C)
    d["N_opt"] = n_opt
    d["D_opt"] = d_opt
    d["N_actual"] = d["Parameters"]
    d["D_actual"] = d["Training dataset size (total)"]
    d["N_ratio"] = d["N_actual"] / d["N_opt"]
    d["D_ratio"] = d["D_actual"] / d["D_opt"]
    d["L_actual"] = chinchilla_loss(d["N_actual"].to_numpy(),
                                    d["D_actual"].to_numpy())
    d["L_opt"] = chinchilla_loss(d["N_opt"].to_numpy(), d["D_opt"].to_numpy())
    d["delta_L"] = d["L_actual"] - d["L_opt"]
    d["delta_L_pct"] = 100.0 * d["delta_L"] / d["L_actual"]
    d["Publication date"] = pd.to_datetime(d["Publication date"], errors="coerce")
    return d


def filter_for_plotting(d: pd.DataFrame) -> pd.DataFrame:
    plot_df = d.dropna(subset=["D_actual", "Publication date"]).copy()
    plot_df = plot_df[plot_df["Training compute (FLOP)"] > 1e23]
    # Exclude fine-tunes and rows where D_actual is implausibly small for C
    # (e.g. continued-training deltas or fine-tuning dataset sizes).
    plot_df = plot_df[plot_df["Base model"].isna()]
    plot_df = plot_df[plot_df["D_actual"] >= 0.05 * plot_df["D_opt"]]
    # Flag MoE: Epoch's reported C uses N_active, so for MoE C / (6*N_total*T) << 1.
    plot_df["is_moe"] = (
        plot_df["Training compute (FLOP)"]
        / (6 * plot_df["N_actual"] * plot_df["D_actual"])
    ) < 0.5
    return plot_df.sort_values("Publication date")
