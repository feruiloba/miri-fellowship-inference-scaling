"""
Per-training-compute frontier curves on the combined AA + ECI log_viewer
dataset (2 panels: GPQA Diamond and AIME).

Half-decade-aligned training-compute bands. MoE rows kept (training compute
is well-defined for both dense and MoE).

Output:
  output/benchmark_vs_tokens/combined_eci_aa/score_vs_compute_bands_by_training_compute.{png,csv}
"""

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _combined_eci_aa import BENCHMARKS, OUT_DIR, ROOT, load_combined  # noqa: E402

EPOCH_CSV = ROOT / "data" / "eci" / "epoch_all_ai_models.csv"
N_BANDS = 8
TOP_K = 10
MIN_BAND_N = 4

_TRAIN_EXPONENTS = np.arange(22, 27.0001, 0.5)
TRAIN_EDGES = np.power(10.0, _TRAIN_EXPONENTS)


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower().strip()).strip("-")


def _exp_label(exp: float) -> str:
    return f"$10^{{{exp:g}}}$"


TRAIN_LABELS = [
    f"{_exp_label(_TRAIN_EXPONENTS[i])}–{_exp_label(_TRAIN_EXPONENTS[i + 1])} FLOP"
    for i in range(len(TRAIN_EDGES) - 1)
]


def _strip_suffix(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(name)).strip()


def _load_train_lookup() -> dict[str, float]:
    ep = pd.read_csv(EPOCH_CSV, usecols=["Model", "Training compute (FLOP)"])
    ep = ep.dropna(subset=["Model", "Training compute (FLOP)"]).copy()
    ep["slug"] = ep["Model"].map(slugify)
    ep = ep.sort_values("Training compute (FLOP)", ascending=False).drop_duplicates(
        "slug", keep="first"
    )
    return dict(zip(ep["slug"], ep["Training compute (FLOP)"]))


def _resolve(model: str, model_slug: str, lookup: dict[str, float]) -> float | None:
    if model_slug in lookup:
        return lookup[model_slug]
    s = slugify(_strip_suffix(model))
    if s in lookup:
        return lookup[s]
    return None


def _make_train_bands(C: pd.Series) -> pd.Series:
    edges = TRAIN_EDGES.copy()
    edges[-1] *= 1.000001
    return pd.cut(C, bins=edges, include_lowest=True, labels=False).astype("Int64")


def _build_panel(benchmark: str, lookup: dict[str, float]) -> tuple | None:
    df = load_combined(benchmark)
    if df.empty:
        return None
    df["train_compute"] = df.apply(
        lambda r: _resolve(r["model"], r["model_slug"], lookup), axis=1
    )
    df = df.dropna(subset=["train_compute"])
    if df.empty:
        return None

    log_tokens = np.log10(df["total_inference_tokens"].to_numpy(dtype=float))
    edges = np.linspace(log_tokens.min(), log_tokens.max(), N_BANDS + 1)
    centers = 10 ** ((edges[:-1] + edges[1:]) / 2)
    df["band"] = pd.cut(log_tokens, bins=edges, include_lowest=True, labels=False)
    df["train_band"] = _make_train_bands(df["train_compute"])
    df = df.dropna(subset=["train_band"])
    if df.empty:
        return None

    rows = []
    for (tband, band), sub in df.groupby(["train_band", "band"], observed=True):
        top = sub["score_raw"].nlargest(TOP_K)
        rows.append({
            "train_band": int(tband),
            "train_band_label": TRAIN_LABELS[int(tband)],
            "band_idx": int(band),
            "band_center_tokens": centers[int(band)],
            "n_band": len(sub),
            "n_top": len(top),
            "top_mean": float(top.mean()),
            "top_max": float(top.max()),
        })
    summary = pd.DataFrame(rows)
    summary = summary[summary["n_band"] >= MIN_BAND_N]
    return df, summary


def _draw_panel(ax, benchmark: str, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    ax.scatter(df["total_inference_tokens"], df["score_raw"],
               s=8, alpha=0.10, color="#888888", edgecolor="none")
    cmap = plt.get_cmap("viridis")
    band_n = df.groupby("train_band").size()
    for tband in sorted(summary["train_band"].unique()):
        sub = summary[summary["train_band"] == tband].sort_values("band_idx")
        color = cmap(int(tband) / max(len(TRAIN_LABELS) - 1, 1))
        ax.plot(sub["band_center_tokens"], sub["top_mean"],
                marker="o", color=color, linewidth=1.6, markersize=5,
                label=f"{TRAIN_LABELS[tband]} (n={int(band_n.get(tband, 0))})")
        for _, row in sub.iterrows():
            ax.annotate(f"{int(row['n_top'])}",
                        (row["band_center_tokens"], row["top_mean"]),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=6, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Total inference tokens")
    ax.set_ylabel(f"Top-{TOP_K} mean score per band")
    ax.set_title(benchmark, fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(title="training compute", loc="lower right", fontsize=6, framealpha=0.9)


def main() -> None:
    lookup = _load_train_lookup()
    panels = []
    for b in BENCHMARKS:
        result = _build_panel(b, lookup)
        if result is None:
            print(f"skipping {b!r}")
            continue
        panels.append((b, *result))
    if not panels:
        return

    cols = 2
    rows_n = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 5 * rows_n), squeeze=False)
    for idx, (b, df, summary) in enumerate(panels):
        _draw_panel(axes[idx // cols][idx % cols], b, df, summary)
    for j in range(len(panels), rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)
    fig.suptitle(
        f"Compute-frontier curves by training compute  "
        f"(top-{TOP_K} mean, half-decade train bands, AA + log_viewer)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "score_vs_compute_bands_by_training_compute.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    combined = pd.concat(
        [s.assign(benchmark=b) for b, _, s in panels], ignore_index=True
    )
    out_csv = OUT_DIR / "score_vs_compute_bands_by_training_compute.csv"
    combined.to_csv(out_csv, index=False)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    for b, df, summary in panels:
        print(f"  {b}: {len(df)} rows with resolved training compute, {len(summary)} plotted points")


if __name__ == "__main__":
    main()
