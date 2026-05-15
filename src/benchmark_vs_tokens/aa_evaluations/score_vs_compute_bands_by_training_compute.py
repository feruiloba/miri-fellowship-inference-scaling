"""
Per-training-compute frontier curves of score vs test-time compute, across
several important AA benchmarks.

For each benchmark, bin every run (whose base model has a known Epoch
training-compute value) into log-spaced inference-token bands and half-decade-
aligned training-compute bands; in each (train_band, inf_band) cell take the
top-10 scores and average them. One curve per training-compute band in each
panel shows how the inference-time scaling lift differs by training scale.

Resolution of training compute per row:
  1. exact model_slug match against slugified Epoch Model
  2. slugified-(model name with trailing "(...)" stripped) match

MoE filter is *not* applied here — training compute is well-defined for both
dense and MoE models, so we keep both.

Output:
  output/benchmark_vs_tokens/aa_evaluations/score_vs_compute_bands_by_training_compute.{png,csv}
"""

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EVAL_CSV = ROOT / "data" / "artificial_analysis" / "aa_evaluations_combined.csv"
EPOCH_CSV = ROOT / "data" / "eci" / "epoch_all_ai_models.csv"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens" / "aa_evaluations"

BENCHMARKS = [
    "gpqa-diamond",
    "aime-2025",
    "humanitys-last-exam",
    "artificial-analysis-long-context-reasoning",
]
N_BANDS = 8
TOP_K = 10
MIN_BAND_N = 2

# Half-decade training-compute edges (in FLOP), exponents 22 → 27 step 0.5.
_TRAIN_EXPONENTS = np.arange(22, 27.0001, 0.5)
TRAIN_EDGES = np.power(10.0, _TRAIN_EXPONENTS)


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower().strip()).strip("-")


def strip_config_suffix(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(name)).strip()


def _load_train_lookup() -> dict[str, float]:
    ep = pd.read_csv(EPOCH_CSV, usecols=["Model", "Training compute (FLOP)"])
    ep = ep.dropna(subset=["Model", "Training compute (FLOP)"]).copy()
    ep["slug"] = ep["Model"].map(slugify)
    ep = ep.sort_values(
        "Training compute (FLOP)", ascending=False
    ).drop_duplicates(subset=["slug"], keep="first")
    return dict(zip(ep["slug"], ep["Training compute (FLOP)"]))


def _resolve_compute(model: str, model_slug: str, lookup: dict[str, float]) -> float | None:
    if model_slug in lookup:
        return lookup[model_slug]
    name_slug = slugify(strip_config_suffix(model))
    if name_slug in lookup:
        return lookup[name_slug]
    return None


def _load(benchmark: str, lookup: dict[str, float]) -> pd.DataFrame:
    ev = pd.read_csv(EVAL_CSV)
    ev = ev[ev["benchmark"] == benchmark].copy()
    ev = ev.dropna(subset=["total_output_tokens", "score_raw", "model_slug", "model"])
    ev = ev[ev["total_output_tokens"] > 0]
    ev["train_compute"] = ev.apply(
        lambda r: _resolve_compute(r["model"], r["model_slug"], lookup), axis=1
    )
    return ev.dropna(subset=["train_compute"])


def _exp_label(exp: float) -> str:
    return f"$10^{{{exp:g}}}$"


TRAIN_LABELS = [
    f"{_exp_label(_TRAIN_EXPONENTS[i])}–{_exp_label(_TRAIN_EXPONENTS[i + 1])} FLOP"
    for i in range(len(TRAIN_EDGES) - 1)
]


def _make_train_bands(C: pd.Series) -> pd.Series:
    edges = TRAIN_EDGES.copy()
    edges[-1] *= 1.000001
    return pd.cut(C, bins=edges, include_lowest=True, labels=False).astype("Int64")


def _build_panel(benchmark: str, lookup: dict[str, float]) -> tuple | None:
    df = _load(benchmark, lookup)
    if df.empty:
        return None

    log_tokens = np.log10(df["total_output_tokens"].to_numpy(dtype=float))
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
    summary = summary.sort_values(["train_band", "band_idx"])
    return df, summary


def _draw_panel(ax, benchmark: str, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    ax.scatter(
        df["total_output_tokens"], df["score_raw"],
        s=8, alpha=0.10, color="#888888", edgecolor="none",
    )

    cmap = plt.get_cmap("viridis")
    band_total_n = df.groupby("train_band").size()
    bands_used = sorted(summary["train_band"].unique())
    for tband in bands_used:
        sub = summary[summary["train_band"] == tband].sort_values("band_idx")
        # Use global slot index so colours match across panels
        color = cmap(int(tband) / max(len(TRAIN_LABELS) - 1, 1))
        ax.plot(
            sub["band_center_tokens"], sub["top_mean"],
            marker="o", color=color, linewidth=1.6, markersize=5,
            label=f"{TRAIN_LABELS[tband]} (n={int(band_total_n.get(tband, 0))})",
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{int(row['n_top'])}",
                (row["band_center_tokens"], row["top_mean"]),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=6, color=color,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Total inference tokens")
    ax.set_ylabel(f"Top-{TOP_K} mean score per band")
    ax.set_title(benchmark, fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(title="training compute", loc="lower right", fontsize=6, framealpha=0.9)


def main(benchmarks: list[str] = BENCHMARKS) -> None:
    lookup = _load_train_lookup()
    panels: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for b in benchmarks:
        result = _build_panel(b, lookup)
        if result is None:
            print(f"skipping {b!r} (no usable rows)")
            continue
        df, summary = result
        panels.append((b, df, summary))

    if not panels:
        print("Nothing to plot.")
        return

    cols = 2
    rows_n = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 5 * rows_n), squeeze=False)
    for idx, (b, df, summary) in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        _draw_panel(ax, b, df, summary)
    for j in range(len(panels), rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        f"Compute-frontier curves by training compute  "
        f"(top-{TOP_K} mean, {N_BANDS} inf-token bands × half-decade train-compute bands)",
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
