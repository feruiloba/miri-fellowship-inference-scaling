"""
Per-parameter-count frontier curves of score vs test-time compute, across
several important AA benchmarks.

For each benchmark, bin every (non-MoE, known-N) run into log-spaced compute
bands and decade-aligned parameter bands; in each (param_band, compute_band)
cell take the top-10 scores and average them. One curve per parameter band
in each panel shows how much of the compute-scaling lift comes from bigger
models versus better use of compute at the same size.

MoE filter: Epoch's `Parameters` column reports *total* parameters for MoE
models, which would inflate dense vs MoE comparisons. We drop any model whose
Epoch `Parameters notes` mention "mixture", "MoE", "expert", or "active".

Output:
  output/benchmark_vs_tokens/aa_evaluations/top10_grouped_by_param_band.{png,csv}
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
    # scicode replaces artificial-analysis-long-context-reasoning here because
    # it has the most non-MoE models with known Epoch parameter counts.
    "scicode",
]
N_BANDS = 8
TOP_K = 10
MIN_BAND_N = 1
MOE_PATTERN = re.compile(r"mixture|\bMoE\b|expert|active", re.IGNORECASE)

# Decade-aligned parameter bands.
DECADE_EDGES = np.array([1e9, 1e10, 1e11, 1e12, 1e13])  # <10B, 10-100B, 100B-1T, ≥1T


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower().strip()).strip("-")


def _format_param_count(n: float) -> str:
    if n >= 1e12:
        return f"{n / 1e12:g}T"
    if n >= 1e9:
        return f"{n / 1e9:g}B"
    if n >= 1e6:
        return f"{n / 1e6:g}M"
    return f"{n:g}"


PARAM_LABELS = [
    f"{_format_param_count(DECADE_EDGES[i])}–{_format_param_count(DECADE_EDGES[i + 1])}"
    for i in range(len(DECADE_EDGES) - 1)
]


def _load_epoch_lookup() -> dict[str, float]:
    """slug → parameters, with MoE rows excluded."""
    ep = pd.read_csv(EPOCH_CSV, usecols=["Model", "Parameters", "Parameters notes"])
    ep = ep.dropna(subset=["Model", "Parameters"]).copy()
    notes = ep["Parameters notes"].fillna("")
    is_moe = notes.str.contains(MOE_PATTERN, regex=True, na=False)
    ep = ep[~is_moe]
    ep["slug"] = ep["Model"].map(slugify)
    ep = ep.sort_values("Parameters", ascending=False).drop_duplicates(
        subset=["slug"], keep="first"
    )
    return dict(zip(ep["slug"], ep["Parameters"]))


def _load(benchmark: str, slug_to_params: dict[str, float]) -> pd.DataFrame:
    ev = pd.read_csv(EVAL_CSV)
    ev = ev[ev["benchmark"] == benchmark].copy()
    ev = ev.dropna(subset=["total_output_tokens", "score_raw", "model_slug"])
    ev = ev[ev["total_output_tokens"] > 0]
    ev["parameters"] = ev["model_slug"].map(slug_to_params)
    return ev.dropna(subset=["parameters"])


def _make_param_bands(params: pd.Series) -> pd.Series:
    edges = DECADE_EDGES.copy()
    edges[-1] *= 1.000001
    return pd.cut(params, bins=edges, include_lowest=True, labels=False).astype("Int64")


def _build_panel(benchmark: str, slug_to_params: dict[str, float]) -> tuple | None:
    df = _load(benchmark, slug_to_params)
    if df.empty:
        return None

    log_tokens = np.log10(df["total_output_tokens"].to_numpy(dtype=float))
    edges = np.linspace(log_tokens.min(), log_tokens.max(), N_BANDS + 1)
    centers = 10 ** ((edges[:-1] + edges[1:]) / 2)
    df["band"] = pd.cut(log_tokens, bins=edges, include_lowest=True, labels=False)
    df["param_band"] = _make_param_bands(df["parameters"])
    df = df.dropna(subset=["param_band"])
    if df.empty:
        return None

    rows = []
    for (pband, band), sub in df.groupby(["param_band", "band"], observed=True):
        top = sub["score_raw"].nlargest(TOP_K)
        rows.append({
            "param_band": int(pband),
            "param_band_label": PARAM_LABELS[int(pband)],
            "band_idx": int(band),
            "band_center_tokens": centers[int(band)],
            "n_band": len(sub),
            "n_top": len(top),
            "top_mean": float(top.mean()),
            "top_max": float(top.max()),
        })
    summary = pd.DataFrame(rows)
    summary = summary[summary["n_band"] >= MIN_BAND_N]
    summary = summary.sort_values(["param_band", "band_idx"])
    return df, summary


def _draw_panel(ax, benchmark: str, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    ax.scatter(
        df["total_output_tokens"], df["score_raw"],
        s=8, alpha=0.10, color="#888888", edgecolor="none",
    )

    cmap = plt.get_cmap("viridis")
    band_total_n = df.groupby("param_band").size()
    bands_used = sorted(summary["param_band"].unique())
    for i, pband in enumerate(bands_used):
        sub = summary[summary["param_band"] == pband].sort_values("band_idx")
        # Color slot is fixed across panels so the same parameter band has the
        # same colour in every benchmark.
        color = cmap(int(pband) / max(len(PARAM_LABELS) - 1, 1))
        ax.plot(
            sub["band_center_tokens"], sub["top_mean"],
            marker="o", color=color, linewidth=1.8, markersize=6,
            label=f"{PARAM_LABELS[pband]} (n={int(band_total_n.get(pband, 0))})",
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
    ax.legend(title="parameters", loc="lower right", fontsize=7, framealpha=0.9)


def main(benchmarks: list[str] = BENCHMARKS) -> None:
    slug_to_params = _load_epoch_lookup()
    panels: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for b in benchmarks:
        result = _build_panel(b, slug_to_params)
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
        f"Per-parameter-count compute-frontier curves  "
        f"(non-MoE only, top-{TOP_K} mean, {N_BANDS} compute bands × decade param bands)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "top10_grouped_by_param_band.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    combined = pd.concat(
        [s.assign(benchmark=b) for b, _, s in panels], ignore_index=True
    )
    out_csv = OUT_DIR / "top10_grouped_by_param_band.csv"
    combined.to_csv(out_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    for b, df, summary in panels:
        print(f"  {b}: {len(df)} non-MoE rows with known N, {len(summary)} plotted points")


if __name__ == "__main__":
    main()
