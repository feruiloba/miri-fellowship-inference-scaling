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
  output/benchmark_vs_tokens/aa_evaluations/top10_grouped_by_training_compute_band.{png,csv}
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

# Optional horizontal reference line per benchmark: shows how many inference
# tokens each training-compute curve needs to reach a given score. The
# crossing for each curve is annotated with the interpolated token count.
REFERENCE_SCORES: dict[str, float] = {
    "gpqa-diamond": 0.7,
}

# Half-decade training-compute bands covering 10^22 to 10^27 FLOP.
TRAIN_EDGE_EXPONENTS = [22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0]
TRAIN_EDGES = np.array([10.0 ** e for e in TRAIN_EDGE_EXPONENTS])


def _fmt_exp(e: float) -> str:
    return f"{e:.1f}" if (e * 2) % 2 else f"{int(e)}"


TRAIN_LABELS = [
    rf"$10^{{{_fmt_exp(TRAIN_EDGE_EXPONENTS[i])}}}$–$10^{{{_fmt_exp(TRAIN_EDGE_EXPONENTS[i+1])}}}$ FLOP"
    for i in range(len(TRAIN_EDGE_EXPONENTS) - 1)
]


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


def _abbr_tokens(n: float) -> str:
    """Human-readable token count: 1234 → '1.2K', 1840000 → '1.8M'."""
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.1f}{suf}".replace(".0", "")
    return f"{n:.0f}"


def _crossing_tokens(centers: np.ndarray, scores: np.ndarray, y: float
                     ) -> float | None:
    """Linearly interpolate (in log10 tokens) the first crossing of `scores`
    through level `y`, scanning from low → high tokens. Returns the token
    value, or None if the curve never reaches `y`."""
    if len(scores) < 2:
        return None
    log_x = np.log10(centers)
    for i in range(len(scores) - 1):
        y0, y1 = scores[i], scores[i + 1]
        if (y0 - y) * (y1 - y) <= 0 and y0 != y1:
            frac = (y - y0) / (y1 - y0)
            return float(10 ** (log_x[i] + frac * (log_x[i + 1] - log_x[i])))
        if y0 == y:
            return float(centers[i])
    return None


def _draw_panel(ax, benchmark: str, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    ax.scatter(
        df["total_output_tokens"], df["score_raw"],
        s=8, alpha=0.10, color="#888888", edgecolor="none",
    )

    cmap = plt.get_cmap("viridis")
    band_total_n = df.groupby("train_band").size()
    bands_used = sorted(summary["train_band"].unique())
    crossings: list[tuple[int, tuple, float]] = []  # (tband, color, tokens)
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
        ref = REFERENCE_SCORES.get(benchmark)
        if ref is not None:
            xc = _crossing_tokens(
                sub["band_center_tokens"].to_numpy(),
                sub["top_mean"].to_numpy(),
                ref,
            )
            if xc is not None:
                crossings.append((int(tband), color, xc))

    ref = REFERENCE_SCORES.get(benchmark)
    if ref is not None:
        ax.axhline(ref, color="black", linestyle="--", linewidth=1.0, alpha=0.55,
                   label=f"reference score = {ref:.2f}")
        # Sort crossings left-to-right and stagger labels upward to avoid
        # overlap when multiple curves cross close together in x.
        crossings_sorted = sorted(crossings, key=lambda c: c[2])
        for i, (tband, color, xc) in enumerate(crossings_sorted):
            ax.axvline(xc, color=color, linestyle=":", linewidth=1.0, alpha=0.7)
            y_off = 10 + 12 * i
            ax.annotate(
                _abbr_tokens(xc),
                (xc, ref),
                xytext=(0, y_off), textcoords="offset points",
                fontsize=8, color=color, fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=color, linewidth=0.6, alpha=0.9),
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
        f"(top-{TOP_K} mean, {N_BANDS} inf-token bands × decade train-compute bands)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "top10_grouped_by_training_compute_band.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    combined = pd.concat(
        [s.assign(benchmark=b) for b, _, s in panels], ignore_index=True
    )
    out_csv = OUT_DIR / "top10_grouped_by_training_compute_band.csv"
    combined.to_csv(out_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    for b, df, summary in panels:
        print(f"  {b}: {len(df)} rows with resolved training compute, {len(summary)} plotted points")


if __name__ == "__main__":
    main()
