"""
Per-parameter-count frontier curves on the combined AA + ECI log_viewer
dataset (2 panels: GPQA Diamond and AIME).

MoE rows excluded the same way as the AA-only version.

Output:
  output/benchmark_vs_tokens/combined_eci_aa/score_vs_compute_bands_by_params.{png,csv}
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
MOE_PATTERN = re.compile(r"mixture|\bMoE\b|expert|active", re.IGNORECASE)

DECADE_EDGES = np.array([1e9, 1e10, 1e11, 1e12, 1e13])


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower().strip()).strip("-")


def _fmt(n: float) -> str:
    if n >= 1e12:
        return f"{n / 1e12:g}T"
    if n >= 1e9:
        return f"{n / 1e9:g}B"
    if n >= 1e6:
        return f"{n / 1e6:g}M"
    return f"{n:g}"


PARAM_LABELS = [
    f"{_fmt(DECADE_EDGES[i])}–{_fmt(DECADE_EDGES[i + 1])}"
    for i in range(len(DECADE_EDGES) - 1)
]


def _load_epoch_lookup() -> dict[str, float]:
    ep = pd.read_csv(EPOCH_CSV, usecols=["Model", "Parameters", "Parameters notes"])
    ep = ep.dropna(subset=["Model", "Parameters"]).copy()
    notes = ep["Parameters notes"].fillna("")
    ep = ep[~notes.str.contains(MOE_PATTERN, regex=True, na=False)]
    ep["slug"] = ep["Model"].map(slugify)
    ep = ep.sort_values("Parameters", ascending=False).drop_duplicates("slug", keep="first")
    return dict(zip(ep["slug"], ep["Parameters"]))


def _make_param_bands(params: pd.Series) -> pd.Series:
    edges = DECADE_EDGES.copy()
    edges[-1] *= 1.000001
    return pd.cut(params, bins=edges, include_lowest=True, labels=False).astype("Int64")


def _build_panel(benchmark: str, slug_to_params: dict[str, float]) -> tuple | None:
    df = load_combined(benchmark)
    if df.empty:
        return None
    df["parameters"] = df["model_slug"].map(slug_to_params)
    df = df.dropna(subset=["parameters"])
    if df.empty:
        return None

    log_tokens = np.log10(df["total_inference_tokens"].to_numpy(dtype=float))
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
    return df, summary


def _draw_panel(ax, benchmark: str, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    ax.scatter(df["total_inference_tokens"], df["score_raw"],
               s=8, alpha=0.10, color="#888888", edgecolor="none")
    cmap = plt.get_cmap("viridis")
    band_n = df.groupby("param_band").size()
    for pband in sorted(summary["param_band"].unique()):
        sub = summary[summary["param_band"] == pband].sort_values("band_idx")
        color = cmap(int(pband) / max(len(PARAM_LABELS) - 1, 1))
        ax.plot(sub["band_center_tokens"], sub["top_mean"],
                marker="o", color=color, linewidth=1.8, markersize=6,
                label=f"{PARAM_LABELS[pband]} (n={int(band_n.get(pband, 0))})")
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
    ax.legend(title="parameters", loc="lower right", fontsize=7, framealpha=0.9)


def main() -> None:
    slug_to_params = _load_epoch_lookup()
    panels = []
    for b in BENCHMARKS:
        result = _build_panel(b, slug_to_params)
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
        f"Per-parameter-count compute-frontier curves  "
        f"(non-MoE, top-{TOP_K} mean, AA + log_viewer)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "score_vs_compute_bands_by_params.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    combined = pd.concat(
        [s.assign(benchmark=b) for b, _, s in panels], ignore_index=True
    )
    out_csv = OUT_DIR / "score_vs_compute_bands_by_params.csv"
    combined.to_csv(out_csv, index=False)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    for b, df, summary in panels:
        print(f"  {b}: {len(df)} rows with known N, {len(summary)} plotted points")


if __name__ == "__main__":
    main()
