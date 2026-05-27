"""
2D heatmap of top-5 mean benchmark score across (training FLOPs, inference
tokens) cells.

Same layout as train_flops_vs_inference_tokens_avg_score.py, but in each
log-spaced (train_flop, inference_tokens) cell we keep only the top-5
scoring runs and average them (or fewer than 5, if the cell is smaller).
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVAL_CSV = ROOT / "data" / "artificial_analysis" / "aa_evaluations_combined.csv"
MERGED_CSV = ROOT / "data" / "merged_datasets.csv"
OUT_DIR = ROOT / "output" / "iso_capability"

BENCH_CONFIGS = [
    {"benchmark": "aime-2025", "label": "AIME 2025", "y_step": 0.5},
    {"benchmark": "gpqa-diamond", "label": "GPQA Diamond"},
    {"benchmark": "humanitys-last-exam", "label": "Humanity's Last Exam"},
    {"benchmark": "artificial-analysis-long-context-reasoning",
     "label": "AA Long Context Reasoning"},
]

TOP_K = 5


def _train_flop_lookup() -> pd.DataFrame:
    m = pd.read_csv(MERGED_CSV, usecols=["AA_slug", "Training compute (FLOP)"])
    m = m.rename(columns={
        "AA_slug": "model_slug",
        "Training compute (FLOP)": "train_flop",
    })
    m["train_flop"] = pd.to_numeric(m["train_flop"], errors="coerce")
    m = m.dropna(subset=["model_slug", "train_flop"])
    return m.drop_duplicates(subset=["model_slug"])


def _load(benchmark: str) -> pd.DataFrame:
    ev = pd.read_csv(EVAL_CSV)
    ev = ev[ev["benchmark"] == benchmark].copy()
    ev = ev.dropna(subset=["total_output_tokens", "score_raw", "model_slug"])
    ev = ev[ev["total_output_tokens"] > 0]
    ev = ev.merge(_train_flop_lookup(), on="model_slug", how="inner")
    ev = ev[ev["train_flop"] > 0].copy()
    return ev


def _oom_edges(values: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Bin edges at log10 multiples of `step` covering the data range."""
    log_min = math.log10(values.min())
    log_max = math.log10(values.max())
    lo = math.floor(log_min / step) * step
    hi = math.ceil(log_max / step) * step
    if hi <= lo:
        hi = lo + step
    n = int(round((hi - lo) / step)) + 1
    return 10.0 ** np.linspace(lo, hi, n)


def _cell_stats(df: pd.DataFrame, x_edges: np.ndarray, y_edges: np.ndarray):
    """Return arrays of shape (nx, ny): top-K mean score, count, top_n."""
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    xb = np.digitize(df["train_flop"].to_numpy(), x_edges) - 1
    yb = np.digitize(df["total_output_tokens"].to_numpy(), y_edges) - 1

    mean_score = np.full((nx, ny), np.nan)
    count = np.zeros((nx, ny), dtype=int)
    top_n = np.zeros((nx, ny), dtype=int)
    scores = df["score_raw"].to_numpy()
    for i in range(nx):
        for j in range(ny):
            mask = (xb == i) & (yb == j)
            n = int(mask.sum())
            if n == 0:
                continue
            cell_scores = scores[mask]
            top = np.sort(cell_scores)[-TOP_K:]
            mean_score[i, j] = top.mean()
            count[i, j] = n
            top_n[i, j] = len(top)
    return mean_score, count, top_n


def _draw_panel(ax, label: str, df: pd.DataFrame,
                x_step: float = 1.0, y_step: float = 1.0):
    x = df["train_flop"].to_numpy(dtype=float)
    y = df["total_output_tokens"].to_numpy(dtype=float)

    x_edges = _oom_edges(x, x_step)
    y_edges = _oom_edges(y, y_step)
    mean_score, count, top_n = _cell_stats(df, x_edges, y_edges)
    nx, ny = mean_score.shape

    valid = mean_score[~np.isnan(mean_score)]
    vmin = float(valid.min()) if valid.size else 0.0
    vmax = float(valid.max()) if valid.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    mesh = ax.pcolormesh(
        x_edges, y_edges, mean_score.T,
        cmap="viridis", shading="flat", vmin=vmin, vmax=vmax,
    )
    ax.scatter(x, y, s=8, alpha=0.45, color="#222222", edgecolor="none",
               zorder=2)

    for i in range(nx):
        for j in range(ny):
            if count[i, j] == 0:
                continue
            cx = 10 ** ((np.log10(x_edges[i]) + np.log10(x_edges[i + 1])) / 2)
            cy = 10 ** ((np.log10(y_edges[j]) + np.log10(y_edges[j + 1])) / 2)
            cell_val = mean_score[i, j]
            txt_color = "white" if cell_val < (vmin + vmax) / 2 else "black"
            ax.text(
                cx, cy,
                f"{cell_val:.2f}\ntop {top_n[i, j]}/{count[i, j]}",
                ha="center", va="center", fontsize=7, color=txt_color, zorder=3,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training compute (FLOP)")
    ax.set_ylabel("Inference tokens (per eval)")
    ax.set_title(label, fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    return mesh, x_edges, y_edges, mean_score, count, top_n


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = []
    for cfg in BENCH_CONFIGS:
        df = _load(cfg["benchmark"])
        if df.empty:
            print(f"skipping {cfg['benchmark']!r}: empty")
            continue
        panels.append((cfg, df))

    if not panels:
        print("Nothing to plot.")
        return

    cols = 2
    rows_n = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7.5 * cols, 5.5 * rows_n),
                             squeeze=False)
    rows_out = []
    for idx, (cfg, df) in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        mesh, x_edges, y_edges, mean_score, count, top_n = _draw_panel(
            ax, cfg["label"], df,
            x_step=cfg.get("x_step", 1.0),
            y_step=cfg.get("y_step", 1.0),
        )
        if mesh is not None:
            cbar = fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02)
            cbar.set_label(f"Top-{TOP_K} mean score", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
        nx, ny = count.shape
        for i in range(nx):
            for j in range(ny):
                if count[i, j] == 0:
                    continue
                rows_out.append({
                    "benchmark": cfg["benchmark"],
                    "train_flop_lo": x_edges[i],
                    "train_flop_hi": x_edges[i + 1],
                    "inference_tokens_lo": y_edges[j],
                    "inference_tokens_hi": y_edges[j + 1],
                    "n": int(count[i, j]),
                    "n_top": int(top_n[i, j]),
                    "top_mean_score": float(mean_score[i, j]),
                })
        print(f"  {cfg['benchmark']}: {len(df)} rows, "
              f"{df['model_slug'].nunique()} models")
    for j in range(len(panels), rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        f"Top-{TOP_K} mean benchmark score across "
        f"(training FLOPs, inference tokens) bands  "
        f"(color scale per panel)",
        fontsize=13,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    out_png = OUT_DIR / "train_flops_vs_inference_tokens_top5_score.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    out_csv = OUT_DIR / "train_flops_vs_inference_tokens_top5_score.csv"
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
