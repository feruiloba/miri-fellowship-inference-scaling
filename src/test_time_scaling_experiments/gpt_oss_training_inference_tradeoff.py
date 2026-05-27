"""
Per-benchmark training-vs-inference compute substitution rate from the
gpt-oss-20b / gpt-oss-120b sibling pair.

For each benchmark we measure four finite-difference slopes from the 2×2
grid of (size, effort):

    α_low  = [s(120, low)  − s(20, low) ] / log10(9)            # ∂s/∂log(C_train) at low effort
    α_high = [s(120, high) − s(20, high)] / log10(9)            # ∂s/∂log(C_train) at high effort
    β_20   = [s(20, high)  − s(20, low) ] / Δlog(tokens_20b)    # ∂s/∂log(C_inf) at 20b
    β_120  = [s(120, high) − s(120, low)] / Δlog(tokens_120b)   # ∂s/∂log(C_inf) at 120b

Substitution rate r = α/β. We report two corner-consistent values:

    r_LL = α_low  / β_20    (lower-left anchor: small model, low effort)
    r_UR = α_high / β_120   (upper-right anchor: large model, high effort)

r > 1 → 1 log-decade of training buys more score than 1 log-decade of
        inference; benchmark is training-bound.
r < 1 → inference scaling compensates well; benchmark is inference-amenable.

Outputs:
  output/test_time_scaling_experiments/gpt_oss_training_inference_tradeoff.{csv,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVAL_CSV = ROOT / "data" / "artificial_analysis" / "aa_evaluations_combined.csv"
OUT_DIR = ROOT / "output" / "test_time_scaling_experiments"

MODEL_SLUGS = {
    "20b":  {"high": "gpt-oss-20b",  "low": "gpt-oss-20b-low"},
    "120b": {"high": "gpt-oss-120b", "low": "gpt-oss-120b-low"},
}

TRAINING_LOG_RATIO = np.log10(4.94e24 / 5.49e23)  # ≈ log10(9.00) ≈ 0.954


def _load_grid() -> dict[str, dict[tuple[str, str], dict]]:
    """Returns {benchmark: {(size, effort): {"tokens", "score"}}}."""
    ev = pd.read_csv(EVAL_CSV)
    all_slugs = [s for sd in MODEL_SLUGS.values() for s in sd.values()]
    ev = ev[ev["model_slug"].isin(all_slugs)].copy()
    ev = ev.dropna(subset=["total_output_tokens", "score_raw"])
    ev = ev[ev["total_output_tokens"] > 0]

    slug_to_corner = {
        slug: (size, effort)
        for size, ed in MODEL_SLUGS.items()
        for effort, slug in ed.items()
    }
    grid: dict[str, dict[tuple[str, str], dict]] = {}
    for _, row in ev.iterrows():
        size, effort = slug_to_corner[row["model_slug"]]
        grid.setdefault(row["benchmark"], {})[(size, effort)] = {
            "tokens": float(row["total_output_tokens"]),
            "score":  float(row["score_raw"]),
        }
    return grid


def _compute_row(bench: str, cells: dict[tuple[str, str], dict]) -> dict | None:
    required = [("20b", "low"), ("20b", "high"), ("120b", "low"), ("120b", "high")]
    if not all(k in cells for k in required):
        return None

    s20L, s20H = cells[("20b", "low")]["score"],  cells[("20b", "high")]["score"]
    s120L, s120H = cells[("120b", "low")]["score"], cells[("120b", "high")]["score"]
    t20L, t20H = cells[("20b", "low")]["tokens"],  cells[("20b", "high")]["tokens"]
    t120L, t120H = cells[("120b", "low")]["tokens"], cells[("120b", "high")]["tokens"]

    dlog_t20  = np.log10(t20H)  - np.log10(t20L)
    dlog_t120 = np.log10(t120H) - np.log10(t120L)

    alpha_low  = (s120L - s20L) / TRAINING_LOG_RATIO
    alpha_high = (s120H - s20H) / TRAINING_LOG_RATIO
    beta_20    = (s20H - s20L) / dlog_t20  if dlog_t20  != 0 else np.nan
    beta_120   = (s120H - s120L) / dlog_t120 if dlog_t120 != 0 else np.nan

    def _safe_ratio(num, denom):
        if denom is None or np.isnan(denom) or denom == 0:
            return np.nan
        return num / denom

    r_LL = _safe_ratio(alpha_low,  beta_20)
    r_UR = _safe_ratio(alpha_high, beta_120)

    return {
        "benchmark": bench,
        "s20_low": s20L, "s20_high": s20H,
        "s120_low": s120L, "s120_high": s120H,
        "alpha_low":  alpha_low,
        "alpha_high": alpha_high,
        "beta_20":    beta_20,
        "beta_120":   beta_120,
        "r_LL": r_LL,
        "r_UR": r_UR,
    }


def _plot(df: pd.DataFrame, out_png: Path) -> None:
    """Single bar per benchmark: slope ratio β_120 / β_20.

    Ratio < 1 → small model scales at least as well per token (inference
                pays off — the 20b uses extra thinking productively).
    Ratio ≈ 1 → both models scale similarly.
    Ratio > 1 → big model gains more per token (small model below
                capability threshold for this benchmark).

    Benchmarks where β_20 ≤ small noise threshold are labelled
    "small model floor-pinned" and shown at the right edge.
    """
    CAP = 6.0    # display cap; bars exceeding this are clipped + annotated

    work = df.copy()
    work = work[(work["beta_20"].notna()) & (work["beta_120"].notna())].copy()

    rows = []
    for _, r in work.iterrows():
        b20, b120 = r["beta_20"], r["beta_120"]
        # Only flag as floor-pinned if the small model's slope is
        # non-positive (i.e. no scaling signal at all). A small positive
        # slope is still a meaningful denominator.
        if b20 <= 0:
            ratio = np.inf
            note = "β_20 ≤ 0"
        else:
            ratio = b120 / b20
            note = ""
        rows.append({"benchmark": r["benchmark"], "ratio": ratio, "note": note,
                     "beta_20": b20, "beta_120": b120})
    rdf = pd.DataFrame(rows)
    # Sort: finite ratios ascending, ∞ ones at the end
    rdf["sort_key"] = rdf["ratio"].replace(np.inf, CAP * 100)
    rdf = rdf.sort_values("sort_key").reset_index(drop=True)
    rdf["ratio_clipped"] = rdf["ratio"].clip(upper=CAP)

    fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * len(rdf))))
    y = np.arange(len(rdf))
    colors = []
    for r in rdf["ratio"]:
        if r == np.inf:
            colors.append("#7f7f7f")     # gray — degenerate
        elif r <= 1.0:
            colors.append("#2ca02c")     # green — small model scales well
        else:
            colors.append("#d62728")     # red — big model scales better

    ax.barh(y, rdf["ratio_clipped"], color=colors, edgecolor="white", linewidth=0.5)

    for yi, r, r_clip, note in zip(y, rdf["ratio"], rdf["ratio_clipped"], rdf["note"]):
        if r == np.inf:
            tag = f"∞  ({note})"
        elif r > CAP:
            tag = f"{r:.1f}  →"
        else:
            tag = f"{r:.2f}"
        ax.annotate(tag, (r_clip, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9)

    ax.axvline(1.0, color="#222", linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(rdf["benchmark"], fontsize=10)
    ax.set_xlim(0, CAP + 0.6)

    # Plain-English bottom labels, positioned BELOW the bars to keep clear
    # of the title.
    label_y = -1.0
    ax.text(0.5, label_y,
            "← Small model scales as well per token\n(inference compensates well)",
            ha="center", va="top", fontsize=9, color="#2ca02c", style="italic")
    ax.text(CAP - 1.0, label_y,
            "Big model gains far more per token →\n(small model below capability threshold)",
            ha="center", va="top", fontsize=9, color="#d62728", style="italic")
    ax.set_ylim(label_y - 1.4, len(rdf) - 0.4)

    ax.set_xlabel(
        "Slope ratio  β(gpt-oss-120b) / β(gpt-oss-20b)\n"
        "(= 1 means both models gain the same score per 10× more tokens)",
        fontsize=10,
    )
    ax.set_title(
        "Inference-token scaling: how much better the big model uses extra thinking\n"
        "(gpt-oss-20b vs 120b, same release day, 9× training compute)",
        fontsize=12,
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")


def main() -> None:
    grid = _load_grid()
    rows = []
    skipped = []
    for bench, cells in grid.items():
        row = _compute_row(bench, cells)
        if row is None:
            skipped.append(bench)
            continue
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("benchmark").reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "gpt_oss_training_inference_tradeoff.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    png_path = OUT_DIR / "gpt_oss_training_inference_tradeoff.png"
    _plot(df, png_path)
    print(f"Wrote {png_path}")

    if skipped:
        print(f"Skipped (missing one of 4 corners): {skipped}")

    print()
    print("Summary (sorted by mean |r|):")
    show = df.copy()
    show["mean_r"] = show[["r_LL", "r_UR"]].abs().mean(axis=1)
    show = show.sort_values("mean_r", ascending=False)
    cols = ["benchmark", "alpha_low", "alpha_high", "beta_20", "beta_120", "r_LL", "r_UR"]
    print(show[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
