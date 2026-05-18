"""
Compare Kimi k1.5 and Gemini 2.5 Flash on GPQA Diamond, using manual-report data.

Kimi k1.5 reports `mean_response_tokens` (Fig 6 of the k1.5 paper);
Gemini 2.5 Flash reports `thinking_tokens` (Gemini 2.5 tech report Fig 4).
Both are inference-time token spend per question, not strictly identical
quantities — keep that in mind when comparing absolute x positions.

Output:
  output/benchmark_vs_tokens/manual_reports/kimi_k15_vs_gemini_25_gpqa.{png,csv}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_CSV = ROOT / "data" / "manual_reports" / "manual_reports_data.csv"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens" / "manual_reports"

BENCHMARK = "GPQA Diamond"
TARGETS = {
    "Kimi k1.5":        "#2ca02c",
    "Gemini 2.5 Flash": "#ff7f0e",
}


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    df = df[df["benchmark"] == BENCHMARK].copy()
    df = df.dropna(subset=["x_value", "performance"])
    df = df[df["model"].isin(TARGETS) & (df["x_value"] > 0)]
    if df.empty:
        print("No rows.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for model, color in TARGETS.items():
        sub = df[df["model"] == model].sort_values("x_value")
        if sub.empty:
            continue
        unit = sub["x_unit"].dropna().iloc[0] if sub["x_unit"].notna().any() else "tokens"
        ax.plot(
            sub["x_value"], sub["performance"],
            marker="o", color=color, linewidth=1.6, markersize=6,
            markeredgecolor="white", markeredgewidth=0.5,
            label=f"{model}  (n={len(sub)}, x_unit={unit})",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference tokens (paper convention; see legend)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{BENCHMARK} — inference-token scaling (manual reports)")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "kimi_k15_vs_gemini_25_gpqa.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    out_csv = OUT_DIR / "kimi_k15_vs_gemini_25_gpqa.csv"
    df.sort_values(["model", "x_value"]).to_csv(out_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    print()
    summary = (
        df.groupby("model")
        .agg(n=("performance", "size"),
             x_min=("x_value", "min"), x_max=("x_value", "max"),
             score_min=("performance", "min"), score_max=("performance", "max"),
             x_unit=("x_unit", lambda s: s.dropna().iloc[0] if s.notna().any() else ""))
    )
    print(summary.to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
