"""
Compare o1, Claude 3.7 Sonnet (extended thinking), and Kimi k1.5 on
OTIS Mock AIME 2024-2025 using manual-report data.

Single-panel plot. Each model gets a curve of token spend vs accuracy.
Caveat on token semantics: o1 and Claude 3.7 (extended thinking) report
`thinking_tokens` straight from their tech reports / blog posts, while
Kimi k1.5 reports `mean_response_tokens` from Fig 6 of its paper. Both
are inference-time token spend per problem, but they aren't strictly the
same quantity. Treat the x-axis as "inference tokens (paper convention)".

Output:
  output/benchmark_vs_tokens/manual_reports/o1_vs_sonnet37_vs_kimi_k15_aime.{png,csv}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_CSV = ROOT / "data" / "manual_reports" / "manual_reports_data.csv"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens" / "manual_reports"

BENCHMARK = "OTIS Mock AIME 2024-2025"
TARGETS = {
    "o1":                                    ("o1",                                     "#1f77b4"),
    "Claude 3.7 Sonnet (extended thinking)": ("Claude 3.7 Sonnet (extended thinking)",  "#d62728"),
    "Kimi k1.5":                             ("Kimi k1.5",                              "#2ca02c"),
    "Gemini 2.5 Flash":                      ("Gemini 2.5 Flash",                       "#ff7f0e"),
}


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    df = df[df["benchmark"] == BENCHMARK].copy()
    df = df.dropna(subset=["x_value", "performance"])
    df = df[df["model"].isin(TARGETS)]
    df = df[df["x_value"] > 0]
    if df.empty:
        print("No rows.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for model, (name, color) in TARGETS.items():
        sub = df[df["model"] == model].sort_values("x_value")
        if sub.empty:
            continue
        unit = sub["x_unit"].dropna().iloc[0] if sub["x_unit"].notna().any() else "tokens"
        ax.plot(
            sub["x_value"], sub["performance"],
            marker="o", color=color, linewidth=1.6, markersize=7,
            markeredgecolor="white", markeredgewidth=0.5,
            label=f"{name}  (n={len(sub)}, x_unit={unit})",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference tokens (paper convention; see legend)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{BENCHMARK} — inference-token scaling (manual reports)")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "o1_vs_sonnet37_vs_kimi_k15_aime.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    out_csv = OUT_DIR / "o1_vs_sonnet37_vs_kimi_k15_aime.csv"
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
