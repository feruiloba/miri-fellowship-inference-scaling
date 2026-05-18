"""
Effort sweep for gpt-5.1 and gpt-5.2 on OTIS Mock AIME 2024-2025,
using only data/eci/benchmarks_runs-PUBLIC VIEW.csv.

The CSV preserves the effort suffix on the `model` field but has no
token counts, so the x-axis here is the effort level itself.

Output:
  output/benchmark_vs_tokens/eci/otis_aime_gpt5_1_vs_gpt5_2_effort.{png,csv}
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data" / "eci" / "benchmarks_runs-PUBLIC VIEW.csv"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens" / "eci"

TASK = "OTIS Mock AIME 2024-2025"
EFFORT_RE = re.compile(r"_(minimal|low|medium|high|xhigh|max)$")
EFFORT_ORDER = ["minimal", "low", "medium", "high", "xhigh", "max"]
EFFORT_RANK = {e: i for i, e in enumerate(EFFORT_ORDER)}

TARGETS = {
    "gpt-5.1-2025-11-13": ("GPT-5.1", "#1f77b4"),
    "gpt-5.2-2025-12-11": ("GPT-5.2", "#d62728"),
}


def _parse_score(s) -> float | None:
    if pd.isna(s):
        return None
    return float(str(s).replace(",", "."))


def main() -> None:
    df = pd.read_csv(CSV)
    df = df[df["task"] == TASK].copy()
    df["effort"] = df["model"].str.extract(EFFORT_RE)
    df["model_base"] = df["model"].apply(lambda m: EFFORT_RE.sub("", m))
    df = df[df["model_base"].isin(TARGETS) & df["effort"].notna()]
    df["score"] = df["Best score (across scorers)"].map(_parse_score)
    df = df.dropna(subset=["score"])
    df["effort_rank"] = df["effort"].map(EFFORT_RANK)

    if df.empty:
        print("No rows.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for model_base, (name, color) in TARGETS.items():
        sub = df[df["model_base"] == model_base].sort_values("effort_rank")
        if sub.empty:
            continue
        ax.plot(
            sub["effort_rank"], sub["score"],
            marker="o", color=color, linewidth=1.8, markersize=8,
            markeredgecolor="white", markeredgewidth=0.6,
            label=f"{name} (efforts: {','.join(sub['effort'])})",
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['score']:.3f}",
                (row["effort_rank"], row["score"]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=7, color=color,
            )

    used = sorted(df["effort_rank"].unique())
    ax.set_xticks(used)
    ax.set_xticklabels([EFFORT_ORDER[r] for r in used])
    ax.set_xlabel("Reasoning effort")
    ax.set_ylabel("Best score")
    ax.set_title(f"{TASK} — GPT-5.1 vs GPT-5.2 effort sweep")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "otis_aime_gpt5_1_vs_gpt5_2_effort.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    out_csv = OUT_DIR / "otis_aime_gpt5_1_vs_gpt5_2_effort.csv"
    df.sort_values(["model_base", "effort_rank"])[
        ["task", "model_base", "effort", "id", "score"]
    ].to_csv(out_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    print()
    print(df.groupby("model_base")
          .agg(n=("score", "size"),
               efforts=("effort", lambda s: ",".join(sorted(s, key=EFFORT_RANK.get))),
               score_min=("score", "min"), score_max=("score", "max"))
          .to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
