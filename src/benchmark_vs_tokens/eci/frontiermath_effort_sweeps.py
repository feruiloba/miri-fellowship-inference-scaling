"""
Effort sweeps on FrontierMath-2025-02-28-Private and FrontierMath-2025-02-28-Public,
using only data/eci/benchmarks_runs-PUBLIC VIEW.csv.

The CSV preserves the effort suffix on the `model` field (e.g.
`gpt-5.2-2025-12-11_xhigh` vs `_high`/`_medium`/`_low`) but does not
contain token counts (its `billable_*_tokens` columns are NaN). So the
x-axis here is the effort level itself, ordered low → medium → high →
xhigh → max.

Only models with ≥3 distinct effort levels are plotted.

Output:
  output/benchmark_vs_tokens/eci/frontiermath_effort_sweeps.{png,csv}
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data" / "eci" / "benchmarks_runs-PUBLIC VIEW.csv"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens" / "eci"

TASKS = [
    "FrontierMath-2025-02-28-Private",
    "FrontierMath-2025-02-28-Public",
]

EFFORT_RE = re.compile(r"_(minimal|low|medium|high|xhigh|max)$")
EFFORT_ORDER = ["minimal", "low", "medium", "high", "xhigh", "max"]
EFFORT_RANK = {e: i for i, e in enumerate(EFFORT_ORDER)}


def _parse_score(s) -> float | None:
    if pd.isna(s):
        return None
    return float(str(s).replace(",", "."))


def _load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df[df["task"].isin(TASKS)].copy()
    df["effort"] = df["model"].str.extract(EFFORT_RE)
    df["model_base"] = df["model"].apply(lambda m: EFFORT_RE.sub("", m))
    df = df.dropna(subset=["effort"])
    df["score"] = df["Best score (across scorers)"].map(_parse_score)
    df = df.dropna(subset=["score"])
    df["effort_rank"] = df["effort"].map(EFFORT_RANK)
    return df


def _draw_panel(ax, df: pd.DataFrame, task: str, color_map: dict) -> None:
    sub = df[df["task"] == task]
    plotted_any = False
    for model_base, color in color_map.items():
        fam = sub[sub["model_base"] == model_base].sort_values("effort_rank")
        if fam["effort"].nunique() < 3:
            continue
        plotted_any = True
        ax.plot(
            fam["effort_rank"], fam["score"],
            marker="o", color=color, linewidth=1.6, markersize=7,
            markeredgecolor="white", markeredgewidth=0.5,
            label=f"{model_base} (efforts: {','.join(fam['effort'])})",
        )

    used_ranks = sorted({EFFORT_RANK[e] for e in sub["effort"].unique()})
    ax.set_xticks(used_ranks)
    ax.set_xticklabels([EFFORT_ORDER[r] for r in used_ranks])
    ax.set_xlabel("Reasoning effort")
    ax.set_ylabel("Best score")
    ax.set_title(task, fontsize=11)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    if plotted_any:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "no models with ≥3 efforts",
                ha="center", va="center", transform=ax.transAxes,
                color="#888888", fontsize=10)


def main() -> None:
    df = _load()
    if df.empty:
        print("No rows.")
        return

    # Common colour palette across panels so the same model shows the same colour
    model_bases = sorted(df["model_base"].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {m: cmap(i % 10) for i, m in enumerate(model_bases)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, task in zip(axes, TASKS):
        _draw_panel(ax, df, task, color_map)
    fig.suptitle(
        "Reasoning-effort sweeps on FrontierMath 2025-02-28  "
        "(from data/eci/benchmarks_runs-PUBLIC VIEW.csv)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "frontiermath_effort_sweeps.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    out_csv = OUT_DIR / "frontiermath_effort_sweeps.csv"
    keep = df.groupby(["task", "model_base"])["effort"].transform("nunique") >= 3
    df[keep].sort_values(["task", "model_base", "effort_rank"])[[
        "task", "model_base", "effort", "id", "score",
    ]].to_csv(out_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    print()
    plotted = df[keep]
    print(
        plotted.groupby(["task", "model_base"])
        .agg(n_efforts=("effort", "nunique"),
             efforts=("effort", lambda s: ",".join(sorted(s.unique(), key=EFFORT_RANK.get))),
             score_min=("score", "min"), score_max=("score", "max"))
        .to_string(float_format=lambda v: f"{v:.4g}")
    )


if __name__ == "__main__":
    main()
