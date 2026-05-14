"""
Group artificial-analysis evaluation rows into model families.

A model family is the set of rows in a single eval CSV that share a common
base model but differ in reasoning-effort / thinking-mode configuration. For
example, on aime-2025:

    GPT-5 nano (minimal), GPT-5 nano (medium), GPT-5 nano (high)
    Claude 4 Sonnet, Claude 4 Sonnet (Thinking)

belong to the same family because they spend different amounts of
reasoning/answer tokens on the same benchmark.

Family detection works on the model slug: strip the trailing effort suffix
(`-low`, `-medium`, `-high`, `-xhigh`, `-minimal`, `-non-reasoning`,
`-reasoning`, `-thinking`, `-adaptive`) and group rows that resolve to the
same base. Rows without a slug fall back to a whitespace-collapsed lowercase
of the display name as the family key — this also catches the slug-duplicate
case (same slug appearing twice with different token counts under different
display-name spellings).

Only families with ≥2 variants are kept.

Usage:
    python src/data_processing/prepare/find_aa_evaluation_families.py
"""

import re
from pathlib import Path

import pandas as pd

EVAL_DIR = Path("data/artificial_analysis/evaluations")
OUTPUT_DIR = Path("data/artificial_analysis/evaluation_families")
COMBINED_PATH = Path("data/artificial_analysis/aa_evaluation_families_combined.csv")

EFFORT_SUFFIXES = (
    "low", "medium", "high", "xhigh", "minimal",
    "non-reasoning", "reasoning", "thinking", "adaptive",
)
_EFFORT_RE = re.compile(rf"-({'|'.join(EFFORT_SUFFIXES)})$")


def family_id(model_slug: str | float, model: str) -> str:
    """Family key: slug with effort suffix stripped, or normalized display name."""
    if isinstance(model_slug, str) and model_slug:
        return _EFFORT_RE.sub("", model_slug)
    # Fallback: lowercase, collapse whitespace, drop trailing parenthetical
    name = re.sub(r"\s*\([^)]*\)\s*$", "", str(model)).strip().lower()
    return re.sub(r"\s+", " ", name)


def build_families(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate `df` with family_id + variant_count, keep families with ≥2 rows."""
    df = df.copy()
    df["family_id"] = [
        family_id(s, m) for s, m in zip(df.get("model_slug"), df["model"])
    ]
    counts = df.groupby("family_id").size().rename("variant_count")
    df = df.merge(counts, left_on="family_id", right_index=True)
    df = df[df["variant_count"] >= 2].copy()
    # Reorder columns: benchmark, family_id, variant_count, model, model_slug, …rest
    front = ["benchmark", "family_id", "variant_count", "model", "model_slug"]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]
    return df.sort_values(["family_id", "total_output_tokens"], kind="stable")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csvs = sorted(EVAL_DIR.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSVs in {EVAL_DIR}/ — run the scraper first.")

    combined = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        if "benchmark" not in df.columns:
            df.insert(0, "benchmark", csv_path.stem)
        fam = build_families(df)
        if fam.empty:
            print(f"[{csv_path.stem}] no families with ≥2 variants")
            continue
        out_path = OUTPUT_DIR / csv_path.name
        fam.to_csv(out_path, index=False)
        n_families = fam["family_id"].nunique()
        print(f"[{csv_path.stem}] {n_families} families, {len(fam)} variant rows → {out_path}")
        combined.append(fam)

    if combined:
        all_fam = pd.concat(combined, ignore_index=True)
        all_fam.to_csv(COMBINED_PATH, index=False)
        print(f"\nCombined → {COMBINED_PATH}  ({len(all_fam)} rows, "
              f"{all_fam.groupby(['benchmark', 'family_id']).ngroups} families)")


if __name__ == "__main__":
    main()
