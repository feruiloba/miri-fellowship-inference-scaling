"""
Estimate ECI from Artificial Analysis per-benchmark evaluations
================================================================

For each (model, benchmark) score in data/artificial_analysis/aa_evaluations_combined.csv
whose benchmark maps to an eci-public benchmark, invert the IRT model to produce
a per-benchmark ECI estimate, and append rows to data/derived_eci/eci_from_benchmarks.csv.

Re-uses the IRT inversion + caching machinery from eci_from_log_viewer.py.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "data_processing" / "derive_eci"))
sys.path.insert(0, str(ROOT / "eci-public" / "src"))

from eci_from_log_viewer import (  # noqa: E402
    OUT_CSV,
    OUT_DIR,
    estimate_eci,
    get_eci_params,
    update_output_csv,
)


AA_EVAL_FILE = ROOT / "data" / "artificial_analysis" / "aa_evaluations_combined.csv"
AA_STATS_FILE = ROOT / "data" / "artificial_analysis" / "artificial_analysis_llm_stats.csv"


# Map AA benchmark slug → eci-public benchmark name. Only mappings where the
# evaluation is the same (or very nearly so) are listed; ambiguous matches like
# math-500 ↔ "MATH level 5" are intentionally omitted.
AA_TO_ECI_BENCH = {
    "gpqa-diamond": "GPQA diamond",
    "humanitys-last-exam": "HLE",
    "apex-agents-aa": "APEX-Agents",
    "terminalbench-hard": "Terminal Bench",
    "aime-2025": "OTIS Mock AIME 2024-2025",
}


def load_company_lookup():
    """Map model_slug → company_slug from the llm_stats CSV."""
    stats = pd.read_csv(AA_STATS_FILE, usecols=["slug", "company", "company_slug"])
    stats = stats.dropna(subset=["slug"]).drop_duplicates(subset=["slug"])
    return stats.set_index("slug")[["company", "company_slug"]]


def collect_aa_rows():
    df = pd.read_csv(AA_EVAL_FILE)
    df = df[df["benchmark"].isin(AA_TO_ECI_BENCH)].copy()
    df = df.dropna(subset=["score_raw"])
    df["eci_benchmark"] = df["benchmark"].map(AA_TO_ECI_BENCH)
    df["accuracy"] = df["score_raw"].astype(float)

    # tokens — null when AA didn't measure
    for col in ("reasoning_tokens", "answer_tokens", "total_output_tokens"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    bench_df, a, b = get_eci_params()

    raw = collect_aa_rows()
    print(
        f"Collected {len(raw)} AA evaluation rows across "
        f"{raw['benchmark'].nunique()} benchmarks"
    )

    scored = estimate_eci(raw, bench_df, a, b)

    companies = load_company_lookup()
    company = scored["model_slug"].map(companies["company"]).fillna("")

    out_rows = pd.DataFrame({
        "model": scored["model"],
        "slug": scored["model_slug"].fillna(""),
        "company": company,
        "benchmark_source": "aa_" + scored["benchmark"].str.replace("-", "_"),
        "benchmark_score": scored["accuracy"],
        "eci_estimated": scored["eci_estimated"],
        "reasoning_tokens": scored["reasoning_tokens"],
        "answer_tokens": scored["answer_tokens"],
        "total_inference_tokens": scored["total_output_tokens"],
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full = update_output_csv(out_rows)
    print(f"Wrote {OUT_CSV} ({len(full)} total rows; {len(out_rows)} new from AA evaluations)")
    print(out_rows.head(10).to_string(index=False))
