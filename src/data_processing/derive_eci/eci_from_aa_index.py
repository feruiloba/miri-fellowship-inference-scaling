"""
Estimate ECI from AA Index
==========================

Method 1: Convert AA Index → ECI using the linear fit from
get_aa_index_to_eci_fit_params.py:

    AA = 1.0515 · ECI - 117.8305     =>     ECI = (AA + 117.8305) / 1.0515

Joins AA stats with output-token counts and writes a CSV with the
estimated ECI alongside reasoning / answer / total output tokens.

Data sources:
- data/artificial_analysis/artificial_analysis_llm_stats.csv
- data/artificial_analysis/aa_output_tokens.csv
"""

import os
import re

import pandas as pd


AA_FILE = "data/artificial_analysis/artificial_analysis_llm_stats.csv"
TOKENS_FILE = "data/artificial_analysis/aa_output_tokens.csv"
FIT_FILE = "output/capability_indices/aa_index_vs_eci_fit.csv"
OUT_DIR = "data/derived_eci"


def load_fit(path: str = FIT_FILE) -> tuple[float, float]:
    """Load (slope, intercept) for AA = slope * ECI + intercept from the fit CSV
    written by get_aa_index_to_eci_fit_params.py — keeps this script in sync with the latest fit."""
    fit = pd.read_csv(path)
    return float(fit["slope"].iloc[0]), float(fit["intercept"].iloc[0])


SLOPE, INTERCEPT = load_fit()


def _norm(s):
    return re.sub(r"[\s\-_]+", "", str(s)).lower()


def aa_to_eci(aa_index):
    return (aa_index - INTERCEPT) / SLOPE


def build_table():
    aa = pd.read_csv(AA_FILE)
    tok = pd.read_csv(TOKENS_FILE)

    aa["aa_index"] = pd.to_numeric(aa["artificial_analysis_index"], errors="coerce")
    tok["reasoning_tokens"] = pd.to_numeric(tok["reasoning_tokens"], errors="coerce")
    tok["answer_tokens"] = pd.to_numeric(tok["answer_tokens"], errors="coerce")

    aa["_k"] = aa["name"].apply(_norm)
    tok["_k"] = tok["model"].apply(_norm)

    merged = aa.merge(
        tok[["_k", "model", "reasoning_tokens", "answer_tokens"]],
        on="_k", how="inner",
    )
    merged = merged[merged["aa_index"].notna()].copy()

    merged["total_output_tokens"] = (
        merged["reasoning_tokens"].fillna(0) + merged["answer_tokens"].fillna(0)
    )
    merged["benchmark_source"] = "aa_index"
    merged["benchmark_score"] = merged["aa_index"]
    merged["eci_estimated"] = aa_to_eci(merged["aa_index"])

    out = merged[[
        "name", "slug", "company",
        "benchmark_source", "benchmark_score", "eci_estimated",
        "reasoning_tokens", "answer_tokens", "total_output_tokens",
    ]].rename(columns={"name": "model"}).sort_values("eci_estimated", ascending=False)

    return out


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build_table()
    out_path = f"{OUT_DIR}/eci_from_benchmarks.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")
    print(df.head(10).to_string(index=False))
