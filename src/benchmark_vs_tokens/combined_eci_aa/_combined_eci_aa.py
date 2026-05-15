"""
Shared loader for the AA + ECI log_viewer combined dataset.

The combined CSV is produced by
src/data_processing/prepare/combine_aa_and_log_viewer.py and lives at
data/combined_eci_aa/aa_and_log_viewer.csv.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
COMBINED_CSV = ROOT / "data" / "combined_eci_aa" / "aa_and_log_viewer.csv"
OUT_DIR = ROOT / "output" / "benchmark_vs_tokens" / "combined_eci_aa"

BENCHMARKS = ["GPQA Diamond", "AIME"]


def load_combined(benchmark: str) -> pd.DataFrame:
    df = pd.read_csv(COMBINED_CSV)
    df = df[df["canonical_benchmark"] == benchmark].copy()
    df = df.dropna(subset=["score_raw", "total_inference_tokens"])
    df = df[df["total_inference_tokens"] > 0]
    return df
