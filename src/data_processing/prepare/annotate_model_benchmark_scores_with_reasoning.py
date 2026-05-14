"""
Add is_reasoning and reasoning_level columns to data/eci/model_benchmark_scores.csv.

Detection logic:
1. Suffix-based: model_version ends with _low, _medium, _high, _xhigh, _max,
   _unknown, _NNK (thinking budget), or _thinking.
2. Parenthetical: model_version contains "(NNK thinking)".
3. Name-based: known reasoning model families without suffixes, e.g.
   DeepSeek-R1, o1-preview, deepseek-reasoner, *-Thinking-*.

reasoning_level is extracted from the suffix when present, or set to
"unknown" for models identified by name only.
"""

import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
INPUT_FILE = DATA_DIR / "eci" / "model_benchmark_scores.csv"
OUTPUT_FILE = INPUT_FILE  # overwrite in place

# Suffix pattern: _low, _medium, _high, _xhigh, _max, _unknown, _NNK, _thinking
SUFFIX_PATTERN = re.compile(
    r"_(low|medium|high|xhigh|max|unknown|thinking|\d+K)$", re.IGNORECASE
)

# Parenthetical pattern: "(16K thinking)", "(24K thinking)"
PAREN_PATTERN = re.compile(
    r"\((\d+K)\s+thinking\)$", re.IGNORECASE
)

# Known reasoning model families (matched against model_version, case-insensitive)
# These are models that reason but don't use suffixes
KNOWN_REASONING_PATTERNS = [
    r"DeepSeek-R1",
    r"deepseek-reasoner",
    r"o1-preview",
    r"o1-mini",
    r"Thinking",        # catches Qwen3-*-Thinking-*, kimi-k2-thinking, etc.
    r"thinking-exp",    # gemini-2.0-flash-thinking-exp
]
KNOWN_REASONING_RE = re.compile(
    "|".join(KNOWN_REASONING_PATTERNS), re.IGNORECASE
)


def classify_reasoning(model_version: str) -> tuple[bool, str | None]:
    """
    Returns (is_reasoning, reasoning_level) for a model_version string.
    """
    if not isinstance(model_version, str) or model_version.strip() == "":
        return False, None

    # Check suffix pattern
    suffix_match = SUFFIX_PATTERN.search(model_version)
    if suffix_match:
        return True, suffix_match.group(1).lower()

    # Check parenthetical pattern: "(16K thinking)"
    paren_match = PAREN_PATTERN.search(model_version)
    if paren_match:
        return True, paren_match.group(1)

    # Check known reasoning model names
    if KNOWN_REASONING_RE.search(model_version):
        return True, "unknown"

    return False, None


def main():
    print(f"Reading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    results = df["model_version"].apply(classify_reasoning)
    df["is_reasoning"] = results.apply(lambda x: x[0])
    df["reasoning_level"] = results.apply(lambda x: x[1])

    reasoning_count = df["is_reasoning"].sum()
    total = len(df)
    print(f"Reasoning rows: {reasoning_count}/{total}")
    print(f"Reasoning level distribution:")
    print(df[df["is_reasoning"]]["reasoning_level"].value_counts().to_string())

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWritten to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
