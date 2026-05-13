#!/usr/bin/env python3
"""Interactive CLI to add rows to manual_reports_data.csv."""

import csv
import os
import re
import sys

from thefuzz import fuzz, process

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "data")
ECI_PATH = os.path.join(DATA_DIR, "model_benchmark_scores.csv")
TOKENS_PATH = os.path.join(DATA_DIR, "manual_reports_data.csv")
EPOCH_MODELS_PATH = os.path.join(DATA_DIR, "epoch_all_ai_models.csv")
AA_STATS_PATH = os.path.join(DATA_DIR, "artificial_analysis_llm_stats.csv")

TOKENS_COLUMNS = [
    "model_id", "model", "benchmark_id", "benchmark",
    "x_value", "x_unit", "performance", "score_unit",
    "data_quality", "source_detail", "source_url", "human_verified",
]

X_UNIT_OPTIONS = ["thinking_tokens", "mean_response_tokens"]
DATA_QUALITY_OPTIONS = ["figure_estimate", "exact_text"]


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_eci():
    return load_csv(ECI_PATH)


def load_tokens():
    return load_csv(TOKENS_PATH)


def load_epoch_models():
    return load_csv(EPOCH_MODELS_PATH)


def load_aa_stats():
    return load_csv(AA_STATS_PATH)


def slugify(name):
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s



def fuzzy_pick(query, choices, label, id_label):
    """Fuzzy match query against choices list of (name, id). Returns (name, id)."""
    if not choices:
        return None

    names = [c[0] for c in choices]
    results = process.extract(query, names, scorer=fuzz.WRatio, limit=10)
    # Filter to score >= 50 (thefuzz returns (name, score) or (name, score, key))
    filtered = []
    for item in results:
        name, score = item[0], item[1]
        if score >= 50:
            filtered.append((name, score))
    results = filtered

    if not results:
        return None

    name_to_id = {c[0]: c[1] for c in choices}

    print(f"\nClosest matches for '{query}':")
    for i, (name, score) in enumerate(results, 1):
        print(f"  {i}: {name}  ({id_label}: {name_to_id[name]})  [match: {score}%]")
    print(f"  0: None of these — add as new {label}")

    while True:
        pick = input(f"\nPick a number [0-{len(results)}]: ").strip()
        if pick == "0":
            return None
        try:
            idx = int(pick)
            if 1 <= idx <= len(results):
                chosen_name = results[idx - 1][0]
                return (chosen_name, name_to_id[chosen_name])
        except ValueError:
            pass
        print("Invalid choice, try again.")


def ask_new_entry(label):
    """Ask user for a new name and generate slugified id."""
    while True:
        name = input(f"Enter the full {label} name: ").strip()
        if name:
            entry_id = slugify(name) + "_unknown"
            print(f"  → {label}_id will be: {entry_id}")
            return (name, entry_id)
        print(f"{label} name cannot be empty.")


def merged_unique_benchmarks(eci_rows, tokens_rows):
    """Merge unique (benchmark, benchmark_id) pairs from eci and tokens CSVs.

    ECI entries take priority for id when the same name appears in both.
    """
    seen = {}
    for r in eci_rows:
        v = r.get("benchmark", "").strip()
        if v and v not in seen:
            seen[v] = r.get("benchmark_id", "").strip()
    for r in tokens_rows:
        v = r.get("benchmark", "").strip()
        if v and v not in seen:
            seen[v] = r.get("benchmark_id", "").strip()
    return [(v, id_v) for v, id_v in seen.items()]


def merged_unique_models(eci_rows, tokens_rows, epoch_rows, aa_rows):
    """Merge unique (model, model_id) pairs from all sources.

    Priority for model_id:
    1. ECI (model_benchmark_scores) — uses model_id
    2. Tokens (manual_reports_data) — uses model_id
    3. Epoch (epoch_all_ai_models) — uses slugified name + _unknown
    4. Artificial Analysis — uses slug column
    """
    seen = {}
    # 1. ECI benchmarks (highest priority)
    for r in eci_rows:
        v = r.get("model", "").strip()
        if v and v not in seen:
            seen[v] = r.get("model_id", "").strip()
    # 2. Tokens file
    for r in tokens_rows:
        v = r.get("model", "").strip()
        if v and v not in seen:
            seen[v] = r.get("model_id", "").strip()
    # 3. Epoch all AI models (Model column)
    for r in epoch_rows:
        v = r.get("Model", "").strip()
        if v and v not in seen:
            seen[v] = slugify(v) + "_unknown"
    # 4. Artificial Analysis (name column, slug for id)
    for r in aa_rows:
        v = r.get("name", "").strip()
        if v and v not in seen:
            slug = r.get("slug", "").strip()
            seen[v] = slug if slug else slugify(v) + "_unknown"
    return [(v, id_v) for v, id_v in seen.items()]


def ask_model(eci_rows, tokens_rows, epoch_rows, aa_rows):
    print("\n--- Model ---")
    query = input("Type a model name (or part of it): ").strip()
    if not query:
        print("Cannot be empty.")
        return ask_model(eci_rows, tokens_rows, epoch_rows, aa_rows)

    choices = merged_unique_models(eci_rows, tokens_rows, epoch_rows, aa_rows)
    result = fuzzy_pick(query, choices, "model", "model_id")
    if result:
        print(f"  ✓ Selected: {result[0]} ({result[1]})")
        return result
    return ask_new_entry("model")


def ask_benchmark(eci_rows, tokens_rows):
    print("\n--- Benchmark ---")
    query = input("Type a benchmark name (or part of it): ").strip()
    if not query:
        print("Cannot be empty.")
        return ask_benchmark(eci_rows, tokens_rows)

    choices = merged_unique_benchmarks(eci_rows, tokens_rows)
    result = fuzzy_pick(query, choices, "benchmark", "benchmark_id")
    if result:
        print(f"  ✓ Selected: {result[0]} ({result[1]})")
        return result
    return ask_new_entry("benchmark")


def most_common(values):
    """Return the most common value from a list, or None if empty."""
    if not values:
        return None
    from collections import Counter
    return Counter(values).most_common(1)[0][0]


def ask_x_unit(existing_rows):
    print("\n--- x_unit ---")
    # Find the most common valid x_unit from existing rows
    existing_units = [
        r.get("x_unit", "").strip() for r in existing_rows
        if r.get("x_unit", "").strip() in X_UNIT_OPTIONS
    ]
    default = most_common(existing_units)
    if default:
        print(f"  Default (from existing rows): {default}")
    for i, opt in enumerate(X_UNIT_OPTIONS, 1):
        print(f"  {i}: {opt}")
    while True:
        if default:
            pick = input(f"Press Enter for '{default}', or pick [1-2]: ").strip()
            if pick == "":
                return default
        else:
            pick = input("Pick [1-2]: ").strip()
        try:
            idx = int(pick)
            if 1 <= idx <= len(X_UNIT_OPTIONS):
                return X_UNIT_OPTIONS[idx - 1]
        except ValueError:
            pass
        print("Invalid choice.")


def ask_x_value():
    print("\n--- x_value (token count) ---")
    while True:
        val = input("Enter the numeric token count: ").strip()
        try:
            n = float(val)
            if n == int(n):
                return str(int(n))
            return val
        except ValueError:
            print("Must be a number.")


def ask_performance():
    print("\n--- performance ---")
    while True:
        val = input("Enter the performance value (0 to 1): ").strip()
        try:
            n = float(val)
            if 0 <= n <= 1:
                return val
            print("Must be between 0 and 1 (inclusive).")
        except ValueError:
            print("Must be a number.")


def get_existing_for_model(tokens_rows, model_name, model_id):
    """Get rows from tokens file matching this model by name or id."""
    return [
        r for r in tokens_rows
        if r.get("model", "").strip() == model_name
        or r.get("model_id", "").strip() == model_id
    ]


def ask_data_quality(existing_rows):
    print("\n--- data_quality ---")
    # Find most common valid data_quality from existing rows
    valid_dq = [
        r.get("data_quality", "").strip() for r in existing_rows
        if r.get("data_quality", "").strip() in DATA_QUALITY_OPTIONS
    ]
    default = most_common(valid_dq)
    if default:
        print(f"  Default (from existing rows): {default}")
        for i, opt in enumerate(DATA_QUALITY_OPTIONS, 1):
            print(f"  {i}: {opt}")
        while True:
            pick = input(f"Press Enter for '{default}', or pick [1-2]: ").strip()
            if pick == "":
                return default
            try:
                idx = int(pick)
                if 1 <= idx <= len(DATA_QUALITY_OPTIONS):
                    return DATA_QUALITY_OPTIONS[idx - 1]
            except ValueError:
                pass
            print("Invalid choice.")
    else:
        for i, opt in enumerate(DATA_QUALITY_OPTIONS, 1):
            print(f"  {i}: {opt}")
        while True:
            pick = input("Pick [1-2]: ").strip()
            try:
                idx = int(pick)
                if 1 <= idx <= len(DATA_QUALITY_OPTIONS):
                    return DATA_QUALITY_OPTIONS[idx - 1]
            except ValueError:
                pass
            print("Invalid choice.")


def ask_with_suggestions(field_name, existing_rows, required=False):
    """Ask for a text field, showing existing values as suggestions."""
    print(f"\n--- {field_name} ---")
    all_vals = [
        r.get(field_name, "").strip() for r in existing_rows
        if r.get(field_name, "").strip()
    ]
    existing_vals = list(dict.fromkeys(all_vals))  # unique, preserving order
    default = most_common(all_vals)

    if default and len(existing_vals) == 1:
        # Single existing value — offer as default
        print(f"  Default (from existing rows): {default}")
        print(f"  0: Type a new value")
        while True:
            pick = input(f"Press Enter for default, or pick 0: ").strip()
            if pick == "":
                return default
            if pick == "0":
                break  # Fall through to manual entry
            # They typed a raw value directly
            return pick

    elif existing_vals:
        print(f"  Most common (from existing rows): {default}")
        print(f"  Other values:")
        for i, val in enumerate(existing_vals, 1):
            print(f"  {i}: {val}")
        print(f"  0: Type a new value")

        while True:
            pick = input(f"Press Enter for most common, pick a number, or type a new value: ").strip()
            if pick == "":
                return default
            try:
                idx = int(pick)
                if idx == 0:
                    break  # Fall through to manual entry
                if 1 <= idx <= len(existing_vals):
                    return existing_vals[idx - 1]
            except ValueError:
                # They typed a raw value
                if pick:
                    return pick
                if required:
                    print("This field is required.")
                    continue
                return ""
            print("Invalid choice.")

    while True:
        val = input(f"Enter {field_name}: ").strip()
        if val:
            return val
        if required:
            print("This field is required.")
        else:
            return ""


def show_summary(row):
    print("\n" + "=" * 60)
    print("  ROW SUMMARY")
    print("=" * 60)
    for col in TOKENS_COLUMNS:
        print(f"  {col:20s}: {row[col]}")
    print("=" * 60)


def append_row(row):
    file_exists = os.path.exists(TOKENS_PATH)
    with open(TOKENS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TOKENS_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def collect_one_row(eci_rows, epoch_rows, aa_rows):
    """Collect all fields for one row. Returns the row dict or None if cancelled."""
    # Reload tokens file each time
    tokens_rows = load_tokens()

    model_name, model_id = ask_model(eci_rows, tokens_rows, epoch_rows, aa_rows)
    benchmark_name, benchmark_id = ask_benchmark(eci_rows, tokens_rows)

    existing_rows = get_existing_for_model(tokens_rows, model_name, model_id)

    x_unit = ask_x_unit(existing_rows)
    x_value = ask_x_value()
    performance = ask_performance()

    data_quality = ask_data_quality(existing_rows)
    source_detail = ask_with_suggestions("source_detail", existing_rows)
    source_url = ask_with_suggestions("source_url", existing_rows, required=True)

    row = {
        "model_id": model_id,
        "model": model_name,
        "benchmark_id": benchmark_id,
        "benchmark": benchmark_name,
        "x_value": x_value,
        "x_unit": x_unit,
        "performance": performance,
        "score_unit": "accuracy",
        "data_quality": data_quality,
        "source_detail": source_detail,
        "source_url": source_url,
        "human_verified": "FALSE",
    }

    EDITABLE_FIELDS = [
        "model", "benchmark", "x_unit", "x_value",
        "performance", "data_quality", "source_detail", "source_url",
    ]

    while True:
        show_summary(row)

        confirm = input("\nPress Enter to confirm, or type [e]dit / [c]ancel: ").strip().lower()
        if confirm == "":
            row["human_verified"] = "yes"
            append_row(row)
            print("\n  ✓ Row appended to manual_reports_data.csv")
            return row
        elif confirm in ("c", "cancel"):
            print("  Row discarded.")
            return None
        elif confirm in ("e", "edit"):
            print("\n  Which field to edit?")
            for i, field in enumerate(EDITABLE_FIELDS, 1):
                print(f"  {i}: {field} = {row[field]}")
            while True:
                pick = input(f"Pick [1-{len(EDITABLE_FIELDS)}]: ").strip()
                try:
                    idx = int(pick)
                    if 1 <= idx <= len(EDITABLE_FIELDS):
                        break
                except ValueError:
                    pass
                print("Invalid choice.")
            field = EDITABLE_FIELDS[idx - 1]
            existing_rows = get_existing_for_model(tokens_rows, row["model"], row["model_id"])
            if field == "model":
                name, mid = ask_model(eci_rows, tokens_rows, epoch_rows, aa_rows)
                row["model"] = name
                row["model_id"] = mid
            elif field == "benchmark":
                name, bid = ask_benchmark(eci_rows, tokens_rows)
                row["benchmark"] = name
                row["benchmark_id"] = bid
            elif field == "x_unit":
                row["x_unit"] = ask_x_unit(existing_rows)
            elif field == "x_value":
                row["x_value"] = ask_x_value()
            elif field == "performance":
                row["performance"] = ask_performance()
            elif field == "data_quality":
                row["data_quality"] = ask_data_quality(existing_rows)
            elif field == "source_detail":
                row["source_detail"] = ask_with_suggestions("source_detail", existing_rows)
            elif field == "source_url":
                row["source_url"] = ask_with_suggestions("source_url", existing_rows, required=True)
        else:
            print("  Please press Enter, or type e or c.")


def main():
    if not os.path.exists(ECI_PATH):
        print(f"Error: Reference file not found: {ECI_PATH}")
        sys.exit(1)

    print("=" * 60)
    print("  Inference Scaling Tokens — Row Entry Tool")
    print("=" * 60)

    eci_rows = load_eci()
    epoch_rows = load_epoch_models()
    aa_rows = load_aa_stats()
    print(f"Loaded {len(eci_rows)} rows from model_benchmark_scores.csv")
    print(f"Loaded {len(epoch_rows)} rows from epoch_all_ai_models.csv")
    print(f"Loaded {len(aa_rows)} rows from artificial_analysis_llm_stats.csv")

    while True:
        collect_one_row(eci_rows, epoch_rows, aa_rows)
        again = input("\nPress Enter to add another row, or type 'q' to quit: ").strip().lower()
        if again in ("q", "quit", "n", "no"):
            print("Done. Goodbye!")
            break


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)
