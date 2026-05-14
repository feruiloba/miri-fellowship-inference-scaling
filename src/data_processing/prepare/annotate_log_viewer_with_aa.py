"""Match unique log-viewer model IDs to entries in the AA stats CSV.

For every distinct `model` value in data/eci/log_viewer_summary/*.json, ranks rows
from data/artificial_analysis/artificial_analysis_llm_stats.csv by similarity
(name / slug / company / company_slug / model_id substrings + release-date
hint) and lets the user pick the best match. The chosen AA row's name,
company_slug, and release_date are written back into every summary JSON that
references that model.

Usage:
    python src/data_processing/match_models_to_aa.py
    python src/data_processing/match_models_to_aa.py --top 15
    python src/data_processing/match_models_to_aa.py --redo   # revisit already-matched models
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUMMARY_DIR = ROOT / "data" / "eci" / "log_viewer_summary"
AA_CSV = ROOT / "data" / "artificial_analysis" / "artificial_analysis_llm_stats.csv"

WRITE_FIELDS = ("aa_model_id", "aa_model_slug", "aa_name", "company_slug", "release_date")


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if t}


def _score(model_id: str, row: dict) -> float:
    """Higher = better match. Combines token overlap and substring similarity."""
    mt = _tokens(model_id)
    if not mt:
        return 0.0
    haystack_parts = [row.get("name", ""), row.get("slug", ""),
                      row.get("company", ""), row.get("company_slug", ""),
                      row.get("model_id", "")]
    rt = set().union(*(_tokens(p) for p in haystack_parts))
    overlap = len(mt & rt) / max(len(mt), 1)

    # also reward substring matches (e.g. "gpt-5.2" → "gpt-5-2")
    norm_model = re.sub(r"[^a-z0-9]+", "", model_id.lower())
    norm_hay = re.sub(r"[^a-z0-9]+", "", " ".join(haystack_parts).lower())
    seq = SequenceMatcher(None, norm_model, norm_hay).ratio() if norm_hay else 0.0

    return overlap * 2.0 + seq


def collect_models() -> dict[str, list[Path]]:
    """Return {model_id: [summary file paths]}."""
    out: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(SUMMARY_DIR.glob("*.json")):
        d = json.load(p.open())
        m = d.get("model")
        if m:
            out[m].append(p)
    return out


def load_aa() -> list[dict]:
    with AA_CSV.open() as f:
        return list(csv.DictReader(f))


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except ValueError:
            return None


def date_from_model_id(model_id: str) -> date | None:
    """Extract a release-date hint from the model id, if one is embedded.

    Handles patterns like '...-2025-01-25', '...-20250805', '...-2025-08'.
    """
    m = re.search(r"(\d{4})-?(\d{2})-?(\d{2})", model_id)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = re.search(r"(\d{4})-(\d{2})(?!\d)", model_id)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            pass
    return None


def already_matched(paths: list[Path]) -> bool:
    for p in paths:
        d = json.load(p.open())
        if not all(d.get(f) for f in WRITE_FIELDS):
            return False
    return True


def write_match(paths: list[Path], aa_row: dict) -> None:
    aa_model_id = aa_row.get("model_id", "")
    aa_model_slug = aa_row.get("slug", "")
    name = aa_row.get("name", "")
    company_slug = aa_row.get("company_slug", "")
    release_date = aa_row.get("release_date", "")
    for p in paths:
        d = json.load(p.open())
        d["aa_model_id"] = aa_model_id
        d["aa_model_slug"] = aa_model_slug
        d["aa_name"] = name
        d["company_slug"] = company_slug
        d["release_date"] = release_date
        with p.open("w") as f:
            json.dump(d, f, indent=2)


def clear_match(paths: list[Path]) -> None:
    for p in paths:
        d = json.load(p.open())
        changed = False
        for f in WRITE_FIELDS:
            if f in d:
                del d[f]
                changed = True
        if changed:
            with p.open("w") as fh:
                json.dump(d, fh, indent=2)


def _sort_by_date(rows: list[dict], target: date | None) -> list[dict]:
    if target is None:
        return rows
    def key(r):
        d = _parse_date(r.get("release_date"))
        return (0, abs((d - target).days)) if d else (1, 0)
    return sorted(rows, key=key)


def prompt_match(model_id: str, paths: list[Path], aa_rows: list[dict], top: int) -> str:
    """Return 'matched' | 'skipped' | 'quit'."""
    hint_date = date_from_model_id(model_id)
    scored = sorted(((_score(model_id, r), r) for r in aa_rows), key=lambda t: t[0], reverse=True)
    candidates = [r for s, r in scored[:top] if s > 0]
    candidates = _sort_by_date(candidates, hint_date)

    print("\n" + "=" * 78)
    print(f"MODEL: {model_id}")
    print(f"  appears in {len(paths)} summary file(s)")
    if hint_date:
        print(f"  date hint from model id: {hint_date.isoformat()} (sorted by closeness)")
    print("=" * 78)

    def _print_table(rows: list[dict]) -> None:
        print(f"  {'#':>3}  {'name':<50} {'company':<14} {'release':<12} {'Δd':>6}  slug")
        print(f"  {'-'*3}  {'-'*50} {'-'*14} {'-'*12} {'-'*6}  {'-'*30}")
        for i, r in enumerate(rows, 1):
            d = _parse_date(r.get("release_date"))
            delta = f"{abs((d - hint_date).days)}d" if (d and hint_date) else ""
            print(f"  {i:>3}  {(r.get('name') or '')[:50]:<50} "
                  f"{(r.get('company') or '')[:14]:<14} "
                  f"{(r.get('release_date') or '')[:12]:<12} "
                  f"{delta:>6}  "
                  f"{r.get('slug') or ''}")

    if not candidates:
        print("  no AA candidates with positive score.")
    else:
        _print_table(candidates)

    print("\n  enter number to match, 's' to skip, 'q' to quit, "
          "or a search string to filter further")
    while True:
        choice = input("  > ").strip()
        if not choice:
            continue
        if choice.lower() == "q":
            return "quit"
        if choice.lower() == "s":
            return "skipped"
        if choice.isdigit():
            i = int(choice)
            if 1 <= i <= len(candidates):
                pick = candidates[i - 1]
                print(f"  → matched to: {pick.get('name')}  "
                      f"(company_slug={pick.get('company_slug')}, "
                      f"release_date={pick.get('release_date')})")
                write_match(paths, pick)
                return "matched"
            print(f"  invalid index (1..{len(candidates)})")
            continue
        # treat as search filter
        ql = choice.lower()
        filtered = [
            r for r in aa_rows
            if ql in (r.get("name", "") + " " + r.get("slug", "") + " "
                      + r.get("company", "") + " " + r.get("company_slug", "")
                      + " " + r.get("model_id", "")).lower()
        ]
        if not filtered:
            print("  no matches for that filter.")
            continue
        candidates = _sort_by_date(filtered[:top], hint_date)
        print(f"  filtered to {len(candidates)} row(s):")
        _print_table(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10, help="number of candidates to show")
    parser.add_argument("--redo", action="store_true",
                        help="revisit models that already have aa_name/company_slug/release_date set")
    args = parser.parse_args()

    aa_rows = load_aa()
    models = collect_models()
    print(f"{len(models)} unique model ids across {sum(len(v) for v in models.values())} summary files")

    pending: list[tuple[str, list[Path]]] = []
    for model_id, paths in sorted(models.items()):
        if already_matched(paths) and not args.redo:
            continue
        pending.append((model_id, paths))

    # Auto-match pass: if the last path segment of model_id equals an AA slug
    # (treating '.' and '-' as equivalent), match without prompting.
    def _slug_key(s: str) -> str:
        return re.sub(r"[.\-]+", "-", (s or "").lower())

    by_slug = {_slug_key(r.get("slug")): r for r in aa_rows if r.get("slug")}
    auto_matched: list[tuple[str, list[Path]]] = []
    still_pending: list[tuple[str, list[Path]]] = []
    for model_id, paths in pending:
        tail = _slug_key(model_id.split("/")[-1])
        row = by_slug.get(tail)
        if row:
            if args.redo:
                clear_match(paths)
            write_match(paths, row)
            auto_matched.append((model_id, paths))
            print(f"  auto-matched {model_id} → {row.get('name')} ({row.get('release_date')})")
        else:
            still_pending.append((model_id, paths))
    print(f"auto-matched {len(auto_matched)} model(s) by exact slug equality")
    pending = still_pending

    print(f"{len(pending)} model(s) to process "
          f"({'including already-matched' if args.redo else 'skipping already-matched'})")

    matched = skipped = 0
    for i, (model_id, paths) in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}]")
        if args.redo:
            clear_match(paths)
        result = prompt_match(model_id, paths, aa_rows, args.top)
        if result == "quit":
            print("\nquitting; previous matches are saved.")
            break
        if result == "matched":
            matched += 1
        else:
            skipped += 1

    print(f"\ndone. matched={matched}  skipped={skipped}")


if __name__ == "__main__":
    main()
