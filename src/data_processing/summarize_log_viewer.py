"""Summarize log_viewer_json runs into per-run token summaries.

For each folder under data/log_viewer_json/, reads header.json + summaries.json
and writes data/log_viewer_summary/<task>__<run_id>.json containing:
  - task / model / eval_id / run_id / dataset_name / epochs_configured
  - samples: per-sample raw per-epoch tokens (copied from summaries.json) and
    per-sample averages across epochs
  - totals: sum across samples of the per-sample averages

Note on file naming: the user asked for files named after the task, but multiple
runs share the same task (different models / epoch counts), so we append the
run_id to keep filenames unique while keeping the task name first.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "data" / "log_viewer_json"
OUT_DIR = ROOT / "data" / "log_viewer_summary"

TOKEN_FIELDS = ("output_tokens", "reasoning_tokens")


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "unknown"


def summarize_run(run_dir: Path) -> dict | None:
    header_path = run_dir / "header.json"
    summaries_path = run_dir / "summaries.json"
    if not (header_path.exists() and summaries_path.exists()):
        return None

    with header_path.open() as f:
        header = json.load(f)
    with summaries_path.open() as f:
        summaries = json.load(f)

    eval_block = header.get("eval", {})
    dataset = eval_block.get("dataset", {}) or {}
    primary_model = eval_block.get("model")

    # sample_id -> field -> list of per-epoch totals (only for the primary model)
    per_sample: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {f: [] for f in TOKEN_FIELDS}
    )

    for s in summaries:
        sid = s.get("id")
        if sid is None:
            continue
        usage = s.get("model_usage") or {}
        u = usage.get(primary_model) or {}
        for f in TOKEN_FIELDS:
            v = u.get(f)
            if v is not None:
                per_sample[sid][f].append(v)

    samples_out: dict[str, dict] = {}
    totals: dict[str, float | None] = {}
    for f in TOKEN_FIELDS:
        totals[f"total_{f}"] = 0.0
        totals[f"total_avg_{f}"] = 0.0
    totals_seen = {f: False for f in TOKEN_FIELDS}

    for sid, fields in per_sample.items():
        entry: dict = {"n_epochs": max((len(v) for v in fields.values()), default=0)}
        for f in TOKEN_FIELDS:
            vals = fields[f]
            entry[f] = vals  # raw per-epoch tokens, copied from summaries.json
            avg = (sum(vals) / len(vals)) if vals else None
            entry[f"avg_{f}"] = avg
            if vals:
                totals[f"total_{f}"] += sum(vals)
                totals[f"total_avg_{f}"] += avg
                totals_seen[f] = True
        samples_out[sid] = entry

    totals_source = "samples" if any(totals_seen.values()) else None

    # Fallback: if no per-sample usage data, use header's aggregate model_usage.
    if not any(totals_seen.values()):
        epochs_cfg = (eval_block.get("config") or {}).get("epochs") or 1
        # `stats` lives inside `eval` in newer logs but at the top level in
        # older ones — check both.
        stats_block = eval_block.get("stats") or header.get("stats") or {}
        agg_usage = stats_block.get("model_usage") or {}
        u = agg_usage.get(primary_model) or {}
        per_field_total = {f: 0 for f in TOKEN_FIELDS}
        any_seen = {f: False for f in TOKEN_FIELDS}
        for f in TOKEN_FIELDS:
            v = u.get(f)
            if v is not None:
                per_field_total[f] = v
                any_seen[f] = True
        if any(any_seen.values()):
            totals_source = "header_stats"
            for f in TOKEN_FIELDS:
                if any_seen[f]:
                    totals[f"total_{f}"] = per_field_total[f]
                    totals[f"total_avg_{f}"] = per_field_total[f] / epochs_cfg
                else:
                    totals[f"total_{f}"] = None
                    totals[f"total_avg_{f}"] = None
        else:
            for f in TOKEN_FIELDS:
                totals[f"total_{f}"] = None
                totals[f"total_avg_{f}"] = None

    # Pull score(s) from results.scores. Most runs have a single scorer with
    # an `accuracy` metric, but we keep all metrics from all scorers in case.
    scores_out: list[dict] = []
    primary_accuracy: float | None = None
    primary_stderr: float | None = None
    results_block = eval_block.get("results") or header.get("results") or {}
    for sc in (results_block.get("scores") or []):
        metrics = sc.get("metrics") or {}
        flat: dict[str, float | None] = {}
        for mname, m in metrics.items():
            flat[mname] = m.get("value") if isinstance(m, dict) else None
        scores_out.append(
            {
                "scorer": sc.get("scorer") or sc.get("name"),
                "reducer": sc.get("reducer"),
                "metrics": flat,
            }
        )
        if primary_accuracy is None and "accuracy" in flat:
            primary_accuracy = flat["accuracy"]
            primary_stderr = flat.get("stderr")

    return {
        "task": eval_block.get("task"),
        "task_display_name": eval_block.get("task_display_name"),
        "model": eval_block.get("model"),
        "eval_id": eval_block.get("eval_id"),
        "run_id": eval_block.get("run_id"),
        "dataset_name": dataset.get("name"),
        "epochs_configured": (eval_block.get("config") or {}).get("epochs"),
        "accuracy": primary_accuracy,
        "stderr": primary_stderr,
        "scores": scores_out,
        "totals": totals,
        "totals_source": totals_source,
        "samples": samples_out,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.json"):
        old.unlink()
    missing_path = OUT_DIR.parent / "log_viewer_summary_missing_tokens.json"
    if missing_path.exists():
        missing_path.unlink()

    folders = sorted(p for p in SRC_DIR.iterdir() if p.is_dir())
    written = 0
    skipped = 0
    used_names: dict[str, int] = {}
    missing: list[dict] = []
    for folder in folders:
        result = summarize_run(folder)
        if result is None:
            skipped += 1
            continue
        task = _sanitize(result.get("task") or "unknown_task")
        model = _sanitize(result.get("model") or "unknown_model")
        base = f"{task}__{model}"
        n = used_names.get(base, 0)
        used_names[base] = n + 1
        name = base if n == 0 else f"{base}__{n + 1}"
        out_path = OUT_DIR / f"{name}.json"
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)
        written += 1

        if result.get("totals_source") is None:
            missing.append(
                {
                    "run_id": result.get("run_id") or folder.name,
                    "task": result.get("task"),
                    "model": result.get("model"),
                    "eval_id": result.get("eval_id"),
                    "folder": str(folder.relative_to(ROOT)),
                }
            )

    with missing_path.open("w") as f:
        json.dump(missing, f, indent=2)
    print(f"Wrote {written} summaries to {OUT_DIR} (skipped {skipped})")
    print(f"Recorded {len(missing)} runs with no token data at {missing_path}")


if __name__ == "__main__":
    main()
