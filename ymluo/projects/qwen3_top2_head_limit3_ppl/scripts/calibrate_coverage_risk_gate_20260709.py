#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


KEY_FIELDS = ("benchmark", "task", "sample_id", "method")


def read_rows(output_dir: Path, *, require_coverage: bool) -> list[dict[str, str]]:
    path = output_dir / "task_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"score", "keep_fraction", "online_seconds"}
    if require_coverage:
        required.update({"ours_query_coverage_recall", "ours_query_coverage_terms"})
    missing = sorted(required - set(rows[0].keys())) if rows else sorted(required)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return rows


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in KEY_FIELDS)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered)) - 1)))
    return ordered[idx]


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def calibrate_task(
    base_rows: list[dict[str, str]],
    reference_by_key: dict[tuple[str, ...], dict[str, str]],
    target_recall: float,
    min_gain: float,
    min_terms: int,
) -> dict[str, Any]:
    paired = []
    for row in base_rows:
        ref = reference_by_key.get(row_key(row))
        if not ref:
            continue
        terms = float(row.get("ours_query_coverage_terms") or 0.0)
        if terms < min_terms:
            continue
        base_score = float(row["score"])
        ref_score = float(ref["score"])
        recall = float(row["ours_query_coverage_recall"])
        paired.append((row, ref, recall, ref_score - base_score))
    beneficial = [item for item in paired if item[3] > min_gain]
    if not paired or not beneficial:
        return {
            "samples": len(paired),
            "beneficial": len(beneficial),
            "threshold": None,
            "benefit_recall": 0.0,
            "trigger_rate": 0.0,
            "stitched_score": mean([float(row["score"]) for row, *_ in paired]),
            "stitched_keep": mean([float(row["keep_fraction"]) for row, *_ in paired]),
            "stitched_online": mean([float(row["online_seconds"]) for row, *_ in paired]),
        }

    # Trigger when coverage recall is below threshold. Pick the smallest threshold
    # that covers the target fraction of beneficial examples.
    threshold = quantile([item[2] for item in beneficial], target_recall)
    triggered = [item for item in paired if item[2] <= threshold]
    triggered_beneficial = [item for item in beneficial if item[2] <= threshold]
    stitched_score = []
    stitched_keep = []
    stitched_online = []
    for row, ref, recall, _gain in paired:
        use_ref = recall <= threshold
        chosen = ref if use_ref else row
        stitched_score.append(float(chosen["score"]))
        stitched_keep.append(float(chosen["keep_fraction"]))
        stitched_online.append(float(chosen["online_seconds"]))
    return {
        "samples": len(paired),
        "beneficial": len(beneficial),
        "threshold": threshold,
        "benefit_recall": len(triggered_beneficial) / max(1, len(beneficial)),
        "trigger_rate": len(triggered) / max(1, len(paired)),
        "stitched_score": mean(stitched_score),
        "stitched_keep": mean(stitched_keep),
        "stitched_online": mean(stitched_online),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--target_recall", type=float, default=0.80)
    parser.add_argument("--min_gain", type=float, default=0.01)
    parser.add_argument("--min_terms", type=int, default=3)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()

    base_rows = read_rows(args.base, require_coverage=True)
    reference_rows = read_rows(args.reference, require_coverage=False)
    reference_by_key = {row_key(row): row for row in reference_rows}
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in base_rows:
        by_task[row["task"]].append(row)

    summaries = []
    for task, rows in sorted(by_task.items()):
        result = calibrate_task(rows, reference_by_key, args.target_recall, args.min_gain, args.min_terms)
        result["task"] = task
        summaries.append(result)

    fieldnames = [
        "task",
        "samples",
        "beneficial",
        "threshold",
        "benefit_recall",
        "trigger_rate",
        "stitched_score",
        "stitched_keep",
        "stitched_online",
    ]
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)
    else:
        print(",".join(fieldnames))
        for row in summaries:
            print(",".join(str(row.get(field, "")) for field in fieldnames))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
