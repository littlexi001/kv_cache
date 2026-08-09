from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METHOD = "qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64"
TASKS = ("lcc", "multifieldqa_en", "qmsum")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_run_root", type=Path, required=True)
    parser.add_argument("--ab_run_root", type=Path, required=True)
    parser.add_argument("--expected_pairs", type=int, default=15)
    parser.add_argument(
        "--ablation",
        choices=("disable_sketch", "tail_alpha0"),
        default="disable_sketch",
    )
    return parser.parse_args()


def read_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("shard[0-9]*/sample_results.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def keyed(
    rows: list[dict[str, str]], method: str
) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("method") != method or row.get("task") not in TASKS:
            continue
        key = (str(row["task"]), str(row["sample_id"]))
        if key in output:
            raise ValueError(f"duplicate row for {key} and method {method}")
        output[key] = row
    return output


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def score_by_task(
    rows: dict[tuple[str, str], dict[str, str]]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for (task, _), row in rows.items():
        grouped[task].append(float(row["score"]))
    return {task: mean(grouped[task]) for task in TASKS}


def macro(scores: dict[str, float]) -> float:
    return mean([scores[task] for task in TASKS])


def summarize(
    reference_rows: list[dict[str, str]],
    ab_rows: list[dict[str, str]],
    expected_pairs: int,
    ablation: str = "disable_sketch",
) -> dict[str, Any]:
    full = keyed(reference_rows, "full_kv")
    current = keyed(reference_rows, METHOD)
    no_value = keyed(ab_rows, METHOD)
    keys = set(full) & set(current) & set(no_value)
    if len(keys) != expected_pairs:
        raise ValueError(
            f"expected {expected_pairs} strict triples, found {len(keys)}; "
            f"full={len(full)} current={len(current)} no_value={len(no_value)}"
        )
    if {task for task, _ in keys} != set(TASKS):
        raise ValueError("strict triples do not cover all three weak tasks")
    for key in keys:
        row = no_value[key]
        if row.get("executed_path") != METHOD:
            raise ValueError(f"unexpected executed_path for {key}")
        debug_disabled = float(
            row.get("packed_qmse_debug_value_sketch_disabled", 0)
        )
        rank = float(row.get("packed_qmse_value_sketch_rank", 0))
        bits = float(row.get("packed_qmse_value_sketch_bits", 0))
        if ablation == "disable_sketch":
            if debug_disabled < 0.99:
                raise ValueError(
                    f"ValueSketch debug switch was not active for {key}"
                )
            if abs(rank) > 1e-6 or abs(bits) > 1e-6:
                raise ValueError(f"ValueSketch was not disabled for {key}")
        elif ablation == "tail_alpha0":
            if debug_disabled > 1e-6:
                raise ValueError(f"debug disable unexpectedly active for {key}")
            if abs(rank - 16.0) > 1e-6 or abs(bits - 4.0) > 1e-6:
                raise ValueError(f"rank-16 INT4 sketch was not preserved for {key}")
            if abs(
                float(row.get("packed_qmse_value_sketch_tail_alpha", -1.0))
            ) > 1e-6:
                raise ValueError(f"tail alpha was not zero for {key}")
        if float(row.get("sampled_candidate_overflow_fraction", 0)) > 1e-8:
            raise ValueError(f"candidate compaction overflowed for {key}")

    conditions = {
        "full_kv": {key: full[key] for key in keys},
        "frozen_current": {key: current[key] for key in keys},
        "no_value_sketch": {key: no_value[key] for key in keys},
    }
    task_scores = {
        name: score_by_task(rows) for name, rows in conditions.items()
    }
    macro_scores = {name: macro(scores) for name, scores in task_scores.items()}
    current_vs_full = macro_scores["frozen_current"] / macro_scores["full_kv"]
    no_value_vs_full = macro_scores["no_value_sketch"] / macro_scores["full_kv"]
    prediction_changes = sum(
        current[key].get("prediction", "") != no_value[key].get("prediction", "")
        for key in keys
    )
    return {
        "strict_triples": len(keys),
        "tasks": list(TASKS),
        "method": METHOD,
        "ablation": ablation,
        "debug_contract": {
            "value_sketch_rank": 0 if ablation == "disable_sketch" else 16,
            "value_sketch_bits": 0 if ablation == "disable_sketch" else 4,
            "tail_alpha": None if ablation == "disable_sketch" else 0.0,
            "full_fallback_count": sum(
                no_value[key].get("executed_path") != METHOD for key in keys
            ),
        },
        "macro_score": macro_scores,
        "macro_quality_retention": {
            "frozen_current": current_vs_full,
            "no_value_sketch": no_value_vs_full,
        },
        "no_value_minus_current_macro": (
            macro_scores["no_value_sketch"] - macro_scores["frozen_current"]
        ),
        "prediction_changes": prediction_changes,
        "per_task": {
            task: {
                "full_kv": task_scores["full_kv"][task],
                "frozen_current": task_scores["frozen_current"][task],
                "no_value_sketch": task_scores["no_value_sketch"][task],
            }
            for task in TASKS
        },
        "claim_boundary": (
            "This weak-task diagnostic isolates ValueSketch quality. Its timing "
            "is not compared with the reference run because diagnostics differ."
        ),
    }


def write_task_csv(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task", "full_kv", "frozen_current", "no_value_sketch"),
        )
        writer.writeheader()
        for task, row in payload["per_task"].items():
            writer.writerow({"task": task, **row})


def main() -> None:
    args = parse_args()
    payload = summarize(
        read_rows(args.reference_run_root),
        read_rows(args.ab_run_root),
        args.expected_pairs,
        args.ablation,
    )
    output = args.ab_run_root / "valuesketch_weak_task_ab_summary.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_task_csv(
        args.ab_run_root / "valuesketch_weak_task_ab_per_task.csv", payload
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
