#!/usr/bin/env python
"""Strict paired summary for post-freeze QKSieve-Robust RULER runs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import qksieve_robust_contract_20260810 as contract
import summarize_qksieve_frozen_c64_ruler_20260807 as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--expected_tasks", default=base.DEFAULT_TASKS)
    parser.add_argument("--expected_length_samples", required=True)
    parser.add_argument("--bootstrap_resamples", default=10000, type=int)
    parser.add_argument("--seed", default=20260810, type=int)
    return parser.parse_args()


def strict_pairs(
    rows: list[dict[str, str]],
    tasks: tuple[str, ...],
    length_samples: dict[int, int],
) -> dict[tuple[str, int, str], dict[str, dict[str, str]]]:
    expected_methods = {base.REFERENCE_METHOD, contract.METHOD}
    expected_pairs = len(tasks) * sum(length_samples.values())
    counts = Counter(row["method"] for row in rows)
    expected_counts = Counter(
        {method: expected_pairs for method in expected_methods}
    )
    if counts != expected_counts:
        raise AssertionError(
            f"method counts differ: expected={expected_counts}, got={counts}"
        )

    grouped: dict[
        tuple[str, int, str], dict[str, dict[str, str]]
    ] = defaultdict(dict)
    for row in rows:
        key = (
            row["base_task"],
            int(row["requested_length"]),
            row["sample_id"],
        )
        method = row["method"]
        if method in grouped[key]:
            raise AssertionError(f"duplicate RULER row: {key}, {method}")
        grouped[key][method] = row
    if len(grouped) != expected_pairs:
        raise AssertionError(
            f"expected {expected_pairs} strict pairs, found {len(grouped)}"
        )
    if any(set(pair) != expected_methods for pair in grouped.values()):
        raise AssertionError("one or more RULER examples are not strictly paired")

    if {key[0] for key in grouped} != set(tasks):
        raise AssertionError("observed RULER task set differs from protocol")
    if {key[1] for key in grouped} != set(length_samples):
        raise AssertionError("observed RULER length set differs from protocol")
    cell_counts = Counter((task, length) for task, length, _ in grouped)
    for task in tasks:
        for length, expected in length_samples.items():
            if cell_counts[(task, length)] != expected:
                raise AssertionError(
                    f"{task}@{length}: expected {expected}, "
                    f"found {cell_counts[(task, length)]}"
                )

    for pair in grouped.values():
        full = pair[base.REFERENCE_METHOD]
        sparse = pair[contract.METHOD]
        contract.audit_sparse_row(sparse)
        if int(full["prompt_tokens"]) != int(sparse["prompt_tokens"]):
            raise AssertionError("Full and QKSieve prompt lengths differ")
        if int(sparse["suffix_tokens"]) <= 0:
            raise AssertionError("QKSieve RULER row has no dense query suffix")
    return grouped


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(
    rows: list[dict[str, str]],
    tasks: tuple[str, ...],
    length_samples: dict[int, int],
    *,
    bootstrap_resamples: int = 10000,
    seed: int = 20260810,
) -> dict[str, Any]:
    grouped = strict_pairs(rows, tasks, length_samples)
    by_cell: dict[
        tuple[str, int], list[dict[str, dict[str, str]]]
    ] = defaultdict(list)
    for (task, length, _), pair in grouped.items():
        by_cell[(task, length)].append(pair)
    per_task_length = {
        f"{task}@{length}": {
            "task": task,
            "length": length,
            **base.cell_metrics(by_cell[(task, length)]),
        }
        for task in tasks
        for length in length_samples
    }
    per_length = {
        str(length): base.aggregate_cells(
            [per_task_length[f"{task}@{length}"] for task in tasks]
        )
        for length in length_samples
    }
    overall = base.aggregate_cells(list(per_task_length.values()))

    rng = random.Random(seed)
    cell_keys = sorted(per_task_length)
    retention_samples: list[float] = []
    delta_samples: list[float] = []
    for _ in range(max(0, bootstrap_resamples)):
        sampled = [rng.choice(cell_keys) for _ in cell_keys]
        full = sum(
            per_task_length[key][base.REFERENCE_METHOD]["score"]
            for key in sampled
        ) / len(sampled)
        ours = sum(
            per_task_length[key][contract.METHOD]["score"]
            for key in sampled
        ) / len(sampled)
        if full > 0.0:
            retention_samples.append(ours / full)
        delta_samples.append(ours - full)

    sparse_rows = [pair[contract.METHOD] for pair in grouped.values()]
    bootstrap: dict[str, Any] = {
        "unit": "task-length cell",
        "resamples": bootstrap_resamples,
        "seed": seed,
    }
    if delta_samples:
        bootstrap["macro_score_delta_95ci"] = [
            _percentile(delta_samples, 0.025),
            _percentile(delta_samples, 0.975),
        ]
    if retention_samples:
        bootstrap["quality_retention_95ci"] = [
            _percentile(retention_samples, 0.025),
            _percentile(retention_samples, 0.975),
        ]
    return {
        "schema": "qksieve_robust_ruler_summary_v1",
        "strict_pairs": len(grouped),
        "rows": len(rows),
        "tasks": list(tasks),
        "length_samples": length_samples,
        "fallback_count": 0,
        "frozen_contract": contract.contract_payload(),
        "attention_tokens_mean": base.mean(
            [float(row["configured_attention_tokens"]) for row in sparse_rows]
        ),
        "attention_fraction_mean": base.mean(
            [float(row["configured_attention_fraction"]) for row in sparse_rows]
        ),
        "effective_sample_count_mean": base.mean(
            [float(row["packed_qmse_sample_count"]) for row in sparse_rows]
        ),
        "overall": overall,
        "per_length": per_length,
        "per_task_length": per_task_length,
        "bootstrap": bootstrap,
        "timing_claim_boundary": (
            "Generation-harness timing is diagnostic because methods may stop "
            "after different token counts. Paper systems claims use isolated "
            "fixed-step benchmarks."
        ),
    }


def main() -> None:
    args = parse_args()
    tasks = base.parse_csv_values(args.expected_tasks)
    length_samples = base.parse_length_samples(args.expected_length_samples)
    rows = base.load_rows(args.run_root)
    payload = summarize(
        rows,
        tasks,
        length_samples,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    output = args.run_root / "paired_summary.json"
    merged = args.run_root / "sample_results.csv"
    with merged.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload["merged_csv_sha256"] = base.sha256(merged)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
