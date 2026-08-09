#!/usr/bin/env python
"""Strict audit and summary for the frozen QKSieve LongBench path."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from run_sample_calibrated_longbench_20260717 import (
    QKSIEVE_FROZEN_C64_METHOD,
    QKSIEVE_FROZEN_C64_SCORE_MODE,
    tail_resolution_sample_count,
)


REFERENCE_METHOD = "full_kv"
TIMING_FIELDS = (
    "prefill_seconds",
    "query_seconds",
    "decode_seconds",
    "online_seconds",
    "total_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--expected_pairs", required=True, type=int)
    parser.add_argument("--expected_tasks", default=16, type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_rows(run_root: Path) -> list[dict[str, str]]:
    paths = sorted(run_root.glob("shard[0-9]*/sample_results.csv"))
    if not paths and (run_root / "sample_results.csv").is_file():
        paths = [run_root / "sample_results.csv"]
    if not paths:
        raise FileNotFoundError(f"no sample_results.csv under {run_root}")
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def expected_attention_tokens(history_tokens: int) -> int:
    return min(history_tokens, max(256, min(math.ceil(0.06 * history_tokens), 1280)))


def strict_index(
    rows: list[dict[str, str]], expected_pairs: int, expected_tasks: int
) -> tuple[
    dict[str, dict[tuple[str, str], dict[str, str]]],
    list[str],
]:
    methods = (REFERENCE_METHOD, QKSIEVE_FROZEN_C64_METHOD)
    counts = Counter(row["method"] for row in rows)
    expected_counts = Counter({method: expected_pairs for method in methods})
    if counts != expected_counts:
        raise AssertionError(
            f"method counts differ: expected={expected_counts}, got={counts}"
        )
    indexed: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    for method in methods:
        selected: dict[tuple[str, str], dict[str, str]] = {}
        for row in rows:
            if row["method"] != method:
                continue
            key = (row["task"], row["sample_id"])
            if key in selected:
                raise AssertionError(f"duplicate {method} row: {key}")
            selected[key] = row
        indexed[method] = selected
    keys = set(indexed[REFERENCE_METHOD])
    if len(keys) != expected_pairs:
        raise AssertionError(f"expected {expected_pairs} strict pairs, got {len(keys)}")
    if set(indexed[QKSIEVE_FROZEN_C64_METHOD]) != keys:
        raise AssertionError("Full and frozen QKSieve keys are not strictly paired")
    tasks = sorted({task for task, _ in keys})
    if len(tasks) != expected_tasks:
        raise AssertionError(f"expected {expected_tasks} tasks, got {len(tasks)}")
    return indexed, tasks


def audit_sparse_row(row: dict[str, str]) -> None:
    if row["executed_path"] != QKSIEVE_FROZEN_C64_METHOD:
        raise AssertionError(
            f"quality/cost fallback detected: {row['executed_path']}"
        )
    if row["configured_score_mode"] != QKSIEVE_FROZEN_C64_SCORE_MODE:
        raise AssertionError("frozen score mode mismatch")
    if abs(float(row["configured_index_bits_per_token"]) - 306.0) > 1e-6:
        raise AssertionError("auxiliary index must be 306 bits/token/head")
    history_tokens = int(row["prefix_tokens"])
    expected_budget = expected_attention_tokens(history_tokens)
    actual_budget = int(float(row["configured_attention_tokens"]))
    if actual_budget != expected_budget:
        raise AssertionError(
            f"budget mismatch for {history_tokens}: {actual_budget} != {expected_budget}"
        )
    expected_samples = tail_resolution_sample_count(
        64, expected_budget / history_tokens
    )
    actual_samples = int(
        float(row["configured_sampled_quantile_sample_count"])
    )
    if actual_samples != expected_samples:
        raise AssertionError(
            f"sample-count mismatch: {actual_samples} != {expected_samples}"
        )
    prebuilt = int(float(row.get("qk_prebuild_layers", 0) or 0))
    batched = int(
        float(row.get("qk_batched_allocation_layers", 0) or 0)
    )
    if prebuilt <= 0 or batched != prebuilt:
        raise AssertionError(
            f"QK prebuild/batched allocation mismatch: {prebuilt}/{batched}"
        )


def method_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": len(rows),
        "score": mean([float(row["score"]) for row in rows]),
        "prompt_tokens": mean([float(row["prompt_tokens"]) for row in rows]),
        "generated_tokens": mean(
            [float(row["generated_tokens"]) for row in rows]
        ),
    }
    for field in TIMING_FIELDS:
        result[field] = mean([float(row[field]) for row in rows])
    return result


def summarize(
    rows: list[dict[str, str]], expected_pairs: int, expected_tasks: int
) -> dict[str, Any]:
    indexed, tasks = strict_index(rows, expected_pairs, expected_tasks)
    for row in indexed[QKSIEVE_FROZEN_C64_METHOD].values():
        audit_sparse_row(row)
    keys_by_task: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in indexed[REFERENCE_METHOD]:
        keys_by_task[key[0]].append(key)

    per_task: dict[str, Any] = {}
    macro_scores = {
        REFERENCE_METHOD: [],
        QKSIEVE_FROZEN_C64_METHOD: [],
    }
    for task in tasks:
        task_result: dict[str, Any] = {"samples": len(keys_by_task[task])}
        for method in (REFERENCE_METHOD, QKSIEVE_FROZEN_C64_METHOD):
            selected = [indexed[method][key] for key in keys_by_task[task]]
            task_result[method] = method_metrics(selected)
            macro_scores[method].append(task_result[method]["score"])
        full = task_result[REFERENCE_METHOD]
        ours = task_result[QKSIEVE_FROZEN_C64_METHOD]
        ours["quality_retention"] = (
            ours["score"] / full["score"] if full["score"] > 0 else None
        )
        for field in TIMING_FIELDS:
            ours[field.replace("_seconds", "_speedup")] = (
                full[field] / ours[field] if ours[field] > 0 else None
            )
        per_task[task] = task_result

    methods: dict[str, Any] = {}
    for method in (REFERENCE_METHOD, QKSIEVE_FROZEN_C64_METHOD):
        selected = list(indexed[method].values())
        methods[method] = method_metrics(selected)
        methods[method]["macro_score"] = mean(macro_scores[method])
    full = methods[REFERENCE_METHOD]
    ours = methods[QKSIEVE_FROZEN_C64_METHOD]
    ours["quality_retention"] = (
        ours["macro_score"] / full["macro_score"]
        if full["macro_score"] > 0
        else None
    )
    for field in TIMING_FIELDS:
        ours[field.replace("_seconds", "_speedup")] = (
            full[field] / ours[field] if ours[field] > 0 else None
        )
    sparse_rows = list(indexed[QKSIEVE_FROZEN_C64_METHOD].values())
    return {
        "schema": "qksieve_frozen_c64_longbench_summary_v1",
        "strict_pairs": expected_pairs,
        "tasks": expected_tasks,
        "full_fallback_count": 0,
        "score_mode": QKSIEVE_FROZEN_C64_SCORE_MODE,
        "auxiliary_index_bits_per_token_per_head": 306.0,
        "attention_tokens_mean": mean(
            [float(row["configured_attention_tokens"]) for row in sparse_rows]
        ),
        "attention_fraction_mean": mean(
            [float(row["configured_attention_fraction"]) for row in sparse_rows]
        ),
        "sample_count_mean": mean(
            [
                float(row["configured_sampled_quantile_sample_count"])
                for row in sparse_rows
            ]
        ),
        "methods": methods,
        "per_task": per_task,
    }


def write_per_task_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    for task, result in payload["per_task"].items():
        full = result[REFERENCE_METHOD]
        ours = result[QKSIEVE_FROZEN_C64_METHOD]
        rows.append(
            {
                "task": task,
                "samples": result["samples"],
                "full_score": full["score"],
                "qksieve_score": ours["score"],
                "quality_retention": ours["quality_retention"],
                "full_online_seconds": full["online_seconds"],
                "qksieve_online_seconds": ours["online_seconds"],
                "online_speedup": ours["online_speedup"],
                "full_total_seconds": full["total_seconds"],
                "qksieve_total_seconds": ours["total_seconds"],
                "total_speedup": ours["total_speedup"],
                "generated_tokens": ours["generated_tokens"],
                "prompt_tokens": ours["prompt_tokens"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    payload = summarize(
        load_rows(args.run_root), args.expected_pairs, args.expected_tasks
    )
    output = args.output or args.run_root / "paired_summary.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_per_task_csv(args.run_root / "per_task.csv", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
