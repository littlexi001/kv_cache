from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import run_sample_calibrated_longbench_20260717 as benchmark
import summarize_qksieve_frozen_longbench_20260807 as frozen_audit


REFERENCE_METHOD = "full_kv"
METHOD = benchmark.QKSIEVE_FROZEN_C64_METHOD
DEFAULT_TASKS = (
    "niah_single_1,niah_single_2,niah_single_3,"
    "niah_multikey_1,niah_multikey_2,niah_multikey_3,"
    "niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad,qa_hotpot"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict paired summary for frozen-c64 RULER runs."
    )
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--expected_tasks", default=DEFAULT_TASKS)
    parser.add_argument(
        "--expected_length_samples",
        required=True,
        help="Comma-separated LENGTH:SAMPLES cells, for example 4096:5,8192:5.",
    )
    return parser.parse_args()


def parse_csv_values(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError("expected_tasks must contain unique task names")
    return parsed


def parse_length_samples(value: str) -> dict[int, int]:
    parsed: dict[int, int] = {}
    for item in value.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise ValueError("length sample entries must be LENGTH:SAMPLES")
        length, samples = (int(field) for field in fields)
        if length <= 0 or samples <= 0 or length in parsed:
            raise ValueError("lengths and sample counts must be positive and unique")
        parsed[length] = samples
    if not parsed:
        raise ValueError("at least one RULER length is required")
    return dict(sorted(parsed.items()))


def load_rows(run_root: Path) -> list[dict[str, str]]:
    paths = sorted(run_root.glob("shard[0-9]*/sample_results.csv"))
    if not paths and (run_root / "sample_results.csv").is_file():
        paths = [run_root / "sample_results.csv"]
    if not paths:
        raise FileNotFoundError(f"no RULER sample CSV under {run_root}")
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def geomean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(mean([math.log(value) for value in values]))


def row_tpot(row: dict[str, str], field: str) -> float:
    generated = int(float(row["generated_tokens"]))
    if generated <= 0:
        raise AssertionError("a RULER timing row generated zero tokens")
    return float(row[field]) / generated


def strict_pairs(
    rows: list[dict[str, str]],
    tasks: tuple[str, ...],
    length_samples: dict[int, int],
) -> dict[tuple[str, int, str], dict[str, dict[str, str]]]:
    expected_methods = {REFERENCE_METHOD, METHOD}
    expected_pairs = len(tasks) * sum(length_samples.values())
    expected_counts = Counter(
        {method: expected_pairs for method in expected_methods}
    )
    counts = Counter(row["method"] for row in rows)
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

    observed_tasks = {key[0] for key in grouped}
    observed_lengths = {key[1] for key in grouped}
    if observed_tasks != set(tasks):
        raise AssertionError(f"unexpected task set: {sorted(observed_tasks)}")
    if observed_lengths != set(length_samples):
        raise AssertionError(f"unexpected length set: {sorted(observed_lengths)}")
    cell_counts = Counter((task, length) for task, length, _ in grouped)
    for task in tasks:
        for length, expected in length_samples.items():
            if cell_counts[(task, length)] != expected:
                raise AssertionError(
                    f"{task}@{length}: expected {expected}, "
                    f"found {cell_counts[(task, length)]}"
                )

    for pair in grouped.values():
        full = pair[REFERENCE_METHOD]
        sparse = pair[METHOD]
        frozen_audit.audit_sparse_row(sparse)
        if int(full["prompt_tokens"]) != int(sparse["prompt_tokens"]):
            raise AssertionError("Full and QKSieve prompt lengths differ")
        if int(sparse["suffix_tokens"]) <= 0:
            raise AssertionError("QKSieve RULER row has no dense query suffix")
    return grouped


def method_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        "score": mean([float(row["score"]) for row in rows]),
        "prompt_tokens": mean([float(row["prompt_tokens"]) for row in rows]),
        "generated_tokens": mean(
            [float(row["generated_tokens"]) for row in rows]
        ),
        "prefill_seconds": mean([float(row["prefill_seconds"]) for row in rows]),
        "query_seconds": mean([float(row["query_seconds"]) for row in rows]),
        "decode_seconds": mean([float(row["decode_seconds"]) for row in rows]),
        "online_seconds": mean([float(row["online_seconds"]) for row in rows]),
        "total_seconds": mean([float(row["total_seconds"]) for row in rows]),
        "decode_tpot_seconds": mean(
            [row_tpot(row, "decode_seconds") for row in rows]
        ),
        "online_tpot_seconds": mean(
            [row_tpot(row, "online_seconds") for row in rows]
        ),
    }


def cell_metrics(
    pairs: list[dict[str, dict[str, str]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"samples": len(pairs)}
    for method in (REFERENCE_METHOD, METHOD):
        result[method] = method_metrics([pair[method] for pair in pairs])
    full = result[REFERENCE_METHOD]
    sparse = result[METHOD]
    result["quality_retention"] = (
        sparse["score"] / full["score"] if full["score"] > 0 else None
    )
    result["score_delta"] = sparse["score"] - full["score"]
    for field in (
        "query_seconds",
        "decode_seconds",
        "online_seconds",
        "total_seconds",
        "decode_tpot_seconds",
        "online_tpot_seconds",
    ):
        result[field.replace("_seconds", "_speedup")] = (
            full[field] / sparse[field] if sparse[field] > 0 else None
        )
    return result


def aggregate_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    full_macro = mean([cell[REFERENCE_METHOD]["score"] for cell in cells])
    sparse_macro = mean([cell[METHOD]["score"] for cell in cells])
    speed_fields = (
        "decode_speedup",
        "online_speedup",
        "total_speedup",
        "decode_tpot_speedup",
        "online_tpot_speedup",
    )
    return {
        "cells": len(cells),
        "full_macro": full_macro,
        "qksieve_macro": sparse_macro,
        "quality_retention": (
            sparse_macro / full_macro if full_macro > 0 else None
        ),
        "score_delta": sparse_macro - full_macro,
        **{
            f"geomean_{field}": geomean(
                [float(cell[field]) for cell in cells]
            )
            for field in speed_fields
        },
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(
    rows: list[dict[str, str]],
    tasks: tuple[str, ...],
    length_samples: dict[int, int],
) -> dict[str, Any]:
    grouped = strict_pairs(rows, tasks, length_samples)
    by_cell_pairs: dict[
        tuple[str, int], list[dict[str, dict[str, str]]]
    ] = defaultdict(list)
    for (task, length, _), pair in grouped.items():
        by_cell_pairs[(task, length)].append(pair)
    per_task_length = {
        f"{task}@{length}": {
            "task": task,
            "length": length,
            **cell_metrics(by_cell_pairs[(task, length)]),
        }
        for task in tasks
        for length in length_samples
    }
    per_length = {
        str(length): aggregate_cells(
            [per_task_length[f"{task}@{length}"] for task in tasks]
        )
        for length in length_samples
    }
    sparse_rows = [pair[METHOD] for pair in grouped.values()]
    return {
        "schema": "qksieve_frozen_c64_ruler_summary_v1",
        "strict_pairs": len(grouped),
        "rows": len(rows),
        "tasks": list(tasks),
        "length_samples": length_samples,
        "fallback_count": 0,
        "score_mode": benchmark.QKSIEVE_FROZEN_C64_SCORE_MODE,
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
        "timing_claim_boundary": (
            "Generation-harness timing is diagnostic because methods may stop "
            "after different token counts. Paper systems claims use isolated "
            "fixed-step benchmarks."
        ),
        "overall": aggregate_cells(list(per_task_length.values())),
        "per_length": per_length,
        "per_task_length": per_task_length,
    }


def main() -> None:
    args = parse_args()
    tasks = parse_csv_values(args.expected_tasks)
    length_samples = parse_length_samples(args.expected_length_samples)
    rows = load_rows(args.run_root)
    payload = summarize(rows, tasks, length_samples)
    output = args.run_root / "paired_summary.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    merged = args.run_root / "sample_results.csv"
    with merged.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload["merged_csv_sha256"] = sha256(merged)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
