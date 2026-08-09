#!/usr/bin/env python3
"""Validate and summarize strictly paired aligned RetroInfer LongBench rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from run_retroinfer_aligned_longbench_20260728 import (
    FULL_METHOD,
    RETROINFER_METHOD,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--expected_pairs", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _key(row: dict[str, str]) -> tuple[str, str]:
    return row["task"], row["sample_id"]


def _macro(rows: dict[tuple[str, str], dict[str, str]]) -> float:
    by_task: dict[str, list[float]] = defaultdict(list)
    for (task, _), row in rows.items():
        by_task[task].append(float(row["score"]))
    return sum(
        sum(values) / len(values)
        for values in by_task.values()
    ) / len(by_task)


def _sum_float(rows: list[dict[str, str]], field: str) -> float:
    return sum(float(row[field]) for row in rows)


def analyze(
    rows: list[dict[str, str]],
    *,
    expected_pairs: int,
) -> dict[str, Any]:
    expected_methods = (FULL_METHOD, RETROINFER_METHOD)
    by_method = {
        method: {
            _key(row): row
            for row in rows
            if row["method"] == method
        }
        for method in expected_methods
    }
    full_keys = set(by_method[FULL_METHOD])
    if (
        len(rows) != expected_pairs * 2
        or len(full_keys) != expected_pairs
        or set(by_method[RETROINFER_METHOD]) != full_keys
        or set(row["method"] for row in rows) != set(expected_methods)
    ):
        raise ValueError("RetroInfer rows are not strict Full/RetroInfer pairs")
    tasks = sorted({task for task, _ in full_keys})
    if len(tasks) != 16:
        raise ValueError("aligned RetroInfer report requires 16 tasks")

    contract_errors: list[str] = []
    for sample_key in sorted(full_keys):
        full = by_method[FULL_METHOD][sample_key]
        retro = by_method[RETROINFER_METHOD][sample_key]
        for field in (
            "prompt_sha256",
            "prompt_tokens",
            "prompt_truncation_mode",
            "prompt_wrapper",
            "stop_token_ids",
            "max_new_tokens",
            "official_repository_commit",
            "model_name_or_path",
            "dtype",
        ):
            if full[field] != retro[field]:
                contract_errors.append(f"{sample_key}: {field}")
        if full["protocol"] != "qksieve_aligned_longbench_v1":
            contract_errors.append(f"{sample_key}: full protocol")
        if retro["protocol"] != "qksieve_aligned_longbench_v1":
            contract_errors.append(f"{sample_key}: RetroInfer protocol")
    if contract_errors:
        raise ValueError(
            "aligned RetroInfer contract errors: "
            + "; ".join(contract_errors[:20])
        )

    full_macro = _macro(by_method[FULL_METHOD])
    retro_macro = _macro(by_method[RETROINFER_METHOD])
    full_rows = list(by_method[FULL_METHOD].values())
    retro_rows = list(by_method[RETROINFER_METHOD].values())
    full_decode_steps = sum(int(row["decode_steps"]) for row in full_rows)
    retro_decode_steps = sum(int(row["decode_steps"]) for row in retro_rows)
    full_decode_tpot = (
        _sum_float(full_rows, "decode_seconds") / full_decode_steps
        if full_decode_steps
        else None
    )
    retro_decode_tpot = (
        _sum_float(retro_rows, "decode_seconds") / retro_decode_steps
        if retro_decode_steps
        else None
    )
    per_task: dict[str, Any] = {}
    for task in tasks:
        task_keys = [key for key in full_keys if key[0] == task]
        full_score = sum(
            float(by_method[FULL_METHOD][key]["score"])
            for key in task_keys
        ) / len(task_keys)
        retro_score = sum(
            float(by_method[RETROINFER_METHOD][key]["score"])
            for key in task_keys
        ) / len(task_keys)
        per_task[task] = {
            "samples": len(task_keys),
            "full_score": full_score,
            "retroinfer_score": retro_score,
            "quality_retention": (
                retro_score / full_score if full_score else None
            ),
        }

    return {
        "schema": "qksieve_retroinfer_aligned_longbench_summary_v1",
        "strict_pairs": expected_pairs,
        "tasks": len(tasks),
        "full_macro_score": full_macro,
        "retroinfer_macro_score": retro_macro,
        "quality_retention": (
            retro_macro / full_macro if full_macro else None
        ),
        "decode_tpot_seconds": {
            FULL_METHOD: full_decode_tpot,
            RETROINFER_METHOD: retro_decode_tpot,
        },
        "decode_speedup": (
            full_decode_tpot / retro_decode_tpot
            if full_decode_tpot and retro_decode_tpot
            else None
        ),
        "request_total_speedup": (
            _sum_float(full_rows, "total_seconds")
            / _sum_float(retro_rows, "total_seconds")
        ),
        "median_fixed_seconds": {
            method: {
                field: median(
                    float(row[field])
                    for row in by_method[method].values()
                )
                for field in (
                    "cache_init_seconds",
                    "cache_prepare_seconds",
                    "graph_capture_seconds",
                )
            }
            for method in expected_methods
        },
        "peak_memory": {
            method: {
                "gpu_peak_allocated_bytes": max(
                    int(row["gpu_peak_allocated_bytes"])
                    for row in by_method[method].values()
                ),
                "gpu_peak_reserved_bytes": max(
                    int(row["gpu_peak_reserved_bytes"])
                    for row in by_method[method].values()
                ),
                "cpu_peak_rss_bytes": max(
                    int(row["cpu_peak_rss_bytes"])
                    for row in by_method[method].values()
                ),
            }
            for method in expected_methods
        },
        "native_operating_point": {
            "retrieval_budget": float(
                retro_rows[0]["retrieval_budget"]
            ),
            "estimation_budget": float(
                retro_rows[0]["estimation_budget"]
            ),
            "cache_ratio": float(retro_rows[0]["cache_ratio"]),
        },
        "per_task": per_task,
        "claim_boundary": (
            "Official RetroInfer backend under aligned evaluation protocol; "
            "not an original RetrievalAttention reproduction."
        ),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.run_root.glob("shard[0-9]*/sample_results.csv"))
    if not paths:
        raise SystemExit("no aligned RetroInfer shard CSVs found")
    report = analyze(
        read_rows(paths),
        expected_pairs=args.expected_pairs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
