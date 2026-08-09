#!/usr/bin/env python
"""Strict matched-sample audit for one sparse LongBench method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_qksieve_deployment_longbench_20260801 import (
    FULL,
    actual_active_ratio,
    index_method,
    macro,
    mean,
    read_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_root", type=Path, required=True)
    parser.add_argument("--sparse_root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--score_mode", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--expected_per_task", type=int, required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=50000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def paired_interval(
    full_scores: list[float],
    sparse_scores: list[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    full = np.asarray(full_scores, dtype=np.float64)
    sparse = np.asarray(sparse_scores, dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(full), size=(replicates, len(full)))
    full_boot = full[draws].mean(axis=1)
    sparse_boot = sparse[draws].mean(axis=1)
    valid = full_boot > 0.0
    retention = sparse_boot[valid] / full_boot[valid]
    delta = sparse_boot - full_boot
    retention_ci = (
        [
            float(np.quantile(retention, 0.025)),
            float(np.quantile(retention, 0.975)),
        ]
        if retention.size
        else [None, None]
    )
    return {
        "macro_delta_ci95": [
            float(np.quantile(delta, 0.025)),
            float(np.quantile(delta, 0.975)),
        ],
        "quality_retention_ci95": retention_ci,
        "probability_sparse_ge_full": float(np.mean(delta >= 0.0)),
    }


def analyze(
    full_rows: list[dict[str, str]],
    sparse_rows: list[dict[str, str]],
    *,
    method: str,
    score_mode: str,
    tasks: list[str],
    expected_per_task: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    sparse_all = index_method(sparse_rows, method)
    sparse = {key: row for key, row in sparse_all.items() if key[0] in tasks}
    full_all = index_method(full_rows, FULL)
    full = {key: full_all[key] for key in sparse if key in full_all}
    expected = expected_per_task * len(tasks)
    if len(sparse) != expected or set(full) != set(sparse):
        raise ValueError(
            f"strict match failed: full={len(full)} sparse={len(sparse)} "
            f"expected={expected}"
        )

    per_task: dict[str, Any] = {}
    for task_index, task in enumerate(tasks):
        keys = sorted(key for key in sparse if key[0] == task)
        if len(keys) != expected_per_task:
            raise ValueError(
                f"{task}: expected {expected_per_task}, got {len(keys)}"
            )
        for key in keys:
            row = sparse[key]
            if row.get("executed_path") != method:
                raise ValueError(f"{key}: executed_path mismatch")
            if row.get("configured_score_mode") != score_mode:
                raise ValueError(f"{key}: score-mode mismatch")
            if float(row.get("sampled_quantile_fallback", "0") or 0) != 0.0:
                raise ValueError(f"{key}: fallback occurred")
            for field in ("prompt_tokens", "prefix_tokens", "suffix_tokens"):
                if row[field] != full[key][field]:
                    raise ValueError(f"{key}: {field} protocol mismatch")
        full_scores = [float(full[key]["score"]) for key in keys]
        sparse_scores = [float(sparse[key]["score"]) for key in keys]
        task_full = mean(full_scores)
        task_sparse = mean(sparse_scores)
        per_task[task] = {
            "samples": len(keys),
            "full": task_full,
            "sparse": task_sparse,
            # A tiny matched subset can have a zero Full score. Its additive
            # delta remains meaningful, but a multiplicative ratio does not.
            "quality_retention": task_sparse / task_full if task_full > 0.0 else None,
            "paired_sample_bootstrap": paired_interval(
                full_scores,
                sparse_scores,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + task_index,
            ),
        }

    full_macro = macro(full)
    sparse_macro = macro(sparse)
    all_keys = sorted(sparse)
    report = {
        "schema": "targeted_sparse_longbench_v1",
        "method": method,
        "score_mode": score_mode,
        "strict_pairs": expected,
        "tasks": tasks,
        "full_macro": full_macro,
        "sparse_macro": sparse_macro,
        "quality_retention": sparse_macro / full_macro if full_macro > 0.0 else None,
        "mean_loaded_token_ratio": mean(
            [actual_active_ratio(sparse[key]) for key in all_keys]
        ),
        "mean_configured_candidate_fraction": mean(
            [float(sparse[key]["configured_candidate_fraction"]) for key in all_keys]
        ),
        "paired_sample_bootstrap": paired_interval(
            [float(full[key]["score"]) for key in all_keys],
            [float(sparse[key]["score"]) for key in all_keys],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "per_task": per_task,
        "fairness_contract": {
            "strict_same_samples": True,
            "same_prompt_lengths": True,
            "full_fallback": False,
        },
    }
    return report


def main() -> None:
    args = parse_args()
    full_rows, _ = read_rows(args.full_root)
    sparse_rows, _ = read_rows(args.sparse_root)
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    report = analyze(
        full_rows,
        sparse_rows,
        method=args.method,
        score_mode=args.score_mode,
        tasks=tasks,
        expected_per_task=args.expected_per_task,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
