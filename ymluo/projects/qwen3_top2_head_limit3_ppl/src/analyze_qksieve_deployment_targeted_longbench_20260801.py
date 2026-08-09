#!/usr/bin/env python
"""Audit deployment QKSieve on targeted LongBench failure modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_qksieve_deployment_longbench_20260801 import (
    DEPLOYMENT,
    DEPLOYMENT_SCORE_MODE,
    FULL,
    actual_active_ratio,
    index_method,
    macro,
    mean,
    paired_task_bootstrap,
    read_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_root", type=Path, required=True)
    parser.add_argument("--deployment_root", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--expected_per_task", type=int, required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=50000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def paired_sample_interval(
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
    retention = np.divide(
        sparse_boot,
        full_boot,
        out=np.full_like(sparse_boot, np.nan),
        where=full_boot > 0.0,
    )
    retention = retention[np.isfinite(retention)]
    delta = sparse_boot - full_boot
    return {
        "macro_delta_ci95": [
            float(np.quantile(delta, 0.025)),
            float(np.quantile(delta, 0.975)),
        ],
        "quality_retention_ci95": [
            float(np.quantile(retention, 0.025)),
            float(np.quantile(retention, 0.975)),
        ],
        "probability_sparse_ge_full": float(np.mean(delta >= 0.0)),
    }


def analyze_targeted(
    full_rows: list[dict[str, str]],
    deployment_rows: list[dict[str, str]],
    *,
    tasks: list[str],
    expected_per_task: int,
    bootstrap_replicates: int = 50000,
    bootstrap_seed: int = 20260801,
) -> dict[str, Any]:
    full_all = index_method(full_rows, FULL)
    deployment_all = index_method(deployment_rows, DEPLOYMENT)
    deployment = {
        key: row for key, row in deployment_all.items() if key[0] in tasks
    }
    full = {key: full_all[key] for key in deployment if key in full_all}
    expected = expected_per_task * len(tasks)
    if len(deployment) != expected or set(full) != set(deployment):
        raise ValueError(
            f"strict targeted match failed: full={len(full)} "
            f"deployment={len(deployment)} expected={expected}"
        )
    for task in tasks:
        count = sum(key[0] == task for key in deployment)
        if count != expected_per_task:
            raise ValueError(f"{task}: expected {expected_per_task}, got {count}")
    for key, row in deployment.items():
        if row["executed_path"] != DEPLOYMENT:
            raise ValueError(f"{key}: executed_path mismatch")
        if row["configured_score_mode"] != DEPLOYMENT_SCORE_MODE:
            raise ValueError(f"{key}: score-mode mismatch")
        if float(row["configured_index_bits_per_token"]) != 240.0:
            raise ValueError(f"{key}: index-rate mismatch")
        if float(row.get("sampled_quantile_fallback", "0") or 0) != 0.0:
            raise ValueError(f"{key}: fallback occurred")
        for field in ("prompt_tokens", "prefix_tokens", "suffix_tokens"):
            if row[field] != full[key][field]:
                raise ValueError(f"{key}: {field} protocol mismatch")

    full_macro = macro(full)
    sparse_macro = macro(deployment)
    per_task: dict[str, Any] = {}
    for task_index, task in enumerate(tasks):
        keys = sorted(key for key in deployment if key[0] == task)
        full_scores = [float(full[key]["score"]) for key in keys]
        sparse_scores = [float(deployment[key]["score"]) for key in keys]
        task_full = mean(full_scores)
        task_sparse = mean(sparse_scores)
        per_task[task] = {
            "samples": len(keys),
            "full": task_full,
            "deployment": task_sparse,
            "quality_retention": task_sparse / task_full,
            "paired_sample_bootstrap": paired_sample_interval(
                full_scores,
                sparse_scores,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + task_index,
            ),
        }
    return {
        "schema": "qksieve_deployment_targeted_longbench_v1",
        "strict_pairs": expected,
        "tasks": tasks,
        "full_macro": full_macro,
        "deployment_macro": sparse_macro,
        "quality_retention": sparse_macro / full_macro,
        "mean_loaded_token_ratio": mean(
            [actual_active_ratio(row) for row in deployment.values()]
        ),
        "paired_task_bootstrap": paired_task_bootstrap(
            full,
            deployment,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "per_task": per_task,
        "fairness_contract": {
            "strict_same_samples": True,
            "same_prompt_lengths": True,
            "full_fallback": False,
            "exact_candidate_rerank": False,
        },
    }


def main() -> None:
    args = parse_args()
    full_rows, _ = read_rows(args.full_root)
    deployment_rows, _ = read_rows(args.deployment_root)
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    report = analyze_targeted(
        full_rows,
        deployment_rows,
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
