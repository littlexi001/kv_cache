#!/usr/bin/env python
"""Join deployment-only QKSieve rows with frozen matched LongBench controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FULL = "full_kv"
REFERENCE = "qksieve_fullprompt_auto_plain_fulltopk"
DEPLOYMENT = "qksieve_global_qkbalanced_keymse_wmma_sampled"
DEPLOYMENT_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_unbiased_packed_direct"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_root", type=Path, required=True)
    parser.add_argument("--deployment_root", type=Path, required=True)
    parser.add_argument("--expected_pairs", type=int, required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=50000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(root: Path) -> tuple[list[dict[str, str]], list[Path]]:
    paths = sorted(root.glob("shard[0-9]*/sample_results.csv"))
    if not paths:
        raise ValueError(f"no shard sample_results.csv files under {root}")
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows, paths


def index_method(
    rows: list[dict[str, str]], method: str
) -> dict[tuple[str, str], dict[str, str]]:
    selected = [row for row in rows if row["method"] == method]
    indexed = {(row["task"], row["sample_id"]): row for row in selected}
    if len(indexed) != len(selected):
        raise ValueError(f"duplicate {method} task/sample rows")
    return indexed


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def macro(rows: dict[tuple[str, str], dict[str, str]]) -> float:
    by_task: dict[str, list[float]] = defaultdict(list)
    for (task, _), row in rows.items():
        by_task[task].append(float(row["score"]))
    return mean([mean(values) for values in by_task.values()])


def actual_active_ratio(row: dict[str, str]) -> float:
    measured = row.get("selected_history_fraction_mean", "")
    if measured not in {"", "nan"} and float(measured) > 0.0:
        return float(measured)
    return min(
        int(row["prefix_tokens"]),
        int(float(row["configured_attention_tokens"])),
    ) / max(1, int(row["prefix_tokens"]))


def paired_task_bootstrap(
    full: dict[tuple[str, str], dict[str, str]],
    sparse: dict[tuple[str, str], dict[str, str]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Resample tasks, then paired examples within every sampled task."""
    tasks = sorted({task for task, _ in full})
    task_keys = [sorted(key for key in full if key[0] == task) for task in tasks]
    sample_counts = {len(keys) for keys in task_keys}
    if len(sample_counts) != 1:
        raise ValueError("bootstrap requires equal sample counts per task")
    samples_per_task = sample_counts.pop()
    full_values = np.asarray(
        [[float(full[key]["score"]) for key in keys] for keys in task_keys],
        dtype=np.float64,
    )
    sparse_values = np.asarray(
        [[float(sparse[key]["score"]) for key in keys] for keys in task_keys],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    task_draws = generator.integers(
        0, len(tasks), size=(replicates, len(tasks))
    )
    sample_draws = generator.integers(
        0,
        samples_per_task,
        size=(replicates, len(tasks), samples_per_task),
    )
    full_boot = full_values[
        task_draws[..., None], sample_draws
    ].mean(axis=-1).mean(axis=-1)
    sparse_boot = sparse_values[
        task_draws[..., None], sample_draws
    ].mean(axis=-1).mean(axis=-1)
    retention = np.divide(
        sparse_boot,
        full_boot,
        out=np.full_like(sparse_boot, np.nan),
        where=full_boot > 0.0,
    )
    retention = retention[np.isfinite(retention)]
    delta = sparse_boot - full_boot
    return {
        "replicates": replicates,
        "seed": seed,
        "protocol": "resample tasks, then paired examples within task",
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


def analyze(
    reference_rows: list[dict[str, str]],
    deployment_rows: list[dict[str, str]],
    *,
    expected_pairs: int,
    bootstrap_replicates: int = 50000,
    bootstrap_seed: int = 20260801,
) -> dict[str, Any]:
    full = index_method(reference_rows, FULL)
    reference = index_method(reference_rows, REFERENCE)
    deployment = index_method(deployment_rows, DEPLOYMENT)
    keys = set(full)
    if not (
        len(keys) == expected_pairs
        and set(reference) == keys
        and set(deployment) == keys
    ):
        raise ValueError(
            "deployment comparison is not strict matched-sample data: "
            f"full={len(full)} reference={len(reference)} "
            f"deployment={len(deployment)} expected={expected_pairs}"
        )
    tasks = sorted({task for task, _ in keys})
    if len(tasks) != 16:
        raise ValueError(f"expected 16 LongBench tasks, got {len(tasks)}")

    for key in sorted(keys):
        reference_row = reference[key]
        deployment_row = deployment[key]
        if deployment_row["executed_path"] != DEPLOYMENT:
            raise ValueError(f"{key}: deployment executed_path mismatch")
        if deployment_row["configured_score_mode"] != DEPLOYMENT_SCORE_MODE:
            raise ValueError(f"{key}: deployment score-mode mismatch")
        if abs(
            float(deployment_row["configured_index_bits_per_token"]) - 240.0
        ) > 1.0e-6:
            raise ValueError(f"{key}: deployment index-rate mismatch")
        if (
            deployment_row["configured_attention_tokens"]
            != reference_row["configured_attention_tokens"]
        ):
            raise ValueError(f"{key}: active-token budget mismatch")
        for field in ("prompt_tokens", "prefix_tokens", "suffix_tokens"):
            if deployment_row[field] != reference_row[field]:
                raise ValueError(f"{key}: {field} protocol mismatch")
        fallback = deployment_row.get("sampled_quantile_fallback", "0")
        if fallback not in {"", "nan"} and float(fallback) != 0.0:
            raise ValueError(f"{key}: sampled-quantile fallback occurred")

    full_macro = macro(full)
    methods: dict[str, Any] = {}
    for method, indexed in (
        (REFERENCE, reference),
        (DEPLOYMENT, deployment),
    ):
        score = macro(indexed)
        methods[method] = {
            "macro_score": score,
            "quality_retention": score / full_macro if full_macro else None,
            "mean_loaded_token_ratio": mean(
                [actual_active_ratio(row) for row in indexed.values()]
            ),
            "paired_task_bootstrap": paired_task_bootstrap(
                full,
                indexed,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            ),
        }

    per_task: dict[str, Any] = {}
    for task in tasks:
        task_keys = sorted(key for key in keys if key[0] == task)
        full_score = mean([float(full[key]["score"]) for key in task_keys])
        per_task[task] = {
            "samples": len(task_keys),
            "full": full_score,
        }
        for method, indexed in (
            (REFERENCE, reference),
            (DEPLOYMENT, deployment),
        ):
            score = mean([float(indexed[key]["score"]) for key in task_keys])
            per_task[task][method] = {
                "score": score,
                "relative_full": score / full_score if full_score else None,
            }

    return {
        "schema": "qksieve_deployment_matched_longbench_v1",
        "strict_pairs": expected_pairs,
        "tasks": len(tasks),
        "full_macro": full_macro,
        "methods": methods,
        "per_task": per_task,
        "fairness_contract": {
            "same_samples": True,
            "same_prompt_protocol": True,
            "same_length_only_active_token_schedule": True,
            "same_exact_selected_kv_attention_consumer": True,
            "full_fallback": False,
            "exact_candidate_rerank": False,
            "recent_or_sink_reservation": False,
            "deployment_selector": (
                "sampled quantile plus fused packed scan and compaction"
            ),
        },
        "latency_claim": {
            "valid": False,
            "reason": (
                "Generation harness time contains Python and model overhead; "
                "direct complete CUDA calls are reported separately."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    reference_rows, reference_paths = read_rows(args.reference_root)
    deployment_rows, deployment_paths = read_rows(args.deployment_root)
    report = analyze(
        reference_rows,
        deployment_rows,
        expected_pairs=args.expected_pairs,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    project_root = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__),
        project_root / "src/run_sample_calibrated_longbench_20260717.py",
        project_root / "src/run_head_top2_targeted_ppl_20260714.py",
    ]
    report["source_sha256"] = {
        str(path.relative_to(project_root)): sha256(path)
        for path in source_paths
    }
    report["input_sha256"] = {
        **{
            f"reference/{path.relative_to(args.reference_root)}": sha256(path)
            for path in reference_paths
        },
        **{
            f"deployment/{path.relative_to(args.deployment_root)}": sha256(path)
            for path in deployment_paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
