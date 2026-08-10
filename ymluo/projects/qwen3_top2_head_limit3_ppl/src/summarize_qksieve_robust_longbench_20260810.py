#!/usr/bin/env python
"""Strict paired summary for post-freeze QKSieve-Robust LongBench runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import qksieve_robust_contract_20260810 as contract
import summarize_qksieve_frozen_longbench_20260807 as base


SAMPLE_COUNT_AUDIT_SCHEMA = "qksieve_decode_mean_sample_count_v1"


def expected_decode_mean_sample_count(row: dict[str, str]) -> float:
    """Reconstruct the mean sample count reported over decode steps."""

    prompt_tokens = int(float(row["prompt_tokens"]))
    generated_tokens = int(float(row["generated_tokens"]))
    configured = int(float(row["configured_sampled_quantile_sample_count"]))
    selected_fraction = float(row["configured_attention_fraction"])
    if prompt_tokens <= 0 or generated_tokens <= 0:
        raise AssertionError("sample-count audit requires non-empty decode")
    if not 0.0 < selected_fraction <= 1.0:
        raise AssertionError("invalid configured attention fraction")

    desired = max(
        256,
        configured,
        math.ceil(16.0 / selected_fraction),
    )
    cap = min(contract.MAX_QUANTILE_SAMPLE_COUNT, desired)
    growing_steps = min(generated_tokens, max(0, cap - prompt_tokens))
    growing_sum = (
        growing_steps * (2 * prompt_tokens + growing_steps - 1) / 2.0
    )
    capped_sum = (generated_tokens - growing_steps) * cap
    return (growing_sum + capped_sum) / generated_tokens


def audit_sparse_result_row(row: dict[str, str]) -> float:
    """Audit raw decode-mean diagnostics, then the frozen static contract."""

    expected = expected_decode_mean_sample_count(row)
    observed = float(row["packed_qmse_sample_count"])
    error = abs(observed - expected)
    if error > 1e-6:
        raise AssertionError(
            "decode-mean effective sample-count mismatch: "
            f"{observed} != {expected}"
        )

    # The frozen contract predates decode-mean aggregation and expects the
    # initial-history value. Normalize only this already-audited diagnostic so
    # that all remaining frozen-path checks still run unchanged.
    normalized = dict(row)
    normalized["packed_qmse_sample_count"] = str(
        contract.expected_effective_sample_count(
            int(float(row["prefix_tokens"]))
        )
    )
    contract.audit_sparse_row(normalized)
    return error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--expected_pairs", required=True, type=int)
    parser.add_argument("--expected_tasks", default=16, type=int)
    parser.add_argument("--bootstrap_resamples", default=10000, type=int)
    parser.add_argument("--seed", default=20260810, type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def task_bootstrap(
    payload: dict[str, Any], resamples: int, seed: int
) -> dict[str, list[float]]:
    if resamples <= 0:
        return {}
    tasks = sorted(payload["per_task"])
    rng = random.Random(seed)
    retentions: list[float] = []
    deltas: list[float] = []
    for _ in range(resamples):
        sampled = [rng.choice(tasks) for _ in tasks]
        full = sum(
            payload["per_task"][task][base.REFERENCE_METHOD]["score"]
            for task in sampled
        ) / len(sampled)
        ours = sum(
            payload["per_task"][task][contract.METHOD]["score"]
            for task in sampled
        ) / len(sampled)
        if full > 0.0:
            retentions.append(ours / full)
        deltas.append(ours - full)
    result = {
        "macro_score_delta_95ci": [
            _percentile(deltas, 0.025),
            _percentile(deltas, 0.975),
        ]
    }
    if retentions:
        result["quality_retention_95ci"] = [
            _percentile(retentions, 0.025),
            _percentile(retentions, 0.975),
        ]
    return result


def summarize(
    rows: list[dict[str, str]],
    expected_pairs: int,
    expected_tasks: int,
    *,
    bootstrap_resamples: int = 10000,
    seed: int = 20260810,
) -> dict[str, Any]:
    indexed, _ = base.strict_index(rows, expected_pairs, expected_tasks)
    sparse_rows = list(indexed[contract.METHOD].values())
    sample_count_errors = [
        audit_sparse_result_row(row) for row in sparse_rows
    ]

    payload = base.summarize(rows, expected_pairs, expected_tasks)
    payload.update(
        schema="qksieve_robust_longbench_summary_v1",
        rows=len(rows),
        frozen_contract=contract.contract_payload(),
        effective_sample_count_mean=base.mean(
            [float(row["packed_qmse_sample_count"]) for row in sparse_rows]
        ),
        sample_count_audit={
            "schema": SAMPLE_COUNT_AUDIT_SCHEMA,
            "rows": len(sparse_rows),
            "max_abs_error": max(sample_count_errors, default=0.0),
            "definition": (
                "mean_j min(prompt_tokens+j,512,max(256,configured,"
                "ceil(16/fraction)))"
            ),
        },
        summarizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        value_sketch_tail_alpha=contract.VALUE_SKETCH_TAIL_ALPHA,
        bootstrap={
            "unit": "task",
            "resamples": bootstrap_resamples,
            "seed": seed,
            **task_bootstrap(payload, bootstrap_resamples, seed),
        },
    )
    return payload


def main() -> None:
    args = parse_args()
    rows = base.load_rows(args.run_root)
    payload = summarize(
        rows,
        args.expected_pairs,
        args.expected_tasks,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    output = args.output or args.run_root / "paired_summary.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    base.write_per_task_csv(args.run_root / "per_task.csv", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
