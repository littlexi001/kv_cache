from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_longmemeval_10m_single_action_utility_probe import load_records
from analyze_longmemeval_10m_pairwise_set_utility_probe import (
    score_diagnostics,
    selection_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare batched and sequential independent-completeness probes."
    )
    parser.add_argument("--probe_pattern", required=True)
    parser.add_argument("--reference_probe_pattern", required=True)
    parser.add_argument("--reader_pattern", required=True)
    parser.add_argument("--selection_pattern", required=True)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args)
    batch_static = np.asarray(
        [record["probe"]["batched_static_completeness_log_odds"] for record in records]
    )
    batch_dynamic = np.asarray(
        [record["probe"]["batched_dynamic_completeness_log_odds"] for record in records]
    )
    batch_delta = batch_dynamic - batch_static
    reference_static = np.asarray(
        [record["reference_probe"]["static_completeness_log_odds"] for record in records]
    )
    reference_dynamic = np.asarray(
        [record["reference_probe"]["dynamic_completeness_log_odds"] for record in records]
    )
    reference_delta = reference_dynamic - reference_static
    changed = np.asarray(
        [not bool(record["probe"]["sets_identical"]) for record in records]
    )
    batch_seconds = np.asarray([record["probe"]["batch_seconds"] for record in records])
    sequential_seconds = np.asarray(
        [
            float(record["reference_probe"]["static_completeness_seconds"])
            + float(record["reference_probe"]["dynamic_completeness_seconds"])
            for record in records
        ]
    )
    absolute_errors = {
        "static": np.abs(batch_static - reference_static),
        "dynamic": np.abs(batch_dynamic - reference_dynamic),
        "delta": np.abs(batch_delta - reference_delta),
    }
    output = {
        "protocol": {
            "memory_scope": "eight independent real 10M-token shards",
            "selection_uses_answer_at_test": False,
            "logical_scores": 2,
            "batched_gpu_model_calls": 1,
            "reference_gpu_model_calls": 2,
        },
        "queries": len(records),
        "changed_candidate_sets": int(changed.sum()),
        "numerical_fidelity_changed": {
            key: {
                "mean_absolute_error": float(values[changed].mean()),
                "max_absolute_error": float(values[changed].max()),
            }
            for key, values in absolute_errors.items()
        },
        "utility_sign_agreement_changed": float(
            np.mean((batch_delta[changed] > 0) == (reference_delta[changed] > 0))
        ),
        "latency_changed": {
            "sequential_two_forward_mean_seconds": float(sequential_seconds[changed].mean()),
            "batched_one_call_mean_seconds": float(batch_seconds[changed].mean()),
            "wall_clock_speedup": float(
                sequential_seconds[changed].mean() / batch_seconds[changed].mean()
            ),
            "paired_mean_seconds_saved": float(
                (sequential_seconds[changed] - batch_seconds[changed]).mean()
            ),
            "queries_batched_faster": int(
                (batch_seconds[changed] < sequential_seconds[changed]).sum()
            ),
            "queries_changed": int(changed.sum()),
        },
        "score_diagnostics": {
            "sequential_reference": score_diagnostics(records, reference_delta),
            "batched": score_diagnostics(records, batch_delta),
        },
        "zero_threshold_gate": selection_summary(
            records,
            batch_delta > 0,
            samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
