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
        description="Compare shared-prefix and sequential completeness probes."
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
    shared_static = np.asarray(
        [record["probe"]["shared_static_completeness_log_odds"] for record in records]
    )
    shared_dynamic = np.asarray(
        [record["probe"]["shared_dynamic_completeness_log_odds"] for record in records]
    )
    shared_delta = shared_dynamic - shared_static
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
    shared_seconds = np.asarray([record["probe"]["total_seconds"] for record in records])
    sequential_seconds = np.asarray(
        [
            float(record["reference_probe"]["static_completeness_seconds"])
            + float(record["reference_probe"]["dynamic_completeness_seconds"])
            for record in records
        ]
    )
    logical_tokens = np.asarray(
        [record["probe"]["logical_prompt_tokens"] for record in records]
    )
    executed_tokens = np.asarray([record["probe"]["executed_tokens"] for record in records])
    output = {
        "protocol": {
            "memory_scope": "eight independent real 10M-token shards",
            "selection_uses_answer_at_test": False,
            "shared_prefix_is_exact_token_prefix": True,
            "logical_completeness_scores": 2,
            "model_calls": "one common prefix plus one batched branch call",
        },
        "queries": len(records),
        "changed_candidate_sets": int(changed.sum()),
        "numerical_fidelity_changed": {
            "static_mean_absolute_error": float(
                np.abs(shared_static[changed] - reference_static[changed]).mean()
            ),
            "dynamic_mean_absolute_error": float(
                np.abs(shared_dynamic[changed] - reference_dynamic[changed]).mean()
            ),
            "delta_mean_absolute_error": float(
                np.abs(shared_delta[changed] - reference_delta[changed]).mean()
            ),
            "utility_sign_agreement": float(
                np.mean((shared_delta[changed] > 0) == (reference_delta[changed] > 0))
            ),
        },
        "compute_changed": {
            "logical_prompt_tokens": float(logical_tokens[changed].mean()),
            "executed_tokens": float(executed_tokens[changed].mean()),
            "token_execution_reduction": float(
                1.0 - executed_tokens[changed].mean() / logical_tokens[changed].mean()
            ),
            "sequential_two_forward_mean_seconds": float(sequential_seconds[changed].mean()),
            "shared_prefix_mean_seconds": float(shared_seconds[changed].mean()),
            "wall_clock_speedup": float(
                sequential_seconds[changed].mean() / shared_seconds[changed].mean()
            ),
            "queries_shared_prefix_faster": int(
                (shared_seconds[changed] < sequential_seconds[changed]).sum()
            ),
            "queries_changed": int(changed.sum()),
        },
        "score_diagnostics": {
            "sequential_reference": score_diagnostics(records, reference_delta),
            "shared_prefix": score_diagnostics(records, shared_delta),
        },
        "zero_threshold_gate": selection_summary(
            records,
            shared_delta > 0,
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
