from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze same-process sequential versus shared-prefix paired timing."
    )
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    rows = [row for row in read_jsonl(Path(args.rows)) if not row["sets_identical"]]
    sequential = np.asarray([row["paired_sequential_seconds"] for row in rows])
    shared = np.asarray([row["total_seconds"] for row in rows])
    difference = shared - sequential
    seq_score = np.asarray([row["paired_sequential_utility_score"] for row in rows])
    shared_score = np.asarray([row["shared_completeness_utility_score"] for row in rows])
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(rows), size=(args.bootstrap_samples, len(rows)))
    sampled_difference = difference[draws].mean(axis=1)
    output = {
        "protocol": {
            "same_process_and_model": True,
            "execution_order_alternates_by_query": True,
            "logical_prompts_are_identical": True,
        },
        "queries": len(rows),
        "latency": {
            "sequential_mean_seconds": float(sequential.mean()),
            "shared_prefix_mean_seconds": float(shared.mean()),
            "wall_clock_speedup": float(sequential.mean() / shared.mean()),
            "shared_minus_sequential_seconds": float(difference.mean()),
            "bootstrap_95_ci_seconds": [
                float(np.quantile(sampled_difference, 0.025)),
                float(np.quantile(sampled_difference, 0.975)),
            ],
            "shared_faster_queries": int((shared < sequential).sum()),
            "shared_slower_queries": int((shared > sequential).sum()),
            "median_per_query_speedup": float(np.median(sequential / shared)),
        },
        "by_execution_order": {
            order: {
                "queries": len(indices),
                "sequential_mean_seconds": float(sequential[indices].mean()),
                "shared_prefix_mean_seconds": float(shared[indices].mean()),
                "wall_clock_speedup": float(
                    sequential[indices].mean() / shared[indices].mean()
                ),
            }
            for order in ("shared_first", "sequential_first")
            for indices in [
                np.asarray(
                    [index for index, row in enumerate(rows) if row["paired_order"] == order]
                )
            ]
            if len(indices)
        },
        "numerical_fidelity": {
            "utility_mean_absolute_error": float(np.abs(shared_score - seq_score).mean()),
            "utility_sign_agreement": float(
                np.mean((shared_score > 0) == (seq_score > 0))
            ),
        },
        "compute": {
            "logical_prompt_tokens": float(
                np.mean([row["logical_prompt_tokens"] for row in rows])
            ),
            "executed_tokens": float(np.mean([row["executed_tokens"] for row in rows])),
            "token_execution_reduction": float(
                1.0
                - np.mean([row["executed_tokens"] for row in rows])
                / np.mean([row["logical_prompt_tokens"] for row in rows])
            ),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
