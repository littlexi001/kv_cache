from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from analyze_stepwise_set_utility import mcnemar_exact_p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired retrieval and reader comparison for two frozen rankings."
    )
    parser.add_argument("--baseline_rows_path", required=True)
    parser.add_argument("--candidate_rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def paired_ci(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    difference = candidate.astype(np.float64) - baseline.astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(difference), size=(samples, len(difference)))
    means = difference[draws].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def compare_metric(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    field: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline = np.asarray([bool(row[field]) for row in baseline_rows])
    candidate = np.asarray([bool(row[field]) for row in candidate_rows])
    wins = int(np.sum(candidate & ~baseline))
    losses = int(np.sum(baseline & ~candidate))
    return {
        "baseline_rate": float(baseline.mean()),
        "candidate_rate": float(candidate.mean()),
        "difference": float(candidate.mean() - baseline.mean()),
        "paired_bootstrap_95_ci": paired_ci(
            baseline, candidate, samples=samples, seed=seed
        ),
        "wins_losses": [wins, losses],
        "mcnemar_p": mcnemar_exact_p(wins, losses),
    }


def compare_group(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "steps": len(baseline_rows),
        "retrieval_at_3": compare_metric(
            baseline_rows,
            candidate_rows,
            "retrieval_target_span_hit_at_k",
            samples=samples,
            seed=seed,
        ),
        "top1_generation": compare_metric(
            baseline_rows,
            candidate_rows,
            "target_hit",
            samples=samples,
            seed=seed + 1,
        ),
        "any_branch_generation": compare_metric(
            baseline_rows,
            candidate_rows,
            "any_branch_target_hit",
            samples=samples,
            seed=seed + 2,
        ),
        "baseline_mean_f1": statistics.fmean(
            float(row["target_f1"]) for row in baseline_rows
        ),
        "candidate_mean_f1": statistics.fmean(
            float(row["target_f1"]) for row in candidate_rows
        ),
        "baseline_mean_parallel_seconds": statistics.fmean(
            float(row["parallel_branch_critical_seconds"]) for row in baseline_rows
        ),
        "candidate_mean_parallel_seconds": statistics.fmean(
            float(row["parallel_branch_critical_seconds"]) for row in candidate_rows
        ),
    }


def main() -> None:
    args = parse_args()
    baseline = {
        (int(row["query_id"]), int(row["step_index"])): row
        for row in read_jsonl(Path(args.baseline_rows_path))
    }
    candidate = {
        (int(row["query_id"]), int(row["step_index"])): row
        for row in read_jsonl(Path(args.candidate_rows_path))
    }
    if set(baseline) != set(candidate):
        raise ValueError("paired reader runs do not cover identical step keys")
    keys = sorted(baseline)
    step_types = sorted({str(baseline[key]["step_type"]) for key in keys})
    groups: dict[str, Any] = {}
    for step_type in [*step_types, "overall"]:
        selected = [
            key
            for key in keys
            if step_type == "overall" or str(baseline[key]["step_type"]) == step_type
        ]
        groups[step_type] = compare_group(
            [baseline[key] for key in selected],
            [candidate[key] for key in selected],
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
    payload = {
        "source": "paired frozen-ranking reader comparison",
        "selection_uses_gold": False,
        "baseline_rows_path": args.baseline_rows_path,
        "candidate_rows_path": args.candidate_rows_path,
        "groups": groups,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
