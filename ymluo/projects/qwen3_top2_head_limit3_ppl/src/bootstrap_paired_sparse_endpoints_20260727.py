from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two sparse endpoints with case resampling and circular "
            "moving-token blocks."
        )
    )
    parser.add_argument("--left_root", type=Path, required=True)
    parser.add_argument("--right_root", type=Path, required=True)
    parser.add_argument("--method", default="direct_countcap")
    parser.add_argument("--block_length", type=int, default=16)
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_method(path: Path, method: str) -> dict[int, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            int(row["target_index"]): float(row["nll"])
            for row in rows
            if row["method"] == method
        }


def circular_block_mean(
    values: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> float:
    token_count = len(values)
    block_count = math.ceil(token_count / block_length)
    starts = rng.integers(0, token_count, size=block_count)
    offsets = np.arange(block_length)
    indices = (starts[:, None] + offsets[None, :]) % token_count
    return float(values[indices.reshape(-1)[:token_count]].mean())


def interval(values: np.ndarray) -> dict[str, float]:
    low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "lower_2p5": float(low),
        "median": float(median),
        "upper_97p5": float(high),
    }


def main() -> None:
    args = parse_args()
    if args.block_length <= 0 or args.replicates <= 0:
        raise ValueError("block_length and replicates must be positive")

    left_paths = {
        path.parent.name: path
        for path in args.left_root.glob("*/token_results.csv")
    }
    right_paths = {
        path.parent.name: path
        for path in args.right_root.glob("*/token_results.csv")
    }
    case_names = sorted(set(left_paths) & set(right_paths))
    if not case_names:
        raise ValueError("the endpoint roots have no cases in common")

    cases = []
    for case_name in case_names:
        left = load_method(left_paths[case_name], args.method)
        right = load_method(right_paths[case_name], args.method)
        if not left or set(left) != set(right):
            raise ValueError(f"{case_name} lacks strict token pairing")
        indices = sorted(left)
        delta = np.asarray(
            [right[index] - left[index] for index in indices],
            dtype=np.float64,
        )
        cases.append((case_name, delta))

    all_delta = np.concatenate([delta for _, delta in cases])
    point_delta = float(all_delta.mean())
    point_retention = math.exp(-point_delta)

    rng = np.random.default_rng(args.seed)
    samples = np.empty(args.replicates, dtype=np.float64)
    for replicate in range(args.replicates):
        sampled_case_indices = rng.integers(0, len(cases), size=len(cases))
        samples[replicate] = np.mean(
            [
                circular_block_mean(
                    cases[index][1],
                    args.block_length,
                    rng,
                )
                for index in sampled_case_indices
            ]
        )
    retention_samples = np.exp(-samples)

    result = {
        "protocol": {
            "left_root": str(args.left_root),
            "right_root": str(args.right_root),
            "method": args.method,
            "cases": len(cases),
            "tokens": int(len(all_delta)),
            "block_length": args.block_length,
            "replicates": args.replicates,
            "seed": args.seed,
            "sign": "retention > 1 means right has lower NLL than left",
        },
        "cases": [
            {
                "case": case_name,
                "tokens": int(len(delta)),
                "mean_delta_nll_right_minus_left": float(delta.mean()),
                "right_vs_left_quality_retention": math.exp(
                    -float(delta.mean())
                ),
            }
            for case_name, delta in cases
        ],
        "point_estimate": {
            "mean_delta_nll_right_minus_left": point_delta,
            "right_vs_left_quality_retention": point_retention,
        },
        "bootstrap_95_percent": {
            "mean_delta_nll_right_minus_left": interval(samples),
            "right_vs_left_quality_retention": interval(retention_samples),
        },
        "probabilities": {
            "right_better_than_left": float(
                (retention_samples > 1.0).mean()
            ),
            "right_not_worse_by_0p5_percent": float(
                (retention_samples >= 0.995).mean()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
