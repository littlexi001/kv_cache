from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


JOIN_FIELDS = (
    "label",
    "layer",
    "heldout_step",
    "kv_head",
    "query_head",
    "selected_fraction_target",
)
BLOCK_FIELDS = ("label", "layer", "heldout_step")
METRICS = (
    "top2_recall",
    "selected_attention_mass",
    "top2_attention_mass_recall",
    "score_pearson",
    "score_rmse",
)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def read_method(
    roots: list[Path],
    method: str,
    selected_fraction: float,
) -> dict[tuple[str, ...], dict[str, float]]:
    output: dict[tuple[str, ...], dict[str, float]] = {}
    for root in roots:
        for path in root.glob("*/per_head.csv"):
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["method"] != method:
                        continue
                    if (
                        abs(
                            float(row["selected_fraction_target"])
                            - selected_fraction
                        )
                        > 1.0e-9
                    ):
                        continue
                    key = tuple(row[field] for field in JOIN_FIELDS)
                    if key in output:
                        raise ValueError(
                            f"duplicate paired case {key} for {method}"
                        )
                    output[key] = {
                        metric: float(row[metric]) for metric in METRICS
                    }
    return output


def paired_block_bootstrap(
    left: dict[tuple[str, ...], dict[str, float]],
    right: dict[tuple[str, ...], dict[str, float]],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError("the selected methods have no strictly paired cases")
    block_indices = tuple(
        JOIN_FIELDS.index(field) for field in BLOCK_FIELDS
    )
    blocks: dict[
        tuple[str, ...],
        list[tuple[dict[str, float], dict[str, float]]],
    ] = defaultdict(list)
    for key in common:
        block = tuple(key[index] for index in block_indices)
        blocks[block].append((left[key], right[key]))

    random_generator = random.Random(seed)
    output: dict[str, Any] = {}
    for metric in METRICS:
        block_differences = [
            sum(
                left_row[metric] - right_row[metric]
                for left_row, right_row in rows
            )
            / len(rows)
            for rows in blocks.values()
        ]
        samples = [
            sum(
                block_differences[
                    random_generator.randrange(len(block_differences))
                ]
                for _ in block_differences
            )
            / len(block_differences)
            for _ in range(iterations)
        ]
        left_mean = sum(left[key][metric] for key in common) / len(common)
        right_mean = sum(right[key][metric] for key in common) / len(common)
        raw_difference = left_mean - right_mean
        oriented_difference = (
            -raw_difference if metric == "score_rmse" else raw_difference
        )
        oriented_samples = (
            [-value for value in samples]
            if metric == "score_rmse"
            else samples
        )
        output[metric] = {
            "left_mean": left_mean,
            "right_mean": right_mean,
            "left_minus_right": raw_difference,
            "improvement_positive_difference": oriented_difference,
            "improvement_ci95_low": quantile(
                oriented_samples,
                0.025,
            ),
            "improvement_ci95_high": quantile(
                oriented_samples,
                0.975,
            ),
            "probability_improvement_positive": sum(
                value > 0.0 for value in oriented_samples
            )
            / iterations,
        }
    output["metadata"] = {
        "left_cases": len(left),
        "right_cases": len(right),
        "strict_pair_count": len(common),
        "bootstrap_blocks": len(blocks),
        "bootstrap_iterations": iterations,
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_roots",
        required=True,
        help="Comma-separated roots containing trace/per_head.csv files.",
    )
    parser.add_argument("--left_method", required=True)
    parser.add_argument("--right_method", required=True)
    parser.add_argument("--selected_fraction", type=float, default=0.01)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [
        Path(item.strip())
        for item in args.input_roots.split(",")
        if item.strip()
    ]
    left = read_method(
        roots,
        args.left_method,
        args.selected_fraction,
    )
    right = read_method(
        roots,
        args.right_method,
        args.selected_fraction,
    )
    comparison = paired_block_bootstrap(
        left,
        right,
        args.bootstrap_iterations,
        args.seed,
    )
    output = {
        "config": {
            "input_roots": [str(root) for root in roots],
            "left_method": args.left_method,
            "right_method": args.right_method,
            "selected_fraction": args.selected_fraction,
            "bootstrap_iterations": args.bootstrap_iterations,
            "seed": args.seed,
        },
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
