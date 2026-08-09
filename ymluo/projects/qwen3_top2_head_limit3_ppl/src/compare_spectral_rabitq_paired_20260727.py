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
METRICS = (
    "top2_recall",
    "selected_attention_mass",
    "score_pearson",
)


def normalize_label(value: str) -> str:
    return (
        value.replace("qwen3_4b_", "qwen3_")
        .replace("llama31_8b_", "llama31_")
        .replace("qwen25_7b_", "qwen25_")
    )


def read_rows(root: Path, method: str, fraction: float) -> dict[tuple[str, ...], dict[str, float]]:
    rows: dict[tuple[str, ...], dict[str, float]] = {}
    for path in root.glob("*/per_head.csv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["method"] != method:
                    continue
                if abs(float(row["selected_fraction_target"]) - fraction) > 1.0e-9:
                    continue
                key = tuple(
                    normalize_label(row[field])
                    if field == "label"
                    else row[field]
                    for field in JOIN_FIELDS
                )
                rows[key] = {metric: float(row[metric]) for metric in METRICS}
    return rows


def quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def paired_block_bootstrap(
    pairs: list[tuple[tuple[str, ...], dict[str, float], dict[str, float]]],
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    blocks: dict[tuple[str, str, str], list[tuple[dict[str, float], dict[str, float]]]] = defaultdict(list)
    for key, left, right in pairs:
        blocks[(key[0], key[1], key[2])].append((left, right))
    block_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for items in blocks.values():
        for metric in METRICS:
            block_values[metric].append(
                sum(left[metric] - right[metric] for left, right in items)
                / len(items)
            )
    rng = random.Random(seed)
    block_count = len(blocks)
    output = {}
    for metric in METRICS:
        values = block_values[metric]
        observed = sum(values) / len(values)
        spectral_raw_mean = sum(left[metric] for _, left, _ in pairs) / len(
            pairs
        )
        rabitq_raw_mean = sum(right[metric] for _, _, right in pairs) / len(
            pairs
        )
        samples = []
        for _ in range(iterations):
            samples.append(
                sum(values[rng.randrange(block_count)] for _ in range(block_count))
                / block_count
            )
        output[metric] = {
            "spectral_raw_mean": spectral_raw_mean,
            "rabitq_raw_mean": rabitq_raw_mean,
            "raw_mean_difference": spectral_raw_mean - rabitq_raw_mean,
            "paired_block_mean_difference": observed,
            "bootstrap_ci95_low": quantile(samples, 0.025),
            "bootstrap_ci95_high": quantile(samples, 0.975),
            "probability_difference_positive": sum(value > 0.0 for value in samples)
            / iterations,
        }
    output["metadata"] = {
        "paired_cases": len(pairs),
        "bootstrap_blocks": block_count,
        "bootstrap_iterations": iterations,
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly pair spectral-index and RaBitQ retrieval cases and "
            "compute block-bootstrap confidence intervals."
        )
    )
    parser.add_argument("--spectral_root", required=True, type=Path)
    parser.add_argument("--rabitq_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--spectral_methods",
        default="auto_qmse_b10,auto_qmse_total_b14,auto_qmse_total_b15",
    )
    parser.add_argument(
        "--rabitq_method", default="rabitq_official_fp_query"
    )
    parser.add_argument("--selected_fraction", type=float, default=0.06)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rabitq = read_rows(
        args.rabitq_root, args.rabitq_method, args.selected_fraction
    )
    output: dict[str, Any] = {
        "config": {
            **vars(args),
            "spectral_root": str(args.spectral_root),
            "rabitq_root": str(args.rabitq_root),
            "output": str(args.output),
        },
        "comparisons": {},
    }
    for method in (
        item.strip()
        for item in args.spectral_methods.split(",")
        if item.strip()
    ):
        spectral = read_rows(
            args.spectral_root, method, args.selected_fraction
        )
        common = sorted(set(spectral) & set(rabitq))
        if not common:
            raise RuntimeError(f"no paired cases for {method}")
        pairs = [(key, spectral[key], rabitq[key]) for key in common]
        comparison = paired_block_bootstrap(
            pairs,
            args.bootstrap_iterations,
            args.seed,
        )
        comparison["spectral_method"] = method
        comparison["rabitq_method"] = args.rabitq_method
        comparison["spectral_case_count"] = len(spectral)
        comparison["rabitq_case_count"] = len(rabitq)
        comparison["strict_pair_count"] = len(common)
        output["comparisons"][method] = comparison
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
