#!/usr/bin/env python3
"""Summarize paired qMSE, Key-MSE, and fixed-layout QKSieve PPL runs."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


TAGS = ("qmse", "keymse", "fixed411111")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--expected_pairs", type=int, default=12)
    parser.add_argument("--bootstrap_iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def weighted_mean(rows: list[dict[str, Any]], field: str) -> float:
    valid = [row for row in rows if row.get(field) is not None]
    total = sum(int(row["tokens"]) for row in valid)
    if not valid or total <= 0:
        raise ValueError(f"no weighted values for {field}")
    return sum(float(row[field]) * int(row["tokens"]) for row in valid) / total


def load_pairs(path: Path) -> dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["topic"]), int(row["window"]))
        method = str(row["method"])
        if method in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate {key}/{method} in {path}")
        grouped[key][method] = row
    output = {}
    for key, methods in grouped.items():
        if set(methods) != {"full_attention", "direct_countcap"}:
            raise ValueError(f"incomplete pair {key}: {sorted(methods)}")
        output[key] = (
            methods["full_attention"],
            methods["direct_countcap"],
        )
    return output


def aggregate(
    pairs: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, float]:
    full = [value[0] for value in pairs.values()]
    sparse = [value[1] for value in pairs.values()]
    full_nll = weighted_mean(full, "nll")
    sparse_nll = weighted_mean(sparse, "nll")
    full_step = weighted_mean(full, "steady_sparse_seconds_per_step")
    sparse_step = weighted_mean(
        sparse, "steady_sparse_seconds_per_step"
    )
    return {
        "pair_count": float(len(pairs)),
        "full_ppl": math.exp(full_nll),
        "sparse_ppl": math.exp(sparse_nll),
        "delta_nll": sparse_nll - full_nll,
        "quality_retention": math.exp(full_nll - sparse_nll),
        "top1_agreement": weighted_mean(sparse, "top1_agreement"),
        "kl_full_to_sparse": weighted_mean(
            sparse, "kl_full_to_sparse_mean"
        ),
        "index_ratio": weighted_mean(
            sparse, "packed_index_ratio_of_full_kv"
        ),
        "attention_tokens": weighted_mean(
            sparse, "actual_attention_tokens_mean"
        ),
        "steady_ms_per_token": 1000.0 * sparse_step,
        "speedup_vs_full": full_step / sparse_step,
    }


def paired_delta(
    left: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]],
    right: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    if set(left) != set(right):
        raise ValueError("paired methods cover different windows")
    values = []
    for key in sorted(left):
        lf, ls = left[key]
        rf, rs = right[key]
        values.append(
            (float(ls["nll"]) - float(lf["nll"]))
            - (float(rs["nll"]) - float(rf["nll"]))
        )
    rng = random.Random(seed)
    samples = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(iterations)
    )

    def percentile(fraction: float) -> float:
        index = round(fraction * (len(samples) - 1))
        return float(samples[index])

    return {
        "mean": sum(values) / len(values),
        "ci95_low": percentile(0.025),
        "ci95_high": percentile(0.975),
        "probability_left_better": (
            sum(value < 0.0 for value in samples) / len(samples)
        ),
    }


def main() -> None:
    args = parse_args()
    pair_sets = {
        tag: load_pairs(args.run_root / tag / "case_summary.json")
        for tag in TAGS
    }
    for tag, pairs in pair_sets.items():
        if len(pairs) != args.expected_pairs:
            raise ValueError(
                f"{tag}: expected {args.expected_pairs}, got {len(pairs)}"
            )
    result = {
        "schema": "qksieve_allocation_structure_ppl_v1",
        "methods": {
            tag: aggregate(pairs) for tag, pairs in pair_sets.items()
        },
        "fixed411111_minus_keymse_delta_nll": paired_delta(
            pair_sets["fixed411111"],
            pair_sets["keymse"],
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        ),
        "keymse_minus_qmse_delta_nll": paired_delta(
            pair_sets["keymse"],
            pair_sets["qmse"],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 1,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
