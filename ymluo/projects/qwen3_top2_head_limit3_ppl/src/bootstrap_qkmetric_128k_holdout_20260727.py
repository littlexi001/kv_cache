from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


FULL_METHOD = "full_attention"
SPARSE_METHOD = "direct_countcap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired hierarchical moving-block bootstrap for the independent "
            "128K QK-balanced holdout."
        )
    )
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument(
        "--full_root",
        type=Path,
        default=None,
        help=(
            "Optional separate root providing matched Full KV token rows. "
            "Defaults to input_root."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--block_length", type=int, default=16)
    parser.add_argument("--replicates", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def read_methods(path: Path) -> dict[str, dict[int, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_method: dict[str, dict[int, float]] = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[int(row["target_index"])] = float(
            row["nll"]
        )
    return by_method


def load_case(
    sparse_path: Path,
    full_path: Path | None = None,
) -> dict[str, object]:
    sparse_methods = read_methods(sparse_path)
    full_methods = (
        read_methods(full_path)
        if full_path is not None
        else sparse_methods
    )
    by_method = {
        FULL_METHOD: full_methods.get(FULL_METHOD, {}),
        SPARSE_METHOD: sparse_methods.get(SPARSE_METHOD, {}),
    }
    if FULL_METHOD not in by_method or SPARSE_METHOD not in by_method:
        raise ValueError(f"{sparse_path} lacks a paired Full/Sparse result")
    if not by_method[FULL_METHOD] or not by_method[SPARSE_METHOD]:
        raise ValueError(f"{sparse_path} lacks a paired Full/Sparse result")
    common = sorted(set(by_method[FULL_METHOD]) & set(by_method[SPARSE_METHOD]))
    if len(common) != len(by_method[FULL_METHOD]) or len(common) != len(
        by_method[SPARSE_METHOD]
    ):
        raise ValueError(f"{sparse_path} has incomplete token pairs")
    full = np.asarray([by_method[FULL_METHOD][index] for index in common])
    sparse = np.asarray([by_method[SPARSE_METHOD][index] for index in common])
    return {
        "case": sparse_path.parent.name,
        "tokens": len(common),
        "full_nll": full,
        "sparse_nll": sparse,
        "delta_nll": sparse - full,
    }


def circular_block_sample(
    values: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    token_count = len(values)
    block_count = math.ceil(token_count / block_length)
    starts = rng.integers(0, token_count, size=block_count)
    offsets = np.arange(block_length)
    indices = (starts[:, None] + offsets[None, :]) % token_count
    return values[indices.reshape(-1)[:token_count]]


def percentile_interval(values: np.ndarray) -> dict[str, float]:
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
    paths = sorted(args.input_root.glob("*/token_results.csv"))
    cases = []
    for path in paths:
        full_path = (
            args.full_root / path.parent.name / "token_results.csv"
            if args.full_root is not None
            else None
        )
        if full_path is not None and not full_path.is_file():
            raise FileNotFoundError(
                f"missing matched Full KV token rows: {full_path}"
            )
        cases.append(load_case(path, full_path))
    if not cases:
        raise ValueError("no token_results.csv files found")

    all_full = np.concatenate([case["full_nll"] for case in cases])
    all_sparse = np.concatenate([case["sparse_nll"] for case in cases])
    all_delta = all_sparse - all_full
    point_full_ppl = math.exp(float(all_full.mean()))
    point_sparse_ppl = math.exp(float(all_sparse.mean()))
    point_retention = point_full_ppl / point_sparse_ppl

    rng = np.random.default_rng(args.seed)
    replicate_deltas = np.empty(args.replicates, dtype=np.float64)
    case_count = len(cases)
    for replicate in range(args.replicates):
        sampled_case_indices = rng.integers(0, case_count, size=case_count)
        sampled_deltas = [
            circular_block_sample(
                cases[index]["delta_nll"],
                args.block_length,
                rng,
            )
            for index in sampled_case_indices
        ]
        replicate_deltas[replicate] = np.concatenate(sampled_deltas).mean()

    replicate_retentions = np.exp(-replicate_deltas)
    summary = {
        "protocol": {
            "sparse_root": str(args.input_root),
            "full_root": str(
                args.full_root
                if args.full_root is not None
                else args.input_root
            ),
            "independent_cases": case_count,
            "tokens": int(len(all_delta)),
            "pairing": "same target token under Full KV and QK-balanced sparse KV",
            "bootstrap": "case resampling plus circular moving-token blocks",
            "block_length": args.block_length,
            "replicates": args.replicates,
            "seed": args.seed,
        },
        "cases": [
            {
                "case": case["case"],
                "tokens": case["tokens"],
                "mean_delta_nll": float(case["delta_nll"].mean()),
                "quality_retention": math.exp(-float(case["delta_nll"].mean())),
            }
            for case in cases
        ],
        "point_estimate": {
            "full_ppl": point_full_ppl,
            "sparse_ppl": point_sparse_ppl,
            "mean_delta_nll": float(all_delta.mean()),
            "quality_retention": point_retention,
        },
        "bootstrap_95_percent": {
            "mean_delta_nll": percentile_interval(replicate_deltas),
            "quality_retention": percentile_interval(replicate_retentions),
        },
        "bootstrap_probabilities": {
            "retention_ge_0p95": float((replicate_retentions >= 0.95).mean()),
            "retention_ge_0p98": float((replicate_retentions >= 0.98).mean()),
            "retention_ge_0p99": float((replicate_retentions >= 0.99).mean()),
            "retention_ge_1p00": float((replicate_retentions >= 1.00).mean()),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "bootstrap_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "bootstrap_samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["replicate", "mean_delta_nll", "quality_retention"])
        writer.writerows(
            (
                index,
                float(delta),
                float(retention),
            )
            for index, (delta, retention) in enumerate(
                zip(replicate_deltas, replicate_retentions, strict=True)
            )
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
