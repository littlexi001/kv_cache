#!/usr/bin/env python
"""Benchmark layer-parallel versus batched request-local QK factors."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import statistics
import time
from pathlib import Path

import torch

import run_head_top2_targeted_ppl_20260714 as qksieve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--workers", type=int, default=36)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(callable_, device: torch.device) -> tuple[float, tuple[torch.Tensor, ...]]:
    synchronize(device)
    start = time.perf_counter()
    output = callable_()
    synchronize(device)
    return time.perf_counter() - start, output


def main() -> None:
    args = parse_args()
    if args.layers <= 0 or args.kv_heads <= 0 or args.head_dim <= 0:
        raise ValueError("matrix dimensions must be positive")
    if args.workers <= 0 or args.repeats <= 0:
        raise ValueError("workers and repeats must be positive")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    shape = (args.layers, 1, args.kv_heads, args.head_dim, args.head_dim)
    identity = torch.eye(args.head_dim, dtype=torch.float32).view(
        1, 1, 1, args.head_dim, args.head_dim
    )
    key_source = torch.randn(shape, dtype=torch.float32)
    query_source = torch.randn(shape, dtype=torch.float32)
    key_covariance = (
        key_source @ key_source.transpose(-1, -2) / args.head_dim
        + 0.05 * identity
    )
    query_covariance = (
        query_source @ query_source.transpose(-1, -2) / args.head_dim
        + 0.05 * identity
    )
    key_square_root, key_inverse_square_root, key_eigenvalues = (
        qksieve._symmetric_covariance_factors_with_spectrum(key_covariance)
    )
    key_square_root = key_square_root.to(device)
    key_inverse_square_root = key_inverse_square_root.to(device)
    key_eigenvalues = key_eigenvalues.to(device)
    query_covariance = query_covariance.to(device)

    def solve_layer(layer: int) -> tuple[torch.Tensor, ...]:
        return qksieve._qk_metric_projection_factors_from_key_factors(
            key_square_root[layer],
            key_inverse_square_root[layer],
            key_eigenvalues[layer],
            query_covariance[layer],
            projection_dim=args.head_dim,
            query_shrinkage=0.75,
        )

    def solve_parallel() -> tuple[torch.Tensor, ...]:
        with ThreadPoolExecutor(
            max_workers=min(args.workers, args.layers)
        ) as executor:
            rows = list(executor.map(solve_layer, range(args.layers)))
        return tuple(torch.stack([row[index] for row in rows], dim=0)
                     for index in range(3))

    def solve_batched() -> tuple[torch.Tensor, ...]:
        return qksieve._qk_metric_projection_factors_from_key_factors(
            key_square_root,
            key_inverse_square_root,
            key_eigenvalues,
            query_covariance,
            projection_dim=args.head_dim,
            query_shrinkage=0.75,
        )

    solve_parallel()
    solve_batched()
    parallel_seconds: list[float] = []
    batched_seconds: list[float] = []
    parallel_output = None
    batched_output = None
    for repeat in range(args.repeats):
        order = ("parallel", "batched") if repeat % 2 == 0 else (
            "batched", "parallel"
        )
        for mode in order:
            if mode == "parallel":
                seconds, parallel_output = timed(solve_parallel, device)
                parallel_seconds.append(seconds)
            else:
                seconds, batched_output = timed(solve_batched, device)
                batched_seconds.append(seconds)
    assert parallel_output is not None and batched_output is not None
    max_abs = [
        float((parallel - batched).abs().max().item())
        for parallel, batched in zip(parallel_output, batched_output)
    ]
    exact = [
        bool(torch.equal(parallel, batched))
        for parallel, batched in zip(parallel_output, batched_output)
    ]
    parallel_median = statistics.median(parallel_seconds)
    batched_median = statistics.median(batched_seconds)
    result = {
        "schema": "qksieve_batched_qk_factor_benchmark_v1",
        "device": str(device),
        "layers": args.layers,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "workers": min(args.workers, args.layers),
        "repeats": args.repeats,
        "parallel_seconds": parallel_seconds,
        "batched_seconds": batched_seconds,
        "parallel_median_seconds": parallel_median,
        "batched_median_seconds": batched_median,
        "batched_speedup": parallel_median / batched_median,
        "output_exact": exact,
        "output_max_abs": max_abs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
