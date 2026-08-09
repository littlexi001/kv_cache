from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import torch

import preallocated_sparse_attention_cuda_20260729 as preallocated_cuda
import qabs_cuda_kernels as sparse_cuda


def measure_gpu_ms(
    function: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iterations


def measure_wall_ms(
    function: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started) / iterations


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="32768,65536,120000")
    parser.add_argument("--candidate_count", type=int, default=1280)
    parser.add_argument("--capacity_fraction", type=float, default=0.06)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260729)
    rows: list[dict[str, float | int]] = []
    for history_count in sorted(
        {int(item) for item in args.lengths.split(",") if item}
    ):
        candidate_count = min(args.candidate_count, history_count)
        capacity = max(
            candidate_count,
            math.ceil(args.capacity_fraction * history_count),
        )
        split = 8 if history_count <= 65536 else 4
        query = torch.randn(
            1, 32, 128, dtype=torch.float16, device="cuda"
        )
        key = torch.randn(
            1,
            8,
            history_count + 1,
            128,
            dtype=torch.float16,
            device="cuda",
        )
        value = torch.randn_like(key)
        indices = torch.empty(
            1, 32, capacity, dtype=torch.long, device="cuda"
        )
        for head in range(32):
            indices[0, head, :candidate_count] = torch.randperm(
                history_count, device="cuda"
            )[:candidate_count]
        counts = torch.full(
            (1, 32),
            candidate_count,
            dtype=torch.long,
            device="cuda",
        )
        scaling = 128.0**-0.5
        workspace = preallocated_cuda.allocate_workspace(query, split)

        def allocating() -> torch.Tensor:
            return sparse_cuda.final_attention_ragged_self_split(
                query,
                key,
                value,
                indices,
                counts,
                scaling,
                split,
            )

        def preallocated() -> torch.Tensor:
            return preallocated_cuda.forward_out(
                query,
                key,
                value,
                indices,
                counts,
                workspace,
                scaling,
                split,
            )

        allocating_output = allocating()
        preallocated_output = preallocated()
        torch.cuda.synchronize()
        iterations = min(
            args.iterations,
            100 if history_count >= 96_000 else args.iterations,
        )
        allocating_gpu_ms = measure_gpu_ms(
            allocating, args.warmup, iterations
        )
        preallocated_gpu_ms = measure_gpu_ms(
            preallocated, args.warmup, iterations
        )
        allocating_wall_ms = measure_wall_ms(
            allocating, args.warmup, iterations
        )
        preallocated_wall_ms = measure_wall_ms(
            preallocated, args.warmup, iterations
        )
        rows.append(
            {
                "history_count": history_count,
                "candidate_count": candidate_count,
                "candidate_capacity": capacity,
                "split": split,
                "max_abs_error": float(
                    (allocating_output - preallocated_output)
                    .abs()
                    .max()
                    .item()
                ),
                "allocating_gpu_ms": allocating_gpu_ms,
                "preallocated_gpu_ms": preallocated_gpu_ms,
                "gpu_speedup": (
                    allocating_gpu_ms / preallocated_gpu_ms
                ),
                "allocating_wall_ms": allocating_wall_ms,
                "preallocated_wall_ms": preallocated_wall_ms,
                "wall_speedup": (
                    allocating_wall_ms / preallocated_wall_ms
                ),
            }
        )
        del query, key, value, indices, workspace
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
