from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Callable

import torch
from torch.utils.cpp_extension import _get_build_directory

import mixedblock_spectral_cuda_20260729 as optimized_cuda
import qabs_cuda_kernels as sparse_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_qksieve_mixedblock_cuda_20260729 import (
    HIGH_ALLOCATIONS,
    sample_count_for,
    selected_fraction,
    unbiased_fraction,
)
from benchmark_qksieve_plain_gqa4_20260729 import (
    allocate_outputs,
    capacity_for,
)


BASELINE_EXTENSION = "qksieve_mixedblock_spectral_20260729_v11"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="32768,65536,120000")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_baseline_extension() -> object:
    build_directory = _get_build_directory(
        BASELINE_EXTENSION, verbose=False
    )
    sys.path.insert(0, build_directory)
    return importlib.import_module(BASELINE_EXTENSION)


def measure_ms(
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


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    baseline = load_baseline_extension()
    optimized_cuda.load_extension()
    rows: list[dict[str, float | int | bool]] = []
    for history_count in sorted(
        {int(value) for value in args.lengths.split(",")}
    ):
        fraction = selected_fraction(history_count)
        sample_count = sample_count_for(fraction)
        threshold_fraction = unbiased_fraction(fraction, sample_count)
        capacity = capacity_for(history_count, fraction, sample_count)
        projected_query = torch.randn(
            1, 8, 4, 128, dtype=torch.float16, device="cuda"
        )
        query_codes, query_scales = varbit_cuda.quantize_projected_query(
            projected_query
        )
        exact_query = torch.randn(
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
        packed_index = varbit_cuda.allocate_packed_index(
            HIGH_ALLOCATIONS.unsqueeze(0).cuda(),
            history_count,
            torch.float16,
        )
        packed_index["packed_codes"].random_(0, 256)
        packed_index["key_scales"].uniform_(0.05, 1.0)
        baseline_outputs = allocate_outputs(capacity)
        optimized_outputs = allocate_outputs(capacity)
        split = 8 if history_count <= 65536 else 4
        scaling = 128.0**-0.5

        def baseline_selection() -> None:
            baseline.plain_sampled_compact_gqa4_indices_out(
                query_codes,
                query_scales,
                packed_index["packed_codes"],
                packed_index["key_scales"],
                packed_index["bit_allocations"],
                packed_index["code_offsets"],
                packed_index["scale_offsets"],
                packed_index["code_bases"],
                packed_index["scale_bases"],
                packed_index["code_strides"],
                packed_index["scale_strides"],
                *baseline_outputs,
                history_count,
                sample_count,
                threshold_fraction,
            )

        def optimized_selection() -> None:
            optimized_cuda.plain_sampled_threshold_compact_gqa4_indices_out(
                query_codes,
                query_scales,
                packed_index,
                *optimized_outputs,
                history_count,
                sample_count,
                threshold_fraction,
            )

        def baseline_pipeline() -> torch.Tensor:
            baseline_selection()
            return sparse_cuda.final_attention_ragged_self_split(
                exact_query,
                key,
                value,
                baseline_outputs[0],
                baseline_outputs[1],
                scaling,
                split,
            )

        def optimized_pipeline() -> torch.Tensor:
            optimized_selection()
            return sparse_cuda.final_attention_ragged_self_split(
                exact_query,
                key,
                value,
                optimized_outputs[0],
                optimized_outputs[1],
                scaling,
                split,
            )

        baseline_result = baseline_pipeline()
        optimized_result = optimized_pipeline()
        torch.cuda.synchronize()
        iterations = min(
            args.iterations,
            50 if history_count >= 96_000 else args.iterations,
        )
        baseline_ms = measure_ms(
            baseline_pipeline, args.warmup, iterations
        )
        optimized_ms = measure_ms(
            optimized_pipeline, args.warmup, iterations
        )
        rows.append(
            {
                "history_count": history_count,
                "sample_count": sample_count,
                "candidate_count_mean": float(
                    baseline_outputs[1].float().mean().item()
                ),
                "split": split,
                "candidate_count_max_abs_diff": int(
                    (baseline_outputs[1] - optimized_outputs[1])
                    .abs()
                    .max()
                    .item()
                ),
                "output_max_abs_error": float(
                    (baseline_result - optimized_result)
                    .abs()
                    .max()
                    .item()
                ),
                "atomic_pipeline_ms": baseline_ms,
                "blockwise_pipeline_ms": optimized_ms,
                "speedup": baseline_ms / optimized_ms,
            }
        )
        del key, value, packed_index
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
