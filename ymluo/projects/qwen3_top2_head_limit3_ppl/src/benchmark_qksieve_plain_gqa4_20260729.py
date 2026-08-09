from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import _get_build_directory

import mixedblock_spectral_cuda_20260729 as optimized_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_qksieve_mixedblock_cuda_20260729 import (
    HIGH_ALLOCATIONS,
    measure_ms,
    sample_count_for,
    selected_fraction,
    unbiased_fraction,
)


BASELINE_EXTENSION = "qksieve_mixedblock_spectral_20260729_v8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="32768,65536,120000")
    parser.add_argument("--warmup", type=int, default=15)
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


def adapter_metadata(
    packed_index: dict[str, torch.Tensor | int],
) -> dict[str, torch.Tensor | int]:
    bit_allocations = packed_index["bit_allocations"].contiguous()
    batch_count, kv_head_count, _ = bit_allocations.shape
    block_size = 32
    block_count = math.ceil(
        int(packed_index["capacity"]) / block_size
    )

    def duplicate(tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack((tensor, tensor), dim=0).contiguous()

    return {
        "bit_allocations": duplicate(bit_allocations),
        "code_offsets": duplicate(
            packed_index["code_offsets"].contiguous()
        ),
        "scale_offsets": duplicate(
            packed_index["scale_offsets"].contiguous()
        ),
        "code_strides": duplicate(
            packed_index["code_strides"].contiguous()
        ),
        "scale_strides": duplicate(
            packed_index["scale_strides"].contiguous()
        ),
        "block_hot_prefix": torch.zeros(
            batch_count * kv_head_count,
            block_count + 1,
            dtype=torch.int16,
            device=bit_allocations.device,
        ),
        "head_code_bases": packed_index["code_bases"]
        .reshape(-1)
        .contiguous(),
        "head_scale_bases": packed_index["scale_bases"]
        .reshape(-1)
        .contiguous(),
        "block_size": block_size,
    }


def capacity_for(
    history_count: int,
    fraction: float,
    sample_count: int,
) -> int:
    standard_error = math.sqrt(
        fraction * (1.0 - fraction) / sample_count
    )
    capacity_fraction = min(
        1.0, max(0.06, fraction + 6.0 * standard_error)
    )
    return max(1, math.ceil(capacity_fraction * history_count))


def allocate_outputs(capacity: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty(
            1, 32, capacity, dtype=torch.long, device="cuda"
        ),
        torch.empty(1, 32, dtype=torch.long, device="cuda"),
        torch.empty(1, 32, dtype=torch.float32, device="cuda"),
        torch.empty(1, 32, dtype=torch.bool, device="cuda"),
    )


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
        capacity = capacity_for(
            history_count, fraction, sample_count
        )
        projected_query = torch.randn(
            1, 8, 4, 128, dtype=torch.float16, device="cuda"
        )
        query_codes, query_scales = varbit_cuda.quantize_projected_query(
            projected_query
        )
        packed_index = varbit_cuda.allocate_packed_index(
            HIGH_ALLOCATIONS.unsqueeze(0).cuda(),
            history_count,
            torch.float16,
        )
        packed_index["packed_codes"].random_(0, 256)
        packed_index["key_scales"].uniform_(0.05, 1.0)
        adapted = adapter_metadata(packed_index)
        baseline_outputs = allocate_outputs(capacity)
        optimized_outputs = allocate_outputs(capacity)

        def baseline_call() -> None:
            baseline.sampled_compact_gqa4_indices_out(
                query_codes,
                query_scales,
                packed_index["packed_codes"],
                packed_index["key_scales"],
                adapted["bit_allocations"],
                adapted["code_offsets"],
                adapted["scale_offsets"],
                adapted["code_strides"],
                adapted["scale_strides"],
                adapted["block_hot_prefix"],
                adapted["head_code_bases"],
                adapted["head_scale_bases"],
                *baseline_outputs,
                history_count,
                int(adapted["block_size"]),
                sample_count,
                threshold_fraction,
            )

        def optimized_call() -> None:
            optimized_cuda.plain_sampled_threshold_compact_gqa4_indices_out(
                query_codes,
                query_scales,
                packed_index,
                *optimized_outputs,
                history_count,
                sample_count,
                threshold_fraction,
            )

        baseline_call()
        optimized_call()
        torch.cuda.synchronize()
        baseline_counts = baseline_outputs[1].clone()
        optimized_counts = optimized_outputs[1].clone()
        baseline_thresholds = baseline_outputs[2].clone()
        optimized_thresholds = optimized_outputs[2].clone()
        iterations = min(
            args.iterations,
            50 if history_count >= 96_000 else args.iterations,
        )
        baseline_ms = measure_ms(
            baseline_call, args.warmup, iterations
        )
        optimized_ms = measure_ms(
            optimized_call, args.warmup, iterations
        )
        set_equal = True
        for row in range(32):
            baseline_count = int(
                baseline_counts.reshape(-1)[row].item()
            )
            optimized_count = int(
                optimized_counts.reshape(-1)[row].item()
            )
            baseline_indices = torch.sort(
                baseline_outputs[0].reshape(32, -1)[
                    row, :baseline_count
                ]
            ).values
            optimized_indices = torch.sort(
                optimized_outputs[0].reshape(32, -1)[
                    row, :optimized_count
                ]
            ).values
            if (
                baseline_count != optimized_count
                or not torch.equal(
                    baseline_indices, optimized_indices
                )
            ):
                set_equal = False
                break
        rows.append(
            {
                "history_count": history_count,
                "candidate_count_mean": float(
                    baseline_counts.float().mean().item()
                ),
                "candidate_count_max_abs_diff": int(
                    (
                        baseline_counts - optimized_counts
                    )
                    .abs()
                    .max()
                    .item()
                ),
                "threshold_max_abs_diff": float(
                    (
                        baseline_thresholds - optimized_thresholds
                    )
                    .abs()
                    .max()
                    .item()
                ),
                "candidate_sets_equal": set_equal,
                "adapter_ms": baseline_ms,
                "plain_layout_ms": optimized_ms,
                "speedup": baseline_ms / optimized_ms,
            }
        )
        del packed_index, adapted
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
