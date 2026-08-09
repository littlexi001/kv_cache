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


def candidate_sets_equal(
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
) -> bool:
    for row in range(32):
        left_count = int(left[1].reshape(-1)[row].item())
        right_count = int(right[1].reshape(-1)[row].item())
        if left_count != right_count:
            return False
        left_indices = torch.sort(
            left[0].reshape(32, -1)[row, :left_count]
        ).values
        right_indices = torch.sort(
            right[0].reshape(32, -1)[row, :right_count]
        ).values
        if not torch.equal(left_indices, right_indices):
            return False
    return True


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
        packed_index = varbit_cuda.allocate_packed_index(
            HIGH_ALLOCATIONS.unsqueeze(0).cuda(),
            history_count,
            torch.float16,
        )
        packed_index["packed_codes"].random_(0, 256)
        packed_index["key_scales"].uniform_(0.05, 1.0)
        baseline_outputs = allocate_outputs(capacity)
        optimized_outputs = allocate_outputs(capacity)

        def baseline_call() -> None:
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
        rows.append(
            {
                "history_count": history_count,
                "sample_count": sample_count,
                "selected_keep": math.ceil(
                    threshold_fraction * sample_count
                ),
                "candidate_count_max_abs_diff": int(
                    (baseline_outputs[1] - optimized_outputs[1])
                    .abs()
                    .max()
                    .item()
                ),
                "threshold_max_abs_diff": float(
                    (baseline_outputs[2] - optimized_outputs[2])
                    .abs()
                    .max()
                    .item()
                ),
                "candidate_sets_equal": candidate_sets_equal(
                    baseline_outputs, optimized_outputs
                ),
                "full_sort_ms": baseline_ms,
                "warpselect_ms": optimized_ms,
                "speedup": baseline_ms / optimized_ms,
            }
        )
        del packed_index
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
