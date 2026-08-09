#!/usr/bin/env python
"""Validate exact equivalence of bitonic and k-way-merge sample thresholds."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path

import torch

import qksieve_query_cuda_20260728 as query_cuda
import qksieve_valuesketch_cuda_20260801 as sketch_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_variablebit_spectral_attention_20260727 import (
    ALLOCATION_PROFILES,
)
from validate_qksieve_valuesketch_deterministic_20260804 import (
    allocate_outputs,
    launch_deterministic,
    measure_ms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_module",
        default="mixedblock_spectral_cuda_v40_reference",
    )
    parser.add_argument(
        "--candidate_module",
        default="mixedblock_spectral_cuda_20260729",
    )
    parser.add_argument("--lengths", default="32768,65536,131072")
    parser.add_argument("--seeds", default="20260807,20260808,20260809")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def target_count(history: int) -> int:
    return min(history, 1280, max(256, math.ceil(0.06 * history)))


def c64_sample_count(history: int, selected: int) -> int:
    fraction = selected / history
    return min(
        8192,
        max(256, 256 * math.ceil(math.ceil(64.0 / fraction) / 256)),
    )


def candidate_capacity(history: int, selected: int, samples: int) -> int:
    fraction = selected / history
    upper_fraction = min(
        1.0,
        fraction
        + 6.0 * math.sqrt(fraction * (1.0 - fraction) / samples),
    )
    return min(history, max(selected + 1, math.ceil(upper_fraction * history)))


def relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        (left - right).abs().max().item()
        / max(1.0, float(left.abs().max().item()))
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    reference_module = importlib.import_module(args.reference_module)
    candidate_module = importlib.import_module(args.candidate_module)
    reference_extension = reference_module.load_extension()
    candidate_extension = candidate_module.load_extension()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    rank = 16
    block_size = 256
    scaling = 128.0**-0.5
    rows: list[dict[str, object]] = []

    for history in lengths:
        selected = target_count(history)
        selected_fraction = selected / history
        sample_count = c64_sample_count(history, selected)
        capacity = candidate_capacity(history, selected, sample_count)
        for seed in seeds:
            torch.manual_seed(seed)
            query = torch.randn(1, 32, 128, dtype=dtype, device=device)
            grouped_query = query.reshape(1, 8, 4, 128)
            key_basis = torch.randn(
                1, 8, 128, 128, dtype=dtype, device=device
            )
            query_codes, query_scales = query_cuda.project_quantize(
                grouped_query, key_basis
            )
            key = torch.randn(
                1, 8, history + 1, 128, dtype=dtype, device=device
            )
            value = torch.randn_like(key)
            value_mean = torch.randn(
                1, 8, 128, dtype=dtype, device=device
            )
            value_basis = (
                torch.randn(
                    1, 8, 128, rank, dtype=dtype, device=device
                )
                / math.sqrt(128.0)
            ).contiguous()
            allocation = ALLOCATION_PROFILES["qmse_total_b15"].unsqueeze(0).cuda()
            index = varbit_cuda.allocate_packed_index(
                allocation, history, dtype
            )
            index["packed_codes"].random_(0, 256)
            index["key_scales"].uniform_(0.01, 0.1)
            block_count = math.ceil(history / block_size)
            value_codes = torch.randint(
                0,
                256,
                (1, 8, history, rank // 2),
                dtype=torch.uint8,
                device=device,
            )
            value_minimum = torch.randn(
                1, 8, block_count, rank, dtype=dtype, device=device
            )
            value_scale = torch.rand_like(value_minimum).mul_(0.1).add_(0.01)
            reference_outputs = allocate_outputs(capacity, rank, device)
            candidate_outputs = allocate_outputs(capacity, rank, device)
            reference_masks = torch.empty(
                1,
                32,
                block_count * 8,
                dtype=torch.int32,
                device=device,
            )
            candidate_masks = torch.empty_like(reference_masks)
            reference_partials = torch.empty(
                1,
                32,
                block_count,
                rank + 2,
                dtype=torch.float32,
                device=device,
            )
            candidate_partials = torch.empty_like(reference_partials)
            reference_attention_workspace = (
                sketch_cuda.allocate_attention_workspace(query, capacity)
            )
            candidate_attention_workspace = (
                sketch_cuda.allocate_attention_workspace(query, capacity)
            )

            def reference_call() -> None:
                launch_deterministic(
                    reference_extension,
                    query_codes,
                    query_scales,
                    index,
                    value_codes,
                    value_minimum,
                    value_scale,
                    reference_masks,
                    reference_partials,
                    reference_outputs,
                    history,
                    sample_count,
                    selected_fraction,
                    scaling,
                )

            def candidate_call() -> None:
                launch_deterministic(
                    candidate_extension,
                    query_codes,
                    query_scales,
                    index,
                    value_codes,
                    value_minimum,
                    value_scale,
                    candidate_masks,
                    candidate_partials,
                    candidate_outputs,
                    history,
                    sample_count,
                    selected_fraction,
                    scaling,
                )

            def reference_attention() -> torch.Tensor:
                return sketch_cuda.exact_selected_plus_tail_out(
                    query,
                    key,
                    value,
                    reference_outputs["indices"],
                    reference_outputs["counts"],
                    reference_outputs["thresholds"],
                    reference_outputs["tail_denominator"],
                    reference_outputs["tail_coefficients"],
                    value_mean,
                    value_basis,
                    reference_attention_workspace,
                    scaling,
                    1.0,
                )

            def candidate_attention() -> torch.Tensor:
                return sketch_cuda.exact_selected_plus_tail_out(
                    query,
                    key,
                    value,
                    candidate_outputs["indices"],
                    candidate_outputs["counts"],
                    candidate_outputs["thresholds"],
                    candidate_outputs["tail_denominator"],
                    candidate_outputs["tail_coefficients"],
                    value_mean,
                    value_basis,
                    candidate_attention_workspace,
                    scaling,
                    1.0,
                )

            def reference_complete() -> torch.Tensor:
                reference_call()
                return reference_attention()

            def candidate_complete() -> torch.Tensor:
                candidate_call()
                return candidate_attention()

            reference_call()
            candidate_call()
            torch.cuda.synchronize()
            candidate_sets_equal = True
            candidate_order_equal = True
            for row_index in range(32):
                reference_count = int(
                    reference_outputs["counts"].reshape(-1)[row_index].item()
                )
                candidate_count = int(
                    candidate_outputs["counts"].reshape(-1)[row_index].item()
                )
                reference_indices = reference_outputs["indices"].reshape(
                    32, capacity
                )[row_index, :reference_count]
                candidate_indices = candidate_outputs["indices"].reshape(
                    32, capacity
                )[row_index, :candidate_count]
                candidate_sets_equal &= bool(
                    torch.equal(reference_indices, candidate_indices)
                )
                candidate_order_equal &= bool(
                    torch.equal(reference_indices, candidate_indices)
                )
            reference_ms = measure_ms(
                reference_call, args.warmup, args.iterations
            )
            candidate_ms = measure_ms(
                candidate_call, args.warmup, args.iterations
            )
            reference_final = reference_attention().clone()
            candidate_final = candidate_attention().clone()
            torch.cuda.synchronize()
            final_difference = (
                reference_final.float() - candidate_final.float()
            ).abs()
            reference_complete_ms = measure_ms(
                reference_complete, args.warmup, args.iterations
            )
            candidate_complete_ms = measure_ms(
                candidate_complete, args.warmup, args.iterations
            )
            row = {
                "history_tokens": history,
                "seed": seed,
                "sample_count": sample_count,
                "candidate_capacity": capacity,
                "selected_keep": math.ceil(
                    selected_fraction * sample_count
                ),
                "thresholds_bitwise_equal": bool(
                    torch.equal(
                        reference_outputs["thresholds"],
                        candidate_outputs["thresholds"],
                    )
                ),
                "selection_masks_bitwise_equal": bool(
                    torch.equal(reference_masks, candidate_masks)
                ),
                "counts_bitwise_equal": bool(
                    torch.equal(
                        reference_outputs["counts"],
                        candidate_outputs["counts"],
                    )
                ),
                "candidate_sets_equal": candidate_sets_equal,
                "candidate_order_equal": candidate_order_equal,
                "overflow_equal": bool(
                    torch.equal(
                        reference_outputs["overflow"],
                        candidate_outputs["overflow"],
                    )
                ),
                "selected_denominator_relative_error": relative_error(
                    reference_outputs["selected_denominator"],
                    candidate_outputs["selected_denominator"],
                ),
                "tail_denominator_relative_error": relative_error(
                    reference_outputs["tail_denominator"],
                    candidate_outputs["tail_denominator"],
                ),
                "tail_coefficients_relative_error": relative_error(
                    reference_outputs["tail_coefficients"],
                    candidate_outputs["tail_coefficients"],
                ),
                "reference_ms": reference_ms,
                "candidate_ms": candidate_ms,
                "speedup": reference_ms / candidate_ms,
                "final_output_bitwise_equal": bool(
                    torch.equal(reference_final, candidate_final)
                ),
                "final_output_max_abs_error": float(
                    final_difference.max().item()
                ),
                "final_output_mean_abs_error": float(
                    final_difference.mean().item()
                ),
                "final_output_top1_equal": bool(
                    torch.equal(
                        reference_final.reshape(-1, 128).argmax(dim=-1),
                        candidate_final.reshape(-1, 128).argmax(dim=-1),
                    )
                ),
                "reference_complete_ms": reference_complete_ms,
                "candidate_complete_ms": candidate_complete_ms,
                "complete_speedup": (
                    reference_complete_ms / candidate_complete_ms
                ),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del (
                query,
                grouped_query,
                key_basis,
                query_codes,
                query_scales,
                key,
                value,
                value_mean,
                value_basis,
                index,
                value_codes,
                value_minimum,
                value_scale,
                reference_outputs,
                candidate_outputs,
                reference_masks,
                candidate_masks,
                reference_partials,
                candidate_partials,
                reference_attention_workspace,
                candidate_attention_workspace,
            )
            torch.cuda.empty_cache()

    equality_fields = (
        "thresholds_bitwise_equal",
        "selection_masks_bitwise_equal",
        "counts_bitwise_equal",
        "candidate_sets_equal",
        "candidate_order_equal",
        "overflow_equal",
    )
    result = {
        "schema": "qksieve_warpmerge_threshold_validation_v1",
        "all_discrete_outputs_equal": all(
            bool(row[field]) for row in rows for field in equality_fields
        ),
        "maximum_tail_relative_error": max(
            max(
                float(row["selected_denominator_relative_error"]),
                float(row["tail_denominator_relative_error"]),
                float(row["tail_coefficients_relative_error"]),
            )
            for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
