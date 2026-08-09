from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch

import mixedblock_spectral_cuda_20260729 as mixedblock_cuda
import qabs_cuda_kernels as qabs_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_qksieve_mixedblock_cuda_20260729 import (
    LOW_PROFILES,
    measure_ms,
    mixed_metadata,
    sample_count_for,
    selected_fraction,
    unbiased_fraction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="32768,65536,120000")
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--hot_fraction", type=float, default=0.10)
    parser.add_argument("--low_profile", default="fixed441")
    parser.add_argument("--split_count", type=int, default=8)
    parser.add_argument("--reference_split_count", type=int, default=8)
    parser.add_argument("--max_local_candidates", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def capacity_fraction(
    history_count: int,
    fraction: float,
    sample_count: int,
) -> float:
    floor = fraction if history_count >= 96_000 else 0.06
    standard_error = math.sqrt(
        fraction * (1.0 - fraction) / sample_count
    )
    return min(1.0, max(floor, fraction + 6.0 * standard_error))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rows: list[dict[str, float | int | bool]] = []
    for history_count in sorted(
        {int(value) for value in args.lengths.split(",")}
    ):
        fraction = selected_fraction(history_count)
        sample_count = sample_count_for(fraction)
        threshold_fraction = unbiased_fraction(fraction, sample_count)
        capacity = max(
            1,
            math.ceil(
                capacity_fraction(
                    history_count, fraction, sample_count
                )
                * history_count
            ),
        )
        projected_query = torch.randn(
            1, 8, 4, 128, dtype=torch.float16, device="cuda"
        )
        query_codes, query_scales = varbit_cuda.quantize_projected_query(
            projected_query
        )
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
        metadata, code_count, scale_count = mixed_metadata(
            history_count,
            args.block_size,
            args.hot_fraction,
            LOW_PROFILES[args.low_profile],
        )
        packed_codes = torch.randint(
            0,
            256,
            (code_count,),
            dtype=torch.uint8,
            device="cuda",
        )
        key_scales = torch.rand(
            scale_count,
            dtype=torch.float16,
            device="cuda",
        )

        candidate_indices = torch.empty(
            1, 32, capacity, dtype=torch.long, device="cuda"
        )
        candidate_counts = torch.empty(
            1, 32, dtype=torch.long, device="cuda"
        )
        thresholds = torch.empty(
            1, 32, dtype=torch.float32, device="cuda"
        )
        overflow = torch.empty(
            1, 32, dtype=torch.bool, device="cuda"
        )
        fused_workspace = mixedblock_cuda.allocate_fused_attention_workspace(
            query, args.split_count
        )

        def retrieve() -> tuple[torch.Tensor, ...]:
            return (
                mixedblock_cuda.sampled_threshold_compact_gqa4_indices_out(
                    query_codes,
                    query_scales,
                    packed_codes,
                    key_scales,
                    metadata,
                    candidate_indices,
                    candidate_counts,
                    thresholds,
                    overflow,
                    history_count,
                    sample_count,
                    threshold_fraction,
                )
            )

        def reference() -> torch.Tensor:
            indices, counts, _, _ = retrieve()
            return qabs_cuda.final_attention_ragged_self_split(
                query,
                key,
                value,
                indices,
                counts,
                128.0**-0.5,
                args.reference_split_count,
            )

        def fused() -> torch.Tensor:
            output, _, _, _ = (
                mixedblock_cuda.sampled_threshold_fused_attention_gqa4_out(
                    query_codes,
                    query_scales,
                    query,
                    key,
                    value,
                    packed_codes,
                    key_scales,
                    metadata,
                    fused_workspace,
                    history_count,
                    sample_count,
                    threshold_fraction,
                    128.0**-0.5,
                    args.split_count,
                    args.max_local_candidates,
                )
            )
            return output

        reference_output = reference()
        reference_counts = candidate_counts.clone()
        reference_thresholds = thresholds.clone()
        fused_output = fused()
        fused_counts = fused_workspace["candidate_counts"].clone()
        fused_thresholds = fused_workspace["thresholds"].clone()
        torch.cuda.synchronize()

        iterations = min(
            args.iterations,
            20 if history_count >= 96_000 else args.iterations,
        )
        reference_ms = measure_ms(
            reference, args.warmup, iterations
        )
        fused_ms = measure_ms(fused, args.warmup, iterations)
        retrieval_ms = measure_ms(
            retrieve, args.warmup, iterations
        )
        rows.append(
            {
                "history_count": history_count,
                "selected_fraction": fraction,
                "sample_count": sample_count,
                "candidate_capacity": capacity,
                "split_count": args.split_count,
                "reference_split_count": args.reference_split_count,
                "max_local_candidates": args.max_local_candidates,
                "candidate_count_mean": float(
                    reference_counts.float().mean().item()
                ),
                "candidate_count_max": int(
                    reference_counts.max().item()
                ),
                "candidate_count_max_abs_diff": int(
                    (reference_counts - fused_counts).abs().max().item()
                ),
                "threshold_max_abs_diff": float(
                    (
                        reference_thresholds - fused_thresholds
                    )
                    .abs()
                    .max()
                    .item()
                ),
                "output_max_abs_error": float(
                    (reference_output - fused_output)
                    .abs()
                    .max()
                    .item()
                ),
                "output_mean_abs_error": float(
                    (reference_output - fused_output)
                    .abs()
                    .mean()
                    .item()
                ),
                "reference_overflow": bool(overflow.any().item()),
                "fused_overflow": bool(
                    fused_workspace["overflow"].any().item()
                ),
                "retrieval_ms": retrieval_ms,
                "reference_complete_ms": reference_ms,
                "fused_complete_ms": fused_ms,
                "fusion_speedup": reference_ms / fused_ms,
            }
        )
        del (
            key,
            value,
            packed_codes,
            key_scales,
            candidate_indices,
            fused_workspace,
        )
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
