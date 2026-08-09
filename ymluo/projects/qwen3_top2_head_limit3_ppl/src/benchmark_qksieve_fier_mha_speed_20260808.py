#!/usr/bin/env python
"""MHA speed benchmark for Full Attention, QKSieve, and FIER.

The tensor layout matches LLaMA-2 7B MHA: 32 query heads, 32 KV heads,
and a head dimension of 128. Full Attention consumes native MHA K/V without
GQA replication. All sparse paths reuse the same resident FP16 K/V tensors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import benchmark_qksieve_mixedblock_cuda_20260729 as mixed_bench
import fier_rtn1_cuda_20260728 as fier_cuda
import mixedblock_spectral_cuda_20260729 as mixed_cuda
import qabs_cuda_kernels as sparse_cuda
import qksieve_query_cuda_20260728 as query_cuda
import qksieve_valuesketch_cuda_20260801 as valuesketch_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda


QUERY_HEADS = 32
KV_HEADS = 32
HEAD_DIM = 128
QUERY_GROUPS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="8192,16384,32768,65536,131072")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--hot_fraction", type=float, default=0.15)
    parser.add_argument("--sample_count", type=int, default=0)
    parser.add_argument("--max_sample_count", type=int, default=0)
    parser.add_argument("--qksieve_split_count", type=int, default=8)
    parser.add_argument("--fier_split_count", type=int, default=8)
    parser.add_argument("--value_tail_alpha", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def measure_ms(
    function: Callable[[], object], warmup: int, iterations: int
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


def threshold_fraction(fraction: float, sample_count: int) -> float:
    rank = max(1, min(sample_count, int(round(fraction * (sample_count + 1)))))
    return (rank - 0.5) / sample_count


def sample_count_for(fraction: float) -> int:
    return min(2048, max(256, math.ceil(16.0 / fraction)))


def capacity_for(history_count: int, fraction: float) -> int:
    samples = sample_count_for(fraction)
    capacity_fraction = min(
        1.0,
        max(
            0.06,
            fraction
            + 6.0 * math.sqrt(fraction * (1.0 - fraction) / samples),
        ),
    )
    return min(history_count, max(1, math.ceil(capacity_fraction * history_count)))


def repeat_head_allocations(allocations: torch.Tensor) -> torch.Tensor:
    if allocations.shape[0] != 8:
        raise ValueError("expected the learned eight-head allocation template")
    return allocations.repeat(KV_HEADS // 8, 1)


def mixed_metadata_mha(
    history_count: int,
    block_size: int,
    hot_fraction: float,
) -> tuple[dict[str, torch.Tensor | int], int, int]:
    low = repeat_head_allocations(mixed_bench.LOW_PROFILES["fixed84"])
    high = repeat_head_allocations(mixed_bench.HIGH_ALLOCATIONS)
    allocations = torch.stack((low, high), dim=0)
    profile = mixed_bench.profile_metadata(allocations)
    block_count = math.ceil(history_count / block_size)
    hot_blocks = max(1, math.ceil(block_count * hot_fraction))
    block_profiles = torch.zeros(KV_HEADS, block_count, dtype=torch.uint8)
    generator = torch.Generator().manual_seed(20260808)
    for head in range(KV_HEADS):
        selected = torch.randperm(block_count, generator=generator)[:hot_blocks]
        block_profiles[head, selected] = 1

    cumulative_hot = block_profiles.to(torch.int32).cumsum(dim=1)
    if int(cumulative_hot.max().item()) > torch.iinfo(torch.int16).max:
        raise ValueError("hot-block prefix exceeds int16 capacity")
    block_hot_prefix = torch.cat(
        (
            torch.zeros(KV_HEADS, 1, dtype=torch.int16),
            cumulative_hot.to(torch.int16),
        ),
        dim=1,
    )
    head_code_bases = torch.empty(KV_HEADS, dtype=torch.int64)
    head_scale_bases = torch.empty(KV_HEADS, dtype=torch.int64)
    code_cursor = 0
    scale_cursor = 0
    for head in range(KV_HEADS):
        head_code_bases[head] = code_cursor
        head_scale_bases[head] = scale_cursor
        for block in range(block_count):
            profile_id = int(block_profiles[head, block].item())
            code_cursor += block_size * int(
                profile["code_strides"][profile_id, head].item()
            )
            scale_cursor += block_size * int(
                profile["scale_strides"][profile_id, head].item()
            )

    metadata: dict[str, torch.Tensor | int] = {
        "bit_allocations": allocations.cuda(),
        "code_offsets": profile["code_offsets"].cuda(),
        "scale_offsets": profile["scale_offsets"].cuda(),
        "code_strides": profile["code_strides"].cuda(),
        "scale_strides": profile["scale_strides"].cuda(),
        "block_hot_prefix": block_hot_prefix.cuda(),
        "head_code_bases": head_code_bases.cuda(),
        "head_scale_bases": head_scale_bases.cuda(),
        "block_size": block_size,
    }
    return metadata, code_cursor, scale_cursor


def plain_metadata_mha(
    history_count: int,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Build the per-head plain layout used by the real decode path."""
    allocations = repeat_head_allocations(
        mixed_bench.LOW_PROFILES["fixed84"]
    )
    profile = mixed_bench.profile_metadata(allocations.unsqueeze(0))
    code_offsets = profile["code_offsets"][0].contiguous()
    scale_offsets = profile["scale_offsets"][0].contiguous()
    code_strides = profile["code_strides"][0].contiguous()
    scale_strides = profile["scale_strides"][0].contiguous()
    code_bases = torch.empty(KV_HEADS, dtype=torch.int64)
    scale_bases = torch.empty(KV_HEADS, dtype=torch.int64)
    code_cursor = 0
    scale_cursor = 0
    for head in range(KV_HEADS):
        code_bases[head] = code_cursor
        scale_bases[head] = scale_cursor
        code_cursor += history_count * int(code_strides[head].item())
        scale_cursor += history_count * int(scale_strides[head].item())
    return (
        {
            "bit_allocations": allocations.unsqueeze(0).cuda(),
            "code_offsets": code_offsets.unsqueeze(0).cuda(),
            "scale_offsets": scale_offsets.unsqueeze(0).cuda(),
            "code_strides": code_strides.unsqueeze(0).cuda(),
            "scale_strides": scale_strides.unsqueeze(0).cuda(),
            "code_bases": code_bases.unsqueeze(0).cuda(),
            "scale_bases": scale_bases.unsqueeze(0).cuda(),
        },
        code_cursor,
        scale_cursor,
    )


def allocate_qksieve_outputs(capacity: int) -> tuple[torch.Tensor, ...]:
    shape = (1, QUERY_HEADS, capacity)
    return (
        torch.empty(shape, dtype=torch.long, device="cuda"),
        torch.empty(shape, dtype=torch.float32, device="cuda"),
        torch.empty(1, QUERY_HEADS, dtype=torch.long, device="cuda"),
        torch.empty(1, QUERY_HEADS, dtype=torch.float32, device="cuda"),
        torch.empty(1, QUERY_HEADS, dtype=torch.bool, device="cuda"),
    )


def allocate_fier_outputs(capacity: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty(1, QUERY_HEADS, capacity, dtype=torch.long, device="cuda"),
        torch.empty(1, QUERY_HEADS, dtype=torch.long, device="cuda"),
        torch.empty(1, QUERY_HEADS, dtype=torch.float32, device="cuda"),
        torch.empty(1, QUERY_HEADS, dtype=torch.bool, device="cuda"),
    )


def sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    split_count: int,
) -> torch.Tensor:
    return sparse_cuda.final_attention_ragged_self_split(
        query,
        key,
        value,
        indices,
        counts,
        HEAD_DIM**-0.5,
        split_count,
    )


def bytes_of(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    lengths = sorted({int(item) for item in args.lengths.split(",") if item})
    rows: list[dict[str, object]] = []

    for history_count in lengths:
        selected_fraction = mixed_bench.selected_fraction(history_count)
        default_sample_count = sample_count_for(selected_fraction)
        if args.sample_count > 0:
            sample_count = args.sample_count
        elif args.max_sample_count > 0:
            sample_count = min(default_sample_count, args.max_sample_count)
        else:
            sample_count = default_sample_count
        selected_threshold = threshold_fraction(selected_fraction, sample_count)
        capacity = capacity_for(history_count, selected_fraction)

        query = torch.randn(
            1, QUERY_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda"
        )
        grouped_query = query.reshape(
            1, KV_HEADS, QUERY_GROUPS, HEAD_DIM
        )
        query_basis = torch.randn(
            1,
            KV_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float16,
            device="cuda",
        )
        key = torch.randn(
            1,
            KV_HEADS,
            history_count,
            HEAD_DIM,
            dtype=torch.float16,
            device="cuda",
        )
        value = torch.randn_like(key)

        projected_query = torch.einsum(
            "bhgd,bhdm->bhgm", grouped_query, query_basis
        )
        query_codes, query_scales = varbit_cuda.quantize_projected_query(
            projected_query
        )
        q_metadata, q_code_count, q_scale_count = plain_metadata_mha(
            history_count
        )
        q_packed_codes = torch.randint(
            0,
            256,
            (q_code_count,),
            dtype=torch.uint8,
            device="cuda",
        )
        q_key_scales = torch.rand(
            q_scale_count, dtype=torch.float16, device="cuda"
        )
        q_outputs = allocate_qksieve_outputs(capacity)
        q_value_outputs = allocate_qksieve_outputs(capacity)
        value_rank = 16
        value_block_size = 256
        value_block_count = math.ceil(history_count / value_block_size)
        packed_value_codes = torch.randint(
            0,
            256,
            (1, KV_HEADS, history_count, value_rank // 2),
            dtype=torch.uint8,
            device="cuda",
        )
        value_minimum = torch.randn(
            1,
            KV_HEADS,
            value_block_count,
            value_rank,
            dtype=torch.float16,
            device="cuda",
        )
        value_scale = torch.rand_like(value_minimum).mul_(0.1).add_(1.0e-3)
        value_mean = torch.randn(
            1, KV_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda"
        )
        value_basis = torch.randn(
            1,
            KV_HEADS,
            HEAD_DIM,
            value_rank,
            dtype=torch.float16,
            device="cuda",
        ).contiguous()
        selected_denominator = torch.empty(
            1, QUERY_HEADS, dtype=torch.float32, device="cuda"
        )
        tail_denominator = torch.empty_like(selected_denominator)
        tail_coefficients = torch.empty(
            1,
            QUERY_HEADS,
            value_rank,
            dtype=torch.float32,
            device="cuda",
        )
        value_attention_workspace = valuesketch_cuda.allocate_attention_workspace(
            query, capacity
        )
        q_packed_index = {
            "packed_codes": q_packed_codes,
            "key_scales": q_key_scales,
            **q_metadata,
        }

        fier_index = fier_cuda.allocate_packed_index(
            1, KV_HEADS, history_count, key.device
        )
        fier_cuda.update_packed_index(key, fier_index, history_count)
        fier_outputs = allocate_fier_outputs(capacity)

        def q_prepare() -> tuple[torch.Tensor, torch.Tensor]:
            return query_cuda.project_quantize(grouped_query, query_basis)

        def q_retrieve_prepared() -> tuple[torch.Tensor, ...]:
            return mixed_cuda.plain_sampled_threshold_compact_gqa4_indices_out(
                query_codes,
                query_scales,
                q_packed_index,
                q_outputs[0],
                q_outputs[2],
                q_outputs[3],
                q_outputs[4],
                history_count,
                sample_count,
                selected_threshold,
            )

        def q_retrieve_complete() -> tuple[torch.Tensor, ...]:
            codes, scales = q_prepare()
            return mixed_cuda.plain_sampled_threshold_compact_gqa4_indices_out(
                codes,
                scales,
                q_packed_index,
                q_outputs[0],
                q_outputs[2],
                q_outputs[3],
                q_outputs[4],
                history_count,
                sample_count,
                selected_threshold,
            )

        def q_sparse_attention() -> torch.Tensor:
            return sparse_attention(
                query,
                key,
                value,
                q_outputs[0],
                q_outputs[2],
                args.qksieve_split_count,
            )

        def q_valuesketch_retrieve_prepared() -> tuple[torch.Tensor, ...]:
            return mixed_cuda.plain_sampled_threshold_compact_gqa4_valuesketch_out(
                query_codes,
                query_scales,
                q_packed_index,
                packed_value_codes,
                value_minimum,
                value_scale,
                q_value_outputs[0],
                q_value_outputs[2],
                q_value_outputs[3],
                q_value_outputs[4],
                selected_denominator,
                tail_denominator,
                tail_coefficients,
                history_count,
                sample_count,
                selected_threshold,
                value_block_size,
                HEAD_DIM**-0.5,
            )

        def q_valuesketch_retrieve_complete() -> tuple[torch.Tensor, ...]:
            codes, scales = q_prepare()
            return mixed_cuda.plain_sampled_threshold_compact_gqa4_valuesketch_out(
                codes,
                scales,
                q_packed_index,
                packed_value_codes,
                value_minimum,
                value_scale,
                q_value_outputs[0],
                q_value_outputs[2],
                q_value_outputs[3],
                q_value_outputs[4],
                selected_denominator,
                tail_denominator,
                tail_coefficients,
                history_count,
                sample_count,
                selected_threshold,
                value_block_size,
                HEAD_DIM**-0.5,
            )

        def q_valuesketch_attention() -> torch.Tensor:
            return valuesketch_cuda.exact_selected_plus_tail_out(
                query,
                key,
                value,
                q_value_outputs[0],
                q_value_outputs[2],
                q_value_outputs[3],
                tail_denominator,
                tail_coefficients,
                value_mean,
                value_basis,
                value_attention_workspace,
                HEAD_DIM**-0.5,
                args.value_tail_alpha,
            )

        def q_valuesketch_complete() -> torch.Tensor:
            q_valuesketch_retrieve_complete()
            return q_valuesketch_attention()

        def q_complete() -> torch.Tensor:
            q_retrieve_complete()
            return q_sparse_attention()

        def fier_retrieve() -> tuple[torch.Tensor, ...]:
            return fier_cuda.sampled_threshold_compact_out(
                query,
                fier_index,
                *fier_outputs,
                history_count,
                sample_count,
                selected_threshold,
            )

        def fier_sparse_attention() -> torch.Tensor:
            return sparse_attention(
                query,
                key,
                value,
                fier_outputs[0],
                fier_outputs[1],
                args.fier_split_count,
            )

        def fier_complete() -> torch.Tensor:
            fier_retrieve()
            return fier_sparse_attention()

        def full_attention() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2), key, value
            )

        # Candidate equivalence is checked with one shared, immutable Query
        # encoding so the A/B changes only the Value-tail computation.
        q_retrieve_prepared()
        q_valuesketch_retrieve_prepared()
        fier_retrieve()
        if (
            bool(q_outputs[4].any().item())
            or bool(q_value_outputs[4].any().item())
            or bool(fier_outputs[3].any().item())
        ):
            raise RuntimeError("candidate buffer overflowed")
        q_counts_equal = bool(torch.equal(q_outputs[2], q_value_outputs[2]))
        q_count_max_abs_diff = int(
            (q_outputs[2] - q_value_outputs[2]).abs().max().item()
        )
        q_threshold_max_abs_diff = float(
            (q_outputs[3] - q_value_outputs[3]).abs().max().item()
        )
        q_candidate_sets_equal = q_counts_equal
        if q_candidate_sets_equal:
            for head in range(QUERY_HEADS):
                count = int(q_outputs[2][0, head].item())
                left = torch.sort(q_outputs[0][0, head, :count]).values
                right = torch.sort(q_value_outputs[0][0, head, :count]).values
                if not torch.equal(left, right):
                    q_candidate_sets_equal = False
                    break

        q_split8_reference = sparse_attention(
            query,
            key,
            value,
            q_outputs[0],
            q_outputs[2],
            8,
        ).clone()
        q_selected_split_output = q_sparse_attention().clone()
        torch.cuda.synchronize()
        split_difference = (
            q_split8_reference.float() - q_selected_split_output.float()
        ).abs()
        split_output_max_abs_error = float(split_difference.max().item())
        split_output_cosine = float(
            F.cosine_similarity(
                q_split8_reference.float().reshape(1, -1),
                q_selected_split_output.float().reshape(1, -1),
            ).item()
        )

        iterations = min(args.iterations, 20 if history_count >= 65536 else args.iterations)
        warmup = min(args.warmup, iterations)
        q_prepare_ms = measure_ms(q_prepare, warmup, iterations)
        q_scan_ms = measure_ms(q_retrieve_prepared, warmup, iterations)
        q_sparse_ms = measure_ms(q_sparse_attention, warmup, iterations)
        q_complete_ms = measure_ms(q_complete, warmup, iterations)
        q_valuesketch_scan_ms = measure_ms(
            q_valuesketch_retrieve_prepared, warmup, iterations
        )
        q_valuesketch_attention_ms = measure_ms(
            q_valuesketch_attention, warmup, iterations
        )
        q_valuesketch_complete_ms = measure_ms(
            q_valuesketch_complete, warmup, iterations
        )
        fier_scan_ms = measure_ms(fier_retrieve, warmup, iterations)
        fier_sparse_ms = measure_ms(fier_sparse_attention, warmup, iterations)
        fier_complete_ms = measure_ms(fier_complete, warmup, iterations)
        full_ms = measure_ms(
            full_attention, min(5, warmup), min(10, iterations)
        )

        full_kv_bytes = bytes_of(key) + bytes_of(value)
        q_index_bytes = (
            bytes_of(q_packed_codes)
            + bytes_of(q_key_scales)
            + sum(bytes_of(tensor) for tensor in q_metadata.values())
        )
        q_valuesketch_bytes = (
            bytes_of(packed_value_codes)
            + bytes_of(value_minimum)
            + bytes_of(value_scale)
            + bytes_of(value_mean)
            + bytes_of(value_basis)
        )
        q_actual_fraction = float(
            (q_outputs[2].float() / history_count).mean().item()
        )
        fier_actual_fraction = float(
            (fier_outputs[1].float() / history_count).mean().item()
        )
        row: dict[str, object] = {
            "history_count": history_count,
            "layout": "MHA_32Q_32KV_D128",
            "target_fraction": selected_fraction,
            "target_tokens": selected_fraction * history_count,
            "sample_count": sample_count,
            "qksieve_split_count": args.qksieve_split_count,
            "fier_split_count": args.fier_split_count,
            "qksieve_actual_fraction": q_actual_fraction,
            "fier_actual_fraction": fier_actual_fraction,
            "split8_selected_output_max_abs_error": split_output_max_abs_error,
            "split8_selected_output_cosine": split_output_cosine,
            "qksieve_query_prepare_ms": q_prepare_ms,
            "qksieve_selector_scan_ms": q_scan_ms,
            "qksieve_sparse_attention_ms": q_sparse_ms,
            "qksieve_complete_ms": q_complete_ms,
            "qksieve_valuesketch_candidate_counts_equal": q_counts_equal,
            "qksieve_valuesketch_candidate_sets_equal": q_candidate_sets_equal,
            "qksieve_valuesketch_tail_alpha": args.value_tail_alpha,
            "qksieve_valuesketch_count_max_abs_diff": q_count_max_abs_diff,
            "qksieve_valuesketch_threshold_max_abs_diff": (
                q_threshold_max_abs_diff
            ),
            "qksieve_valuesketch_scan_ms": q_valuesketch_scan_ms,
            "qksieve_valuesketch_attention_ms": q_valuesketch_attention_ms,
            "qksieve_valuesketch_complete_ms": q_valuesketch_complete_ms,
            "fier_selector_scan_ms": fier_scan_ms,
            "fier_sparse_attention_ms": fier_sparse_ms,
            "fier_complete_ms": fier_complete_ms,
            "full_mha_sdpa_ms": full_ms,
            "qksieve_attention_speedup": full_ms / q_complete_ms,
            "qksieve_valuesketch_attention_speedup": (
                full_ms / q_valuesketch_complete_ms
            ),
            "qksieve_valuesketch_slowdown_vs_no_value": (
                q_valuesketch_complete_ms / q_complete_ms
            ),
            "fier_attention_speedup": full_ms / fier_complete_ms,
            "qksieve_vs_fier_speedup": fier_complete_ms / q_complete_ms,
            "qksieve_index_bytes": q_index_bytes,
            "qksieve_valuesketch_bytes": q_valuesketch_bytes,
            "fier_index_bytes": fier_cuda.allocated_bytes(fier_index),
            "full_kv_bytes": full_kv_bytes,
            "qksieve_index_ratio_of_full_kv": q_index_bytes / full_kv_bytes,
            "qksieve_total_auxiliary_ratio_with_valuesketch": (
                (q_index_bytes + q_valuesketch_bytes) / full_kv_bytes
            ),
            "fier_index_ratio_of_full_kv": (
                fier_cuda.allocated_bytes(fier_index) / full_kv_bytes
            ),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        del (
            query,
            grouped_query,
            query_basis,
            key,
            value,
            q_packed_codes,
            q_key_scales,
            q_outputs,
            q_value_outputs,
            packed_value_codes,
            value_minimum,
            value_scale,
            value_mean,
            value_basis,
            selected_denominator,
            tail_denominator,
            tail_coefficients,
            value_attention_workspace,
            fier_index,
            fier_outputs,
        )
        torch.cuda.empty_cache()

    payload = {
        "benchmark": "QKSieve/FIER native-MHA attention speed",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "qksieve_valuesketch_tail_alpha": args.value_tail_alpha,
        "rows": rows,
        "claim_boundary": (
            "Synthetic MHA-shaped layer benchmark with resident FP16 K/V; "
            "it measures attention-path speed, not whole-model decode quality."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
