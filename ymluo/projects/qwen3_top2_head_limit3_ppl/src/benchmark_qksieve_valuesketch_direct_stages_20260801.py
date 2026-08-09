#!/usr/bin/env python
"""Direct CUDA timing for QKSieve with a fused low-rank Value tail.

Each stage and each complete path is invoked and timed independently. Complete
latency is never reconstructed by adding separately measured stage latencies.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import mixedblock_spectral_cuda_20260729 as mixed_cuda
import qabs_cuda_kernels as sparse_cuda
import qksieve_query_cuda_20260728 as query_cuda
import qksieve_valuesketch_cuda_20260801 as sketch_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_variablebit_spectral_attention_20260727 import (
    ALLOCATION_PROFILES,
)
from run_head_top2_targeted_ppl_20260714 import _packed_value_sketch_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths", default="8192,16384,32768,65536,131072"
    )
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--value_block_size", type=int, default=256)
    parser.add_argument("--tail_alpha", type=float, default=0.5)
    parser.add_argument(
        "--allocation_profile",
        choices=sorted(ALLOCATION_PROFILES),
        default="qmse_total_b15",
    )
    parser.add_argument(
        "--recent_counts",
        default="128,256",
        help="Comma-separated fixed-quota recent-token counts to time.",
    )
    parser.add_argument(
        "--selected_count",
        type=int,
        default=0,
        help="Override the length-dependent exact attention count when positive.",
    )
    parser.add_argument(
        "--candidate_capacity_override",
        type=int,
        default=0,
        help="Override candidate workspace capacity when positive.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def measure_ms(
    function: Callable[[], object], warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    stop.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(stop)) / iterations


def measure_wall_ms_once(function: Callable[[], object]) -> tuple[float, object]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start), result


def selected_count(history_tokens: int) -> int:
    return min(
        history_tokens,
        1280,
        max(256, math.ceil(0.06 * history_tokens)),
    )


def sample_count_for(selected_fraction: float) -> int:
    return min(2048, max(256, math.ceil(16.0 / selected_fraction)))


def candidate_capacity(
    history_tokens: int, selected_fraction: float, sample_count: int
) -> int:
    fraction = min(
        1.0,
        max(
            selected_fraction,
            selected_fraction
            + 6.0
            * math.sqrt(
                selected_fraction
                * (1.0 - selected_fraction)
                / sample_count
            ),
        ),
    )
    return min(
        history_tokens,
        math.ceil(fraction * history_tokens),
    )


def gpu_tensor_bytes(mapping: dict[str, object]) -> int:
    return sum(
        int(value.numel()) * int(value.element_size())
        for value in mapping.values()
        if isinstance(value, torch.Tensor) and value.is_cuda
    )


def allocate_outputs(capacity: int, rank: int) -> dict[str, torch.Tensor]:
    return {
        "indices": torch.empty(
            1, 32, capacity, dtype=torch.long, device="cuda"
        ),
        "counts": torch.empty(1, 32, dtype=torch.long, device="cuda"),
        "thresholds": torch.empty(
            1, 32, dtype=torch.float32, device="cuda"
        ),
        "overflow": torch.empty(1, 32, dtype=torch.bool, device="cuda"),
        "tail_denominator": torch.empty(
            1, 32, dtype=torch.float32, device="cuda"
        ),
        "tail_coefficients": torch.empty(
            1, 32, rank, dtype=torch.float32, device="cuda"
        ),
        "selected_denominator": torch.empty(
            1, 32, dtype=torch.float32, device="cuda"
        ),
        "refinement_flags": torch.empty(
            1, 32, dtype=torch.bool, device="cuda"
        ),
    }


def ordinary_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    target_count: int,
    candidate_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    scaling = 128.0**-0.5
    indices = outputs["indices"] if candidate_indices is None else candidate_indices
    if target_count >= 1280:
        split_count = 8 if indices.shape[-1] <= 4096 else 4
        return sparse_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            indices,
            outputs["counts"],
            scaling,
            split_count,
        )
    if target_count >= 900:
        return sparse_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            indices,
            outputs["counts"],
            scaling,
            2,
        )
    return sparse_cuda.final_attention_ragged_self(
        query,
        key,
        value,
        indices,
        outputs["counts"],
        scaling,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    lengths = sorted(
        {int(item) for item in args.lengths.split(",") if item.strip()}
    )
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("lengths must contain positive integers")
    if args.selected_count < 0:
        raise ValueError("selected_count must be non-negative")
    recent_counts = sorted(
        {int(item) for item in args.recent_counts.split(",") if item.strip()}
    )
    if any(count <= 0 for count in recent_counts):
        raise ValueError("recent_counts must contain positive integers")
    torch.manual_seed(args.seed)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    scaling = 128.0**-0.5
    allocation = ALLOCATION_PROFILES[args.allocation_profile].unsqueeze(0).cuda()
    rows: list[dict[str, object]] = []

    for history in lengths:
        target = (
            min(history, args.selected_count)
            if args.selected_count > 0
            else selected_count(history)
        )
        selected_fraction = target / history
        samples = sample_count_for(selected_fraction)
        capacity = candidate_capacity(history, selected_fraction, samples)
        if args.candidate_capacity_override > 0:
            capacity = min(history, args.candidate_capacity_override)
        iterations = min(args.iterations, 20 if history >= 65536 else args.iterations)
        warmup = min(args.warmup, iterations)

        query = torch.randn(1, 32, 128, dtype=dtype, device="cuda")
        grouped_query = query.reshape(1, 8, 4, 128)
        key_basis = torch.randn(
            1, 8, 128, 128, dtype=dtype, device="cuda"
        )
        key = torch.randn(
            1, 8, history + 1, 128, dtype=dtype, device="cuda"
        )
        value = torch.randn_like(key)
        index = varbit_cuda.allocate_packed_index(
            allocation, history + 1, dtype
        )
        index["packed_codes"].random_(0, 256)
        index["key_scales"].uniform_(0.01, 0.1)
        ordinary = allocate_outputs(capacity, 8)
        mean_tail = allocate_outputs(capacity, 1)
        mean_tail["mean"] = value[..., :history, :].float().mean(dim=2).to(dtype)
        mean_tail["basis"] = torch.zeros(
            1, 8, 128, 1, dtype=dtype, device="cuda"
        )

        value_build_state: dict[str, object] = {
            "capacity": history + 1,
            "qk_metric_rebuild_count": 1,
        }
        value_index_build_wall_ms, built_value_index = measure_wall_ms_once(
            lambda: _packed_value_sketch_state(
                value[..., :history, :],
                value_build_state,
                rank=16,
                bits=4,
                block_size=args.value_block_size,
            )
        )
        value_index_build_bytes = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in built_value_index[:5]
        )
        del built_value_index, value_build_state
        warm_value_build_state: dict[str, object] = {
            "capacity": history + 1,
            "qk_metric_rebuild_count": 1,
        }
        value_index_warm_build_wall_ms, warm_built_value_index = (
            measure_wall_ms_once(
                lambda: _packed_value_sketch_state(
                    value[..., :history, :],
                    warm_value_build_state,
                    rank=16,
                    bits=4,
                    block_size=args.value_block_size,
                )
            )
        )
        del warm_built_value_index, warm_value_build_state

        sketches: dict[int, dict[str, torch.Tensor]] = {}
        for rank in (8, 12, 16, 32):
            block_count = math.ceil(history / args.value_block_size)
            sketches[rank] = {
                "packed_codes": torch.randint(
                    0,
                    256,
                    (1, 8, history, rank // 2),
                    dtype=torch.uint8,
                    device="cuda",
                ),
                "minimum": torch.randn(
                    1, 8, block_count, rank, dtype=dtype, device="cuda"
                ),
                "scale": torch.rand(
                    1, 8, block_count, rank, dtype=dtype, device="cuda"
                ).mul_(0.1).add_(0.01),
                "mean": torch.randn(
                    1, 8, 128, dtype=dtype, device="cuda"
                ),
                "basis": torch.randn(
                    1, 8, 128, rank, dtype=dtype, device="cuda"
                ).mul_(0.1),
                **allocate_outputs(capacity, rank),
            }
        value_attention_workspaces = {
            rank: sketch_cuda.allocate_attention_workspace(query, capacity)
            for rank in sketches
        }
        conditional_block_size = 1024
        conditional_rank = 8
        conditional_block_count = math.ceil(
            history / conditional_block_size
        )
        conditional = {
            **allocate_outputs(capacity, conditional_rank),
            "tail_block_denominator": torch.empty(
                1,
                32,
                conditional_block_count,
                dtype=torch.float32,
                device="cuda",
            ),
            "mean_x": torch.randn(
                1,
                8,
                conditional_block_count,
                conditional_rank,
                dtype=dtype,
                device="cuda",
            ).mul_(0.1),
            "mean_v": torch.randn(
                1,
                8,
                conditional_block_count,
                128,
                dtype=dtype,
                device="cuda",
            ).mul_(0.1),
            "linear_map": torch.randn(
                1,
                8,
                128,
                conditional_rank,
                dtype=dtype,
                device="cuda",
            ).mul_(0.1),
        }
        progressive_residuals: dict[float, torch.Tensor] = {}
        for refine_fraction in (0.0, 0.25, 0.375, 0.5, 0.625, 1.0):
            residual = torch.zeros(
                1, 8, dtype=torch.float32, device="cuda"
            )
            refine_kv_heads = int(round(8 * refine_fraction))
            if refine_kv_heads:
                residual[:, :refine_kv_heads] = 1.0
            progressive_residuals[refine_fraction] = residual

        def query_prepare_full() -> tuple[torch.Tensor, torch.Tensor]:
            return query_cuda.project_quantize(grouped_query, key_basis)

        def query_prepare_active() -> tuple[torch.Tensor, torch.Tensor]:
            return query_cuda.project_quantize_active(
                grouped_query, key_basis, allocation
            )

        query_codes, query_scales = query_prepare_active()
        recent_indices = {
            count: torch.arange(
                history - min(count, history - 1),
                history,
                dtype=torch.long,
                device="cuda",
            )
            .reshape(1, 1, -1)
            .expand(1, 32, -1)
            .contiguous()
            for count in recent_counts
            if min(count, history - 1) < capacity
        }

        def ordinary_retrieval() -> tuple[torch.Tensor, ...]:
            return mixed_cuda.plain_sampled_threshold_compact_gqa4_indices_out(
                query_codes,
                query_scales,
                index,
                ordinary["indices"],
                ordinary["counts"],
                ordinary["thresholds"],
                ordinary["overflow"],
                history,
                samples,
                selected_fraction,
            )

        def ordinary_complete() -> torch.Tensor:
            codes, scales = query_prepare_active()
            mixed_cuda.plain_sampled_threshold_compact_gqa4_indices_out(
                codes,
                scales,
                index,
                ordinary["indices"],
                ordinary["counts"],
                ordinary["thresholds"],
                ordinary["overflow"],
                history,
                samples,
                selected_fraction,
            )
            return ordinary_sparse_attention(query, key, value, ordinary, target)

        def recent_quota_merge(recent_count: int) -> torch.Tensor:
            return sparse_cuda.quota_merge_candidates(
                ordinary["indices"], recent_indices[recent_count]
            )

        def recent_complete(recent_count: int) -> torch.Tensor:
            codes, scales = query_prepare_active()
            mixed_cuda.plain_sampled_threshold_compact_gqa4_indices_out(
                codes,
                scales,
                index,
                ordinary["indices"],
                ordinary["counts"],
                ordinary["thresholds"],
                ordinary["overflow"],
                history,
                samples,
                selected_fraction,
            )
            merged = recent_quota_merge(recent_count)
            return ordinary_sparse_attention(
                query,
                key,
                value,
                ordinary,
                target,
                candidate_indices=merged,
            )

        def value_retrieval(rank: int) -> tuple[torch.Tensor, ...]:
            sketch = sketches[rank]
            return mixed_cuda.plain_sampled_threshold_compact_gqa4_valuesketch_out(
                query_codes,
                query_scales,
                index,
                sketch["packed_codes"],
                sketch["minimum"],
                sketch["scale"],
                sketch["indices"],
                sketch["counts"],
                sketch["thresholds"],
                sketch["overflow"],
                sketch["selected_denominator"],
                sketch["tail_denominator"],
                sketch["tail_coefficients"],
                history,
                samples,
                selected_fraction,
                args.value_block_size,
                scaling,
            )

        def mean_tail_retrieval() -> tuple[torch.Tensor, ...]:
            return mixed_cuda.plain_sampled_threshold_compact_gqa4_mass_out(
                query_codes,
                query_scales,
                index,
                mean_tail["indices"],
                mean_tail["counts"],
                mean_tail["thresholds"],
                mean_tail["overflow"],
                mean_tail["selected_denominator"],
                mean_tail["tail_denominator"],
                history,
                samples,
                selected_fraction,
                scaling,
            )

        def mean_tail_exact_combine() -> torch.Tensor:
            return sketch_cuda.exact_selected_plus_tail(
                query,
                key,
                value,
                mean_tail["indices"],
                mean_tail["counts"],
                mean_tail["thresholds"],
                mean_tail["tail_denominator"],
                mean_tail["tail_coefficients"],
                mean_tail["mean"],
                mean_tail["basis"],
                scaling,
                args.tail_alpha,
            )

        def mean_tail_complete() -> torch.Tensor:
            codes, scales = query_prepare_active()
            mixed_cuda.plain_sampled_threshold_compact_gqa4_mass_out(
                codes,
                scales,
                index,
                mean_tail["indices"],
                mean_tail["counts"],
                mean_tail["thresholds"],
                mean_tail["overflow"],
                mean_tail["selected_denominator"],
                mean_tail["tail_denominator"],
                history,
                samples,
                selected_fraction,
                scaling,
            )
            return mean_tail_exact_combine()

        def conditional_retrieval() -> tuple[torch.Tensor, ...]:
            return mixed_cuda.plain_sampled_threshold_compact_gqa4_condtail_out(
                query_codes,
                query_scales,
                index,
                conditional["indices"],
                conditional["counts"],
                conditional["thresholds"],
                conditional["overflow"],
                conditional["selected_denominator"],
                conditional["tail_denominator"],
                conditional["tail_block_denominator"],
                conditional["tail_coefficients"],
                history,
                samples,
                selected_fraction,
                conditional_block_size,
                scaling,
            )

        def conditional_tail_numerator() -> torch.Tensor:
            return sketch_cuda.reduce_conditional_tail_moments(
                conditional["tail_block_denominator"],
                conditional["tail_coefficients"],
                conditional["mean_x"],
                conditional["mean_v"],
                conditional["linear_map"],
            )

        def conditional_exact_combine() -> torch.Tensor:
            return sketch_cuda.exact_selected_plus_conditional_tail(
                query,
                key,
                value,
                conditional["indices"],
                conditional["counts"],
                conditional["thresholds"],
                conditional["tail_denominator"],
                conditional_tail_numerator(),
                scaling,
            )

        def conditional_complete() -> torch.Tensor:
            codes, scales = query_prepare_active()
            mixed_cuda.plain_sampled_threshold_compact_gqa4_condtail_out(
                codes,
                scales,
                index,
                conditional["indices"],
                conditional["counts"],
                conditional["thresholds"],
                conditional["overflow"],
                conditional["selected_denominator"],
                conditional["tail_denominator"],
                conditional["tail_block_denominator"],
                conditional["tail_coefficients"],
                history,
                samples,
                selected_fraction,
                conditional_block_size,
                scaling,
            )
            return conditional_exact_combine()

        def value_exact_combine(rank: int) -> torch.Tensor:
            sketch = sketches[rank]
            return sketch_cuda.exact_selected_plus_tail(
                query,
                key,
                value,
                sketch["indices"],
                sketch["counts"],
                sketch["thresholds"],
                sketch["tail_denominator"],
                sketch["tail_coefficients"],
                sketch["mean"],
                sketch["basis"],
                scaling,
                args.tail_alpha,
            )

        def value_exact_combine_persistent(rank: int) -> torch.Tensor:
            sketch = sketches[rank]
            return sketch_cuda.exact_selected_plus_tail_out(
                query,
                key,
                value,
                sketch["indices"],
                sketch["counts"],
                sketch["thresholds"],
                sketch["tail_denominator"],
                sketch["tail_coefficients"],
                sketch["mean"],
                sketch["basis"],
                value_attention_workspaces[rank],
                scaling,
                args.tail_alpha,
            )

        def value_complete(rank: int) -> torch.Tensor:
            sketch = sketches[rank]
            codes, scales = query_prepare_active()
            mixed_cuda.plain_sampled_threshold_compact_gqa4_valuesketch_out(
                codes,
                scales,
                index,
                sketch["packed_codes"],
                sketch["minimum"],
                sketch["scale"],
                sketch["indices"],
                sketch["counts"],
                sketch["thresholds"],
                sketch["overflow"],
                sketch["selected_denominator"],
                sketch["tail_denominator"],
                sketch["tail_coefficients"],
                history,
                samples,
                selected_fraction,
                args.value_block_size,
                scaling,
            )
            return value_exact_combine(rank)

        def value_complete_persistent(rank: int) -> torch.Tensor:
            sketch = sketches[rank]
            codes, scales = query_prepare_active()
            mixed_cuda.plain_sampled_threshold_compact_gqa4_valuesketch_out(
                codes,
                scales,
                index,
                sketch["packed_codes"],
                sketch["minimum"],
                sketch["scale"],
                sketch["indices"],
                sketch["counts"],
                sketch["thresholds"],
                sketch["overflow"],
                sketch["selected_denominator"],
                sketch["tail_denominator"],
                sketch["tail_coefficients"],
                history,
                samples,
                selected_fraction,
                args.value_block_size,
                scaling,
            )
            return value_exact_combine_persistent(rank)

        def registered_general_complete() -> torch.Tensor:
            if history <= 65536:
                return ordinary_complete()
            return value_complete(16)

        def progressive_retrieval(
            requested_refine_fraction: float,
        ) -> tuple[torch.Tensor, ...]:
            sketch = sketches[32]
            return mixed_cuda.plain_sampled_threshold_compact_gqa4_valuesketch_progressive_out(
                query_codes,
                query_scales,
                index,
                sketch["packed_codes"],
                sketch["minimum"],
                sketch["scale"],
                progressive_residuals[requested_refine_fraction],
                sketch["indices"],
                sketch["counts"],
                sketch["thresholds"],
                sketch["overflow"],
                sketch["selected_denominator"],
                sketch["tail_denominator"],
                sketch["tail_coefficients"],
                sketch["refinement_flags"],
                history,
                samples,
                selected_fraction,
                args.value_block_size,
                scaling,
                0.0,
            )

        def progressive_complete(requested_refine_fraction: float) -> torch.Tensor:
            sketch = sketches[32]
            codes, scales = query_prepare_active()
            mixed_cuda.plain_sampled_threshold_compact_gqa4_valuesketch_progressive_out(
                codes,
                scales,
                index,
                sketch["packed_codes"],
                sketch["minimum"],
                sketch["scale"],
                progressive_residuals[requested_refine_fraction],
                sketch["indices"],
                sketch["counts"],
                sketch["thresholds"],
                sketch["overflow"],
                sketch["selected_denominator"],
                sketch["tail_denominator"],
                sketch["tail_coefficients"],
                sketch["refinement_flags"],
                history,
                samples,
                selected_fraction,
                args.value_block_size,
                scaling,
                0.0,
            )
            return value_exact_combine(32)

        full_key = key.repeat_interleave(4, dim=1)
        full_value = value.repeat_interleave(4, dim=1)

        def full_attention() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2), full_key, full_value, is_causal=False
            )

        ordinary_retrieval()
        mean_tail_retrieval()
        conditional_retrieval()
        value_retrieval(8)
        value_retrieval(16)
        value_retrieval(32)
        torch.cuda.synchronize()
        if (
            bool(ordinary["overflow"].any().item())
            or bool(mean_tail["overflow"].any().item())
        ) or any(
            bool(sketches[rank]["overflow"].any().item())
            for rank in (8, 16, 32)
        ) or bool(conditional["overflow"].any().item()):
            raise RuntimeError(f"candidate output overflow at {history}")

        shared_indices = conditional["indices"][:, ::4, :].contiguous()
        shared_counts = conditional["counts"][:, ::4].contiguous()
        repeated_shared_indices = shared_indices.repeat_interleave(4, dim=1)
        repeated_shared_counts = shared_counts.repeat_interleave(4, dim=1)
        zero_tail_denominator = torch.zeros_like(
            conditional["tail_denominator"]
        )
        zero_tail_numerator = torch.zeros(
            1, 32, 128, dtype=torch.float32, device="cuda"
        )

        def per_head_shared_set_exact() -> torch.Tensor:
            return sketch_cuda.exact_selected_plus_conditional_tail(
                query,
                key,
                value,
                repeated_shared_indices,
                repeated_shared_counts,
                conditional["thresholds"],
                zero_tail_denominator,
                zero_tail_numerator,
                scaling,
            )

        def shared_gqa_exact() -> torch.Tensor:
            return sketch_cuda.exact_shared_gqa_selected_plus_conditional_tail(
                query,
                key,
                value,
                shared_indices,
                shared_counts,
                conditional["thresholds"],
                zero_tail_denominator,
                zero_tail_numerator,
                scaling,
            )

        dense_iterations = min(10, iterations)
        full_ms = measure_ms(full_attention, min(5, warmup), dense_iterations)
        row: dict[str, object] = {
            "history_tokens": history,
            "target_tokens": target,
            "target_fraction": selected_fraction,
            "value_tail_alpha": args.tail_alpha,
            "sample_count": samples,
            "candidate_capacity": capacity,
            "full_attention_ms": full_ms,
            "query_prepare_ms": measure_ms(
                query_prepare_active, warmup, iterations
            ),
            "query_prepare_active_ms": measure_ms(
                query_prepare_active, warmup, iterations
            ),
            "query_prepare_full128_ms": measure_ms(
                query_prepare_full, warmup, iterations
            ),
            "ordinary_retrieval_ms": measure_ms(
                ordinary_retrieval, warmup, iterations
            ),
            "ordinary_exact_sparse_ms": measure_ms(
                lambda: ordinary_sparse_attention(
                    query, key, value, ordinary, target
                ),
                warmup,
                iterations,
            ),
            "ordinary_complete_ms": measure_ms(
                ordinary_complete, warmup, iterations
            ),
            "ordinary_complete_speedup": 0.0,
            "key_index_bytes": gpu_tensor_bytes(index),
            "full_kv_bytes": int(key.numel() + value.numel())
            * key.element_size(),
            "value_r16_i4_first_index_build_wall_ms": value_index_build_wall_ms,
            "value_r16_i4_warm_index_build_wall_ms": (
                value_index_warm_build_wall_ms
            ),
            "value_r16_i4_first_index_build_bytes": value_index_build_bytes,
        }
        row["ordinary_complete_speedup"] = full_ms / float(
            row["ordinary_complete_ms"]
        )
        mean_tail_scan_ms = measure_ms(
            mean_tail_retrieval, warmup, iterations
        )
        mean_tail_combine_ms = measure_ms(
            mean_tail_exact_combine, warmup, iterations
        )
        mean_tail_complete_ms = measure_ms(
            mean_tail_complete, warmup, iterations
        )
        mean_tail_constant_bytes = sum(
            int(mean_tail[name].numel()) * int(mean_tail[name].element_size())
            for name in ("mean", "basis")
        )
        row.update(
            {
                "mean_tail_value_mean_build_ms": measure_ms(
                    lambda: value[..., :history, :].float().mean(dim=2),
                    min(5, warmup),
                    min(10, iterations),
                ),
                "mean_tail_scan_ms": mean_tail_scan_ms,
                "mean_tail_exact_combine_ms": mean_tail_combine_ms,
                "mean_tail_complete_ms": mean_tail_complete_ms,
                "mean_tail_complete_speedup": full_ms / mean_tail_complete_ms,
                "mean_tail_candidate_count_mean": float(
                    mean_tail["counts"].float().mean().item()
                ),
                "mean_tail_constant_bytes": mean_tail_constant_bytes,
                "mean_tail_total_index_ratio_of_full_kv": (
                    int(row["key_index_bytes"]) + mean_tail_constant_bytes
                )
                / int(row["full_kv_bytes"]),
            }
        )
        conditional_scan_ms = measure_ms(
            conditional_retrieval, warmup, iterations
        )
        conditional_reduce_ms = measure_ms(
            conditional_tail_numerator, warmup, iterations
        )
        conditional_combine_ms = measure_ms(
            conditional_exact_combine, warmup, iterations
        )
        per_head_shared_set_exact_ms = measure_ms(
            per_head_shared_set_exact, warmup, iterations
        )
        shared_gqa_exact_ms = measure_ms(
            shared_gqa_exact, warmup, iterations
        )
        conditional_complete_ms = measure_ms(
            conditional_complete, warmup, iterations
        )
        conditional_index_bytes = sum(
            int(conditional[name].numel())
            * int(conditional[name].element_size())
            for name in ("mean_x", "mean_v", "linear_map")
        )
        row.update(
            {
                "condtail_r8_b1024_scan_ms": conditional_scan_ms,
                "condtail_r8_b1024_tail_reduce_ms": conditional_reduce_ms,
                "condtail_r8_b1024_exact_combine_ms": (
                    conditional_combine_ms
                ),
                "per_head_shared_set_exact_ms": per_head_shared_set_exact_ms,
                "shared_gqa_exact_ms": shared_gqa_exact_ms,
                "shared_gqa_exact_speedup_over_per_head": (
                    per_head_shared_set_exact_ms / shared_gqa_exact_ms
                ),
                "condtail_r8_b1024_complete_ms": conditional_complete_ms,
                "condtail_r8_b1024_complete_speedup": (
                    full_ms / conditional_complete_ms
                ),
                "condtail_r8_b1024_candidate_count_mean": float(
                    conditional["counts"].float().mean().item()
                ),
                "condtail_r8_b1024_value_index_bytes": (
                    conditional_index_bytes
                ),
                "condtail_r8_b1024_total_index_ratio_of_full_kv": (
                    int(row["key_index_bytes"]) + conditional_index_bytes
                )
                / int(row["full_kv_bytes"]),
            }
        )
        for recent_count in recent_indices:
            prefix = f"recent_quota_{recent_count}"
            merge_ms = measure_ms(
                lambda count=recent_count: recent_quota_merge(count),
                warmup,
                iterations,
            )
            complete_ms = measure_ms(
                lambda count=recent_count: recent_complete(count),
                warmup,
                iterations,
            )
            row.update(
                {
                    f"{prefix}_merge_ms": merge_ms,
                    f"{prefix}_complete_ms": complete_ms,
                    f"{prefix}_complete_speedup": full_ms / complete_ms,
                }
            )
        for rank in (8, 12, 16, 32):
            prefix = f"value_r{rank}_i4"
            scan_ms = measure_ms(
                lambda rank=rank: value_retrieval(rank), warmup, iterations
            )
            combine_ms = measure_ms(
                lambda rank=rank: value_exact_combine(rank), warmup, iterations
            )
            persistent_combine_ms = measure_ms(
                lambda rank=rank: value_exact_combine_persistent(rank),
                warmup,
                iterations,
            )
            complete_ms = measure_ms(
                lambda rank=rank: value_complete(rank), warmup, iterations
            )
            persistent_complete_ms = measure_ms(
                lambda rank=rank: value_complete_persistent(rank),
                warmup,
                iterations,
            )
            reference_output = value_exact_combine(rank)
            persistent_output = value_exact_combine_persistent(rank)
            torch.cuda.synchronize()
            persistent_max_abs_error = float(
                (reference_output.float() - persistent_output.float())
                .abs()
                .max()
                .item()
            )
            sketch = sketches[rank]
            sketch_index_bytes = sum(
                int(sketch[name].numel()) * int(sketch[name].element_size())
                for name in ("packed_codes", "minimum", "scale", "mean", "basis")
            )
            row.update(
                {
                    f"{prefix}_tail_scan_ms": scan_ms,
                    f"{prefix}_exact_combine_ms": combine_ms,
                    f"{prefix}_persistent_exact_combine_ms": (
                        persistent_combine_ms
                    ),
                    f"{prefix}_persistent_combine_speedup": (
                        combine_ms / persistent_combine_ms
                    ),
                    f"{prefix}_complete_ms": complete_ms,
                    f"{prefix}_complete_speedup": full_ms / complete_ms,
                    f"{prefix}_persistent_complete_ms": persistent_complete_ms,
                    f"{prefix}_persistent_complete_speedup": (
                        full_ms / persistent_complete_ms
                    ),
                    f"{prefix}_persistent_path_speedup": (
                        complete_ms / persistent_complete_ms
                    ),
                    f"{prefix}_persistent_max_abs_error": (
                        persistent_max_abs_error
                    ),
                    f"{prefix}_candidate_count_mean": float(
                        sketch["counts"].float().mean().item()
                    ),
                    f"{prefix}_value_index_bytes": sketch_index_bytes,
                    f"{prefix}_total_index_ratio_of_full_kv": (
                        int(row["key_index_bytes"]) + sketch_index_bytes
                    )
                    / int(row["full_kv_bytes"]),
                }
            )
        general_complete_ms = measure_ms(
            registered_general_complete, warmup, iterations
        )
        row.update(
            {
                "general_profile_branch": (
                    "fast"
                    if history <= 65536
                    else f"robust_r16_i4_alpha{args.tail_alpha:g}"
                ),
                "general_profile_complete_ms": general_complete_ms,
                "general_profile_complete_speedup": full_ms / general_complete_ms,
            }
        )
        for requested_refine_fraction in (0.0, 0.25, 0.375, 0.5, 0.625, 1.0):
            prefix = f"progressive_r8to32_f{requested_refine_fraction:g}"
            progressive_retrieval(requested_refine_fraction)
            torch.cuda.synchronize()
            realized_refine_fraction = float(
                sketches[32]["refinement_flags"].float().mean().item()
            )
            complete_ms = measure_ms(
                lambda fraction=requested_refine_fraction: progressive_complete(
                    fraction
                ),
                warmup,
                iterations,
            )
            row.update(
                {
                    f"{prefix}_realized_refine_fraction": (
                        realized_refine_fraction
                    ),
                    f"{prefix}_complete_ms": complete_ms,
                    f"{prefix}_complete_speedup": full_ms / complete_ms,
                }
            )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del (
            full_key,
            full_value,
            key,
            value,
            index,
            sketches,
            value_attention_workspaces,
            conditional,
            ordinary,
        )
        torch.cuda.empty_cache()

    result = {
        "schema": "qksieve_valuesketch_direct_stages_v1",
        "device": torch.cuda.get_device_name(),
        "dtype": args.dtype,
        "allocation_profile": args.allocation_profile,
        "timing_contract": {
            "independent_direct_invocation": True,
            "complete_path_measured_directly": True,
            "complete_path_not_reconstructed_from_stage_sum": True,
            "scope": "one Llama-3.1-8B attention layer, batch=1, decode query=1",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
