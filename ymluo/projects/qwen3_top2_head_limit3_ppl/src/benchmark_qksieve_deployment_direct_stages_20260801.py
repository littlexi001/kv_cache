#!/usr/bin/env python
"""Direct CUDA timing for the deployed sampled-quantile QKSieve path.

Every field is measured by invoking that CUDA path directly.  The complete
attention speedup is never reconstructed from separately measured stages.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

import mixedblock_spectral_cuda_20260729 as mixed_cuda
import qabs_cuda_kernels as sparse_cuda
import qksieve_mass_cuda_20260801 as mass_cuda
import qksieve_query_cuda_20260728 as query_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_variablebit_spectral_attention_20260727 import (
    ALLOCATION_PROFILES,
)
PROFILES = {
    "fixed200_b48": torch.tensor(
        [[2, 0, 0, 0, 0, 0, 0, 0]] * 8, dtype=torch.int8
    ),
    "fixed110_b64": torch.tensor(
        [[1, 1, 0, 0, 0, 0, 0, 0]] * 8, dtype=torch.int8
    ),
    "fixed210_b80": torch.tensor(
        [[2, 1, 0, 0, 0, 0, 0, 0]] * 8, dtype=torch.int8
    ),
    "fixed400_b80": torch.tensor(
        [[4, 0, 0, 0, 0, 0, 0, 0]] * 8, dtype=torch.int8
    ),
    "fixed410_b112": torch.tensor(
        [[4, 1, 0, 0, 0, 0, 0, 0]] * 8, dtype=torch.int8
    ),
    "fixed4221_b208": torch.tensor(
        [[4, 2, 2, 1, 0, 0, 0, 0]] * 8, dtype=torch.int8
    ),
    "fixed4421_b240": torch.tensor(
        [[4, 4, 2, 1, 0, 0, 0, 0]] * 8, dtype=torch.int8
    ),
    "auto240_reference": ALLOCATION_PROFILES["qmse_total_b15"],
}

PROFILE_LOGICAL_BITS = {
    "fixed200_b48": 48,
    "fixed110_b64": 64,
    "fixed210_b80": 80,
    "fixed400_b80": 80,
    "fixed410_b112": 112,
    "fixed4221_b208": 208,
    "fixed4421_b240": 240,
    "auto240_reference": 240,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths", default="8192,16384,32768,65536,131072"
    )
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default="auto240_reference"
    )
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--index_chunk_tokens", type=int, default=4096)
    parser.add_argument("--tail_resolution_target", type=int, default=16)
    parser.add_argument("--capacity_sigma", type=float, default=6.0)
    parser.add_argument(
        "--minimum_capacity_fraction", type=float, default=0.0
    )
    parser.add_argument("--sample_mass_correction", action="store_true")
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


def selected_count(history_tokens: int) -> int:
    return min(
        history_tokens,
        1280,
        max(256, math.ceil(0.06 * history_tokens)),
    )


def sample_count_for(
    selected_fraction: float,
    tail_resolution_target: int = 16,
) -> int:
    if tail_resolution_target <= 0:
        raise ValueError("tail_resolution_target must be positive")
    return min(
        2048,
        max(256, math.ceil(tail_resolution_target / selected_fraction)),
    )


def selected_sample_count(
    selected_fraction: float,
    sample_count: int,
) -> int:
    return max(
        1,
        min(
            sample_count,
            int(round(selected_fraction * (sample_count + 1))),
        ),
    )


def threshold_fraction(
    selected_fraction: float, sample_count: int
) -> float:
    rank = max(
        1,
        min(
            sample_count,
            int(round(selected_fraction * (sample_count + 1))),
        ),
    )
    return (rank - 0.5) / sample_count


def candidate_capacity(
    history_tokens: int,
    selected_fraction: float,
    sample_count: int,
    sigma: float = 6.0,
    minimum_fraction: float = 0.0,
) -> int:
    if sigma < 0.0:
        raise ValueError("capacity sigma must be non-negative")
    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError("minimum capacity fraction must lie in [0, 1]")
    capacity_fraction = min(
        1.0,
        max(
            minimum_fraction,
            selected_fraction
            + sigma
            * math.sqrt(
                selected_fraction
                * (1.0 - selected_fraction)
                / sample_count
            ),
        ),
    )
    return min(
        history_tokens,
        max(1, math.ceil(capacity_fraction * history_tokens)),
    )


def allocate_outputs(
    capacity: int,
    *,
    collect_proxy_mass: bool,
) -> tuple[torch.Tensor, ...]:
    threshold_shape = (2, 1, 32) if collect_proxy_mass else (1, 32)
    return (
        torch.empty(
            1, 32, capacity, dtype=torch.long, device="cuda"
        ),
        torch.empty(1, 32, dtype=torch.long, device="cuda"),
        torch.empty(
            threshold_shape, dtype=torch.float32, device="cuda"
        ),
        torch.empty(1, 32, dtype=torch.bool, device="cuda"),
    )


def sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
    target_count: int,
) -> torch.Tensor:
    indices, counts, _, _ = outputs
    scaling = 128.0**-0.5
    if target_count >= 1280:
        split_count = 8 if indices.shape[-1] <= 4096 else 4
        return sparse_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            indices,
            counts,
            scaling,
            split_count,
        )
    if target_count >= 900:
        return sparse_cuda.final_attention_ragged_self_split(
            query, key, value, indices, counts, scaling, 2
        )
    return sparse_cuda.final_attention_ragged_self(
        query, key, value, indices, counts, scaling
    )


def gpu_tensor_bytes(mapping: dict[str, Any]) -> int:
    return sum(
        int(value.numel()) * int(value.element_size())
        for value in mapping.values()
        if isinstance(value, torch.Tensor) and value.is_cuda
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    lengths = sorted(
        {int(item) for item in args.lengths.split(",") if item.strip()}
    )
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("lengths must contain positive integers")
    torch.manual_seed(args.seed)
    allocation = PROFILES[args.profile].unsqueeze(0).cuda()
    rows: list[dict[str, Any]] = []

    for history_tokens in lengths:
        target_count = selected_count(history_tokens)
        selected_fraction = target_count / history_tokens
        samples = sample_count_for(
            selected_fraction,
            args.tail_resolution_target,
        )
        selected_samples = selected_sample_count(
            selected_fraction,
            samples,
        )
        threshold = threshold_fraction(selected_fraction, samples)
        capacity = candidate_capacity(
            history_tokens,
            selected_fraction,
            samples,
            args.capacity_sigma,
            args.minimum_capacity_fraction,
        )
        iterations = min(
            args.iterations,
            20 if history_tokens >= 65536 else args.iterations,
        )
        warmup = min(args.warmup, iterations)

        query = torch.randn(1, 32, 128, dtype=dtype, device="cuda")
        grouped_query = query.reshape(1, 8, 4, 128)
        basis = torch.randn(
            1, 8, 128, 128, dtype=dtype, device="cuda"
        )
        key = torch.randn(
            1,
            8,
            history_tokens + 1,
            128,
            dtype=dtype,
            device="cuda",
        )
        value = torch.randn_like(key)
        index = varbit_cuda.allocate_packed_index(
            allocation, history_tokens + 1, dtype
        )
        index["packed_codes"].random_(0, 256)
        index["key_scales"].uniform_(0.01, 0.1)
        outputs = allocate_outputs(
            capacity,
            collect_proxy_mass=args.sample_mass_correction,
        )

        def query_prepare() -> tuple[torch.Tensor, torch.Tensor]:
            return query_cuda.project_quantize(grouped_query, basis)

        query_codes, query_scales = query_prepare()

        def fused_retrieval(
            codes: torch.Tensor = query_codes,
            scales: torch.Tensor = query_scales,
        ) -> tuple[torch.Tensor, ...]:
            return mixed_cuda.plain_sampled_threshold_compact_gqa4_indices_out(
                codes,
                scales,
                index,
                *outputs,
                history_tokens,
                samples,
                threshold,
            )

        def query_plus_retrieval() -> tuple[torch.Tensor, ...]:
            codes, scales = query_prepare()
            return fused_retrieval(codes, scales)

        fused_retrieval()
        if bool(outputs[3].any().item()):
            raise RuntimeError("candidate output overflow during warmup")
        if args.sample_mass_correction:
            initial_mass = outputs[2][1]
            if not bool(torch.isfinite(initial_mass).all().item()):
                raise RuntimeError("proxy selected mass is non-finite")
            if not bool(
                ((initial_mass >= 0.0) & (initial_mass <= 1.0)).all().item()
            ):
                raise RuntimeError("proxy selected mass left [0, 1]")

        def exact_sparse_attention() -> torch.Tensor:
            return sparse_attention(
                query, key, value, outputs, target_count
            )

        prefill_value_sum = value[..., :history_tokens, :].float().sum(dim=2)
        current_value = value[..., history_tokens, :].float()

        def value_mean_update() -> torch.Tensor:
            return (prefill_value_sum + current_value) / float(
                history_tokens + 1
            )

        def blend_sampled_mass(
            sparse_output: torch.Tensor,
            mass: torch.Tensor,
            value_mean: torch.Tensor,
        ) -> torch.Tensor:
            return mass_cuda.mean_value_blend(
                sparse_output,
                value_mean,
                mass,
            )

        sparse_output_for_stage = exact_sparse_attention()
        def proxy_mass_correction_stage() -> torch.Tensor:
            mass = outputs[2][1]
            return blend_sampled_mass(
                sparse_output_for_stage,
                mass,
                value_mean_update(),
            )

        def attention_complete() -> torch.Tensor:
            query_plus_retrieval()
            sparse_output = exact_sparse_attention()
            if not args.sample_mass_correction:
                return sparse_output
            return blend_sampled_mass(
                sparse_output,
                outputs[2][1],
                value_mean_update(),
            )

        full_key = key.repeat_interleave(4, dim=1)
        full_value = value.repeat_interleave(
            4, dim=1
        )

        def full_attention() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2),
                full_key,
                full_value,
                is_causal=False,
            )

        def index_append() -> torch.Tensor:
            projected = torch.einsum(
                "bhkd,bhdm->bhkm",
                key[..., history_tokens : history_tokens + 1, :],
                basis,
            )
            varbit_cuda.encode_projected_keys_into(
                projected.contiguous(), index, history_tokens
            )
            return index["packed_codes"]

        build_index = varbit_cuda.allocate_packed_index(
            allocation, history_tokens, dtype
        )

        def historical_index_build() -> torch.Tensor:
            for start in range(0, history_tokens, args.index_chunk_tokens):
                stop = min(
                    history_tokens, start + args.index_chunk_tokens
                )
                projected = torch.einsum(
                    "bhkd,bhdm->bhkm",
                    key[..., start:stop, :],
                    basis,
                )
                varbit_cuda.encode_projected_keys_into(
                    projected.contiguous(), build_index, start
                )
            return build_index["packed_codes"]

        dense_iterations = min(10, iterations)
        dense_warmup = min(5, warmup)
        full_ms = measure_ms(
            full_attention, dense_warmup, dense_iterations
        )
        query_prepare_ms = measure_ms(
            query_prepare, warmup, iterations
        )
        fused_retrieval_ms = measure_ms(
            fused_retrieval, warmup, iterations
        )
        query_plus_retrieval_ms = measure_ms(
            query_plus_retrieval, warmup, iterations
        )
        exact_sparse_ms = measure_ms(
            exact_sparse_attention, warmup, iterations
        )
        value_mean_update_ms = measure_ms(
            value_mean_update, warmup, iterations
        )
        proxy_mass_correction_ms = (
            measure_ms(
                proxy_mass_correction_stage,
                warmup,
                iterations,
            )
            if args.sample_mass_correction
            else 0.0
        )
        attention_complete_ms = measure_ms(
            attention_complete, warmup, iterations
        )
        append_ms = measure_ms(index_append, warmup, iterations)
        historical_build_ms = measure_ms(historical_index_build, 0, 1)
        fused_retrieval()
        counts = outputs[1].float()
        row = {
            "history_tokens": history_tokens,
            "target_tokens_per_query_head": target_count,
            "target_fraction": selected_fraction,
            "sample_count": samples,
            "selected_sample_count": selected_samples,
            "candidate_capacity": capacity,
            "mean_selected_tokens_per_query_head": float(
                counts.mean().item()
            ),
            "min_selected_tokens_per_query_head": int(
                counts.min().item()
            ),
            "max_selected_tokens_per_query_head": int(
                counts.max().item()
            ),
            "overflow": bool(outputs[3].any().item()),
            "profile": args.profile,
            "allocation_by_kv_head": allocation.squeeze(0).cpu().tolist(),
            "logical_index_bits_per_token_per_kv_head": (
                PROFILE_LOGICAL_BITS[args.profile]
            ),
            "allocated_index_bytes": gpu_tensor_bytes(index),
            "full_preexpanded_sdpa_direct_ms": full_ms,
            "query_prepare_direct_ms": query_prepare_ms,
            "fused_sampled_retrieval_direct_ms": fused_retrieval_ms,
            "query_plus_retrieval_direct_ms": query_plus_retrieval_ms,
            "exact_sparse_attention_direct_ms": exact_sparse_ms,
            "proxy_selected_mass_mean": (
                float(outputs[2][1].mean().item())
                if args.sample_mass_correction
                else None
            ),
            "proxy_selected_mass_min": (
                float(outputs[2][1].min().item())
                if args.sample_mass_correction
                else None
            ),
            "proxy_selected_mass_max": (
                float(outputs[2][1].max().item())
                if args.sample_mass_correction
                else None
            ),
            "value_mean_update_direct_ms": value_mean_update_ms,
            "proxy_mass_correction_stage_direct_ms": (
                proxy_mass_correction_ms
            ),
            "attention_complete_direct_ms": attention_complete_ms,
            "per_token_index_append_direct_ms": append_ms,
            "historical_index_build_direct_ms": historical_build_ms,
            "attention_speedup_vs_full_preexpanded_sdpa": (
                full_ms / attention_complete_ms
            ),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del (
            query,
            grouped_query,
            basis,
            key,
            value,
            index,
            outputs,
            query_codes,
            query_scales,
            full_key,
            full_value,
            build_index,
        )
        torch.cuda.empty_cache()

    result = {
        "schema": "qksieve_deployment_direct_cuda_stages_v2",
        "hardware": torch.cuda.get_device_name(0),
        "contract": {
            "batch": 1,
            "query_heads": 32,
            "kv_heads": 8,
            "gqa_group_size": 4,
            "head_dimension": 128,
            "dtype": args.dtype,
            "profile": args.profile,
            "tail_resolution_target": args.tail_resolution_target,
            "capacity_sigma": args.capacity_sigma,
            "minimum_capacity_fraction": args.minimum_capacity_fraction,
            "sample_mass_correction": args.sample_mass_correction,
            "candidate_schedule": (
                "min(N,1280,max(256,ceil(0.06*N)))"
            ),
            "candidate_capacity": (
                "ceil(N * max(minimum_fraction, r + sigma * "
                "sqrt(r*(1-r)/samples)))"
            ),
            "selector": (
                "fused 256-to-2048-point sampled-quantile scan and "
                "candidate compaction; no exact proxy top-k"
            ),
            "tail_mass_estimator": (
                "proxy softmax mass fused into sampled-quantile retrieval "
                "plus online mean-Value correction"
                if args.sample_mass_correction
                else "disabled"
            ),
            "final_consumer": (
                "exact ragged sparse QK-softmax-AV CUDA kernel"
            ),
            "timing": (
                "CUDA events; every stage and complete path measured directly"
            ),
            "stage_sums_used_for_speedup": False,
            "historical_index_build_in_attention_speedup": False,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
