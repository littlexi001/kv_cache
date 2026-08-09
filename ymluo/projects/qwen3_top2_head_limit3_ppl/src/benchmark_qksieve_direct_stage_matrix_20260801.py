"""Direct CUDA timing matrix for QKSieve, FIER, and dense attention.

Every reported latency is measured by CUDA events around the named operation.
The complete attention paths are timed directly; stage sums are intentionally
not used to infer end-to-end speed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

import fier_rtn1_cuda_20260728 as fier_cuda
import qabs_cuda_kernels as sparse_cuda
import qksieve_query_cuda_20260728 as query_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_variablebit_spectral_attention_20260727 import (
    ALLOCATION_PROFILES,
)


PROFILES = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths", default="8192,16384,32768,65536,131072"
    )
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--index_chunk_tokens", type=int, default=4096)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument(
        "--include_fier",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_csv_ints(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",") if item.strip()})
    if not result or any(item <= 0 for item in result):
        raise ValueError("lengths must contain positive integers")
    return result


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
    return min(history_tokens, 1280, max(256, math.ceil(0.06 * history_tokens)))


def sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    count = int(indices.shape[-1])
    scaling = 128.0**-0.5
    if count >= 1280:
        return sparse_cuda.final_attention_ragged_self_split(
            query, key, value, indices, counts, scaling, 4
        )
    if count >= 900:
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


def qksieve_row(
    *,
    name: str,
    allocation: torch.Tensor,
    history_tokens: int,
    query: torch.Tensor,
    grouped_query: torch.Tensor,
    query_basis: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    count: int,
    warmup: int,
    iterations: int,
    index_chunk_tokens: int,
    full_preexpanded_ms: float,
    scale_dtype: torch.dtype,
) -> dict[str, Any]:
    allocation = allocation.unsqueeze(0).to(device=query.device)
    packed_index = varbit_cuda.allocate_packed_index(
        allocation, history_tokens + 1, scale_dtype
    )
    packed_index["packed_codes"].random_(0, 256)
    packed_index["key_scales"].uniform_(0.01, 0.1)

    def query_prepare() -> tuple[torch.Tensor, torch.Tensor]:
        return query_cuda.project_quantize(grouped_query, query_basis)

    query_codes, query_scales = query_prepare()

    def score_scan() -> torch.Tensor:
        return varbit_cuda.scores(
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
            history_tokens,
        ).reshape(1, 32, history_tokens)

    scores = score_scan()

    def topk() -> torch.Tensor:
        return torch.topk(scores, k=count, dim=-1, sorted=False).indices

    indices = topk().contiguous()
    counts = torch.full(
        (1, 32), count, dtype=torch.long, device=query.device
    )

    def exact_attention() -> torch.Tensor:
        return sparse_attention(query, key, value, indices, counts)

    def selection_complete() -> torch.Tensor:
        current_codes, current_scales = query_prepare()
        current_scores = varbit_cuda.scores(
            current_codes,
            current_scales,
            packed_index["packed_codes"],
            packed_index["key_scales"],
            packed_index["bit_allocations"],
            packed_index["code_offsets"],
            packed_index["scale_offsets"],
            packed_index["code_bases"],
            packed_index["scale_bases"],
            packed_index["code_strides"],
            packed_index["scale_strides"],
            history_tokens,
        ).reshape(1, 32, history_tokens)
        return torch.topk(
            current_scores, k=count, dim=-1, sorted=False
        ).indices

    def attention_complete() -> torch.Tensor:
        current_indices = selection_complete().contiguous()
        return sparse_attention(
            query, key, value, current_indices, counts
        )

    def index_append() -> torch.Tensor:
        projected = torch.einsum(
            "bhkd,bhdm->bhkm",
            key[..., history_tokens : history_tokens + 1, :],
            query_basis,
        )
        varbit_cuda.encode_projected_keys_into(
            projected.contiguous(), packed_index, history_tokens
        )
        return packed_index["packed_codes"]

    build_index = varbit_cuda.allocate_packed_index(
        allocation, history_tokens, scale_dtype
    )

    def historical_index_build() -> torch.Tensor:
        for start in range(0, history_tokens, index_chunk_tokens):
            stop = min(history_tokens, start + index_chunk_tokens)
            projected = torch.einsum(
                "bhkd,bhdm->bhkm",
                key[..., start:stop, :],
                query_basis,
            )
            varbit_cuda.encode_projected_keys_into(
                projected.contiguous(), build_index, start
            )
        return build_index["packed_codes"]

    query_prepare_ms = measure_ms(query_prepare, warmup, iterations)
    score_scan_ms = measure_ms(score_scan, warmup, iterations)
    topk_ms = measure_ms(topk, warmup, iterations)
    exact_attention_ms = measure_ms(exact_attention, warmup, iterations)
    selection_complete_ms = measure_ms(
        selection_complete, warmup, iterations
    )
    attention_complete_ms = measure_ms(
        attention_complete, warmup, iterations
    )
    append_ms = measure_ms(index_append, warmup, iterations)
    historical_build_ms = measure_ms(historical_index_build, 0, 1)
    return {
        "method": f"qksieve_{name}_fulltopk",
        "profile": name,
        "allocation_by_kv_head": allocation.squeeze(0).cpu().tolist(),
        "selected_tokens_per_query_head": count,
        "selected_fraction": count / history_tokens,
        "index_bytes": gpu_tensor_bytes(packed_index),
        "index_ratio_of_full_fp16_kv": (
            gpu_tensor_bytes(packed_index) / (history_tokens * 4096.0)
        ),
        "query_prepare_ms": query_prepare_ms,
        "proxy_score_scan_ms": score_scan_ms,
        "topk_ms": topk_ms,
        "exact_sparse_attention_ms": exact_attention_ms,
        "selection_complete_direct_ms": selection_complete_ms,
        "attention_complete_direct_ms": attention_complete_ms,
        "per_token_index_append_ms": append_ms,
        "historical_index_build_direct_ms": historical_build_ms,
        "attention_speedup_vs_full_preexpanded_sdpa": (
            full_preexpanded_ms / attention_complete_ms
        ),
    }


def fier_row(
    *,
    history_tokens: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    count: int,
    warmup: int,
    iterations: int,
    full_preexpanded_ms: float,
) -> dict[str, Any]:
    packed_index = fier_cuda.allocate_packed_index(
        1, 8, history_tokens + 32, query.device
    )
    fier_cuda.update_packed_index(
        key[..., :history_tokens, :], packed_index, history_tokens
    )

    def score_scan() -> torch.Tensor:
        return fier_cuda.scores(query, packed_index, history_tokens)

    scores = score_scan()

    def topk() -> torch.Tensor:
        return torch.topk(scores, k=count, dim=-1, sorted=False).indices

    indices = topk().contiguous()
    counts = torch.full(
        (1, 32), count, dtype=torch.long, device=query.device
    )

    def exact_attention() -> torch.Tensor:
        return sparse_attention(query, key, value, indices, counts)

    def selection_complete() -> torch.Tensor:
        current_scores = score_scan()
        return torch.topk(
            current_scores, k=count, dim=-1, sorted=False
        ).indices

    def attention_complete() -> torch.Tensor:
        current_indices = selection_complete().contiguous()
        return sparse_attention(
            query, key, value, current_indices, counts
        )

    # FIER shares one scale pair across each 32-token group.  Time one full
    # group update and divide by 32 to avoid reporting only the cheap first
    # token of a new group.
    append_index = fier_cuda.allocate_packed_index(
        1, 8, history_tokens + 32, query.device
    )
    fier_cuda.update_packed_index(
        key[..., :history_tokens, :], append_index, history_tokens
    )

    def append_group() -> torch.Tensor:
        append_index["indexed_count"] = history_tokens
        fier_cuda.update_packed_index(
            key[..., : history_tokens + 32, :],
            append_index,
            history_tokens + 32,
        )
        return append_index["packed_codes"]

    build_index = fier_cuda.allocate_packed_index(
        1, 8, history_tokens, query.device
    )

    def historical_index_build() -> torch.Tensor:
        build_index["indexed_count"] = 0
        fier_cuda.update_packed_index(
            key[..., :history_tokens, :], build_index, history_tokens
        )
        return build_index["packed_codes"]

    score_scan_ms = measure_ms(score_scan, warmup, iterations)
    topk_ms = measure_ms(topk, warmup, iterations)
    exact_attention_ms = measure_ms(exact_attention, warmup, iterations)
    selection_complete_ms = measure_ms(
        selection_complete, warmup, iterations
    )
    attention_complete_ms = measure_ms(
        attention_complete, warmup, iterations
    )
    append_per_token_ms = measure_ms(append_group, warmup, iterations) / 32.0
    historical_build_ms = measure_ms(historical_index_build, 0, 1)
    return {
        "method": "fier_rtn1_g32_fulltopk",
        "selected_tokens_per_query_head": count,
        "selected_fraction": count / history_tokens,
        "index_bytes": fier_cuda.allocated_bytes(packed_index),
        "index_ratio_of_full_fp16_kv": (
            fier_cuda.allocated_bytes(packed_index)
            / (history_tokens * 4096.0)
        ),
        "query_prepare_ms": 0.0,
        "proxy_score_scan_ms": score_scan_ms,
        "topk_ms": topk_ms,
        "exact_sparse_attention_ms": exact_attention_ms,
        "selection_complete_direct_ms": selection_complete_ms,
        "attention_complete_direct_ms": attention_complete_ms,
        "per_token_index_append_ms": append_per_token_ms,
        "historical_index_build_direct_ms": historical_build_ms,
        "attention_speedup_vs_full_preexpanded_sdpa": (
            full_preexpanded_ms / attention_complete_ms
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    names = [part.strip() for part in args.profiles.split(",") if part.strip()]
    unknown = sorted(set(names) - set(PROFILES))
    if not names or unknown:
        raise ValueError(f"unknown profiles: {unknown}")
    if args.dtype == "bfloat16" and args.include_fier:
        raise ValueError("the local FIER kernel is FP16-only; use --no-include_fier")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    torch.manual_seed(args.seed)
    lengths: list[dict[str, Any]] = []

    for history_tokens in parse_csv_ints(args.lengths):
        count = selected_count(history_tokens)
        iterations = min(
            args.iterations,
            20 if history_tokens >= 65536 else args.iterations,
        )
        warmup = min(args.warmup, iterations)
        query = torch.randn(
            1, 32, 128, dtype=dtype, device="cuda"
        )
        grouped_query = query.reshape(1, 8, 4, 128)
        query_basis = torch.randn(
            1, 8, 128, 128, dtype=dtype, device="cuda"
        )
        key = torch.randn(
            1,
            8,
            history_tokens + 32,
            128,
            dtype=dtype,
            device="cuda",
        )
        value = torch.randn_like(key)
        full_key = key[..., :history_tokens, :].repeat_interleave(4, dim=1)
        full_value = value[..., :history_tokens, :].repeat_interleave(4, dim=1)

        def dense_preexpanded() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2), full_key, full_value, is_causal=False
            )

        def dense_native_gqa() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2),
                key[..., :history_tokens, :],
                value[..., :history_tokens, :],
                is_causal=False,
                enable_gqa=True,
            )

        def dense_expand_plus_sdpa() -> torch.Tensor:
            expanded_key = key[..., :history_tokens, :].repeat_interleave(
                4, dim=1
            )
            expanded_value = value[..., :history_tokens, :].repeat_interleave(
                4, dim=1
            )
            return F.scaled_dot_product_attention(
                query.unsqueeze(2),
                expanded_key,
                expanded_value,
                is_causal=False,
            )

        dense_iterations = min(iterations, 10)
        dense_warmup = min(warmup, 5)
        full_preexpanded_ms = measure_ms(
            dense_preexpanded, dense_warmup, dense_iterations
        )
        native_gqa_error: str | None = None
        try:
            full_native_gqa_ms: float | None = measure_ms(
                dense_native_gqa, dense_warmup, dense_iterations
            )
        except TypeError as error:
            full_native_gqa_ms = None
            native_gqa_error = str(error)
        full_expand_plus_sdpa_ms = measure_ms(
            dense_expand_plus_sdpa, dense_warmup, dense_iterations
        )

        method_rows = [
            qksieve_row(
                name=name,
                allocation=PROFILES[name],
                history_tokens=history_tokens,
                query=query,
                grouped_query=grouped_query,
                query_basis=query_basis,
                key=key,
                value=value,
                count=count,
                warmup=warmup,
                iterations=iterations,
                index_chunk_tokens=args.index_chunk_tokens,
                full_preexpanded_ms=full_preexpanded_ms,
                scale_dtype=dtype,
            )
            for name in names
        ]
        if args.include_fier:
            method_rows.append(
                fier_row(
                    history_tokens=history_tokens,
                    query=query,
                    key=key,
                    value=value,
                    count=count,
                    warmup=warmup,
                    iterations=iterations,
                    full_preexpanded_ms=full_preexpanded_ms,
                )
            )
        length_row = {
            "history_tokens": history_tokens,
            "selected_tokens_per_query_head": count,
            "full_attention": {
                "preexpanded_sdpa_direct_ms": full_preexpanded_ms,
                "native_gqa_sdpa_direct_ms": full_native_gqa_ms,
                "native_gqa_unavailable_error": native_gqa_error,
                "repeat_interleave_plus_sdpa_direct_ms": (
                    full_expand_plus_sdpa_ms
                ),
            },
            "methods": method_rows,
        }
        lengths.append(length_row)
        print(json.dumps(length_row, sort_keys=True), flush=True)
        del query, grouped_query, query_basis, key, value, full_key, full_value
        torch.cuda.empty_cache()

    output = {
        "schema": "qksieve_direct_cuda_stage_matrix_v1",
        "hardware": torch.cuda.get_device_name(0),
        "contract": {
            "batch": 1,
            "query_heads": 32,
            "kv_heads": 8,
            "head_dimension": 128,
            "dtype": args.dtype,
            "candidate_schedule": "min(N, 1280, max(256, ceil(0.06*N)))",
            "quality_aligned_selector": "materialized proxy scores plus exact torch.topk",
            "final_consumer": "shared exact ragged sparse QK-softmax-AV CUDA kernel",
            "timing": "CUDA events; every field is measured directly",
            "stage_sums_used_for_speedup": False,
            "fier_status": "faithful local RTN-1 g32 CUDA implementation, not official FIER kernel",
            "historical_index_build_in_attention_speedup": False,
            "per_token_index_append_in_attention_speedup": False,
        },
        "lengths": lengths,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
