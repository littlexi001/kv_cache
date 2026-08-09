from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import qabs_cuda_kernels as qabs_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda


DEFAULT_SCHEDULE = {
    2048: 0.125,
    4096: 0.0625,
    8192: 0.06,
    16384: 0.06,
    32768: 0.04,
    65536: 0.02,
    131072: 0.01,
}

# Representative b10 qMSE allocations observed on Qwen3-4B. Each row sums to
# ten bit-per-coordinate units over eight independent 16D spectral bands.
ALLOCATION_PROFILES = {
    "qmse_b10": torch.tensor(
        [
            [8, 1, 1, 0, 0, 0, 0, 0],
            [8, 1, 1, 0, 0, 0, 0, 0],
            [8, 1, 1, 0, 0, 0, 0, 0],
            [8, 1, 1, 0, 0, 0, 0, 0],
            [4, 4, 1, 1, 0, 0, 0, 0],
            [4, 4, 1, 1, 0, 0, 0, 0],
            [4, 1, 1, 1, 1, 1, 1, 0],
            [4, 1, 1, 1, 1, 1, 0, 1],
        ],
        dtype=torch.int8,
    ),
    # Per-head modal layouts from Qwen3-4B sports/medicine traces under a
    # 15-unit budget that charges one FP16 scale for each active 16D band.
    "qmse_total_b15": torch.tensor(
        [
            [8, 4, 0, 0, 0, 0, 0, 0],
            [8, 1, 1, 1, 0, 0, 0, 0],
            [4, 4, 4, 0, 0, 0, 0, 0],
            [8, 1, 1, 1, 0, 0, 0, 0],
            [4, 4, 4, 0, 0, 0, 0, 0],
            [8, 4, 0, 0, 0, 0, 0, 0],
            [8, 1, 1, 1, 0, 0, 0, 0],
            [8, 1, 1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.int8,
    ),
    "fixed_low192": torch.tensor(
        [[4, 4, 1, 0, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
    "fixed410": torch.tensor(
        [[4, 1, 0, 0, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
    "fixed400_b80": torch.tensor(
        [[4, 0, 0, 0, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
    "fixed210_b80": torch.tensor(
        [[2, 1, 0, 0, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
    "fixed4221_b208": torch.tensor(
        [[4, 2, 2, 1, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
    "fixed4421_b240": torch.tensor(
        [[4, 4, 2, 1, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
}


def parse_ints(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("expected at least one integer")
    return result


def selected_fraction(history_tokens: int) -> float:
    return DEFAULT_SCHEDULE.get(
        history_tokens,
        min(0.06, 1280.0 / history_tokens),
    )


def variance_controlled_sample_count(
    selected_fraction_target: float,
    minimum: int,
    maximum: int,
    target_tail_samples: int,
) -> int:
    return min(
        maximum,
        max(
            minimum,
            math.ceil(
                target_tail_samples / selected_fraction_target
            ),
        ),
    )


def measure_ms(
    function: Callable[[], object],
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


def make_metadata(
    allocations: torch.Tensor,
    history_tokens: int,
) -> dict[str, torch.Tensor | int | float]:
    allocations = allocations.to(torch.int8).cpu()
    batch_count, kv_heads, bands = allocations.shape
    if bands != 8:
        raise ValueError("expected eight 16D bands")
    code_offsets = torch.zeros_like(allocations, dtype=torch.int16)
    scale_offsets = torch.full_like(allocations, -1, dtype=torch.int8)
    code_strides = torch.zeros(
        batch_count, kv_heads, dtype=torch.int16
    )
    scale_strides = torch.zeros(
        batch_count, kv_heads, dtype=torch.int8
    )
    code_bases = torch.zeros(
        batch_count, kv_heads, dtype=torch.int64
    )
    scale_bases = torch.zeros(
        batch_count, kv_heads, dtype=torch.int64
    )
    code_cursor = 0
    scale_cursor = 0
    for batch in range(batch_count):
        for head in range(kv_heads):
            code_bases[batch, head] = code_cursor
            scale_bases[batch, head] = scale_cursor
            token_code_offset = 0
            token_scale_offset = 0
            for band in range(8):
                bits = int(allocations[batch, head, band])
                if bits not in {0, 1, 2, 4, 8}:
                    raise ValueError(f"unsupported bit width: {bits}")
                code_offsets[batch, head, band] = token_code_offset
                if bits:
                    scale_offsets[batch, head, band] = token_scale_offset
                    token_code_offset += 2 * bits
                    token_scale_offset += 1
            code_strides[batch, head] = token_code_offset
            scale_strides[batch, head] = token_scale_offset
            code_cursor += history_tokens * token_code_offset
            scale_cursor += history_tokens * token_scale_offset
    full_kv_bytes = batch_count * kv_heads * history_tokens * 512
    packed_bytes = code_cursor + 2 * scale_cursor
    return {
        "bit_allocations": allocations.cuda(),
        "code_offsets": code_offsets.cuda(),
        "scale_offsets": scale_offsets.cuda(),
        "code_bases": code_bases.cuda(),
        "scale_bases": scale_bases.cuda(),
        "code_strides": code_strides.cuda(),
        "scale_strides": scale_strides.cuda(),
        "total_code_bytes": code_cursor,
        "total_scale_values": scale_cursor,
        "packed_bytes": packed_bytes,
        "index_ratio_of_full_kv": packed_bytes / full_kv_bytes,
    }


def quantize_query(
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = query.float().reshape(*query.shape[:-1], 8, 16)
    scales = (
        grouped.abs().amax(dim=-1).clamp_min(1.0e-8) / 127.0
    )
    codes = torch.round(grouped / scales.unsqueeze(-1)).clamp(-127, 127)
    return codes.to(torch.int8).reshape_as(query), scales.to(query.dtype)


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    unsigned = (codes.to(torch.int16) & ((1 << bits) - 1)).to(torch.uint8)
    if bits == 8:
        return unsigned
    if bits == 4:
        grouped = unsigned.reshape(unsigned.shape[0], 8, 2)
        return grouped[..., 0] | (grouped[..., 1] << 4)
    if bits == 2:
        grouped = unsigned.reshape(unsigned.shape[0], 4, 4)
        return (
            grouped[..., 0]
            | (grouped[..., 1] << 2)
            | (grouped[..., 2] << 4)
            | (grouped[..., 3] << 6)
        )
    if bits == 1:
        positive = (codes > 0).to(torch.uint8).reshape(
            codes.shape[0], 2, 8
        )
        weights = (1 << torch.arange(8, device=codes.device)).to(torch.uint8)
        return (positive * weights).sum(dim=-1).to(torch.uint8)
    raise ValueError(f"unsupported bit width: {bits}")


def validate_scores(
    representative_allocations: torch.Tensor,
) -> dict[str, float | int]:
    torch.manual_seed(20260727)
    history_tokens = 257
    allocations = representative_allocations.unsqueeze(0).cuda()
    metadata = make_metadata(allocations, history_tokens)
    query_codes = torch.randint(
        -127,
        128,
        (1, 8, 4, 128),
        dtype=torch.int8,
        device="cuda",
    )
    query_scales = (
        torch.rand(1, 8, 4, 8, dtype=torch.float16, device="cuda") * 0.03
        + 0.001
    )
    unpacked = torch.zeros(
        1, 8, history_tokens, 128, dtype=torch.int8, device="cuda"
    )
    scale_matrix = torch.zeros(
        1, 8, history_tokens, 8, dtype=torch.float16, device="cuda"
    )
    packed_codes = torch.zeros(
        int(metadata["total_code_bytes"]),
        dtype=torch.uint8,
        device="cuda",
    )
    key_scales = torch.zeros(
        int(metadata["total_scale_values"]),
        dtype=torch.float16,
        device="cuda",
    )
    for head in range(8):
        code_stride = int(metadata["code_strides"][0, head].item())
        scale_stride = int(metadata["scale_strides"][0, head].item())
        head_codes = torch.zeros(
            history_tokens, code_stride, dtype=torch.uint8, device="cuda"
        )
        head_scales = torch.zeros(
            history_tokens, scale_stride, dtype=torch.float16, device="cuda"
        )
        for band in range(8):
            bits = int(allocations[0, head, band].item())
            if bits == 0:
                continue
            maximum = 1 if bits == 1 else (1 << (bits - 1)) - 1
            band_codes = torch.randint(
                -maximum,
                maximum + 1,
                (history_tokens, 16),
                dtype=torch.int8,
                device="cuda",
            )
            if bits == 1:
                band_codes = torch.where(
                    band_codes >= 0,
                    torch.ones_like(band_codes),
                    -torch.ones_like(band_codes),
                )
            scale = (
                torch.rand(
                    history_tokens, dtype=torch.float16, device="cuda"
                )
                * 0.03
                + 0.001
            )
            unpacked[0, head, :, band * 16 : (band + 1) * 16] = band_codes
            scale_matrix[0, head, :, band] = scale
            code_offset = int(metadata["code_offsets"][0, head, band].item())
            packed_band = pack_codes(band_codes, bits)
            head_codes[
                :, code_offset : code_offset + packed_band.shape[-1]
            ] = packed_band
            scale_offset = int(
                metadata["scale_offsets"][0, head, band].item()
            )
            head_scales[:, scale_offset] = scale
        code_base = int(metadata["code_bases"][0, head].item())
        scale_base = int(metadata["scale_bases"][0, head].item())
        packed_codes[
            code_base : code_base + head_codes.numel()
        ] = head_codes.flatten()
        key_scales[
            scale_base : scale_base + head_scales.numel()
        ] = head_scales.flatten()

    tested = varbit_cuda.scores(
        query_codes,
        query_scales,
        packed_codes,
        key_scales,
        metadata["bit_allocations"],
        metadata["code_offsets"],
        metadata["scale_offsets"],
        metadata["code_bases"],
        metadata["scale_bases"],
        metadata["code_strides"],
        metadata["scale_strides"],
        history_tokens,
    )
    reconstructed_key = (
        unpacked.float()
        * scale_matrix.repeat_interleave(16, dim=-1).float()
    )
    reconstructed_query = (
        query_codes.float()
        * query_scales.repeat_interleave(16, dim=-1).float()
    )
    reference = torch.einsum(
        "bhgd,bhkd->bhgk",
        reconstructed_query,
        reconstructed_key,
    ).reshape_as(tested)
    return {
        "history_tokens": history_tokens,
        "max_abs_error": float((tested - reference).abs().max().item()),
        "mean_abs_error": float((tested - reference).abs().mean().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the packed variable-bit qMSE spectral retrieval index "
            "and its complete exact-candidate sparse-attention path."
        )
    )
    parser.add_argument(
        "--lengths",
        default="2048,4096,8192,16384,32768,65536,131072",
    )
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument("--adaptive_sample_count", action="store_true")
    parser.add_argument(
        "--unbiased_quantile",
        action="store_true",
        help=(
            "Use the nearest finite-sample order statistic instead of "
            "ceil(target_fraction * sample_count)."
        ),
    )
    parser.add_argument("--maximum_sample_count", type=int, default=2048)
    parser.add_argument("--target_tail_samples", type=int, default=16)
    parser.add_argument("--capacity_multiplier", type=float, default=2.0)
    parser.add_argument("--minimum_capacity_fraction", type=float, default=0.06)
    parser.add_argument("--split_count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--full_iterations", type=int, default=20)
    parser.add_argument(
        "--allocation_profile",
        choices=sorted(ALLOCATION_PROFILES),
        default="qmse_b10",
    )
    parser.add_argument(
        "--compare_sharedtail",
        action="store_true",
        help="Also benchmark the fixed 240-bit 4/4/shared-sign index.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    representative_allocations = ALLOCATION_PROFILES[
        args.allocation_profile
    ]
    correctness = validate_scores(representative_allocations)
    if correctness["max_abs_error"] > 0.02:
        raise RuntimeError(f"variable-bit score mismatch: {correctness}")

    rows = []
    allocations = representative_allocations.unsqueeze(0)
    for history_tokens in parse_ints(args.lengths):
        fraction = selected_fraction(history_tokens)
        sample_count = (
            variance_controlled_sample_count(
                fraction,
                args.sample_count,
                args.maximum_sample_count,
                args.target_tail_samples,
            )
            if args.adaptive_sample_count
            else args.sample_count
        )
        capacity_fraction = min(
            1.0,
            max(
                args.minimum_capacity_fraction,
                args.capacity_multiplier * fraction,
            ),
        )
        capacity = min(
            history_tokens,
            max(1, math.ceil(capacity_fraction * history_tokens)),
        )
        threshold_fraction = fraction
        threshold_rank = max(
            1,
            min(
                sample_count,
                int(round(fraction * (sample_count + 1))),
            ),
        )
        if args.unbiased_quantile:
            threshold_fraction = (threshold_rank - 0.5) / sample_count
        metadata = make_metadata(allocations, history_tokens)
        packed_codes = torch.randint(
            0,
            256,
            (int(metadata["total_code_bytes"]),),
            dtype=torch.uint8,
            device="cuda",
        )
        key_scales = (
            torch.rand(
                int(metadata["total_scale_values"]),
                dtype=torch.float16,
                device="cuda",
            )
            * 0.03
            + 0.001
        )
        projected_query = torch.randn(
            1, 8, 4, 128, dtype=torch.float16, device="cuda"
        )
        query_codes, query_scales = quantize_query(projected_query)
        sharedtail_amplitude = (
            torch.rand(
                1,
                8,
                128,
                dtype=torch.float32,
                device="cuda",
            )
            * 0.75
            + 0.25
        )
        sharedtail_query_codes, sharedtail_query_scales = (
            varbit_cuda.quantize_sharedtail_projected_query(
                projected_query,
                sharedtail_amplitude,
            )
        )
        sharedtail_index = {
            "packed_codes": torch.randint(
                0,
                256,
                (1, 8, history_tokens, 24),
                dtype=torch.uint8,
                device="cuda",
            ),
            "key_scales": (
                torch.rand(
                    1,
                    8,
                    history_tokens,
                    3,
                    dtype=torch.float16,
                    device="cuda",
                )
                * 0.03
                + 0.001
            ),
            "coordinate_amplitude": sharedtail_amplitude,
            "capacity": history_tokens,
        }
        query = torch.randn(
            1, 32, 128, dtype=torch.float16, device="cuda"
        )
        key = torch.randn(
            1,
            8,
            history_tokens + 1,
            128,
            dtype=torch.float16,
            device="cuda",
        )
        value = torch.randn_like(key)
        full_key = key.repeat_interleave(4, dim=1)
        full_value = value.repeat_interleave(4, dim=1)
        scaling = 128.0**-0.5

        def scan() -> torch.Tensor:
            return varbit_cuda.scores(
                query_codes,
                query_scales,
                packed_codes,
                key_scales,
                metadata["bit_allocations"],
                metadata["code_offsets"],
                metadata["scale_offsets"],
                metadata["code_bases"],
                metadata["scale_bases"],
                metadata["code_strides"],
                metadata["scale_strides"],
                history_tokens,
            )

        def retrieve() -> tuple[torch.Tensor, ...]:
            return varbit_cuda.sampled_threshold_compact_out(
                query_codes,
                query_scales,
                packed_codes,
                key_scales,
                metadata["bit_allocations"],
                metadata["code_offsets"],
                metadata["scale_offsets"],
                metadata["code_bases"],
                metadata["scale_bases"],
                metadata["code_strides"],
                metadata["scale_strides"],
                candidate_indices,
                candidate_proxy_scores,
                candidate_counts,
                candidate_thresholds,
                candidate_overflow,
                history_tokens,
                sample_count,
                threshold_fraction,
            )

        def retrieve_allocating() -> tuple[torch.Tensor, ...]:
            return varbit_cuda.sampled_threshold_compact(
                query_codes,
                query_scales,
                packed_codes,
                key_scales,
                metadata["bit_allocations"],
                metadata["code_offsets"],
                metadata["scale_offsets"],
                metadata["code_bases"],
                metadata["scale_bases"],
                metadata["code_strides"],
                metadata["scale_strides"],
                history_tokens,
                sample_count,
                threshold_fraction,
                capacity,
            )

        candidate_indices = torch.empty(
            1, 32, capacity, dtype=torch.long, device="cuda"
        )
        candidate_proxy_scores = torch.empty(
            1, 32, capacity, dtype=torch.float32, device="cuda"
        )
        candidate_counts = torch.empty(
            1, 32, dtype=torch.long, device="cuda"
        )
        candidate_thresholds = torch.empty(
            1, 32, dtype=torch.float32, device="cuda"
        )
        candidate_overflow = torch.empty(
            1, 32, dtype=torch.bool, device="cuda"
        )
        sharedtail_candidate_indices = torch.empty(
            1, 32, capacity, dtype=torch.long, device="cuda"
        )
        sharedtail_candidate_scores = torch.empty(
            1, 32, capacity, dtype=torch.float32, device="cuda"
        )
        sharedtail_candidate_counts = torch.empty(
            1, 32, dtype=torch.long, device="cuda"
        )
        sharedtail_candidate_thresholds = torch.empty(
            1, 32, dtype=torch.float32, device="cuda"
        )
        sharedtail_candidate_overflow = torch.empty(
            1, 32, dtype=torch.bool, device="cuda"
        )

        def sharedtail_scan() -> torch.Tensor:
            return varbit_cuda.sharedtail_scores(
                sharedtail_query_codes,
                sharedtail_query_scales,
                sharedtail_index,
                history_tokens,
            )

        def sharedtail_retrieve() -> tuple[torch.Tensor, ...]:
            return varbit_cuda.sharedtail_sampled_threshold_compact_out(
                sharedtail_query_codes,
                sharedtail_query_scales,
                sharedtail_index,
                sharedtail_candidate_indices,
                sharedtail_candidate_scores,
                sharedtail_candidate_counts,
                sharedtail_candidate_thresholds,
                sharedtail_candidate_overflow,
                history_tokens,
                sample_count,
                fraction,
            )

        indices, _, counts, _, overflow = retrieve()
        (
            sharedtail_indices,
            _,
            sharedtail_counts,
            _,
            sharedtail_overflow,
        ) = sharedtail_retrieve()

        def consume() -> torch.Tensor:
            return qabs_cuda.final_attention_ragged_self_split(
                query,
                key,
                value,
                indices,
                counts,
                scaling,
                args.split_count,
            )

        def complete() -> torch.Tensor:
            current_indices, _, current_counts, _, _ = retrieve()
            return qabs_cuda.final_attention_ragged_self_split(
                query,
                key,
                value,
                current_indices,
                current_counts,
                scaling,
                args.split_count,
            )

        def sharedtail_consume() -> torch.Tensor:
            return qabs_cuda.final_attention_ragged_self_split(
                query,
                key,
                value,
                sharedtail_indices,
                sharedtail_counts,
                scaling,
                args.split_count,
            )

        def sharedtail_complete() -> torch.Tensor:
            (
                current_indices,
                _,
                current_counts,
                _,
                _,
            ) = sharedtail_retrieve()
            return qabs_cuda.final_attention_ragged_self_split(
                query,
                key,
                value,
                current_indices,
                current_counts,
                scaling,
                args.split_count,
            )

        def full_attention() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2),
                full_key,
                full_value,
            )

        iterations = (
            min(args.iterations, 20)
            if history_tokens >= 65536
            else args.iterations
        )
        scan_ms = measure_ms(scan, args.warmup, iterations)
        query_prepare_ms = measure_ms(
            lambda: varbit_cuda.quantize_projected_query(projected_query),
            args.warmup,
            iterations,
        )
        allocating_retrieve_ms = measure_ms(
            retrieve_allocating, args.warmup, iterations
        )
        retrieve_ms = measure_ms(retrieve, args.warmup, iterations)
        consume_ms = measure_ms(consume, args.warmup, iterations)
        complete_ms = measure_ms(complete, args.warmup, iterations)
        full_ms = measure_ms(
            full_attention,
            min(args.warmup, 10),
            min(args.full_iterations, iterations),
        )
        if args.compare_sharedtail:
            sharedtail_query_prepare_ms = measure_ms(
                lambda: varbit_cuda.quantize_sharedtail_projected_query(
                    projected_query,
                    sharedtail_amplitude,
                ),
                args.warmup,
                iterations,
            )
            sharedtail_scan_ms = measure_ms(
                sharedtail_scan, args.warmup, iterations
            )
            sharedtail_retrieve_ms = measure_ms(
                sharedtail_retrieve, args.warmup, iterations
            )
            sharedtail_consume_ms = measure_ms(
                sharedtail_consume, args.warmup, iterations
            )
            sharedtail_complete_ms = measure_ms(
                sharedtail_complete, args.warmup, iterations
            )
        else:
            sharedtail_query_prepare_ms = 0.0
            sharedtail_scan_ms = 0.0
            sharedtail_retrieve_ms = 0.0
            sharedtail_consume_ms = 0.0
            sharedtail_complete_ms = 0.0
        mean_fraction = float(
            (counts.float() / history_tokens).mean().item()
        )
        packed_bytes = int(metadata["packed_bytes"])
        row = {
            "history_tokens": history_tokens,
            "selected_fraction_target": fraction,
            "quantile_sample_count": sample_count,
            "expected_quantile_tail_samples": sample_count * fraction,
            "quantile_threshold_fraction": threshold_fraction,
            "quantile_threshold_rank": threshold_rank,
            "selected_fraction_mean": mean_fraction,
            "selected_fraction_max": float(
                (counts.float() / history_tokens).max().item()
            ),
            "capacity_fraction": capacity / history_tokens,
            "overflow": bool(overflow.any().item()),
            "code_bytes_per_kv_token_head": (
                int(metadata["total_code_bytes"]) / (8 * history_tokens)
            ),
            "scale_bytes_per_kv_token_head": (
                2 * int(metadata["total_scale_values"])
                / (8 * history_tokens)
            ),
            "index_ratio_of_full_kv": float(
                metadata["index_ratio_of_full_kv"]
            ),
            "index_bytes_total": packed_bytes,
            "full_score_scan_ms": scan_ms,
            "query_prepare_ms": query_prepare_ms,
            "allocating_fused_retrieval_ms": allocating_retrieve_ms,
            "fused_retrieval_ms": retrieve_ms,
            "exact_candidate_attention_ms": consume_ms,
            "complete_attention_ms": complete_ms,
            "complete_attention_plus_query_prepare_ms": (
                complete_ms + query_prepare_ms
            ),
            "full_sdpa_ms": full_ms,
            "full_sdpa_over_complete_attention": full_ms / complete_ms,
            "full_sdpa_over_complete_plus_query_prepare": (
                full_ms / (complete_ms + query_prepare_ms)
            ),
            "effective_index_bandwidth_gbps": (
                packed_bytes / scan_ms / 1.0e6
            ),
            "sharedtail_index_ratio_of_full_kv": 240 / 4096,
            "sharedtail_selected_fraction_mean": float(
                (sharedtail_counts.float() / history_tokens).mean().item()
            ),
            "sharedtail_selected_fraction_max": float(
                (sharedtail_counts.float() / history_tokens).max().item()
            ),
            "sharedtail_overflow": bool(sharedtail_overflow.any().item()),
            "sharedtail_full_score_scan_ms": sharedtail_scan_ms,
            "sharedtail_query_prepare_ms": sharedtail_query_prepare_ms,
            "sharedtail_fused_retrieval_ms": sharedtail_retrieve_ms,
            "sharedtail_exact_candidate_attention_ms": (
                sharedtail_consume_ms
            ),
            "sharedtail_complete_attention_ms": sharedtail_complete_ms,
            "sharedtail_complete_plus_query_prepare_ms": (
                sharedtail_complete_ms + sharedtail_query_prepare_ms
            ),
            "full_sdpa_over_sharedtail_complete_attention": (
                full_ms / sharedtail_complete_ms
                if sharedtail_complete_ms > 0.0
                else 0.0
            ),
            "full_sdpa_over_sharedtail_complete_plus_query_prepare": (
                full_ms
                / (
                    sharedtail_complete_ms
                    + sharedtail_query_prepare_ms
                )
                if sharedtail_complete_ms > 0.0
                else 0.0
            ),
            "sharedtail_over_variablebit_complete": (
                complete_ms / sharedtail_complete_ms
                if sharedtail_complete_ms > 0.0
                else 0.0
            ),
            "sharedtail_effective_index_bandwidth_gbps": (
                (history_tokens * 8 * 30)
                / sharedtail_scan_ms
                / 1.0e6
                if sharedtail_scan_ms > 0.0
                else 0.0
            ),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del (
            metadata,
            packed_codes,
            key_scales,
            projected_query,
            query_codes,
            query_scales,
            query,
            key,
            value,
            full_key,
            full_value,
            candidate_indices,
            candidate_proxy_scores,
            candidate_counts,
            candidate_thresholds,
            candidate_overflow,
            sharedtail_amplitude,
            sharedtail_query_codes,
            sharedtail_query_scales,
            sharedtail_index,
            sharedtail_candidate_indices,
            sharedtail_candidate_scores,
            sharedtail_candidate_counts,
            sharedtail_candidate_thresholds,
            sharedtail_candidate_overflow,
            sharedtail_indices,
            sharedtail_counts,
            sharedtail_overflow,
            indices,
            counts,
            overflow,
        )
        torch.cuda.empty_cache()

    output = {
        "correctness": correctness,
        "config": {
            **vars(args),
            "output": str(args.output) if args.output else None,
            "allocations": representative_allocations.tolist(),
            "selected_fraction_schedule": DEFAULT_SCHEDULE,
        },
        "scope": (
            "One decode attention layer with 32 query heads, 8 KV heads, "
            "head dimension 128. The complete path includes the packed "
            "variable-bit proxy scan, sampled threshold and compaction, exact "
            "candidate QK, current-token attention, and sparse V aggregation. "
            "It excludes index construction and all non-attention model work."
        ),
        "rows": rows,
    }
    text = json.dumps(output, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
