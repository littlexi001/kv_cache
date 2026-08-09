from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import mixedblock_spectral_cuda_20260729 as mixedblock_cuda
import qabs_cuda_kernels as qabs_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda


HIGH_ALLOCATIONS = torch.tensor(
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
)
LOW_ALLOCATIONS = torch.tensor(
    [[4, 4, 1, 0, 0, 0, 0, 0]] * 8,
    dtype=torch.int8,
)
LOW_PROFILES = {
    "fixed441": LOW_ALLOCATIONS,
    "fixed84": torch.tensor(
        [[8, 4, 0, 0, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
    "fixed811": torch.tensor(
        [[8, 1, 1, 0, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="32768,65536,131072")
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--hot_fraction", type=float, default=0.10)
    parser.add_argument(
        "--low_profile",
        choices=tuple(LOW_PROFILES),
        default="fixed441",
    )
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selected_fraction(history_count: int) -> float:
    return min(0.06, 1280.0 / history_count)


def sample_count_for(fraction: float) -> int:
    return min(2048, max(256, math.ceil(16.0 / fraction)))


def unbiased_fraction(fraction: float, sample_count: int) -> float:
    rank = max(
        1,
        min(sample_count, int(round(fraction * (sample_count + 1)))),
    )
    return (rank - 0.5) / sample_count


def profile_metadata(allocations: torch.Tensor) -> dict[str, torch.Tensor]:
    profiles, heads, bands = allocations.shape
    code_offsets = torch.zeros_like(allocations, dtype=torch.int16)
    scale_offsets = torch.full_like(allocations, -1, dtype=torch.int8)
    code_strides = torch.zeros(profiles, heads, dtype=torch.int16)
    scale_strides = torch.zeros(profiles, heads, dtype=torch.int8)
    for profile in range(profiles):
        for head in range(heads):
            code_cursor = 0
            scale_cursor = 0
            for band in range(bands):
                bits = int(allocations[profile, head, band].item())
                code_offsets[profile, head, band] = code_cursor
                if bits:
                    scale_offsets[profile, head, band] = scale_cursor
                    code_cursor += 2 * bits
                    scale_cursor += 1
            code_strides[profile, head] = code_cursor
            scale_strides[profile, head] = scale_cursor
    return {
        "code_offsets": code_offsets,
        "scale_offsets": scale_offsets,
        "code_strides": code_strides,
        "scale_strides": scale_strides,
    }


def mixed_metadata(
    history_count: int,
    block_size: int,
    hot_fraction: float,
    low_allocations: torch.Tensor,
) -> tuple[dict[str, torch.Tensor | int], int, int]:
    allocations = torch.stack((low_allocations, HIGH_ALLOCATIONS), dim=0)
    profile = profile_metadata(allocations)
    head_count = int(allocations.shape[1])
    block_count = math.ceil(history_count / block_size)
    hot_blocks = max(1, math.ceil(block_count * hot_fraction))
    block_profiles = torch.zeros(
        head_count, block_count, dtype=torch.uint8
    )
    generator = torch.Generator().manual_seed(20260729)
    for head in range(head_count):
        selected = torch.randperm(
            block_count, generator=generator
        )[:hot_blocks]
        block_profiles[head, selected] = 1

    cumulative_hot = block_profiles.to(torch.int32).cumsum(dim=1)
    if int(cumulative_hot.max().item()) > torch.iinfo(torch.int16).max:
        raise ValueError("hot-block prefix exceeds int16 capacity")
    block_hot_prefix = torch.cat(
        (
            torch.zeros(head_count, 1, dtype=torch.int16),
            cumulative_hot.to(torch.int16),
        ),
        dim=1,
    )
    head_code_bases = torch.empty(head_count, dtype=torch.int64)
    head_scale_bases = torch.empty_like(head_code_bases)
    code_cursor = 0
    scale_cursor = 0
    for head in range(head_count):
        head_code_bases[head] = code_cursor
        head_scale_bases[head] = scale_cursor
        for block in range(block_count):
            selected_profile = int(block_profiles[head, block].item())
            code_cursor += (
                block_size
                * int(
                    profile["code_strides"][
                        selected_profile, head
                    ].item()
                )
            )
            scale_cursor += (
                block_size
                * int(
                    profile["scale_strides"][
                        selected_profile, head
                    ].item()
                )
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


def sortedblock_metadata(
    history_count: int,
    block_size: int,
    hot_fraction: float,
    low_allocations: torch.Tensor,
) -> tuple[dict[str, torch.Tensor | int], int, int]:
    allocations = torch.stack((low_allocations, HIGH_ALLOCATIONS), dim=0)
    profile = profile_metadata(allocations)
    head_count = int(allocations.shape[1])
    block_count = math.ceil(history_count / block_size)
    hot_blocks = max(1, math.ceil(block_count * hot_fraction))
    original_blocks = torch.empty(
        head_count, block_count, dtype=torch.int32
    )
    generator = torch.Generator().manual_seed(20260729)
    for head in range(head_count):
        permutation = torch.randperm(block_count, generator=generator)
        hot = permutation[:hot_blocks]
        hot_mask = torch.zeros(block_count, dtype=torch.bool)
        hot_mask[hot] = True
        original_blocks[head] = torch.cat(
            (hot, torch.arange(block_count)[~hot_mask])
        ).to(torch.int32)

    head_code_bases = torch.empty(head_count, dtype=torch.int64)
    head_scale_bases = torch.empty_like(head_code_bases)
    code_cursor = 0
    scale_cursor = 0
    for head in range(head_count):
        head_code_bases[head] = code_cursor
        head_scale_bases[head] = scale_cursor
        code_cursor += block_size * (
            hot_blocks * int(profile["code_strides"][1, head].item())
            + (block_count - hot_blocks)
            * int(profile["code_strides"][0, head].item())
        )
        scale_cursor += block_size * (
            hot_blocks * int(profile["scale_strides"][1, head].item())
            + (block_count - hot_blocks)
            * int(profile["scale_strides"][0, head].item())
        )
    metadata: dict[str, torch.Tensor | int] = {
        "bit_allocations": allocations.cuda(),
        "code_offsets": profile["code_offsets"].cuda(),
        "scale_offsets": profile["scale_offsets"].cuda(),
        "code_strides": profile["code_strides"].cuda(),
        "scale_strides": profile["scale_strides"].cuda(),
        "head_code_bases": head_code_bases.cuda(),
        "head_scale_bases": head_scale_bases.cuda(),
        "original_blocks": original_blocks.cuda(),
        "block_size": block_size,
        "hot_block_count": hot_blocks,
    }
    return metadata, code_cursor, scale_cursor


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
    return float(start.elapsed_time(end) / iterations)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rows = []
    for history_count in sorted(
        {int(value) for value in args.lengths.split(",")}
    ):
        fraction = selected_fraction(history_count)
        sample_count = sample_count_for(fraction)
        threshold_fraction = unbiased_fraction(fraction, sample_count)
        capacity_fraction = min(
            1.0,
            max(
                0.06,
                fraction
                + 6.0
                * math.sqrt(
                    fraction * (1.0 - fraction) / sample_count
                ),
            ),
        )
        capacity = min(
            history_count,
            max(1, math.ceil(capacity_fraction * history_count)),
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
        full_key = key.repeat_interleave(4, dim=1)
        full_value = value.repeat_interleave(4, dim=1)

        high_allocation = HIGH_ALLOCATIONS[None].cuda()
        high_meta = varbit_cuda.make_packed_metadata(
            high_allocation,
            history_count,
        )
        high_codes = torch.randint(
            0,
            256,
            (int(high_meta["total_code_bytes"]),),
            dtype=torch.uint8,
            device="cuda",
        )
        high_scales = torch.rand(
            int(high_meta["total_scale_values"]),
            dtype=torch.float16,
            device="cuda",
        )
        mixed_meta, mixed_code_count, mixed_scale_count = mixed_metadata(
            history_count,
            args.block_size,
            args.hot_fraction,
            LOW_PROFILES[args.low_profile],
        )
        mixed_codes = torch.randint(
            0,
            256,
            (mixed_code_count,),
            dtype=torch.uint8,
            device="cuda",
        )
        mixed_scales = torch.rand(
            mixed_scale_count,
            dtype=torch.float16,
            device="cuda",
        )
        sorted_meta, sorted_code_count, sorted_scale_count = (
            sortedblock_metadata(
                history_count,
                args.block_size,
                args.hot_fraction,
                LOW_PROFILES[args.low_profile],
            )
        )
        sorted_codes = torch.randint(
            0,
            256,
            (sorted_code_count,),
            dtype=torch.uint8,
            device="cuda",
        )
        sorted_scales = torch.rand(
            sorted_scale_count,
            dtype=torch.float16,
            device="cuda",
        )

        shape = (1, 32, capacity)
        high_outputs = (
            torch.empty(shape, dtype=torch.long, device="cuda"),
            torch.empty(shape, dtype=torch.float32, device="cuda"),
            torch.empty(1, 32, dtype=torch.long, device="cuda"),
            torch.empty(1, 32, dtype=torch.float32, device="cuda"),
            torch.empty(1, 32, dtype=torch.bool, device="cuda"),
        )
        mixed_outputs = tuple(torch.empty_like(value) for value in high_outputs)
        gqa4_outputs = (
            torch.empty_like(high_outputs[0]),
            torch.empty_like(high_outputs[2]),
            torch.empty_like(high_outputs[3]),
            torch.empty_like(high_outputs[4]),
        )
        sorted_outputs = tuple(
            torch.empty_like(value) for value in high_outputs
        )

        def high_retrieve() -> tuple[torch.Tensor, ...]:
            return varbit_cuda.sampled_threshold_compact_out(
                query_codes,
                query_scales,
                high_codes,
                high_scales,
                high_meta["bit_allocations"],
                high_meta["code_offsets"],
                high_meta["scale_offsets"],
                high_meta["code_bases"],
                high_meta["scale_bases"],
                high_meta["code_strides"],
                high_meta["scale_strides"],
                *high_outputs,
                history_count,
                sample_count,
                threshold_fraction,
            )

        def mixed_retrieve() -> tuple[torch.Tensor, ...]:
            return mixedblock_cuda.sampled_threshold_compact_out(
                query_codes,
                query_scales,
                mixed_codes,
                mixed_scales,
                mixed_meta,
                *mixed_outputs,
                history_count,
                sample_count,
                threshold_fraction,
            )

        def gqa4_retrieve() -> tuple[torch.Tensor, ...]:
            return (
                mixedblock_cuda.sampled_threshold_compact_gqa4_indices_out(
                    query_codes,
                    query_scales,
                    mixed_codes,
                    mixed_scales,
                    mixed_meta,
                    *gqa4_outputs,
                    history_count,
                    sample_count,
                    threshold_fraction,
                )
            )

        def sorted_retrieve() -> tuple[torch.Tensor, ...]:
            return mixedblock_cuda.sortedblock_sampled_threshold_compact_out(
                query_codes,
                query_scales,
                sorted_codes,
                sorted_scales,
                sorted_meta,
                *sorted_outputs,
                history_count,
                sample_count,
                threshold_fraction,
            )

        high_retrieve()
        mixed_retrieve()
        gqa4_retrieve()
        sorted_retrieve()

        def high_complete() -> torch.Tensor:
            indices, _, counts, _, _ = high_retrieve()
            return qabs_cuda.final_attention_ragged_self_split(
                query,
                key,
                value,
                indices,
                counts,
                128.0**-0.5,
                16,
            )

        def mixed_complete() -> torch.Tensor:
            indices, _, counts, _, _ = mixed_retrieve()
            return qabs_cuda.final_attention_ragged_self_split(
                query,
                key,
                value,
                indices,
                counts,
                128.0**-0.5,
                16,
            )

        def gqa4_complete() -> torch.Tensor:
            indices, counts, _, _ = gqa4_retrieve()
            return qabs_cuda.final_attention_ragged_self_split(
                query,
                key,
                value,
                indices,
                counts,
                128.0**-0.5,
                16,
            )

        def sorted_complete() -> torch.Tensor:
            indices, _, counts, _, _ = sorted_retrieve()
            return qabs_cuda.final_attention_ragged_self_split(
                query,
                key,
                value,
                indices,
                counts,
                128.0**-0.5,
                16,
            )

        def full_attention() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2),
                full_key,
                full_value,
            )

        iterations = min(
            args.iterations,
            30 if history_count >= 65536 else args.iterations,
        )
        high_retrieval_ms = measure_ms(
            high_retrieve, args.warmup, iterations
        )
        mixed_retrieval_ms = measure_ms(
            mixed_retrieve, args.warmup, iterations
        )
        gqa4_retrieval_ms = measure_ms(
            gqa4_retrieve, args.warmup, iterations
        )
        sorted_retrieval_ms = measure_ms(
            sorted_retrieve, args.warmup, iterations
        )
        high_complete_ms = measure_ms(
            high_complete, args.warmup, iterations
        )
        mixed_complete_ms = measure_ms(
            mixed_complete, args.warmup, iterations
        )
        gqa4_complete_ms = measure_ms(
            gqa4_complete, args.warmup, iterations
        )
        sorted_complete_ms = measure_ms(
            sorted_complete, args.warmup, iterations
        )
        full_ms = measure_ms(
            full_attention,
            min(10, args.warmup),
            min(20, iterations),
        )
        high_counts = high_outputs[2].float()
        mixed_counts = mixed_outputs[2].float()
        gqa4_counts = gqa4_outputs[1].float()
        sorted_counts = sorted_outputs[2].float()
        mixed_reference_output = mixed_complete()
        mixed_repeat_output = qabs_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            mixed_outputs[0],
            mixed_outputs[2],
            128.0**-0.5,
            16,
        )
        gqa4_output = gqa4_complete()
        gqa4_repeat_output = qabs_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            gqa4_outputs[0],
            gqa4_outputs[1],
            128.0**-0.5,
            16,
        )
        sorted_mixed_indices = mixed_outputs[0].clone()
        sorted_gqa4_indices = gqa4_outputs[0].clone()
        set_recalls = []
        set_precisions = []
        for row in range(32):
            mixed_count = int(mixed_outputs[2].reshape(-1)[row].item())
            gqa4_count = int(gqa4_outputs[1].reshape(-1)[row].item())
            mixed_indices = mixed_outputs[0].reshape(32, -1)[
                row, :mixed_count
            ]
            gqa4_indices = gqa4_outputs[0].reshape(32, -1)[
                row, :gqa4_count
            ]
            mixed_set = set(mixed_indices.detach().cpu().tolist())
            gqa4_set = set(gqa4_indices.detach().cpu().tolist())
            intersection = len(mixed_set & gqa4_set)
            set_recalls.append(intersection / max(1, len(mixed_set)))
            set_precisions.append(intersection / max(1, len(gqa4_set)))
            sorted_mixed_indices.reshape(32, -1)[row, :mixed_count] = (
                torch.sort(mixed_indices).values
            )
            sorted_gqa4_indices.reshape(32, -1)[row, :gqa4_count] = (
                torch.sort(gqa4_indices).values
            )
        sorted_mixed_output = qabs_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            sorted_mixed_indices,
            mixed_outputs[2],
            128.0**-0.5,
            16,
        )
        sorted_gqa4_output = qabs_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            sorted_gqa4_indices,
            gqa4_outputs[1],
            128.0**-0.5,
            16,
        )
        mixed_unsplit_output = qabs_cuda.final_attention_ragged_self(
            query,
            key,
            value,
            mixed_outputs[0],
            mixed_outputs[2],
            128.0**-0.5,
        )
        gqa4_unsplit_output = qabs_cuda.final_attention_ragged_self(
            query,
            key,
            value,
            gqa4_outputs[0],
            gqa4_outputs[1],
            128.0**-0.5,
        )
        mixed_metadata_bytes = (
            2 * mixed_meta["block_hot_prefix"].numel()
            + 8 * mixed_meta["head_code_bases"].numel()
            + 8 * mixed_meta["head_scale_bases"].numel()
        )
        mixed_index_bytes = (
            mixed_code_count
            + 2 * mixed_scale_count
            + mixed_metadata_bytes
        )
        sorted_metadata_bytes = (
            8 * sorted_meta["head_code_bases"].numel()
            + 8 * sorted_meta["head_scale_bases"].numel()
            + 4 * sorted_meta["original_blocks"].numel()
        )
        sorted_index_bytes = (
            sorted_code_count
            + 2 * sorted_scale_count
            + sorted_metadata_bytes
        )
        row = {
            "history_count": history_count,
            "target_fraction": fraction,
            "sample_count": sample_count,
            "high_candidate_fraction": float(
                (high_counts / history_count).mean().item()
            ),
            "mixed_candidate_fraction": float(
                (mixed_counts / history_count).mean().item()
            ),
            "gqa4_candidate_fraction": float(
                (gqa4_counts / history_count).mean().item()
            ),
            "sorted_candidate_fraction": float(
                (sorted_counts / history_count).mean().item()
            ),
            "high_retrieval_ms": high_retrieval_ms,
            "mixed_retrieval_ms": mixed_retrieval_ms,
            "gqa4_retrieval_ms": gqa4_retrieval_ms,
            "sorted_retrieval_ms": sorted_retrieval_ms,
            "high_complete_ms": high_complete_ms,
            "mixed_complete_ms": mixed_complete_ms,
            "gqa4_complete_ms": gqa4_complete_ms,
            "sorted_complete_ms": sorted_complete_ms,
            "full_ms": full_ms,
            "high_attention_speedup": full_ms / high_complete_ms,
            "mixed_attention_speedup": full_ms / mixed_complete_ms,
            "gqa4_attention_speedup": full_ms / gqa4_complete_ms,
            "gqa4_vs_mixed_retrieval_speedup": (
                mixed_retrieval_ms / gqa4_retrieval_ms
            ),
            "gqa4_vs_mixed_complete_speedup": (
                mixed_complete_ms / gqa4_complete_ms
            ),
            "gqa4_candidate_count_max_abs_diff": float(
                (mixed_counts - gqa4_counts).abs().max().item()
            ),
            "gqa4_candidate_set_recall": float(
                sum(set_recalls) / len(set_recalls)
            ),
            "gqa4_candidate_set_precision": float(
                sum(set_precisions) / len(set_precisions)
            ),
            "gqa4_output_max_abs_error": float(
                (mixed_reference_output - gqa4_output).abs().max().item()
            ),
            "mixed_split_repeat_max_abs_error": float(
                (
                    mixed_reference_output - mixed_repeat_output
                ).abs().max().item()
            ),
            "gqa4_split_repeat_max_abs_error": float(
                (gqa4_output - gqa4_repeat_output).abs().max().item()
            ),
            "sorted_split_output_max_abs_error": float(
                (
                    sorted_mixed_output - sorted_gqa4_output
                ).abs().max().item()
            ),
            "unsplit_output_max_abs_error": float(
                (
                    mixed_unsplit_output - gqa4_unsplit_output
                ).abs().max().item()
            ),
            "sorted_attention_speedup": full_ms / sorted_complete_ms,
            "mixed_vs_high_complete_speedup": (
                high_complete_ms / mixed_complete_ms
            ),
            "sorted_vs_high_complete_speedup": (
                high_complete_ms / sorted_complete_ms
            ),
            "sorted_vs_mixed_complete_speedup": (
                mixed_complete_ms / sorted_complete_ms
            ),
            "high_index_ratio_of_full_kv": (
                (
                    int(high_meta["total_code_bytes"])
                    + 2 * int(high_meta["total_scale_values"])
                )
                / (8 * history_count * 512)
            ),
            "mixed_index_ratio_of_full_kv": (
                mixed_index_bytes / (8 * history_count * 512)
            ),
            "sorted_index_ratio_of_full_kv": (
                sorted_index_bytes / (8 * history_count * 512)
            ),
            "mixed_payload_bytes_per_token_head": (
                (mixed_code_count + 2 * mixed_scale_count)
                / (8 * history_count)
            ),
            "mixed_metadata_bytes_per_token_head": (
                mixed_metadata_bytes / (8 * history_count)
            ),
            "sorted_payload_bytes_per_token_head": (
                (sorted_code_count + 2 * sorted_scale_count)
                / (8 * history_count)
            ),
            "sorted_metadata_bytes_per_token_head": (
                sorted_metadata_bytes / (8 * history_count)
            ),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
        del (
            projected_query,
            query_codes,
            query_scales,
            query,
            key,
            value,
            full_key,
            full_value,
            high_codes,
            high_scales,
            mixed_codes,
            mixed_scales,
            sorted_codes,
            sorted_scales,
        )
        torch.cuda.empty_cache()

    result = {
        "config": vars(args) | {"output": str(args.output)},
        "scope": (
            "One decode attention layer. Both methods include sampled "
            "threshold, full index scan, compaction, exact sparse QK/V, and "
            "V aggregation; index construction is excluded."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
