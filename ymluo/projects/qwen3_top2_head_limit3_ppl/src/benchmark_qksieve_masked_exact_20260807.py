#!/usr/bin/env python
"""Test a quality-equivalent mask-to-exact-attention QKSieve data path.

The existing deterministic selector emits both a bitmask and an int64 candidate
array.  This benchmark holds that selector fixed and compares:

1. the deployed exact attention that consumes the int64 candidate array; and
2. a new GQA-4 exact attention kernel that consumes the same bitmask directly.

The test accepts an optimization only when both paths consume the same selected
tokens and their final ValueSketch-combined outputs agree within BF16/FP16
rounding tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch

import mixedblock_spectral_cuda_20260729 as mixed_cuda
import qksieve_masked_valuesketch_cuda_20260807 as masked_cuda
import qksieve_query_cuda_20260728 as query_cuda
import qksieve_valuesketch_cuda_20260801 as sketch_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_variablebit_spectral_attention_20260727 import (
    ALLOCATION_PROFILES,
)
from validate_qksieve_valuesketch_deterministic_20260804 import (
    launch_deterministic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="32768,65536,131072")
    parser.add_argument("--split_tokens", default="512,1024,2048")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--tail_alpha", type=float, default=1.0)
    parser.add_argument("--skip_mask", action="store_true")
    parser.add_argument(
        "--cache_capacity_tokens",
        type=int,
        default=0,
        help="Physical preallocated KV capacity; zero uses exact logical size.",
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


def target_count(history: int) -> int:
    return min(history, 1280, max(256, math.ceil(0.06 * history)))


def c64_sample_count(history: int, selected: int) -> int:
    fraction = selected / history
    raw = math.ceil(64.0 / fraction)
    return min(8192, max(256, 256 * math.ceil(raw / 256)))


def candidate_capacity(history: int, selected: int, samples: int) -> int:
    fraction = selected / history
    upper_fraction = min(
        1.0,
        fraction
        + 6.0 * math.sqrt(fraction * (1.0 - fraction) / samples),
    )
    return min(history, max(selected + 1, math.ceil(upper_fraction * history)))


def allocate_selector_outputs(
    capacity: int, rank: int, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "indices": torch.empty(
            1, 32, capacity, dtype=torch.int64, device=device
        ),
        "counts": torch.empty(1, 32, dtype=torch.int64, device=device),
        "thresholds": torch.empty(
            1, 32, dtype=torch.float32, device=device
        ),
        "overflow": torch.empty(1, 32, dtype=torch.bool, device=device),
        "selected_denominator": torch.empty(
            1, 32, dtype=torch.float32, device=device
        ),
        "tail_denominator": torch.empty(
            1, 32, dtype=torch.float32, device=device
        ),
        "tail_coefficients": torch.empty(
            1, 32, rank, dtype=torch.float32, device=device
        ),
    }


def union_statistics(selection_masks: torch.Tensor, history: int) -> dict[str, float]:
    masks = selection_masks.reshape(1, 8, 4, -1).cpu()
    per_head = 0
    union = 0
    for kv_head in range(8):
        for word in range(masks.shape[-1]):
            group_words = [
                int(masks[0, kv_head, group, word].item()) & 0xFFFFFFFF
                for group in range(4)
            ]
            valid_bits = max(0, min(32, history - word * 32))
            valid_mask = (
                0xFFFFFFFF if valid_bits == 32 else (1 << valid_bits) - 1
            )
            per_head += sum((item & valid_mask).bit_count() for item in group_words)
            union += (
                (group_words[0] | group_words[1] | group_words[2] | group_words[3])
                & valid_mask
            ).bit_count()
    return {
        "selected_events": float(per_head),
        "gqa_union_tokens": float(union),
        "gqa_reuse": float(per_head / max(union, 1)),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    lengths = sorted({int(value) for value in args.lengths.split(",")})
    split_sizes = sorted(
        {int(value) for value in args.split_tokens.split(",")}
    )
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("lengths must contain positive integers")
    if not split_sizes or any(not 0 < size <= 2048 for size in split_sizes):
        raise ValueError("split_tokens must contain values in [1,2048]")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    rank = 16
    block_size = 256
    scaling = 128.0**-0.5
    allocation = ALLOCATION_PROFILES["qmse_total_b15"].unsqueeze(0).cuda()
    selector_extension = mixed_cuda.load_extension()
    rows: list[dict[str, object]] = []

    for history in lengths:
        selected = target_count(history)
        selected_fraction = selected / history
        samples = c64_sample_count(history, selected)
        capacity = candidate_capacity(history, selected, samples)
        block_count = math.ceil(history / block_size)

        query = torch.randn(1, 32, 128, dtype=dtype, device=device)
        cache_capacity = args.cache_capacity_tokens or (history + 1)
        if cache_capacity < history + 1:
            raise ValueError(
                "cache_capacity_tokens must cover history plus the self token"
            )
        key_storage = torch.randn(
            1, 8, cache_capacity, 128, dtype=dtype, device=device
        )
        value_storage = torch.randn_like(key_storage)
        key = key_storage[:, :, : history + 1, :]
        value = value_storage[:, :, : history + 1, :]
        key_basis = torch.randn(1, 8, 128, 128, dtype=dtype, device=device)
        index = varbit_cuda.allocate_packed_index(allocation, history, dtype)
        index["packed_codes"].random_(0, 256)
        index["key_scales"].uniform_(0.01, 0.1)
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
        value_mean = torch.randn(1, 8, 128, dtype=dtype, device=device)
        value_basis = (
            torch.randn(1, 8, 128, rank, dtype=dtype, device=device)
            / math.sqrt(128.0)
        ).contiguous()
        selector = allocate_selector_outputs(capacity, rank, device)
        selection_masks = torch.empty(
            1,
            32,
            block_count * 8,
            dtype=torch.int32,
            device=device,
        )
        tail_partials = torch.empty(
            1,
            32,
            block_count,
            rank + 2,
            dtype=torch.float32,
            device=device,
        )
        index_workspace = sketch_cuda.allocate_attention_workspace(
            query, capacity
        )
        tiled_workspace = sketch_cuda.allocate_attention_workspace(
            query, capacity
        )
        mask_workspaces = (
            {}
            if args.skip_mask
            else {
                split_tokens: masked_cuda.allocate_workspace(
                    query, history, split_tokens
                )
                for split_tokens in split_sizes
            }
        )

        def select() -> None:
            grouped_query = query.reshape(1, 8, 4, 128)
            query_codes, query_scales = query_cuda.project_quantize(
                grouped_query, key_basis
            )
            launch_deterministic(
                selector_extension,
                query_codes,
                query_scales,
                index,
                value_codes,
                value_minimum,
                value_scale,
                selection_masks,
                tail_partials,
                selector,
                history,
                samples,
                selected_fraction,
                scaling,
            )

        def index_attention() -> torch.Tensor:
            return sketch_cuda.exact_selected_plus_tail_out(
                query,
                key,
                value,
                selector["indices"],
                selector["counts"],
                selector["thresholds"],
                selector["tail_denominator"],
                selector["tail_coefficients"],
                value_mean,
                value_basis,
                index_workspace,
                scaling,
                args.tail_alpha,
            )

        def legacy_attention() -> torch.Tensor:
            return sketch_cuda.exact_selected_plus_tail(
                query,
                key,
                value,
                selector["indices"],
                selector["counts"],
                selector["thresholds"],
                selector["tail_denominator"],
                selector["tail_coefficients"],
                value_mean,
                value_basis,
                scaling,
                args.tail_alpha,
            )

        def mask_attention(split_tokens: int) -> torch.Tensor:
            return masked_cuda.exact_masked_gqa4_plus_tail_out(
                query,
                key,
                value,
                selection_masks,
                selector["thresholds"],
                selector["tail_denominator"],
                selector["tail_coefficients"],
                value_mean,
                value_basis,
                mask_workspaces[split_tokens],
                scaling,
                args.tail_alpha,
                split_tokens,
            )

        def tiled_attention() -> torch.Tensor:
            return sketch_cuda.exact_selected_plus_tail_tiled_out(
                query,
                key,
                value,
                selector["indices"],
                selector["counts"],
                selector["thresholds"],
                selector["tail_denominator"],
                selector["tail_coefficients"],
                value_mean,
                value_basis,
                tiled_workspace,
                scaling,
                args.tail_alpha,
            )

        def tiled_contiguous_attention() -> torch.Tensor:
            return sketch_cuda.exact_selected_plus_tail_tiled_out(
                query,
                key.contiguous(),
                value.contiguous(),
                selector["indices"],
                selector["counts"],
                selector["thresholds"],
                selector["tail_denominator"],
                selector["tail_coefficients"],
                value_mean,
                value_basis,
                tiled_workspace,
                scaling,
                args.tail_alpha,
            )

        def index_complete() -> torch.Tensor:
            select()
            return index_attention()

        def mask_complete(split_tokens: int) -> torch.Tensor:
            select()
            return mask_attention(split_tokens)

        def tiled_complete() -> torch.Tensor:
            select()
            return tiled_attention()

        def legacy_complete() -> torch.Tensor:
            select()
            return legacy_attention()

        def tiled_contiguous_complete() -> torch.Tensor:
            select()
            return tiled_contiguous_attention()

        select()
        torch.cuda.synchronize()
        if bool(selector["overflow"].any().item()):
            raise RuntimeError(
                f"candidate capacity {capacity} overflowed at history={history}"
            )
        reference = index_attention().clone()
        legacy = legacy_attention().clone()
        tiled = tiled_attention().clone()
        tiled_contiguous = tiled_contiguous_attention().clone()
        torch.cuda.synchronize()
        legacy_difference = (legacy.float() - reference.float()).abs()
        tiled_difference = (tiled.float() - reference.float()).abs()
        tiled_contiguous_difference = (
            tiled_contiguous.float() - reference.float()
        ).abs()
        tiled_denominator = reference.float().abs().clamp_min(1.0e-3)
        tiled_attention_ms = measure_ms(
            tiled_attention, args.warmup, args.iterations
        )
        tiled_complete_ms = measure_ms(
            tiled_complete, args.warmup, args.iterations
        )
        legacy_attention_ms = measure_ms(
            legacy_attention, args.warmup, args.iterations
        )
        legacy_complete_ms = measure_ms(
            legacy_complete, args.warmup, args.iterations
        )
        tiled_contiguous_attention_ms = measure_ms(
            tiled_contiguous_attention, args.warmup, args.iterations
        )
        tiled_contiguous_complete_ms = measure_ms(
            tiled_contiguous_complete, args.warmup, args.iterations
        )
        tiled_attention()
        tiled_repeat_reference = tiled_workspace["output"].clone()
        tiled_attention()
        selector_ms = measure_ms(select, args.warmup, args.iterations)
        index_attention_ms = measure_ms(
            index_attention, args.warmup, args.iterations
        )
        index_complete_ms = measure_ms(
            index_complete, args.warmup, args.iterations
        )
        count_values = selector["counts"].float()
        row: dict[str, object] = {
            "history_tokens": history,
            "cache_capacity_tokens": cache_capacity,
            "key_is_contiguous": key.is_contiguous(),
            "key_head_stride": int(key.stride(1)),
            "target_tokens": selected,
            "target_fraction": selected_fraction,
            "sample_count": samples,
            "candidate_capacity": capacity,
            "candidate_count_mean": float(count_values.mean().item()),
            "candidate_count_min": int(selector["counts"].min().item()),
            "candidate_count_max": int(selector["counts"].max().item()),
            "selector_ms": selector_ms,
            "index_attention_ms": index_attention_ms,
            "index_complete_ms": index_complete_ms,
            "tiled_attention_ms": tiled_attention_ms,
            "tiled_complete_ms": tiled_complete_ms,
            "tiled_attention_speedup": index_attention_ms
            / tiled_attention_ms,
            "tiled_complete_speedup": index_complete_ms
            / tiled_complete_ms,
            "tiled_max_abs_error": float(tiled_difference.max().item()),
            "tiled_mean_abs_error": float(tiled_difference.mean().item()),
            "tiled_max_relative_error": float(
                (tiled_difference / tiled_denominator).max().item()
            ),
            "tiled_top1_equal": bool(
                torch.equal(
                    tiled.reshape(-1, 128).argmax(dim=-1),
                    reference.reshape(-1, 128).argmax(dim=-1),
                )
            ),
            "tiled_bitwise_repeat": bool(
                torch.equal(
                    tiled_repeat_reference,
                    tiled_workspace["output"],
                )
            ),
            "legacy_attention_ms": legacy_attention_ms,
            "legacy_complete_ms": legacy_complete_ms,
            "legacy_max_abs_error": float(legacy_difference.max().item()),
            "tiled_contiguous_attention_ms": tiled_contiguous_attention_ms,
            "tiled_contiguous_complete_ms": tiled_contiguous_complete_ms,
            "tiled_contiguous_max_abs_error": float(
                tiled_contiguous_difference.max().item()
            ),
            **union_statistics(selection_masks, history),
        }
        for split_tokens in (() if args.skip_mask else split_sizes):
            candidate = mask_attention(split_tokens).clone()
            torch.cuda.synchronize()
            difference = (candidate.float() - reference.float()).abs()
            denominator = reference.float().abs().clamp_min(1.0e-3)
            mask_ms = measure_ms(
                lambda size=split_tokens: mask_attention(size),
                args.warmup,
                args.iterations,
            )
            complete_ms = measure_ms(
                lambda size=split_tokens: mask_complete(size),
                args.warmup,
                args.iterations,
            )
            mask_attention(split_tokens)
            repeat_reference = mask_workspaces[split_tokens]["output"].clone()
            mask_attention(split_tokens)
            deterministic = bool(
                torch.equal(
                    repeat_reference,
                    mask_workspaces[split_tokens]["output"],
                )
            )
            prefix = f"mask_s{split_tokens}"
            row.update(
                {
                    f"{prefix}_attention_ms": mask_ms,
                    f"{prefix}_complete_ms": complete_ms,
                    f"{prefix}_attention_speedup": row["index_attention_ms"]
                    / mask_ms,
                    f"{prefix}_complete_speedup": row["index_complete_ms"]
                    / complete_ms,
                    f"{prefix}_max_abs_error": float(difference.max().item()),
                    f"{prefix}_mean_abs_error": float(difference.mean().item()),
                    f"{prefix}_max_relative_error": float(
                        (difference / denominator).max().item()
                    ),
                    f"{prefix}_top1_equal": bool(
                        torch.equal(
                            candidate.reshape(-1, 128).argmax(dim=-1),
                            reference.reshape(-1, 128).argmax(dim=-1),
                        )
                    ),
                    f"{prefix}_bitwise_repeat": deterministic,
                    f"{prefix}_workspace_bytes": sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in mask_workspaces[split_tokens].values()
                        if tensor.data_ptr()
                        != mask_workspaces[split_tokens]["output_view"].data_ptr()
                    ),
                }
            )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)

        del (
            query,
            key,
            value,
            key_storage,
            value_storage,
            key_basis,
            index,
            value_codes,
            value_minimum,
            value_scale,
            value_mean,
            value_basis,
            selector,
            selection_masks,
            tail_partials,
            index_workspace,
            tiled_workspace,
            mask_workspaces,
        )
        torch.cuda.empty_cache()

    result = {
        "schema": "qksieve_masked_exact_benchmark_v1",
        "seed": args.seed,
        "dtype": args.dtype,
        "tail_alpha": args.tail_alpha,
        "quality_contract": (
            "same deterministic selection masks and ValueSketch tail; only "
            "the exact-attention consumer changes"
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
