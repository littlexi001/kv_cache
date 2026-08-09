#!/usr/bin/env python
"""Benchmark a two-index frequency-tiered QKSieve decode path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import benchmark_variablebit_spectral_attention_20260727 as varbit_bench
import qabs_cuda_kernels as sparse_cuda
import qksieve_query_cuda_20260728 as query_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda


HIGH_ALLOCATION = varbit_bench.ALLOCATION_PROFILES["qmse_total_b15"]
LOW_192_ALLOCATION = torch.tensor(
    [[4, 4, 1, 0, 0, 0, 0, 0]] * 8,
    dtype=torch.int8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="8192,16384,32768,65536,131072")
    parser.add_argument("--hot_fraction", type=float, default=0.04)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def target_count(history_tokens: int) -> int:
    return min(
        history_tokens,
        1280,
        max(256, math.ceil(0.06 * history_tokens)),
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
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    stop.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(stop)) / iterations


def random_index(
    allocation: torch.Tensor,
    token_count: int,
) -> tuple[dict[str, torch.Tensor | int | float], torch.Tensor, torch.Tensor]:
    metadata = varbit_bench.make_metadata(
        allocation.unsqueeze(0),
        token_count,
    )
    codes = torch.randint(
        0,
        256,
        (int(metadata["total_code_bytes"]),),
        dtype=torch.uint8,
        device="cuda",
    )
    scales = (
        torch.rand(
            int(metadata["total_scale_values"]),
            dtype=torch.float16,
            device="cuda",
        )
        * 0.03
        + 0.001
    )
    return metadata, codes, scales


def scan_index(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    metadata: dict[str, torch.Tensor | int | float],
    packed_codes: torch.Tensor,
    key_scales: torch.Tensor,
    token_count: int,
) -> torch.Tensor:
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
        token_count,
    ).reshape(1, 32, token_count)


def exact_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    selected_count = int(indices.shape[-1])
    if selected_count >= 1280:
        return sparse_cuda.final_attention_ragged_self_split(
            query, key, value, indices, counts, scaling, 4
        )
    if selected_count >= 900:
        return sparse_cuda.final_attention_ragged_self_split(
            query, key, value, indices, counts, scaling, 2
        )
    return sparse_cuda.final_attention_ragged_self(
        query, key, value, indices, counts, scaling
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(20260729)
    rows = []
    for history_tokens in (
        int(item) for item in args.lengths.split(",") if item.strip()
    ):
        selected_count = target_count(history_tokens)
        hot_count = max(1, math.ceil(history_tokens * args.hot_fraction))
        cold_count = history_tokens - hot_count
        baseline_meta, baseline_codes, baseline_scales = random_index(
            HIGH_ALLOCATION, history_tokens
        )
        low_full_meta, low_full_codes, low_full_scales = random_index(
            LOW_192_ALLOCATION, history_tokens
        )
        cold_meta, cold_codes, cold_scales = random_index(
            LOW_192_ALLOCATION, cold_count
        )
        hot_meta, hot_codes, hot_scales = random_index(
            HIGH_ALLOCATION, hot_count
        )
        cold_ids = (
            torch.arange(cold_count, dtype=torch.long, device="cuda")
            .view(1, 1, cold_count)
            .expand(1, 8, cold_count)
            .repeat_interleave(4, dim=1)
            .contiguous()
        )
        hot_ids = (
            torch.arange(
                cold_count,
                history_tokens,
                dtype=torch.long,
                device="cuda",
            )
            .view(1, 1, hot_count)
            .expand(1, 8, hot_count)
            .repeat_interleave(4, dim=1)
            .contiguous()
        )
        query = torch.randn(
            1, 32, 128, dtype=torch.float16, device="cuda"
        )
        grouped_query = query.reshape(1, 8, 4, 128)
        query_basis = torch.randn(
            1, 8, 128, 128, dtype=torch.float16, device="cuda"
        )
        key = torch.randn(
            1, 8, history_tokens, 128, dtype=torch.float16, device="cuda"
        )
        value = torch.randn_like(key)
        full_key = key.repeat_interleave(4, dim=1)
        full_value = value.repeat_interleave(4, dim=1)
        scaling = 128.0**-0.5
        counts = torch.full(
            (1, 32),
            selected_count,
            dtype=torch.long,
            device="cuda",
        )

        def prepare_query() -> tuple[torch.Tensor, torch.Tensor]:
            return query_cuda.project_quantize(grouped_query, query_basis)

        query_codes, query_scales = prepare_query()

        def baseline_scan() -> torch.Tensor:
            return scan_index(
                query_codes,
                query_scales,
                baseline_meta,
                baseline_codes,
                baseline_scales,
                history_tokens,
            )

        def tier_scans() -> tuple[torch.Tensor, torch.Tensor]:
            return (
                scan_index(
                    query_codes,
                    query_scales,
                    cold_meta,
                    cold_codes,
                    cold_scales,
                    cold_count,
                ),
                scan_index(
                    query_codes,
                    query_scales,
                    hot_meta,
                    hot_codes,
                    hot_scales,
                    hot_count,
                ),
            )

        def low_full_scan() -> torch.Tensor:
            return scan_index(
                query_codes,
                query_scales,
                low_full_meta,
                low_full_codes,
                low_full_scales,
                history_tokens,
            )

        baseline_scores = baseline_scan()
        low_full_scores = low_full_scan()
        cold_score, hot_score = tier_scans()

        def baseline_select_from(
            scores: torch.Tensor = baseline_scores,
        ) -> torch.Tensor:
            return torch.topk(
                scores,
                k=selected_count,
                dim=-1,
                sorted=False,
            ).indices

        def tier_select_from(
            cold_scores: torch.Tensor = cold_score,
            hot_scores: torch.Tensor = hot_score,
        ) -> torch.Tensor:
            local_cold_count = min(selected_count, cold_count)
            local_hot_count = min(selected_count, hot_count)
            cold_values, cold_local = torch.topk(
                cold_scores,
                k=local_cold_count,
                dim=-1,
                sorted=False,
            )
            hot_values, hot_local = torch.topk(
                hot_scores,
                k=local_hot_count,
                dim=-1,
                sorted=False,
            )
            local_values = torch.cat((cold_values, hot_values), dim=-1)
            global_ids = torch.cat(
                (
                    cold_ids.gather(-1, cold_local),
                    hot_ids.gather(-1, hot_local),
                ),
                dim=-1,
            )
            winners = torch.topk(
                local_values,
                k=selected_count,
                dim=-1,
                sorted=False,
            ).indices
            return global_ids.gather(-1, winners).contiguous()

        baseline_indices = baseline_select_from()
        def low_full_select_from(
            scores: torch.Tensor = low_full_scores,
        ) -> torch.Tensor:
            return torch.topk(
                scores,
                k=selected_count,
                dim=-1,
                sorted=False,
            ).indices

        low_full_indices = low_full_select_from()
        tier_indices = tier_select_from()

        def baseline_selection_path() -> torch.Tensor:
            codes, scales = prepare_query()
            scores = scan_index(
                codes,
                scales,
                baseline_meta,
                baseline_codes,
                baseline_scales,
                history_tokens,
            )
            return baseline_select_from(scores)

        def tier_selection_path() -> torch.Tensor:
            codes, scales = prepare_query()
            cold_scores = scan_index(
                codes,
                scales,
                cold_meta,
                cold_codes,
                cold_scales,
                cold_count,
            )
            hot_scores = scan_index(
                codes,
                scales,
                hot_meta,
                hot_codes,
                hot_scales,
                hot_count,
            )
            return tier_select_from(cold_scores, hot_scores)

        def low_full_selection_path() -> torch.Tensor:
            codes, scales = prepare_query()
            scores = scan_index(
                codes,
                scales,
                low_full_meta,
                low_full_codes,
                low_full_scales,
                history_tokens,
            )
            return low_full_select_from(scores)

        def baseline_complete() -> torch.Tensor:
            return exact_sparse_attention(
                query,
                key,
                value,
                baseline_selection_path(),
                counts,
                scaling,
            )

        def tier_complete() -> torch.Tensor:
            return exact_sparse_attention(
                query,
                key,
                value,
                tier_selection_path(),
                counts,
                scaling,
            )

        def low_full_complete() -> torch.Tensor:
            return exact_sparse_attention(
                query,
                key,
                value,
                low_full_selection_path(),
                counts,
                scaling,
            )

        def full_attention() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2),
                full_key,
                full_value,
                is_causal=False,
                scale=scaling,
            )

        iterations = min(
            args.iterations,
            15 if history_tokens >= 65536 else args.iterations,
        )
        warmup = min(args.warmup, iterations)
        query_ms = measure_ms(prepare_query, warmup, iterations)
        baseline_scan_ms = measure_ms(
            baseline_scan, warmup, iterations
        )
        low_full_scan_ms = measure_ms(
            low_full_scan, warmup, iterations
        )
        tier_scan_ms = measure_ms(tier_scans, warmup, iterations)
        baseline_topk_ms = measure_ms(
            baseline_select_from, warmup, iterations
        )
        low_full_topk_ms = measure_ms(
            low_full_select_from, warmup, iterations
        )
        tier_topk_merge_ms = measure_ms(
            tier_select_from, warmup, iterations
        )
        baseline_selection_ms = measure_ms(
            baseline_selection_path, warmup, iterations
        )
        low_full_selection_ms = measure_ms(
            low_full_selection_path, warmup, iterations
        )
        tier_selection_ms = measure_ms(
            tier_selection_path, warmup, iterations
        )
        baseline_complete_ms = measure_ms(
            baseline_complete, warmup, iterations
        )
        low_full_complete_ms = measure_ms(
            low_full_complete, warmup, iterations
        )
        tier_complete_ms = measure_ms(
            tier_complete, warmup, iterations
        )
        full_ms = measure_ms(
            full_attention,
            min(5, warmup),
            min(10, iterations),
        )
        baseline_bytes = int(baseline_meta["packed_bytes"])
        tier_packed_bytes = (
            int(cold_meta["packed_bytes"])
            + int(hot_meta["packed_bytes"])
        )
        dense_id_map_bytes = 4 * history_tokens * 8
        succinct_bitmap_bytes = math.ceil(history_tokens / 8) * 8
        rows.append(
            {
                "history_tokens": history_tokens,
                "selected_tokens": selected_count,
                "hot_fraction": args.hot_fraction,
                "baseline_index_bytes_per_token_head": (
                    baseline_bytes / history_tokens / 8
                ),
                "tier_index_bytes_per_token_head": (
                    (tier_packed_bytes + succinct_bitmap_bytes)
                    / history_tokens
                    / 8
                ),
                "tier_index_ratio_vs_baseline": (
                    (tier_packed_bytes + succinct_bitmap_bytes)
                    / baseline_bytes
                ),
                "tier_dense_id_map_bytes_per_token_head": (
                    (tier_packed_bytes + dense_id_map_bytes)
                    / history_tokens
                    / 8
                ),
                "tier_dense_id_map_ratio_vs_baseline": (
                    (tier_packed_bytes + dense_id_map_bytes)
                    / baseline_bytes
                ),
                "query_prepare_ms": query_ms,
                "baseline_scan_ms": baseline_scan_ms,
                "low192_single_scan_ms": low_full_scan_ms,
                "low192_scan_speedup": (
                    baseline_scan_ms / low_full_scan_ms
                ),
                "tier_two_scan_ms": tier_scan_ms,
                "scan_speedup": baseline_scan_ms / tier_scan_ms,
                "baseline_topk_ms": baseline_topk_ms,
                "low192_topk_ms": low_full_topk_ms,
                "tier_two_topk_merge_ms": tier_topk_merge_ms,
                "baseline_selection_ms": baseline_selection_ms,
                "low192_selection_ms": low_full_selection_ms,
                "low192_selection_speedup": (
                    baseline_selection_ms / low_full_selection_ms
                ),
                "tier_selection_ms": tier_selection_ms,
                "selection_speedup": baseline_selection_ms / tier_selection_ms,
                "baseline_complete_ms": baseline_complete_ms,
                "low192_complete_ms": low_full_complete_ms,
                "low192_vs_baseline_complete_speedup": (
                    baseline_complete_ms / low_full_complete_ms
                ),
                "tier_complete_ms": tier_complete_ms,
                "tier_vs_baseline_complete_speedup": (
                    baseline_complete_ms / tier_complete_ms
                ),
                "full_sdpa_ms": full_ms,
                "baseline_attention_speedup_vs_full": (
                    full_ms / baseline_complete_ms
                ),
                "low192_attention_speedup_vs_full": (
                    full_ms / low_full_complete_ms
                ),
                "tier_attention_speedup_vs_full": (
                    full_ms / tier_complete_ms
                ),
            }
        )
        del (
            baseline_codes,
            baseline_scales,
            low_full_codes,
            low_full_scales,
            cold_codes,
            cold_scales,
            hot_codes,
            hot_scales,
            query,
            grouped_query,
            query_basis,
            key,
            value,
            full_key,
            full_value,
            baseline_scores,
            low_full_scores,
            cold_score,
            hot_score,
            baseline_indices,
            low_full_indices,
            tier_indices,
        )
        torch.cuda.empty_cache()

    output = {
        "scope": (
            "One Qwen-style GQA layer. Tier path includes fused Query "
            "projection/quantization, separate low-rate cold and QKSieve-rate "
            "hot scans, two local top-k operations, 2k-to-k merge, 32-bit "
            "token-ID mapping, and exact sparse attention. Runtime uses a "
            "dense per-head ID map. Storage reports both that implemented "
            "map and the target one-bit tier bitmap requiring a fused "
            "rank/select mapper."
        ),
        "high_allocation": HIGH_ALLOCATION.tolist(),
        "low_192_allocation": LOW_192_ALLOCATION.tolist(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
