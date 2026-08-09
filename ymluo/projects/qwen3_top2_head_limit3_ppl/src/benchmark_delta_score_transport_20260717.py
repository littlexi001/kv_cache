from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import qabs_cuda_kernels as kernels


def measure(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / repeats


def benchmark_length(
    length: int,
    rank: int,
    batch_count: int,
    refresh_count: int,
    candidate_fraction: float,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    kv_heads = 8
    groups = 4
    projection_dim = 64
    chunk_count = projection_dim // 16
    packed = torch.randint(
        0,
        256,
        (batch_count, kv_heads, chunk_count, length, 8),
        dtype=torch.uint8,
        device="cuda",
    )
    scales = torch.rand(
        batch_count, kv_heads, length, 1, dtype=torch.float16, device="cuda"
    )
    projected_query = torch.randn(
        batch_count,
        kv_heads,
        groups,
        projection_dim,
        dtype=torch.float32,
        device="cuda",
    )
    previous_query = torch.randn_like(projected_query)
    projected_query_half = projected_query.half()
    previous_query_half = previous_query.half()
    spectral_weights = torch.rand(
        batch_count,
        kv_heads,
        projection_dim,
        dtype=torch.float32,
        device="cuda",
    )
    anchor_query = projected_query_half.clone()
    far_anchor_query = projected_query_half + 1.0
    refresh_mask = torch.empty(
        batch_count, kv_heads, dtype=torch.uint8, device="cuda"
    )
    gate_signal = torch.empty(
        batch_count, kv_heads, dtype=torch.float32, device="cuda"
    )
    refresh_indices = torch.empty(
        batch_count, kv_heads, dtype=torch.int32, device="cuda"
    )
    score_cache = torch.randn(
        batch_count,
        kv_heads * groups,
        length,
        dtype=torch.float16,
        device="cuda",
    )
    candidate_count = max(1, int(candidate_fraction * length + 0.999999))

    def full_refresh() -> None:
        query_scale = (
            projected_query.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
            / 127.0
        )
        query_codes = torch.round(projected_query / query_scale).clamp(
            -127, 127
        ).to(torch.int8)
        scores = kernels.pca_int4_chunked_prefix_scores(
            query_codes, packed, scales, length, projection_dim
        )
        scores = scores * query_scale.reshape(batch_count, kv_heads * groups, 1)
        score_cache.copy_(scores.to(score_cache.dtype))

    def delta_update() -> None:
        delta = projected_query - previous_query
        dimensions = torch.topk(delta.abs(), k=rank, dim=-1).indices
        selected = torch.gather(delta, -1, dimensions)
        delta_scale = (
            selected.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
            / 127.0
        )
        delta_codes = torch.round(delta / delta_scale).clamp(-127, 127).to(
            torch.int8
        )
        delta_scores = kernels.pca_int4_chunked_selected_scores(
            delta_codes,
            packed,
            scales,
            dimensions.to(torch.int32),
            length,
        )
        delta_scores = delta_scores * delta_scale.reshape(
            batch_count, kv_heads * groups, 1
        )
        score_cache.copy_((score_cache.float() + delta_scores).to(score_cache.dtype))

    def shared_delta_update() -> None:
        delta = projected_query - previous_query
        shared_dimensions = torch.topk(
            delta.float().square().sum(dim=2), k=rank, dim=-1
        ).indices
        dimensions = shared_dimensions.unsqueeze(2).expand(-1, -1, groups, -1)
        selected = torch.gather(delta, -1, dimensions)
        delta_scale = (
            selected.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
            / 127.0
        )
        delta_codes = torch.round(delta / delta_scale).clamp(-127, 127).to(
            torch.int8
        )
        delta_scores = kernels.pca_int4_chunked_shared_selected_scores(
            delta_codes,
            packed,
            scales,
            shared_dimensions.to(torch.int32),
            length,
        )
        delta_scores = delta_scores * delta_scale.reshape(
            batch_count, kv_heads * groups, 1
        )
        score_cache.copy_((score_cache.float() + delta_scores).to(score_cache.dtype))

    def fused_shared_delta_update() -> None:
        delta = projected_query - previous_query
        shared_dimensions = torch.topk(
            delta.float().square().sum(dim=2), k=rank, dim=-1
        ).indices
        dimensions = shared_dimensions.unsqueeze(2).expand(-1, -1, groups, -1)
        selected = torch.gather(delta, -1, dimensions)
        delta_scale = (
            selected.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
            / 127.0
        )
        delta_codes = torch.round(delta / delta_scale).clamp(-127, 127).to(
            torch.int8
        )
        kernels.pca_int4_chunked_shared_selected_add(
            delta_codes,
            packed,
            scales,
            shared_dimensions,
            delta_scale,
            score_cache,
            length,
        )

    def fused_fixed_tail_delta_update() -> None:
        delta = (projected_query - previous_query)[..., -rank:]
        delta_scale = (
            delta.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 127.0
        )
        delta_codes = torch.round(delta / delta_scale).clamp(-127, 127).to(
            torch.int8
        )
        kernels.pca_int4_chunked_contiguous_add(
            delta_codes,
            packed,
            scales,
            delta_scale,
            score_cache,
            length,
            projection_dim - rank,
        )

    def direct_fixed_tail_delta_update() -> None:
        kernels.pca_int4_chunked_contiguous_delta_add(
            projected_query_half,
            previous_query_half,
            packed,
            scales,
            score_cache,
            length,
            projection_dim - rank,
            rank,
        )

    def spectral_no_refresh_update() -> None:
        kernels.pca_int4_chunked_spectral_gated_delta_add(
            projected_query_half,
            previous_query_half,
            packed,
            scales,
            spectral_weights,
            anchor_query,
            refresh_mask,
            gate_signal,
            refresh_indices,
            score_cache,
            length,
            projection_dim - rank,
            0.08,
        )

    def reset_anchor() -> None:
        anchor_query.copy_(far_anchor_query)

    def spectral_all_refresh_update() -> None:
        anchor_query.copy_(far_anchor_query)
        kernels.pca_int4_chunked_spectral_gated_delta_add(
            projected_query_half,
            previous_query_half,
            packed,
            scales,
            spectral_weights,
            anchor_query,
            refresh_mask,
            gate_signal,
            refresh_indices,
            score_cache,
            length,
            projection_dim - rank,
            0.08,
        )

    def spectral_topk_update() -> None:
        kernels.pca_int4_chunked_spectral_gated_delta_add(
            projected_query_half,
            previous_query_half,
            packed,
            scales,
            spectral_weights,
            anchor_query,
            refresh_mask,
            gate_signal,
            refresh_indices,
            score_cache,
            length,
            projection_dim - rank,
            0.0,
            refresh_count,
        )

    def full_refresh_and_select() -> None:
        query_scale = (
            projected_query.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
            / 127.0
        )
        query_codes = torch.round(projected_query / query_scale).clamp(
            -127, 127
        ).to(torch.int8)
        scores = kernels.pca_int4_chunked_prefix_scores(
            query_codes, packed, scales, length, projection_dim
        )
        scores = scores * query_scale.reshape(batch_count, kv_heads * groups, 1)
        torch.topk(
            scores,
            k=candidate_count,
            dim=-1,
            largest=True,
            sorted=True,
        )

    def spectral_topk_update_and_select() -> None:
        spectral_topk_update()
        torch.topk(
            score_cache,
            k=candidate_count,
            dim=-1,
            largest=True,
            sorted=True,
        )

    def spectral_topk_update_cast_and_select() -> None:
        spectral_topk_update()
        torch.topk(
            score_cache.float(),
            k=candidate_count,
            dim=-1,
            largest=True,
            sorted=True,
        )

    def spectral_topk_update_select_then_sort() -> None:
        spectral_topk_update()
        selected_scores, selected_indices = torch.topk(
            score_cache,
            k=candidate_count,
            dim=-1,
            largest=True,
            sorted=False,
        )
        _, order = torch.sort(selected_scores, dim=-1, descending=True)
        torch.gather(selected_indices, -1, order)

    full_ms = measure(full_refresh, warmup, repeats)
    delta_ms = measure(delta_update, warmup, repeats)
    shared_delta_ms = measure(shared_delta_update, warmup, repeats)
    fused_shared_delta_ms = measure(fused_shared_delta_update, warmup, repeats)
    fused_fixed_tail_delta_ms = measure(
        fused_fixed_tail_delta_update, warmup, repeats
    )
    direct_fixed_tail_delta_ms = measure(
        direct_fixed_tail_delta_update, warmup, repeats
    )
    anchor_reset_ms = measure(reset_anchor, warmup, repeats)
    spectral_no_refresh_ms = measure(
        spectral_no_refresh_update, warmup, repeats
    )
    spectral_all_refresh_with_reset_ms = measure(
        spectral_all_refresh_update, warmup, repeats
    )
    spectral_topk_ms = measure(spectral_topk_update, warmup, repeats)
    full_select_ms = measure(full_refresh_and_select, warmup, repeats)
    spectral_select_ms = measure(
        spectral_topk_update_and_select, warmup, repeats
    )
    spectral_cast_select_ms = measure(
        spectral_topk_update_cast_and_select, warmup, repeats
    )
    spectral_select_sort_ms = measure(
        spectral_topk_update_select_then_sort, warmup, repeats
    )
    spectral_all_refresh_ms = max(
        0.0, spectral_all_refresh_with_reset_ms - anchor_reset_ms
    )
    return {
        "batch": batch_count,
        "length": length,
        "rank": rank,
        "spectral_refresh_count": refresh_count,
        "full_refresh_ms": full_ms,
        "delta_update_ms": delta_ms,
        "delta_vs_full_speedup": full_ms / delta_ms,
        "shared_delta_update_ms": shared_delta_ms,
        "shared_delta_vs_full_speedup": full_ms / shared_delta_ms,
        "shared_vs_independent_speedup": delta_ms / shared_delta_ms,
        "fused_shared_delta_update_ms": fused_shared_delta_ms,
        "fused_shared_delta_vs_full_speedup": full_ms / fused_shared_delta_ms,
        "fused_vs_unfused_shared_speedup": shared_delta_ms / fused_shared_delta_ms,
        "fused_fixed_tail_delta_update_ms": fused_fixed_tail_delta_ms,
        "fused_fixed_tail_delta_vs_full_speedup": (
            full_ms / fused_fixed_tail_delta_ms
        ),
        "fixed_tail_vs_dynamic_shared_speedup": (
            fused_shared_delta_ms / fused_fixed_tail_delta_ms
        ),
        "direct_fixed_tail_delta_update_ms": direct_fixed_tail_delta_ms,
        "direct_fixed_tail_delta_vs_full_speedup": (
            full_ms / direct_fixed_tail_delta_ms
        ),
        "direct_vs_quantized_fixed_tail_speedup": (
            fused_fixed_tail_delta_ms / direct_fixed_tail_delta_ms
        ),
        "spectral_gate_tail_update_ms": spectral_no_refresh_ms,
        "spectral_gate_full_refresh_ms": spectral_all_refresh_ms,
        "spectral_gate_full_refresh_with_anchor_reset_ms": (
            spectral_all_refresh_with_reset_ms
        ),
        "anchor_reset_ms": anchor_reset_ms,
        "spectral_topk_update_ms": spectral_topk_ms,
        "spectral_topk_vs_full_speedup": full_ms / spectral_topk_ms,
        "candidate_fraction": candidate_fraction,
        "candidate_count": candidate_count,
        "full_refresh_and_select_ms": full_select_ms,
        "spectral_topk_update_and_select_ms": spectral_select_ms,
        "spectral_update_select_vs_full_speedup": (
            full_select_ms / spectral_select_ms
        ),
        "spectral_topk_update_cast_and_select_ms": spectral_cast_select_ms,
        "spectral_cast_select_vs_full_speedup": (
            full_select_ms / spectral_cast_select_ms
        ),
        "spectral_topk_update_select_then_sort_ms": spectral_select_sort_ms,
        "spectral_select_then_sort_vs_full_speedup": (
            full_select_ms / spectral_select_sort_ms
        ),
        "score_cache_fraction_of_full_kv": groups / (2.0 * 128),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="32000,64000,128000")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch_counts", default="1")
    parser.add_argument("--refresh_counts", default="2")
    parser.add_argument("--candidate_fraction", type=float, default=0.08)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()

    lengths = tuple(int(item) for item in args.lengths.split(",") if item)
    batch_counts = tuple(
        int(item) for item in args.batch_counts.split(",") if item
    )
    refresh_counts = tuple(
        int(item) for item in args.refresh_counts.split(",") if item
    )
    results = [
        benchmark_length(
            length,
            args.rank,
            batch_count,
            refresh_count,
            args.candidate_fraction,
            args.warmup,
            args.repeats,
        )
        for batch_count in batch_counts
        for length in lengths
        for refresh_count in refresh_counts
    ]
    report = {
        "protocol": {
            "batch_counts": batch_counts,
            "refresh_counts": refresh_counts,
            "kv_heads": 8,
            "query_heads": 32,
            "projection_dim": 64,
            "rank": args.rank,
            "candidate_fraction": args.candidate_fraction,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "includes": [
                "query quantization",
                "dimension top-k",
                "INT4 index scan",
                "FP16 score-cache read/add/write",
            ],
        },
        "results": results,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
