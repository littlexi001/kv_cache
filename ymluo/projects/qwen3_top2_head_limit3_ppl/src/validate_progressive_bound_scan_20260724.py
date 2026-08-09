#!/usr/bin/env python3
"""Validate and benchmark the strict progressive PCA/INT4 scanner."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import qabs_cuda_kernels as kernels


def _time_cuda(callable_, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        callable_()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / repeats


def _sorted_candidates(
    indices: torch.Tensor,
    scores: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[list[list[int]], list[list[float]]]:
    flattened_indices = indices.reshape(-1, indices.shape[-1]).cpu()
    flattened_scores = scores.reshape(-1, scores.shape[-1]).cpu()
    flattened_counts = counts.reshape(-1).cpu()
    candidate_sets: list[list[int]] = []
    candidate_scores: list[list[float]] = []
    for row, count_tensor in enumerate(flattened_counts):
        count = int(count_tensor)
        pairs = sorted(
            zip(
                flattened_indices[row, :count].tolist(),
                flattened_scores[row, :count].tolist(),
                strict=True,
            )
        )
        candidate_sets.append([int(index) for index, _ in pairs])
        candidate_scores.append([float(score) for _, score in pairs])
    return candidate_sets, candidate_scores


def run_case(
    key_count: int,
    projection_dim: int,
    kv_head_count: int,
    group_count: int,
    selected_fraction: float,
    sample_count: int,
    repeats: int,
) -> dict[str, float | int | bool]:
    torch.manual_seed(20260724 + key_count)
    device = torch.device("cuda")
    dtype = torch.float16
    chunk_count = projection_dim // 16
    capacity = key_count
    dimension_scale = torch.exp(
        -torch.arange(projection_dim, device=device, dtype=torch.float32) / 18.0
    )
    projected_key = (
        torch.randn(
            1,
            kv_head_count,
            key_count,
            projection_dim,
            device=device,
            dtype=torch.float32,
        )
        * dimension_scale
    ).to(dtype)
    packed = torch.empty(
        1,
        kv_head_count,
        chunk_count,
        capacity,
        8,
        device=device,
        dtype=torch.uint8,
    )
    scales = torch.empty(
        1,
        kv_head_count,
        capacity,
        1,
        device=device,
        dtype=dtype,
    )
    exponents = torch.empty(
        1,
        kv_head_count,
        capacity,
        math.ceil(chunk_count / 2),
        device=device,
        dtype=torch.uint8,
    )
    chunk_squared_norms = torch.empty(
        1,
        kv_head_count,
        capacity,
        chunk_count,
        device=device,
        dtype=torch.int16,
    )
    kernels.pca_int4_logscale16_pack_into(
        projected_key,
        packed,
        scales,
        exponents,
        0,
    )
    kernels.pca_int4_logscale16_chunk_norms_into(
        packed,
        chunk_squared_norms,
        0,
        key_count,
    )
    del projected_key

    query_scale = torch.exp(
        -torch.arange(projection_dim, device=device, dtype=torch.float32) / 18.0
    )
    query_float = (
        torch.randn(
            1,
            kv_head_count,
            group_count,
            projection_dim,
            device=device,
        )
        * query_scale
    )
    query_codes = torch.round(
        query_float
        / query_float.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
        * 127.0
    ).clamp(-127, 127).to(torch.int8)
    candidate_capacity = min(
        key_count,
        math.ceil(key_count * max(0.12, 2.0 * selected_fraction)),
    )

    def full_call():
        return kernels.pca_int4_logscale16_sampled_quantile_candidates(
            query_codes,
            packed,
            scales,
            exponents,
            key_count,
            sample_count,
            selected_fraction,
            candidate_capacity,
            use_dp4a=True,
            write_proxy_scores=True,
        )

    def bound_call(statistics: bool = False):
        return kernels.pca_int4_logscale16_sampled_quantile_bound_candidates(
            query_codes,
            packed,
            scales,
            exponents,
            chunk_squared_norms,
            key_count,
            sample_count,
            selected_fraction,
            candidate_capacity,
            use_dp4a=True,
            write_proxy_scores=True,
            collect_statistics=statistics,
        )

    full = full_call()
    bounded = bound_call(True)
    torch.cuda.synchronize()
    full_indices, full_scores, full_counts, full_boundaries, full_overflow = full
    (
        bounded_indices,
        bounded_scores,
        bounded_counts,
        bounded_boundaries,
        bounded_overflow,
        key_chunk_evaluations,
        query_chunk_evaluations,
    ) = bounded
    full_sets, full_sorted_scores = _sorted_candidates(
        full_indices, full_scores, full_counts
    )
    bounded_sets, bounded_sorted_scores = _sorted_candidates(
        bounded_indices, bounded_scores, bounded_counts
    )
    maximum_score_error = max(
        (
            max(
                (abs(left - right) for left, right in zip(a, b, strict=True)),
                default=0.0,
            )
            for a, b in zip(full_sorted_scores, bounded_sorted_scores, strict=True)
        ),
        default=0.0,
    )
    exact_candidates = (
        full_sets == bounded_sets
        and torch.equal(full_counts, bounded_counts)
        and torch.equal(full_overflow, bounded_overflow)
    )
    full_ms = _time_cuda(full_call, warmup=5, repeats=repeats)
    bounded_ms = _time_cuda(
        lambda: bound_call(False), warmup=5, repeats=repeats
    )
    key_fraction = float(
        key_chunk_evaluations.sum().item()
        / (kv_head_count * key_count * chunk_count)
    )
    query_fraction = float(
        query_chunk_evaluations.sum().item()
        / (kv_head_count * group_count * key_count * chunk_count)
    )
    return {
        "key_count": key_count,
        "projection_dim": projection_dim,
        "exact_candidates": exact_candidates,
        "boundary_exact": bool(
            torch.equal(full_boundaries, bounded_boundaries)
        ),
        "maximum_proxy_score_error": maximum_score_error,
        "key_chunk_fraction": key_fraction,
        "query_chunk_fraction": query_fraction,
        "full_scan_ms": full_ms,
        "bounded_scan_ms": bounded_ms,
        "scan_speedup": full_ms / bounded_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 16384, 32768])
    parser.add_argument("--projection-dim", type=int, default=48)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--selected-fraction", type=float, default=0.06)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.time()
    rows = [
        run_case(
            length,
            args.projection_dim,
            args.kv_heads,
            args.groups,
            args.selected_fraction,
            args.sample_count,
            args.repeats,
        )
        for length in args.lengths
    ]
    result = {
        "configuration": vars(args) | {"output": str(args.output) if args.output else None},
        "rows": rows,
        "elapsed_seconds": time.time() - started,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
