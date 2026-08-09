#!/usr/bin/env python
"""Measure a graph-captured frozen QKSieve attention layer.

The benchmark keeps the numerical path fixed. It compares eager launches with
CUDA Graph replay and rejects the graph result unless candidate IDs, thresholds,
counts, and attention outputs match the eager path for a fresh query.
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
import qksieve_query_cuda_20260728 as query_cuda
import qksieve_valuesketch_cuda_20260801 as value_cuda
import variablebit_spectral_cuda_20260727 as variablebit_cuda
from benchmark_variablebit_spectral_attention_20260727 import (
    ALLOCATION_PROFILES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="32768,65536")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument(
        "--allocation_profile",
        choices=sorted(ALLOCATION_PROFILES),
        default="qmse_total_b15",
    )
    return parser.parse_args()


def selected_count(history: int) -> int:
    return min(history, 1280, max(256, math.ceil(0.06 * history)))


def sample_count(selected_fraction: float, history: int) -> int:
    return min(
        history,
        8192,
        max(256, math.ceil(16.0 / selected_fraction)),
    )


def candidate_capacity(
    history: int, selected_fraction: float, samples: int
) -> int:
    standard_deviation = math.sqrt(
        selected_fraction * (1.0 - selected_fraction) / samples
    )
    fraction = min(
        1.0,
        max(0.06, selected_fraction + 6.0 * standard_deviation),
    )
    return min(history, max(1, math.ceil(fraction * history)))


def wall_ms(
    function: Callable[[], object], warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start) / iterations


def cuda_ms(
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


def valid_candidate_ids(
    indices: torch.Tensor, counts: torch.Tensor
) -> list[list[int]]:
    indices_cpu = indices.cpu()
    counts_cpu = counts.cpu()
    return [
        indices_cpu[0, head, : int(counts_cpu[0, head])].tolist()
        for head in range(indices.shape[1])
    ]


@torch.inference_mode()
def run_length(
    history: int,
    allocation: torch.Tensor,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    dtype = torch.float16
    scaling = 128.0**-0.5
    target = selected_count(history)
    selected_fraction = target / history
    samples = sample_count(selected_fraction, history)
    capacity = candidate_capacity(history, selected_fraction, samples)
    threshold_rank = max(
        1,
        min(samples, int(round(selected_fraction * (samples + 1)))),
    )
    threshold_fraction = (threshold_rank - 0.5) / samples
    scan_blocks = math.ceil(history / 256)
    value_blocks = math.ceil(history / 256)

    static_query = torch.randn(1, 32, 128, dtype=dtype, device="cuda")
    grouped_query = static_query.view(1, 8, 4, 128)
    query_basis = torch.randn(
        1, 8, 128, 128, dtype=dtype, device="cuda"
    ).contiguous()
    key = torch.randn(
        1, 8, history, 128, dtype=dtype, device="cuda"
    )
    value = torch.randn_like(key)

    packed_index = variablebit_cuda.allocate_packed_index(
        allocation, history, dtype
    )
    packed_index["packed_codes"].random_(0, 256)
    packed_index["key_scales"].uniform_(0.01, 0.1)

    value_codes = torch.randint(
        0,
        256,
        (1, 8, history, 8),
        dtype=torch.uint8,
        device="cuda",
    )
    value_minimum = torch.randn(
        1, 8, value_blocks, 16, dtype=dtype, device="cuda"
    )
    value_scale = torch.rand(
        1, 8, value_blocks, 16, dtype=dtype, device="cuda"
    ).mul_(0.1).add_(0.01)
    value_mean = torch.randn(1, 8, 128, dtype=dtype, device="cuda")
    value_basis = torch.randn(
        1, 8, 128, 16, dtype=dtype, device="cuda"
    ).mul_(0.1)

    query_codes = torch.empty_like(grouped_query, dtype=torch.int8)
    query_scales = torch.empty(
        1, 8, 4, 8, dtype=dtype, device="cuda"
    )
    selection_masks = torch.empty(
        1, 32, scan_blocks * 8, dtype=torch.int32, device="cuda"
    )
    tail_partials = torch.empty(
        1, 32, scan_blocks, 18, dtype=torch.float32, device="cuda"
    )
    candidate_indices = torch.empty(
        1, 32, capacity, dtype=torch.long, device="cuda"
    )
    candidate_counts = torch.empty(1, 32, dtype=torch.long, device="cuda")
    thresholds = torch.empty(1, 32, dtype=torch.float32, device="cuda")
    overflow = torch.empty(1, 32, dtype=torch.bool, device="cuda")
    selected_denominator = torch.empty(
        1, 32, dtype=torch.float32, device="cuda"
    )
    tail_denominator = torch.empty(
        1, 32, dtype=torch.float32, device="cuda"
    )
    tail_coefficients = torch.empty(
        1, 32, 16, dtype=torch.float32, device="cuda"
    )
    workspace = value_cuda.allocate_attention_workspace(
        static_query, capacity
    )

    query_extension = query_cuda.load_extension()
    scan_extension = mixed_cuda.load_extension()
    value_extension = value_cuda.load_extension()

    reference_codes, reference_scales = tuple(
        query_extension.qksieve_project_quantize_wmma_forward(
            grouped_query, query_basis
        )
    )
    query_extension.qksieve_project_quantize_wmma_out(
        grouped_query,
        query_basis,
        query_codes,
        query_scales,
    )
    torch.cuda.synchronize()
    projection_codes_equal = bool(torch.equal(reference_codes, query_codes))
    projection_scales_equal = bool(
        torch.equal(reference_scales, query_scales)
    )

    def sparse_step() -> torch.Tensor:
        query_extension.qksieve_project_quantize_wmma_out(
            grouped_query,
            query_basis,
            query_codes,
            query_scales,
        )
        scan_extension.plain_sampled_compact_gqa4_valuesketch_deterministic_out(
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
            value_codes,
            value_minimum,
            value_scale,
            selection_masks,
            tail_partials,
            candidate_indices,
            candidate_counts,
            thresholds,
            overflow,
            selected_denominator,
            tail_denominator,
            tail_coefficients,
            history,
            samples,
            threshold_fraction,
            16,
            256,
            scaling,
        )
        value_extension.qksieve_valuesketch_attention_out(
            static_query,
            key,
            value,
            candidate_indices,
            candidate_counts,
            thresholds,
            tail_denominator,
            tail_coefficients,
            value_mean,
            value_basis,
            workspace["output"],
            workspace["partial_output"],
            workspace["partial_maximum"],
            workspace["partial_sum"],
            scaling,
            1.0,
        )
        return workspace["output"]

    def full_step() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            static_query.unsqueeze(2),
            key,
            value,
            is_causal=False,
            enable_gqa=True,
        )

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(5):
            sparse_step()
            full_step()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    sparse_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(sparse_graph):
        sparse_step()
    full_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(full_graph):
        full_output = full_step()
    torch.cuda.synchronize()

    probe_query = torch.randn_like(static_query)
    static_query.copy_(probe_query)
    sparse_step()
    torch.cuda.synchronize()
    eager_output = workspace["output"].clone()
    eager_counts = candidate_counts.clone()
    eager_thresholds = thresholds.clone()
    eager_overflow = overflow.clone()
    eager_ids = valid_candidate_ids(candidate_indices, candidate_counts)

    static_query.copy_(probe_query)
    sparse_graph.replay()
    torch.cuda.synchronize()
    graph_output = workspace["output"].clone()
    graph_counts = candidate_counts.clone()
    graph_thresholds = thresholds.clone()
    graph_overflow = overflow.clone()
    graph_ids = valid_candidate_ids(candidate_indices, candidate_counts)

    output_max_abs = float((eager_output - graph_output).abs().max().item())
    threshold_max_abs = float(
        (eager_thresholds - graph_thresholds).abs().max().item()
    )
    counts_equal = bool(torch.equal(eager_counts, graph_counts))
    overflow_equal = bool(torch.equal(eager_overflow, graph_overflow))
    candidate_ids_equal = eager_ids == graph_ids
    equivalent = (
        projection_codes_equal
        and projection_scales_equal
        and output_max_abs == 0.0
        and threshold_max_abs == 0.0
        and counts_equal
        and overflow_equal
        and candidate_ids_equal
    )

    query_source = torch.randn_like(static_query)

    def sparse_graph_with_copy() -> None:
        static_query.copy_(query_source)
        sparse_graph.replay()

    def full_graph_with_copy() -> None:
        static_query.copy_(query_source)
        full_graph.replay()

    timings = {
        "sparse_eager_wall_ms": wall_ms(sparse_step, warmup, iterations),
        "sparse_eager_cuda_ms": cuda_ms(sparse_step, warmup, iterations),
        "sparse_graph_wall_ms": wall_ms(
            sparse_graph.replay, warmup, iterations
        ),
        "sparse_graph_cuda_ms": cuda_ms(
            sparse_graph.replay, warmup, iterations
        ),
        "sparse_graph_copy_wall_ms": wall_ms(
            sparse_graph_with_copy, warmup, iterations
        ),
        "full_eager_wall_ms": wall_ms(full_step, warmup, iterations),
        "full_eager_cuda_ms": cuda_ms(full_step, warmup, iterations),
        "full_graph_wall_ms": wall_ms(full_graph.replay, warmup, iterations),
        "full_graph_cuda_ms": cuda_ms(full_graph.replay, warmup, iterations),
        "full_graph_copy_wall_ms": wall_ms(
            full_graph_with_copy, warmup, iterations
        ),
    }
    timings["sparse_graph_vs_eager"] = (
        timings["sparse_eager_wall_ms"] / timings["sparse_graph_wall_ms"]
    )
    timings["graphed_sparse_vs_graphed_full"] = (
        timings["full_graph_wall_ms"] / timings["sparse_graph_wall_ms"]
    )

    del full_output
    return {
        "history_tokens": history,
        "target_tokens_per_head": target,
        "selected_fraction": selected_fraction,
        "sample_count": samples,
        "candidate_capacity": capacity,
        "equivalence": {
            "passed": equivalent,
            "projection_codes_equal": projection_codes_equal,
            "projection_scales_equal": projection_scales_equal,
            "output_max_abs": output_max_abs,
            "threshold_max_abs": threshold_max_abs,
            "counts_equal": counts_equal,
            "overflow_equal": overflow_equal,
            "candidate_ids_equal": candidate_ids_equal,
        },
        "timings": timings,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    lengths = sorted(
        {int(item) for item in args.lengths.split(",") if item.strip()}
    )
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("lengths must contain positive integers")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    allocation = ALLOCATION_PROFILES[args.allocation_profile].unsqueeze(0).cuda()
    rows = [
        run_length(length, allocation, args.warmup, args.iterations)
        for length in lengths
    ]
    if not all(bool(row["equivalence"]["passed"]) for row in rows):
        raise RuntimeError("CUDA Graph equivalence check failed")

    payload = {
        "benchmark": "frozen_qksieve_cuda_graph",
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "allocation_profile": args.allocation_profile,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
