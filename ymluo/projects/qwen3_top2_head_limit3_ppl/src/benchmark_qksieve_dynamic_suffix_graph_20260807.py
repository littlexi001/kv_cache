#!/usr/bin/env python
"""Validate a graph-replayable static-prefix plus exact-suffix attention path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import mixedblock_spectral_cuda_20260729 as mixed_cuda
import qksieve_query_cuda_20260728 as query_cuda
import qksieve_valuesketch_cuda_20260801 as value_cuda
import variablebit_spectral_cuda_20260727 as variablebit_cuda
from benchmark_qksieve_cuda_graph_20260807 import (
    candidate_capacity,
    cuda_ms,
    sample_count,
    selected_count,
    valid_candidate_ids,
    wall_ms,
)
from benchmark_variablebit_spectral_attention_20260727 import (
    ALLOCATION_PROFILES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tokens", type=int, default=4096)
    parser.add_argument("--suffix_capacity", type=int, default=64)
    parser.add_argument("--suffix_counts", default="0,1,7,31,63")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def allocate_scan_outputs(
    capacity: int, prefix_tokens: int
) -> dict[str, torch.Tensor]:
    scan_blocks = math.ceil(prefix_tokens / 256)
    return {
        "selection_masks": torch.empty(
            1, 32, scan_blocks * 8, dtype=torch.int32, device="cuda"
        ),
        "tail_partials": torch.empty(
            1, 32, scan_blocks, 18, dtype=torch.float32, device="cuda"
        ),
        "indices": torch.empty(
            1, 32, capacity, dtype=torch.long, device="cuda"
        ),
        "counts": torch.empty(1, 32, dtype=torch.long, device="cuda"),
        "thresholds": torch.empty(
            1, 32, dtype=torch.float32, device="cuda"
        ),
        "overflow": torch.empty(1, 32, dtype=torch.bool, device="cuda"),
        "selected_denominator": torch.empty(
            1, 32, dtype=torch.float32, device="cuda"
        ),
        "tail_denominator": torch.empty(
            1, 32, dtype=torch.float32, device="cuda"
        ),
        "tail_coefficients": torch.empty(
            1, 32, 16, dtype=torch.float32, device="cuda"
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.prefix_tokens <= 0 or args.suffix_capacity <= 0:
        raise ValueError("prefix_tokens and suffix_capacity must be positive")
    suffix_counts = sorted(
        {int(item) for item in args.suffix_counts.split(",") if item.strip()}
    )
    if (
        not suffix_counts
        or suffix_counts[0] < 0
        or suffix_counts[-1] >= args.suffix_capacity
    ):
        raise ValueError("suffix_counts must lie in [0, suffix_capacity)")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = torch.float16
    scaling = 128.0**-0.5
    prefix = args.prefix_tokens
    physical_key_count = prefix + args.suffix_capacity
    target = selected_count(prefix)
    fraction = target / prefix
    samples = sample_count(fraction, prefix)
    threshold_rank = max(
        1, min(samples, int(round(fraction * (samples + 1))))
    )
    threshold_fraction = (threshold_rank - 0.5) / samples
    prefix_capacity = candidate_capacity(prefix, fraction, samples)
    total_capacity = prefix_capacity + args.suffix_capacity

    allocation = ALLOCATION_PROFILES["qmse_total_b15"].unsqueeze(0).cuda()
    static_query = torch.randn(1, 32, 128, dtype=dtype, device="cuda")
    grouped_query = static_query.view(1, 8, 4, 128)
    query_basis = torch.randn(
        1, 8, 128, 128, dtype=dtype, device="cuda"
    ).contiguous()
    query_codes = torch.empty_like(grouped_query, dtype=torch.int8)
    query_scales = torch.empty(
        1, 8, 4, 8, dtype=dtype, device="cuda"
    )
    key = torch.randn(
        1, 8, physical_key_count, 128, dtype=dtype, device="cuda"
    )
    value = torch.randn_like(key)
    packed_index = variablebit_cuda.allocate_packed_index(
        allocation, prefix, dtype
    )
    packed_index["packed_codes"].random_(0, 256)
    packed_index["key_scales"].uniform_(0.01, 0.1)

    value_blocks = math.ceil(prefix / 256)
    value_codes = torch.randint(
        0,
        256,
        (1, 8, prefix, 8),
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
    outputs = allocate_scan_outputs(total_capacity, prefix)
    active_key_count = torch.tensor(
        [prefix + 1], dtype=torch.int32, device="cuda"
    )
    active_workspace = value_cuda.allocate_attention_workspace(
        static_query, total_capacity
    )
    reference_workspace = value_cuda.allocate_attention_workspace(
        static_query, total_capacity
    )

    query_extension = query_cuda.load_extension()
    scan_extension = mixed_cuda.load_extension()
    value_extension = value_cuda.load_extension()

    def dynamic_step() -> torch.Tensor:
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
            outputs["selection_masks"],
            outputs["tail_partials"],
            outputs["indices"],
            outputs["counts"],
            outputs["thresholds"],
            outputs["overflow"],
            outputs["selected_denominator"],
            outputs["tail_denominator"],
            outputs["tail_coefficients"],
            prefix,
            samples,
            threshold_fraction,
            16,
            256,
            scaling,
        )
        value_extension.qksieve_append_suffix_candidates_out(
            outputs["indices"],
            outputs["counts"],
            active_key_count,
            prefix,
            physical_key_count,
        )
        value_extension.qksieve_valuesketch_attention_active_out(
            static_query,
            key,
            value,
            outputs["indices"],
            outputs["counts"],
            outputs["thresholds"],
            outputs["tail_denominator"],
            outputs["tail_coefficients"],
            value_mean,
            value_basis,
            active_key_count,
            active_workspace["output"],
            active_workspace["partial_output"],
            active_workspace["partial_maximum"],
            active_workspace["partial_sum"],
            scaling,
            1.0,
        )
        return active_workspace["output"]

    for _ in range(5):
        dynamic_step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dynamic_step()
    torch.cuda.synchronize()

    rows: list[dict[str, object]] = []
    for suffix_history_count in suffix_counts:
        logical_key_count = prefix + suffix_history_count + 1
        probe_query = torch.randn_like(static_query)
        static_query.copy_(probe_query)
        active_key_count.fill_(logical_key_count)
        dynamic_step()
        torch.cuda.synchronize()
        eager_output = active_workspace["output"].clone()
        eager_counts = outputs["counts"].clone()
        eager_ids = valid_candidate_ids(outputs["indices"], outputs["counts"])

        value_extension.qksieve_valuesketch_attention_out(
            static_query,
            key[..., :logical_key_count, :],
            value[..., :logical_key_count, :],
            outputs["indices"],
            outputs["counts"],
            outputs["thresholds"],
            outputs["tail_denominator"],
            outputs["tail_coefficients"],
            value_mean,
            value_basis,
            reference_workspace["output"],
            reference_workspace["partial_output"],
            reference_workspace["partial_maximum"],
            reference_workspace["partial_sum"],
            scaling,
            1.0,
        )
        torch.cuda.synchronize()
        reference_output = reference_workspace["output"].clone()

        static_query.copy_(probe_query)
        active_key_count.fill_(logical_key_count)
        graph.replay()
        torch.cuda.synchronize()
        graph_output = active_workspace["output"].clone()
        graph_counts = outputs["counts"].clone()
        graph_ids = valid_candidate_ids(outputs["indices"], outputs["counts"])

        rows.append(
            {
                "suffix_history_tokens": suffix_history_count,
                "logical_key_count": logical_key_count,
                "active_vs_sliced_reference_max_abs": float(
                    (eager_output - reference_output).abs().max().item()
                ),
                "graph_vs_eager_max_abs": float(
                    (graph_output - eager_output).abs().max().item()
                ),
                "counts_equal": bool(torch.equal(eager_counts, graph_counts)),
                "candidate_ids_equal": eager_ids == graph_ids,
                "count_min": int(graph_counts.min().item()),
                "count_max": int(graph_counts.max().item()),
            }
        )

    active_key_count.fill_(prefix + suffix_counts[-1] + 1)
    timing = {
        "eager_wall_ms": wall_ms(
            dynamic_step, args.warmup, args.iterations
        ),
        "eager_cuda_ms": cuda_ms(
            dynamic_step, args.warmup, args.iterations
        ),
        "graph_wall_ms": wall_ms(graph.replay, args.warmup, args.iterations),
        "graph_cuda_ms": cuda_ms(graph.replay, args.warmup, args.iterations),
    }
    timing["graph_vs_eager_speedup"] = (
        timing["eager_wall_ms"] / timing["graph_wall_ms"]
    )

    passed = all(
        row["active_vs_sliced_reference_max_abs"] == 0.0
        and row["graph_vs_eager_max_abs"] == 0.0
        and row["counts_equal"]
        and row["candidate_ids_equal"]
        for row in rows
    )
    payload = {
        "benchmark": "qksieve_dynamic_exact_suffix_graph",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "prefix_tokens": prefix,
        "suffix_capacity": args.suffix_capacity,
        "prefix_candidate_capacity": prefix_capacity,
        "total_candidate_capacity": total_capacity,
        "passed": passed,
        "rows": rows,
        "timing": timing,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("dynamic suffix graph equivalence failed")


if __name__ == "__main__":
    main()
