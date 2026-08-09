from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_coaccess_kv_layout import (
    complete_query_head_groups,
    coaccess_physical_positions,
    union_query_heads,
)
from evaluate_streaming_hypergraph_layout import streaming_hypergraph_positions


def build_page_workload(
    path: Path,
    *,
    train_observations: int,
    test_queries: int,
    group_size: int,
    page_size: int,
    microblock_size: int,
    neighbor_count: int,
    layout_method: str = "coaccess_graph",
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float | int]]:
    payload = np.load(path)
    indices = payload["indices"]
    token_count = int(payload["context_token_ids"].shape[0])
    selected_heads = payload["selected_heads"]
    query_head_groups = complete_query_head_groups(selected_heads, group_size)
    if not query_head_groups:
        raise ValueError("shared KV-head evaluation requires at least one complete query-head group")
    if train_observations + test_queries > indices.shape[2]:
        raise ValueError("requested queries exceed the stored observations")

    pages_per_group = (token_count + page_size - 1) // page_size
    chronological_by_query: list[list[np.ndarray]] = [[] for _ in range(test_queries)]
    coaccess_by_query: list[list[np.ndarray]] = [[] for _ in range(test_queries)]
    group_index = 0
    layout_started = time.perf_counter()
    for layer_slot in range(indices.shape[0]):
        for group_slots in query_head_groups:
            token_sets = union_query_heads(indices[layer_slot, group_slots])
            if layout_method == "coaccess_graph":
                positions = coaccess_physical_positions(
                    token_sets[:train_observations],
                    token_count,
                    page_size=page_size,
                    microblock_size=microblock_size,
                    neighbor_count=neighbor_count,
                )
            elif layout_method == "streaming_hypergraph":
                positions = streaming_hypergraph_positions(
                    token_sets[:train_observations], token_count
                )
            else:
                raise ValueError(f"unsupported layout method: {layout_method}")
            page_offset = group_index * pages_per_group
            for query_index, tokens in enumerate(
                token_sets[train_observations : train_observations + test_queries]
            ):
                chronological_by_query[query_index].append(
                    np.unique(tokens // page_size).astype(np.int64) + page_offset
                )
                coaccess_by_query[query_index].append(
                    np.unique(positions[tokens] // page_size).astype(np.int64) + page_offset
                )
            group_index += 1
    layout_seconds = time.perf_counter() - layout_started

    chronological = [np.concatenate(parts) for parts in chronological_by_query]
    coaccess = [np.concatenate(parts) for parts in coaccess_by_query]
    metadata: dict[str, float | int] = {
        "context_tokens": token_count,
        "layers": int(indices.shape[0]),
        "kv_heads": len(query_head_groups),
        "physical_kv_groups": group_index,
        "pages_per_group": pages_per_group,
        "total_pages": group_index * pages_per_group,
        "layout_build_seconds_python": layout_seconds,
        "layout_method": layout_method,
    }
    return chronological, coaccess, metadata


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean_ms_per_query": statistics.mean(samples),
        "median_ms_per_query": statistics.median(samples),
        "p90_ms_per_query": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
    }


def benchmark_gpu_gather(
    workloads: dict[str, list[torch.Tensor]],
    *,
    total_pages: int,
    page_size: int,
    row_elements: int,
    rounds: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    source = torch.empty(
        (total_pages, page_size, row_elements),
        dtype=torch.bfloat16,
        device=device,
    ).normal_()
    maximum_pages = max(index.numel() for values in workloads.values() for index in values)
    destination = torch.empty(
        (maximum_pages, page_size, row_elements),
        dtype=source.dtype,
        device=device,
    )
    timings: dict[str, list[float]] = {name: [] for name in workloads}
    method_order = list(workloads)
    for round_index in range(rounds + 1):
        for name in method_order[round_index % len(method_order) :] + method_order[: round_index % len(method_order)]:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for index in workloads[name]:
                torch.index_select(source, 0, index, out=destination[: index.numel()])
            torch.cuda.synchronize(device)
            elapsed_ms = 1000.0 * (time.perf_counter() - started) / len(workloads[name])
            if round_index > 0:
                timings[name].append(elapsed_ms)
    return {name: _summary(samples) for name, samples in timings.items()}


def benchmark_host_miss_fill(
    workloads: dict[str, list[np.ndarray]],
    *,
    total_pages: int,
    page_size: int,
    row_elements: int,
    rounds: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    source = torch.empty(
        (total_pages, page_size, row_elements),
        dtype=torch.bfloat16,
        pin_memory=True,
    ).normal_()
    maximum_pages = max(index.size for values in workloads.values() for index in values)
    staging = torch.empty(
        (maximum_pages, page_size, row_elements),
        dtype=source.dtype,
        pin_memory=True,
    )
    destination = torch.empty_like(staging, device=device)
    cpu_indices = {
        name: [torch.from_numpy(index) for index in values]
        for name, values in workloads.items()
    }
    timings: dict[str, list[float]] = {name: [] for name in workloads}
    method_order = list(workloads)
    for round_index in range(rounds + 1):
        for name in method_order[round_index % len(method_order) :] + method_order[: round_index % len(method_order)]:
            started = time.perf_counter()
            for index in cpu_indices[name]:
                page_count = index.numel()
                torch.index_select(source, 0, index, out=staging[:page_count])
                destination[:page_count].copy_(staging[:page_count], non_blocking=True)
                torch.cuda.synchronize(device)
            elapsed_ms = 1000.0 * (time.perf_counter() - started) / len(cpu_indices[name])
            if round_index > 0:
                timings[name].append(elapsed_ms)
    return {name: _summary(samples) for name, samples in timings.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark co-access-aware physical KV page reads.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train_observations", type=int, default=256)
    parser.add_argument("--test_queries", type=int, default=64)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--page_size", type=int, default=16)
    parser.add_argument("--microblock_size", type=int, default=1)
    parser.add_argument("--neighbor_count", type=int, default=16)
    parser.add_argument(
        "--layout_method",
        choices=("coaccess_graph", "streaming_hypergraph"),
        default="coaccess_graph",
    )
    parser.add_argument("--row_elements", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    chronological, coaccess, metadata = build_page_workload(
        Path(args.input),
        train_observations=args.train_observations,
        test_queries=args.test_queries,
        group_size=args.group_size,
        page_size=args.page_size,
        microblock_size=args.microblock_size,
        neighbor_count=args.neighbor_count,
        layout_method=args.layout_method,
    )
    numpy_workloads = {"chronological": chronological, "coaccess": coaccess}
    gpu_workloads = {
        name: [torch.from_numpy(index).to(device=device) for index in values]
        for name, values in numpy_workloads.items()
    }
    page_counts = {
        name: float(np.mean([index.size for index in values]))
        for name, values in numpy_workloads.items()
    }
    results = {
        **metadata,
        "page_tokens": args.page_size,
        "microblock_tokens": args.microblock_size,
        "row_elements_bf16": args.row_elements,
        "test_queries": args.test_queries,
        "mean_pages_per_query": page_counts,
        "page_reduction_fraction": 1.0 - page_counts["coaccess"] / page_counts["chronological"],
        "gpu_resident_page_gather": benchmark_gpu_gather(
            gpu_workloads,
            total_pages=int(metadata["total_pages"]),
            page_size=args.page_size,
            row_elements=args.row_elements,
            rounds=args.rounds,
            device=device,
        ),
        "host_pinned_miss_fill": benchmark_host_miss_fill(
            numpy_workloads,
            total_pages=int(metadata["total_pages"]),
            page_size=args.page_size,
            row_elements=args.row_elements,
            rounds=args.rounds,
            device=device,
        ),
        "quality_contract": "exact token IDs unchanged; only physical page addresses differ",
    }
    for benchmark in ("gpu_resident_page_gather", "host_pinned_miss_fill"):
        baseline = results[benchmark]["chronological"]["median_ms_per_query"]
        packed = results[benchmark]["coaccess"]["median_ms_per_query"]
        results[benchmark]["coaccess_speedup"] = baseline / packed

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
