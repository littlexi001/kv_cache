from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty measurement list")
    ordered = sorted(values)

    def quantile(fraction: float) -> float:
        index = round(fraction * (len(ordered) - 1))
        return float(ordered[min(len(ordered) - 1, max(0, index))])

    return {
        "mean_ms": float(statistics.fmean(values)),
        "p50_ms": quantile(0.50),
        "p95_ms": quantile(0.95),
        "minimum_ms": float(ordered[0]),
        "maximum_ms": float(ordered[-1]),
    }


def cuda_time_ms(operation: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def wall_time_ms(operation: Callable[[], None]) -> float:
    started = time.perf_counter()
    operation()
    return 1000.0 * (time.perf_counter() - started)


def make_candidate_batch(
    *,
    history_count: int,
    kv_heads: int,
    groups: int,
    candidate_count: int,
    common_fraction: float,
    page_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Create controlled per-query-head candidates and a GQA-union fetch list."""

    common_count = round(candidate_count * common_fraction)
    unique_count = candidate_count - common_count
    required = common_count + groups * unique_count
    if required > history_count:
        raise ValueError(
            f"candidate construction needs {required} distinct tokens but history has "
            f"only {history_count}"
        )

    candidate_rows: list[torch.Tensor] = []
    fetch_rows: list[torch.Tensor] = []
    useful_union_counts: list[int] = []
    for _ in range(kv_heads):
        permutation = torch.randperm(history_count, generator=generator)
        common = permutation[:common_count]
        cursor = common_count
        per_query: list[torch.Tensor] = []
        for _ in range(groups):
            unique = permutation[cursor : cursor + unique_count]
            cursor += unique_count
            per_query.append(torch.sort(torch.cat((common, unique))).values)
        candidates = torch.stack(per_query, dim=0)
        useful_union = torch.unique(candidates.reshape(-1), sorted=True)
        useful_union_counts.append(int(useful_union.numel()))
        if page_size == 1:
            fetched = useful_union
        else:
            pages = torch.unique(useful_union // page_size, sorted=True)
            offsets = torch.arange(page_size, dtype=torch.long)
            fetched = (pages[:, None] * page_size + offsets[None, :]).reshape(-1)
            fetched = fetched[fetched < history_count]
        candidate_rows.append(candidates)
        fetch_rows.append(fetched)

    fetch_count = max(int(row.numel()) for row in fetch_rows)
    fetch_ids = torch.zeros((kv_heads, fetch_count), dtype=torch.long)
    mappings = torch.empty(
        (kv_heads, groups, candidate_count), dtype=torch.long
    )
    fetched_counts: list[int] = []
    for kv_head, (candidates, fetched) in enumerate(
        zip(candidate_rows, fetch_rows, strict=True)
    ):
        count = int(fetched.numel())
        fetched_counts.append(count)
        fetch_ids[kv_head, :count] = fetched
        if count < fetch_count:
            fetch_ids[kv_head, count:] = fetched[-1]
        mapping = torch.searchsorted(fetched, candidates)
        if not torch.equal(fetched[mapping], candidates):
            raise AssertionError("every exact candidate must be present in the fetch set")
        mappings[kv_head] = mapping

    mean_useful = statistics.fmean(useful_union_counts)
    mean_fetched = statistics.fmean(fetched_counts)
    metrics = {
        "common_candidates_per_query_head": float(common_count),
        "useful_union_tokens_per_kv_head_mean": float(mean_useful),
        "padded_fetch_tokens_per_kv_head": float(fetch_count),
        "fetched_tokens_per_kv_head_mean": float(mean_fetched),
        "gqa_union_factor": float(mean_useful / candidate_count),
        "page_transfer_expansion": float(mean_fetched / mean_useful),
        "padding_expansion": float(fetch_count / mean_fetched),
    }
    return fetch_ids, mappings, metrics


def make_candidate_batch_from_trace(
    *,
    record: dict[str, object],
    history_count: int,
    kv_heads: int,
    groups: int,
    candidate_count: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    indices = record.get("indices")
    counts = record.get("counts")
    if not isinstance(indices, torch.Tensor) or not isinstance(counts, torch.Tensor):
        raise ValueError("candidate trace record is missing indices or counts")
    if indices.shape[0] != kv_heads * groups or counts.shape[0] != kv_heads * groups:
        raise ValueError("candidate trace head count does not match benchmark setup")

    candidate_rows: list[torch.Tensor] = []
    fetch_rows: list[torch.Tensor] = []
    useful_union_counts: list[int] = []
    intersection_counts: list[int] = []
    for kv_head in range(kv_heads):
        per_query: list[torch.Tensor] = []
        valid_sets: list[set[int]] = []
        for group in range(groups):
            head = kv_head * groups + group
            count = min(int(counts[head].item()), int(indices.shape[1]))
            if count <= 0:
                raise ValueError("candidate trace contains an empty query head")
            valid = indices[head, :count].long().clamp(0, history_count - 1)
            valid_sets.append(set(valid.tolist()))
            if count >= candidate_count:
                dense = valid[:candidate_count]
            else:
                dense = torch.cat(
                    (valid, valid[-1:].expand(candidate_count - count)), dim=0
                )
            per_query.append(dense)
        candidates = torch.stack(per_query, dim=0)
        useful_union = torch.tensor(
            sorted(set().union(*valid_sets)), dtype=torch.long
        )
        useful_union_counts.append(int(useful_union.numel()))
        intersection_counts.append(len(set.intersection(*valid_sets)))
        if page_size == 1:
            fetched = useful_union
        else:
            pages = torch.unique(useful_union // page_size, sorted=True)
            offsets = torch.arange(page_size, dtype=torch.long)
            fetched = (pages[:, None] * page_size + offsets[None, :]).reshape(-1)
            fetched = fetched[fetched < history_count]
        candidate_rows.append(candidates)
        fetch_rows.append(fetched)

    fetch_count = max(int(row.numel()) for row in fetch_rows)
    fetch_ids = torch.zeros((kv_heads, fetch_count), dtype=torch.long)
    mappings = torch.empty((kv_heads, groups, candidate_count), dtype=torch.long)
    fetched_counts: list[int] = []
    for kv_head, (candidates, fetched) in enumerate(
        zip(candidate_rows, fetch_rows, strict=True)
    ):
        count = int(fetched.numel())
        fetched_counts.append(count)
        fetch_ids[kv_head, :count] = fetched
        if count < fetch_count:
            fetch_ids[kv_head, count:] = fetched[-1]
        mapping = torch.searchsorted(fetched, candidates)
        if not torch.equal(fetched[mapping], candidates):
            raise AssertionError("trace candidates are missing from the fetch set")
        mappings[kv_head] = mapping

    mean_useful = statistics.fmean(useful_union_counts)
    mean_fetched = statistics.fmean(fetched_counts)
    mean_head_count = statistics.fmean(
        min(int(value.item()), candidate_count) for value in counts
    )
    metrics = {
        "common_candidates_per_query_head": float(
            statistics.fmean(intersection_counts)
        ),
        "useful_union_tokens_per_kv_head_mean": float(mean_useful),
        "padded_fetch_tokens_per_kv_head": float(fetch_count),
        "fetched_tokens_per_kv_head_mean": float(mean_fetched),
        "gqa_union_factor": float(mean_useful / mean_head_count),
        "page_transfer_expansion": float(mean_fetched / mean_useful),
        "padding_expansion": float(fetch_count / mean_fetched),
    }
    return fetch_ids, mappings, metrics


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.query_heads % args.kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    if not 0.0 <= args.common_fraction <= 1.0:
        raise ValueError("common_fraction must be in [0, 1]")
    if args.page_size <= 0:
        raise ValueError("page_size must be positive")

    torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    groups = args.query_heads // args.kv_heads
    batch_count = args.warmup + args.repeats
    generator = torch.Generator().manual_seed(args.seed)

    if args.candidate_trace is None:
        batches = [
            make_candidate_batch(
                history_count=args.history_count,
                kv_heads=args.kv_heads,
                groups=groups,
                candidate_count=args.candidate_count,
                common_fraction=args.common_fraction,
                page_size=args.page_size,
                generator=generator,
            )
            for _ in range(batch_count)
        ]
        candidate_source = "controlled_synthetic"
    else:
        trace = torch.load(args.candidate_trace, map_location="cpu", weights_only=False)
        trace_records = trace.get("records") if isinstance(trace, dict) else None
        if not isinstance(trace_records, list) or not trace_records:
            raise ValueError("candidate trace does not contain records")
        batches = [
            make_candidate_batch_from_trace(
                record=trace_records[
                    (index * len(trace_records) // batch_count) % len(trace_records)
                ],
                history_count=args.history_count,
                kv_heads=args.kv_heads,
                groups=groups,
                candidate_count=args.candidate_count,
                page_size=args.page_size,
            )
            for index in range(batch_count)
        ]
        candidate_source = str(args.candidate_trace)
    # The host tensor represents one layer. Reusing it is valid because it is much
    # larger than last-level cache, while keeping the benchmark memory bounded.
    host_full = torch.empty(
        (2, args.kv_heads, args.history_count, args.head_dim),
        dtype=torch.float16,
        pin_memory=True,
    )
    host_full.fill_(0.015625)
    row_values = (
        torch.arange(2 * args.kv_heads * args.history_count, dtype=torch.int32)
        .remainder_(1024)
        .to(torch.float16)
        .div_(1024.0)
        .reshape(2, args.kv_heads, args.history_count)
    )
    host_full[..., 0].copy_(row_values)
    host_selected_batches = [
        torch.empty(
            (2, args.kv_heads, int(fetch.shape[1]), args.head_dim),
            dtype=torch.float16,
            pin_memory=True,
        )
        for fetch, _, _ in batches
    ]
    host_id_batches = [
        torch.empty_like(fetch, pin_memory=True) for fetch, _, _ in batches
    ]
    device_selected_batches = [
        torch.empty_like(selected, device=device)
        for selected in host_selected_batches
    ]
    device_fetch_batches = [fetch.to(device) for fetch, _, _ in batches]
    device_mapping_batches = [mapping.to(device) for _, mapping, _ in batches]
    query = torch.randn(
        (1, args.query_heads, 1, args.head_dim),
        dtype=torch.float16,
        device=device,
    )

    validation_fetch = batches[0][0]
    validation_count = int(validation_fetch.shape[1])
    validation_index = validation_fetch.reshape(
        1, args.kv_heads, validation_count, 1
    ).expand(2, args.kv_heads, validation_count, args.head_dim)
    validation_gather = torch.empty(
        (2, args.kv_heads, validation_count, args.head_dim),
        dtype=torch.float16,
    )
    validation_offsets = (
        torch.arange(2 * args.kv_heads, dtype=torch.long)
        .reshape(2, args.kv_heads, 1)
        .mul_(args.history_count)
    )
    validation_rows = (
        validation_offsets + validation_fetch.unsqueeze(0)
    ).reshape(-1)
    validation_select = torch.empty_like(validation_gather)
    torch.gather(host_full, 2, validation_index, out=validation_gather)
    torch.index_select(
        host_full.reshape(-1, args.head_dim),
        0,
        validation_rows,
        out=validation_select.reshape(-1, args.head_dim),
    )
    if not torch.equal(validation_gather, validation_select):
        raise AssertionError("row-wise index_select does not match torch.gather")

    retrieval_measurements: dict[str, list[float]] = {
        "candidate_ids_d2h": [],
        "cpu_exact_kv_gather": [],
        "exact_kv_h2d": [],
        "gpu_remap_and_sparse_attention": [],
        "sequential_total": [],
    }
    retrieval_checksum = 0.0

    def retrieval_iteration(batch_index: int, measure: bool) -> None:
        nonlocal retrieval_checksum
        fetch_gpu = device_fetch_batches[batch_index]
        mapping_gpu = device_mapping_batches[batch_index]
        host_ids = host_id_batches[batch_index]
        host_selected = host_selected_batches[batch_index]
        device_selected = device_selected_batches[batch_index]
        fetch_count = int(fetch_gpu.shape[1])

        def ids_d2h() -> None:
            host_ids.copy_(fetch_gpu, non_blocking=True)
            torch.cuda.synchronize()

        id_ms = wall_time_ms(ids_d2h)
        gather_index = host_ids.reshape(
            1, args.kv_heads, fetch_count, 1
        ).expand(2, args.kv_heads, fetch_count, args.head_dim)
        row_offsets = (
            torch.arange(2 * args.kv_heads, dtype=torch.long)
            .reshape(2, args.kv_heads, 1)
            .mul_(args.history_count)
        )
        gather_rows = (row_offsets + host_ids.unsqueeze(0)).reshape(-1)

        def cpu_gather() -> None:
            if args.cpu_gather_mode == "gather":
                torch.gather(host_full, 2, gather_index, out=host_selected)
            elif args.cpu_gather_mode == "index_select":
                torch.index_select(
                    host_full.reshape(-1, args.head_dim),
                    0,
                    gather_rows,
                    out=host_selected.reshape(-1, args.head_dim),
                )
            else:
                raise ValueError(f"unsupported CPU gather mode: {args.cpu_gather_mode}")

        gather_ms = wall_time_ms(cpu_gather)

        def exact_h2d() -> None:
            device_selected.copy_(host_selected, non_blocking=True)

        h2d_ms = cuda_time_ms(exact_h2d)

        def sparse_attention() -> None:
            nonlocal retrieval_checksum
            source_key = device_selected[0].unsqueeze(1).expand(
                args.kv_heads,
                groups,
                fetch_count,
                args.head_dim,
            )
            source_value = device_selected[1].unsqueeze(1).expand_as(source_key)
            gather_mapping = mapping_gpu.unsqueeze(-1).expand(
                args.kv_heads,
                groups,
                args.candidate_count,
                args.head_dim,
            )
            selected_key = torch.gather(source_key, 2, gather_mapping).reshape(
                1, args.query_heads, args.candidate_count, args.head_dim
            )
            selected_value = torch.gather(
                source_value, 2, gather_mapping
            ).reshape(1, args.query_heads, args.candidate_count, args.head_dim)
            output = F.scaled_dot_product_attention(
                query,
                selected_key,
                selected_value,
                is_causal=False,
            )
            retrieval_checksum = float(output[0, 0, 0, 0].float().item())

        attention_ms = cuda_time_ms(sparse_attention)
        total_ms = id_ms + gather_ms + h2d_ms + attention_ms
        if measure:
            retrieval_measurements["candidate_ids_d2h"].append(id_ms)
            retrieval_measurements["cpu_exact_kv_gather"].append(gather_ms)
            retrieval_measurements["exact_kv_h2d"].append(h2d_ms)
            retrieval_measurements["gpu_remap_and_sparse_attention"].append(
                attention_ms
            )
            retrieval_measurements["sequential_total"].append(total_ms)

    for batch_index in range(args.warmup):
        retrieval_iteration(batch_index, False)
    for batch_index in range(args.warmup, batch_count):
        retrieval_iteration(batch_index, True)

    del device_selected_batches
    torch.cuda.empty_cache()
    device_full = torch.empty_like(host_full, device=device)
    full_measurements: dict[str, list[float]] = {
        "full_kv_h2d": [],
        "gpu_native_gqa_attention": [],
        "sequential_total": [],
    }
    full_checksum = 0.0

    def full_iteration(measure: bool) -> None:
        nonlocal full_checksum

        def full_h2d() -> None:
            device_full.copy_(host_full, non_blocking=True)

        h2d_ms = cuda_time_ms(full_h2d)

        def full_attention() -> None:
            nonlocal full_checksum
            output = F.scaled_dot_product_attention(
                query,
                device_full[0].unsqueeze(0),
                device_full[1].unsqueeze(0),
                is_causal=False,
                enable_gqa=True,
            )
            full_checksum = float(output[0, 0, 0, 0].float().item())

        attention_ms = cuda_time_ms(full_attention)
        if measure:
            full_measurements["full_kv_h2d"].append(h2d_ms)
            full_measurements["gpu_native_gqa_attention"].append(attention_ms)
            full_measurements["sequential_total"].append(h2d_ms + attention_ms)

    for _ in range(args.warmup):
        full_iteration(False)
    for _ in range(args.repeats):
        full_iteration(True)

    candidate_metrics = [metrics for _, _, metrics in batches[args.warmup :]]
    mean_metric = {
        key: float(statistics.fmean(row[key] for row in candidate_metrics))
        for key in candidate_metrics[0]
    }
    bytes_per_full_layer = host_full.numel() * host_full.element_size()
    measured_selected_buffers = host_selected_batches[args.warmup :]
    bytes_per_retrieval_layers = [
        selected.numel() * selected.element_size()
        for selected in measured_selected_buffers
    ]
    bytes_per_retrieval_layer = statistics.fmean(bytes_per_retrieval_layers)
    retrieval_summary = {
        key: summarize(values) for key, values in retrieval_measurements.items()
    }
    full_summary = {
        key: summarize(values) for key, values in full_measurements.items()
    }
    direct_subsystem_speedup = (
        full_summary["sequential_total"]["p50_ms"]
        / retrieval_summary["sequential_total"]["p50_ms"]
    )
    host_retrieval_extra_per_layer = sum(
        retrieval_summary[key]["p50_ms"]
        for key in (
            "candidate_ids_d2h",
            "cpu_exact_kv_gather",
            "exact_kv_h2d",
        )
    )
    estimated_full_decode_ms = (
        args.native_full_resident_ms
        + args.layers * full_summary["full_kv_h2d"]["p50_ms"]
    )
    estimated_retrieval_decode_ms = (
        args.qksieve_resident_ms + args.layers * host_retrieval_extra_per_layer
    )

    return {
        "setup": {
            "history_count": args.history_count,
            "candidate_count_per_query_head": args.candidate_count,
            "common_fraction_within_gqa_group": args.common_fraction,
            "page_size": args.page_size,
            "layers": args.layers,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "cpu_threads": args.cpu_threads,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "candidate_source": candidate_source,
            "cpu_gather_mode": args.cpu_gather_mode,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "candidate_layout": mean_metric,
        "storage_and_traffic": {
            "full_kv_mib_per_layer": bytes_per_full_layer / 2**20,
            "full_kv_gib_all_layers": bytes_per_full_layer * args.layers / 2**30,
            "retrieved_kv_mib_per_layer": bytes_per_retrieval_layer / 2**20,
            "retrieved_kv_mib_per_layer_min": min(bytes_per_retrieval_layers)
            / 2**20,
            "retrieved_kv_mib_per_layer_max": max(bytes_per_retrieval_layers)
            / 2**20,
            "retrieved_kv_mib_all_layers": (
                bytes_per_retrieval_layer * args.layers / 2**20
            ),
            "host_traffic_reduction": bytes_per_full_layer
            / bytes_per_retrieval_layer,
            "qksieve_240bit_index_gib_all_layers": (
                args.layers
                * args.kv_heads
                * args.history_count
                * 240
                / 8
                / 2**30
            ),
            "qksieve_index_over_full_fp16_kv": 240 / (2 * args.head_dim * 16),
        },
        "retrieval_per_layer": retrieval_summary,
        "full_offload_per_layer": full_summary,
        "direct_subsystem_speedup_full_over_retrieval": direct_subsystem_speedup,
        "conservative_decode_estimate": {
            "native_full_resident_ms": args.native_full_resident_ms,
            "qksieve_resident_ms": args.qksieve_resident_ms,
            "full_offload_ms_per_token": estimated_full_decode_ms,
            "qksieve_host_ms_per_token": estimated_retrieval_decode_ms,
            "speedup": estimated_full_decode_ms / estimated_retrieval_decode_ms,
            "note": (
                "Adds measured host-transfer stages to prior resident decode. "
                "It is conservative because it does not subtract the old GPU "
                "candidate-gather cost from the resident QKSieve measurement."
            ),
        },
        "checksums": {
            "retrieval": retrieval_checksum,
            "full": full_checksum,
            "finite": math.isfinite(retrieval_checksum) and math.isfinite(full_checksum),
            "gather_index_select_exact_match": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-count", type=int, required=True)
    parser.add_argument("--candidate-count", type=int, default=1280)
    parser.add_argument("--common-fraction", type=float, default=1.0)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--candidate-trace", type=Path)
    parser.add_argument(
        "--cpu-gather-mode",
        choices=("gather", "index_select"),
        default="gather",
    )
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--native-full-resident-ms", type=float, default=54.443)
    parser.add_argument("--qksieve-resident-ms", type=float, default=51.357)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
