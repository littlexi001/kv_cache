from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch

import qabs_cuda_kernels as sparse_cuda
import sorted_sparse_attention_cuda_20260729 as sorted_sparse_cuda


def measure_ms(
    function: Callable[[], torch.Tensor],
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


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="32768,65536,120000")
    parser.add_argument("--candidate_count", type=int, default=1280)
    parser.add_argument("--capacity_fraction", type=float, default=0.06)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260729)
    rows: list[dict[str, float | int | str]] = []
    for history_count in sorted(
        {int(item) for item in args.lengths.split(",") if item}
    ):
        candidate_count = min(args.candidate_count, history_count)
        capacity = max(
            candidate_count,
            math.ceil(args.capacity_fraction * history_count),
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
        random_indices = torch.empty(
            1, 32, capacity, dtype=torch.long, device="cuda"
        )
        for head in range(32):
            random_indices[0, head, :candidate_count] = torch.randperm(
                history_count, device="cuda"
            )[:candidate_count]
        sorted_indices = random_indices.clone()
        sorted_indices[..., :candidate_count] = torch.sort(
            sorted_indices[..., :candidate_count], dim=-1
        ).values
        chunk_shuffled_indices = sorted_indices.clone()
        chunk_width = 32
        full_chunk_count = candidate_count // chunk_width
        if full_chunk_count > 1:
            prefix = chunk_shuffled_indices[
                ..., : full_chunk_count * chunk_width
            ].reshape(1, 32, full_chunk_count, chunk_width)
            for head in range(32):
                order = torch.randperm(
                    full_chunk_count, device="cuda"
                )
                prefix[0, head] = prefix[0, head].index_select(0, order)
        orders = {
            "random": random_indices,
            "chunk_shuffled": chunk_shuffled_indices,
            "sorted": sorted_indices,
        }
        counts = torch.full(
            (1, 32),
            candidate_count,
            dtype=torch.long,
            device="cuda",
        )
        scaling = 128.0**-0.5
        split = 8 if history_count <= 65536 else 4
        outputs: dict[tuple[str, str], torch.Tensor] = {}
        for order_name, indices in orders.items():
            baseline_function = (
                lambda indices=indices: sparse_cuda.final_attention_ragged_self_split(
                    query,
                    key,
                    value,
                    indices,
                    counts,
                    scaling,
                    split,
                )
            )
            sorted_function = (
                lambda indices=indices: sorted_sparse_cuda.forward(
                    query,
                    key,
                    value,
                    indices,
                    counts,
                    scaling,
                    split,
                )
            )
            baseline_output = baseline_function()
            sorted_output = sorted_function()
            outputs[(order_name, "baseline")] = baseline_output
            outputs[(order_name, "local_sort")] = sorted_output
            for kernel_name, function, output in (
                ("baseline", baseline_function, baseline_output),
                ("local_sort", sorted_function, sorted_output),
            ):
                latency = measure_ms(
                    function, args.warmup, args.iterations
                )
                rows.append(
                    {
                        "history_count": history_count,
                        "candidate_count": candidate_count,
                        "candidate_capacity": capacity,
                        "split": split,
                        "order": order_name,
                        "kernel": kernel_name,
                        "latency_ms": latency,
                        "max_abs_error_vs_baseline_same_order": float(
                            (output - baseline_output).abs().max().item()
                        ),
                    }
                )
        reference = outputs[("random", "baseline")]
        current_rows = rows[-2 * len(orders) :]
        random_baseline_ms = next(
            item["latency_ms"]
            for item in current_rows
            if item["order"] == "random"
            and item["kernel"] == "baseline"
        )
        for row in current_rows:
            row["speedup_vs_random"] = (
                random_baseline_ms / row["latency_ms"]
            )
            row["max_abs_error_vs_random"] = float(
                (
                    outputs[(row["order"], row["kernel"])] - reference
                ).abs().max().item()
            )
        del query, key, value, outputs, orders
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
