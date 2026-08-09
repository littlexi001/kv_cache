from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(quantile * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_count", type=int, default=131_072)
    parser.add_argument("--new_token_fraction", type=float, default=0.0064)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    selected_count = max(1, math.ceil(args.history_count * args.new_token_fraction))
    shape = (2, args.kv_heads, args.history_count, args.head_dim)
    selected_shape = (2, args.kv_heads, selected_count, args.head_dim)
    host_full = torch.empty(shape, dtype=torch.float16, pin_memory=True)
    host_full.random_(-64, 64)
    host_selected = torch.empty(selected_shape, dtype=torch.float16, pin_memory=True)
    device_selected = torch.empty(selected_shape, dtype=torch.float16, device="cuda")

    generator = torch.Generator().manual_seed(20260715)
    index_batches = []
    for _ in range(args.warmup + 2 * args.repeats):
        indices = torch.randint(
            0,
            args.history_count,
            (args.kv_heads, selected_count),
            generator=generator,
            dtype=torch.long,
        )
        index_batches.append(
            indices.unsqueeze(0)
            .unsqueeze(-1)
            .expand(2, -1, -1, args.head_dim)
            .contiguous()
        )

    def gather(index: torch.Tensor) -> None:
        torch.gather(host_full, dim=2, index=index, out=host_selected)

    def transfer() -> None:
        device_selected.copy_(host_selected, non_blocking=True)
        torch.cuda.synchronize()

    for index in index_batches[: args.warmup]:
        gather(index)
        transfer()

    gather_ms = []
    transfer_ms = []
    combined_ms = []
    measured_indices = index_batches[args.warmup : args.warmup + args.repeats]
    combined_indices = index_batches[args.warmup + args.repeats :]
    for index, combined_index in zip(measured_indices, combined_indices, strict=True):
        started = time.perf_counter()
        gather(index)
        gather_ms.append(1000.0 * (time.perf_counter() - started))

        started = time.perf_counter()
        transfer()
        transfer_ms.append(1000.0 * (time.perf_counter() - started))

        started = time.perf_counter()
        gather(combined_index)
        transfer()
        combined_ms.append(1000.0 * (time.perf_counter() - started))

    bytes_per_layer = host_selected.numel() * host_selected.element_size()
    result = {
        "history_count": args.history_count,
        "new_token_fraction": args.new_token_fraction,
        "selected_count_per_kv_head": selected_count,
        "bytes_per_layer": bytes_per_layer,
        "mib_per_layer": bytes_per_layer / 2**20,
        "mib_all_layers": bytes_per_layer * args.layers / 2**20,
        "cpu_gather_mean_ms_per_layer": sum(gather_ms) / len(gather_ms),
        "cpu_gather_p50_ms_per_layer": percentile(gather_ms, 0.50),
        "cpu_gather_p95_ms_per_layer": percentile(gather_ms, 0.95),
        "h2d_mean_ms_per_layer": sum(transfer_ms) / len(transfer_ms),
        "h2d_p50_ms_per_layer": percentile(transfer_ms, 0.50),
        "h2d_p95_ms_per_layer": percentile(transfer_ms, 0.95),
        "sequential_mean_ms_per_layer": sum(combined_ms) / len(combined_ms),
        "sequential_p50_ms_per_layer": percentile(combined_ms, 0.50),
        "sequential_p95_ms_per_layer": percentile(combined_ms, 0.95),
    }
    result["sequential_mean_ms_all_layers"] = (
        result["sequential_mean_ms_per_layer"] * args.layers
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
