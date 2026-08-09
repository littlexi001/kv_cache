from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import qabs_cuda_kernels as kernels


def timed_ms(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history_lengths",
        type=int,
        nargs="+",
        default=[8192, 16000, 32000],
    )
    parser.add_argument("--candidate_fraction", type=float, default=0.06)
    parser.add_argument("--capacity_fraction", type=float, default=0.12)
    parser.add_argument("--split_counts", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.capacity_fraction < args.candidate_fraction:
        raise ValueError("capacity fraction must cover the candidate fraction")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16
    query_heads = 32
    kv_heads = 8
    head_dim = 128
    scaling = head_dim**-0.5
    rows = []

    for history_count in args.history_lengths:
        key_count = history_count + 1
        candidate_count = math.ceil(args.candidate_fraction * history_count)
        capacity = math.ceil(args.capacity_fraction * history_count)
        query = torch.randn(
            (1, query_heads, head_dim), device=device, dtype=dtype
        )
        key = torch.randn(
            (1, kv_heads, key_count, head_dim), device=device, dtype=dtype
        )
        value = torch.randn_like(key)
        rank = torch.arange(
            capacity, device=device, dtype=torch.long
        ).view(1, 1, -1)
        head = torch.arange(
            query_heads, device=device, dtype=torch.long
        ).view(1, -1, 1)
        indices = (
            rank * 8191 + head * 131071
        ) % history_count
        counts = torch.full(
            (1, query_heads),
            candidate_count,
            device=device,
            dtype=torch.long,
        )

        def baseline() -> torch.Tensor:
            return kernels.final_attention_ragged_self(
                query, key, value, indices, counts, scaling
            )

        reference = baseline()
        baseline_ms = timed_ms(baseline, args.warmup, args.repeats)
        methods = {
            "single_block": {
                "milliseconds": baseline_ms,
                "speedup": 1.0,
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
            }
        }
        for split_count in args.split_counts:
            def split_attention(
                split_count: int = split_count,
            ) -> torch.Tensor:
                return kernels.final_attention_ragged_self_split(
                    query,
                    key,
                    value,
                    indices,
                    counts,
                    scaling,
                    split_count,
                )

            output = split_attention()
            elapsed_ms = timed_ms(
                split_attention, args.warmup, args.repeats
            )
            methods[f"split{split_count}"] = {
                "milliseconds": elapsed_ms,
                "speedup": baseline_ms / elapsed_ms,
                "max_abs_error": float(
                    (reference - output).abs().max().item()
                ),
                "mean_abs_error": float(
                    (reference - output).abs().mean().item()
                ),
            }

        row = {
            "history_count": history_count,
            "candidate_count": candidate_count,
            "capacity": capacity,
            "methods": methods,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del query, key, value, indices, counts, reference
        torch.cuda.empty_cache()

    payload = {
        "config": {
            "candidate_fraction": args.candidate_fraction,
            "capacity_fraction": args.capacity_fraction,
            "split_counts": args.split_counts,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
