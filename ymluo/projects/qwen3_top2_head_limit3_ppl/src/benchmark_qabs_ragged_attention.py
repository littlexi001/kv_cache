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


def build_ragged_indices(
    history_count: int,
    fractions: tuple[float, ...],
    head_histogram: tuple[int, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(fractions) != len(head_histogram):
        raise ValueError("fractions and head_histogram must have the same length")
    history_counts = torch.cat(
        [
            torch.full((head_count,), max(1, math.ceil(history_count * fraction)), device=device, dtype=torch.long)
            for fraction, head_count in zip(fractions, head_histogram, strict=True)
        ]
    ).unsqueeze(0)
    max_history_count = max(1, math.ceil(history_count * max(fractions)))
    rank = torch.arange(max_history_count + 1, device=device, dtype=torch.long).view(1, 1, -1)
    head = torch.arange(history_counts.shape[1], device=device, dtype=torch.long).view(1, -1, 1)
    indices = (rank * 8191 + head * 131071) % history_count
    indices.scatter_(-1, history_counts.unsqueeze(-1), history_count)
    counts = history_counts + 1
    valid = rank < counts.unsqueeze(-1)
    return indices.contiguous(), counts.contiguous(), valid.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_lengths", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0025, 0.005, 0.01, 0.02, 0.04])
    parser.add_argument("--head_histogram", type=int, nargs="+", default=[11, 10, 6, 3, 2])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16
    head_count = sum(args.head_histogram)
    scaling = args.head_dim**-0.5
    rows = []

    for history_count in args.history_lengths:
        key_count = history_count + 1
        query = torch.randn((1, head_count, args.head_dim), device=device, dtype=dtype)
        key = torch.randn((1, head_count, key_count, args.head_dim), device=device, dtype=dtype)
        value = torch.randn_like(key)
        indices, counts, valid = build_ragged_indices(
            history_count,
            tuple(args.fractions),
            tuple(args.head_histogram),
            device,
        )

        def full_attention() -> torch.Tensor:
            scores = torch.matmul(query.unsqueeze(2), key.transpose(2, 3)) * scaling
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
            return torch.matmul(weights, value)

        def padded_attention() -> torch.Tensor:
            return kernels.final_attention(query, key, value, indices, valid, scaling)

        def ragged_attention() -> torch.Tensor:
            return kernels.final_attention_ragged(query, key, value, indices, counts, scaling)

        padded_output = padded_attention()
        ragged_output = ragged_attention()
        max_abs_error = float((padded_output - ragged_output).abs().max().item())
        mean_abs_error = float((padded_output - ragged_output).abs().mean().item())
        full_ms = timed_ms(full_attention, args.warmup, args.repeats)
        padded_ms = timed_ms(padded_attention, args.warmup, args.repeats)
        ragged_ms = timed_ms(ragged_attention, args.warmup, args.repeats)
        mean_history_fraction = float(((counts - 1).float() / history_count).mean().item())
        row = {
            "history_count": history_count,
            "mean_history_fraction": mean_history_fraction,
            "max_history_fraction": max(args.fractions),
            "full_ms": full_ms,
            "padded_ms": padded_ms,
            "ragged_ms": ragged_ms,
            "full_over_padded": full_ms / padded_ms,
            "full_over_ragged": full_ms / ragged_ms,
            "padded_over_ragged": padded_ms / ragged_ms,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)
        del query, key, value, indices, counts, valid, padded_output, ragged_output
        torch.cuda.empty_cache()

    payload = {
        "config": {
            "head_count": head_count,
            "head_dim": args.head_dim,
            "fractions": args.fractions,
            "head_histogram": args.head_histogram,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
