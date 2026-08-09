from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import qabs_cuda_kernels


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
    end.synchronize()
    return float(start.elapsed_time(end) / repeats)


def validate_ranges(device: torch.device) -> dict[str, float]:
    torch.manual_seed(20260717)
    batch_count, query_heads, kv_heads = 2, 8, 2
    key_count, candidate_count, head_dim = 8192, 512, 128
    query = torch.randn(
        batch_count, query_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(
        batch_count,
        kv_heads,
        key_count,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    indices = torch.randint(
        key_count,
        (batch_count, query_heads, candidate_count),
        device=device,
        dtype=torch.long,
    )
    scaling = head_dim**-0.5
    reference = qabs_cuda_kernels.candidate_compact_scores(
        query, key, indices, scaling
    )
    output = torch.full_like(reference, -torch.inf)
    starts = torch.zeros((batch_count, query_heads), device=device, dtype=torch.long)
    first_ends = torch.tensor(
        [[32, 64, 96, 128, 160, 192, 224, 256]],
        device=device,
        dtype=torch.long,
    ).expand(batch_count, -1)
    qabs_cuda_kernels.candidate_compact_scores_range(
        query, key, indices, starts, first_ends, output, scaling
    )
    positions = torch.arange(candidate_count, device=device).view(1, 1, -1)
    first_valid = positions < first_ends.unsqueeze(-1)
    first_error = (output[first_valid] - reference[first_valid]).abs().max().item()
    first_untouched = bool(torch.isneginf(output[~first_valid]).all().item())

    second_ends = torch.tensor(
        [[64, 64, 160, 192, 160, 320, 384, 512]],
        device=device,
        dtype=torch.long,
    ).expand(batch_count, -1)
    qabs_cuda_kernels.candidate_compact_scores_range(
        query, key, indices, first_ends, second_ends, output, scaling
    )
    second_valid = positions < second_ends.unsqueeze(-1)
    second_error = (output[second_valid] - reference[second_valid]).abs().max().item()
    second_untouched = bool(torch.isneginf(output[~second_valid]).all().item())
    if first_error != 0.0 or second_error != 0.0:
        raise AssertionError(
            f"incremental scores differ: first={first_error}, second={second_error}"
        )
    if not first_untouched or not second_untouched:
        raise AssertionError("incremental scorer modified an unopened candidate range")
    return {
        "first_max_abs_error": first_error,
        "second_max_abs_error": second_error,
        "unopened_ranges_untouched": 1.0,
    }


def benchmark_128k(
    device: torch.device, warmup: int, repeats: int
) -> dict[str, float]:
    torch.manual_seed(20260718)
    batch_count, query_heads, kv_heads = 1, 32, 8
    key_count, head_dim = 131072, 128
    candidate_count = math.ceil(0.08 * key_count)
    query = torch.randn(
        batch_count, query_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(
        batch_count,
        kv_heads,
        key_count,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    indices = torch.randint(
        key_count,
        (batch_count, query_heads, candidate_count),
        device=device,
        dtype=torch.long,
    )
    starts = torch.zeros((batch_count, query_heads), device=device, dtype=torch.long)
    full_ends = torch.full_like(starts, candidate_count)
    fractions = [0.01] * 18 + [0.02] * 4 + [0.04] * 3 + [0.06] * 2 + [0.08] * 5
    dynamic_ends = torch.tensor(
        [min(candidate_count, math.ceil(fraction * key_count)) for fraction in fractions],
        device=device,
        dtype=torch.long,
    ).view(1, query_heads)
    scaling = head_dim**-0.5
    full_output = torch.empty(
        (batch_count, query_heads, candidate_count),
        device=device,
        dtype=torch.float32,
    )
    dynamic_output = torch.full_like(full_output, -torch.inf)

    full_range_ms = timed_ms(
        lambda: qabs_cuda_kernels.candidate_compact_scores_range(
            query, key, indices, starts, full_ends, full_output, scaling
        ),
        warmup,
        repeats,
    )
    dynamic_range_ms = timed_ms(
        lambda: qabs_cuda_kernels.candidate_compact_scores_range(
            query, key, indices, starts, dynamic_ends, dynamic_output, scaling
        ),
        warmup,
        repeats,
    )
    legacy_full_ms = timed_ms(
        lambda: qabs_cuda_kernels.candidate_compact_scores(
            query, key, indices, scaling
        ),
        warmup,
        repeats,
    )
    dynamic_fraction = dynamic_ends.float().mean().item() / key_count
    return {
        "history_tokens": float(key_count),
        "candidate_capacity_fraction": candidate_count / key_count,
        "dynamic_exact_qk_fraction": dynamic_fraction,
        "full_range_ms": full_range_ms,
        "dynamic_range_ms": dynamic_range_ms,
        "legacy_full_ms": legacy_full_ms,
        "range_kernel_speedup": full_range_ms / dynamic_range_ms,
        "legacy_to_dynamic_speedup": legacy_full_ms / dynamic_range_ms,
        "ideal_qk_traffic_speedup": (candidate_count / key_count) / dynamic_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    result = {
        "validation": validate_ranges(device),
        "benchmark_128k": benchmark_128k(device, args.warmup, args.repeats),
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
