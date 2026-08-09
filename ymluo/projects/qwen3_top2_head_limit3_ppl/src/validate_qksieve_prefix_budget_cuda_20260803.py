from __future__ import annotations

import argparse
import json

import torch

from qksieve_prefix_budget_cuda_20260803 import (
    prefix_tail_budget_candidates,
)


def timed_ms(callable_, repeats: int) -> float:
    for _ in range(3):
        callable_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        callable_()
    stop.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(stop) / repeats)


def exact_prefix(
    scores: torch.Tensor,
    tail_weights: torch.Tensor,
    allowed_tail: torch.Tensor,
    floor_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sorted_scores, sorted_indices = torch.sort(
        scores, dim=-1, descending=True
    )
    del sorted_scores
    ordered_weights = torch.gather(tail_weights, -1, sorted_indices)
    cumulative = torch.cumsum(ordered_weights, dim=-1)
    tail = (cumulative[..., -1:] - cumulative).clamp_min(0.0)
    counts = torch.sum(tail > allowed_tail.unsqueeze(-1), dim=-1) + 1
    counts = counts.clamp(min=floor_k, max=scores.shape[-1])
    return sorted_indices, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=131_008)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--floor_k", type=int, default=1280)
    parser.add_argument("--tail_fraction", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260803)
    scores = torch.randn(
        1,
        args.heads,
        args.history_tokens,
        generator=generator,
        device="cuda",
    )
    tail_weights = torch.rand(
        scores.shape,
        generator=generator,
        device="cuda",
    ).square()
    allowed_tail = tail_weights.sum(dim=-1) * args.tail_fraction

    indices, counts, _ = prefix_tail_budget_candidates(
        scores, tail_weights, allowed_tail, args.floor_k
    )
    capacity = indices.shape[-1]
    valid = (
        torch.arange(capacity, device="cuda").reshape(1, 1, -1)
        < counts.unsqueeze(-1)
    )
    selected_mask = torch.zeros_like(scores, dtype=torch.bool)
    coordinates = torch.nonzero(valid, as_tuple=False)
    selected_mask[
        coordinates[:, 0],
        coordinates[:, 1],
        indices[
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        ],
    ] = True
    omitted_tail = tail_weights.masked_fill(selected_mask, 0.0).sum(dim=-1)
    selected_floor = scores.masked_fill(~selected_mask, torch.inf).amin(dim=-1)
    omitted_ceiling = scores.masked_fill(selected_mask, -torch.inf).amax(dim=-1)
    exact_indices, exact_counts = exact_prefix(
        scores, tail_weights, allowed_tail, args.floor_k
    )
    del exact_indices

    histogram_ms = timed_ms(
        lambda: prefix_tail_budget_candidates(
            scores, tail_weights, allowed_tail, args.floor_k
        ),
        args.repeats,
    )
    sort_ms = timed_ms(
        lambda: exact_prefix(scores, tail_weights, allowed_tail, args.floor_k),
        args.repeats,
    )
    payload = {
        "history_tokens": args.history_tokens,
        "heads": args.heads,
        "floor_k": args.floor_k,
        "tail_fraction": args.tail_fraction,
        "histogram_ms": histogram_ms,
        "exact_sort_ms": sort_ms,
        "speedup": sort_ms / histogram_ms,
        "mean_count": float(counts.float().mean()),
        "mean_exact_count": float(exact_counts.float().mean()),
        "count_overhead_ratio": float(
            counts.float().sum() / exact_counts.float().sum()
        ),
        "minimum_count": int(counts.min()),
        "maximum_count": int(counts.max()),
        "maximum_tail_violation": float(
            (omitted_tail - allowed_tail).max()
        ),
        "minimum_prefix_gap": float(
            (selected_floor - omitted_ceiling).min()
        ),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
