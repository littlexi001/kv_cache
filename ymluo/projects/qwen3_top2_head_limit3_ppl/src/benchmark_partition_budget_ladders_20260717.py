from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from run_head_top2_targeted_ppl_20260714 import (
    _partition_global_sample_budget_ladder,
    _partition_proxy_ucb_budget_ladder,
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=131072)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--sample_fraction", type=float, default=0.0025)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(20260719)
    history_count = args.history_tokens
    fractions = (0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08)
    keep_counts = tuple(math.ceil(fraction * history_count) for fraction in fractions)
    candidate_count = keep_counts[-1]
    sample_count = math.ceil(args.sample_fraction * history_count)
    stride = max(1, history_count // sample_count)
    sample_indices = torch.arange(sample_count, device=device) * stride
    sample_indices = sample_indices.clamp_max(history_count - 1)

    proxy_scores = torch.randn(1, args.heads, history_count, device=device)
    proxy_scores = proxy_scores * 0.8
    candidate_indices = torch.topk(
        proxy_scores, candidate_count, dim=-1, sorted=True
    ).indices
    sample_scores = proxy_scores.index_select(-1, sample_indices) + 0.25 * torch.randn(
        1, args.heads, sample_count, device=device
    )
    self_scores = torch.randn(1, args.heads, device=device)

    old_ms = timed_ms(
        lambda: _partition_proxy_ucb_budget_ladder(
            proxy_scores,
            candidate_indices,
            sample_scores,
            sample_indices,
            self_scores,
            keep_counts,
            0.75,
            0.0,
        ),
        args.warmup,
        args.repeats,
    )
    global_ms = timed_ms(
        lambda: _partition_global_sample_budget_ladder(
            proxy_scores,
            candidate_indices,
            sample_scores,
            sample_indices,
            self_scores,
            keep_counts,
            0.75,
            0.0,
        ),
        args.warmup,
        args.repeats,
    )
    result = {
        "history_tokens": history_count,
        "heads": args.heads,
        "sample_fraction": args.sample_fraction,
        "candidate_fraction": fractions[-1],
        "rung_count": len(fractions),
        "prefix_conditioned_ms": old_ms,
        "global_vectorized_ms": global_ms,
        "controller_speedup": old_ms / global_ms,
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
