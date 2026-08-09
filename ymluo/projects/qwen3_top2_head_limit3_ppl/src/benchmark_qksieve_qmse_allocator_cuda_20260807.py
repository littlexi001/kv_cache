#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import qksieve_qmse_allocator_cuda_20260807 as fused
from run_head_top2_targeted_ppl_20260714 import (
    _feasible_hierarchical_rate_allocations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=288)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def reference(
    costs: torch.Tensor,
    allocations: torch.Tensor,
    bit_indices: torch.Tensor,
    used_bits: torch.Tensor,
) -> torch.Tensor:
    candidate_costs = torch.zeros(
        (*costs.shape[:2], allocations.shape[0]),
        dtype=torch.float64,
        device=costs.device,
    )
    for band in range(8):
        candidate_costs += costs[:, :, band, :].index_select(
            -1, bit_indices[:, band]
        )
    minimum_cost = candidate_costs.amin(dim=-1, keepdim=True)
    minimum_mask = candidate_costs == minimum_cost
    used = used_bits.reshape(1, 1, -1)
    maximum_used = torch.where(
        minimum_mask,
        used,
        torch.full_like(used, -1),
    ).amax(dim=-1, keepdim=True)
    preferred = minimum_mask & (used == maximum_used)
    best = preferred.to(torch.int8).argmax(dim=-1)
    return allocations.index_select(0, best.reshape(-1)).reshape(
        *costs.shape[:2], 8
    )


def timed(function, repeats: int) -> float:
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.rows % 8:
        raise ValueError("rows must be a positive multiple of eight")
    device = torch.device("cuda")
    bit_levels = (0, 1, 2, 4, 8)
    host = _feasible_hierarchical_rate_allocations(bit_levels, 15, True)
    allocations, bit_indices, used_bits = (
        value.to(device) for value in host
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    random_costs = torch.rand(
        args.rows // 8,
        8,
        8,
        len(bit_levels),
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    cases = {
        "random": random_costs,
        "all_equal": torch.zeros_like(random_costs),
        "quantized_ties": torch.round(random_costs * 4.0) / 4.0,
    }
    rows = []
    for name, costs in cases.items():
        expected = reference(costs, allocations, bit_indices, used_bits)
        actual = fused.allocate(
            costs, allocations, bit_indices, used_bits
        )
        reference_seconds = timed(
            lambda: reference(costs, allocations, bit_indices, used_bits),
            args.repeats,
        )
        fused_seconds = timed(
            lambda: fused.allocate(
                costs, allocations, bit_indices, used_bits
            ),
            args.repeats,
        )
        rows.append(
            {
                "case": name,
                "bitwise_equal": bool(torch.equal(expected, actual)),
                "reference_us": reference_seconds * 1.0e6,
                "fused_us": fused_seconds * 1.0e6,
                "speedup": reference_seconds / fused_seconds,
            }
        )
    payload = {
        "rows_per_call": args.rows,
        "allocation_count": int(allocations.shape[0]),
        "repeats": args.repeats,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
