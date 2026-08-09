from __future__ import annotations

import math
import time

import torch

import qksieve_valuesketch_cuda_20260801 as value_sketch_cuda
from qksieve_dual_mass_cuda_20260803 import (
    dual_mass_candidates_with_value_tail,
)


def main() -> None:
    torch.manual_seed(20260803)
    device = torch.device("cuda")
    batch, query_heads, kv_heads, history, head_dim, rank = (
        1,
        32,
        8,
        4096,
        128,
        16,
    )
    scaling = 1.0 / math.sqrt(head_dim)
    proxy = torch.randn(batch, query_heads, history, device=device) * 2.5
    risk = torch.randn(batch, kv_heads, history, device=device) * 0.5
    slope = torch.ones(batch, query_heads, device=device)
    intercept = torch.zeros_like(slope)
    packed_codes = torch.randint(
        0,
        256,
        (batch, kv_heads, history, rank // 2),
        dtype=torch.uint8,
        device=device,
    )
    block_count = math.ceil(history / 256)
    minimum = torch.randn(
        batch, kv_heads, block_count, rank, device=device
    ) * 0.1
    scale = 0.01 + torch.rand_like(minimum) * 0.05
    query = torch.randn(
        batch, query_heads, head_dim, dtype=torch.float16, device=device
    )
    key = torch.randn(
        batch,
        kv_heads,
        history + 1,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    value = torch.randn_like(key)
    value_mean = torch.randn(
        batch, kv_heads, head_dim, dtype=torch.float16, device=device
    )
    value_basis = torch.randn(
        batch,
        kv_heads,
        head_dim,
        rank,
        dtype=torch.float16,
        device=device,
    )

    start = time.perf_counter()
    result = dual_mass_candidates_with_value_tail(
        proxy,
        risk,
        slope,
        intercept,
        packed_codes,
        minimum,
        scale,
        0.975,
        492,
        candidate_capacity=history,
    )
    torch.cuda.synchronize()
    print("selector_seconds", time.perf_counter() - start, flush=True)
    indices, counts, _, _, overflow, anchor, denominator, coefficients = result
    capacity = int(counts.max().item())
    indices = indices[..., :capacity].contiguous()
    print(
        "capacity",
        capacity,
        "mean",
        float(counts.float().mean().item()),
        "overflow",
        int(overflow.sum().item()),
        flush=True,
    )

    start = time.perf_counter()
    output = value_sketch_cuda.exact_selected_plus_tail(
        query,
        key,
        value,
        indices,
        counts,
        anchor / scaling,
        denominator,
        coefficients,
        value_mean,
        value_basis,
        scaling,
        1.0,
    )
    torch.cuda.synchronize()
    print(
        "attention_seconds",
        time.perf_counter() - start,
        "finite",
        bool(torch.isfinite(output).all().item()),
        flush=True,
    )


if __name__ == "__main__":
    main()
