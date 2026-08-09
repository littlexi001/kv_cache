from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import qabs_cuda_kernels as kernels


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_count", type=int, default=63744)
    parser.add_argument("--projection_dim", type=int, default=32)
    parser.add_argument("--candidate_fraction", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    batch, kv_heads, groups = 1, 8, 4
    generator = torch.Generator(device="cuda").manual_seed(20260715)
    query_codes = torch.randint(
        -127,
        128,
        (batch, kv_heads, groups, args.projection_dim),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    )
    key_codes = torch.randint(
        -127,
        128,
        (batch, kv_heads, args.history_count + 32, args.projection_dim),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    )
    key_scales = torch.rand(
        (batch, kv_heads, args.history_count + 32, 1),
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    padded_query = torch.zeros(
        (batch, kv_heads, 16, args.projection_dim),
        dtype=torch.int8,
        device="cuda",
    )
    padded_query[..., :groups, :].copy_(query_codes)
    reference = kernels.pca_int8_scores(
        query_codes, key_codes, args.history_count
    ).float() * key_scales[..., : args.history_count, 0].unsqueeze(2).float()
    actual = kernels.pca_int8_wmma_scores(
        padded_query,
        key_codes,
        key_scales,
        args.history_count,
        groups,
    ).reshape_as(reference)
    keep_count = math.ceil(args.history_count * args.candidate_fraction)
    reference_shared = torch.topk(
        reference.sum(dim=2), keep_count, dim=-1, sorted=False
    ).indices
    actual_shared = torch.topk(
        actual.sum(dim=2), keep_count, dim=-1, sorted=False
    ).indices
    overlap = (
        reference_shared.unsqueeze(-1)
        .eq(actual_shared.unsqueeze(-2))
        .any(dim=-1)
        .float()
        .mean()
    )
    result = {
        "history_count": args.history_count,
        "projection_dim": args.projection_dim,
        "keep_count": keep_count,
        "reference_shape": list(reference.shape),
        "actual_shape": list(actual.shape),
        "score_max_abs_error": float(
            (reference - actual.float()).abs().max().item()
        ),
        "score_mean_abs_error": float(
            (reference - actual.float()).abs().mean().item()
        ),
        "shared_topk_overlap": float(overlap.item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
