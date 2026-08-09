#!/usr/bin/env python3
"""Validate fused query projection plus sampled-quantile thresholding."""

from __future__ import annotations

import json
import math

import torch

import qabs_cuda_kernels as kernels


def time_cuda(callable_, warmup: int = 20, repeats: int = 500) -> float:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        callable_()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / repeats


def candidate_sets(
    indices: torch.Tensor,
    counts: torch.Tensor,
) -> list[list[int]]:
    indices_cpu = indices.reshape(-1, indices.shape[-1]).cpu()
    counts_cpu = counts.reshape(-1).cpu()
    return [
        sorted(indices_cpu[row, : int(count)].tolist())
        for row, count in enumerate(counts_cpu)
    ]


def main() -> None:
    torch.manual_seed(20260724)
    device = torch.device("cuda")
    dtype = torch.float16
    key_count = 8192
    projection_dim = 48
    kv_heads = 8
    groups = 4
    head_dim = 128
    chunks = projection_dim // 16
    capacity = key_count
    candidate_capacity = math.ceil(0.12 * key_count)
    sample_count = 128
    query = torch.randn(
        1, kv_heads * groups, head_dim, device=device, dtype=dtype
    )
    basis = torch.linalg.qr(
        torch.randn(
            1,
            kv_heads,
            head_dim,
            projection_dim,
            device=device,
            dtype=torch.float32,
        )
    ).Q.to(dtype)
    projected_key = torch.randn(
        1,
        kv_heads,
        key_count,
        projection_dim,
        device=device,
        dtype=dtype,
    )
    packed = torch.empty(
        1, kv_heads, chunks, capacity, 8, device=device, dtype=torch.uint8
    )
    scales = torch.empty(
        1, kv_heads, capacity, 1, device=device, dtype=dtype
    )
    exponents = torch.empty(
        1,
        kv_heads,
        capacity,
        math.ceil(chunks / 2),
        device=device,
        dtype=torch.uint8,
    )
    kernels.pca_int4_logscale16_pack_into(
        projected_key, packed, scales, exponents, 0
    )
    grouped_query = query.reshape(1, kv_heads, groups, head_dim)

    def reference():
        _, codes, query_scales = kernels.pca_project_query_int8(
            grouped_query, basis
        )
        candidates = kernels.pca_int4_logscale16_sampled_quantile_candidates(
            codes,
            packed,
            scales,
            exponents,
            key_count,
            sample_count,
            0.06,
            candidate_capacity,
            use_dp4a=False,
            write_proxy_scores=False,
        )
        return (*candidates, codes, query_scales)

    def fused():
        return kernels.pca_int4_logscale16_raw_query_sampled_quantile_candidates(
            query,
            basis,
            packed,
            scales,
            exponents,
            key_count,
            sample_count,
            0.06,
            candidate_capacity,
            use_dp4a=False,
            write_proxy_scores=False,
        )

    reference_outputs = reference()
    fused_outputs = fused()
    torch.cuda.synchronize()
    reference_sets = candidate_sets(reference_outputs[0], reference_outputs[2])
    fused_sets = candidate_sets(fused_outputs[0], fused_outputs[2])
    reference_ms = time_cuda(reference)
    fused_ms = time_cuda(fused)
    print(
        json.dumps(
            {
                "candidate_sets_exact": reference_sets == fused_sets,
                "sample_count": sample_count,
                "counts_exact": bool(
                    torch.equal(reference_outputs[2], fused_outputs[2])
                ),
                "boundaries_exact": bool(
                    torch.equal(reference_outputs[3], fused_outputs[3])
                ),
                "query_codes_exact": bool(
                    torch.equal(reference_outputs[5], fused_outputs[5])
                ),
                "query_scale_max_abs_error": float(
                    (
                        reference_outputs[6].reshape_as(fused_outputs[6])
                        - fused_outputs[6]
                    ).abs().max().item()
                ),
                "reference_ms": reference_ms,
                "fused_ms": fused_ms,
                "speedup": reference_ms / fused_ms,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
