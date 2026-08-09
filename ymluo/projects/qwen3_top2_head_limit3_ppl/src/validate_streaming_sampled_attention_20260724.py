#!/usr/bin/env python3
"""Validate one-block sampled-quantile scan plus exact sparse attention."""

from __future__ import annotations

import argparse
import json
import math

import torch

import qabs_cuda_kernels as kernels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="8192,16000,32000")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def time_cuda(callable_, warmup: int, repeats: int) -> float:
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
    counts_cpu = counts.reshape(-1).clamp_max(indices.shape[-1]).cpu()
    return [
        sorted(indices_cpu[row, : int(count)].tolist())
        for row, count in enumerate(counts_cpu)
    ]


def run_length(
    history_count: int,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | bool]:
    torch.manual_seed(20260724 + history_count)
    device = torch.device("cuda")
    dtype = torch.float16
    projection_dim = 48
    kv_heads = 8
    groups = 4
    head_dim = 128
    chunks = projection_dim // 16
    candidate_capacity = math.ceil(0.12 * history_count)
    sample_count = 256
    scaling = head_dim**-0.5
    query = torch.randn(
        1, kv_heads * groups, head_dim, device=device, dtype=dtype
    )
    key = torch.randn(
        1,
        kv_heads,
        history_count + 1,
        head_dim,
        device=device,
        dtype=dtype,
    )
    value = torch.randn_like(key)
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
    projected_key = torch.einsum(
        "bhnd,bhdm->bhnm",
        key[..., :history_count, :],
        basis,
    )
    packed = torch.empty(
        1,
        kv_heads,
        chunks,
        history_count,
        8,
        device=device,
        dtype=torch.uint8,
    )
    scales = torch.empty(
        1,
        kv_heads,
        history_count,
        1,
        device=device,
        dtype=dtype,
    )
    exponents = torch.empty(
        1,
        kv_heads,
        history_count,
        math.ceil(chunks / 2),
        device=device,
        dtype=torch.uint8,
    )
    kernels.pca_int4_logscale16_pack_into(
        projected_key,
        packed,
        scales,
        exponents,
        0,
    )

    def reference():
        result = (
            kernels.pca_int4_logscale16_raw_query_sampled_quantile_candidates(
                query,
                basis,
                packed,
                scales,
                exponents,
                history_count,
                sample_count,
                0.06,
                candidate_capacity,
                use_dp4a=False,
                write_proxy_scores=False,
            )
        )
        output = kernels.final_attention_ragged_self(
            query,
            key,
            value,
            result[0],
            result[2],
            scaling,
        )
        return output, result

    def streaming():
        return kernels.pca_int4_logscale16_streaming_attention(
            query,
            key,
            value,
            basis,
            packed,
            scales,
            exponents,
            history_count,
            sample_count,
            0.06,
            candidate_capacity,
            scaling,
            use_dp4a=False,
        )

    reference_output, reference_result = reference()
    streaming_result = streaming()
    torch.cuda.synchronize()
    streaming_output = streaming_result[0].unsqueeze(1)
    output_difference = (
        streaming_output.float() - reference_output.float()
    )
    reference_sets = candidate_sets(
        reference_result[0],
        reference_result[2],
    )
    streaming_sets = candidate_sets(
        streaming_result[1],
        streaming_result[2],
    )
    candidate_jaccards = [
        len(set(reference) & set(streaming))
        / max(1, len(set(reference) | set(streaming)))
        for reference, streaming in zip(
            reference_sets,
            streaming_sets,
        )
    ]
    reference_ms = time_cuda(reference, warmup, repeats)
    streaming_ms = time_cuda(streaming, warmup, repeats)
    return {
        "history_count": history_count,
        "candidate_sets_exact": (
            reference_sets == streaming_sets
        ),
        "counts_exact": bool(
            torch.equal(reference_result[2], streaming_result[2])
        ),
        "boundaries_exact": bool(
            torch.equal(reference_result[3], streaming_result[3])
        ),
        "boundary_max_abs_error": float(
            (
                reference_result[3] - streaming_result[3]
            ).abs().max().item()
        ),
        "count_max_abs_error": int(
            (
                reference_result[2] - streaming_result[2]
            ).abs().max().item()
        ),
        "reference_count_mean": float(
            reference_result[2].float().mean().item()
        ),
        "streaming_count_mean": float(
            streaming_result[2].float().mean().item()
        ),
        "reference_count_min": int(reference_result[2].min().item()),
        "reference_count_max": int(reference_result[2].max().item()),
        "streaming_count_min": int(streaming_result[2].min().item()),
        "streaming_count_max": int(streaming_result[2].max().item()),
        "candidate_jaccard_mean": (
            sum(candidate_jaccards) / len(candidate_jaccards)
        ),
        "query_codes_exact": bool(
            torch.equal(reference_result[5], streaming_result[5])
        ),
        "output_max_abs_error": float(output_difference.abs().max().item()),
        "output_mean_abs_error": float(output_difference.abs().mean().item()),
        "reference_ms": reference_ms,
        "streaming_ms": streaming_ms,
        "speedup": reference_ms / streaming_ms,
    }


def main() -> None:
    args = parse_args()
    lengths = [
        int(value.strip())
        for value in args.lengths.split(",")
        if value.strip()
    ]
    rows = [
        run_length(length, args.warmup, args.repeats)
        for length in lengths
    ]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
