#!/usr/bin/env python3
"""Directly time the request-local QKSieve index-build stages."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch

import variablebit_spectral_cuda_20260727 as varbit_cuda
from run_head_top2_targeted_ppl_20260714 import (
    _hierarchical_key_rate_allocation,
    _qk_metric_projection_factors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=131072)
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--prefill_query_tokens", type=int, default=8)
    parser.add_argument("--chunk_tokens", type=int, default=4096)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def cuda_ms(function: Callable[[], Any], repeats: int, warmup: int = 1) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return float(torch.tensor(samples).median().item())


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    history = args.history_tokens
    key = torch.randn(1, 8, history, 128, dtype=dtype, device="cuda")
    grouped_queries = torch.randn(
        1,
        8,
        args.prefill_query_tokens * 4,
        128,
        dtype=dtype,
        device="cuda",
    )
    sampled_key = key[..., :: args.sample_stride, :].float().contiguous()

    key_second_moment = torch.einsum(
        "bhkd,bhke->bhde", sampled_key, sampled_key
    ) / float(sampled_key.shape[-2])
    query_second_moment = torch.einsum(
        "bhnd,bhne->bhde", grouped_queries.float(), grouped_queries.float()
    ) / float(grouped_queries.shape[-2])
    query_factor, key_factor = _qk_metric_projection_factors(
        key_second_moment,
        query_second_moment,
        projection_dim=128,
        query_shrinkage=args.query_shrinkage,
    )
    basis = key_factor.to(dtype).contiguous()
    projected_sample = torch.einsum(
        "bhkd,bhdm->bhkm", sampled_key.to(dtype), basis
    )
    allocation = _hierarchical_key_rate_allocation(
        projected_sample,
        bit_budget_per_coordinate=15,
        allow_zero_bits=True,
        include_scale_metadata=True,
    ).to(torch.int8)
    packed_index = varbit_cuda.allocate_packed_index(
        allocation, history, dtype
    )
    projected_full = torch.empty_like(key)

    def materialize_sample() -> torch.Tensor:
        return key[..., :: args.sample_stride, :].float().contiguous()

    def key_moment() -> torch.Tensor:
        return torch.einsum(
            "bhkd,bhke->bhde", sampled_key, sampled_key
        ) / float(sampled_key.shape[-2])

    def query_moment() -> torch.Tensor:
        query_float = grouped_queries.float()
        return torch.einsum(
            "bhnd,bhne->bhde", query_float, query_float
        ) / float(query_float.shape[-2])

    def key_pca_eigh() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.eigh(key_second_moment)

    def solve_transform() -> tuple[torch.Tensor, torch.Tensor]:
        return _qk_metric_projection_factors(
            key_second_moment,
            query_second_moment,
            projection_dim=128,
            query_shrinkage=args.query_shrinkage,
        )

    def project_sample() -> torch.Tensor:
        return torch.einsum(
            "bhkd,bhdm->bhkm", sampled_key.to(dtype), basis
        )

    def allocate_bits() -> torch.Tensor:
        return _hierarchical_key_rate_allocation(
            projected_sample,
            bit_budget_per_coordinate=15,
            allow_zero_bits=True,
            include_scale_metadata=True,
        )

    def current_code_calibration() -> torch.Tensor:
        current_sample = key[..., :: args.sample_stride, :].float().contiguous()
        current_key_moment = torch.einsum(
            "bhkd,bhke->bhde", current_sample, current_sample
        ) / float(current_sample.shape[-2])
        # The current implementation first computes this Key-PCA solution, then
        # replaces it with the QK-balanced factors below.
        _, current_key_vectors = torch.linalg.eigh(current_key_moment)
        initial_basis = current_key_vectors.flip(-1).contiguous().to(dtype)
        torch.einsum(
            "bhkd,bhdm->bhkm", current_sample.to(dtype), initial_basis
        )
        current_query_float = grouped_queries.float()
        current_query_moment = torch.einsum(
            "bhnd,bhne->bhde", current_query_float, current_query_float
        ) / float(current_query_float.shape[-2])
        _, current_key_factor = _qk_metric_projection_factors(
            current_key_moment,
            current_query_moment,
            projection_dim=128,
            query_shrinkage=args.query_shrinkage,
        )
        current_basis = current_key_factor.to(dtype)
        current_projected_sample = torch.einsum(
            "bhkd,bhdm->bhkm", current_sample.to(dtype), current_basis
        )
        return _hierarchical_key_rate_allocation(
            current_projected_sample,
            bit_budget_per_coordinate=15,
            allow_zero_bits=True,
            include_scale_metadata=True,
        )

    def project_all_keys() -> torch.Tensor:
        for start in range(0, history, args.chunk_tokens):
            stop = min(history, start + args.chunk_tokens)
            projected_full[..., start:stop, :] = torch.einsum(
                "bhkd,bhdm->bhkm", key[..., start:stop, :], basis
            )
        return projected_full

    def pack_all_projected() -> torch.Tensor:
        for start in range(0, history, args.chunk_tokens):
            stop = min(history, start + args.chunk_tokens)
            varbit_cuda.encode_projected_keys_into(
                projected_full[..., start:stop, :].contiguous(),
                packed_index,
                start,
            )
        return packed_index["packed_codes"]

    def project_and_pack_all() -> torch.Tensor:
        for start in range(0, history, args.chunk_tokens):
            stop = min(history, start + args.chunk_tokens)
            projected = torch.einsum(
                "bhkd,bhdm->bhkm", key[..., start:stop, :], basis
            )
            varbit_cuda.encode_projected_keys_into(
                projected.contiguous(), packed_index, start
            )
        return packed_index["packed_codes"]

    # Warm the extension and all cuSOLVER paths before collecting timings.
    project_all_keys()
    pack_all_projected()
    solve_transform()
    torch.cuda.synchronize()

    stages = {
        "sample_materialize_ms": cuda_ms(materialize_sample, args.repeats),
        "key_second_moment_ms": cuda_ms(key_moment, args.repeats),
        "query_second_moment_ms": cuda_ms(query_moment, args.repeats),
        "redundant_key_pca_eigh_ms": cuda_ms(key_pca_eigh, args.repeats),
        "eigh_svd_transform_solve_ms": cuda_ms(solve_transform, args.repeats),
        "sample_key_projection_ms": cuda_ms(project_sample, args.repeats),
        # This function intentionally includes its Python control flow and GPU
        # synchronizations caused by scalar .item() calls.
        "key_mse_bit_allocation_ms": cuda_ms(allocate_bits, args.repeats, warmup=0),
        "all_key_projection_ms": cuda_ms(project_all_keys, args.repeats),
        "all_key_quantize_pack_ms": cuda_ms(pack_all_projected, args.repeats),
        "all_key_project_plus_pack_ms": cuda_ms(
            project_and_pack_all, args.repeats
        ),
        "current_code_calibration_complete_ms": cuda_ms(
            current_code_calibration, args.repeats, warmup=0
        ),
    }
    diagnostic_stage_names = (
        "sample_materialize_ms",
        "key_second_moment_ms",
        "query_second_moment_ms",
        "redundant_key_pca_eigh_ms",
        "eigh_svd_transform_solve_ms",
        "sample_key_projection_ms",
        "key_mse_bit_allocation_ms",
        "all_key_project_plus_pack_ms",
    )
    output = {
        "schema": "qksieve_requestlocal_index_build_stages_v1",
        "hardware": torch.cuda.get_device_name(0),
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "sampled_key_tokens": int(sampled_key.shape[-2]),
        "allocation_by_kv_head": allocation.squeeze(0).cpu().tolist(),
        "stages": stages,
        "diagnostic_serial_stage_sum_ms": float(
            sum(stages[name] for name in diagnostic_stage_names)
        ),
        "measured_calibration_plus_project_pack_ms": float(
            stages["current_code_calibration_complete_ms"]
            + stages["all_key_project_plus_pack_ms"]
        ),
        "notes": [
            "Each stage is measured independently with CUDA events after warmup.",
            "The diagnostic serial sum uses disjoint conceptual stages; the measured calibration total follows current code.",
            "It excludes model prefill and the rank-16 Value-tail index.",
        ],
    }
    text = json.dumps(output, indent=2)
    print(text, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
