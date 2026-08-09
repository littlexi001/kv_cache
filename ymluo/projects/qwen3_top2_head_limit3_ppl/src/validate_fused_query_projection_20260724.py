#!/usr/bin/env python3
"""Validate and benchmark fused Key-PCA query projection/quantization."""

from __future__ import annotations

import json
import time

import torch

import qabs_cuda_kernels as kernels


def time_cuda(callable_, warmup: int = 20, repeats: int = 1000) -> float:
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


def main() -> None:
    torch.manual_seed(20260724)
    device = torch.device("cuda")
    query = torch.randn(1, 8, 4, 128, device=device, dtype=torch.float16)
    basis = torch.linalg.qr(
        torch.randn(1, 8, 128, 48, device=device, dtype=torch.float32)
    ).Q.to(torch.float16)

    def reference():
        projected = torch.einsum("bhgd,bhdm->bhgm", query, basis)
        scale = (
            projected.float()
            .abs()
            .amax(dim=-1, keepdim=True)
            .clamp_min(1.0e-8)
            / 127.0
        )
        codes = (
            torch.round(projected.float() / scale)
            .clamp(-127, 127)
            .to(torch.int8)
        )
        return projected, codes, scale

    def fused():
        return kernels.pca_project_query_int8(query, basis)

    reference_outputs = reference()
    fused_outputs = fused()
    torch.cuda.synchronize()
    reference_ms = time_cuda(reference)
    fused_ms = time_cuda(fused)
    result = {
        "projected_max_abs_error": float(
            (
                reference_outputs[0].float()
                - fused_outputs[0].float()
            ).abs().max().item()
        ),
        "projected_mean_abs_error": float(
            (
                reference_outputs[0].float()
                - fused_outputs[0].float()
            ).abs().mean().item()
        ),
        "code_exact_fraction": float(
            (reference_outputs[1] == fused_outputs[1]).float().mean().item()
        ),
        "code_max_abs_difference": int(
            (
                reference_outputs[1].to(torch.int16)
                - fused_outputs[1].to(torch.int16)
            ).abs().max().item()
        ),
        "scale_max_abs_error": float(
            (reference_outputs[2] - fused_outputs[2]).abs().max().item()
        ),
        "reference_ms": reference_ms,
        "fused_ms": fused_ms,
        "speedup": reference_ms / fused_ms,
        "elapsed_seconds": time.time(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
