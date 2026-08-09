#!/usr/bin/env python3
"""Compare algebraically equivalent QK-balanced factor solvers."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from run_head_top2_targeted_ppl_20260714 import (
    _hierarchical_qmse_rate_allocation,
    _hierarchical_quantize_band,
    _qk_metric_projection_factors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=32768)
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--query_tokens", type=int, default=32)
    parser.add_argument("--query_shrinkage", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def regularize_query_covariance(
    query_covariance: torch.Tensor,
    shrinkage: float,
) -> torch.Tensor:
    head_dim = int(query_covariance.shape[-1])
    isotropic_scale = query_covariance.diagonal(
        dim1=-2,
        dim2=-1,
    ).mean(dim=-1)
    identity = torch.eye(
        head_dim,
        dtype=torch.float32,
        device=query_covariance.device,
    )
    return (
        (1.0 - shrinkage) * query_covariance.float()
        + shrinkage * isotropic_scale[..., None, None] * identity
    )


def symmetric_factors(
    covariance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
    floor = eigenvalues.amax(dim=-1, keepdim=True) * 1.0e-8 + 1.0e-12
    stable = eigenvalues.clamp_min(floor)
    square_root = torch.einsum(
        "...di,...i,...ei->...de",
        eigenvectors,
        stable.sqrt(),
        eigenvectors,
    )
    inverse_square_root = torch.einsum(
        "...di,...i,...ei->...de",
        eigenvectors,
        stable.rsqrt(),
        eigenvectors,
    )
    return square_root, inverse_square_root, eigenvalues


def generalized_symmetric_solver(
    key_covariance: torch.Tensor,
    query_covariance: torch.Tensor,
    shrinkage: float,
    *,
    solve_on_cpu: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_device = key_covariance.device
    key_work = key_covariance.detach().float()
    query_work = regularize_query_covariance(query_covariance, shrinkage)
    if solve_on_cpu:
        key_work = key_work.cpu()
        query_work = query_work.cpu()
    key_root, key_inverse_root, _ = symmetric_factors(key_work)
    product = key_root @ query_work @ key_root
    eigenvalues, eigenvectors = torch.linalg.eigh(product)
    eigenvalues = eigenvalues.flip(-1).clamp_min(1.0e-20)
    eigenvectors = eigenvectors.flip(-1).contiguous()
    fourth_root = eigenvalues.sqrt().sqrt().unsqueeze(-2)
    key_factor = (key_inverse_root @ eigenvectors) * fourth_root
    query_factor = (key_root @ eigenvectors) / fourth_root
    return (
        query_factor.to(output_device).contiguous(),
        key_factor.to(output_device).contiguous(),
    )


def generalized_cholesky_solver(
    key_covariance: torch.Tensor,
    query_covariance: torch.Tensor,
    shrinkage: float,
    *,
    solve_on_cpu: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_device = key_covariance.device
    key_work = key_covariance.detach().float()
    query_work = regularize_query_covariance(query_covariance, shrinkage)
    if solve_on_cpu:
        key_work = key_work.cpu()
        query_work = query_work.cpu()
    dimension = int(key_work.shape[-1])
    trace_scale = key_work.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    jitter = trace_scale[..., None, None] * 1.0e-8 + 1.0e-12
    identity = torch.eye(dimension, dtype=torch.float32, device=key_work.device)
    lower = torch.linalg.cholesky(key_work + jitter * identity)
    product = lower.transpose(-1, -2) @ query_work @ lower
    eigenvalues, eigenvectors = torch.linalg.eigh(product)
    eigenvalues = eigenvalues.flip(-1).clamp_min(1.0e-20)
    eigenvectors = eigenvectors.flip(-1).contiguous()
    fourth_root = eigenvalues.sqrt().sqrt().unsqueeze(-2)
    key_factor = torch.linalg.solve_triangular(
        lower.transpose(-1, -2),
        eigenvectors,
        upper=True,
    ) * fourth_root
    query_factor = (lower @ eigenvectors) / fourth_root
    return (
        query_factor.to(output_device).contiguous(),
        key_factor.to(output_device).contiguous(),
    )


def median_seconds(function: Callable[[], object], repeats: int) -> float:
    for _ in range(2):
        function()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return float(statistics.median(samples))


def reconstruct_projected_keys(
    projected_keys: torch.Tensor,
    allocation: torch.Tensor,
) -> torch.Tensor:
    bands = projected_keys.reshape(*projected_keys.shape[:-1], 8, 16)
    reconstructed = torch.zeros_like(bands)
    for bits in (1, 2, 4, 8):
        candidate = _hierarchical_quantize_band(bands, bits)
        mask = (allocation == bits).unsqueeze(-2).unsqueeze(-1)
        reconstructed = torch.where(mask, candidate, reconstructed)
    return reconstructed.reshape_as(projected_keys)


def transformation_metrics(
    query_factor: torch.Tensor,
    key_factor: torch.Tensor,
    key_covariance: torch.Tensor,
    query_covariance: torch.Tensor,
) -> dict[str, float]:
    dimension = int(key_factor.shape[-1])
    identity = torch.eye(dimension, device=key_factor.device).expand(
        *key_factor.shape[:-2],
        dimension,
        dimension,
    )
    dual_error = (query_factor @ key_factor.transpose(-1, -2) - identity).abs()
    key_metric = key_factor.transpose(-1, -2) @ key_covariance @ key_factor
    query_metric = (
        query_factor.transpose(-1, -2) @ query_covariance @ query_factor
    )
    diagonal = torch.diagonal(key_metric, dim1=-2, dim2=-1)
    off_diagonal = key_metric - torch.diag_embed(diagonal)
    balance_error = (key_metric - query_metric).abs()
    return {
        "dual_identity_max_abs": float(dual_error.max().item()),
        "balanced_covariance_max_abs": float(balance_error.max().item()),
        "key_covariance_offdiag_rms_over_diag_rms": float(
            off_diagonal.square().mean().sqrt().item()
            / diagonal.square().mean().sqrt().clamp_min(1.0e-20).item()
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    key = torch.randn(
        1,
        8,
        args.history_tokens,
        128,
        dtype=torch.float16,
        device="cuda",
    )
    query = torch.randn(
        1,
        8,
        args.query_tokens,
        128,
        dtype=torch.float16,
        device="cuda",
    )
    sampled_key = key[..., :: args.sample_stride, :].float().contiguous()
    key_covariance = torch.einsum(
        "bhkd,bhke->bhde",
        sampled_key,
        sampled_key,
    ) / float(sampled_key.shape[-2])
    query_float = query.float()
    query_covariance = torch.einsum(
        "bhqd,bhqe->bhde",
        query_float,
        query_float,
    ) / float(query_float.shape[-2])
    regularized_query = regularize_query_covariance(
        query_covariance,
        args.query_shrinkage,
    )

    solvers: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]] = {
        "legacy_cpu_eigh_svd": lambda: _qk_metric_projection_factors(
            key_covariance,
            query_covariance,
            128,
            args.query_shrinkage,
        ),
        "symmetric_cpu": lambda: generalized_symmetric_solver(
            key_covariance,
            query_covariance,
            args.query_shrinkage,
            solve_on_cpu=True,
        ),
        "symmetric_cuda": lambda: generalized_symmetric_solver(
            key_covariance,
            query_covariance,
            args.query_shrinkage,
            solve_on_cpu=False,
        ),
        "cholesky_cpu": lambda: generalized_cholesky_solver(
            key_covariance,
            query_covariance,
            args.query_shrinkage,
            solve_on_cpu=True,
        ),
        "cholesky_cuda": lambda: generalized_cholesky_solver(
            key_covariance,
            query_covariance,
            args.query_shrinkage,
            solve_on_cpu=False,
        ),
    }
    factors = {name: solver() for name, solver in solvers.items()}
    timings = {
        name: median_seconds(solver, args.repeats)
        for name, solver in solvers.items()
    }
    exact_scores = torch.einsum(
        "bhqd,bhkd->bhqk",
        query.float(),
        key.float(),
    )
    keep_count = min(
        args.history_tokens,
        1280,
        max(256, int((0.06 * args.history_tokens) + 0.999999)),
    )
    exact_topk = exact_scores.topk(keep_count, dim=-1).indices
    exact_mask = torch.zeros_like(exact_scores, dtype=torch.bool).scatter_(
        -1,
        exact_topk,
        True,
    )
    exact_attention = torch.softmax(exact_scores / (128.0**0.5), dim=-1)

    rows = {}
    legacy_allocation = None
    for name, (query_factor, key_factor) in factors.items():
        projected_sample = torch.einsum(
            "bhkd,bhdm->bhkm",
            sampled_key.to(torch.float16),
            key_factor.to(torch.float16),
        )
        projected_query = torch.einsum(
            "bhqd,bhdm->bhqm",
            query,
            query_factor.to(torch.float16),
        )
        allocation = _hierarchical_qmse_rate_allocation(
            projected_sample,
            projected_query,
            15,
            allow_zero_bits=True,
            include_scale_metadata=True,
            query_covariance_shrinkage="oas",
        )
        reconstructed_key = reconstruct_projected_keys(
            torch.einsum(
                "bhkd,bhdm->bhkm",
                key,
                key_factor.to(torch.float16),
            ),
            allocation,
        )
        proxy_scores = torch.einsum(
            "bhqd,bhkd->bhqk",
            projected_query.float(),
            reconstructed_key.float(),
        )
        proxy_topk = proxy_scores.topk(keep_count, dim=-1).indices
        proxy_mask = torch.zeros_like(exact_mask).scatter_(
            -1,
            proxy_topk,
            True,
        )
        recall = (proxy_mask & exact_mask).sum(dim=-1).float() / keep_count
        mass = (exact_attention * proxy_mask).sum(dim=-1)
        if legacy_allocation is None:
            legacy_allocation = allocation
        rows[name] = {
            "solver_seconds": timings[name],
            "speedup_vs_legacy": timings["legacy_cpu_eigh_svd"] / timings[name],
            **transformation_metrics(
                query_factor,
                key_factor,
                key_covariance,
                regularized_query,
            ),
            "allocation_equal_to_legacy": bool(
                torch.equal(allocation, legacy_allocation)
            ),
            "allocation": allocation.squeeze(0).cpu().tolist(),
            "exact_topk_recall_mean": float(recall.mean().item()),
            "exact_topk_recall_min": float(recall.min().item()),
            "exact_attention_mass_mean": float(mass.mean().item()),
            "exact_attention_mass_min": float(mass.min().item()),
        }

    output = {
        "schema": "qksieve_qk_factor_solver_benchmark_v1",
        "hardware": torch.cuda.get_device_name(0),
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "sampled_key_tokens": int(sampled_key.shape[-2]),
        "keep_count": keep_count,
        "rows": rows,
    }
    rendered = json.dumps(output, indent=2)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
