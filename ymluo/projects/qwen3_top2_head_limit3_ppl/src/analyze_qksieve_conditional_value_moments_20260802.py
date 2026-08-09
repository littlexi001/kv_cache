#!/usr/bin/env python
"""Audit blockwise K-conditioned completion of the omitted Value tail.

The sparse selector already scans a quantized, request-local Key proxy.  This
experiment asks whether the same scan can recover the omitted Value numerator
without a per-token Value sketch.  Every history block stores a small linear
conditional moment model

    V ~= mean(V) + A (X - mean(X)),

where X is a prefix of the QK-balanced proxy coordinate.  During retrieval we
accumulate the proxy tail's weight and weighted X mean per block.  Applying A
once per block estimates the omitted Value numerator.  This is an offline
mechanism audit; it deliberately makes no CUDA latency claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from analyze_automatic_spectral_rate_allocation_20260727 import (
    FULL_KV_BITS,
    GROUP_SIZE,
    ZERO_BIT_LEVELS,
    allocate_bits,
)
from analyze_hierarchical_spectral_quantization_20260727 import query_int8
from analyze_qk_balanced_spectral_rate_20260727 import (
    distortion_table_from_bands,
    qk_balanced_factors,
)
from analyze_qk_progressive_refinement_20260727 import (
    allocation_rate,
    quantized_bands,
    reconstruct,
)


def parse_ints(specification: str) -> tuple[int, ...]:
    values = tuple(sorted({int(x) for x in specification.split(",") if x.strip()}))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_floats(specification: str) -> tuple[float, ...]:
    values = tuple(
        sorted({float(x) for x in specification.split(",") if x.strip()})
    )
    if not values:
        raise ValueError("expected at least one float")
    return values


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p99": float(torch.quantile(tensor, 0.99)),
        "maximum": float(tensor.max()),
    }


def output_metrics(
    output: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    relative_l2 = torch.linalg.vector_norm(output - reference) / (
        torch.linalg.vector_norm(reference).clamp_min(1.0e-12)
    )
    cosine = F.cosine_similarity(output[None], reference[None], dim=-1)[0]
    return {
        "relative_l2": float(relative_l2),
        "cosine": float(cosine),
    }


def symmetric_quantize(
    tensor: torch.Tensor,
    bits: int,
    reduce_dims: tuple[int, ...],
) -> torch.Tensor:
    if bits >= 16:
        return tensor
    if bits < 2:
        raise ValueError("moment quantization requires at least two bits")
    maximum_code = float((1 << (bits - 1)) - 1)
    scale = tensor.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1.0e-12)
    return (
        torch.round(tensor / scale * maximum_code)
        .clamp(-maximum_code, maximum_code)
        * (scale / maximum_code)
    )


def fit_block_models(
    coordinates: torch.Tensor,
    values: torch.Tensor,
    block_size: int,
    ridge: float,
    moment_bits: int,
    linear_group_blocks: int = 1,
    linear_fit_stride: int = 1,
) -> dict[str, torch.Tensor | int | float]:
    if linear_fit_stride <= 0:
        raise ValueError("linear_fit_stride must be positive")
    token_count, coordinate_dim = coordinates.shape
    value_dim = values.shape[-1]
    block_count = math.ceil(token_count / block_size)
    padded_count = block_count * block_size
    padding = padded_count - token_count
    if padding:
        coordinates = F.pad(coordinates, (0, 0, 0, padding))
        values = F.pad(values, (0, 0, 0, padding))
    x = coordinates.reshape(block_count, block_size, coordinate_dim)
    v = values.reshape(block_count, block_size, value_dim)
    counts = torch.full(
        (block_count, 1, 1),
        float(block_size),
        dtype=x.dtype,
        device=x.device,
    )
    if padding:
        counts[-1] = float(block_size - padding)
    valid = torch.arange(block_size, device=x.device)[None, :, None] < counts
    mask = valid.to(x.dtype)
    mean_x = (x * mask).sum(dim=1) / counts[:, 0]
    mean_v = (v * mask).sum(dim=1) / counts[:, 0]
    centered_x = (x - mean_x[:, None]) * mask
    centered_v = (v - mean_v[:, None]) * mask
    if linear_group_blocks <= 0:
        linear_group_blocks = block_count
    linear_group_blocks = min(linear_group_blocks, block_count)
    linear_group_ids = torch.arange(block_count, device=x.device) // linear_group_blocks
    linear_group_count = int(linear_group_ids[-1]) + 1
    fit_x = centered_x[:, ::linear_fit_stride]
    fit_v = centered_v[:, ::linear_fit_stride]
    fit_mask = mask[:, ::linear_fit_stride]
    covariance_per_block = torch.einsum("bti,btj->bij", fit_x, fit_x)
    cross_per_block = torch.einsum("btd,bti->bdi", fit_v, fit_x)
    covariance = torch.zeros(
        linear_group_count,
        coordinate_dim,
        coordinate_dim,
        device=x.device,
        dtype=x.dtype,
    )
    cross_covariance = torch.zeros(
        linear_group_count,
        value_dim,
        coordinate_dim,
        device=x.device,
        dtype=x.dtype,
    )
    covariance.index_add_(0, linear_group_ids, covariance_per_block)
    cross_covariance.index_add_(0, linear_group_ids, cross_per_block)
    fit_counts = fit_mask.sum(dim=1)[:, 0]
    degrees_per_block = (fit_counts - 1.0).clamp_min(1.0)
    normalizer = torch.zeros(
        linear_group_count, device=x.device, dtype=x.dtype
    )
    normalizer.index_add_(0, linear_group_ids, degrees_per_block)
    covariance = covariance / normalizer[:, None, None]
    cross_covariance = cross_covariance / normalizer[:, None, None]
    ridge_scale = covariance.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    ridge_scale = (ridge_scale * ridge).clamp_min(1.0e-8)
    identity = torch.eye(coordinate_dim, device=x.device, dtype=x.dtype)
    regularized = covariance + ridge_scale[:, None, None] * identity
    # Solve A Sigma = Cov(V, X) without forming a batched inverse.
    linear_map = torch.linalg.solve(
        regularized,
        cross_covariance.transpose(-1, -2),
    ).transpose(-1, -2)

    mean_x = symmetric_quantize(mean_x, moment_bits, (1,))
    mean_v = symmetric_quantize(mean_v, moment_bits, (1,))
    linear_map = symmetric_quantize(linear_map, moment_bits, (1, 2))
    mean_bits = moment_bits * (coordinate_dim + value_dim) * block_count
    linear_bits = (
        moment_bits * coordinate_dim * value_dim * linear_group_count
    )
    metadata_bits = 0
    if moment_bits < 16:
        metadata_bits = 2 * 16 * block_count + 16 * linear_group_count
    stored_bits = mean_bits + linear_bits + metadata_bits
    return {
        "mean_x": mean_x,
        "mean_v": mean_v,
        "linear_map": linear_map,
        "linear_group_ids": linear_group_ids,
        "linear_group_blocks": linear_group_blocks,
        "linear_group_count": linear_group_count,
        "linear_fit_stride": linear_fit_stride,
        "block_size": block_size,
        "block_count": block_count,
        "counts": counts[:, 0, 0],
        "moment_bits_per_token": stored_bits / token_count,
    }


def selected_conditioned_residual_mean(
    coordinates: torch.Tensor,
    values: torch.Tensor,
    selected: torch.Tensor,
    model: dict[str, torch.Tensor | int | float],
) -> torch.Tensor:
    """Infer each block's omitted residual mean from selected residuals."""
    mean_x = model["mean_x"]
    mean_v = model["mean_v"]
    linear_map = model["linear_map"]
    linear_group_ids = model["linear_group_ids"]
    counts = model["counts"]
    assert isinstance(mean_x, torch.Tensor)
    assert isinstance(mean_v, torch.Tensor)
    assert isinstance(linear_map, torch.Tensor)
    assert isinstance(linear_group_ids, torch.Tensor)
    assert isinstance(counts, torch.Tensor)
    block_size = int(model["block_size"])
    block_count = int(model["block_count"])
    selected_blocks = selected // block_size
    selected_coordinates = coordinates.index_select(0, selected).float()
    selected_values = values.index_select(0, selected).float()
    token_maps = linear_map.index_select(
        0, linear_group_ids.index_select(0, selected_blocks).long()
    ).float()
    centered = selected_coordinates - mean_x.index_select(
        0, selected_blocks
    ).float()
    predicted = mean_v.index_select(0, selected_blocks).float() + torch.einsum(
        "ndi,ni->nd", token_maps, centered
    )
    selected_residual = selected_values - predicted
    selected_residual_sum = torch.zeros(
        block_count,
        values.shape[1],
        device=values.device,
        dtype=torch.float32,
    )
    selected_residual_sum.index_add_(
        0, selected_blocks, selected_residual
    )
    selected_counts = torch.zeros(
        block_count, device=values.device, dtype=torch.float32
    )
    selected_counts.index_add_(
        0,
        selected_blocks,
        torch.ones_like(selected_blocks, dtype=torch.float32),
    )
    omitted_counts = counts.float() - selected_counts
    return -selected_residual_sum / omitted_counts[:, None].clamp_min(1.0)


def fit_gaussian_tilt_moments(
    score_coordinates: torch.Tensor,
    block_size: int,
    moment_bits: int,
    covariance_mode: str,
) -> dict[str, torch.Tensor | int | float | str]:
    """Fit block moments needed by the Gaussian exponential-tilt closure."""
    if covariance_mode not in ("diag", "full"):
        raise ValueError("covariance mode must be 'diag' or 'full'")
    token_count, coordinate_dim = score_coordinates.shape
    block_count = math.ceil(token_count / block_size)
    padded_count = block_count * block_size
    padding = padded_count - token_count
    if padding:
        score_coordinates = F.pad(score_coordinates, (0, 0, 0, padding))
    blocked = score_coordinates.reshape(block_count, block_size, coordinate_dim)
    counts = torch.full(
        (block_count,),
        float(block_size),
        dtype=blocked.dtype,
        device=blocked.device,
    )
    if padding:
        counts[-1] = float(block_size - padding)
    valid = torch.arange(block_size, device=blocked.device)[None, :] < counts[:, None]
    mask = valid.to(blocked.dtype)[:, :, None]
    mean = (blocked * mask).sum(dim=1) / counts[:, None]
    centered = (blocked - mean[:, None]) * mask
    denominator = (counts - 1.0).clamp_min(1.0)
    if covariance_mode == "diag":
        covariance = centered.square().sum(dim=1) / denominator[:, None]
        covariance = symmetric_quantize(covariance, moment_bits, (1,)).clamp_min(0.0)
        covariance_elements = coordinate_dim
        covariance_key = "variance"
    else:
        covariance = torch.einsum("bti,btj->bij", centered, centered)
        covariance = covariance / denominator[:, None, None]
        diagonal_scale = covariance.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        jitter = diagonal_scale.clamp_min(1.0e-8) * 1.0e-5
        identity = torch.eye(
            coordinate_dim, device=blocked.device, dtype=blocked.dtype
        )
        covariance = covariance + jitter[:, None, None] * identity
        factor = torch.linalg.cholesky(covariance)
        covariance = symmetric_quantize(factor, moment_bits, (1, 2))
        covariance_elements = coordinate_dim * (coordinate_dim + 1) // 2
        covariance_key = "covariance_factor"
    mean = symmetric_quantize(mean, moment_bits, (1,))
    stored_bits = (
        moment_bits
        * (coordinate_dim + covariance_elements)
        * block_count
    )
    if moment_bits < 16:
        stored_bits += 2 * 16 * block_count
    result: dict[str, torch.Tensor | int | float | str] = {
        "mean": mean,
        covariance_key: covariance,
        "covariance_mode": covariance_mode,
        "block_size": block_size,
        "block_count": block_count,
        "counts": counts,
        "coordinate_dim": coordinate_dim,
        "moment_bits_per_token": stored_bits / token_count,
    }
    return result


def gaussian_tilt_tail_statistics(
    calibrated_proxy_scores: torch.Tensor,
    score_direction: torch.Tensor,
    score_intercept: float,
    conditional_coordinates: torch.Tensor,
    selected: torch.Tensor,
    model: dict[str, torch.Tensor | int | float | str],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Approximate omitted exponential moments from block Gaussian closure."""
    mean = model["mean"]
    counts = model["counts"]
    assert isinstance(mean, torch.Tensor)
    assert isinstance(counts, torch.Tensor)
    block_size = int(model["block_size"])
    block_count = int(model["block_count"])
    direction = score_direction.float()
    if direction.numel() != mean.shape[1]:
        raise ValueError("score direction and Gaussian moment dimensions differ")
    covariance_mode = str(model["covariance_mode"])
    if covariance_mode == "diag":
        variance = model["variance"]
        assert isinstance(variance, torch.Tensor)
        covariance_direction = variance.float() * direction[None, :]
        quadratic = (covariance_direction * direction[None, :]).sum(dim=-1)
    else:
        factor = model["covariance_factor"]
        assert isinstance(factor, torch.Tensor)
        factor = factor.float()
        factor_t_direction = torch.einsum("bji,j->bi", factor, direction)
        covariance_direction = torch.einsum(
            "bij,bj->bi", factor, factor_t_direction
        )
        quadratic = factor_t_direction.square().sum(dim=-1)
    log_partition = (
        counts.float().log()
        + float(score_intercept)
        + mean.float() @ direction
        + 0.5 * quadratic
    )
    reference = calibrated_proxy_scores.index_select(0, selected).amin().float()
    all_denominator = torch.exp((log_partition - reference).clamp(-80.0, 80.0))
    conditional_dim = conditional_coordinates.shape[1]
    tilted_mean = mean[:, :conditional_dim].float() + covariance_direction[
        :, :conditional_dim
    ]
    all_weighted_x = all_denominator[:, None] * tilted_mean

    selected_blocks = selected // block_size
    selected_weights = torch.exp(
        (
            calibrated_proxy_scores.index_select(0, selected).float()
            - reference
        ).clamp(-80.0, 80.0)
    )
    selected_denominator = torch.zeros_like(all_denominator)
    selected_denominator.index_add_(0, selected_blocks, selected_weights)
    selected_weighted_x = torch.zeros_like(all_weighted_x)
    selected_weighted_x.index_add_(
        0,
        selected_blocks,
        selected_weights[:, None]
        * conditional_coordinates.index_select(0, selected).float(),
    )
    raw_tail_denominator = all_denominator - selected_denominator
    negative = raw_tail_denominator < 0.0
    tail_denominator = raw_tail_denominator.clamp_min(0.0)
    tail_weighted_x = all_weighted_x - selected_weighted_x
    tail_weighted_x[tail_denominator == 0.0] = 0.0
    deficit = (selected_denominator - all_denominator).clamp_min(0.0).sum()
    diagnostics = {
        "negative_block_fraction": float(negative.float().mean()),
        "selected_mass_deficit_ratio": float(
            deficit / selected_denominator.sum().clamp_min(1.0e-20)
        ),
    }
    return tail_denominator, tail_weighted_x, diagnostics


def gaussian_tilt_tail_statistics_selected_conditioned(
    calibrated_proxy_scores: torch.Tensor,
    score_direction: torch.Tensor,
    score_intercept: float,
    score_coordinates: torch.Tensor,
    conditional_coordinates: torch.Tensor,
    selected: torch.Tensor,
    model: dict[str, torch.Tensor | int | float | str],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Fit the Gaussian closure to each block after removing selected tokens."""
    mean = model["mean"]
    counts = model["counts"]
    assert isinstance(mean, torch.Tensor)
    assert isinstance(counts, torch.Tensor)
    mean = mean.float()
    counts = counts.float()
    block_size = int(model["block_size"])
    block_count = int(model["block_count"])
    coordinate_dim = int(model["coordinate_dim"])
    direction = score_direction.float()
    if score_coordinates.shape[1] != coordinate_dim:
        raise ValueError("score coordinates and Gaussian moments differ")
    covariance_mode = str(model["covariance_mode"])
    selected_blocks = selected // block_size
    selected_coordinates = score_coordinates.index_select(0, selected).float()
    selected_counts = torch.zeros(
        block_count, device=mean.device, dtype=torch.float32
    )
    selected_counts.index_add_(
        0, selected_blocks, torch.ones_like(selected_blocks, dtype=torch.float32)
    )
    tail_counts = (counts - selected_counts).clamp_min(0.0)
    raw_sum = counts[:, None] * mean
    selected_sum = torch.zeros_like(raw_sum)
    selected_sum.index_add_(0, selected_blocks, selected_coordinates)
    tail_sum = raw_sum - selected_sum
    tail_mean = tail_sum / tail_counts[:, None].clamp_min(1.0)

    if covariance_mode == "diag":
        variance = model["variance"]
        assert isinstance(variance, torch.Tensor)
        raw_square_sum = (
            (counts - 1.0).clamp_min(0.0)[:, None] * variance.float()
            + counts[:, None] * mean.square()
        )
        selected_square_sum = torch.zeros_like(raw_square_sum)
        selected_square_sum.index_add_(
            0, selected_blocks, selected_coordinates.square()
        )
        centered_square_sum = (
            raw_square_sum
            - selected_square_sum
            - tail_counts[:, None] * tail_mean.square()
        )
        negative_variance = centered_square_sum < 0.0
        tail_variance = centered_square_sum.clamp_min(0.0) / (
            tail_counts - 1.0
        ).clamp_min(1.0)[:, None]
        covariance_direction = tail_variance * direction[None, :]
        quadratic = (covariance_direction * direction[None, :]).sum(dim=-1)
    else:
        factor = model["covariance_factor"]
        assert isinstance(factor, torch.Tensor)
        covariance = factor.float() @ factor.float().transpose(-1, -2)
        raw_outer_sum = (
            (counts - 1.0).clamp_min(0.0)[:, None, None] * covariance
            + counts[:, None, None]
            * torch.einsum("bi,bj->bij", mean, mean)
        )
        selected_outer_sum = torch.zeros_like(raw_outer_sum)
        selected_outer_sum.index_add_(
            0,
            selected_blocks,
            torch.einsum(
                "bi,bj->bij", selected_coordinates, selected_coordinates
            ),
        )
        centered_outer_sum = (
            raw_outer_sum
            - selected_outer_sum
            - tail_counts[:, None, None]
            * torch.einsum("bi,bj->bij", tail_mean, tail_mean)
        )
        tail_covariance = centered_outer_sum / (
            tail_counts - 1.0
        ).clamp_min(1.0)[:, None, None]
        tail_covariance = 0.5 * (
            tail_covariance + tail_covariance.transpose(-1, -2)
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(tail_covariance)
        negative_variance = eigenvalues < 0.0
        tail_covariance = (
            eigenvectors
            * eigenvalues.clamp_min(0.0)[:, None, :]
        ) @ eigenvectors.transpose(-1, -2)
        covariance_direction = tail_covariance @ direction
        quadratic = (covariance_direction * direction[None, :]).sum(dim=-1)

    log_partition = (
        tail_counts.clamp_min(1.0).log()
        + float(score_intercept)
        + tail_mean @ direction
        + 0.5 * quadratic
    )
    reference = calibrated_proxy_scores.index_select(0, selected).amin().float()
    tail_denominator = torch.exp(
        (log_partition - reference).clamp(-80.0, 80.0)
    ).masked_fill(tail_counts == 0.0, 0.0)
    conditional_dim = conditional_coordinates.shape[1]
    tilted_mean = tail_mean[:, :conditional_dim] + covariance_direction[
        :, :conditional_dim
    ]
    tail_weighted_x = tail_denominator[:, None] * tilted_mean
    diagnostics = {
        "negative_block_fraction": 0.0,
        "selected_mass_deficit_ratio": 0.0,
        "negative_variance_fraction": float(
            negative_variance.float().mean()
        ),
    }
    return tail_denominator, tail_weighted_x, diagnostics


def gaussian_tilt_tail_statistics_hybrid(
    calibrated_proxy_scores: torch.Tensor,
    score_direction: torch.Tensor,
    score_intercept: float,
    score_coordinates: torch.Tensor,
    conditional_coordinates: torch.Tensor,
    selected: torch.Tensor,
    model: dict[str, torch.Tensor | int | float | str],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Repair only blocks whose all-token closure cannot cover selected mass."""
    denominator, weighted_x, diagnostics = gaussian_tilt_tail_statistics(
        calibrated_proxy_scores,
        score_direction,
        score_intercept,
        conditional_coordinates,
        selected,
        model,
    )
    if diagnostics["negative_block_fraction"] == 0.0:
        diagnostics["repaired_block_fraction"] = 0.0
        return denominator, weighted_x, diagnostics
    conditioned_denominator, conditioned_x, conditioned_diagnostics = (
        gaussian_tilt_tail_statistics_selected_conditioned(
            calibrated_proxy_scores,
            score_direction,
            score_intercept,
            score_coordinates,
            conditional_coordinates,
            selected,
            model,
        )
    )
    repair = (denominator == 0.0) & (conditioned_denominator > 0.0)
    denominator = torch.where(repair, conditioned_denominator, denominator)
    weighted_x = torch.where(repair[:, None], conditioned_x, weighted_x)
    diagnostics["repaired_block_fraction"] = float(repair.float().mean())
    diagnostics["negative_variance_fraction"] = conditioned_diagnostics[
        "negative_variance_fraction"
    ]
    return denominator, weighted_x, diagnostics


def conditional_block_numerator(
    denominator: torch.Tensor,
    weighted_x: torch.Tensor,
    model: dict[str, torch.Tensor | int | float],
) -> torch.Tensor:
    """Return the conditional-Value numerator separately for every block."""
    mean_x = model["mean_x"]
    mean_v = model["mean_v"]
    linear_map = model["linear_map"]
    assert isinstance(mean_x, torch.Tensor)
    assert isinstance(mean_v, torch.Tensor)
    assert isinstance(linear_map, torch.Tensor)
    linear_group_ids = model.get("linear_group_ids")
    if isinstance(linear_group_ids, torch.Tensor):
        linear_map = linear_map.index_select(0, linear_group_ids.long())
    weighted_mean_x = weighted_x / denominator[:, None].clamp_min(1.0e-20)
    predicted_mean_v = mean_v + torch.einsum(
        "bdi,bi->bd",
        linear_map,
        weighted_mean_x - mean_x,
    )
    return denominator[:, None] * predicted_mean_v


def gaussian_tilt_block_control_values(
    score_direction: torch.Tensor,
    score_intercept: float,
    reference: torch.Tensor | float,
    conditional_model: dict[str, torch.Tensor | int | float],
    gaussian_model: dict[str, torch.Tensor | int | float | str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one denominator/numerator control value per token in each block.

    These values need not be an exact probabilistic model.  A subsequent
    Horvitz--Thompson residual correction remains unbiased for the unnormalized
    softmax totals; a better block model only lowers its sampling variance.
    """
    mean = gaussian_model["mean"]
    counts = gaussian_model["counts"]
    assert isinstance(mean, torch.Tensor)
    assert isinstance(counts, torch.Tensor)
    direction = score_direction.float()
    mean = mean.float()
    counts = counts.float()
    if direction.numel() != mean.shape[1]:
        raise ValueError("score direction and Gaussian moment dimensions differ")
    covariance_mode = str(gaussian_model["covariance_mode"])
    if covariance_mode == "diag":
        variance = gaussian_model["variance"]
        assert isinstance(variance, torch.Tensor)
        covariance_direction = variance.float() * direction[None, :]
        quadratic = (covariance_direction * direction[None, :]).sum(dim=-1)
    else:
        factor = gaussian_model["covariance_factor"]
        assert isinstance(factor, torch.Tensor)
        factor = factor.float()
        factor_t_direction = torch.einsum("bji,j->bi", factor, direction)
        covariance_direction = torch.einsum(
            "bij,bj->bi", factor, factor_t_direction
        )
        quadratic = factor_t_direction.square().sum(dim=-1)
    reference_tensor = torch.as_tensor(
        reference, device=mean.device, dtype=torch.float32
    )
    log_per_token_denominator = (
        float(score_intercept)
        + mean @ direction
        + 0.5 * quadratic
        - reference_tensor
    )
    per_token_denominator = torch.exp(
        log_per_token_denominator.clamp(-80.0, 80.0)
    )
    conditional_dim = int(
        torch.as_tensor(conditional_model["mean_x"]).shape[-1]
    )
    tilted_mean = mean[:, :conditional_dim] + covariance_direction[
        :, :conditional_dim
    ]
    block_denominator = counts * per_token_denominator
    block_weighted_x = block_denominator[:, None] * tilted_mean
    block_numerator = conditional_block_numerator(
        block_denominator, block_weighted_x, conditional_model
    )
    return per_token_denominator, block_numerator / counts[:, None]


def stratified_uniform_sample_indices(
    token_count: int,
    block_size: int,
    samples_per_block: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Draw a fixed-size uniform reservoir independently in every block."""
    if token_count <= 0 or block_size <= 0 or samples_per_block <= 0:
        raise ValueError("token, block, and sample counts must be positive")
    sampled: list[torch.Tensor] = []
    for start in range(0, token_count, block_size):
        count = min(block_size, token_count - start)
        take = min(samples_per_block, count)
        sampled.append(
            torch.randperm(count, generator=generator)[:take] + start
        )
    return torch.cat(sampled).to(device=device, dtype=torch.long)


def control_variate_tail_statistics(
    exact_scores: torch.Tensor,
    values: torch.Tensor,
    selected: torch.Tensor,
    sample_indices: torch.Tensor,
    block_size: int,
    base_denominator_per_token: torch.Tensor,
    base_numerator_per_token: torch.Tensor,
    reference: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Estimate omitted softmax totals with a stratified control variate.

    The reservoir is sampled from each complete block before observing the
    query.  Conditioning on any query-dependent selected set therefore keeps
    the residual correction unbiased, including when sampled and selected
    tokens overlap.
    """
    token_count = exact_scores.numel()
    if values.shape[0] != token_count:
        raise ValueError("score and Value token dimensions differ")
    if selected.ndim != 1 or sample_indices.ndim != 1:
        raise ValueError("selected and sample indices must be one-dimensional")
    block_count = math.ceil(token_count / block_size)
    if base_denominator_per_token.shape != (block_count,):
        raise ValueError("base denominator must contain one value per block")
    if base_numerator_per_token.shape != (block_count, values.shape[-1]):
        raise ValueError("base numerator must contain one vector per block")
    device = exact_scores.device
    selected = selected.to(device=device, dtype=torch.long)
    sample_indices = sample_indices.to(device=device, dtype=torch.long)
    block_starts = torch.arange(
        block_count, device=device, dtype=torch.long
    ) * block_size
    block_counts = (token_count - block_starts).clamp(min=0, max=block_size)
    selected_blocks = selected // block_size
    selected_counts = torch.zeros(block_count, device=device, dtype=torch.float32)
    selected_counts.index_add_(
        0, selected_blocks, torch.ones_like(selected_blocks, dtype=torch.float32)
    )
    tail_counts = block_counts.float() - selected_counts
    block_tail_denominator = (
        tail_counts * base_denominator_per_token.float()
    )
    block_tail_numerator = (
        tail_counts[:, None] * base_numerator_per_token.float()
    )

    sample_blocks = sample_indices // block_size
    sample_counts = torch.zeros(block_count, device=device, dtype=torch.float32)
    sample_counts.index_add_(
        0, sample_blocks, torch.ones_like(sample_blocks, dtype=torch.float32)
    )
    inclusion = sample_counts / block_counts.float().clamp_min(1.0)
    selected_mask = torch.zeros(token_count, device=device, dtype=torch.bool)
    selected_mask[selected] = True
    tail_sample = sample_indices[~selected_mask.index_select(0, sample_indices)]
    tail_sample_blocks = tail_sample // block_size
    inverse_inclusion = inclusion.index_select(
        0, tail_sample_blocks
    ).clamp_min(1.0e-12).reciprocal()
    reference_tensor = torch.as_tensor(
        reference, device=device, dtype=torch.float32
    )
    sample_weights = torch.exp(
        (exact_scores.index_select(0, tail_sample).float() - reference_tensor)
        .clamp(-80.0, 80.0)
    )
    denominator_residual = sample_weights - base_denominator_per_token.index_select(
        0, tail_sample_blocks
    ).float()
    numerator_residual = (
        sample_weights[:, None] * values.index_select(0, tail_sample).float()
        - base_numerator_per_token.index_select(0, tail_sample_blocks).float()
    )
    block_tail_denominator.index_add_(
        0, tail_sample_blocks, inverse_inclusion * denominator_residual
    )
    block_tail_numerator.index_add_(
        0, tail_sample_blocks, inverse_inclusion[:, None] * numerator_residual
    )
    diagnostics = {
        "sample_tokens": float(sample_indices.numel()),
        "sample_selected_overlap": float(sample_indices.numel() - tail_sample.numel()),
        "negative_tail_block_fraction": float(
            (block_tail_denominator < 0.0).float().mean()
        ),
        "estimated_tail_denominator": float(block_tail_denominator.sum()),
    }
    return (
        block_tail_denominator.sum(),
        block_tail_numerator.sum(dim=0),
        diagnostics,
    )


def combine_selected_and_tail(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    values: torch.Tensor,
    selected: torch.Tensor,
    tail_numerator: torch.Tensor,
    tail_denominator: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    threshold = proxy_scores[selected].amin()
    selected_scores = exact_scores[selected]
    maximum = torch.maximum(selected_scores.amax(), threshold)
    selected_weights = torch.exp((selected_scores - maximum).clamp_min(-80.0))
    tail_factor = torch.exp((threshold - maximum).clamp(min=-80.0, max=80.0))
    scaled_tail_denominator = float(alpha) * tail_factor * tail_denominator
    numerator = selected_weights @ values[selected]
    numerator = numerator + float(alpha) * tail_factor * tail_numerator
    denominator = selected_weights.sum() + scaled_tail_denominator
    return numerator / denominator.clamp_min(1.0e-20)


def tail_statistics(
    proxy_scores: torch.Tensor,
    coordinates: torch.Tensor,
    values: torch.Tensor,
    selected: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_count = proxy_scores.numel()
    threshold = proxy_scores[selected].amin()
    tail_mask = torch.ones(token_count, dtype=torch.bool, device=proxy_scores.device)
    tail_mask[selected] = False
    weights = torch.exp((proxy_scores - threshold).clamp(min=-80.0, max=0.0))
    weights = weights * tail_mask.to(weights.dtype)
    block_ids = torch.arange(token_count, device=proxy_scores.device) // block_size
    block_count = math.ceil(token_count / block_size)
    denominator = torch.zeros(block_count, device=weights.device, dtype=torch.float32)
    denominator.index_add_(0, block_ids, weights.float())
    weighted_x = torch.zeros(
        block_count,
        coordinates.shape[-1],
        device=weights.device,
        dtype=torch.float32,
    )
    weighted_x.index_add_(0, block_ids, weights[:, None] * coordinates.float())
    empirical_numerator = weights.float() @ values.float()
    return denominator, weighted_x, empirical_numerator


def conditional_tail_numerator(
    denominator: torch.Tensor,
    weighted_x: torch.Tensor,
    model: dict[str, torch.Tensor | int | float],
) -> torch.Tensor:
    block_numerator = conditional_block_numerator(
        denominator, weighted_x, model
    )
    return block_numerator[denominator > 0].sum(dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--max_heldout_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--rate_budget", type=int, default=15)
    parser.add_argument("--fractions", default="0.02,0.04")
    parser.add_argument("--coordinate_dims", default="4,8,16,32")
    parser.add_argument("--block_sizes", default="64,128,256,512,1024")
    parser.add_argument("--moment_bits", default="4,8,16")
    parser.add_argument("--alphas", default="0.5,1.0")
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--layers", default="")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    traces = tuple(Path(x) for x in args.traces.split(",") if x.strip())
    fractions = parse_floats(args.fractions)
    coordinate_dims = parse_ints(args.coordinate_dims)
    block_sizes = parse_ints(args.block_sizes)
    moment_bits_values = parse_ints(args.moment_bits)
    alphas = parse_floats(args.alphas)
    requested_layers = set(parse_ints(args.layers)) if args.layers.strip() else None
    if any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("fractions must lie in (0, 1)")
    if any(value not in (4, 8, 16) for value in moment_bits_values):
        raise ValueError("moment bits must be 4, 8, or 16")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []

    for trace_path in traces:
        payload = torch.load(trace_path, map_location="cpu", weights_only=False)
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload["records"]:
            layer = int(record["layer"])
            if requested_layers is None or layer in requested_layers:
                by_layer[layer].append(record)
        for layer, records in sorted(by_layer.items()):
            records.sort(key=lambda item: int(item.get("step", 0)))
            state = next(
                (
                    record
                    for record in records
                    if isinstance(record.get("key"), torch.Tensor)
                    and isinstance(record.get("value"), torch.Tensor)
                ),
                None,
            )
            if state is None:
                raise ValueError(f"layer {layer} has no materialized K/V state")
            key = state["key"].to(device).float()[0]
            value = state["value"].to(device).float()[0]
            calibration_records = records[: min(args.calibration_steps, len(records))]
            heldout_records = records[len(calibration_records) :]
            if not heldout_records:
                heldout_records = calibration_records[-1:]
            if args.max_heldout_steps > 0:
                heldout_records = heldout_records[: args.max_heldout_steps]
            calibration = torch.stack(
                [
                    record["query"].to(device).float()[0, :, 0, :]
                    for record in calibration_records
                ],
                dim=0,
            )
            scaling = float(state["scaling"])
            kv_heads, token_count, head_dim = key.shape
            query_heads = calibration.shape[1]
            if query_heads % kv_heads:
                raise ValueError("query heads must be divisible by KV heads")
            groups = query_heads // kv_heads

            for kv_head in range(kv_heads):
                head_key = key[kv_head]
                head_value = value[kv_head]
                head_calibration = calibration[
                    :, kv_head * groups : (kv_head + 1) * groups
                ].reshape(-1, head_dim)
                query_factor, key_factor, _ = qk_balanced_factors(
                    head_key[:: args.sample_stride],
                    head_calibration,
                    args.query_shrinkage,
                )
                raw_coordinates = head_key @ key_factor
                projected_calibration = head_calibration @ query_factor
                bands = quantized_bands(raw_coordinates, projected_calibration)
                allocation = allocate_bits(
                    distortion_table_from_bands(
                        raw_coordinates,
                        projected_calibration,
                        bands,
                    ),
                    args.rate_budget,
                    ZERO_BIT_LEVELS,
                    include_scale_metadata=True,
                )
                proxy_coordinates = reconstruct(bands, allocation).float()
                active_dimensions = torch.tensor(
                    [
                        dimension
                        for band, bits in enumerate(allocation)
                        if bits > 0
                        for dimension in range(
                            band * GROUP_SIZE,
                            (band + 1) * GROUP_SIZE,
                        )
                    ],
                    device=device,
                    dtype=torch.long,
                )
                if not active_dimensions.numel():
                    raise RuntimeError("QKSieve allocation contains no active dimension")
                available_dims = min(
                    max(coordinate_dims), int(active_dimensions.numel())
                )
                selected_dimensions = active_dimensions[:available_dims]
                conditional_coordinates = proxy_coordinates.index_select(
                    1, selected_dimensions
                )
                fitted: dict[tuple[int, int, int], dict[str, Any]] = {}
                for coordinate_dim in coordinate_dims:
                    actual_dim = min(coordinate_dim, available_dims)
                    x = conditional_coordinates[:, :actual_dim]
                    for block_size in block_sizes:
                        for moment_bits in moment_bits_values:
                            fitted[(coordinate_dim, block_size, moment_bits)] = (
                                fit_block_models(
                                    x,
                                    head_value,
                                    block_size,
                                    args.ridge,
                                    moment_bits,
                                )
                            )

                for record in heldout_records:
                    query = record["query"].to(device).float()[0, :, 0, :]
                    step = int(record.get("step", 0))
                    for group in range(groups):
                        query_head = kv_head * groups + group
                        projected_query = query[query_head] @ query_factor
                        proxy_query = query_int8(projected_query).float()
                        exact_scores = head_key @ query[query_head] * scaling
                        proxy_scores = proxy_coordinates @ proxy_query * scaling
                        full_weights = torch.softmax(exact_scores, dim=0)
                        full_output = full_weights @ head_value

                        for fraction in fractions:
                            keep = min(
                                token_count,
                                max(1, math.ceil(fraction * token_count)),
                            )
                            selected = torch.topk(
                                proxy_scores, k=keep, sorted=False
                            ).indices
                            sparse = torch.softmax(
                                exact_scores[selected], dim=0
                            ) @ head_value[selected]
                            selected_mass = float(full_weights[selected].sum())
                            base = {
                                "trace": trace_path.stem,
                                "layer": layer,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "step": step,
                                "fraction": fraction,
                                "selected_tokens": keep,
                                "allocation": "-".join(map(str, allocation)),
                                "key_index_bits_per_token": (
                                    GROUP_SIZE * allocation_rate(allocation)
                                ),
                                "exact_selected_mass": selected_mass,
                            }
                            rows.append(
                                {
                                    **base,
                                    "method": "proxy_sparse",
                                    "coordinate_dim": 0,
                                    "block_size": 0,
                                    "moment_bits": 0,
                                    "alpha": 0.0,
                                    "moment_bits_per_token": 0.0,
                                    "total_aux_ratio_of_full_kv": (
                                        GROUP_SIZE
                                        * allocation_rate(allocation)
                                        / FULL_KV_BITS
                                    ),
                                    **output_metrics(sparse, full_output),
                                }
                            )

                            largest_x = conditional_coordinates[:, :available_dims]
                            tail_cache: dict[
                                tuple[int, int],
                                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                            ] = {}
                            for coordinate_dim in coordinate_dims:
                                actual_dim = min(coordinate_dim, available_dims)
                                x = largest_x[:, :actual_dim]
                                for block_size in block_sizes:
                                    tail_cache[(coordinate_dim, block_size)] = (
                                        tail_statistics(
                                            proxy_scores,
                                            x,
                                            head_value,
                                            selected,
                                            block_size,
                                        )
                                    )

                            # This oracle scans tail Values and is only an upper bound.
                            denominator, _, empirical_numerator = tail_cache[
                                (coordinate_dims[0], block_sizes[0])
                            ]
                            for alpha in alphas:
                                empirical = combine_selected_and_tail(
                                    exact_scores,
                                    proxy_scores,
                                    head_value,
                                    selected,
                                    empirical_numerator,
                                    denominator.sum(),
                                    alpha,
                                )
                                rows.append(
                                    {
                                        **base,
                                        "method": "empirical_proxy_tail_oracle",
                                        "coordinate_dim": 0,
                                        "block_size": 0,
                                        "moment_bits": 0,
                                        "alpha": alpha,
                                        "moment_bits_per_token": 0.0,
                                        "total_aux_ratio_of_full_kv": (
                                            GROUP_SIZE
                                            * allocation_rate(allocation)
                                            / FULL_KV_BITS
                                        ),
                                        **output_metrics(empirical, full_output),
                                    }
                                )

                            for coordinate_dim in coordinate_dims:
                                for block_size in block_sizes:
                                    denominator, weighted_x, _ = tail_cache[
                                        (coordinate_dim, block_size)
                                    ]
                                    for moment_bits in moment_bits_values:
                                        model = fitted[
                                            (coordinate_dim, block_size, moment_bits)
                                        ]
                                        tail_numerator = conditional_tail_numerator(
                                            denominator,
                                            weighted_x,
                                            model,
                                        )
                                        moment_rate = float(
                                            model["moment_bits_per_token"]
                                        )
                                        for alpha in alphas:
                                            output = combine_selected_and_tail(
                                                exact_scores,
                                                proxy_scores,
                                                head_value,
                                                selected,
                                                tail_numerator,
                                                denominator.sum(),
                                                alpha,
                                            )
                                            rows.append(
                                                {
                                                    **base,
                                                    "method": "conditional_block_moment",
                                                    "coordinate_dim": min(
                                                        coordinate_dim,
                                                        available_dims,
                                                    ),
                                                    "block_size": block_size,
                                                    "moment_bits": moment_bits,
                                                    "alpha": alpha,
                                                    "moment_bits_per_token": moment_rate,
                                                    "total_aux_ratio_of_full_kv": (
                                                        GROUP_SIZE
                                                        * allocation_rate(allocation)
                                                        + moment_rate
                                                    )
                                                    / FULL_KV_BITS,
                                                    **output_metrics(
                                                        output, full_output
                                                    ),
                                                }
                                            )
            print(
                json.dumps(
                    {
                        "trace": trace_path.stem,
                        "layer": layer,
                        "rows": len(rows),
                    }
                ),
                flush=True,
            )
            del key, value
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("no rows were produced")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (args.output_dir / "per_query.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    group_fields = (
        "method",
        "fraction",
        "coordinate_dim",
        "block_size",
        "moment_bits",
        "alpha",
    )
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    summary: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        result = dict(zip(group_fields, key))
        result["cases"] = len(items)
        for metric in (
            "relative_l2",
            "cosine",
            "exact_selected_mass",
            "moment_bits_per_token",
            "total_aux_ratio_of_full_kv",
        ):
            for statistic, value in summarize(
                float(item[metric]) for item in items
            ).items():
                result[f"{metric}_{statistic}"] = value
        summary.append(result)
    practical = [
        row
        for row in summary
        if row["method"] == "conditional_block_moment"
        and row["moment_bits"] <= 8
    ]
    best_by_fraction: dict[str, dict[str, Any]] = {}
    for fraction in fractions:
        current = [row for row in practical if row["fraction"] == fraction]
        if current:
            best_by_fraction[f"{fraction:g}"] = min(
                current,
                key=lambda row: (
                    row["relative_l2_mean"],
                    row["total_aux_ratio_of_full_kv_mean"],
                ),
            )
    report = {
        "schema": "qksieve_conditional_value_moments_v1",
        "traces": [str(path) for path in traces],
        "contract": {
            "calibration_steps": args.calibration_steps,
            "max_heldout_steps": args.max_heldout_steps,
            "rate_budget": args.rate_budget,
            "ridge": args.ridge,
            "full_fallback": False,
            "router": False,
            "quality_boundary": (
                "Offline real-QKV held-out mechanism audit. Tail weights and "
                "weighted low-dimensional K moments are available from the "
                "Key scan; empirical_proxy_tail_oracle additionally scans "
                "tail Values and is not deployable."
            ),
        },
        "best_practical_by_fraction": best_by_fraction,
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
