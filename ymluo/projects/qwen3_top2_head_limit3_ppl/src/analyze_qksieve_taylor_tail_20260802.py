#!/usr/bin/env python
"""Bounded low-order block moments for omitted-attention completion."""

from __future__ import annotations

import math

import torch

from analyze_qksieve_conditional_value_moments_20260802 import symmetric_quantize


def unsigned_quantize(
    tensor: torch.Tensor,
    bits: int,
    reduce_dims: tuple[int, ...],
) -> torch.Tensor:
    if bits >= 16:
        return tensor
    if bits < 1:
        raise ValueError("unsigned moment quantization needs at least one bit")
    maximum_code = float((1 << bits) - 1)
    scale = tensor.amax(dim=reduce_dims, keepdim=True).clamp_min(1.0e-12)
    return torch.round(tensor / scale * maximum_code) * (scale / maximum_code)


def fit_taylor_block_tail(
    score_coordinates: torch.Tensor,
    full_keys: torch.Tensor,
    values: torch.Tensor,
    block_size: int,
    key_mean_bits: int = 8,
    value_mean_bits: int = 4,
    variance_bits: int = 4,
    cross_bits: int = 4,
    cross_key_dim: int = 16,
    cross_value_dim: int = 8,
) -> dict[str, torch.Tensor | int | float]:
    """Fit zero-, first-, and second-order block summaries."""
    if not (score_coordinates.shape[0] == full_keys.shape[0] == values.shape[0]):
        raise ValueError("coordinates, Keys, and Values must share token count")
    if block_size <= 0:
        raise ValueError("block size must be positive")
    token_count, score_dim = score_coordinates.shape
    key_dim = full_keys.shape[-1]
    value_dim = values.shape[-1]
    cross_key_dim = min(cross_key_dim, score_dim)
    cross_value_dim = min(cross_value_dim, value_dim)
    block_count = math.ceil(token_count / block_size)

    gram = values.float().T @ values.float()
    _, eigenvectors = torch.linalg.eigh(gram)
    value_basis = eigenvectors[:, -cross_value_dim:].flip(dims=(1,))
    projected_values = values.float() @ value_basis

    counts = torch.zeros(block_count, dtype=torch.float32, device=values.device)
    mean_key = torch.zeros(
        block_count, key_dim, dtype=torch.float32, device=values.device
    )
    mean_value = torch.zeros(
        block_count, value_dim, dtype=torch.float32, device=values.device
    )
    diagonal_variance = torch.zeros(
        block_count, score_dim, dtype=torch.float32, device=values.device
    )
    cross_moment = torch.zeros(
        block_count,
        cross_key_dim,
        cross_value_dim,
        dtype=torch.float32,
        device=values.device,
    )
    for block in range(block_count):
        start = block * block_size
        stop = min(token_count, start + block_size)
        block_coordinates = score_coordinates[start:stop].float()
        block_keys = full_keys[start:stop].float()
        block_values = values[start:stop].float()
        block_projected_values = projected_values[start:stop]
        count = stop - start
        counts[block] = float(count)
        mean_key[block] = block_keys.mean(dim=0)
        mean_value[block] = block_values.mean(dim=0)
        centered_coordinates = block_coordinates - block_coordinates.mean(dim=0)
        centered_projected_values = (
            block_projected_values - block_projected_values.mean(dim=0)
        )
        diagonal_variance[block] = centered_coordinates.square().mean(dim=0)
        cross_moment[block] = (
            centered_coordinates[:, :cross_key_dim].T
            @ centered_projected_values
            / float(count)
        )

    mean_key = symmetric_quantize(mean_key, key_mean_bits, (1,))
    mean_value = symmetric_quantize(mean_value, value_mean_bits, (1,))
    diagonal_variance = unsigned_quantize(diagonal_variance, variance_bits, (1,))
    cross_moment = symmetric_quantize(cross_moment, cross_bits, (1, 2))

    table_bits = block_count * (
        key_mean_bits * key_dim
        + value_mean_bits * value_dim
        + variance_bits * score_dim
        + cross_bits * cross_key_dim * cross_value_dim
    )
    scale_count = sum(
        int(bits < 16)
        for bits in (key_mean_bits, value_mean_bits, variance_bits, cross_bits)
    )
    scale_bits = block_count * scale_count * 16
    count_bits = block_count * math.ceil(math.log2(block_size + 1))
    basis_bits = value_dim * cross_value_dim * 16
    return {
        "counts": counts,
        "mean_key": mean_key,
        "mean_value": mean_value,
        "diagonal_variance": diagonal_variance,
        "cross_moment": cross_moment,
        "value_basis": value_basis,
        "block_size": block_size,
        "block_count": block_count,
        "cross_key_dim": cross_key_dim,
        "cross_value_dim": cross_value_dim,
        "bits_per_token": (table_bits + scale_bits + count_bits + basis_bits)
        / token_count,
    }


def taylor_block_tail_statistics(
    full_score_direction: torch.Tensor,
    score_direction: torch.Tensor,
    selected: torch.Tensor,
    reference: torch.Tensor | float,
    model: dict[str, torch.Tensor | int | float],
    use_variance: bool,
    use_cross: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Estimate omitted softmax totals with a finite Taylor polynomial."""
    counts = model["counts"]
    mean_key = model["mean_key"]
    mean_value = model["mean_value"]
    diagonal_variance = model["diagonal_variance"]
    cross_moment = model["cross_moment"]
    value_basis = model["value_basis"]
    assert isinstance(counts, torch.Tensor)
    assert isinstance(mean_key, torch.Tensor)
    assert isinstance(mean_value, torch.Tensor)
    assert isinstance(diagonal_variance, torch.Tensor)
    assert isinstance(cross_moment, torch.Tensor)
    assert isinstance(value_basis, torch.Tensor)

    block_size = int(model["block_size"])
    remaining_counts = counts.float().clone()
    selected_blocks = selected.long() // block_size
    remaining_counts.index_add_(
        0,
        selected_blocks,
        -torch.ones_like(selected_blocks, dtype=torch.float32),
    )
    remaining_counts.clamp_min_(0.0)
    reference_tensor = torch.as_tensor(
        reference, dtype=torch.float32, device=mean_key.device
    )
    mean_scores = mean_key.float() @ full_score_direction.float()
    base_weights = remaining_counts * torch.exp(
        (mean_scores - reference_tensor).clamp(-80.0, 40.0)
    )

    if use_variance:
        score_variance = diagonal_variance.float() @ score_direction.float().square()
        mass_factor = 1.0 + 0.5 * score_variance.clamp_min(0.0)
    else:
        score_variance = torch.zeros_like(base_weights)
        mass_factor = torch.ones_like(base_weights)
    block_values = mass_factor[:, None] * mean_value.float()

    if use_cross:
        cross_key_dim = int(model["cross_key_dim"])
        projected_correction = torch.einsum(
            "r,brv->bv",
            score_direction[:cross_key_dim].float(),
            cross_moment.float(),
        )
        value_correction = projected_correction @ value_basis.float().T
        block_values = block_values + value_correction
    else:
        value_correction = torch.zeros_like(block_values)

    denominator = (base_weights * mass_factor).sum()
    numerator = torch.einsum("b,bd->d", base_weights, block_values)
    diagnostics = {
        "maximum_score_variance": float(score_variance.max()),
        "mean_correction_norm": float(value_correction.norm(dim=-1).mean()),
        "bits_per_token": float(model["bits_per_token"]),
    }
    return denominator, numerator, diagnostics
