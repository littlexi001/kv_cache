from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_qkbalanced_additivity_closure_20260727 import (
    band_error_decomposition,
    block_offdiagonal_ratio,
)
from analyze_qk_balanced_spectral_rate_20260727 import (
    covariance,
    qk_balanced_factors,
)


def test_block_offdiagonal_ratio_is_zero_for_block_diagonal() -> None:
    matrix = torch.block_diag(
        torch.randn(16, 16),
        torch.randn(16, 16),
    )
    assert block_offdiagonal_ratio(matrix) == 0.0


def test_band_error_decomposition_is_an_exact_identity() -> None:
    generator = torch.Generator().manual_seed(23)
    residual = torch.randn(101, 128, generator=generator)
    query = torch.randn(128, generator=generator)
    actual, diagonal, cross = band_error_decomposition(residual, query)
    direct = float((residual @ query).square().mean().item())

    assert abs(actual - direct) < 1.0e-4
    assert abs(actual - diagonal - cross) < 1.0e-8


def test_qk_balanced_factors_preserve_scores_and_balance_covariances() -> None:
    generator = torch.Generator().manual_seed(29)
    dimension = 32
    keys = torch.randn(1024, dimension, generator=generator)
    queries = torch.randn(256, dimension, generator=generator)
    queries[:, :8] *= 3.0
    keys[:, 8:16] *= 2.0
    shrinkage = 0.75

    query_factor, key_factor, singular_values = qk_balanced_factors(
        keys,
        queries,
        shrinkage,
    )

    identity = query_factor @ key_factor.transpose(0, 1)
    assert torch.allclose(
        identity,
        torch.eye(dimension),
        atol=2.0e-4,
        rtol=2.0e-4,
    )
    assert torch.allclose(
        (queries @ query_factor)
        @ (keys @ key_factor).transpose(0, 1),
        queries @ keys.transpose(0, 1),
        atol=2.0e-3,
        rtol=2.0e-4,
    )

    query_covariance = covariance(queries)
    isotropic_scale = query_covariance.diagonal().mean()
    regularized_query_covariance = (
        (1.0 - shrinkage) * query_covariance
        + shrinkage
        * isotropic_scale
        * torch.eye(dimension)
    )
    target = torch.diag(singular_values)
    transformed_query_covariance = (
        query_factor.transpose(0, 1)
        @ regularized_query_covariance
        @ query_factor
    )
    transformed_key_covariance = (
        key_factor.transpose(0, 1)
        @ covariance(keys)
        @ key_factor
    )
    assert torch.allclose(
        transformed_query_covariance,
        target,
        atol=2.0e-3,
        rtol=2.0e-3,
    )
    assert torch.allclose(
        transformed_key_covariance,
        target,
        atol=2.0e-3,
        rtol=2.0e-3,
    )
