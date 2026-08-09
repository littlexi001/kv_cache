from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_balanced_pca_int4 import (  # noqa: E402
    block_hadamard,
    grouped_scores,
    quantize_per_band_int4,
    quantize_per_band_logscale_int4,
)


def test_dual_scaling_preserves_scores() -> None:
    torch.manual_seed(7)
    key = torch.randn(2, 11, 8)
    query = torch.randn(3, 4, 8)
    variance = torch.rand(2, 8).clamp_min(0.1)
    root = variance.sqrt()
    balanced_key = key / root.unsqueeze(1)
    grouped_query = query.reshape(3, 2, 2, 8)
    balanced_query = (grouped_query * root.unsqueeze(0).unsqueeze(2)).reshape(
        3, 4, 8
    )
    expected = grouped_scores(key, query, 2)
    actual = grouped_scores(balanced_key, balanced_query, 2)
    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)


def test_block_hadamard_preserves_scores() -> None:
    torch.manual_seed(11)
    key = torch.randn(2, 13, 16)
    query = torch.randn(5, 4, 16)
    expected = grouped_scores(key, query, 2)
    actual = grouped_scores(
        block_hadamard(key, 8), block_hadamard(query, 8), 2
    )
    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)


def test_per_band_int4_uses_independent_scales() -> None:
    value = torch.tensor([[0.1, -0.1, 10.0, -10.0]])
    quantized = quantize_per_band_int4(value, 2)
    assert quantized[0, 0].abs() > 0
    assert quantized[0, 2].abs() > quantized[0, 0].abs()


def test_logscale_band_int4_is_finite() -> None:
    value = torch.tensor([[0.02, -0.01, 10.0, -9.0]])
    quantized = quantize_per_band_logscale_int4(value, 2, 0.5)
    assert torch.isfinite(quantized).all()
    assert quantized[0, 0].abs() > 0
