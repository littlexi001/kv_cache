from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_query_adaptive_rank_pca_20260717 import (  # noqa: E402
    choose_rank,
    hierarchical_projected_scores,
    index_storage_ratio,
    quantize_dequantize_binary_sign,
)
from analyze_residual_certified_pca_20260717 import (  # noqa: E402
    quantize_dequantize_int4,
)


def test_energy_policy_uses_smallest_sufficient_rank_and_respects_cap() -> None:
    coverage = {64: 0.68, 80: 0.79, 96: 0.91, 112: 0.97}

    assert choose_rank(coverage, threshold=0.75, maximum_rank=112) == 80
    assert choose_rank(coverage, threshold=0.90, maximum_rank=112) == 96
    assert choose_rank(coverage, threshold=0.95, maximum_rank=96) == 96


def test_hierarchical_rank64_matches_direct_int4_projection() -> None:
    torch.manual_seed(4)
    query = torch.randn(128)
    key = torch.randn(37, 128)
    basis = torch.linalg.qr(torch.randn(128, 128)).Q

    scores, coverage = hierarchical_projected_scores(
        query,
        key,
        basis,
        ranks=[64, 80, 96],
        base_rank=64,
        group_size=16,
        scaling=0.125,
    )
    projected_key = key @ basis[:, :64]
    projected_query = query @ basis[:, :64]
    expected = quantize_dequantize_int4(projected_key) @ projected_query * 0.125

    assert torch.allclose(scores[64], expected, atol=1.0e-5, rtol=1.0e-5)
    assert 0.0 <= coverage[64] <= coverage[80] <= coverage[96] <= 1.0


def test_int2_residual_keeps_base_rank64_unchanged() -> None:
    torch.manual_seed(6)
    query = torch.randn(128)
    key = torch.randn(41, 128)
    basis = torch.linalg.qr(torch.randn(128, 128)).Q

    int4_scores, _ = hierarchical_projected_scores(
        query,
        key,
        basis,
        ranks=[64, 72],
        base_rank=64,
        group_size=8,
        scaling=0.125,
        residual_precision="int4",
    )
    int2_scores, _ = hierarchical_projected_scores(
        query,
        key,
        basis,
        ranks=[64, 72],
        base_rank=64,
        group_size=8,
        scaling=0.125,
        residual_precision="int2_uniform4",
    )

    assert torch.equal(int2_scores[64], int4_scores[64])
    assert not torch.equal(int2_scores[72], int4_scores[72])


def test_binary_sign_uses_one_bit_per_residual_dimension() -> None:
    matrix = torch.tensor(
        [[-3.0, 2.0], [-1.0, -4.0], [2.0, 6.0]], dtype=torch.float32
    )
    restored = quantize_dequantize_binary_sign(matrix)

    expected_scale = matrix.abs().mean(dim=0)
    assert torch.equal(restored.sign(), torch.where(matrix >= 0, 1.0, -1.0))
    assert torch.allclose(restored.abs(), expected_scale.expand_as(matrix))
    assert index_storage_ratio(96, 128, 64, 32, "binary_sign") == 0.07421875


def test_hierarchical_projected_scores_supports_binary_residual() -> None:
    torch.manual_seed(7)
    query = torch.randn(8)
    key = torch.randn(64, 8)

    scores, coverage = hierarchical_projected_scores(
        query,
        key,
        torch.eye(8),
        ranks=[4, 8],
        base_rank=4,
        group_size=4,
        scaling=0.5,
        residual_precision="binary_sign",
    )

    assert set(scores) == {4, 8}
    assert coverage[4] <= coverage[8]
