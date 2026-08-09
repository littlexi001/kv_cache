from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_residual_certified_pca_20260717 import (  # noqa: E402
    exact_rerank,
    fixed_budget_residual_rescue,
    interval_candidate_mask,
    projection_error_terms,
    quantize_dequantize_positive,
    union_rescue_candidates,
)


def orthonormal_projection(dimension: int, rank: int) -> torch.Tensor:
    torch.manual_seed(11)
    matrix = torch.randn(dimension, rank)
    return torch.linalg.qr(matrix, mode="reduced").Q


def test_projection_error_bounds_cover_exact_score_error() -> None:
    torch.manual_seed(5)
    query = torch.randn(16)
    key = torch.randn(127, 16)
    projection = orthonormal_projection(16, 7)

    exact, approximate, two_part, single, _ = projection_error_terms(
        query, key, projection, scaling=0.25
    )
    error = (exact - approximate).abs()

    assert torch.all(error <= two_part + 1.0e-5)
    assert torch.all(error <= single + 1.0e-5)


def test_interval_candidate_is_certified_topk_superset() -> None:
    torch.manual_seed(9)
    query = torch.randn(24)
    key = torch.randn(211, 24)
    projection = orthonormal_projection(24, 9)
    exact, approximate, two_part, _, _ = projection_error_terms(
        query, key, projection, scaling=0.2
    )
    true_top = torch.topk(exact, k=7).indices
    candidate_mask = interval_candidate_mask(approximate, two_part, top_count=7)

    assert bool(candidate_mask[true_top].all())
    candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    reranked = exact_rerank(exact, candidates, top_count=7)
    assert set(reranked.tolist()) == set(true_top.tolist())


def test_residual_rescue_preserves_exact_budget() -> None:
    approximate = torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0, 3.0])
    error_norm = torch.tensor([0.0, 0.0, 0.0, 9.0, 8.0, 7.0])

    selected = fixed_budget_residual_rescue(
        approximate, error_norm, top_count=4, rescue_count=2
    )

    assert selected.numel() == 4
    assert {0, 1, 3, 4}.issubset(set(selected.tolist()))


def test_union_rescue_never_drops_primary_candidates() -> None:
    approximate = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
    rescue = torch.tensor([0.0, 0.0, 0.0, 1.0, 3.0, 2.0])

    candidates = union_rescue_candidates(
        approximate, rescue, primary_count=3, total_count=5
    )

    assert candidates.numel() == 5
    assert {0, 1, 2}.issubset(set(candidates.tolist()))
    assert {4, 5}.issubset(set(candidates.tolist()))


def test_positive_scalar_quantization_respects_requested_bit_grid() -> None:
    values = torch.tensor([0.0, 0.1, 0.4, 0.8, 1.0])

    restored = quantize_dequantize_positive(values, bits=4)

    assert restored.min() >= 0.0
    assert restored.max() <= values.max() + 1.0e-6
    assert torch.unique(restored).numel() <= 16
