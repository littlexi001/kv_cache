from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_bandef_numerics import (
    binary_auc,
    candidate_contains,
    fixed_pca_rank_scores,
    gaussian_boundary_recall_estimate,
    gaussian_expected_outside_crossings,
    gaussian_tail_density_crossings,
    gaussian_tail_density_crossings_from_top_values,
    quantize_dequantize_projected_int4,
)


def test_fixed_pca_rank_scores_use_largest_variance_tail_dimensions() -> None:
    query = torch.tensor([[100.0, 10.0, 2.0, 3.0]])
    key = torch.tensor([[100.0, 10.0, 5.0, 7.0]])

    scores = fixed_pca_rank_scores(query, key, rank=2)

    assert torch.equal(scores, torch.tensor([[31.0]]))


def test_projected_int4_round_trip_error_is_bounded_by_scale() -> None:
    key = torch.tensor([[[-7.0, -3.0, 2.0, 7.0]]])

    restored = quantize_dequantize_projected_int4(key)
    scale = key.abs().amax(dim=-1, keepdim=True) / 7.0

    assert torch.max(torch.abs(key - restored)) <= scale.max()


def test_binary_auc_is_one_for_perfect_ordering() -> None:
    assert binary_auc([0.9, 0.8, 0.2, 0.1], [1.0, 1.0, 0.0, 0.0]) == 1.0


def test_candidate_contains_operates_independently_per_head() -> None:
    candidates = torch.tensor([[1, 2, 3], [7, 8, 9]])
    targets = torch.tensor([[2, 5], [7, 3]])

    contained = candidate_contains(candidates, targets)

    assert torch.equal(contained, torch.tensor([[True, False], [True, False]]))


def test_gaussian_crossing_risk_increases_with_residual_sigma() -> None:
    scores = torch.tensor([[10.0, 9.0, 3.0, 1.0]])

    low_risk = gaussian_expected_outside_crossings(
        scores, torch.tensor([0.1]), keep_count=1, candidate_count=2
    )
    high_risk = gaussian_expected_outside_crossings(
        scores, torch.tensor([10.0]), keep_count=1, candidate_count=2
    )

    assert low_risk.item() < high_risk.item()


def test_gaussian_boundary_recall_tracks_score_correlation() -> None:
    current = torch.tensor([[1.0, 0.0]])
    identical = gaussian_boundary_recall_estimate(
        current,
        current,
        torch.ones(2),
        target_fraction=0.02,
        candidate_fraction=0.08,
    )
    orthogonal = gaussian_boundary_recall_estimate(
        torch.tensor([[0.0, 1.0]]),
        current,
        torch.ones(2),
        target_fraction=0.02,
        candidate_fraction=0.08,
    )

    assert identical.item() > 0.99
    assert orthogonal.item() < 0.10


def test_tail_density_crossings_increase_with_residual_sigma() -> None:
    scores = torch.linspace(10.0, 0.0, 100).repeat(2, 1)
    risk = gaussian_tail_density_crossings(
        scores,
        torch.tensor([0.1, 2.0]),
        keep_count=10,
        candidate_count=30,
    )

    assert risk.shape == (2,)
    assert risk[1] > risk[0]


def test_tail_density_top_values_matches_full_score_entry() -> None:
    scores = torch.linspace(10.0, 0.0, 100).repeat(2, 1)
    sigma = torch.tensor([0.5, 2.0])
    direct = gaussian_tail_density_crossings(
        scores, sigma, keep_count=10, candidate_count=30
    )
    top_values = torch.topk(scores, k=30, dim=-1).values
    from_top = gaussian_tail_density_crossings_from_top_values(
        top_values, sigma, keep_count=10, total_token_count=100
    )

    assert torch.allclose(direct, from_top)
