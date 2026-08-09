from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_head_top2_targeted_ppl_20260714 import (
    _gaussian_expected_outside_crossings,
    _gaussian_tail_density_outside_crossings,
    _plan_one_shot_density_band_counts,
)


def test_expected_crossing_risk_increases_with_residual_sigma() -> None:
    scores = torch.tensor([[[10.0, 9.0, 3.0, 1.0]]])

    low = _gaussian_expected_outside_crossings(
        scores,
        torch.tensor([[0.1]]),
        keep_count=1,
        candidate_count=2,
    )
    high = _gaussian_expected_outside_crossings(
        scores,
        torch.tensor([[10.0]]),
        keep_count=1,
        candidate_count=2,
    )

    assert low.item() < high.item()


def test_expected_crossing_risk_excludes_current_candidate_pool() -> None:
    scores = torch.tensor([[[10.0, 9.0, 8.0, 7.0]]])
    sigma = torch.tensor([[1.0]])

    risk_two = _gaussian_expected_outside_crossings(scores, sigma, 1, 2)
    risk_three = _gaussian_expected_outside_crossings(scores, sigma, 1, 3)

    assert risk_three.item() < risk_two.item()


def test_expected_crossing_accepts_per_head_keep_counts() -> None:
    scores = torch.tensor(
        [[[10.0, 9.0, 8.0, 7.0]], [[10.0, 9.0, 8.0, 7.0]]]
    )
    sigma = torch.ones((2, 1))

    risk = _gaussian_expected_outside_crossings(
        scores,
        sigma,
        keep_count=torch.tensor([[1], [2]]),
        candidate_count=3,
    )

    assert risk.shape == (2, 1)
    assert risk[1, 0] > risk[0, 0]


def test_expected_crossing_rejects_mismatched_per_head_keep_counts() -> None:
    scores = torch.zeros((1, 2, 4))
    sigma = torch.ones((1, 2))

    try:
        _gaussian_expected_outside_crossings(
            scores,
            sigma,
            keep_count=torch.ones((1, 1), dtype=torch.long),
            candidate_count=3,
        )
    except ValueError as exc:
        assert "per-row keep counts" in str(exc)
    else:
        raise AssertionError("mismatched keep-count shape was accepted")


def test_tail_density_risk_supports_per_head_keep_counts() -> None:
    scores = torch.linspace(10.0, 0.0, 100).reshape(1, 1, -1).repeat(1, 2, 1)
    risk = _gaussian_tail_density_outside_crossings(
        scores,
        residual_sigma=torch.tensor([[0.5, 2.0]]),
        keep_count=torch.tensor([[10, 30]]),
        candidate_count=30,
    )

    assert risk.shape == (1, 2)
    assert torch.isfinite(risk).all()
    assert (risk >= 0.0).all()


def test_one_shot_planner_stops_when_anchor_matches_query() -> None:
    query = torch.zeros((1, 1, 2, 64))
    top_values = torch.linspace(10.0, 1.0, 8).reshape(1, 1, -1).repeat(1, 2, 1)

    planned, risk = _plan_one_shot_density_band_counts(
        query,
        query.clone(),
        spectral_weights=torch.ones((1, 1, 64)),
        initial_top_values=top_values,
        keep_count=2,
        total_token_count=100,
        target_recall=0.95,
        band_size=16,
    )

    assert torch.equal(planned, torch.ones_like(planned))
    assert (risk == 0.0).all()


def test_one_shot_planner_expands_flat_unsafe_boundary() -> None:
    query = torch.ones((1, 1, 2, 64))
    anchor = torch.zeros_like(query)
    top_values = torch.zeros((1, 2, 8))

    planned, _ = _plan_one_shot_density_band_counts(
        query,
        anchor,
        spectral_weights=torch.ones((1, 1, 64)),
        initial_top_values=top_values,
        keep_count=2,
        total_token_count=100,
        target_recall=0.95,
        band_size=16,
    )

    assert torch.equal(planned, torch.full_like(planned, 4))
