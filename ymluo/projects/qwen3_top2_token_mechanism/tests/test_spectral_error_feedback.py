from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_spectral_error_feedback import (
    boundary_rescue_candidates,
    residual_score_radius,
    select_energy_band,
    update_query_state,
)


def test_select_energy_band_uses_covariance_weighted_group_energy() -> None:
    residual = torch.tensor([[3.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0]])
    eigenvalues = torch.ones(4)

    assert select_energy_band(residual, eigenvalues, band_size=2) == 0


def test_update_query_state_only_changes_selected_contiguous_band() -> None:
    state = torch.zeros((2, 4))
    current = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    updated = update_query_state(state, current, band=1, band_size=2)

    assert torch.equal(updated[:, :2], torch.zeros((2, 2)))
    assert torch.equal(updated[:, 2:], current[:, 2:])


def test_residual_score_radius_bounds_exact_omitted_score() -> None:
    residual = torch.tensor([[3.0, 4.0, 1.0, -2.0]])
    key = torch.tensor([[1.0, -2.0, 4.0, 3.0], [-2.0, 1.0, 0.5, -1.0]])

    radius = residual_score_radius(residual, key, band_size=2)
    exact = residual @ key.T

    assert torch.all(exact.abs() <= radius + 1.0e-6)


def test_boundary_rescue_preserves_primary_and_adds_optimistic_token() -> None:
    scores = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0])
    radius = torch.tensor([0.0, 0.0, 0.0, 10.0, 0.0])

    candidates = boundary_rescue_candidates(scores, radius, 2, 3)

    assert set(candidates.tolist()) == {0, 1, 3}
