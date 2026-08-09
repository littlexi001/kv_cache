from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_qksieve_layerwise_rate_distortion_20260802 import (  # noqa: E402
    conditional_value_leverage,
    shared_normalized_max_priority,
    shared_output_bound_priority,
)
from analyze_qksieve_conditional_value_moments_20260802 import (  # noqa: E402
    control_variate_tail_statistics,
    fit_gaussian_tilt_moments,
    gaussian_tilt_block_control_values,
    gaussian_tilt_tail_statistics,
    gaussian_tilt_tail_statistics_hybrid,
    gaussian_tilt_tail_statistics_selected_conditioned,
    stratified_uniform_sample_indices,
)


def test_conditional_value_leverage_matches_residual_norm() -> None:
    coordinates = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    values = torch.tensor(
        [[1.0, 0.0], [2.0, 2.0], [4.0, 1.0], [7.0, 3.0]]
    )
    model = {
        "mean_x": torch.tensor([[0.5], [2.5]]),
        "mean_v": torch.tensor([[1.5, 1.0], [5.5, 2.0]]),
        "linear_map": torch.tensor([[[1.0], [2.0]]]),
        "linear_group_ids": torch.tensor([0, 0]),
        "linear_group_blocks": 2,
        "linear_group_count": 1,
        "block_size": 2,
        "block_count": 2,
    }
    actual, rate = conditional_value_leverage(
        coordinates, values, model, bits=16
    )
    block_ids = torch.tensor([0, 0, 1, 1])
    predicted = model["mean_v"].index_select(0, block_ids) + (
        coordinates - model["mean_x"].index_select(0, block_ids)
    ) @ model["linear_map"][0].T
    expected = torch.linalg.vector_norm(values - predicted, dim=-1).clamp_min(
        1.0e-8
    ).log()
    torch.testing.assert_close(actual, expected)
    assert rate == 16.0


def test_four_bit_value_leverage_has_bounded_block_range() -> None:
    coordinates = torch.arange(8, dtype=torch.float32)[:, None]
    values = torch.stack((coordinates[:, 0], coordinates[:, 0].square()), dim=-1)
    model = {
        "mean_x": torch.tensor([[1.5], [5.5]]),
        "mean_v": torch.zeros(2, 2),
        "linear_map": torch.zeros(1, 2, 1),
        "linear_group_ids": torch.tensor([0, 0]),
        "linear_group_blocks": 2,
        "linear_group_count": 1,
        "block_size": 4,
        "block_count": 2,
    }
    actual, rate = conditional_value_leverage(
        coordinates, values, model, bits=4
    )
    assert torch.isfinite(actual).all()
    assert actual.shape == (8,)
    assert rate == 12.0


def test_projected_leverage_tracks_each_gqa_output_slice() -> None:
    coordinates = torch.zeros(4, 1)
    values = torch.tensor(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]]
    )
    model = {
        "mean_x": torch.zeros(2, 1),
        "mean_v": torch.zeros(2, 2),
        "linear_map": torch.zeros(1, 2, 1),
        "linear_group_ids": torch.tensor([0, 0]),
        "linear_group_blocks": 2,
        "linear_group_count": 1,
        "block_size": 2,
        "block_count": 2,
    }
    grams = torch.stack((torch.eye(2), torch.diag(torch.tensor([4.0, 1.0]))))
    actual, rate = conditional_value_leverage(
        coordinates, values, model, bits=16, projection_grams=grams
    )
    expected = torch.stack(
        (
            torch.linalg.vector_norm(values, dim=-1).log(),
            (4.0 * values[:, 0].square() + values[:, 1].square()).sqrt().log(),
        )
    )
    torch.testing.assert_close(actual, expected)
    assert rate == 32.0


def test_shared_output_priority_is_softmax_shift_invariant() -> None:
    scores = torch.tensor(
        [[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]], dtype=torch.float32
    )
    log_z = torch.logsumexp(scores, dim=-1)
    leverage = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.2, 0.0]])
    expected = shared_output_bound_priority(scores, log_z, leverage, 1.0)
    shifts = torch.tensor([17.0, -9.0])
    actual = shared_output_bound_priority(
        scores + shifts[:, None], log_z + shifts, leverage, 1.0
    )
    torch.testing.assert_close(actual, expected)


def test_normalized_max_is_bounded_approximation_to_group_sum() -> None:
    generator = torch.Generator().manual_seed(9)
    scores = torch.randn(4, 17, generator=generator)
    log_z = torch.logsumexp(scores, dim=-1)
    leverage = torch.randn(4, 17, generator=generator)
    exact = shared_output_bound_priority(scores, log_z, leverage, 1.0)
    approximate = shared_normalized_max_priority(
        scores, log_z, leverage, 1.0
    )
    assert torch.all(approximate <= exact + 1.0e-6)
    assert torch.all(exact <= approximate + torch.log(torch.tensor(4.0)) + 1.0e-6)


def test_gaussian_tilt_closure_is_exact_for_constant_block() -> None:
    coordinates = torch.tensor([[2.0, 3.0]]).repeat(8, 1)
    direction = torch.tensor([0.25, -0.5])
    intercept = 0.7
    scores = coordinates @ direction + intercept
    selected = torch.tensor([1, 5])
    model = fit_gaussian_tilt_moments(
        coordinates, block_size=4, moment_bits=16, covariance_mode="diag"
    )
    denominator, weighted_x, diagnostics = gaussian_tilt_tail_statistics(
        scores,
        direction,
        intercept,
        coordinates,
        selected,
        model,
    )
    # Statistics are represented relative to the selected-score threshold.
    torch.testing.assert_close(denominator, torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(
        weighted_x, torch.tensor([[6.0, 9.0], [6.0, 9.0]])
    )
    assert diagnostics["negative_block_fraction"] == 0.0
    conditioned_denominator, conditioned_x, _ = (
        gaussian_tilt_tail_statistics_selected_conditioned(
            scores,
            direction,
            intercept,
            coordinates,
            coordinates,
            selected,
            model,
        )
    )
    torch.testing.assert_close(conditioned_denominator, denominator)
    torch.testing.assert_close(conditioned_x, weighted_x)
    hybrid_denominator, hybrid_x, hybrid_diagnostics = (
        gaussian_tilt_tail_statistics_hybrid(
            scores,
            direction,
            intercept,
            coordinates,
            coordinates,
            selected,
            model,
        )
    )
    torch.testing.assert_close(hybrid_denominator, denominator)
    torch.testing.assert_close(hybrid_x, weighted_x)
    assert hybrid_diagnostics["repaired_block_fraction"] == 0.0


def test_full_gaussian_tilt_moments_remain_finite() -> None:
    generator = torch.Generator().manual_seed(7)
    coordinates = torch.randn(16, 4, generator=generator)
    model = fit_gaussian_tilt_moments(
        coordinates, block_size=8, moment_bits=8, covariance_mode="full"
    )
    scores = coordinates @ torch.tensor([0.2, -0.1, 0.3, 0.4])
    denominator, weighted_x, diagnostics = gaussian_tilt_tail_statistics(
        scores,
        torch.tensor([0.2, -0.1, 0.3, 0.4]),
        0.0,
        coordinates[:, :2],
        torch.tensor([0, 9]),
        model,
    )
    assert torch.isfinite(denominator).all()
    assert torch.isfinite(weighted_x).all()
    assert diagnostics["negative_block_fraction"] >= 0.0


def test_control_variate_is_exact_when_reservoir_covers_every_token() -> None:
    generator = torch.Generator().manual_seed(17)
    coordinates = torch.randn(8, 3, generator=generator)
    values = torch.randn(8, 5, generator=generator)
    query = torch.tensor([0.2, -0.3, 0.4])
    scores = coordinates @ query
    selected = torch.tensor([1, 6])
    conditional_model = {
        "mean_x": torch.zeros(2, 2),
        "mean_v": torch.zeros(2, 5),
        "linear_map": torch.zeros(1, 5, 2),
        "linear_group_ids": torch.tensor([0, 0]),
    }
    gaussian_model = fit_gaussian_tilt_moments(
        coordinates, block_size=4, moment_bits=16, covariance_mode="diag"
    )
    reference = scores.index_select(0, selected).amin()
    base_z, base_y = gaussian_tilt_block_control_values(
        query, 0.0, reference, conditional_model, gaussian_model
    )
    sample = stratified_uniform_sample_indices(
        8, 4, 4, torch.Generator().manual_seed(3)
    )
    denominator, numerator, diagnostics = control_variate_tail_statistics(
        scores,
        values,
        selected,
        sample,
        4,
        base_z,
        base_y,
        reference,
    )
    tail = torch.ones(8, dtype=torch.bool)
    tail[selected] = False
    weights = torch.exp(scores - reference) * tail
    torch.testing.assert_close(denominator, weights.sum())
    torch.testing.assert_close(numerator, weights @ values)
    assert diagnostics["sample_selected_overlap"] == 2.0


def test_stratified_control_variate_is_unbiased_with_selected_overlap() -> None:
    generator = torch.Generator().manual_seed(31)
    scores = torch.randn(12, generator=generator)
    values = torch.randn(12, 3, generator=generator)
    selected = torch.topk(scores, 3, sorted=False).indices
    reference = scores.index_select(0, selected).amin()
    base_z = torch.tensor([0.7, 1.1, 0.4])
    base_y = torch.randn(3, 3, generator=generator)
    estimates_z = []
    estimates_y = []
    for seed in range(1200):
        sample = stratified_uniform_sample_indices(
            12, 4, 1, torch.Generator().manual_seed(seed)
        )
        denominator, numerator, _ = control_variate_tail_statistics(
            scores,
            values,
            selected,
            sample,
            4,
            base_z,
            base_y,
            reference,
        )
        estimates_z.append(denominator)
        estimates_y.append(numerator)
    tail = torch.ones(12, dtype=torch.bool)
    tail[selected] = False
    weights = torch.exp(scores - reference) * tail
    torch.testing.assert_close(
        torch.stack(estimates_z).mean(), weights.sum(), atol=0.12, rtol=0.03
    )
    torch.testing.assert_close(
        torch.stack(estimates_y).mean(dim=0),
        weights @ values,
        atol=0.22,
        rtol=0.08,
    )
