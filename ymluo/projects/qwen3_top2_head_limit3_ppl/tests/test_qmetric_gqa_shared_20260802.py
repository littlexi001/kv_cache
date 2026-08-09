from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_qmetric_gqa_shared_layer0_20260802 import (  # noqa: E402
    evenly_spaced_indices,
    gqa_shared_scores,
)
from analyze_qmetric_global_holdout_layer0_20260802 import (  # noqa: E402
    additive_rvq_tail_output,
    fit_additive_request_centroids,
    proxy_mass_control_variate_output,
    replan_exact_after_refinement,
    rerank_exact_after_refinement,
    rms_standardized_error_scale,
    select_by_output_rms_bound,
    solve_three_action_rms_budget,
)
from run_jointkv_residual_ppl_20260802 import (  # noqa: E402
    augment_selected_indices,
    output_error_feedback_multipliers,
)


def test_evenly_spaced_indices_are_bounded_and_unique() -> None:
    indices = evenly_spaced_indices(100, 17, torch.device("cpu"))
    assert indices.shape == (17,)
    assert indices.unique().numel() == 17
    assert int(indices.min()) == 0
    assert int(indices.max()) == 99


def test_shared_scores_retain_one_priority_per_history_token() -> None:
    generator = torch.Generator().manual_seed(53)
    scores = torch.randn(4, 512, generator=generator)
    for mode in ("raw_max", "margin_max", "mass_sum"):
        shared = gqa_shared_scores(scores, 0.06, mode, sample_count=64)
        assert shared.shape == (512,)
        assert torch.isfinite(shared).all()


def test_mass_sum_prefers_a_token_important_to_multiple_heads() -> None:
    scores = torch.full((2, 4), -4.0)
    scores[0, 0] = 4.0
    scores[1, 1] = 4.0
    scores[:, 2] = 3.5
    shared = gqa_shared_scores(scores, 0.25, "mass_sum")
    assert int(torch.argmax(shared)) == 2


def test_additive_request_centroids_recover_separable_values() -> None:
    primary = torch.tensor([[1.0, -2.0], [-0.5, 3.0]])
    residual = torch.tensor([[0.25, 0.5], [-0.75, 0.125]])
    primary_ids = torch.tensor([0, 0, 1, 1, 0, 1])
    residual_ids = torch.tensor([0, 1, 0, 1, 0, 1])
    values = primary[primary_ids] + residual[residual_ids]
    fitted_primary, fitted_residual, _ = fit_additive_request_centroids(
        values,
        primary_ids,
        residual_ids,
        primary_clusters=2,
        residual_clusters=2,
        bits=16,
        iterations=4,
    )
    reconstructed = fitted_primary[primary_ids] + fitted_residual[residual_ids]
    torch.testing.assert_close(reconstructed, values, atol=5.0e-4, rtol=5.0e-4)


def test_additive_tail_is_exact_for_exact_scores_and_values() -> None:
    scores = torch.tensor([0.2, -0.4, 1.1, 0.7, -0.8, 0.3])
    primary = torch.tensor([[1.0, -2.0], [-0.5, 3.0]])
    residual = torch.tensor([[0.25, 0.5], [-0.75, 0.125]])
    primary_ids = torch.tensor([0, 0, 1, 1, 0, 1])
    residual_ids = torch.tensor([0, 1, 0, 1, 0, 1])
    values = primary[primary_ids] + residual[residual_ids]
    selected = torch.tensor([2, 4])
    output = additive_rvq_tail_output(
        scores,
        scores,
        values,
        selected,
        primary_ids,
        primary,
        residual_ids,
        residual,
    )
    expected = torch.softmax(scores, dim=0) @ values
    torch.testing.assert_close(output, expected, atol=1.0e-6, rtol=1.0e-6)


def test_proxy_mass_control_variate_preserves_exact_baseline() -> None:
    scores = torch.tensor([0.2, -0.4, 1.1, 0.7, -0.8, 0.3])
    values = torch.tensor(
        [[1.0, 2.0], [2.0, -1.0], [0.5, 0.2], [1.5, 2.5], [-1.0, 0.0], [0.1, 0.4]]
    )
    selected = torch.tensor([2, 4])
    output, diagnostics = proxy_mass_control_variate_output(
        scores,
        scores,
        values,
        values,
        selected,
        sample_count=3,
    )
    expected = torch.softmax(scores, dim=0) @ values
    torch.testing.assert_close(output, expected, atol=1.0e-6, rtol=1.0e-6)
    assert diagnostics["sample_tokens"] == 3.0


def test_output_feedback_uses_projected_global_layer_error() -> None:
    full = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sparse = torch.tensor([[1.4, 0.0], [0.0, 1.0]])
    result = output_error_feedback_multipliers(
        full,
        sparse,
        torch.eye(4),
        residual_stream_norm=4.0,
        tolerances=torch.full((2,), 0.1),
        mode="projected_layer_global",
    )
    expected = torch.full((2,), 0.4 / (0.1 * 2.0**0.5))
    torch.testing.assert_close(result["multipliers"], expected)


def test_output_feedback_rss_does_not_overweight_zero_error_head() -> None:
    full = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sparse = torch.tensor([[1.4, 0.0], [0.0, 1.0]])
    result = output_error_feedback_multipliers(
        full,
        sparse,
        torch.eye(4),
        residual_stream_norm=4.0,
        tolerances=torch.full((2,), 0.1),
        mode="projected_layer_rss",
    )
    torch.testing.assert_close(result["multipliers"], torch.tensor([4.0, 1.0]))


def test_augment_selected_indices_preserves_required_actions() -> None:
    priorities = torch.tensor([0.1, 0.9, 0.4, 0.8, 0.2])
    selected = torch.tensor([2])
    expanded = augment_selected_indices(selected, priorities, target_count=3)
    assert set(expanded.tolist()) == {1, 2, 3}


def test_output_rms_tolerance_controls_budget_monotonically() -> None:
    generator = torch.Generator().manual_seed(71)
    scores = torch.randn(128, generator=generator)
    uncertainty = 0.05 + 0.1 * torch.rand(128, generator=generator)
    values = torch.randn(128, 16, generator=generator)
    value_errors = 0.1 + 0.2 * torch.rand(128, generator=generator)
    strict, strict_diagnostics = select_by_output_rms_bound(
        scores, uncertainty, values, value_errors, 0.05
    )
    loose, loose_diagnostics = select_by_output_rms_bound(
        scores, uncertainty, values, value_errors, 0.20
    )
    assert strict.numel() >= loose.numel()
    assert strict_diagnostics["predicted_relative_error_after_selection"] <= 0.05
    assert loose_diagnostics["predicted_relative_error_after_selection"] <= 0.20


def test_three_action_solver_meets_bound_with_disjoint_actions() -> None:
    generator = torch.Generator().manual_seed(79)
    scores = torch.randn(256, generator=generator)
    base_uncertainty = 0.25 + 0.2 * torch.rand(256, generator=generator)
    refined_uncertainty = 0.03 + 0.03 * torch.rand(256, generator=generator)
    values = torch.randn(256, 16, generator=generator)
    value_errors = 0.01 * torch.rand(256, generator=generator)
    refined, exact, diagnostics = solve_three_action_rms_budget(
        scores,
        base_uncertainty,
        refined_uncertainty,
        values,
        value_errors,
        relative_tolerance=0.08,
        refinement_cost=48.0,
        exact_cost=4096.0,
    )
    assert diagnostics["predicted_relative_error_after_actions"] <= 0.08
    assert not set(refined.tolist()).intersection(exact.tolist())
    assert diagnostics["refined_tokens"] == float(refined.numel())
    assert diagnostics["exact_tokens"] == float(exact.numel())


def test_three_action_solver_avoids_useless_refinement() -> None:
    generator = torch.Generator().manual_seed(83)
    scores = torch.randn(64, generator=generator)
    uncertainty = 0.5 * torch.ones(64)
    values = torch.randn(64, 8, generator=generator)
    refined, exact, diagnostics = solve_three_action_rms_budget(
        scores,
        uncertainty,
        uncertainty,
        values,
        torch.zeros(64),
        relative_tolerance=0.05,
        refinement_cost=1.0,
        exact_cost=10.0,
    )
    assert refined.numel() == 0
    assert exact.numel() > 0
    assert diagnostics["predicted_relative_error_after_actions"] <= 0.05


def test_replan_consumes_refined_scores_before_exact_selection() -> None:
    base_scores = torch.tensor([2.0, 1.0, 0.0, -1.0])
    refined_scores = torch.tensor([2.0, 1.0, 4.0, -1.0])
    base_uncertainty = torch.full((4,), 0.5)
    refined_uncertainty = torch.full((4,), 0.05)
    values = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0], [-1.0, 0.0]]
    )
    value_errors = torch.full((4,), 0.1)
    refined_indices = torch.tensor([2])
    exact, hybrid_scores, hybrid_uncertainty, diagnostics = (
        replan_exact_after_refinement(
            base_scores,
            refined_scores,
            base_uncertainty,
            refined_uncertainty,
            values,
            value_errors,
            relative_tolerance=0.20,
            refined_indices=refined_indices,
        )
    )
    assert float(hybrid_scores[2]) == 4.0
    assert abs(float(hybrid_uncertainty[2]) - 0.05) < 1.0e-6
    assert int(torch.argmax(hybrid_scores)) == 2
    assert exact.numel() > 0
    assert diagnostics["predicted_relative_error_after_selection"] <= 0.20


def test_fixed_cost_rerank_preserves_count_and_uses_refined_risk() -> None:
    base_scores = torch.tensor([2.0, 1.0, 0.0, -1.0])
    refined_scores = torch.tensor([2.0, 1.0, 4.0, -1.0])
    base_uncertainty = torch.full((4,), 0.5)
    refined_uncertainty = torch.full((4,), 0.05)
    values = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0], [-1.0, 0.0]]
    )
    value_errors = torch.full((4,), 0.1)
    exact, hybrid_scores, _, diagnostics = rerank_exact_after_refinement(
        base_scores,
        refined_scores,
        base_uncertainty,
        refined_uncertainty,
        values,
        value_errors,
        refined_indices=torch.tensor([2]),
        exact_count=1,
    )
    assert exact.numel() == 1
    assert int(exact[0]) == 2
    assert int(torch.argmax(hybrid_scores)) == 2
    assert diagnostics["fixed_exact_tokens"] == 1.0


def test_rms_score_scale_recovers_known_multiplier() -> None:
    proxy = torch.tensor([0.1, -0.3, 0.7, 1.2])
    uncertainty = torch.tensor([0.2, 0.4, 0.1, 0.5])
    signs = torch.tensor([1.0, -1.0, 1.0, -1.0])
    exact = proxy + 1.75 * uncertainty * signs
    scale = rms_standardized_error_scale(
        exact,
        proxy,
        uncertainty,
        torch.arange(4),
    )
    assert abs(scale - 1.75) < 1.0e-6
