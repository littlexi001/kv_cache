from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from analyze_pca_int4_delta_invariance_20260726 import (  # noqa: E402
    candidate_metrics,
    covariance_basis,
    production_int4_dequantize,
    sampled_quantile_metrics,
)


def test_uncentered_pca_basis_matches_right_singular_subspace() -> None:
    generator = torch.Generator().manual_seed(7)
    matrix = torch.randn(80, 12, generator=generator)
    _, basis = covariance_basis(matrix, rank=5)
    _, _, right = torch.linalg.svd(matrix, full_matrices=False)
    right_basis = right[:5].transpose(0, 1)
    overlap = torch.linalg.svdvals(basis.transpose(0, 1) @ right_basis)
    assert torch.allclose(overlap, torch.ones_like(overlap), atol=1.0e-5)


def test_grouped_int4_error_is_small_for_exactly_representable_values() -> None:
    values = torch.zeros(3, 48)
    values[0, 0] = 7.0
    values[0, 1] = -6.0
    values[1, 16] = 3.5
    values[2, 47] = -1.0
    restored = production_int4_dequantize(values)
    assert torch.allclose(restored, values, atol=1.0e-6)


def test_dropped_mass_bounds_exact_candidate_attention_output() -> None:
    exact = torch.tensor([4.0, 2.0, 1.0, -1.0])
    current = torch.tensor(0.5)
    proxy = exact.clone()
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    current_value = torch.tensor([0.25, 0.25])
    attention = torch.softmax(torch.cat((exact, current.view(1))), dim=0)
    metrics = candidate_metrics(
        proxy,
        exact,
        current,
        attention,
        values,
        current_value,
        fraction=0.5,
    )
    assert metrics["output_bound_satisfied"] == 1.0
    assert metrics["retained_attention_mass"] > 0.9


def test_proxy_topk_retained_mass_obeys_uniform_error_bound() -> None:
    exact_scores = torch.tensor([4.0, 3.0, 2.0, 1.0, -1.0])
    proxy_scores = torch.tensor([3.7, 2.6, 2.5, 1.2, -0.8])
    current_score = torch.tensor(0.5)
    full_attention = torch.softmax(
        torch.cat((exact_scores, current_score.view(1))),
        dim=0,
    )

    metrics = candidate_metrics(
        proxy_scores,
        exact_scores,
        current_score,
        full_attention,
        history_value=None,
        current_value=None,
        fraction=0.4,
    )

    assert metrics["deterministic_mass_bound_satisfied"] == 1.0
    assert (
        metrics["retained_attention_mass"]
        >= metrics["deterministic_retained_mass_lower_bound"]
    )
    assert metrics["attention_mass_weighted_topk_recall"] >= metrics["topk_recall"]
    assert metrics["retained_attention_mass_regret"] == pytest.approx(
        metrics["missed_exact_top_attention_mass"]
        - metrics["selected_extra_attention_mass"]
    )


def test_sampled_quantile_reports_variable_budget_and_capacity() -> None:
    exact_scores = torch.arange(32, dtype=torch.float32)
    current_score = torch.tensor(0.0)
    full_attention = torch.softmax(
        torch.cat((exact_scores, current_score.view(1))),
        dim=0,
    )

    metrics = sampled_quantile_metrics(
        exact_scores,
        exact_scores,
        current_score,
        full_attention,
        history_value=None,
        current_value=None,
        fraction=0.25,
        sample_count=8,
        capacity_fraction=0.5,
    )

    assert metrics["keep_count"] == 8
    assert metrics["candidate_capacity"] == 16
    assert metrics["sampled_selected_count"] > 0
    assert metrics["sampled_selected_fraction"] <= 0.5
    assert metrics["sampled_candidate_overflow"] == 0.0
