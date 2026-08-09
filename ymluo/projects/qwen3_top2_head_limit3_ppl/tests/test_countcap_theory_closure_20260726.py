from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from analyze_countcap_margin_certificate_20260726 import (  # noqa: E402
    ranking_certificate_metrics,
    sampled_threshold_metrics,
)
from analyze_countcap_prefix_drift_20260726 import (  # noqa: E402
    prefix_drift_metrics,
)
from fit_countcap_cost_model_20260726 import (  # noqa: E402
    find_crossover,
    fit_model,
    target_budget,
)


def test_prefix_drift_decomposition_is_exact_when_basis_matches() -> None:
    generator = torch.Generator().manual_seed(31)
    key = torch.randn(64, 8, generator=generator)
    queries = torch.randn(12, 8, generator=generator)
    metrics = prefix_drift_metrics(
        key,
        queries,
        rank=4,
        prefix_tokens=64,
        sample_stride=1,
    )

    assert metrics["projector_spectral_distance"] == pytest.approx(
        0.0,
        abs=1.0e-3,
    )
    assert metrics["direct_qk_drift_root_relative_error"] == pytest.approx(
        0.0,
        abs=1.0e-5,
    )
    assert metrics["prefix_qk_root_relative_error"] == pytest.approx(
        metrics["intrinsic_qk_root_relative_error"],
        abs=1.0e-5,
    )
    assert metrics["key_excess_risk_ratio"] == pytest.approx(
        0.0,
        abs=1.0e-5,
    )
    assert metrics["gap_free_key_excess_bound_satisfied"] == 1.0
    assert metrics["decomposition_relative_error"] < 1.0e-5


def test_prefix_drift_detects_late_subspace_shift_and_obeys_triangle() -> None:
    early = torch.tensor(
        [[4.0, 0.0, 0.1, 0.0], [0.0, 4.0, 0.0, 0.1]]
    ).repeat(8, 1)
    late = torch.tensor(
        [[0.0, 0.1, 4.0, 0.0], [0.1, 0.0, 0.0, 4.0]]
    ).repeat(24, 1)
    key = torch.cat((early, late), dim=0)
    queries = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    metrics = prefix_drift_metrics(
        key,
        queries,
        rank=2,
        prefix_tokens=16,
        sample_stride=1,
    )

    assert metrics["projector_spectral_distance"] > 0.9
    assert metrics["direct_qk_drift_root_relative_error"] > 0.5
    assert (
        metrics["prefix_qk_root_relative_error"]
        <= metrics["direct_triangle_rhs"] + 1.0e-6
    )
    assert metrics["key_excess_risk_ratio"] > 0.0
    assert metrics["gap_free_key_excess_bound_satisfied"] == 1.0
    assert metrics["decomposition_relative_error"] < 1.0e-5


def test_uniform_margin_core_is_always_selected() -> None:
    exact = torch.tensor([5.0, 4.0, 1.0, 0.0, -1.0])
    proxy = torch.tensor([5.05, 3.95, 1.02, -0.03, -0.98])
    current = torch.tensor(0.5)
    attention = torch.softmax(torch.cat((exact, current.view(1))), dim=0)
    metrics = ranking_certificate_metrics(
        exact,
        proxy,
        attention,
        fraction=0.4,
        gamma_grid=(0.05, 0.1, 0.2, 0.5),
    )

    assert metrics["robust_core_inclusion_satisfied"] == 1.0
    assert metrics["robust_core_attention_mass"] > 0.0
    assert (
        metrics["actual_proxy_retained_mass"]
        >= metrics["uniform_margin_retained_mass_lower_bound"]
    )
    assert metrics["best_l2_bound_satisfied"] == 1.0


def test_tokenwise_score_bounds_certify_a_high_margin_core() -> None:
    exact = torch.tensor([7.0, 5.0, 2.0, 1.0, 0.0])
    proxy = torch.tensor([6.9, 5.1, 2.1, 0.9, 0.05])
    error_bound = torch.tensor([0.11, 0.11, 0.11, 0.11, 0.06])
    current = torch.tensor(0.0)
    attention = torch.softmax(torch.cat((exact, current.view(1))), dim=0)
    metrics = ranking_certificate_metrics(
        exact,
        proxy,
        attention,
        fraction=0.4,
        gamma_grid=(0.1, 0.2, 0.5),
        score_error_upper_bound=error_bound,
    )

    assert metrics["norm_error_bound_satisfied"] == 1.0
    assert metrics["norm_tokenwise_core_attention_mass"] > 0.0
    assert metrics["norm_tokenwise_core_inclusion_satisfied"] == 1.0


def test_sampled_threshold_margin_core_is_always_selected() -> None:
    exact = torch.tensor([6.0, 5.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0])
    proxy = exact + torch.tensor(
        [0.1, -0.1, 0.05, -0.05, 0.0, 0.02, -0.02, 0.01]
    )
    current = torch.tensor(0.0)
    attention = torch.softmax(torch.cat((exact, current.view(1))), dim=0)
    metrics = sampled_threshold_metrics(
        exact,
        proxy,
        attention,
        fraction=0.25,
        sample_count=8,
    )

    assert metrics["sampled_robust_core_inclusion_satisfied"] == 1.0
    assert (
        metrics["sampled_retained_attention_mass"]
        >= metrics["sampled_uniform_margin_retained_mass_lower_bound"]
    )


def test_cost_model_recovers_nonnegative_linear_coefficients() -> None:
    predictors = [
        [1.0, 2.0, 0.25],
        [1.0, 4.0, 0.25],
        [1.0, 8.0, 0.50],
        [1.0, 16.0, 1.00],
        [1.0, 32.0, 1.28],
    ]
    targets = [
        30.0 + 0.4 * row[1] + 5.0 * row[2]
        for row in predictors
    ]
    fit = fit_model(
        predictors,
        targets,
        ("intercept_ms", "history_ktokens_ms", "attention_ktokens_ms"),
    )

    assert fit["coefficients"]["intercept_ms"] == pytest.approx(30.0)
    assert fit["coefficients"]["history_ktokens_ms"] == pytest.approx(0.4)
    assert fit["coefficients"]["attention_ktokens_ms"] == pytest.approx(5.0)
    assert fit["r_squared"] == pytest.approx(1.0)


def test_budget_and_crossover_respect_frozen_piecewise_rule() -> None:
    assert target_budget(2048) == 256
    assert target_budget(8192) == 492
    assert target_budget(32000) == 1280
    assert target_budget(128000) == 1280

    crossover = find_crossover(
        {
            "intercept_ms": 20.0,
            "history_ktokens_ms": 2.0,
        },
        {
            "intercept_ms": 35.0,
            "history_ktokens_ms": 0.5,
            "attention_ktokens_ms": 0.0,
        },
        budget_inflation=1.0,
        minimum=512,
        maximum=32000,
    )
    assert crossover == 10000
