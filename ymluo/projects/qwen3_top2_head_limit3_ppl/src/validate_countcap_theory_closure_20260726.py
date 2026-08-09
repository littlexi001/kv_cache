from __future__ import annotations

import json

import torch

from analyze_countcap_margin_certificate_20260726 import (
    ranking_certificate_metrics,
    sampled_threshold_metrics,
)
from analyze_countcap_prefix_drift_20260726 import prefix_drift_metrics
from fit_countcap_cost_model_20260726 import (
    find_crossover,
    fit_model,
    target_budget,
)


def assert_close(left: float, right: float, tolerance: float = 1.0e-5) -> None:
    if abs(left - right) > tolerance:
        raise AssertionError(f"{left} != {right} within {tolerance}")


def validate_prefix() -> dict[str, float]:
    generator = torch.Generator().manual_seed(31)
    key = torch.randn(64, 8, generator=generator)
    queries = torch.randn(12, 8, generator=generator)
    matched = prefix_drift_metrics(
        key,
        queries,
        rank=4,
        prefix_tokens=64,
        sample_stride=1,
    )
    # Principal-angle distance takes sqrt(1-cos^2), so float32 roundoff
    # near cos=1 is amplified to roughly 1e-3.
    assert_close(
        matched["projector_spectral_distance"],
        0.0,
        tolerance=1.0e-3,
    )
    assert_close(matched["direct_qk_drift_root_relative_error"], 0.0)
    assert_close(
        matched["prefix_qk_root_relative_error"],
        matched["intrinsic_qk_root_relative_error"],
    )
    assert_close(matched["key_excess_risk_ratio"], 0.0)
    if matched["gap_free_key_excess_bound_satisfied"] != 1.0:
        raise AssertionError("gap-free excess-risk bound failed")
    if matched["decomposition_relative_error"] >= 1.0e-5:
        raise AssertionError("prefix residual decomposition is not exact")

    early = torch.tensor(
        [[4.0, 0.0, 0.1, 0.0], [0.0, 4.0, 0.0, 0.1]]
    ).repeat(8, 1)
    late = torch.tensor(
        [[0.0, 0.1, 4.0, 0.0], [0.1, 0.0, 0.0, 4.0]]
    ).repeat(24, 1)
    shifted = prefix_drift_metrics(
        torch.cat((early, late), dim=0),
        torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        ),
        rank=2,
        prefix_tokens=16,
        sample_stride=1,
    )
    if shifted["projector_spectral_distance"] <= 0.9:
        raise AssertionError("late subspace shift was not detected")
    if (
        shifted["prefix_qk_root_relative_error"]
        > shifted["direct_triangle_rhs"] + 1.0e-6
    ):
        raise AssertionError("QK triangle bound failed")
    if shifted["gap_free_key_excess_bound_satisfied"] != 1.0:
        raise AssertionError("shifted gap-free bound failed")
    return {
        "matched_decomposition_error": matched[
            "decomposition_relative_error"
        ],
        "shifted_projector_distance": shifted[
            "projector_spectral_distance"
        ],
    }


def validate_margin() -> dict[str, float]:
    exact = torch.tensor([7.0, 5.0, 2.0, 1.0, 0.0])
    proxy = torch.tensor([6.9, 5.1, 2.1, 0.9, 0.05])
    error_bound = torch.tensor([0.11, 0.11, 0.11, 0.11, 0.06])
    attention = torch.softmax(
        torch.cat((exact, torch.tensor([0.0]))),
        dim=0,
    )
    ranked = ranking_certificate_metrics(
        exact,
        proxy,
        attention,
        fraction=0.4,
        gamma_grid=(0.1, 0.2, 0.5),
        score_error_upper_bound=error_bound,
    )
    for field in (
        "robust_core_inclusion_satisfied",
        "best_l2_bound_satisfied",
        "norm_error_bound_satisfied",
        "norm_tokenwise_core_inclusion_satisfied",
    ):
        if ranked[field] != 1.0:
            raise AssertionError(f"{field} failed")

    sampled = sampled_threshold_metrics(
        torch.tensor([6.0, 5.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]),
        torch.tensor(
            [6.1, 4.9, 2.05, 0.95, 0.0, -0.98, -2.02, -2.99]
        ),
        torch.softmax(
            torch.tensor([6.0, 5.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, 0.0]),
            dim=0,
        ),
        fraction=0.25,
        sample_count=8,
    )
    if sampled["sampled_robust_core_inclusion_satisfied"] != 1.0:
        raise AssertionError("sampled threshold margin core failed")
    if (
        sampled["sampled_retained_attention_mass"]
        < sampled["sampled_uniform_margin_retained_mass_lower_bound"]
    ):
        raise AssertionError("sampled retained-mass lower bound failed")
    return {
        "tokenwise_core_mass": ranked[
            "norm_tokenwise_core_attention_mass"
        ],
        "sampled_mass_recall": sampled[
            "sampled_mass_weighted_topk_recall"
        ],
    }


def validate_cost() -> dict[str, float]:
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
    assert_close(fit["coefficients"]["intercept_ms"], 30.0)
    assert_close(fit["coefficients"]["history_ktokens_ms"], 0.4)
    assert_close(fit["coefficients"]["attention_ktokens_ms"], 5.0)
    assert_close(fit["r_squared"], 1.0)

    expected_budgets = {
        2048: 256,
        8192: 492,
        32000: 1280,
        128000: 1280,
    }
    for history_tokens, expected in expected_budgets.items():
        if target_budget(history_tokens) != expected:
            raise AssertionError("frozen budget rule changed")
    crossover = find_crossover(
        {"intercept_ms": 20.0, "history_ktokens_ms": 2.0},
        {
            "intercept_ms": 35.0,
            "history_ktokens_ms": 0.5,
            "attention_ktokens_ms": 0.0,
        },
        budget_inflation=1.0,
        minimum=512,
        maximum=32000,
    )
    if crossover != 10000:
        raise AssertionError(f"unexpected crossover: {crossover}")
    return {
        "synthetic_fit_r_squared": fit["r_squared"],
        "synthetic_crossover_tokens": float(crossover),
    }


def main() -> None:
    print(
        json.dumps(
            {
                "prefix": validate_prefix(),
                "margin": validate_margin(),
                "cost": validate_cost(),
                "status": "PASS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
