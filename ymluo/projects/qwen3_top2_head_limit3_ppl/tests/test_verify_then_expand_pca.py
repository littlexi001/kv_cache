import torch

from analyze_verify_then_expand_pca_20260717 import (
    conformal_upper_quantile,
    evaluate_selection,
)


def test_conformal_upper_quantile_uses_finite_sample_correction() -> None:
    values = [0.1, 0.2, 0.3, 0.4]
    assert conformal_upper_quantile(values, 0.5) == 0.3
    assert conformal_upper_quantile(values, 0.9) == 0.4


def test_verify_then_expand_stops_when_tail_is_certified() -> None:
    exact = torch.tensor([10.0, 9.0, 2.0, 1.0, 0.0])
    approximate = torch.tensor([9.8, 8.8, 2.1, 1.2, 0.3])
    bound = torch.full_like(exact, 0.1)
    attention = torch.softmax(exact, dim=0)
    result = evaluate_selection(
        exact,
        approximate,
        bound,
        attention,
        top_count=2,
        fractions=(0.4, 0.6, 0.8),
        alpha=1.0,
    )
    assert result["candidate_ratio"] == 0.4
    assert result["stopped_by_bound"] == 1.0
    assert result["top2_recall"] == 1.0


def test_verify_then_expand_expands_after_proxy_miss() -> None:
    exact = torch.tensor([10.0, 9.0, 8.0, 1.0, 0.0])
    approximate = torch.tensor([9.8, 2.0, 8.1, 1.2, 0.3])
    bound = torch.tensor([0.2, 8.0, 0.2, 0.2, 0.2])
    attention = torch.softmax(exact, dim=0)
    result = evaluate_selection(
        exact,
        approximate,
        bound,
        attention,
        top_count=2,
        fractions=(0.4, 0.6, 0.8),
        alpha=1.0,
    )
    assert result["candidate_ratio"] == 0.6
    assert result["top2_recall"] == 1.0
