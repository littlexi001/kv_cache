import torch

from analyze_progressive_attention_stability_20260717 import (
    progressive_stability_select,
    sparse_probability,
    total_variation,
)


def test_sparse_probability_is_normalized() -> None:
    scores = torch.tensor([3.0, 2.0, 1.0, 0.0])
    probability = sparse_probability(scores, torch.tensor([0, 2]))
    assert torch.isclose(probability.sum(), torch.tensor(1.0))
    assert probability[1] == 0.0


def test_total_variation_matches_disjoint_distributions() -> None:
    left = torch.tensor([1.0, 0.0])
    right = torch.tensor([0.0, 1.0])
    assert total_variation(left, right) == 1.0


def test_progressive_stability_stops_after_required_stable_steps() -> None:
    exact = torch.tensor([10.0, 9.0, 2.0, 1.0, 0.0])
    approximate = exact.clone()
    selected, ratio, transitions = progressive_stability_select(
        exact,
        approximate,
        top_count=2,
        fractions=(0.4, 0.6, 0.8, 1.0),
        tv_threshold=1.0e-6,
        required_stable_steps=2,
    )
    assert ratio == 0.8
    assert len(transitions) == 2
    assert set(selected.tolist()) == {0, 1}
