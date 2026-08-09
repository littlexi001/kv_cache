import torch

from analyze_two_stage_dimension_cascade_20260717 import (
    normalized_work,
    two_stage_candidates,
)


def test_two_stage_candidates_refine_prefix_set() -> None:
    prefix = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0])
    full = torch.tensor([1.0, 2.0, 10.0, 9.0, 8.0])
    selected = two_stage_candidates(prefix, full, 0.8, 0.4)
    assert set(selected.tolist()) == {2, 3}


def test_normalized_work() -> None:
    assert abs(normalized_work(32, 0.2) - 0.6) < 1.0e-8
