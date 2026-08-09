import torch

from analyze_progressive_dimension_cascade_20260717 import (
    cascade_candidates,
    normalized_index_work,
)


def test_cascade_candidates_refines_nested_sets() -> None:
    first = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    second = torch.tensor([1.0, 8.0, 7.0, 6.0, 10.0, 9.0])
    final = torch.tensor([1.0, 2.0, 9.0, 8.0, 10.0, 7.0])
    selected = cascade_candidates(
        (first, second, final), (1.0, 4.0 / 6.0, 2.0 / 6.0)
    )
    assert set(selected.tolist()) == {2, 4}


def test_normalized_index_work_counts_incremental_dimensions() -> None:
    work = normalized_index_work((16, 32, 64), (0.2, 0.08, 0.04))
    assert abs(work - 0.34) < 1.0e-8
