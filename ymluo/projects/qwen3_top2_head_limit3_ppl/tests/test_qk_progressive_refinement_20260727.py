from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_qk_progressive_refinement_20260727 import (
    allocation_rate,
    conformal_radius,
    interval_candidates,
    refine_and_select,
    systematic_sample_indices,
)


def test_interval_refinement_recovers_full_topk_under_valid_radius() -> None:
    base = torch.tensor([10.0, 8.8, 8.5, 8.3, 1.0, -2.0])
    full = torch.tensor([9.7, 8.4, 8.9, 8.2, 1.2, -1.8])
    radius = torch.tensor(0.4)

    selected, candidate = refine_and_select(base, full, radius, 2)

    expected = set(torch.topk(full, k=2).indices.tolist())
    assert set(selected.tolist()) == expected
    assert candidate[torch.tensor(sorted(expected))].all()


def test_interval_candidate_contains_topk_if_all_intervals_are_valid() -> None:
    generator = torch.Generator().manual_seed(7)
    for token_count in (17, 101):
        for top_count in (1, 3, 7):
            base = torch.randn(token_count, generator=generator)
            error = 0.2 * torch.randn(
                token_count,
                generator=generator,
            )
            full = base + error
            radius = error.abs().max()
            candidate, _ = interval_candidates(base, radius, top_count)
            full_top = torch.topk(full, k=top_count).indices
            assert candidate[full_top].all()


def test_systematic_sample_is_unique_and_in_range() -> None:
    sample = systematic_sample_indices(
        token_count=97,
        sample_count=31,
        phase=11,
        device=torch.device("cpu"),
    )
    assert sample.numel() == 31
    assert torch.unique(sample).numel() == sample.numel()
    assert int(sample.min()) >= 0
    assert int(sample.max()) < 97


def test_conformal_radius_uses_finite_sample_rank() -> None:
    errors = torch.arange(1, 11, dtype=torch.float32)
    assert float(conformal_radius(errors, 0.20)) == 9.0
    assert float(conformal_radius(errors, 0.01)) == 10.0


def test_allocation_rate_counts_scale_metadata() -> None:
    assert allocation_rate((4, 4, 4, 0, 0, 0, 0, 0)) == 15
    assert allocation_rate((8, 4, 0, 0, 0, 0, 0, 0)) == 14
