from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import analyze_qksieve_shrinkage_grid_20260810 as grid  # noqa: E402
import analyze_qk_balanced_spectral_rate_20260727 as reference  # noqa: E402
from analyze_automatic_spectral_rate_allocation_20260727 import (  # noqa: E402
    GROUP_COUNT,
    GROUP_SIZE,
    ZERO_BIT_LEVELS,
    allocate_bits,
    quantize_band,
    reconstruct,
)


def test_selected_reconstruction_matches_reference_path() -> None:
    generator = torch.Generator().manual_seed(7)
    coefficients = torch.randn(256, 128, generator=generator)
    queries = torch.randn(16, 128, generator=generator)
    observed, allocation = grid.selected_reconstruction(
        coefficients, queries, total_rate_budget=15
    )

    expected_allocation = allocate_bits(
        reference.distortion_table(coefficients, queries),
        15,
        ZERO_BIT_LEVELS,
        include_scale_metadata=True,
    )
    bands = []
    for group_index in range(GROUP_COUNT):
        start = group_index * GROUP_SIZE
        band = coefficients[:, start : start + GROUP_SIZE]
        bands.append(
            {bits: quantize_band(band, bits) for bits in ZERO_BIT_LEVELS}
        )
    expected = reconstruct(bands, expected_allocation)

    assert allocation == expected_allocation
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_batched_metrics_match_reference_scalar_formula() -> None:
    generator = torch.Generator().manual_seed(11)
    exact = torch.randn(5, 257, generator=generator)
    approximate = exact + 0.2 * torch.randn(5, 257, generator=generator)
    true_top = torch.topk(exact, k=6, dim=-1).indices
    fractions = (0.01, 0.02, 0.04)
    observed = grid.batched_selection_metrics(
        exact, approximate, true_top, fractions
    )

    attention = torch.softmax(exact, dim=-1)
    for fraction in fractions:
        for row in range(exact.shape[0]):
            expected = reference.selection_metrics(
                exact[row],
                attention[row],
                approximate[row],
                true_top[row],
                fraction,
            )
            for name, value in expected.items():
                actual = observed[fraction][name][row].item()
                assert actual == pytest.approx(value, abs=2e-6, rel=2e-6)


def test_append_rows_preserves_strict_pairing_key() -> None:
    exact = torch.tensor([[3.0, 2.0, 1.0], [2.0, 3.0, 1.0]])
    approximate = exact.clone()
    true_top = torch.topk(exact, k=1, dim=-1).indices
    metrics = grid.batched_selection_metrics(
        exact, approximate, true_top, (1 / 3,)
    )
    rows: list[dict[str, object]] = []
    grid.append_metric_rows(
        rows,
        metrics,
        label="trace",
        layer=8,
        evaluation_start=4,
        kv_head=2,
        query_head_start=6,
        groups=2,
    )
    assert [(row["heldout_step"], row["query_head"]) for row in rows] == [
        (4, 6),
        (4, 7),
    ]
