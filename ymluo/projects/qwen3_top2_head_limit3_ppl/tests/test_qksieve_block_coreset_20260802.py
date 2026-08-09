from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_qksieve_block_coreset_20260802 import (  # noqa: E402
    block_coreset_tail_statistics,
    coreset_corrected_proxy_scores,
    fit_block_coreset,
)


def test_singleton_coreset_recovers_exact_tail() -> None:
    generator = torch.Generator().manual_seed(23)
    coordinates = torch.randn(8, 3, generator=generator)
    values = torch.randn(8, 5, generator=generator)
    direction = torch.tensor([0.2, -0.4, 0.1])
    scores = coordinates @ direction
    selected = torch.tensor([1, 6])
    reference = scores.index_select(0, selected).amin()
    model = fit_block_coreset(
        coordinates,
        values,
        block_size=4,
        cluster_count=4,
        moment_bits=16,
        iterations=4,
    )
    denominator, numerator, diagnostics = block_coreset_tail_statistics(
        coordinates,
        values,
        direction,
        selected,
        reference,
        model,
    )
    tail = torch.ones(8, dtype=torch.bool)
    tail[selected] = False
    weights = torch.exp(scores - reference) * tail
    torch.testing.assert_close(denominator, weights.sum())
    torch.testing.assert_close(numerator, weights @ values)
    assert diagnostics["nonempty_prototypes"] == 6.0


def test_joint_coreset_has_finite_rate_and_statistics() -> None:
    generator = torch.Generator().manual_seed(41)
    coordinates = torch.randn(32, 6, generator=generator)
    values = torch.randn(32, 8, generator=generator)
    model = fit_block_coreset(
        coordinates,
        values,
        block_size=16,
        cluster_count=4,
        moment_bits=8,
        iterations=4,
        value_projection_dim=3,
        value_weight=0.5,
    )
    denominator, numerator, diagnostics = block_coreset_tail_statistics(
        coordinates,
        values,
        torch.randn(6, generator=generator),
        torch.tensor([0, 7, 20]),
        0.0,
        model,
    )
    assert float(model["bits_per_token"]) > 0.0
    assert torch.isfinite(denominator)
    assert torch.isfinite(numerator).all()
    assert diagnostics["maximum_log_weight"] < 100.0


def test_singleton_cluster_score_correction_recovers_full_score() -> None:
    generator = torch.Generator().manual_seed(53)
    full_coordinates = torch.randn(8, 3, generator=generator)
    proxy_coordinates = full_coordinates[:, :1]
    values = torch.randn(8, 2, generator=generator)
    model = fit_block_coreset(
        proxy_coordinates,
        values,
        block_size=4,
        cluster_count=4,
        moment_bits=16,
        full_score_coordinates=full_coordinates,
    )
    full_direction = torch.tensor([0.2, -0.5, 0.7])
    proxy_direction = full_direction[:1]
    proxy_scores = proxy_coordinates @ proxy_direction
    corrected, diagnostics = coreset_corrected_proxy_scores(
        proxy_scores,
        proxy_direction,
        full_direction,
        model,
    )
    torch.testing.assert_close(corrected, full_coordinates @ full_direction)
    assert diagnostics["correction_abs_maximum"] > 0.0
