from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_qksieve_taylor_tail_20260802 import (  # noqa: E402
    fit_taylor_block_tail,
    taylor_block_tail_statistics,
)


def test_constant_blocks_are_exact() -> None:
    keys = torch.tensor([[1.0, -0.5]]).repeat(8, 1)
    values = torch.tensor([[0.2, 0.8, -0.3]]).repeat(8, 1)
    coordinates = keys.clone()
    model = fit_taylor_block_tail(
        coordinates,
        keys,
        values,
        block_size=4,
        key_mean_bits=16,
        value_mean_bits=16,
        variance_bits=16,
        cross_bits=16,
        cross_key_dim=2,
        cross_value_dim=2,
    )
    direction = torch.tensor([0.3, -0.2])
    selected = torch.tensor([0, 5])
    reference = keys[0] @ direction
    denominator, numerator, diagnostics = taylor_block_tail_statistics(
        direction,
        direction,
        selected,
        reference,
        model,
        use_variance=True,
        use_cross=True,
    )
    assert torch.allclose(denominator, torch.tensor(6.0), atol=1.0e-6)
    assert torch.allclose(numerator, 6.0 * values[0], atol=1.0e-6)
    assert diagnostics["bits_per_token"] > 0.0


def test_cross_moment_changes_value_not_mass() -> None:
    coordinates = torch.tensor([[-1.0], [-0.5], [0.5], [1.0]])
    keys = coordinates.clone()
    values = torch.cat((coordinates, -coordinates), dim=-1)
    model = fit_taylor_block_tail(
        coordinates,
        keys,
        values,
        block_size=4,
        key_mean_bits=16,
        value_mean_bits=16,
        variance_bits=16,
        cross_bits=16,
        cross_key_dim=1,
        cross_value_dim=2,
    )
    args = (torch.tensor([0.4]), torch.tensor([0.4]), torch.empty(0, dtype=torch.long), 0.0, model)
    denominator_without, numerator_without, _ = taylor_block_tail_statistics(
        *args, use_variance=True, use_cross=False
    )
    denominator_with, numerator_with, _ = taylor_block_tail_statistics(
        *args, use_variance=True, use_cross=True
    )
    assert torch.allclose(denominator_with, denominator_without)
    assert numerator_with.norm() > numerator_without.norm()
