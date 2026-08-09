from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from export_rope_pair_contributions_8b import (  # noqa: E402
    invert_rope,
    pair_contributions,
    rope_cos_sin_on_device,
    rotate_half,
)


def test_invert_scaled_split_half_rope() -> None:
    torch.manual_seed(0)
    values = torch.randn(2, 7, 8)
    angles = torch.randn(7, 4)
    scale = 1.13
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1) * scale
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1) * scale
    rotated = values * cos + rotate_half(values) * sin
    restored = invert_rope(rotated, cos, sin)
    torch.testing.assert_close(restored, values, atol=1e-5, rtol=1e-5)


def test_pair_sum_matches_scaled_dot_product() -> None:
    torch.manual_seed(1)
    query = torch.randn(3, 8)
    keys = torch.randn(3, 10, 8)
    scaling = 8 ** -0.5
    contributions = pair_contributions(query, keys, scaling, bin_size=1)
    expected = (query[:, None, :] * keys).sum(dim=-1) * scaling
    torch.testing.assert_close(contributions.sum(dim=-1), expected)


def test_pair_binning_preserves_bin_mean_total() -> None:
    torch.manual_seed(2)
    query = torch.randn(2, 8)
    keys = torch.randn(2, 9, 8)
    contributions = pair_contributions(query, keys, scaling=1.0, bin_size=4)
    totals = (query[:, None, :] * keys).sum(dim=-1)
    expected = torch.stack(
        (totals[:, 0:4].mean(dim=1), totals[:, 4:8].mean(dim=1), totals[:, 8:9].mean(dim=1)),
        dim=1,
    )
    torch.testing.assert_close(contributions.sum(dim=-1), expected)


def test_local_rope_table_uses_split_half_layout_and_scaling() -> None:
    inv_freq = torch.tensor([1.0, 0.25])
    cos, sin = rope_cos_sin_on_device(
        inv_freq,
        attention_scaling=1.2,
        key_length=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_angles = torch.arange(3, dtype=torch.float32)[:, None] * inv_freq[None, :]
    torch.testing.assert_close(cos, torch.cat((expected_angles.cos(), expected_angles.cos()), dim=-1) * 1.2)
    torch.testing.assert_close(sin, torch.cat((expected_angles.sin(), expected_angles.sin()), dim=-1) * 1.2)
