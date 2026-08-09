from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_lowfreq_quantization_sweep import fixed_clip_int2
from analyze_lowfreq_temporal_reuse import cached_indices, merge_cached_rescue
from analyze_distance_gated_lowfreq_rescue import binary_sign, mask_recent_fraction


def test_fixed_clip_int2_uses_four_symmetric_levels() -> None:
    values = torch.linspace(-2.0, 2.0, 101)
    quantized = fixed_clip_int2(values, 0.75)
    expected = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    assert torch.allclose(torch.unique(quantized), expected)


def test_refresh_four_reuses_only_previous_anchor() -> None:
    current = torch.arange(10).reshape(10, 1, 1)
    reused = cached_indices(current, 4).flatten()
    assert reused.tolist() == [0, 0, 0, 0, 4, 4, 4, 4, 8, 8]


def test_cached_rescue_merge_is_unique_and_fixed_size() -> None:
    base_scores = torch.tensor([[[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]]])
    rescue = torch.tensor([[[1, 5]]])
    merged = merge_cached_rescue(base_scores, rescue, base_count=4)
    assert merged.tolist() == [[[1, 5, 0, 2, 3, 4]]]
    assert torch.unique(merged).numel() == merged.numel()


def test_distance_gate_keeps_only_oldest_prefix() -> None:
    scores = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    gated = mask_recent_fraction(scores, 0.25)
    assert gated[..., :2].tolist() == [[[0.0, 1.0]]]
    assert torch.isneginf(gated[..., 2:]).all()


def test_binary_sign_has_unit_norm() -> None:
    value = torch.tensor([[2.0, -1.0, 0.0, -4.0]])
    quantized = binary_sign(value)
    assert quantized.tolist() == [[0.5, -0.5, 0.5, -0.5]]
    assert torch.allclose(torch.linalg.vector_norm(quantized, dim=-1), torch.ones(1))
