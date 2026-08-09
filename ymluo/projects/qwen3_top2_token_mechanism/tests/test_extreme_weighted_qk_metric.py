from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from analyze_extreme_weighted_qk_metric import (  # noqa: E402
    weighted_key_covariance,
)


def test_uniform_weights_match_second_moment() -> None:
    torch.manual_seed(41)
    key = torch.randn(3, 17, 8)
    expected = torch.einsum("hnd,hne->hde", key, key) / key.shape[1]
    actual = weighted_key_covariance(key, torch.ones(3, 17))
    torch.testing.assert_close(actual, expected)


def test_weight_scale_does_not_change_covariance() -> None:
    torch.manual_seed(43)
    key = torch.randn(2, 13, 6)
    weights = torch.rand(2, 13).clamp_min(0.01)
    expected = weighted_key_covariance(key, weights)
    actual = weighted_key_covariance(key, 9.0 * weights)
    torch.testing.assert_close(actual, expected)
