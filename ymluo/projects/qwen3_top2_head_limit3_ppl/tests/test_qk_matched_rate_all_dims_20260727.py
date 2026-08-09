from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_automatic_spectral_rate_allocation_20260727 import (
    GROUP_SIZE,
)
from analyze_qk_matched_rate_all_dims_20260727 import (
    MATCHED_CONFIGS,
    allocation_rate,
    diagonal_covariance_samples,
)


def test_uniform_allocations_have_exact_declared_rates() -> None:
    for allocation, declared_rate in MATCHED_CONFIGS.values():
        assert allocation_rate(allocation) == declared_rate


def test_matched_rates_include_fp16_per_band_scale_metadata() -> None:
    int1, _ = MATCHED_CONFIGS["uniform_all_dims_int1"]
    int2, _ = MATCHED_CONFIGS["uniform_all_dims_int2"]
    int4, _ = MATCHED_CONFIGS["uniform_all_dims_int4"]

    assert GROUP_SIZE * allocation_rate(int1) == 256
    assert GROUP_SIZE * allocation_rate(int2) == 384
    assert GROUP_SIZE * allocation_rate(int4) == 640


def test_manual_hierarchical_allocations_include_tail_cost() -> None:
    tail0, _ = MATCHED_CONFIGS["manual_head8_mid4_tail0"]
    tail1, _ = MATCHED_CONFIGS["manual_head8_mid4_tail1"]
    tail2, _ = MATCHED_CONFIGS["manual_head8_mid4_tail2"]

    assert GROUP_SIZE * allocation_rate(tail0) == 304
    assert GROUP_SIZE * allocation_rate(tail1) == 464
    assert GROUP_SIZE * allocation_rate(tail2) == 544


def test_fixed_240_bit_patterns_are_exactly_rate_matched() -> None:
    for name in (
        "fixed_spectral_4421",
        "fixed_spectral_4440",
        "fixed_spectral_8111",
    ):
        allocation, declared_rate = MATCHED_CONFIGS[name]
        assert declared_rate == 15
        assert GROUP_SIZE * allocation_rate(allocation) == 240


def test_diagonal_covariance_samples_match_target_exactly() -> None:
    diagonal = torch.tensor([0.25, 1.0, 4.0, 9.0])
    samples = diagonal_covariance_samples(diagonal)
    covariance = samples.transpose(0, 1) @ samples / samples.shape[0]

    assert torch.allclose(covariance, torch.diag(diagonal))
