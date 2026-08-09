from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_automatic_spectral_rate_allocation_20260727 import (  # noqa: E402
    GROUP_COUNT,
    GROUP_SIZE,
    ZERO_BIT_LEVELS,
    allocate_bits,
    quantize_band,
)
from analyze_qk_balanced_spectral_rate_20260727 import (  # noqa: E402
    boundary_scale_quantized_bands,
    distortion_table,
    feasible_allocations,
    joint_qmse_allocation,
    metric_scale_quantize_band,
    topk_boundary_weights,
)


def score_mse(
    coefficients: torch.Tensor,
    queries: torch.Tensor,
    allocation: tuple[int, ...],
) -> torch.Tensor:
    sampled = coefficients[::32]
    reconstructed = []
    for group, bits in enumerate(allocation):
        start = group * GROUP_SIZE
        stop = start + GROUP_SIZE
        reconstructed.append(quantize_band(sampled[:, start:stop], bits))
    approximate = torch.cat(reconstructed, dim=-1)
    score_error = queries @ (sampled - approximate).transpose(0, 1)
    return score_error.square().mean()


def test_feasible_joint_rate_space_is_complete() -> None:
    allocations = feasible_allocations(15)
    assert len(allocations) == 13_817
    assert all(len(allocation) == GROUP_COUNT for allocation in allocations)
    assert all(
        sum(bits + int(bits > 0) for bits in allocation) <= 15
        for allocation in allocations
    )


def test_joint_qmse_never_worse_than_additive_qmse_on_calibration() -> None:
    generator = torch.Generator().manual_seed(20260727)
    coefficients = torch.randn(
        257,
        GROUP_COUNT * GROUP_SIZE,
        generator=generator,
    )
    queries = torch.randn(
        11,
        GROUP_COUNT * GROUP_SIZE,
        generator=generator,
    )
    additive = allocate_bits(
        distortion_table(coefficients, queries),
        15,
        ZERO_BIT_LEVELS,
        include_scale_metadata=True,
    )
    joint = joint_qmse_allocation(coefficients, queries, 15)
    assert sum(bits + int(bits > 0) for bits in joint) <= 15
    assert float(score_mse(coefficients, queries, joint).item()) <= (
        float(score_mse(coefficients, queries, additive).item()) + 1.0e-5
    )


def test_metric_optimal_scale_reduces_calibration_score_mse() -> None:
    generator = torch.Generator().manual_seed(91)
    values = torch.randn(193, GROUP_SIZE, generator=generator)
    queries = torch.randn(13, GROUP_SIZE, generator=generator)
    queries[:, :4] *= 5.0
    for bits in (1, 2, 4, 8):
        baseline = quantize_band(values, bits)
        optimized = metric_scale_quantize_band(values, bits, queries)
        baseline_error = (
            queries @ (values - baseline).transpose(0, 1)
        ).square().mean()
        optimized_error = (
            queries @ (values - optimized).transpose(0, 1)
        ).square().mean()
        assert float(optimized_error.item()) <= (
            float(baseline_error.item()) + 1.0e-5
        )


def test_lsq_and_diagonal_metric_scales_optimize_their_objectives() -> None:
    generator = torch.Generator().manual_seed(711)
    values = torch.randn(193, GROUP_SIZE, generator=generator)
    queries = torch.randn(13, GROUP_SIZE, generator=generator)
    variances = queries.square().mean(dim=0)
    for bits in (1, 2, 4, 8):
        baseline = quantize_band(values, bits)
        lsq = metric_scale_quantize_band(
            values,
            bits,
            queries,
            "identity",
        )
        diagonal = metric_scale_quantize_band(
            values,
            bits,
            queries,
            "diagonal",
        )
        assert float((values - lsq).square().mean().item()) <= (
            float((values - baseline).square().mean().item()) + 1.0e-6
        )
        baseline_diagonal = (
            (values - baseline).square() * variances
        ).mean()
        optimized_diagonal = (
            (values - diagonal).square() * variances
        ).mean()
        assert float(optimized_diagonal.item()) <= (
            float(baseline_diagonal.item()) + 1.0e-6
        )


def test_boundary_scale_reduces_boundary_weighted_score_error() -> None:
    generator = torch.Generator().manual_seed(901)
    coefficients = torch.randn(
        257,
        GROUP_COUNT * GROUP_SIZE,
        generator=generator,
    )
    queries = torch.randn(
        11,
        GROUP_COUNT * GROUP_SIZE,
        generator=generator,
    )
    weights = topk_boundary_weights(
        coefficients,
        queries,
        top_fraction=0.06,
        include_global_floor=True,
    )
    assert weights.shape == (11, 257)
    torch.testing.assert_close(
        weights.mean(dim=-1),
        torch.full((11,), 2.0),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    bands = boundary_scale_quantized_bands(
        coefficients,
        queries,
        weights,
    )
    for group in range(GROUP_COUNT):
        start = group * GROUP_SIZE
        stop = start + GROUP_SIZE
        values = coefficients[:, start:stop]
        query_band = queries[:, start:stop]
        for bits in (1, 2, 4, 8):
            baseline = quantize_band(values, bits)
            baseline_error = query_band @ (
                values - baseline
            ).transpose(0, 1)
            optimized_error = query_band @ (
                values - bands[group][bits]
            ).transpose(0, 1)
            baseline_cost = (
                weights * baseline_error.square()
            ).sum()
            optimized_cost = (
                weights * optimized_error.square()
            ).sum()
            assert float(optimized_cost.item()) <= (
                float(baseline_cost.item()) + 1.0e-3
            )
