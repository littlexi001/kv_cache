from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    _hierarchical_metric_scale_quantize_bands,
    _hierarchical_quantize_band,
)
from analyze_qk_balanced_spectral_rate_20260727 import (  # noqa: E402
    boundary_scale_quantized_bands,
    fit_empirical_bayes_score_bias,
    fit_token_score_affine,
    quantize_token_score_metadata,
    query_int8,
    topk_boundary_weights,
)


def test_qmetric_scale_reduces_batched_calibration_score_error() -> None:
    generator = torch.Generator().manual_seed(20260727)
    values = torch.randn(1, 2, 193, 8, 16, generator=generator)
    queries = torch.randn(1, 2, 17, 8, 16, generator=generator)
    queries[..., :5] *= 3.0
    metrics = torch.einsum(
        "bhqgd,bhqge->bhgde",
        queries,
        queries,
    ) / queries.shape[2]
    for bits in (1, 2, 4, 8):
        baseline = _hierarchical_quantize_band(values, bits)
        optimized = _hierarchical_metric_scale_quantize_bands(
            values,
            bits,
            metrics,
        )
        baseline_residual = values - baseline
        optimized_residual = values - optimized
        baseline_cost = torch.einsum(
            "bhkgd,bhgde,bhkge->",
            baseline_residual,
            metrics,
            baseline_residual,
        )
        optimized_cost = torch.einsum(
            "bhkgd,bhgde,bhkge->",
            optimized_residual,
            metrics,
            optimized_residual,
        )
        assert float(optimized_cost.item()) <= (
            float(baseline_cost.item()) + 1.0e-4
        )


def test_boundary_scale_reduces_boundary_weighted_score_error() -> None:
    generator = torch.Generator().manual_seed(20260728)
    keys = torch.randn(257, 128, generator=generator)
    queries = torch.randn(9, 128, generator=generator)
    weights = topk_boundary_weights(
        keys,
        queries,
        top_fraction=0.05,
        include_global_floor=False,
    )
    quantized = boundary_scale_quantized_bands(
        keys,
        queries,
        weights,
    )

    assert weights.shape == (9, 257)
    assert bool(torch.isfinite(weights).all())
    assert bool((weights >= 0).all())
    assert bool((weights.sum(dim=-1) > 0).all())

    band_keys = keys[:, :16]
    band_queries = queries[:, :16]
    exact_scores = band_queries @ band_keys.transpose(0, 1)
    baseline_keys = _hierarchical_quantize_band(
        band_keys.reshape(1, 1, 257, 1, 16),
        2,
    ).reshape(257, 16)
    baseline_error = (
        weights
        * (
            band_queries @ baseline_keys.transpose(0, 1)
            - exact_scores
        ).square()
    ).sum()
    boundary_error = (
        weights
        * (
            band_queries @ quantized[0][2].transpose(0, 1)
            - exact_scores
        ).square()
    ).sum()
    assert float(boundary_error.item()) <= (
        float(baseline_error.item()) + 1.0e-4
    )


def test_token_affine_reduces_calibration_score_error() -> None:
    generator = torch.Generator().manual_seed(20260729)
    keys = torch.randn(193, 128, generator=generator)
    queries = torch.randn(24, 128, generator=generator)
    reconstructed = keys * 0.82 + 0.05 * torch.randn(
        keys.shape,
        generator=generator,
    )
    gain, bias = fit_token_score_affine(
        keys,
        reconstructed,
        queries,
        ridge=0.1,
        fit_bias=True,
    )
    exact = queries @ keys.transpose(0, 1)
    proxy = torch.stack(
        [query_int8(query) for query in queries],
        dim=0,
    ) @ reconstructed.transpose(0, 1)
    corrected = proxy * gain + bias
    assert gain.shape == (193,)
    assert bias.shape == (193,)
    assert float((corrected - exact).square().mean().item()) < float(
        (proxy - exact).square().mean().item()
    )

    gain8, bias8 = quantize_token_score_metadata(
        gain,
        bias,
        bits=8,
        include_gain=True,
    )
    gain4, bias4 = quantize_token_score_metadata(
        gain,
        bias,
        bits=4,
        include_gain=True,
    )
    assert gain8.shape == gain.shape
    assert bias8.shape == bias.shape
    assert float((gain8 - gain).square().mean().item()) <= float(
        (gain4 - gain).square().mean().item()
    )
    assert float((bias8 - bias).square().mean().item()) <= float(
        (bias4 - bias).square().mean().item()
    )

    eb_gain, eb_bias, shrinkage = fit_empirical_bayes_score_bias(
        keys,
        reconstructed,
        queries,
    )
    assert eb_gain.shape == gain.shape
    assert eb_bias.shape == bias.shape
    assert 0.0 <= shrinkage <= 1.0
