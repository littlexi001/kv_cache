from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_fractional_tail_spectral_rate_20260727 import (  # noqa: E402
    JL_RATE_OPTIONS,
    rademacher_projection,
    shared_envelope,
    shared_envelope_parameters,
)


def test_fixed_sharedtail_physical_rate_is_240_bits() -> None:
    core_codes = 2 * 16 * 4
    tail_signs = 64
    fp16_scales = 3 * 16
    total = core_codes + tail_signs + fp16_scales
    assert total == 240
    assert total / (2 * 128 * 16) == 0.05859375


def test_coordinate_amplitude_is_weighted_least_squares_optimum() -> None:
    generator = torch.Generator().manual_seed(20260727)
    coefficients = torch.randn(4096, 128, generator=generator)
    coordinate_rms, amplitude = shared_envelope_parameters(coefficients)
    envelope = (
        (coefficients / coordinate_rms)
        .square()
        .mean(dim=-1, keepdim=True)
        .sqrt()
    )
    target = coefficients.abs()
    residual = target - envelope * amplitude
    gradient = (envelope * residual).sum(dim=0)
    assert float(gradient.abs().max().item()) < 2.0e-2

    optimum = residual.square().sum()
    perturbation = torch.linspace(-0.25, 0.25, 128)
    alternative = (
        target - envelope * amplitude * (1.0 + perturbation)
    ).square().sum()
    assert float(optimum.item()) <= float(alternative.item())

    quantized_envelope = shared_envelope(coefficients, coordinate_rms)
    relative_envelope_error = (
        (quantized_envelope - envelope).square().mean().sqrt()
        / envelope.square().mean().sqrt()
    )
    assert float(relative_envelope_error.item()) < 5.0e-4


def test_common_key_shift_preserves_attention_distribution() -> None:
    generator = torch.Generator().manual_seed(13)
    query = torch.randn(128, generator=generator)
    keys = torch.randn(257, 128, generator=generator)
    key_mean = torch.randn(128, generator=generator)
    original = torch.softmax(keys @ query, dim=0)
    centered = torch.softmax((keys - key_mean) @ query, dim=0)
    torch.testing.assert_close(original, centered, atol=2.0e-6, rtol=2.0e-6)


def test_uniform_score_error_margin_certifies_topk() -> None:
    exact = torch.tensor((8.0, 7.0, 3.0, 2.0, 1.0))
    error = torch.tensor((-0.2, 0.1, 0.3, -0.3, 0.2))
    approximate = exact + error
    top_count = 2
    margin = exact[top_count - 1] - exact[top_count]
    epsilon = error.abs().max()
    assert margin > 2.0 * epsilon
    assert torch.equal(
        torch.topk(exact, top_count).indices.sort().values,
        torch.topk(approximate, top_count).indices.sort().values,
    )


def test_jl_tail_rate_matches_or_improves_240_bit_budget() -> None:
    core_bits = 2 * (16 * 4 + 16)
    totals = {
        (dimension, bits): core_bits + dimension * bits + 16
        for dimension, bits in JL_RATE_OPTIONS
    }
    assert totals[(8, 8)] == 240
    assert totals[(16, 4)] == 240
    assert totals[(32, 2)] == 240
    assert totals[(8, 4)] == 208
    assert totals[(16, 2)] == 208


def test_full_support_shared_sign_rate_accounting() -> None:
    shared_scale = 16
    assert 8 * 16 + shared_scale == 144
    assert (16 * 4 + 16) + 7 * 16 + shared_scale == 208
    assert (
        (16 * 4 + 16)
        + (16 * 2 + 16)
        + 6 * 16
        + shared_scale
        == 240
    )
    assert 3 * (16 * 2 + 16) + 5 * 16 + shared_scale == 240
    assert 8 * (16 + 16) == 256
    int8_core = 16 * 8 + 16
    assert int8_core + 4 * 16 + shared_scale == 224
    assert int8_core + 5 * 16 + shared_scale == 240
    assert int8_core + 6 * 16 + shared_scale == 256
    assert int8_core + 7 * 16 + shared_scale == 272


def test_rademacher_tail_score_is_unbiased_over_projections() -> None:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(96, generator=generator, dtype=torch.float64)
    key = torch.randn(96, generator=generator, dtype=torch.float64)
    exact = float(query @ key)
    estimates = []
    for seed in range(2048):
        projection = rademacher_projection(
            96,
            16,
            seed,
            torch.device("cpu"),
        ).double()
        estimates.append(float((query @ projection) @ (key @ projection)))
    empirical = sum(estimates) / len(estimates)
    standard_error = torch.tensor(estimates).std().item() / len(estimates) ** 0.5
    assert abs(empirical - exact) < 4.0 * standard_error
