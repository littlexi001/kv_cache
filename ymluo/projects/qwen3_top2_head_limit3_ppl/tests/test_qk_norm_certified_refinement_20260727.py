from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_qk_norm_certified_refinement_20260727 import (
    conservative_log_quantize,
    residual_norm_bound,
)


def test_conservative_log_quantization_never_rounds_down() -> None:
    values = torch.logspace(-4, 3, 1000)
    values[::17] = 0
    for bits in (2, 4, 8):
        quantized = conservative_log_quantize(values, bits)
        assert torch.all(quantized >= values)
        assert torch.all(quantized[values == 0] == 0)


def test_global_norm_bound_covers_all_dot_products() -> None:
    generator = torch.Generator().manual_seed(11)
    residual = torch.randn(31, 128, generator=generator)
    query = torch.randn(128, generator=generator)
    bound, code_bits, metadata_bits = residual_norm_bound(
        residual,
        query,
        tuple(range(8)),
        bits=4,
        mode="global",
    )
    error = (residual @ query).abs()

    assert torch.all(bound + 1.0e-5 >= error)
    assert code_bits == 4
    assert metadata_bits == 32


def test_per_band_bound_is_no_looser_than_global_bound() -> None:
    generator = torch.Generator().manual_seed(19)
    residual = torch.randn(47, 128, generator=generator)
    query = torch.randn(128, generator=generator)
    bands = (1, 3, 5)
    mask = torch.zeros(128, dtype=torch.bool)
    for band in bands:
        mask[band * 16 : (band + 1) * 16] = True
    residual[:, ~mask] = 0
    global_bound, _, _ = residual_norm_bound(
        residual,
        query,
        bands,
        bits=16,
        mode="global",
    )
    per_band_bound, code_bits, metadata_bits = residual_norm_bound(
        residual,
        query,
        bands,
        bits=16,
        mode="per_band",
    )

    assert torch.all(per_band_bound <= global_bound + 1.0e-4)
    assert torch.all(per_band_bound + 1.0e-5 >= (residual @ query).abs())
    assert code_bits == 48
    assert metadata_bits == 96
