from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_qksieve_certified_band_refinement_20260801 import (
    certified_bound,
    interval_candidate_mask,
    upward_block_quantize,
)


def test_upward_block_quantization_never_rounds_down() -> None:
    generator = torch.Generator().manual_seed(13)
    values = torch.rand(513, generator=generator).square() * 100
    values[::19] = 0
    for bits in (2, 4, 8, 16):
        for block_size in (17, 128, 1024):
            quantized = upward_block_quantize(
                values, bits=bits, block_size=block_size
            )
            assert torch.all(quantized >= values)
            assert torch.all(quantized[values == 0] == 0)


def test_certified_bounds_cover_residual_dot_products() -> None:
    generator = torch.Generator().manual_seed(29)
    residual = torch.randn(211, 128, generator=generator)
    query = torch.randn(128, generator=generator)
    omitted_bands = (1, 3, 6)
    keep = torch.zeros(128, dtype=torch.bool)
    for band in omitted_bands:
        keep[band * 16 : (band + 1) * 16] = True
    residual[:, ~keep] = 0
    error = (residual @ query).abs()

    for mode in ("global", "bandwise"):
        bound, streams = certified_bound(
            residual,
            query,
            omitted_bands,
            mode=mode,
            norm_bits=4,
            norm_block_size=32,
        )
        assert torch.all(bound + 1.0e-5 >= error)
        assert streams == (1 if mode == "global" else 3)


def test_certified_intervals_contain_full_topk() -> None:
    generator = torch.Generator().manual_seed(31)
    base = torch.randn(401, generator=generator)
    residual = 0.3 * torch.randn(401, generator=generator)
    full = base + residual
    candidate, _ = interval_candidate_mask(base, residual.abs(), top_count=17)
    full_top = torch.topk(full, k=17).indices
    assert candidate[full_top].all()
