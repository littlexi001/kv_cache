from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    _pack_projected_int4_logscale16,
)


def test_logscale16_packs_odd_number_of_bands() -> None:
    torch.manual_seed(7)
    projected = torch.randn(1, 2, 5, 48, dtype=torch.float32)

    packed, base_scale, packed_exponents = _pack_projected_int4_logscale16(
        projected
    )

    assert packed.shape == (1, 2, 5, 24)
    assert base_scale.shape == (1, 2, 5, 1)
    assert packed_exponents.shape == (1, 2, 5, 2)
    assert bool(((packed_exponents[..., -1] >> 4) == 0).all())

    exponent = torch.empty((1, 2, 5, 3), dtype=torch.uint8)
    exponent[..., 0::2] = packed_exponents & 0x0F
    exponent[..., 1::2] = packed_exponents[..., :1] >> 4
    scales = base_scale.unsqueeze(-2) * torch.exp2(
        -0.25 * exponent.float().unsqueeze(-1)
    )
    low = (packed & 0x0F).float() - 7.0
    high = (packed >> 4).float() - 7.0
    codes = torch.stack((low, high), dim=-1).flatten(-2).reshape(
        1, 2, 5, 3, 16
    )
    restored = (codes * scales).flatten(-2)

    error = (restored - projected).abs()
    # Quarter-octave scale rounding can add at most about 0.13 scale units
    # beyond nearest-integer rounding when an endpoint clips at INT4 range.
    assert bool((error <= 0.64 * scales.expand_as(codes).flatten(-2)).all())
