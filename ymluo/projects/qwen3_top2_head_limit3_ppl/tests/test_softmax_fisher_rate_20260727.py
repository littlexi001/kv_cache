from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_qk_balanced_spectral_rate_20260727 import (  # noqa: E402
    softmax_fisher_cost,
)


def test_softmax_fisher_cost_ignores_common_logit_shift() -> None:
    generator = torch.Generator().manual_seed(20260727)
    logits = torch.randn(5, 31, generator=generator, dtype=torch.float64)
    attention = torch.softmax(logits, dim=-1)
    common_error = torch.randn(
        5, 1, generator=generator, dtype=torch.float64
    ).expand_as(logits)
    cost = softmax_fisher_cost(common_error, attention)
    assert float(cost.item()) < 1.0e-12


def test_softmax_fisher_cost_is_local_kl_quadratic() -> None:
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(7, 43, generator=generator, dtype=torch.float64)
    error = torch.randn(7, 43, generator=generator, dtype=torch.float64)
    attention = torch.softmax(logits, dim=-1)
    fisher = softmax_fisher_cost(error, attention)
    step = 1.0e-3
    perturbed = torch.softmax(logits + step * error, dim=-1)
    kl = (
        attention
        * (attention.clamp_min(1.0e-30).log() - perturbed.log())
    ).sum(dim=-1).mean()
    approximation = 0.5 * step * step * fisher
    torch.testing.assert_close(kl, approximation, rtol=2.0e-3, atol=1.0e-12)
