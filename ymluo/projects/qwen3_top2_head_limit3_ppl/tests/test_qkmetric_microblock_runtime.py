from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    _refresh_qkmetric_microblock_summaries,
)


def test_microblock_summaries_match_direct_statistics_and_update() -> None:
    torch.manual_seed(7)
    key = torch.randn(1, 2, 11, 8)
    basis = torch.randn(1, 2, 8, 4)
    state = {
        "basis": basis,
        "capacity": 32,
        "qk_metric_rebuild_count": 1,
    }
    _refresh_qkmetric_microblock_summaries(key, state, 4, 4)

    projected = torch.einsum("bhnd,bhdr->bhnr", key, basis)
    for block, (start, end) in enumerate(((0, 4), (4, 8), (8, 11))):
        direct = projected[..., start:end, :]
        assert torch.allclose(
            state["microblock_mean"][..., block, :],
            direct.mean(dim=2),
            atol=1.0e-5,
        )
        assert torch.allclose(
            state["microblock_variance"][..., block, :],
            direct.var(dim=2, correction=0),
            atol=1.0e-5,
        )

    extended = torch.cat((key, torch.randn(1, 2, 2, 8)), dim=2)
    _refresh_qkmetric_microblock_summaries(extended, state, 4, 4)
    extended_projected = torch.einsum("bhnd,bhdr->bhnr", extended, basis)
    assert state["microblock_summary_indexed_count"] == 13
    assert torch.allclose(
        state["microblock_mean"][..., 2, :],
        extended_projected[..., 8:12, :].mean(dim=2),
        atol=1.0e-5,
    )
    assert torch.allclose(
        state["microblock_mean"][..., 3, :],
        extended_projected[..., 12:13, :].mean(dim=2),
        atol=1.0e-5,
    )


def test_quantized_microblock_summaries_respect_rowwise_error_bounds() -> None:
    torch.manual_seed(11)
    key = torch.randn(1, 2, 13, 8)
    basis = torch.randn(1, 2, 8, 4)
    state = {
        "basis": basis,
        "capacity": 32,
        "qk_metric_rebuild_count": 1,
    }
    _refresh_qkmetric_microblock_summaries(
        key, state, 4, 4, quantized=True
    )

    assert "microblock_mean" not in state
    assert "microblock_variance" not in state
    mean = (
        state["microblock_mean_q8"].float()
        * state["microblock_mean_scales"].float()
    )
    variance = (
        state["microblock_variance_q8"].float()
        * state["microblock_variance_scales"].float()
    )
    projected = torch.einsum("bhnd,bhdr->bhnr", key, basis)
    for block, (start, end) in enumerate(((0, 4), (4, 8), (8, 12), (12, 13))):
        direct = projected[..., start:end, :]
        direct_mean = direct.mean(dim=2)
        direct_variance = direct.var(dim=2, correction=0)
        mean_bound = state["microblock_mean_scales"][..., block, 0].float() / 2
        variance_bound = (
            state["microblock_variance_scales"][..., block, 0].float() / 2
        )
        assert torch.all(
            (mean[..., block, :] - direct_mean).abs()
            <= mean_bound.unsqueeze(-1) + 1.0e-6
        )
        assert torch.all(
            (variance[..., block, :] - direct_variance).abs()
            <= variance_bound.unsqueeze(-1) + 1.0e-6
        )
