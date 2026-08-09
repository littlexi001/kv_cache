from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    _pack_projected_binary_sign,
    _pack_projected_int2_uniform4,
    _pca_int4_adaptive_rank_partial_scores,
    _pca_int4_partial_scores,
    _record_adaptive_rank_diagnostics,
    _unpack_projected_binary_sign,
    _unpack_projected_int2_uniform4,
)


def test_low_threshold_adaptive_rank_matches_fixed_pca64() -> None:
    torch.manual_seed(17)
    key = torch.randn(1, 2, 96, 128, dtype=torch.float32)
    query = torch.randn(1, 8, 128, dtype=torch.float32)

    fixed = _pca_int4_partial_scores(query, key, {}, projection_dim=64)
    adaptive_state: dict[str, object] = {}
    adaptive = _pca_int4_adaptive_rank_partial_scores(
        query,
        key,
        adaptive_state,
        maximum_projection_dim=128,
        energy_threshold=1.0e-6,
    )

    assert torch.allclose(adaptive, fixed, atol=2.0e-4, rtol=2.0e-4)
    selected = adaptive_state["last_selected_projection_rank"]
    assert isinstance(selected, torch.Tensor)
    assert bool((selected == 64).all())


def test_high_threshold_uses_nested_rank_ladder() -> None:
    torch.manual_seed(19)
    key = torch.randn(1, 1, 64, 128, dtype=torch.float32)
    query = torch.randn(1, 4, 128, dtype=torch.float32)
    state: dict[str, object] = {}

    scores = _pca_int4_adaptive_rank_partial_scores(
        query,
        key,
        state,
        maximum_projection_dim=128,
        energy_threshold=0.9,
    )
    selected = state["last_selected_projection_rank"]
    coverage = state["last_selected_projection_coverage"]

    assert scores.shape == (1, 4, 64)
    assert isinstance(selected, torch.Tensor)
    assert isinstance(coverage, torch.Tensor)
    assert set(selected.flatten().tolist()).issubset({64, 80, 96, 112, 128})
    assert bool((coverage >= 0.0).all() and (coverage <= 1.0).all())

    diagnostics: dict[str, object] = {}
    _record_adaptive_rank_diagnostics(
        diagnostics, "pca_int4_adaptive_rank", state
    )
    assert 64.0 <= float(diagnostics["selected_projection_rank_mean"]) <= 128.0
    assert 64.0 <= float(diagnostics["selected_projection_rank_max"]) <= 128.0
    assert 0.0 <= float(diagnostics["selected_projection_coverage_mean"]) <= 1.0


def test_int2_residual_pack_round_trip_and_adaptive_scores() -> None:
    values = torch.tensor(
        [[-1.0, -0.25, 0.25, 1.0], [0.2, -0.4, 0.8, -0.6]],
        dtype=torch.float32,
    )
    packed, scale = _pack_projected_int2_uniform4(values)
    restored = _unpack_projected_int2_uniform4(packed, scale, values.dtype)

    assert packed.shape == (2, 1)
    assert restored.shape == values.shape
    assert torch.isfinite(restored).all()

    torch.manual_seed(23)
    key = torch.randn(1, 1, 64, 128)
    query = torch.randn(1, 4, 128)
    state: dict[str, object] = {}
    scores = _pca_int4_adaptive_rank_partial_scores(
        query,
        key,
        state,
        maximum_projection_dim=72,
        energy_threshold=0.8,
        residual_precision="int2_uniform4",
    )
    assert scores.shape == (1, 4, 64)
    assert torch.isfinite(scores).all()


def test_binary_residual_pack_round_trip_and_adaptive_scores() -> None:
    values = torch.tensor(
        [
            [-3.0, 2.0, -1.0, 4.0, 5.0, -2.0, 7.0, -8.0],
            [1.0, -4.0, 3.0, -2.0, -5.0, 6.0, -7.0, 8.0],
        ],
        dtype=torch.float32,
    )
    packed, scale = _pack_projected_binary_sign(values)
    restored = _unpack_projected_binary_sign(packed, scale, values.dtype)

    assert packed.shape == (2, 1)
    assert scale.shape == (1, 8)
    assert torch.equal(restored.sign(), values.sign())
    assert torch.allclose(restored.abs(), values.abs().mean(dim=0).expand_as(values))

    torch.manual_seed(29)
    key = torch.randn(1, 1, 64, 128)
    query = torch.randn(1, 4, 128)
    state: dict[str, object] = {}
    scores = _pca_int4_adaptive_rank_partial_scores(
        query,
        key,
        state,
        maximum_projection_dim=96,
        energy_threshold=0.8,
        residual_precision="binary_sign",
        residual_group_dim=32,
    )
    assert scores.shape == (1, 4, 64)
    assert torch.isfinite(scores).all()

    original_scale = state["adaptive_rank_scales"][1].clone()
    extended_key = torch.cat((key, torch.randn(1, 1, 3, 128)), dim=2)
    extended_scores = _pca_int4_adaptive_rank_partial_scores(
        query,
        extended_key,
        state,
        maximum_projection_dim=96,
        energy_threshold=0.8,
        residual_precision="binary_sign",
    )
    assert extended_scores.shape == (1, 4, 67)
    assert torch.equal(state["adaptive_rank_scales"][1], original_scale)
