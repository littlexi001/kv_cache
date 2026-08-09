from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    _pca_int4_partial_scores,
    _residual_sentinel_candidates,
    qabs_sampled_head_adaptive_attention,
)


def exact_gqa_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    groups = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(groups, dim=1)
    return torch.einsum("bhd,bhkd->bhk", query.float(), expanded_key.float())


def decoded_radius(state: dict[str, object], count: int) -> torch.Tensor:
    codes = state["error_radius_codes"]
    scale = state["error_radius_scale"]
    assert isinstance(codes, torch.Tensor)
    assert isinstance(scale, torch.Tensor)
    return codes[..., :count, :].float() * scale.float()


def test_quantized_single_radius_is_a_valid_score_upper_bound() -> None:
    torch.manual_seed(41)
    key = torch.randn(1, 2, 96, 128)
    query = torch.randn(1, 8, 128)
    state: dict[str, object] = {}

    approximate = _pca_int4_partial_scores(
        query, key, state, projection_dim=64, build_error_radius=True
    )
    upper = state["last_residual_upper_scores"]

    assert isinstance(upper, torch.Tensor)
    assert approximate.shape == upper.shape == (1, 8, 96)
    assert bool((exact_gqa_scores(query, key) <= upper + 2.0e-4).all())
    assert state["error_radius_codes"].dtype == torch.uint8


def test_radius_scale_grows_conservatively_for_streamed_outlier() -> None:
    torch.manual_seed(43)
    key = 0.01 * torch.randn(1, 1, 96, 128)
    query = torch.randn(1, 4, 128)
    state: dict[str, object] = {}

    _pca_int4_partial_scores(
        query, key, state, projection_dim=64, build_error_radius=True
    )
    old_scale = state["error_radius_scale"].clone()
    old_radius = decoded_radius(state, key.shape[2]).clone()

    extended_key = torch.cat((key, 100.0 * torch.randn(1, 1, 1, 128)), dim=2)
    _pca_int4_partial_scores(
        query,
        extended_key,
        state,
        projection_dim=64,
        build_error_radius=True,
    )
    new_scale = state["error_radius_scale"]
    new_radius = decoded_radius(state, extended_key.shape[2])
    upper = state["last_residual_upper_scores"]

    assert bool((new_scale >= old_scale).all())
    assert bool((new_scale > old_scale).any())
    assert bool((new_radius[..., : key.shape[2], :] + 1.0e-7 >= old_radius).all())
    assert bool((exact_gqa_scores(query, extended_key) <= upper + 2.0e-3).all())


def test_residual_sentinel_union_keeps_every_primary_candidate() -> None:
    approximate = torch.tensor([[[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]]])
    upper = torch.tensor([[[9.0, 8.0, 7.0, 7.0, 12.0, 11.0]]])

    candidates, primary = _residual_sentinel_candidates(
        approximate, upper, primary_count=3, candidate_count=5
    )

    assert candidates.shape == (1, 1, 5)
    assert primary.shape == (1, 1, 3)
    assert set(primary.flatten().tolist()) == {0, 1, 2}
    assert set(candidates.flatten().tolist()) == {0, 1, 2, 4, 5}
    assert torch.unique(candidates).numel() == candidates.numel()


def test_residual_sentinel_attention_cpu_reference_is_finite() -> None:
    torch.manual_seed(47)
    query = torch.randn(1, 4, 1, 128)
    key = torch.randn(1, 1, 101, 128)
    value = torch.randn(1, 1, 101, 128)
    state: dict[str, object] = {}
    diagnostics: dict[str, object] = {}

    output, indices = qabs_sampled_head_adaptive_attention(
        query,
        key,
        value,
        attention_mask=None,
        scaling=128.0**-0.5,
        mass_threshold=0.9,
        budget_fractions=(0.02,),
        sample_fraction=0.1,
        qabs_dim_count=8,
        candidate_fraction=0.03,
        diagnostics=diagnostics,
        score_mode="pca_int4_residual_sentinel",
        projection_dim=64,
        pca_state=state,
    )

    assert output.shape == (1, 1, 4, 128)
    assert indices.shape == (1, 4, 1, 3)
    assert torch.isfinite(output).all()
    assert int(state["indexed_count"]) == 100
    assert state["error_radius_codes"].dtype == torch.uint8
    assert diagnostics["index_mode"] == "pca_int4_residual_sentinel"
