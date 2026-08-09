from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_head_top2_targeted_ppl_20260714 as runner  # noqa: E402


def test_proxy_selected_mass_includes_current_token() -> None:
    proxy_scores = torch.tensor([[[[math.log(3.0), 0.0]]]])
    query = torch.zeros(1, 1, 2)
    current_key = torch.zeros(1, 1, 2)
    candidate_indices = torch.tensor([[[0]]])
    candidate_counts = torch.ones(1, 1, dtype=torch.long)

    mass = runner._proxy_selected_attention_mass(
        proxy_scores,
        query,
        current_key,
        candidate_indices,
        candidate_counts,
        scaling=1.0,
    )

    assert torch.allclose(mass, torch.tensor([[0.8]]), atol=1.0e-6)


def test_mean_value_score_mode_is_explicit() -> None:
    state: dict[str, object] = {}
    mode = (
        "pca_hierarchical_autoqmsetotal15z_qkmetric_"
        "meanvalue_packed_fulltopk"
    )
    runner._configure_packed_qmse_state(state, mode)

    assert state["packed_qmse_full_topk"]
    assert state["packed_qmse_mean_value_correction"]


def test_mean_value_correction_broadcasts_gqa_heads() -> None:
    sparse_output = torch.tensor([[[[2.0], [4.0]]]])
    value = torch.tensor([[[[0.0], [2.0]]]])
    approximate_mass = torch.tensor([[0.25, 0.5]])

    corrected = runner._mean_value_corrected_output(
        sparse_output,
        value,
        approximate_mass,
        query_head_count=2,
        state={},
    )

    expected = torch.tensor([[[[1.25], [2.5]]]])
    assert torch.allclose(corrected, expected)


def test_stratified_exact_top_mass_matches_complete_toy_history() -> None:
    query = torch.ones(1, 1, 1)
    key_history = torch.tensor(
        [[[[math.log(3.0)], [0.0], [0.0], [0.0]]]]
    )
    current_key = torch.zeros(1, 1, 1)

    mass = runner._stratified_exact_top_mass(
        query,
        key_history,
        current_key,
        sample_count=4,
        selected_keep=1,
        scaling=1.0,
    )

    assert torch.allclose(mass, torch.tensor([[4.0 / 7.0]]), atol=1.0e-6)


def test_sampled_mass_score_mode_is_fast_and_explicit() -> None:
    state: dict[str, object] = {}
    mode = (
        "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
        "kappend_samplemass_unbiased_packed_direct"
    )
    runner._configure_packed_qmse_state(state, mode)

    assert mode in runner._QKSIEVE_FAST_RUNTIME_MODES
    assert state["packed_qmse_sampled_mass_correction"]
    assert not state["packed_qmse_full_topk"]


def test_proxy_mass_score_mode_is_fast_and_excludes_exact_sampler() -> None:
    state: dict[str, object] = {}
    mode = (
        "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
        "kappend_proxymass_unbiased_packed_direct"
    )
    runner._configure_packed_qmse_state(state, mode)

    assert mode in runner._QKSIEVE_FAST_RUNTIME_MODES
    assert state["packed_qmse_proxy_mass_correction"]
    assert not state["packed_qmse_sampled_mass_correction"]
    assert not state["packed_qmse_full_topk"]
