from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_head_top2_targeted_ppl_20260714 as runner  # noqa: E402
import run_qksieve_coldskip_longcontext_quality_20260730 as quality  # noqa: E402


SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch8i4shared_wometric_blockcondres8b256m8global_"
    "packed_fulltopk_oas"
)


def test_block_conditional_score_mode_contract() -> None:
    state: dict[str, object] = {}

    runner._configure_packed_qmse_state(state, SCORE_MODE)

    assert SCORE_MODE in runner._PACKED_QMSE_SCORE_MODES
    assert state["packed_qmse_value_sketch_rank"] == 8
    assert state["packed_qmse_value_sketch_bits"] == 4
    assert state["packed_qmse_block_conditional_residual_dim"] == 8
    assert state["packed_qmse_block_conditional_residual_block_size"] == 256
    assert state["packed_qmse_block_conditional_residual_moment_bits"] == 8
    assert state["packed_qmse_conditional_value_residual_dim"] == 0


def test_block_conditional_score_mode_is_registered_in_ppl_runner() -> None:
    source = (
        PROJECT_ROOT / "src/run_direct_countcap_denseprompt_ppl_20260725.py"
    ).read_text(encoding="utf-8")
    normalized = "".join(source.replace('"', "").split())

    assert SCORE_MODE in normalized


def test_block_conditional_variant_uses_frozen_dynamic_budget() -> None:
    variant = "qksieve_qmse_oas_requestlocal_blockcondres8_r8_m8_k1120"

    assert quality.VARIANTS[variant] == SCORE_MODE
    assert quality.variant_attention_token_budget(variant, 4096) == 256
    assert quality.variant_attention_token_budget(variant, 32768) == 1120
    assert quality.variant_attention_token_budget(variant, 131072) == 1120
    assert quality.variant_attention_token_budget(
        variant.replace("k1120", "k1280"), 65536
    ) == 1280
    assert quality.variant_attention_token_budget(
        variant.replace("k1120", "k2560"), 65536
    ) == 2560


def test_block_moment_int8_quantization_uses_vector_local_scale() -> None:
    values = torch.tensor(
        [[[[1.0, -0.5, 0.25], [100.0, -50.0, 25.0]]]]
    )

    quantized = runner._quantize_qksieve_block_moment(values, 8, (-1,))

    torch.testing.assert_close(quantized[..., 0, 0], values[..., 0, 0])
    torch.testing.assert_close(quantized[..., 1, 0], values[..., 1, 0])
    assert float((quantized[..., 0, :] - values[..., 0, :]).abs().max()) < 0.01
    assert float((quantized[..., 1, :] - values[..., 1, :]).abs().max()) < 1.0


def test_block_conditional_correction_removes_selected_token_from_moments() -> None:
    grouped_tail_weights = torch.tensor([[[[1.0, 0.0, 1.0, 3.0]]]])
    coordinates = torch.tensor([[[[0.0], [2.0], [10.0], [14.0]]]])
    block_coordinate_means = torch.tensor([[[[1.0], [12.0]]]])
    block_residual_means = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]]]]
    )
    block_counts = torch.tensor([[[2.0, 2.0]]])
    linear_map = torch.tensor([[[[2.0], [-1.0]]]])
    candidate_indices = torch.tensor([[[1]]])
    candidate_counts = torch.tensor([[1]])
    selected_residual = torch.tensor([[[[2.0, 4.0]]]])

    correction = runner._block_conditional_value_residual_correction(
        grouped_tail_weights,
        coordinates,
        block_coordinate_means,
        block_residual_means,
        block_counts,
        linear_map,
        candidate_indices,
        candidate_counts,
        selected_residual,
        block_size=2,
    )

    # Block 0 contributes zero after removing token 1.  Block 1 contributes
    # 4*[3,4] plus B*(52 - 4*12) = [12,16] + [8,-4].
    torch.testing.assert_close(correction, torch.tensor([[[20.0, 12.0]]]))


def test_block_conditional_correction_broadcasts_gqa_groups() -> None:
    grouped_tail_weights = torch.tensor(
        [[[[1.0, 1.0]], [[2.0, 0.0]]]]
    ).reshape(1, 1, 2, 2)
    coordinates = torch.tensor([[[[0.0], [2.0]]]])
    block_coordinate_means = torch.tensor([[[[1.0]]]])
    block_residual_means = torch.tensor([[[[3.0]]]])
    block_counts = torch.tensor([[[2.0]]])
    linear_map = torch.tensor([[[[2.0]]]])
    candidate_indices = torch.zeros(1, 2, 1, dtype=torch.long)
    candidate_counts = torch.zeros(1, 2, dtype=torch.long)
    selected_residual = torch.zeros(1, 2, 1, 1)

    correction = runner._block_conditional_value_residual_correction(
        grouped_tail_weights,
        coordinates,
        block_coordinate_means,
        block_residual_means,
        block_counts,
        linear_map,
        candidate_indices,
        candidate_counts,
        selected_residual,
        block_size=2,
    )

    # Group 0: W=2 and G=2, so [2*3 + 2*(2-2*1)] = 6.
    # Group 1: W=2 and G=0, so [2*3 + 2*(0-2*1)] = 2.
    torch.testing.assert_close(correction, torch.tensor([[[6.0], [2.0]]]))


def test_block_conditional_state_appends_without_refitting_prefix() -> None:
    torch.manual_seed(7)
    token_count = 16
    head_dimension = 128
    rank = 8
    key_history = torch.randn(1, 1, token_count, head_dimension)
    value_mean = torch.randn(1, 1, head_dimension)
    value_basis = torch.eye(head_dimension)[:, :rank].reshape(
        1, 1, head_dimension, rank
    )
    coefficients = torch.randn(1, 1, token_count, rank)
    reconstructed = value_mean.unsqueeze(2) + torch.einsum(
        "bhnr,bhdr->bhnd", coefficients, value_basis
    )
    residual = 0.05 * key_history[..., :head_dimension]
    value_history = reconstructed + residual
    state: dict[str, object] = {
        "basis": torch.eye(head_dimension).reshape(
            1, 1, head_dimension, head_dimension
        ),
        "packed_qmse_allocation": torch.full(
            (1, 1, 8), 8, dtype=torch.int64
        ),
        "qk_metric_rebuild_count": 1,
    }

    initial = runner._block_conditional_value_residual_state(
        key_history,
        value_history,
        value_mean,
        value_basis,
        coefficients,
        state,
        dimension=8,
        block_size=4,
        moment_bits=8,
    )
    initial_map_pointer = initial["linear_map"].data_ptr()

    new_key = torch.randn(1, 1, 1, head_dimension)
    new_coefficients = torch.randn(1, 1, 1, rank)
    new_reconstructed = value_mean.unsqueeze(2) + torch.einsum(
        "bhnr,bhdr->bhnd", new_coefficients, value_basis
    )
    new_value = new_reconstructed + 0.05 * new_key
    appended = runner._block_conditional_value_residual_state(
        torch.cat((key_history, new_key), dim=2),
        torch.cat((value_history, new_value), dim=2),
        value_mean,
        value_basis,
        torch.cat((coefficients, new_coefficients), dim=2),
        state,
        dimension=8,
        block_size=4,
        moment_bits=8,
    )

    assert appended is initial
    assert appended["indexed_count"] == token_count + 1
    assert appended["coordinates"].shape[2] == token_count + 1
    assert float(appended["block_counts"].sum()) == token_count + 1
    assert appended["linear_map"].data_ptr() == initial_map_pointer
