from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import run_blockwise_phase_transport_probe_8b as target  # noqa: E402
import run_rope_retrieval_repair_8b as rope_repair  # noqa: E402


def test_block_selector_uses_exact_pre_rope_max_and_respects_budget() -> None:
    scores = torch.zeros(2, 41)
    # With sink=2, local=4, B=4, remote blocks begin at positions 2, 6, ...
    # Head 0 should choose block ids 1 and 5; head 1 chooses 0 and 6.
    scores[0, 6] = 9.0
    scores[0, 22] = 8.0
    scores[1, 2] = 7.0
    scores[1, 26] = 6.0

    selected = target.blockwise_premax_selection(
        scores,
        keep_count=15,
        local_window=4,
        sink_tokens=2,
        block_size=4,
    )

    assert selected.actual_keep_count == 15
    assert selected.actual_keep_count <= selected.requested_keep_count
    assert selected.selected_block_ids.tolist() == [[1, 5], [0, 6]]
    assert selected.anchor_positions.tolist() == [[6, 22], [2, 26]]
    assert selected.positions[0].tolist() == [
        0,
        1,
        6,
        7,
        8,
        9,
        22,
        23,
        24,
        25,
        36,
        37,
        38,
        39,
        40,
    ]
    assert int(selected.remote_mask[0].sum()) == 8


def test_block_selector_realized_budget_is_within_one_block() -> None:
    scores = torch.randn(3, 100)
    selected = target.blockwise_premax_selection(
        scores,
        keep_count=21,
        local_window=8,
        sink_tokens=3,
        block_size=4,
    )

    assert selected.actual_keep_count <= 21
    assert 21 - selected.actual_keep_count < 4


def test_per_head_distance_scores_match_direct_pair_formula() -> None:
    torch.manual_seed(17)
    query = torch.randn(1, 2, 1, 8, dtype=torch.float64)
    key = torch.randn(1, 2, 3, 8, dtype=torch.float64)
    distance = torch.tensor([[3, 7, 11], [4, 8, 12]])
    inv_freq = torch.tensor([1.0, 0.2, 0.04, 0.008])
    scale = 1.13
    score_scale = 1.0 / math.sqrt(8)

    actual = target.scores_at_head_distances(
        query,
        key,
        distance,
        inv_freq,
        rope_repair.rotate_half,
        scale,
        score_scale,
    )
    pair_phase = distance.double().unsqueeze(-1) * inv_freq.view(1, 1, -1)
    embedding = torch.cat((pair_phase, pair_phase), dim=-1)
    rotated_key = key[0] * embedding.cos() - rope_repair.rotate_half(key[0]) * embedding.sin()
    expected = (
        query[0, :, 0].unsqueeze(1) * rotated_key
    ).sum(dim=-1) * score_scale * scale**2

    # Qwen stores inverse frequencies in float32, so phase formation is
    # intentionally float32 even when the synthetic Q/K tensors are float64.
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)


def _one_block_selection() -> target.BlockSelection:
    return target.BlockSelection(
        positions=torch.tensor([[2, 3, 7]]),
        remote_mask=torch.tensor([[True, True, False]]),
        selected_block_ids=torch.tensor([[1]]),
        anchor_positions=torch.tensor([[2]]),
        block_size=2,
        requested_keep_count=3,
        actual_keep_count=3,
        local_start=7,
        remote_start=0,
        remote_end=6,
    )


def test_triggered_transport_moves_one_block_and_preserves_internal_offsets() -> None:
    selection = _one_block_selection()
    query = torch.tensor([[[[1.0, 0.0]]]])
    all_keys = torch.zeros(1, 1, 8, 2)
    all_keys[0, 0, 2] = torch.tensor([1.0, 0.0])
    all_keys[0, 0, 3] = torch.tensor([0.5, 0.0])
    all_keys[0, 0, 7] = torch.tensor([0.0, 1.0])
    distances = (7 - torch.arange(8)).view(1, -1)
    native_values = target.scores_at_head_distances(
        query,
        all_keys,
        distances,
        torch.tensor([1.0]),
        rope_repair.rotate_half,
        attention_scale=1.0,
        score_scale=1.0,
    )
    native = native_values.view(1, 1, 1, 8)

    transport = target.counterfactual_transport(
        query,
        all_keys,
        native,
        selection,
        current_position=7,
        local_anchor_distance=1,
        inv_freq=torch.tensor([1.0]),
        rotate_half=rope_repair.rotate_half,
        attention_scale=1.0,
        score_scale=1.0,
    )

    assert transport["trigger"].tolist() == [[True]]
    assert transport["tau"].tolist() == [[4]]
    assert transport["actual_distance"].tolist() == [[5, 4, 0]]
    assert transport["transport_distance"].tolist() == [[1, 0, 0]]
    assert target.maximum_block_relative_error(
        transport["actual_distance"],
        transport["transport_distance"],
        selection,
    ) == 0.0


def test_independent_clipping_does_not_preserve_block_offsets() -> None:
    selection = _one_block_selection()
    original = torch.tensor([[5, 4, 0]])
    clipped = torch.where(
        selection.remote_mask, original.clamp_max(1), original
    )

    assert clipped.tolist() == [[1, 1, 0]]
    assert target.maximum_block_relative_error(original, clipped, selection) == 1.0


def test_random_control_matches_trigger_count_per_head_exactly() -> None:
    trigger = torch.tensor(
        [[True, False, True, False], [False, True, False, False]]
    )
    block_ids = torch.tensor([[1, 3, 5, 8], [0, 2, 7, 9]])

    first = target.matched_random_trigger_mask(trigger, block_ids, layer_index=12)
    second = target.matched_random_trigger_mask(trigger, block_ids, layer_index=12)

    assert torch.equal(first, second)
    assert torch.equal(first.sum(dim=-1), trigger.sum(dim=-1))


def test_mass_preservation_matches_native_remote_log_partition() -> None:
    native = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
    corrected = torch.tensor([[1.0, 5.0, 3.0, 0.5]])
    remote = torch.tensor([[False, True, True, False]])

    preserved = target.phase.preserve_remote_partition(corrected, native, remote)

    torch.testing.assert_close(
        torch.logsumexp(preserved.masked_fill(~remote, -torch.inf), dim=-1),
        torch.logsumexp(native.masked_fill(~remote, -torch.inf), dim=-1),
    )
    torch.testing.assert_close(preserved[~remote], corrected[~remote])


def test_all_required_ablation_variants_exist_for_both_block_sizes() -> None:
    for block_size in (16, 32):
        prefix = f"block{block_size}_"
        for suffix in (
            "selector_only",
            "clipped_consumer",
            "transport",
            "transport_masspreserve",
            "random_matched",
        ):
            assert prefix + suffix in target.BLOCKWISE_VARIANTS
