from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import run_queryspan_prerope_retrieval_probe_8b as target  # noqa: E402


def test_anchor_positions_are_even_label_free_and_prefix_bounded() -> None:
    positions = target.select_query_anchor_positions((100, 200), 180, 5)
    assert positions == (100, 120, 140, 159, 179)
    assert all(100 <= value < 180 for value in positions)
    assert target.select_query_anchor_positions((4, 9), 9, 1) == (8,)


def test_budget_layout_reserves_sink_local_and_current_exactly() -> None:
    layout = target.compute_budget_layout(
        key_count=1000,
        keep_count=200,
        local_window=128,
        sink_tokens=16,
    )
    assert layout.current == 999
    assert layout.local_start == 871
    assert layout.local_count == 128
    assert layout.sink_count == 16
    assert layout.remote_start == 16
    assert layout.remote_end == 871
    assert layout.remote_count == 55
    assert 16 + 55 + 128 + 1 == 200


def test_exact_selector_has_strict_budget_unique_support_and_fixed_regions() -> None:
    torch.manual_seed(3)
    scores = torch.randn(3, 1000)
    selected, layout = target.exact_final_pre_selection(scores, 200, 128, 16)
    assert selected.shape == (3, 200)
    for head in range(3):
        values = set(map(int, selected[head].tolist()))
        assert len(values) == 200
        assert set(range(16)).issubset(values)
        assert set(range(871, 1000)).issubset(values)
    assert layout.remote_count == 55


def _conjunction_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    # Two query facets.  Block 0 contains one exact match for each facet;
    # block 1 contains only the first facet.  Position 4 is the current token.
    anchors = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    keys = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [-1.0, -1.0]]]]
    )
    return anchors, keys


def test_late_interaction_prefers_block_matching_both_question_facets() -> None:
    anchors, keys = _conjunction_tensors()
    block_scores, lengths = target.queryspan_block_scores(
        anchors,
        keys,
        remote_start=0,
        remote_end=4,
        block_size=2,
        score_chunk_blocks=1,
    )
    assert lengths == [2, 2]
    assert float(block_scores[0, 0]) > float(block_scores[0, 1])
    selected, layout, diagnostics = target.queryspan_block_selection(
        anchors,
        keys,
        keep_count=3,
        local_window=0,
        sink_tokens=0,
        block_size=2,
        score_chunk_blocks=1,
    )
    assert layout.remote_count == 2
    assert set(map(int, selected[0].tolist())) == {0, 1, 4}
    assert diagnostics["selected_blocks_mean"] == 1.0


def test_late_interaction_masks_padding_in_short_last_block() -> None:
    anchors = torch.tensor([[[[-1.0, 0.0]]]])
    # The last block has one valid negatively aligned key.  Zero padding would
    # incorrectly score 0 and beat -1 if it were not masked before max().
    keys = torch.tensor([[[[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]]])
    scores, lengths = target.queryspan_block_scores(
        anchors, keys, 0, 5, block_size=4, score_chunk_blocks=2
    )
    assert lengths == [4, 1]
    assert abs(float(scores[0, 0]) - 1.0) < 1e-6
    assert abs(float(scores[0, 1]) + 1.0) < 1e-6


def test_block_selector_partial_block_still_obeys_exact_budget() -> None:
    torch.manual_seed(9)
    anchors = torch.randn(1, 4, 3, 8)
    # Two KV heads, two query groups.
    keys = torch.randn(1, 2, 50, 8)
    selected, layout, _ = target.queryspan_block_selection(
        anchors,
        keys,
        keep_count=13,
        local_window=3,
        sink_tokens=2,
        block_size=4,
        score_chunk_blocks=2,
    )
    assert selected.shape == (4, 13)
    assert layout.remote_count == 7
    for row in selected:
        assert len(set(map(int, row.tolist()))) == 13
        assert {0, 1, 46, 47, 48, 49}.issubset(set(map(int, row.tolist())))


def test_tokenmax_selector_is_multi_anchor_and_gqa_safe() -> None:
    anchors, keys = _conjunction_tensors()
    selected, layout = target.queryspan_tokenmax_selection(
        anchors, keys, keep_count=3, local_window=0, sink_tokens=0
    )
    assert selected.shape == (1, 3)
    assert int(selected[0, -1]) == layout.current


def test_gqa_scores_and_selected_value_gather_match_explicit_repeat() -> None:
    torch.manual_seed(11)
    query = torch.randn(1, 4, 2, 6)
    key = torch.randn(1, 2, 7, 6)
    value = torch.randn(1, 2, 7, 5)
    actual_scores = target.gqa_query_key_scores(query, key, 0.3)
    repeated_key = key.repeat_interleave(2, dim=1)
    expected_scores = torch.matmul(query, repeated_key.transpose(2, 3)) * 0.3
    torch.testing.assert_close(actual_scores, expected_scores)

    positions = torch.tensor([[0, 3], [1, 4], [2, 5], [0, 6]])
    actual_value = target.gather_per_query_head_gqa(value, positions, groups=2)
    repeated_value = value.repeat_interleave(2, dim=1)
    expected_value = repeated_value.gather(
        2, positions.view(1, 4, 2, 1).expand(1, 4, 2, 5)
    )
    torch.testing.assert_close(actual_value, expected_value)


def test_metric_labels_do_not_change_selector_output() -> None:
    anchors, keys = _conjunction_tensors()
    first, _, _ = target.queryspan_block_selection(
        anchors, keys, 3, 0, 0, 2, 1
    )
    # The selection function has no evidence/conflict argument.  Changing the
    # downstream evaluator therefore cannot feed back into its support.
    controller_a = target.QuerySpanController(
        "queryspan_block_top2_postscore", 0.02, 0, 0, 0, 0, 2, 1,
        {"gold_evidence": (0,), "conflict_evidence": (2,), "lexical_format_distractor": ()},
        {"gold_evidence": ((0, 1),), "conflict_evidence": ((2, 3),)},
    )
    controller_b = target.QuerySpanController(
        "queryspan_block_top2_postscore", 0.02, 0, 0, 0, 0, 2, 1,
        {"gold_evidence": (2,), "conflict_evidence": (0,), "lexical_format_distractor": ()},
        {"gold_evidence": ((2, 3),), "conflict_evidence": ((0, 1),)},
    )
    second, _, _ = target.queryspan_block_selection(
        anchors, keys, 3, 0, 0, 2, 1
    )
    torch.testing.assert_close(first, second)
    assert controller_a.evaluation_positions != controller_b.evaluation_positions


def test_controller_audits_budget_sink_local_current_and_probability_partition() -> None:
    selected = torch.tensor([[0, 2, 3, 4], [0, 1, 3, 4]])
    layout = target.compute_budget_layout(5, 4, 1, 1)
    weights = torch.tensor([[[[0.1, 0.2, 0.3, 0.4]], [[0.25, 0.25, 0.25, 0.25]]]])
    controller = target.QuerySpanController(
        "exact_final_pre_top2_postscore",
        0.8,
        0,
        0,
        1,
        1,
        2,
        1,
        {"gold_evidence": (2,), "conflict_evidence": (1,), "lexical_format_distractor": ()},
        {"gold_evidence": ((2, 3),), "conflict_evidence": ((1, 2),)},
    )
    controller.record(selected, weights, layout, None, None)
    summary = controller.metrics.summary()
    assert summary["token_budget_violation_fraction"] == 0.0
    assert summary["duplicate_support_violation_fraction"] == 0.0
    assert summary["sink_coverage_violation_fraction"] == 0.0
    assert summary["local_coverage_violation_fraction"] == 0.0
    assert summary["current_coverage_violation_fraction"] == 0.0
    total_mass = (
        summary["gold_evidence_attention_mass"]
        + summary["conflict_attention_mass"]
        + summary["lexical_distractor_attention_mass"]
        + summary["other_attention_mass"]
    )
    assert abs(total_mass - 1.0) < 1e-6
    assert summary["selector_used_evidence_labels"] == 0


def test_none_controller_is_an_exact_noop_dispatch() -> None:
    sentinel = object()

    class FakeAttention:
        def _queryspan_original_forward(self, **kwargs):
            assert kwargs["hidden_states"].shape[-2] == 1
            return sentinel

    with target.activate(None):
        result = target.queryspan_attention_forward(
            FakeAttention(),
            torch.zeros(1, 1, 4),
            (torch.ones(1), torch.zeros(1)),
        )
    assert result is sentinel


def test_capture_hooks_store_all_keys_but_only_requested_queries(monkeypatch) -> None:
    class FakeAttention:
        _queryspan_pre_key_chunks: list[torch.Tensor] = []
        _queryspan_anchor_chunks: list[torch.Tensor] = []
        _queryspan_anchor_position_chunks: list[int] = []
        _queryspan_capture_cursor = 0

    attention = FakeAttention()
    monkeypatch.setattr(target, "_PREFIX_KEY_STORAGE", "cpu")
    monkeypatch.setattr(target, "_CAPTURE_PREFIX", True)
    monkeypatch.setattr(target, "_CAPTURE_ANCHOR_POSITIONS", (1, 3))
    query_output = torch.arange(1 * 4 * 2 * 3).reshape(1, 4, 2, 3).float()
    key_output = torch.arange(1 * 4 * 1 * 3).reshape(1, 4, 1, 3).float()
    target._make_query_capture_hook(attention)(None, (), query_output)
    target._make_key_capture_hook(attention)(None, (), key_output)
    assert attention._queryspan_capture_cursor == 4
    assert attention._queryspan_anchor_position_chunks == [1, 3]
    assert attention._queryspan_anchor_chunks[0].shape == (1, 2, 2, 3)
    assert attention._queryspan_pre_key_chunks[0].shape == (1, 1, 4, 3)
    torch.testing.assert_close(
        attention._queryspan_anchor_chunks[0].transpose(1, 2),
        query_output[:, [1, 3]],
    )


def test_read_only_query_kv_does_not_mutate_prefix_cache() -> None:
    class Cache:
        key_cache = [torch.randn(1, 2, 5, 3)]
        value_cache = [torch.randn(1, 2, 5, 4)]

    cache = Cache()
    key_before = cache.key_cache[0].clone()
    value_before = cache.value_cache[0].clone()
    key, value = target.read_only_final_query_kv(
        cache,
        0,
        torch.randn(1, 2, 1, 3),
        torch.randn(1, 2, 1, 4),
    )
    assert key.shape[-2] == 6 and value.shape[-2] == 6
    torch.testing.assert_close(cache.key_cache[0], key_before)
    torch.testing.assert_close(cache.value_cache[0], value_before)
