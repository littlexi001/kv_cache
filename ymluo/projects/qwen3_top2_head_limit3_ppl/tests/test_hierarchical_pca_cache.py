from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from hierarchical_pca_cache_20260715 import (
    HierarchicalPCACache,
    bounded_fraction_count,
    exact_cache_capacity,
    hierarchical_attention_mode,
    exact_per_head_rerank_slots,
    exact_shared_rerank_slots,
    head_balanced_topk_offsets,
    merge_streamed_gqa_outputs,
    pack_projected_int4,
    update_sorted_directory,
)


def make_empty_cache_for_runtime_actions() -> HierarchicalPCACache:
    return HierarchicalPCACache(
        [],
        projection_dim=64,
        index_bits=4,
        candidate_fraction=0.015,
        attention_fraction=0.015,
        candidate_selection_mode="per_head_stream",
        rerank_selection_mode="shared_sum",
        original_gpu_bytes=0,
        directory_backend="fused",
        record_traces=False,
        recent_fraction=0.0,
        debug_directory=False,
        stream_group_size=2,
    )


def make_directory(capacity: int = 6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sentinel = torch.iinfo(torch.int32).max
    ids = torch.full((1, 1, capacity), sentinel, dtype=torch.int32)
    slots = torch.arange(capacity, dtype=torch.int32).reshape(1, 1, -1)
    ages = torch.zeros_like(ids, dtype=torch.uint8)
    return ids, slots, ages


def test_sorted_directory_populates_then_hits_same_candidates() -> None:
    ids, slots, ages = make_directory()
    candidates = torch.tensor([[[7, 2, 11, 5]]], dtype=torch.int32)

    first = update_sorted_directory(candidates, ids, slots, ages)
    assert first.hit_rate == 0.0
    assert len(torch.unique(first.final_slots)) == candidates.shape[-1]
    assert sorted(ids[0, 0, :4].tolist()) == [2, 5, 7, 11]

    second = update_sorted_directory(candidates, ids, slots, ages)
    assert second.hit_rate == 1.0
    assert second.misses == []
    assert torch.equal(second.final_slots, first.final_slots)


def test_sorted_directory_refreshes_hits_and_evicts_oldest() -> None:
    ids, slots, ages = make_directory(capacity=4)
    first = torch.tensor([[[1, 2, 3]]], dtype=torch.int32)
    update_sorted_directory(first, ids, slots, ages)
    resident_slot_for_three = slots[0, 0, torch.searchsorted(ids[0, 0], torch.tensor(3))].item()

    second = torch.tensor([[[3, 4, 5]]], dtype=torch.int32)
    result = update_sorted_directory(second, ids, slots, ages)

    assert result.hit_rate == pytest.approx(1.0 / 3.0)
    assert result.final_slots[0, 0, 0].item() == resident_slot_for_three
    assert set(ids[0, 0].tolist()) >= {3, 4, 5}


def test_sorted_directory_rejects_candidate_set_larger_than_cache() -> None:
    ids, slots, ages = make_directory(capacity=2)
    candidates = torch.tensor([[[1, 2, 3]]], dtype=torch.int32)

    with pytest.raises(ValueError, match="cannot exceed"):
        update_sorted_directory(candidates, ids, slots, ages)


def test_exact_shared_rerank_uses_all_query_heads_for_a_kv_head() -> None:
    grouped_query = torch.tensor([[[[2.0, 0.0], [0.0, 1.0]]]])
    key_cache = torch.tensor(
        [[[[1.0, 0.0], [0.0, 3.0], [0.5, 0.5], [-1.0, -1.0]]]]
    )
    candidate_slots = torch.tensor([[[0, 1, 2]]], dtype=torch.int32)

    reranked = exact_shared_rerank_slots(
        grouped_query, key_cache, candidate_slots, keep_count=2
    )

    assert set(reranked.flatten().tolist()) == {0, 1}


def test_exact_shared_max_rerank_protects_a_single_query_head_peak() -> None:
    grouped_query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    key_cache = torch.tensor(
        [[[[6.0, 0.0], [4.0, 4.0], [0.0, 5.0], [-1.0, -1.0]]]]
    )
    candidate_slots = torch.tensor([[[0, 1, 2]]], dtype=torch.int32)

    reranked = exact_shared_rerank_slots(
        grouped_query,
        key_cache,
        candidate_slots,
        keep_count=1,
        selection_mode="shared_max",
    )

    assert reranked.item() == 0


def test_head_balanced_topk_protects_each_query_head() -> None:
    scores = torch.tensor(
        [[[[10.0, 9.0, 0.0, 0.0], [0.0, 0.0, 8.0, 7.0]]]]
    )

    selected = head_balanced_topk_offsets(scores, keep_count=2)

    assert set(selected.flatten().tolist()) == {0, 2}


def test_exact_per_head_rerank_keeps_distinct_virtual_candidates() -> None:
    grouped_query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    key_cache = torch.tensor(
        [[[[3.0, 0.0], [2.0, 0.0], [0.0, 4.0], [0.0, 1.0]]]]
    )
    candidate_slots = torch.tensor([[[[0, 1], [2, 3]]]], dtype=torch.int32)

    reranked = exact_per_head_rerank_slots(
        grouped_query, key_cache, candidate_slots, keep_count=1
    )

    assert reranked.flatten().tolist() == [0, 2]


def test_pack_projected_int4_round_trip_is_bounded() -> None:
    projected = torch.tensor([[[[-7.0, -3.0, 2.0, 7.0]]]])

    packed, scales = pack_projected_int4(projected)
    low = (packed & 0x0F).to(torch.float32) - 7.0
    high = (packed >> 4).to(torch.float32) - 7.0
    restored = torch.stack((low, high), dim=-1).flatten(-2) * scales

    assert packed.dtype == torch.uint8
    assert torch.max(torch.abs(projected - restored)).item() <= scales.max().item()


def test_exact_cache_capacity_covers_short_prefix_decode_window() -> None:
    capacity = exact_cache_capacity(
        sequence_length=1000,
        max_new_tokens=512,
        exact_cache_fraction=0.032,
        candidate_fraction=0.025,
        stream_group_size=1,
    )

    assert capacity == 38


def test_exact_cache_capacity_preserves_configured_long_context_target() -> None:
    capacity = exact_cache_capacity(
        sequence_length=128_000,
        max_new_tokens=256,
        exact_cache_fraction=0.032,
        candidate_fraction=0.01,
        stream_group_size=2,
    )

    assert capacity == 4096


@pytest.mark.parametrize(
    ("sequence_length", "expected"),
    [
        (2_000, 256),
        (4_000, 256),
        (8_000, 480),
        (16_000, 960),
        (24_000, 1_280),
        (32_000, 1_280),
        (64_000, 1_280),
        (128_000, 1_280),
    ],
)
def test_bounded_fraction_count_matches_frozen_countcap_budget(
    sequence_length: int,
    expected: int,
) -> None:
    assert (
        bounded_fraction_count(
            sequence_length,
            0.06,
            minimum_tokens=256,
            maximum_tokens=1_280,
        )
        == expected
    )


def test_exact_cache_capacity_respects_candidate_floor_and_cap() -> None:
    short_capacity = exact_cache_capacity(
        sequence_length=2_000,
        max_new_tokens=64,
        exact_cache_fraction=0.08,
        candidate_fraction=0.06,
        stream_group_size=1,
        candidate_min_tokens=256,
        candidate_max_tokens=1_280,
    )
    long_capacity = exact_cache_capacity(
        sequence_length=128_000,
        max_new_tokens=256,
        exact_cache_fraction=0.032,
        candidate_fraction=0.06,
        stream_group_size=2,
        candidate_min_tokens=256,
        candidate_max_tokens=1_280,
    )

    assert short_capacity == 256
    assert long_capacity == 4096


def test_capped_candidate_budget_does_not_require_six_percent_hot_cache() -> None:
    cache = HierarchicalPCACache(
        [],
        projection_dim=128,
        index_bits=4,
        candidate_fraction=0.06,
        attention_fraction=0.06,
        candidate_selection_mode="per_head_stream",
        rerank_selection_mode="shared_sum",
        original_gpu_bytes=0,
        directory_backend="fused",
        record_traces=False,
        recent_fraction=0.0,
        debug_directory=False,
        stream_group_size=1,
        index_mode="qk_variable",
        candidate_min_tokens=256,
        candidate_max_tokens=1_280,
    )

    assert cache.candidate_max_tokens == 1_280


def test_merge_streamed_gqa_outputs_restores_query_head_order() -> None:
    group_zero = torch.tensor([[[[0.0], [10.0]]]])
    group_one = torch.tensor([[[[1.0], [11.0]]]])

    merged = merge_streamed_gqa_outputs(
        [group_zero, group_one], query_heads=4, kv_heads=2
    )

    assert merged.shape == (1, 1, 4, 1)
    assert merged.flatten().tolist() == [0.0, 1.0, 10.0, 11.0]


def test_merge_two_group_streams_restores_query_head_order() -> None:
    first_pair = torch.tensor([[[[0.0], [1.0], [10.0], [11.0]]]])
    second_pair = torch.tensor([[[[2.0], [3.0], [12.0], [13.0]]]])

    merged = merge_streamed_gqa_outputs(
        [first_pair, second_pair], query_heads=8, kv_heads=2
    )

    assert merged.shape == (1, 1, 8, 1)
    assert merged.flatten().tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        10.0,
        11.0,
        12.0,
        13.0,
    ]


def test_runtime_action_switches_within_construction_maximum() -> None:
    cache = make_empty_cache_for_runtime_actions()

    cache.set_runtime_action(candidate_fraction=0.01, stream_group_size=2)

    assert cache.candidate_fraction == pytest.approx(0.01)
    assert cache.attention_fraction == pytest.approx(0.01)
    assert cache.stream_group_size == 2


def test_runtime_action_preserves_temporal_candidates_for_unchanged_budget() -> None:
    cache = make_empty_cache_for_runtime_actions()
    cached = torch.tensor([1, 2], dtype=torch.int32)
    state = SimpleNamespace(
        capacity=1001,
        exact_cache_count=100,
        cached_retrieved_candidates=cached,
        candidate_cache_age=1,
    )
    cache.states = [state]

    cache.set_runtime_action(candidate_fraction=0.015, stream_group_size=2)
    assert state.cached_retrieved_candidates is cached
    assert state.candidate_cache_age == 1

    cache.set_runtime_action(candidate_fraction=0.01, stream_group_size=2)
    assert state.cached_retrieved_candidates is None
    assert state.candidate_cache_age == 0


def test_host_append_is_async_by_default_and_can_be_disabled() -> None:
    cache = make_empty_cache_for_runtime_actions()

    assert cache.async_host_append is True

    cache = HierarchicalPCACache(
        [],
        projection_dim=64,
        index_bits=4,
        candidate_fraction=0.015,
        attention_fraction=0.015,
        candidate_selection_mode="per_head_stream",
        rerank_selection_mode="shared_sum",
        original_gpu_bytes=0,
        directory_backend="fused",
        record_traces=False,
        recent_fraction=0.0,
        debug_directory=False,
        stream_group_size=2,
        async_host_append=False,
    )
    assert cache.async_host_append is False


def test_runtime_action_rejects_larger_unallocated_budget() -> None:
    cache = make_empty_cache_for_runtime_actions()

    with pytest.raises(ValueError, match="construction-time maximum"):
        cache.set_runtime_action(candidate_fraction=0.02, stream_group_size=1)


def test_sequence_checkpoint_and_restore_rewind_all_layers() -> None:
    cache = make_empty_cache_for_runtime_actions()
    cache.states = [
        SimpleNamespace(length=12, initial_length=10),
        SimpleNamespace(length=12, initial_length=10),
    ]

    assert cache.sequence_checkpoint() == 12
    cache.restore_sequence_length(11)

    assert [state.length for state in cache.states] == [11, 11]


def test_restore_sequence_length_rejects_prefill_rewind() -> None:
    cache = make_empty_cache_for_runtime_actions()
    cache.states = [SimpleNamespace(length=12, initial_length=10)]

    with pytest.raises(ValueError, match="initial and current"):
        cache.restore_sequence_length(9)


def test_retrieval_features_aggregate_preceding_forward_diagnostics() -> None:
    cache = make_empty_cache_for_runtime_actions()
    cache.states = [
        SimpleNamespace(
            last_retrieval_probe=torch.tensor([1, 2]),
            last_retrieval_score_spread=torch.tensor(0.2),
            last_retrieval_candidate_stability=torch.tensor(0.5),
            last_retrieval_refreshed=torch.tensor(1.0),
        ),
        SimpleNamespace(
            last_retrieval_probe=torch.tensor([3, 4]),
            last_retrieval_score_spread=torch.tensor(0.6),
            last_retrieval_candidate_stability=torch.tensor(0.75),
            last_retrieval_refreshed=torch.tensor(0.0),
        ),
    ]

    features = cache.retrieval_features()

    assert features["retrieval_feature_valid"] == 1.0
    assert features["retrieval_score_spread"] == pytest.approx(0.4)
    assert features["retrieval_candidate_stability"] == pytest.approx(0.625)
    assert features["retrieval_refreshed_fraction"] == pytest.approx(0.5)


def test_retrieval_diagnostic_checkpoint_isolates_counterfactual_probes() -> None:
    cache = make_empty_cache_for_runtime_actions()
    state = SimpleNamespace(
        last_retrieval_probe=torch.tensor([1, 2]),
        last_retrieval_score_spread=torch.tensor(0.2),
        last_retrieval_candidate_stability=torch.tensor(0.5),
        last_retrieval_refreshed=torch.tensor(1.0),
    )
    cache.states = [state]
    checkpoint = cache.retrieval_diagnostic_checkpoint()
    state.last_retrieval_probe = torch.tensor([9, 10])
    state.last_retrieval_score_spread = torch.tensor(0.9)

    cache.restore_retrieval_diagnostic(checkpoint)

    assert state.last_retrieval_probe.tolist() == [1, 2]
    assert state.last_retrieval_score_spread.item() == pytest.approx(0.2)


def test_qwen3_handler_preserves_standard_dense_forward() -> None:
    torch.manual_seed(11)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            attention_dropout=0.0,
        )
    ).eval()
    input_ids = torch.randint(0, 128, (1, 8))

    baseline = model(input_ids=input_ids, use_cache=True).logits
    with hierarchical_attention_mode(model):
        patched = model(input_ids=input_ids, use_cache=True).logits

    assert torch.allclose(patched, baseline, atol=1.0e-6, rtol=1.0e-5)
