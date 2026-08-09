from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_binarypc_exact_rerank_trace_20260801 import (
    evaluate_selection,
    parse_factors,
    reference_key_record,
)
from benchmark_binarypc_exact_rerank_direct_stages_20260801 import (
    selected_count,
)


def test_selected_count_uses_length_only_capped_schedule() -> None:
    assert selected_count(2048) == 256
    assert selected_count(4096) == 256
    assert selected_count(8192) == 492
    assert selected_count(16384) == 984
    assert selected_count(24576) == 1280
    assert selected_count(131072) == 1280


def test_overfetch_factors_are_sorted_and_validated() -> None:
    assert parse_factors("2,1,1.5,2") == [1.0, 1.5, 2.0]
    try:
        parse_factors("0.5,1")
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("an invalid overfetch factor was accepted")


def test_exact_rerank_maps_local_positions_back_to_history_tokens() -> None:
    exact_scores = torch.tensor(
        [[[[10.0, 9.0, 8.0, 0.0], [0.0, 8.0, 9.0, 10.0]]]]
    )
    shared_candidates = torch.tensor([[[0, 1, 2]]], dtype=torch.long)
    metrics = evaluate_selection(
        exact_scores, shared_candidates, target_count=2
    )

    assert metrics["candidate_exact_topk_recall"] == 0.75
    assert metrics["exact_topk_recall_after_rerank"] == 0.75
    assert metrics["top1_recall_after_rerank"] == 0.5
    assert metrics["query_heads"] == 2


def test_reference_key_record_supports_key_deduplicated_traces() -> None:
    first = {"step": 0, "key": torch.ones(1, 1, 4, 2)}
    later = {"step": 7, "key": None}
    assert reference_key_record([first, later]) is first
