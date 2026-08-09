from __future__ import annotations

import pytest

from summarize_128k_candidate_overlap_20260716 import summarize_payload


def test_candidate_overlap_reports_duplicate_rate() -> None:
    row = summarize_payload(
        {
            "record_candidate_overlap": True,
            "candidate_selection_mode": "per_head_stream",
            "topic": "computer",
            "history_tokens": 128000,
            "eval_tokens": 8,
            "candidate_fraction": 0.01,
            "stream_group_size": 2,
            "mean_candidate_union_fraction": 0.015,
            "max_candidate_union_fraction": 0.019,
            "ppl": 2.0,
            "online_seconds": 1.0,
        }
    )

    assert row["raw_concatenated_fraction"] == pytest.approx(0.02)
    assert row["mean_duplicate_candidate_rate"] == pytest.approx(0.25)


def test_candidate_overlap_rejects_union_larger_than_concatenation() -> None:
    with pytest.raises(ValueError, match="physical bound"):
        summarize_payload(
            {
                "record_candidate_overlap": True,
                "candidate_selection_mode": "per_head_stream",
                "topic": "computer",
                "history_tokens": 128000,
                "eval_tokens": 8,
                "candidate_fraction": 0.01,
                "stream_group_size": 2,
                "mean_candidate_union_fraction": 0.021,
                "max_candidate_union_fraction": 0.021,
                "ppl": 2.0,
                "online_seconds": 1.0,
            }
        )
