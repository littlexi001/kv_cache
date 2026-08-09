from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_ruler32k_rope_sparse as runner  # noqa: E402
import summarize_ruler32k_rope_sparse as summarizer  # noqa: E402


def test_all_subsequences_returns_every_exact_match() -> None:
    assert runner.all_subsequences([1, 2, 1, 2, 3], [1, 2]) == [(0, 2), (2, 4)]
    assert runner.all_subsequences([1, 2], []) == []


def test_merge_spans_unifies_tokenization_views_of_one_occurrence() -> None:
    assert runner.merge_spans([(10, 18), (11, 18), (30, 31)]) == ((10, 18), (30, 31))


def test_paired_summary_uses_same_samples() -> None:
    rows = []
    for sample, full, exact, ours in (("a", 1.0, 0.0, 1.0), ("b", 0.0, 1.0, 1.0)):
        for variant, score in (("native_full", full), ("rope_top2", exact), ("local_global_postscore", ours)):
            rows.append(
                {
                    "sample_id": sample,
                    "task": "niah_single_1",
                    "variant": variant,
                    "official_score": score,
                    "first_answer_next_token_nll": 1.0,
                    "query_seconds": 1.0,
                    "generation_seconds": 1.0,
                    "answer_evidence_span_count": 0,
                }
            )
    summary = summarizer.summarize(rows, "rope_top2", 20, 7)
    comparison = summary["comparisons"]["local_global_postscore_minus_rope_top2"]
    assert comparison["paired_samples"] == 2
    assert comparison["official_score_delta_points"] == 50.0
    assert comparison["rescues"] == 1
    assert comparison["harms"] == 0
    assert comparison["improved_samples"] == 1
    assert comparison["worsened_samples"] == 0
