from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_hybrid_block_retrieval import (  # noqa: E402
    allocate_multi_record_blocks,
    parse_record_schedules,
)
from rerank_candidate_blocks_by_question_nll import quota_combine  # noqa: E402
from train_candidate_block_reranker import build_query_features  # noqa: E402
from train_retrieval_action_router import (  # noqa: E402
    FEATURE_NAMES,
    build_feature_vector,
    overlap_fraction,
)
from run_single_query_dynamic_kv_generation import (  # noqa: E402
    answer_hit,
    first_answer_end,
    select_shared_qk_blocks,
)
from run_dynamic_kv_multisample import (  # noqa: E402
    extract_first_final_answer,
    token_f1,
)


class HybridRetrievalTest(unittest.TestCase):
    def test_learned_reranker_features_are_answer_free(self) -> None:
        rows = []
        for index in range(4):
            rows.append(
                {
                    "question_nll": str(1.0 + index),
                    "block_bm25_score": str(4.0 - index),
                    "record_bm25_score": str(2.0 - index // 2),
                    "record_id": str(index // 2),
                    "record_bm25_rank": str(index // 2 + 1),
                    "block_position_fraction": str(index % 2),
                    "record_blocks": "2",
                    "dataset": "qa",
                    "is_gold": str(float(index == 1)),
                    "block_id": str(10 + index),
                }
            )
        matrix, labels, block_ids, _, feature_names = build_query_features(rows, ["qa"])
        self.assertEqual(matrix.shape, (4, len(feature_names)))
        self.assertEqual(labels.tolist(), [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(block_ids, [10, 11, 12, 13])
        self.assertFalse(any("gold" in name or "source" in name for name in feature_names))

    def test_quota_combine_refills_after_overlap(self) -> None:
        self.assertEqual(
            quota_combine([1, 2, 3, 4], [2, 5, 6], primary_quota=2, target=5),
            [1, 2, 5, 6, 3],
        )

    def test_schedule_parser(self) -> None:
        self.assertEqual(
            parse_record_schedules("head=3,1;balanced=2,1,1"),
            {"head": [3, 1], "balanced": [2, 1, 1]},
        )

    def test_multi_record_allocation_is_strict_and_deduplicated(self) -> None:
        selected = allocate_multi_record_blocks(
            record_order=[4, 7],
            rankings_by_record={4: [10, 11, 12], 7: [20, 21]},
            quotas=[2, 1],
            target_blocks=5,
            fallback_ranking=[11, 30, 31, 32],
        )
        self.assertEqual(selected, [10, 11, 20, 30, 31])

    def test_action_router_overlap_is_prefix_bounded(self) -> None:
        self.assertEqual(overlap_fraction([1, 2, 3], [2, 3, 4], 2), 0.5)
        self.assertEqual(overlap_fraction([1, 2, 3], [2, 3, 4], 3), 2 / 3)

    def test_action_router_features_exclude_answer_labels(self) -> None:
        vector = build_feature_vector(
            question="Where was the event in 2020?",
            deep_row={
                "selected_block_ids": "[0, 1, 2]",
                "ranked_block_ids": "[0, 1, 2, 3]",
            },
            hybrid_row={
                "selected_block_ids": "[1, 2, 3]",
                "ranked_block_ids": "[1, 2, 3, 0]",
            },
            routing_row={
                "relative_bm25_margin": "0.4",
                "used_likelihood": "True",
                "bm25_record": "0",
                "likelihood_record": "1",
            },
            record_score_rows=[
                {
                    "record_id": "0",
                    "bm25_rank": "1",
                    "bm25_score": "5",
                    "question_nll": "2.5",
                    "is_source_record": "1",
                },
                {
                    "record_id": "1",
                    "bm25_rank": "2",
                    "bm25_score": "4",
                    "question_nll": "2",
                    "is_source_record": "0",
                },
            ],
            candidate_row={
                "candidate_blocks": "100",
                "mean_block_question_nll": "4",
                "min_block_question_nll": "2",
                "candidate_any_oracle": "1",
            },
            block_to_record={0: 0, 1: 0, 2: 1, 3: 1},
        )
        self.assertEqual(vector.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.all(np.isfinite(vector)))
        self.assertFalse(
            any("answer" in name or "gold" in name or "source" in name for name in FEATURE_NAMES)
        )

    def test_dynamic_generation_answer_hit(self) -> None:
        answers = ["Dayton, Ohio"]
        self.assertTrue(answer_hit("Final answer: Dayton, Ohio.", answers))
        self.assertFalse(answer_hit("The answer concerns Marion Byron.", answers))
        self.assertEqual(
            first_answer_end("She was born in Dayton, Ohio.", answers), 6
        )

    def test_shared_qk_block_selection(self) -> None:
        logits = torch.tensor([[0.0, 0.0, 4.0, 4.0, 1.0, 1.0]])
        self.assertEqual(
            select_shared_qk_blocks(
                logits, context_length=6, block_tokens=2, budget_blocks=2
            ),
            [1, 2],
        )

    def test_multisample_final_answer_scoring(self) -> None:
        text = "Reasoning: first hop.\nFinal answer: Dayton, Ohio.\nMore text"
        self.assertEqual(extract_first_final_answer(text), "Dayton, Ohio.")
        self.assertEqual(token_f1("Dayton, Ohio", "Dayton, Ohio"), 1.0)
        self.assertAlmostEqual(token_f1("Dayton", "Dayton, Ohio"), 2 / 3)


if __name__ == "__main__":
    unittest.main()
