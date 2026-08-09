from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_global_bridge_controller_batch import (  # noqa: E402
    first_rank_in_record,
    gold_ranks,
    optional_rank_in_record,
    ranked_block_ids,
    record_range_for_block,
    merge_channel_selections,
    select_queries,
    selected_hits,
    summarize_rows,
)
from run_global_dynamic_svd_kv_single import mask_scores_to_block_range  # noqa: E402
from analyze_bm25_block_windows import best_contiguous_window  # noqa: E402
from analyze_bridge_controller_ablation import (  # noqa: E402
    paired_binary,
    paired_continuous,
)
from analyze_bridge_entity_extraction import quota_merge  # noqa: E402
from verified_step_state import (  # noqa: E402
    AtomicFact,
    StepAction,
    fact_supported,
    fact_chain_connects,
    final_action_verified,
    parse_step_action,
)


class GlobalBridgeBatchTest(unittest.TestCase):
    def test_query_selection_freezes_development_examples_out(self) -> None:
        queries = [
            {"query_id": 0, "dataset": "2wikimqa"},
            {"query_id": 1, "dataset": "hotpotqa"},
            {"query_id": 2, "dataset": "qasper"},
            {"query_id": 6, "dataset": "2wikimqa"},
            {"query_id": 9, "dataset": "musique"},
        ]
        selected = select_queries(
            queries,
            datasets={"2wikimqa", "hotpotqa", "musique"},
            query_ids=set(),
            excluded_ids={0, 6},
            max_queries=0,
        )
        self.assertEqual([item["query_id"] for item in selected], [1, 9])

    def test_ranking_and_hits_keep_record_and_gold_separate(self) -> None:
        scores = np.asarray([0.1, 0.8, 0.8, 0.4, 0.3], dtype=np.float32)
        ranked = ranked_block_ids(scores)
        self.assertEqual(ranked, [1, 2, 3, 4, 0])
        self.assertEqual(first_rank_in_record(ranked, 2, 2), 2)
        self.assertEqual(optional_rank_in_record(ranked[:1], 2, 2), None)
        self.assertEqual(gold_ranks(ranked, [3]), {"3": 3})
        self.assertEqual(
            selected_hits([2], block_start=2, block_count=2, gold_block_ids=[3]),
            (True, False),
        )

    def test_summary_excludes_controller_errors_from_rates(self) -> None:
        common = {
            "initial_retriever": "qk",
            "question_bm25_record_hit": True,
            "question_bm25_gold_hit": False,
            "hop1_record_hit": True,
            "hop1_gold_hit": False,
            "any_search_record_hit": True,
            "any_search_gold_hit": True,
            "final_search_gold_hit": True,
            "answer_hit": True,
            "answer_f1": 0.75,
            "qk_capture_seconds": 0.1,
            "qk_retrieval_seconds": 0.2,
            "bridge_bm25_seconds": 0.3,
            "bridge_generation_seconds": 0.4,
            "answer_generation_seconds": 0.5,
            "initial_retrieval_seconds": 0.2,
            "online_seconds": 1.5,
        }
        rows = [
            {**common, "controller_error": None},
            {**common, "controller_error": "parse failed", "answer_hit": False},
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["queries"], 2)
        self.assertEqual(summary["valid_queries"], 1)
        self.assertEqual(summary["controller_errors"], 1)
        self.assertEqual(summary["answer_hit_rate"], 1.0)
        self.assertEqual(summary["all_query_answer_hit_rate"], 0.5)
        self.assertEqual(summary["mean_answer_f1"], 0.75)
        self.assertEqual(summary["median_online_seconds"], 1.5)
        self.assertAlmostEqual(summary["mean_qk_retrieval_seconds"], 0.2)

    def test_record_route_maps_global_block_to_half_open_range(self) -> None:
        records = [
            {"block_start": 0, "block_count": 3},
            {"block_start": 3, "block_count": 2},
        ]
        self.assertEqual(record_range_for_block(records, 3), (3, 5))
        self.assertEqual(record_range_for_block(records, 4), (3, 5))
        with self.assertRaises(ValueError):
            record_range_for_block(records, 5)

    def test_block_range_mask_applies_to_coarse_and_exact_shapes(self) -> None:
        local_scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        local_ids = torch.tensor([10, 11, 12, 13])
        masked = mask_scores_to_block_range(local_scores, local_ids, 11, 13)
        self.assertTrue(torch.isneginf(masked[0, 0]))
        self.assertEqual(masked[0, 1:3].tolist(), [2.0, 3.0])
        self.assertTrue(torch.isneginf(masked[0, 3]))

        exact_scores = torch.tensor([[4.0, 3.0], [2.0, 1.0]])
        candidate_ids = torch.tensor([[12, 30], [11, 31]])
        exact_masked = mask_scores_to_block_range(
            exact_scores, candidate_ids, 10, 20
        )
        self.assertEqual(exact_masked[:, 0].tolist(), [4.0, 2.0])
        self.assertTrue(torch.isneginf(exact_masked[:, 1]).all())

    def test_contiguous_window_does_not_cross_record_boundary(self) -> None:
        scores = np.asarray([0.0, 5.0, 5.0, 0.0], dtype=np.float32)
        records = [
            {"block_start": 0, "block_count": 2},
            {"block_start": 2, "block_count": 2},
        ]
        selected = best_contiguous_window(scores, records, 3, "sum")
        self.assertEqual(selected, [0, 1])

    def test_paired_binary_reports_query_level_wins_and_losses(self) -> None:
        baseline = [
            {"query_id": 1, "answer_hit": False},
            {"query_id": 2, "answer_hit": True},
        ]
        candidate = [
            {"query_id": 1, "answer_hit": True},
            {"query_id": 2, "answer_hit": False},
        ]
        result = paired_binary(baseline, candidate, "answer_hit")
        self.assertEqual(result["wins"], [1])
        self.assertEqual(result["losses"], [2])
        self.assertEqual(result["net"], 0)
        self.assertEqual(result["mcnemar_exact_p"], 1.0)

        for row, value in zip(baseline, [0.0, 1.0], strict=True):
            row["answer_f1"] = value
        for row, value in zip(candidate, [1.0, 0.0], strict=True):
            row["answer_f1"] = value
        continuous = paired_continuous(
            baseline, candidate, "answer_f1", bootstrap_samples=100
        )
        self.assertEqual(continuous["mean_delta"], 0.0)
        self.assertEqual(continuous["wins"], 1)
        self.assertEqual(continuous["losses"], 1)

    def test_verified_final_requires_supported_connected_facts(self) -> None:
        memory = (
            "Arnold Richards was a former chair of YIVO. "
            "YIVO is a member organization of the Center for Jewish History."
        )
        action = StepAction(
            facts=(
                AtomicFact("Arnold Richards", "former chair", "YIVO"),
                AtomicFact("YIVO", "member of", "Center for Jewish History"),
            ),
            kind="final",
            value="YIVO",
        )
        question = (
            "Arnold Richards was the former chair of what organization that is a "
            "member of the Center for Jewish History?"
        )
        self.assertTrue(final_action_verified(action, memory, question))

        unsupported = StepAction(
            facts=(AtomicFact("Arnold Richards", "former chair", "Made Up Org"),),
            kind="final",
            value="Made Up Org",
        )
        self.assertFalse(final_action_verified(unsupported, memory, question))

    def test_step_action_parser_requires_one_action(self) -> None:
        parsed = parse_step_action(
            "FACT: Lou Breslow | spouse | Marion Byron | Lou Breslow married Marion Byron\n"
            "SEARCH: Marion Byron | place of birth"
        )
        self.assertEqual(parsed.kind, "search")
        self.assertEqual(parsed.value, "Marion Byron | place of birth")
        self.assertEqual(parsed.facts[0].object, "Marion Byron")
        self.assertIn("married", parsed.facts[0].evidence)

    def test_fact_support_does_not_join_unrelated_blocks(self) -> None:
        fact = AtomicFact("Arnold Richards", "former chair", "YIVO")
        memory = "Arnold Richards was a psychoanalyst.\n\nYIVO preserves history."
        self.assertFalse(fact_supported(fact, memory))

        quoted = AtomicFact(
            "Arnold Richards",
            "former chair",
            "YIVO",
            "Arnold Richards was a former chair of YIVO",
        )
        supported_memory = "Arnold Richards was a former chair of YIVO."
        self.assertTrue(
            fact_supported(quoted, supported_memory, require_evidence=True)
        )
        self.assertFalse(
            fact_supported(fact, supported_memory, require_evidence=True)
        )

    def test_verified_fact_ledger_connects_across_retrieval_steps(self) -> None:
        facts = [
            AtomicFact("Lou Breslow", "spouse", "Marion Byron"),
            AtomicFact("Marion Byron", "born in", "Dayton, Ohio"),
        ]
        question = "Where was the wife of Lou Breslow born?"
        self.assertTrue(fact_chain_connects(facts, "Dayton, Ohio", question))
        self.assertFalse(fact_chain_connects(facts[:1], "Dayton, Ohio", question))

    def test_bridge_channel_quota_preserves_both_channels(self) -> None:
        selected = quota_merge([1, 2, 3], [2, 4, 5], 2, 3)
        self.assertEqual(selected, [1, 2, 4])
        runtime_selected = merge_channel_selections(
            [1, 2, 3], [2, 4, 5], primary_quota=2, target_blocks=3
        )
        self.assertEqual(runtime_selected, selected)


if __name__ == "__main__":
    unittest.main()
