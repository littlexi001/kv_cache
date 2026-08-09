from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_iterative_condition_retrieval import BM25Index  # noqa: E402
from run_model_guided_condition_retrieval import normalized_terms  # noqa: E402
from run_model_guided_condition_retrieval import (  # noqa: E402
    invalid_status_flags,
    multichannel_candidate_sentences,
    score_candidate_choice,
    score_relevance,
)
from annotate_stepwise_sufficiency import build_step_rows  # noqa: E402
from evaluate_stepwise_set_utility import (  # noqa: E402
    mode_inputs,
    mode_memory,
    select_evidence_span,
)
from analyze_stepwise_set_utility import mcnemar_exact_p  # noqa: E402
from profile_step_state_q import step_state_text  # noqa: E402
from run_step_state_kv_span_retrieval import (  # noqa: E402
    find_subsequence,
    specialist_heads,
    window_starts,
)
from analyze_branch_transition_verifier import choose_branch  # noqa: E402
from prepare_synthetic_controlled_corpus import (  # noqa: E402
    make_blind_test_payload_v5,
    make_blind_test_payload_v6,
)
from prepare_controlled_10m_mixed_corpus import build_block_mapping  # noqa: E402
from evaluate_global_step_hybrid_candidates import (  # noqa: E402
    AnchorInvertedIndex,
    reciprocal_rank_fusion,
    stable_rank,
    step_anchor_text,
)
from run_global_step_block_retrieval import (  # noqa: E402
    parse_profile_indices,
    select_profiles,
)
from run_global_candidate_sentence_kv_rerank import (  # noqa: E402
    MMapRawKIndex,
    overlap_fraction,
    rank_sentence_records,
)
from prepare_verified_chained_answer_steps import (  # noqa: E402
    clean_generated_state,
    rewrite_answer_step_with_generated_state,
)
from evaluate_global_step_branch_generation import generated_state_text  # noqa: E402
from aggregate_support_extract_branches import selected_indices  # noqa: E402
from profile_sparse_candidate_k import candidate_block_ids  # noqa: E402
from rerank_candidate_blocks_allhead import (  # noqa: E402
    rank_target as rank_allhead_target,
    select_channels,
)
from generate_atomic_subquestions import atomic_prompt, clean_subquestion  # noqa: E402
from prepare_musique_official_10m import (  # noqa: E402
    render_atomic_question,
    valid_two_hop,
)
from derive_real_2wiki_step_labels import (  # noqa: E402
    decompose_question,
    derive_chain,
    generic_decomposition,
    parse_passages,
)
from rerank_sparse_candidate_blocks_svd import (  # noqa: E402
    max_attention_diagnostics,
    max_attention_scores,
    rank_ids,
)
from build_sparse_k_svd_sidecar import fit_second_moment_basis  # noqa: E402
from analyze_train_calibrated_qk_fusion import fused_rank, zscore  # noqa: E402
from train_qk_rescue_gate import (  # noqa: E402
    choose_threshold,
    order_correlation,
    ranked_gap,
)
from analyze_qk_norm_bias import correlation, rank_values  # noqa: E402
from analyze_monotonic_qk_expansion import evaluate as evaluate_qk_expansion  # noqa: E402
from analyze_qk_channel_token_diagnostics import (  # noqa: E402
    profile_ranks,
    token_category,
)
from train_pairwise_qk_passage_head import (  # noqa: E402
    candidate_features,
    pairwise_examples,
    runtime_passage_scores,
)
from compare_paired_reader_runs import paired_ci  # noqa: E402
from prepare_lexical_sentence_branches import (  # noqa: E402
    overlap_fraction as sentence_overlap_fraction,
    rank_sentence_spans,
)
from evaluate_parallel_bridge_hypotheses import (  # noqa: E402
    equal_rrf_fusion,
    round_robin_fusion,
)
from analyze_bridge_hypothesis_allocations import (  # noqa: E402
    allocated_candidates,
    monotonic_allocations,
)
from rerank_parallel_bridge_candidates_svd import (  # noqa: E402
    aggregate_hypothesis_scores,
)
from train_dynamic_bridge_pool_head import dynamic_candidate_features  # noqa: E402
from evaluate_transition_support_verifier import (  # noqa: E402
    logit_group_margin,
    support_prompt,
)
from train_transition_support_head import (  # noqa: E402
    runtime_transition_scores,
    transition_features,
)
from summarize_deployed_strict_chain import paired_summary  # noqa: E402
from analyze_transition_confidence_gate import (  # noqa: E402
    confidence_record,
    gate_summary,
)
from prepare_confidence_gated_extensions import should_expand  # noqa: E402
from merge_gated_extension_generations import merge_generation_row  # noqa: E402
from compare_adaptive_extension import selected_hits  # noqa: E402


class StepwiseRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            "Harbor control uses Garden-0375 when communicating with LumaKela Dolphin.",
            "The retired label Garden-0375 occurs in a simulation whose light field is empty.",
            "The active wayfinding light carried by XaraQuorin Meridian shows crimson meridian.",
            "The active wayfinding light carried by LumaKela Dolphin shows scarlet voyager.",
        ]
        self.index = BM25Index(
            self.documents,
            min_df=1,
            max_df=1.0,
            k1=1.2,
            b=0.75,
        )

    def test_bm25_postings_scores_match_matrix_scores(self) -> None:
        queries = ["Garden-0375 LumaKela", "active wayfinding light"]
        np.testing.assert_allclose(
            self.index.score_postings(queries),
            self.index.score(queries),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    def test_sparse_svd_reranker_scores_and_tie_breaks_blocks(self) -> None:
        query = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        keys = torch.tensor(
            [
                [[[0.1, 0.0]], [[0.0, 0.2]]],
                [[[0.9, 0.0]], [[0.0, 0.1]]],
            ]
        )
        scores = max_attention_scores(query, keys)
        self.assertGreater(float(scores[1]), float(scores[0]))
        self.assertEqual(rank_ids([9, 4], [1.0, 1.0]), [4, 9])
        blocks, profiles, positions = max_attention_diagnostics(query, keys)
        self.assertEqual(tuple(blocks.shape), (2,))
        self.assertEqual(tuple(profiles.shape), (2, 1))
        self.assertEqual(tuple(positions.shape), (2, 1))

    def test_sparse_k_basis_is_orthonormal(self) -> None:
        rng = np.random.default_rng(3)
        raw = rng.normal(size=(4, 5, 2, 6)).astype(np.float16)
        basis, retained = fit_second_moment_basis(
            raw,
            np.asarray([0, 1, 2, 3], dtype=np.int64),
            rank=3,
            batch_blocks=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(basis.shape), (2, 6, 3))
        for profile_basis in basis:
            np.testing.assert_allclose(
                (profile_basis.T @ profile_basis).numpy(),
                np.eye(3),
                rtol=1.0e-5,
                atol=1.0e-5,
            )
        self.assertTrue(all(0.0 < value <= 1.0 for value in retained))

    def test_train_calibrated_fusion_preserves_candidate_identity(self) -> None:
        normalized = zscore([2.0, 2.0, 2.0])
        np.testing.assert_array_equal(normalized, np.zeros(3))
        row = {
            "candidate_candidates": [11, 12, 13],
            "target_block_id": 12,
            "bm25_scores": [3.0, 2.0, 1.0],
            "svd_scores": [0.0, 3.0, 1.0],
        }
        self.assertEqual(fused_rank(row, "svd", 0.0), 1)
        self.assertEqual(fused_rank(row, "svd", 1.0), 2)

    def test_qk_rescue_gate_uses_ranked_confidence_and_conservative_threshold(self) -> None:
        self.assertGreater(ranked_gap([1.0, 4.0, 2.0, 3.0], 0, 1), 0.0)
        self.assertAlmostEqual(order_correlation([1, 2, 3], [1, 2, 3]), 1.0)
        threshold, _ = choose_threshold(
            np.asarray([0.1, 0.9]),
            np.asarray([True, False]),
            np.asarray([False, True]),
        )
        self.assertGreater(threshold, 0.1)

    def test_qk_norm_and_expansion_diagnostics(self) -> None:
        self.assertAlmostEqual(correlation([1, 2, 3], [2, 4, 6]), 1.0)
        np.testing.assert_array_equal(rank_values([3, 1, 2]), [2, 0, 1])
        rows = [
            {
                "candidate_candidates": [1, 2, 3, 4],
                "svd_candidates": [4, 2, 1, 3],
                "target_block_id": 4,
            }
        ]
        result = evaluate_qk_expansion(rows, "svd", base_blocks=3, extras=1)
        self.assertEqual(result["qk_expansion_recall"], 1.0)
        self.assertEqual(result["equal_budget_lexical_recall"], 1.0)
        self.assertEqual(token_category("?"), "punctuation")
        self.assertEqual(token_category(" the"), "stopword")
        self.assertEqual(token_category(" Dayton"), "content")
        diagnostic_row = {
            "candidate_candidates": [1, 2],
            "target_block_id": 2,
            "full128_profile_scores": [[2.0, 0.0], [1.0, 3.0]],
        }
        self.assertEqual(profile_ranks(diagnostic_row, "full128"), [2, 1])

    def test_pairwise_passage_head_builds_symmetric_differences(self) -> None:
        row = {
            "candidate_candidates": [1, 2, 3],
            "target_block_id": 2,
            "bm25_scores": [3.0, 2.0, 1.0],
            "full128_profile_scores": [
                [1.0, 0.0],
                [3.0, 2.0],
                [0.0, 1.0],
            ],
            "svd_profile_scores": [
                [0.0, 1.0],
                [2.0, 3.0],
                [1.0, 0.0],
            ],
        }
        self.assertEqual(candidate_features(row, "both").shape, (3, 5))
        examples, labels, reachable = pairwise_examples([row], "full128")
        self.assertEqual(reachable, 1)
        self.assertEqual(examples.shape, (4, 3))
        np.testing.assert_array_equal(examples[0], -examples[1])
        np.testing.assert_array_equal(labels, [1, 0, 1, 0])
        scores = runtime_passage_scores(
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            {
                "feature_mean": [1.0, 1.0],
                "feature_scale": [2.0, 1.0],
                "linear_weight": [2.0, 1.0],
                "linear_intercept": 0.0,
            },
        )
        np.testing.assert_allclose(scores, [1.0, 5.0])
        interval = paired_ci(
            np.asarray([False, False]),
            np.asarray([True, True]),
            samples=100,
            seed=3,
        )
        np.testing.assert_allclose(interval, [1.0, 1.0])

    def test_lexical_sentence_selector_penalizes_passage_heading(self) -> None:
        class TinyTokenizer:
            @staticmethod
            def decode(values, skip_special_tokens=True):
                return " ".join(values)

        tokens = ["Passage:", "Orange", "River", "Origin", "is", "Thaba", "Putsoa."]
        ranked = rank_sentence_spans(
            tokens,
            [(0, 3), (3, 7)],
            "What is the origin of Orange River?",
            TinyTokenizer(),
        )
        self.assertEqual(ranked[0], (3, 7))
        self.assertEqual(sentence_overlap_fraction((3, 7), (5, 7)), 1.0)

    def test_multihop_step_labels_separate_bridge_and_answer_blocks(self) -> None:
        query = {
            "query_id": 4,
            "dataset": "synthetic_multihop",
            "task_type": "multihop",
            "split": "test",
            "question": "What light belongs to Harbor-0004?",
            "record_id": 0,
            "block_start": 0,
            "block_count": 100,
            "hard_negative_block_ids": [8],
            "gold_block_ids": [12, 72],
            "evidence_texts": [
                "Harbor-0004 refers to Luma Meridian.",
                "Luma Meridian shows crimson voyager.",
            ],
            "entity": "Luma Meridian",
            "answers": ["crimson voyager"],
        }
        steps = build_step_rows(query)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["target_output"], "Luma Meridian")
        self.assertNotIn("Luma Meridian", steps[0]["step_question"])
        self.assertIn("Harbor-0004", steps[0]["step_question"])
        self.assertEqual(steps[0]["lookup_key"], "Harbor-0004")
        self.assertEqual(steps[0]["step_operator"], "resolve_identifier")
        self.assertEqual(steps[0]["minimal_sufficient_block_ids"], [12])
        self.assertEqual(steps[1]["compact_state_before"], ["BRIDGE_ENTITY: Luma Meridian"])
        self.assertEqual(steps[1]["full_state_before"], [query["evidence_texts"][0]])
        self.assertEqual(steps[1]["minimal_sufficient_block_ids"], [72])
        self.assertEqual(steps[1]["previous_evidence_block_ids"], [12])

        target_ids, state = mode_inputs(steps[1], "target_with_state")
        self.assertEqual(target_ids, [72])
        self.assertEqual(state, ["BRIDGE_ENTITY: Luma Meridian"])
        combined_ids, combined_state = mode_inputs(
            steps[1], "target_plus_previous_with_state"
        )
        self.assertEqual(combined_ids, [72, 12])
        self.assertEqual(combined_state, state)
        no_state_ids, no_state = mode_inputs(steps[1], "target_no_state")
        self.assertEqual(no_state_ids, [72])
        self.assertEqual(no_state, [])

        fact_ids, fact_state = mode_inputs(steps[0], "fact_only")
        self.assertEqual(fact_ids, [])
        self.assertEqual(fact_state, [])
        fact_with_state_ids, fact_with_state = mode_inputs(
            steps[1], "fact_with_state"
        )
        self.assertEqual(fact_with_state_ids, [])
        self.assertEqual(fact_with_state, ["BRIDGE_ENTITY: Luma Meridian"])
        full_state_ids, full_state = mode_inputs(steps[1], "target_with_full_state")
        self.assertEqual(full_state_ids, [72])
        self.assertEqual(full_state, [query["evidence_texts"][0]])
        memory, memory_kind = mode_memory(
            None,
            np.empty((0, 0), dtype=np.int64),
            steps[1],
            "fact_with_state",
            [],
            ["BRIDGE_ENTITY: Luma Meridian"],
        )
        self.assertEqual(memory, query["evidence_texts"][1])
        self.assertEqual(memory_kind, "exact_target_fact")
        profiled_state = step_state_text(steps[1])
        self.assertIn("BRIDGE_ENTITY: Luma Meridian", profiled_state)
        self.assertNotIn("crimson voyager", profiled_state)

    def test_automatic_span_selection_uses_current_step_state(self) -> None:
        memory = (
            "A retired label appears in a simulation. "
            "Harbor control uses Garden-0375 to reach LumaKela Dolphin. "
            "The orientation lamp fitted to LumaKela Dolphin radiates scarlet voyager."
        )
        bridge_span = select_evidence_span(
            memory,
            "Which entity is linked to Garden-0375?",
        )
        answer_span = select_evidence_span(
            memory,
            "BRIDGE_ENTITY: LumaKela Dolphin What does the orientation lamp radiate?",
        )
        self.assertIn("Garden-0375", bridge_span)
        self.assertNotIn("scarlet voyager", bridge_span)
        self.assertIn("scarlet voyager", answer_span)

    def test_rare_anchor_requires_all_anchor_terms(self) -> None:
        question = (
            "Report the active wayfinding light shown by the craft addressed as Garden-0375."
        )
        anchor = self.index.rare_query_text(question, max_terms=1)
        self.assertEqual(anchor, "garden 0375")
        matching = [
            index
            for index, document in enumerate(self.documents)
            if normalized_terms(anchor) <= normalized_terms(document)
        ]
        self.assertEqual(matching, [0, 1])

    def test_novel_feedback_introduces_intermediate_entity(self) -> None:
        question = (
            "Report the active wayfinding light shown by the craft addressed as Garden-0375."
        )
        feedback = self.index.rare_novel_text(
            self.documents[0],
            exclude_text=question,
            max_terms=4,
        )
        self.assertIn("lumakela dolphin", feedback.casefold())
        scores = self.index.score([f"{feedback} {question}"])[0]
        self.assertGreater(float(scores[3]), float(scores[2]))
        sentence_ids, anchor, _, _ = multichannel_candidate_sentences(
            self.index,
            f"{feedback} {question}",
            scores,
            np.arange(len(self.documents), dtype=np.int32),
            candidate_blocks_count=4,
            anchor_candidate_blocks=2,
            anchor_terms=1,
            sentence_scan=4,
            anchor_text_override=feedback,
        )
        self.assertEqual(anchor, "LumaKela Dolphin")
        self.assertIn(3, sentence_ids)

    def test_global_candidate_fusion_is_deterministic_and_rewards_consensus(self) -> None:
        scores = np.asarray([0.5, 0.5, 0.2, 0.8], dtype=np.float32)
        self.assertEqual(stable_rank(scores, budget=3), [3, 0, 1])
        fused = reciprocal_rank_fusion([[4, 2, 1], [2, 3, 4]], budget=4, rrf_k=60.0)
        self.assertEqual(fused[0], 2)
        self.assertEqual(set(fused), {1, 2, 3, 4})

    def test_operator_profile_routes_preserve_contiguous_views(self) -> None:
        self.assertEqual(parse_profile_indices("", 4), [0, 1, 2, 3])
        self.assertEqual(parse_profile_indices("2,3", 4), [2, 3])
        values = np.arange(2 * 4 * 3).reshape(2, 4, 3)
        tensor = torch.from_numpy(values)
        selected = select_profiles(tensor, [2, 3], axis=1)
        self.assertEqual(tuple(selected.shape), (2, 2, 3))
        self.assertEqual(selected.data_ptr(), tensor.data_ptr() + 2 * 3 * tensor.element_size())

    def test_anchor_index_routes_compact_state_without_target_value(self) -> None:
        documents = [
            "Garden-0375 is retired and contains no active light.",
            "Garden-0375 refers to LumaKela Dolphin in the active registry.",
            "The orientation lamp on LumaKela Dolphin radiates scarlet voyager.",
        ]
        index = AnchorInvertedIndex(documents)
        first = {
            "lookup_key": "Garden-0375",
            "compact_state_before": [],
        }
        second = {
            "lookup_key": "Garden-0375",
            "compact_state_before": ["BRIDGE_ENTITY: LumaKela Dolphin"],
        }
        self.assertEqual(step_anchor_text(first), "Garden-0375")
        self.assertEqual(step_anchor_text(second), "LumaKela Dolphin")
        self.assertEqual(
            index.search("LumaKela Dolphin", "What does its lamp radiate?", 4), [2, 1]
        )

    def test_sentence_records_rank_blocks_and_measure_target_overlap(self) -> None:
        ranked, blocks = rank_sentence_records(
            [
                {"block_id": 9, "start": 0, "end": 8, "score": 0.2},
                {"block_id": 4, "start": 10, "end": 20, "score": 0.8},
                {"block_id": 9, "start": 8, "end": 16, "score": 0.7},
            ]
        )
        self.assertEqual([item["block_id"] for item in ranked], [4, 9, 9])
        self.assertEqual(blocks, [4, 9])
        self.assertEqual(overlap_fraction((8, 20), (10, 15)), 1.0)

    def test_generated_state_cleaner_is_generic(self) -> None:
        self.assertEqual(
            clean_generated_state(" Final answer: LumaKela Dolphin.\n"),
            "LumaKela Dolphin",
        )

    def test_generated_bridge_rewrites_every_active_retrieval_field(self) -> None:
        rewritten = rewrite_answer_step_with_generated_state(
            {
                "lookup_key": "Golestan Province",
                "step_question": "Where is Golestan Province located?",
                "question": "Where is the place reached in the first step located?",
                "retrieval_state": (
                    "Golestan Province Where is Golestan Province located?"
                ),
                "official_raw_step_question": "Where is #1 located?",
                "compact_state_before": ["BRIDGE_ENTITY: Golestan Province"],
                "full_state_before": ["BRIDGE_ENTITY: Golestan Province"],
            },
            "Gonbad-e Qabus County",
        )
        self.assertEqual(rewritten["lookup_key"], "Gonbad-e Qabus County")
        self.assertEqual(
            rewritten["step_question"],
            "Where is Gonbad-e Qabus County located?",
        )
        self.assertEqual(
            rewritten["retrieval_state"],
            "Gonbad-e Qabus County Where is Gonbad-e Qabus County located?",
        )
        self.assertEqual(
            rewritten["compact_state_before"],
            ["BRIDGE_ENTITY: Gonbad-e Qabus County"],
        )
        self.assertNotIn("Golestan Province", rewritten["retrieval_state"])

    def test_parallel_hypothesis_fusion_is_unique_and_deterministic(self) -> None:
        rankings = [[5, 2, 8], [7, 2, 9], [6, 5, 10]]
        self.assertEqual(
            round_robin_fusion(rankings, [1, 0, 2]),
            [7, 5, 6, 2, 9, 8, 10],
        )
        rrf = equal_rrf_fusion(rankings, 60.0)
        self.assertEqual(rrf[0], 5)
        self.assertEqual(len(rrf), len(set(rrf)))

    def test_hypothesis_allocation_follows_verifier_rank(self) -> None:
        rankings = [[5, 2, 8], [7, 2, 9], [6, 5, 10]]
        self.assertEqual(
            allocated_candidates(rankings, [1, 0, 2], [2, 1, 0]),
            [7, 2, 5],
        )
        self.assertEqual(
            monotonic_allocations(3),
            [(1, 1, 1), (2, 1, 0), (3, 0, 0)],
        )

    def test_parallel_q_aggregation_applies_verifier_bonus(self) -> None:
        scores = np.asarray([[1.0, 0.0], [0.5, 2.0], [0.0, 1.0]])
        methods = aggregate_hypothesis_scores(
            scores,
            selected_index=1,
            branch_order=[1, 0, 2],
            bonus_lambdas=[1.0],
        )
        np.testing.assert_array_equal(methods["selected_passage"], [0.5, 2.0])
        np.testing.assert_array_equal(methods["max_passage"], [1.0, 2.0])
        np.testing.assert_array_equal(methods["bonus_1p0"], [2.5, 4.0])

    def test_dynamic_pool_features_are_candidate_aligned(self) -> None:
        row = {
            "selected_index": 0,
            "hypothesis_passage_scores": [[1.0, 0.0], [0.0, 2.0]],
            "hypothesis_bm25_scores": [[3.0, 1.0], [1.0, 4.0]],
            "hypothesis_candidate_ranks": [[1, 2], [2, 1]],
        }
        features = dynamic_candidate_features(row)
        self.assertEqual(features.shape, (2, 10))
        self.assertEqual(features[0, 0], 1.0)
        self.assertEqual(features[1, 1], 2.0)

    def test_transition_support_margin_and_prompt(self) -> None:
        logits = torch.tensor([[0.0, 2.0, 1.0, -1.0]])
        margin = logit_group_margin(logits, [1, 2], [0, 3])
        self.assertGreater(float(margin[0]), 1.0)
        prompt = support_prompt("Who directed it?", "It was directed by A.", "A")
        self.assertIn("Proposed answer:\nA", prompt)
        self.assertIn("only Yes or No", prompt)

    def test_transition_features_preserve_branch_alignment(self) -> None:
        row = {
            "heuristic_trace": [
                {
                    "branch_index": 1,
                    "score": 2.0,
                    "anchor_present": False,
                    "repeats_anchor": True,
                    "relation_supported": False,
                    "output_grounded": True,
                    "query_memory_overlap": 1,
                    "grounding_ratio": 0.5,
                    "novel_grounded_terms": ["b"],
                },
                {
                    "branch_index": 0,
                    "score": 4.0,
                    "anchor_present": True,
                    "repeats_anchor": False,
                    "relation_supported": True,
                    "output_grounded": True,
                    "query_memory_overlap": 2,
                    "grounding_ratio": 1.0,
                    "novel_grounded_terms": ["a", "c"],
                },
            ],
            "yes_no_scores": [3.0, 1.0],
            "answer_logprob_scores": [-0.1, -1.0],
            "branch_retrieval_ranks": [1, 2],
            "branch_generated_tokens": [3, 5],
        }
        features = transition_features(row)
        self.assertEqual(features.shape, (2, 12))
        self.assertEqual(features[0, 0], 4.0)
        self.assertEqual(features[0, 1], 3.0)
        self.assertEqual(features[1, 10], -2.0)

    def test_runtime_transition_scores_apply_exported_scaler(self) -> None:
        scores = runtime_transition_scores(
            np.asarray([[3.0, 5.0], [1.0, 1.0]]),
            {
                "feature_mean": [1.0, 1.0],
                "feature_scale": [2.0, 4.0],
                "linear_weight": [2.0, 3.0],
                "linear_intercept": -1.0,
            },
        )
        np.testing.assert_allclose(scores, [4.0, -1.0])

    def test_strict_chain_paired_summary_counts_direction(self) -> None:
        summary = paired_summary(
            [True, False, False, True],
            [True, True, False, False],
        )
        self.assertEqual(summary["baseline_hits"], 2)
        self.assertEqual(summary["selected_hits"], 2)
        self.assertEqual(summary["wins_losses"], [1, 1])

    def test_transition_confidence_gate_tracks_expansion_opportunities(self) -> None:
        record = confidence_record(
            {
                "query_id": 9,
                "branches": [{"target_hit": False}, {"target_hit": True}],
                "any_branch_target_hit": True,
            },
            {
                "heuristic_structured_index": 0,
                "heuristic_structured_scores": [2.0, 1.5],
            },
            method="heuristic_structured",
            retrieval_rank=8,
        )
        self.assertTrue(record["expansion_opportunity"])
        summary = gate_summary([record], "margin", threshold=0.5)
        self.assertEqual(summary["expansion_fraction"], 1.0)
        self.assertEqual(summary["expansion_opportunity_capture"], 1.0)

    def test_confidence_extension_and_generation_merge(self) -> None:
        self.assertTrue(
            should_expand(
                {"head_scores": [1.0, 2.0, 1.5]}, "head", threshold=2.0
            )
        )
        base = {
            "branches": [
                {
                    "rank": 1,
                    "selected_block": 3,
                    "retrieval_target_span_hit": False,
                    "target_hit": False,
                    "target_f1": 0.0,
                    "generation_seconds": 0.2,
                }
            ]
        }
        extension = {
            "branches": [
                {
                    "rank": 4,
                    "selected_block": 8,
                    "retrieval_target_span_hit": True,
                    "target_hit": True,
                    "target_f1": 1.0,
                    "generation_seconds": 0.3,
                }
            ]
        }
        merged = merge_generation_row(base, extension)
        self.assertTrue(merged["any_branch_target_hit"])
        self.assertEqual(len(merged["branches"]), 2)
        self.assertAlmostEqual(merged["total_branch_generation_seconds"], 0.5)
        hits = selected_hits(
            {1: merged},
            {1: {"head_index": 1}},
            [1],
            "head",
        )
        self.assertEqual(hits, [True])

    def test_v6_multihop_template_is_disjoint_and_ordered(self) -> None:
        payload = make_blind_test_payload_v6(380, "multihop")
        self.assertIn("routing token", payload["question"])
        self.assertIn(payload["entity"], payload["evidence"][0])
        self.assertIn(payload["entity"], payload["evidence"][1])
        self.assertIn(payload["answer"], payload["evidence"][1])
        self.assertNotEqual(
            payload["question"], make_blind_test_payload_v5(380, "multihop")["question"]
        )

    def test_sparse_profile_candidate_ids_are_unique_and_sorted(self) -> None:
        rows = [
            {"anchor_candidates": [9, 2, 9, 4]},
            {"anchor_candidates": [4, 3]},
        ]
        self.assertEqual(candidate_block_ids(rows, "anchor_candidates", 3), [2, 3, 4, 9])

    def test_real_2wiki_chain_derivation_requires_unique_title_path(self) -> None:
        context = (
            "Passage 1:\nLou Breslow\nLou Breslow's wife was Marion Byron.\n\n"
            "Passage 2:\nMarion Byron\nMarion Byron was born in Dayton, Ohio.\n"
        )
        self.assertEqual(len(parse_passages(context)), 2)
        chain, reason = derive_chain(
            "Where was the wife of Lou Breslow born?", ["Dayton, Ohio"], context
        )
        self.assertEqual(reason, "ok")
        self.assertEqual(chain["lookup_key"], "Lou Breslow")
        self.assertEqual(chain["bridge"], "Marion Byron")
        decomposition = decompose_question(
            "Who is the spouse of the director of film Emergency Wedding?",
            "Emergency Wedding",
        )
        self.assertEqual(decomposition["step0_operator"], "resolve_director")

        generic = generic_decomposition(
            "Where was the wife of Lou Breslow born?", "Lou Breslow"
        )
        self.assertEqual(generic["step0_operator"], "generic_link")
        self.assertEqual(generic["step1_operator"], "generic_answer")
        self.assertIn("Where was the wife", generic["step0_question"])

    def test_generic_verifier_prefers_exactly_grounded_output(self) -> None:
        step = {
            "step_type": "resolve_bridge",
            "step_operator": "generic_link",
            "lookup_key": "Lou Breslow",
            "question": "Where was the wife of Lou Breslow born?",
            "step_question": "Which new entity is needed for the next lookup?",
            "compact_state_before": [],
        }
        branches = [
            {
                "memory_text": "Lou Breslow worked with several actors.",
                "generated_text": "Alice Example",
            },
            {
                "memory_text": "Lou Breslow's wife was Marion Byron.",
                "generated_text": "Marion Byron",
            },
        ]
        selected, scores = choose_branch(step, branches)
        self.assertEqual(selected, 1)
        self.assertTrue(scores[1]["output_grounded"])

    def test_allhead_channel_selection_uses_train_target_ranks(self) -> None:
        scores = np.asarray(
            [
                [[0.9, 0.2, 0.1], [0.1, 0.2, 0.9]],
                [[0.1, 0.8, 0.2], [0.8, 0.1, 0.2]],
            ],
            dtype=np.float32,
        )
        rows = [
            {"split": "train", "step_type": "resolve_bridge"},
            {"split": "train", "step_type": "resolve_bridge"},
        ]
        channels = select_channels(scores, rows, [0, 1], "resolve_bridge", 1)
        self.assertEqual(channels, [0])
        self.assertEqual(rank_allhead_target(scores[0, 0], 0), 1)

    def test_atomic_subquestion_prompt_does_not_use_target_label(self) -> None:
        step = {
            "step_type": "resolve_bridge",
            "question": "Where was the wife of Lou Breslow born?",
            "lookup_key": "Lou Breslow",
            "target_output": "SECRET_TARGET",
        }
        prompt = atomic_prompt(step)
        self.assertNotIn("SECRET_TARGET", prompt)
        self.assertEqual(clean_subquestion("NEXT: Who is Lou's wife?"), "Who is Lou's wife?")

    def test_official_musique_decomposition_is_rendered_without_relation_rules(self) -> None:
        row = {
            "answerable": True,
            "paragraphs": [
                {"title": "A", "paragraph_text": "Film A stars Person B."},
                {"title": "B", "paragraph_text": "Person B married Person C."},
            ],
            "question_decomposition": [
                {
                    "question": "Film A >> performer",
                    "answer": "Person B",
                    "paragraph_support_idx": 0,
                },
                {
                    "question": "#1 >> spouse",
                    "answer": "Person C",
                    "paragraph_support_idx": 1,
                },
            ],
        }
        self.assertTrue(valid_two_hop(row))
        self.assertEqual(
            render_atomic_question("#1 >> spouse", ["Person B"]),
            "What is the spouse of Person B?",
        )

    def test_alias_aware_anchor_recovers_surname_reference(self) -> None:
        documents = [
            "Edward Buzzell directed Emergency Wedding.",
            "In 1926, Buzzell married actress Ona Munson.",
            "Edward Buzzell also worked as a screenwriter.",
        ]
        index = AnchorInvertedIndex(documents)
        ranked = index.search_alias_aware(
            "Edward Buzzell", "Who did Edward Buzzell marry?", budget=3
        )
        self.assertIn(1, ranked)

    def test_typed_spouse_verifier_prefers_relation_supported_alias_memory(self) -> None:
        step = {
            "step_type": "resolve_answer_from_bridge",
            "step_operator": "resolve_spouse",
            "question": "Who is the spouse of the director?",
            "compact_state_before": ["BRIDGE_ENTITY: Edward Buzzell"],
        }
        branches = [
            {
                "memory_text": "Edward Buzzell directed the film starring Barbara Hale.",
                "generated_text": "Edward Buzzell's spouse is Barbara Hale.",
            },
            {
                "memory_text": "Buzzell married actress Ona Munson in 1926.",
                "generated_text": "Edward Buzzell's spouse is Ona Munson.",
            },
        ]
        selected, _ = choose_branch(step, branches)
        self.assertEqual(selected, 1)

    def test_generic_invalid_status_filter(self) -> None:
        sentences = [
            "Only amber anchor carries legal force as the unit's permission grade.",
            "The motion lapsed and acquired no force.",
            "A discarded badge mock-up lacks any approved operative entry.",
            "An archived exercise uses the code as a fictional label with no deployed unit.",
        ]
        self.assertEqual(invalid_status_flags(sentences).tolist(), [0.0, 1.0, 1.0, 1.0])

    def test_empty_model_candidate_batch_is_a_noop(self) -> None:
        relevance = score_relevance(
            None,
            None,
            None,
            question="question",
            search_need="need",
            sentences=[],
            yes_token_id=0,
            no_token_id=1,
        )
        choices = score_candidate_choice(
            None,
            None,
            None,
            question="question",
            search_need="need",
            sentences=[],
        )
        self.assertEqual(relevance.shape, (0,))
        self.assertEqual(choices.shape, (0,))
        self.assertEqual(relevance.dtype, np.float32)
        self.assertEqual(choices.dtype, np.float32)

    def test_mcnemar_exact_p_handles_balanced_and_one_sided_pairs(self) -> None:
        self.assertEqual(mcnemar_exact_p(0, 0), 1.0)
        self.assertEqual(mcnemar_exact_p(1, 1), 1.0)
        self.assertAlmostEqual(mcnemar_exact_p(0, 5), 0.0625)

    def test_reasoned_bridge_parser_uses_only_structured_entity_line(self) -> None:
        generated = (
            "Relevant fact: Alex P. Keaton was played by Michael J. Fox.\n"
            "Bridge entity: Michael J. Fox"
        )
        self.assertEqual(
            generated_state_text(generated, "bridge_reasoned"), "Michael J. Fox"
        )
        self.assertEqual(generated_state_text(generated, "atomic"), generated)
        self.assertEqual(
            generated_state_text("Relevant fact: unsupported", "bridge_reasoned"), ""
        )

    def test_support_extract_parser_and_deterministic_aggregation(self) -> None:
        self.assertEqual(
            generated_state_text("SUPPORTED: Dayton, Ohio", "support_extract"),
            "Dayton, Ohio",
        )
        self.assertEqual(
            generated_state_text("NOT_SUPPORTED", "support_extract"), ""
        )
        row = {
            "branches": [
                {
                    "block_rank": 1,
                    "state_text": "",
                    "memory_text": "irrelevant",
                },
                {
                    "block_rank": 2,
                    "state_text": "Dayton, Ohio",
                    "memory_text": "The answer is Dayton, Ohio.",
                },
                {
                    "block_rank": 3,
                    "state_text": "Dayton, Ohio",
                    "memory_text": "Dayton, Ohio is supported.",
                },
            ]
        }
        self.assertEqual(
            selected_indices(row),
            {"first_supported": 1, "first_grounded": 1, "consensus": 1},
        )

    def test_kv_span_helpers_find_fact_and_train_specialist_head(self) -> None:
        self.assertEqual(find_subsequence([1, 2, 3, 2], [2, 3]), (1, 3))
        starts = window_starts(256, 32, 10)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 224)
        examples = [
            {
                "split": "train",
                "scenario": "target_plus_negative",
                "step_type": "resolve_bridge",
                "features": np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
                "overlap": np.asarray([1.0, 0.0], dtype=np.float32),
            }
        ]
        specialists = specialist_heads(examples, count=1)
        self.assertEqual(specialists["resolve_bridge"]["indices"].tolist(), [0])

    def test_transition_verifier_prefers_grounded_new_entity(self) -> None:
        step = {
            "step_type": "resolve_bridge",
            "lookup_key": "Garden-0375",
            "question": "What light belongs to Garden-0375?",
            "compact_state_before": [],
        }
        branches = [
            {
                "memory_text": "A drill borrowed Garden-0375 for an imaginary unit.",
                "generated_text": "Garden-0375",
            },
            {
                "memory_text": "Operators use Garden-0375 to reach LumaKela Dolphin.",
                "generated_text": "LumaKela Dolphin",
            },
        ]
        selected, scores = choose_branch(step, branches)
        self.assertEqual(selected, 1)
        self.assertGreater(scores[1]["score"], scores[0]["score"])

    def test_v5_multihop_template_separates_mapping_and_answer(self) -> None:
        payload = make_blind_test_payload_v5(375, "multihop")
        self.assertIn("pairs", payload["evidence"][0])
        self.assertIn(payload["entity"], payload["evidence"][0])
        self.assertNotIn(payload["answer"], payload["evidence"][0])
        self.assertIn(payload["answer"], payload["evidence"][1])

    def test_mixed_corpus_mapping_is_deterministic_and_one_to_one(self) -> None:
        first = build_block_mapping(100, 12, seed=7)
        second = build_block_mapping(100, 12, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(range(12)))
        self.assertEqual(len(set(first.values())), 12)


if __name__ == "__main__":
    unittest.main()
