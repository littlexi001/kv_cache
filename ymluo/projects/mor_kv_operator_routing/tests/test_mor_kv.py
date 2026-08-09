from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from run_mor_kv_offline import (  # noqa: E402
    Action,
    ScoreSignatureRouter,
    combine_rankings,
    gqa_key,
    query_metric,
    qk_ranking,
)
from analyze_natural_operator_library import (  # noqa: E402
    question_features,
    ridge_predict,
    route_from_predictions,
)
from analyze_proxy_route import (  # noqa: E402
    actionwise_proxy_regret_predict,
    route_proxy,
    within_query_zscore,
)
from run_lodo_natural_specialist_retrieval import (  # noqa: E402
    combine_rankings as combine_specialist_rankings,
    compile_specialists,
)
from build_operator_portfolios import allocate_portfolio  # noqa: E402
from train_head_distortion_router import (  # noqa: E402
    SCORE_FEATURES,
    choose_alphas_group_cv_multi,
    conformal_corrections,
    feature_vector,
    gqa_physical_summary,
    higher_quantile,
)
from merge_sparse_attention_reference_nll import paired_bootstrap_ci  # noqa: E402
from evaluate_sparse_attention_reference_nll import (  # noqa: E402
    first_answer_token,
    select_shared_mass_blocks,
    select_shared_output_blocks,
    summarize_head_rows,
)
from benchmark_gqa_grouped_sdpa import (  # noqa: E402
    evenly_spaced_blocks,
    pack_ragged_indices,
    token_indices_for_blocks,
)
from analyze_query_risk_gate import (  # noqa: E402
    average_ranks,
    correlation,
    parse_layer_weights,
)
from analyze_compiled_gqa_oracle import (  # noqa: E402
    physical_union_summary,
    structured_gqa_oracle_summary,
)
from profile_postrope_k_basis import basis_from_moments  # noqa: E402


class MorKvTest(unittest.TestCase):
    def test_postrope_basis_recovers_dominant_axis(self) -> None:
        samples = torch.tensor(
            [[[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]
        )
        sums = samples.sum(dim=1)
        grams = torch.einsum("ksd,kse->kde", samples, samples)
        basis, retained = basis_from_moments(sums, grams, count=4, target_rank=1)
        self.assertAlmostEqual(abs(float(basis[0, 0, 0])), 1.0, places=6)
        self.assertAlmostEqual(float(retained[0]), 1.0, places=6)

    def test_multi_action_alpha_cv_returns_each_action(self) -> None:
        matrix = np.arange(12, dtype=np.float64).reshape(6, 2)
        targets = np.column_stack([matrix[:, 0], np.ones(6)])
        selected = choose_alphas_group_cv_multi(
            matrix,
            targets,
            np.arange(6),
            ["linear", "constant"],
            [0.1, 1.0],
        )
        self.assertEqual(set(selected), {"linear", "constant"})
        self.assertTrue(all(value in {0.1, 1.0} for value in selected.values()))

    def test_layer_score_interaction_is_layer_local(self) -> None:
        row = {
            "layer": 1,
            "query_head": 0,
            **{
                name: float(index + 1)
                for index, name in enumerate(SCORE_FEATURES)
            },
        }
        features = feature_vector(row, 2, 1, interaction_mode="layer_score")
        width = len(SCORE_FEATURES)
        interaction = features[-2 * width :]
        self.assertEqual(interaction[:width].tolist(), [0.0] * width)
        self.assertEqual(
            interaction[width:].tolist(),
            [float(index + 1) for index in range(width)],
        )

    def test_compiled_gqa_oracle_unions_shared_head_blocks(self) -> None:
        compiled = [
            {
                "query_id": 0,
                "layer": 0,
                "query_head": head,
                "query_position": 9,
                "chosen_action": "sparse",
                "selected_blocks": 2,
            }
            for head in (0, 1)
        ]
        selected = {
            (0, 0, 0, 9, "sparse"): {0, 1},
            (0, 0, 1, 9, "sparse"): {1, 2},
        }
        full = {
            (0, 0, 0, 9): {0, 1, 2, 3},
            (0, 0, 1, 9): {0, 1, 2, 3},
        }
        summary = physical_union_summary(compiled, selected, full, 2)
        self.assertEqual(summary["mean_physical_gqa_blocks"], 3.0)
        self.assertEqual(summary["physical_gqa_saving_rate"], 0.25)

    def test_structured_gqa_oracle_prefers_shared_physical_union(self) -> None:
        action_rows = {
            (0, 0, 0, 9): [
                {"action": "full", "selected_blocks": 4, "selected_block_ids": {0, 1, 2, 3}, "relative_output_l2": 0.0},
                {"action": "a", "selected_blocks": 2, "selected_block_ids": {0, 1}, "relative_output_l2": 0.01},
                {"action": "b", "selected_blocks": 2, "selected_block_ids": {2, 3}, "relative_output_l2": 0.02},
            ],
            (0, 0, 1, 9): [
                {"action": "full", "selected_blocks": 4, "selected_block_ids": {0, 1, 2, 3}, "relative_output_l2": 0.0},
                {"action": "a", "selected_blocks": 2, "selected_block_ids": {2, 3}, "relative_output_l2": 0.01},
                {"action": "b", "selected_blocks": 2, "selected_block_ids": {0, 1}, "relative_output_l2": 0.02},
            ],
        }
        summary = structured_gqa_oracle_summary(action_rows, 0.05, 2)
        self.assertEqual(summary["mean_physical_gqa_blocks"], 2.0)
        self.assertEqual(summary["physical_gqa_saving_rate"], 0.5)

    def test_shared_mass_oracle_uses_one_budgeted_block_set(self) -> None:
        attention = torch.tensor(
            [
                [0.05, 0.05, 0.35, 0.35, 0.05, 0.05, 0.05, 0.05],
                [0.10, 0.10, 0.05, 0.05, 0.25, 0.25, 0.10, 0.10],
            ]
        )
        selected = select_shared_mass_blocks(
            attention,
            context_length=8,
            block_tokens=2,
            budget_blocks=3,
            mandatory={0},
        )
        self.assertEqual(selected, [0, 1, 2])

    def test_shared_mass_rerank_stays_inside_proposals(self) -> None:
        attention = torch.tensor(
            [
                [0.05, 0.05, 0.35, 0.35, 0.05, 0.05, 0.05, 0.05],
                [0.10, 0.10, 0.05, 0.05, 0.25, 0.25, 0.10, 0.10],
            ]
        )
        selected = select_shared_mass_blocks(
            attention,
            context_length=8,
            block_tokens=2,
            budget_blocks=2,
            mandatory={0},
            allowed_blocks={0, 2, 3},
        )
        self.assertEqual(selected, [0, 2])

    def test_shared_output_oracle_respects_mandatory_and_budget(self) -> None:
        attention = torch.tensor([[0.1, 0.4, 0.5]])
        values = torch.tensor([[[0.0], [10.0], [2.0]]])
        selected = select_shared_output_blocks(
            attention,
            values,
            context_length=3,
            block_tokens=1,
            budget_blocks=2,
            mandatory={0},
        )
        self.assertEqual(selected, [0, 1])

    def test_query_risk_gate_average_ranks_handles_ties(self) -> None:
        ranks = average_ranks(np.asarray([3.0, 1.0, 1.0, 2.0]))
        self.assertEqual(ranks.tolist(), [3.0, 0.5, 0.5, 2.0])
        self.assertAlmostEqual(
            correlation(ranks, np.asarray([3.0, 0.5, 0.5, 2.0])), 1.0
        )
        self.assertEqual(
            parse_layer_weights("0-1:0.5,3-3:2"), {0: 0.5, 1: 0.5, 3: 2.0}
        )

    def test_pack_ragged_indices_builds_valid_mask(self) -> None:
        padded, valid = pack_ragged_indices(
            [torch.tensor([1, 3]), torch.tensor([2])]
        )
        self.assertEqual(padded.tolist(), [[1, 3], [2, 0]])
        self.assertEqual(valid.tolist(), [[True, True], [True, False]])

    def test_reference_violation_uses_requested_threshold(self) -> None:
        rows = [
            {
                "layer": 0,
                "query_position": 3,
                "kv_head": 0,
                "selected_block_ids": [0],
                "full_blocks": 2,
                "selected_blocks": 1,
                "relative_output_l2": error,
                "chosen_action": "streaming",
            }
            for error in [0.005, 0.015]
        ]
        self.assertEqual(summarize_head_rows(rows, 0.01)["violation_rate"], 0.5)
        self.assertEqual(summarize_head_rows(rows, 0.02)["violation_rate"], 0.0)

    def test_gqa_kernel_block_indices_are_unique_and_bounded(self) -> None:
        blocks = evenly_spaced_blocks(total_blocks=5, active_blocks=3)
        self.assertEqual(blocks, [0, 2, 4])
        indices = token_indices_for_blocks(
            blocks, sequence_length=18, block_tokens=4, device=torch.device("cpu")
        )
        self.assertEqual(indices.tolist(), [0, 1, 2, 3, 8, 9, 10, 11, 16, 17])

    def test_first_answer_token_includes_prompt_separator_space(self) -> None:
        class RecordingTokenizer:
            def __init__(self) -> None:
                self.text = ""

            def __call__(self, text: str, add_special_tokens: bool = False):
                self.text = text
                return {"input_ids": [17]}

        tokenizer = RecordingTokenizer()
        self.assertEqual(first_answer_token(tokenizer, ["  answer  "]), 17)
        self.assertEqual(tokenizer.text, " answer")

    def test_paired_bootstrap_ci_is_deterministic_and_ordered(self) -> None:
        values = np.asarray([-0.2, 0.0, 0.1, 0.3], dtype=np.float64)
        first = paired_bootstrap_ci(values, samples=2000, seed=7)
        second = paired_bootstrap_ci(values, samples=2000, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], values.mean())
        self.assertGreaterEqual(first[1], values.mean())

    def test_gqa_mapping(self) -> None:
        self.assertEqual(gqa_key(3, 0, 2), (3, 0))
        self.assertEqual(gqa_key(3, 1, 2), (3, 0))
        self.assertEqual(gqa_key(3, 2, 2), (3, 1))

    def test_combine_respects_quota_and_deduplicates(self) -> None:
        selected = combine_rankings([1, 2, 3, 4], [2, 5, 6], budget=4, bm25_quota=2)
        self.assertEqual(selected, [1, 2, 5, 6])

    def test_query_metric(self) -> None:
        query = {"gold_block_ids": [2, 4], "hard_negative_block_ids": [3]}
        row = query_metric(query, [2, 3], negative_penalty=0.25)
        self.assertEqual(row["any_evidence_recall"], 1.0)
        self.assertEqual(row["all_evidence_recall"], 0.0)
        self.assertEqual(row["evidence_fraction"], 0.5)
        self.assertEqual(row["hard_negative_hit_rate"], 1.0)
        self.assertEqual(row["utility"], 0.25)

    def test_minority_max_preserves_specialist_nomination(self) -> None:
        blocks = np.asarray([[[[7, 8], [9, 10]]]], dtype=np.int32)
        quality = [(0.8, 0, 0), (0.2, 0, 1)]
        action = Action(head_count=2, depth=2, mode="minority_max", bm25_quota=0)
        ranked = qk_ranking(0, blocks, quality, action)
        self.assertEqual(set(ranked), {7, 8, 9, 10})
        self.assertLess(ranked.index(9), ranked.index(10))

    def test_submodular_ranking_returns_unique_budgeted_blocks(self) -> None:
        blocks = np.asarray([[[[1, 2, 4], [1, 3, 5]]]], dtype=np.int32)
        quality = [(1.0, 0, 0), (1.0, 0, 1)]
        action = Action(head_count=2, depth=3, mode="submodular", bm25_quota=0)
        ranked = qk_ranking(0, blocks, quality, action, max_blocks=3)
        self.assertEqual(len(ranked), 3)
        self.assertEqual(len(set(ranked)), 3)
        self.assertEqual(ranked[0], 1)

    def test_score_signature_router(self) -> None:
        scores = np.zeros((4, 1, 1, 4), dtype=np.float32)
        scores[0, 0, 0] = [4.0, 1.0, 0.5, 0.1]
        scores[1, 0, 0] = [3.8, 1.0, 0.4, 0.1]
        scores[2, 0, 0] = [1.1, 1.0, 0.9, 0.8]
        scores[3, 0, 0] = [1.2, 1.0, 0.95, 0.85]
        labels = ["sharp", "sharp", "flat", "flat"]
        router = ScoreSignatureRouter()
        router.fit(scores, labels, np.asarray([True, True, True, True]))
        self.assertEqual(router.predict(scores), labels)

    def test_natural_router_risk_gate_falls_back(self) -> None:
        predicted = np.asarray([[0.0, -0.4], [0.0, -0.05]], dtype=np.float64)
        residual = np.asarray([0.0, 0.1], dtype=np.float64)
        routed = route_from_predictions(
            predicted,
            residual,
            baseline_index=0,
            threshold=0.1,
            risk_z=1.0,
        )
        self.assertEqual(routed.tolist(), [1, 0])

    def test_ridge_predict_learns_action_delta(self) -> None:
        train_x = np.asarray([[-1.0], [0.0], [1.0]], dtype=np.float64)
        train_y = np.column_stack([np.zeros(3), -train_x[:, 0]])
        predicted, residual = ridge_predict(
            train_x, train_y, np.asarray([[2.0]], dtype=np.float64), alpha=0.0
        )
        self.assertAlmostEqual(predicted[0, 0], 0.0, places=6)
        self.assertAlmostEqual(predicted[0, 1], -2.0, places=6)
        self.assertTrue(np.all(residual < 1.0e-6))

    def test_question_features_are_finite(self) -> None:
        names, values = question_features("Which 2 people arrived before Alice?")
        self.assertEqual(len(names), len(values))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertEqual(values[names.index("q_starts_which")], 1.0)

    def test_proxy_route_respects_fallback_threshold(self) -> None:
        proxy_z = np.zeros((2, 2, 4), dtype=np.float64)
        proxy_z[0, :, 0] = [0.0, -1.0]
        proxy_z[1, :, 0] = [0.0, -0.1]
        actions = route_proxy(
            proxy_z,
            action_prior=np.asarray([-1.0, 1.0]),
            weights=np.asarray([1.0, 0.0, 0.0, 0.0]),
            prior_beta=0.0,
            fallback_threshold=0.5,
        )
        self.assertEqual(actions.tolist(), [1, 0])

    def test_within_query_zscore_handles_ties(self) -> None:
        values = np.asarray([[2.0, 2.0], [1.0, 3.0]], dtype=np.float64)
        standardized = within_query_zscore(values)
        self.assertEqual(standardized[0].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(float(standardized[1].mean()), 0.0)

    def test_actionwise_proxy_regret_predict_shape(self) -> None:
        proxy = np.zeros((4, 2, 4), dtype=np.float64)
        proxy[:, 1, 0] = [-1.0, -0.5, 0.5, 1.0]
        nll = np.column_stack(
            [np.ones(4), np.asarray([0.0, 0.5, 1.5, 2.0], dtype=np.float64)]
        )
        prediction, residual = actionwise_proxy_regret_predict(
            proxy[:3], nll[:3], proxy[3:], baseline_index=0, alpha=0.1
        )
        self.assertEqual(prediction.shape, (1, 2))
        self.assertEqual(residual.shape, (2,))
        self.assertEqual(prediction[0, 0], 0.0)

    def test_specialist_compilation_deduplicates_gqa_group(self) -> None:
        block_ids = np.asarray(
            [
                [[
                    [1, 8],
                    [2, 8],
                    [3, 8],
                    [4, 8],
                ]],
                [[
                    [7, 8],
                    [2, 8],
                    [3, 8],
                    [4, 8],
                ]],
            ],
            dtype=np.int32,
        )
        queries = [{"gold_block_ids": [1]}, {"gold_block_ids": [7]}]
        specialists = compile_specialists(
            block_ids, queries, train_indices=[0, 1], gqa_group_size=2
        )
        self.assertEqual(len(specialists), 2)
        self.assertEqual(specialists[0][1:], (0, 0))

    def test_specialist_combination_fills_budget(self) -> None:
        selected = combine_specialist_rankings(
            specialist=[3, 4], bm25=[1, 2, 3, 5], target_blocks=4, bm25_quota=1
        )
        self.assertEqual(selected, [1, 3, 4, 2])

    def test_portfolio_allocation_deduplicates_and_refills(self) -> None:
        selected = allocate_portfolio(
            rankings={"deep": [1, 2, 3, 4], "specialist": [2, 5, 6, 7]},
            components=[("deep", 2), ("specialist", 2)],
            target_blocks=5,
        )
        self.assertEqual(selected, [1, 2, 5, 6, 3])

    def test_head_local_conformal_correction(self) -> None:
        keys = [(0, 1, 0, 9), (1, 1, 0, 9), (0, 1, 1, 9), (1, 1, 1, 9)]
        target_keys = [(2, 1, 0, 9), (2, 1, 1, 9)]
        corrections, values = conformal_corrections(
            np.asarray([0.1, 0.2, 0.3, 0.4]),
            keys,
            target_keys,
            quantile=0.9,
            scope="head",
        )
        self.assertEqual(corrections.tolist(), [0.2, 0.4])
        self.assertEqual(sorted(values), [0.2, 0.4])
        self.assertEqual(higher_quantile(np.asarray([1.0, 2.0, 3.0]), 0.9), 3.0)

    def test_gqa_physical_union_deduplicates_shared_blocks(self) -> None:
        keys = [(0, 0, 0, 9), (0, 0, 1, 9)]
        grouped = {
            keys[0]: {
                "qk_top_blocks": {"selected_block_ids": [1, 2], "selected_blocks": 2},
                "full": {"selected_block_ids": [1, 2, 3, 4], "selected_blocks": 4},
            },
            keys[1]: {
                "qk_top_blocks": {"selected_block_ids": [2, 3], "selected_blocks": 2},
                "full": {"selected_block_ids": [1, 2, 3, 4], "selected_blocks": 4},
            },
        }
        summary = gqa_physical_summary(
            keys, grouped, ["qk_top_blocks", "qk_top_blocks"], gqa_group_size=2
        )
        self.assertEqual(summary["mean_physical_gqa_blocks"], 3.0)
        self.assertEqual(summary["physical_gqa_saving_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
