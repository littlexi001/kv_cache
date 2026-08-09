from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_lbv2_budget_change_router_20260714 as training
from src.run_controlled_public_kv_benchmark_v1 import (
    Example,
    budget_change_router_numeric_features,
    budget_change_router_text,
)


class BudgetChangeRouterFeatureParityTest(unittest.TestCase):
    def test_training_and_runtime_features_match(self) -> None:
        raw = {
            "domain": "Single-Document QA",
            "sub_domain": "Paper",
            "question": "Which result is supported?",
            "choice_A": "Alpha",
            "choice_B": "Beta result",
            "choice_C": "Gamma",
            "choice_D": "Delta",
            "context": "context text",
        }
        query = (
            "What is the correct answer to this question: Which result is supported?\n"
            "Choices:\n(A) Alpha\n(B) Beta result\n(C) Gamma\n(D) Delta"
        )
        example = Example(
            benchmark="longbench_v2",
            task="lbv2_single_document_qa",
            sample_id="sample",
            context=raw["context"],
            query=query,
            answers=["B"],
            prefix_template="",
            suffix_template="",
            metric="longbench_v2_mc",
            max_new_tokens=128,
            length=0,
            all_classes=[],
            domain=raw["domain"],
            sub_domain=raw["sub_domain"],
        )
        base_row = {key: str(idx + 1) for idx, key in enumerate(training.NUMERIC_KEYS)}
        runtime_map = {key: float(base_row[key]) for key in training.NUMERIC_KEYS}
        with patch(
            "src.run_controlled_public_kv_benchmark_v1.learned_router_feature_map",
            return_value=runtime_map,
        ):
            runtime_numeric = budget_change_router_numeric_features(
                example,
                SimpleNamespace(),
                [],
                SimpleNamespace(),
                list(training.NUMERIC_KEYS),
            )
        self.assertEqual(training.query_text(raw), budget_change_router_text(example))
        self.assertEqual(training.numeric_features(base_row, raw), runtime_numeric)


if __name__ == "__main__":
    unittest.main()
