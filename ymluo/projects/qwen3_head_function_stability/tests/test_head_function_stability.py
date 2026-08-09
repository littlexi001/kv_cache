from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from run_head_function_stability import (  # noqa: E402
    build_controlled_samples,
    cosine_on_common,
    rankdata,
    score_attention_category,
    spans_to_token_indices,
)


class HeadFunctionStabilityTest(unittest.TestCase):
    def test_controlled_suite_has_expected_coverage(self) -> None:
        samples = build_controlled_samples()
        self.assertEqual(len(samples), 32)
        self.assertEqual(len({sample.sample_id for sample in samples}), len(samples))
        categories = {probe.category for sample in samples for probe in sample.probes}
        self.assertEqual(
            categories,
            {"semantic_evidence", "syntactic_dependency", "structural_anchor"},
        )

    def test_span_to_tokens_uses_overlap(self) -> None:
        offsets = [(0, 3), (3, 4), (5, 9)]
        self.assertEqual(spans_to_token_indices(offsets, (2, 6)), [0, 1, 2])

    def test_attention_enrichment(self) -> None:
        attention = torch.zeros((2, 4, 4), dtype=torch.float32)
        attention[0, 3] = torch.tensor([0.8, 0.1, 0.05, 0.05])
        attention[1, 3] = torch.tensor([0.1, 0.2, 0.3, 0.4])
        mass, score, availability, count = score_attention_category(
            attention, [(3, (0,))]
        )
        np.testing.assert_allclose(mass, [0.8, 0.1], rtol=1e-6)
        self.assertAlmostEqual(availability, 0.25)
        self.assertEqual(count, 1)
        self.assertGreater(score[0], 1.0)
        self.assertLess(score[1], 0.0)

    def test_rankdata_ties(self) -> None:
        np.testing.assert_allclose(rankdata(np.asarray([3.0, 1.0, 1.0])), [3.0, 1.5, 1.5])

    def test_cosine_common_categories(self) -> None:
        left = {"a": 1.0, "b": 2.0, "c": 3.0, "left_only": 9.0}
        right = {"a": 2.0, "b": 4.0, "c": 6.0, "right_only": 9.0}
        self.assertAlmostEqual(cosine_on_common(left, right), 1.0)


if __name__ == "__main__":
    unittest.main()
