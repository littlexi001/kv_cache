from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from run_category_link_distortion import category_link_metrics  # noqa: E402


class CategoryLinkDistortionTest(unittest.TestCase):
    def test_removing_dominant_link_changes_output_more(self) -> None:
        attention = torch.zeros((2, 3, 3), dtype=torch.float32)
        attention[:, 2] = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.45, 0.45]])
        values = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        result = category_link_metrics(attention, values, [(2, (0,))])
        removed = result["removed_mass"]
        distortion = result["relative_output_l2"]
        assert isinstance(removed, torch.Tensor)
        assert isinstance(distortion, torch.Tensor)
        self.assertAlmostEqual(float(removed[0]), 0.8, places=6)
        self.assertAlmostEqual(float(removed[1]), 0.1, places=6)
        self.assertGreater(float(distortion[0]), float(distortion[1]))

    def test_metrics_average_multiple_probes(self) -> None:
        attention = torch.zeros((1, 3, 3), dtype=torch.float32)
        attention[0, 1, :2] = torch.tensor([0.6, 0.4])
        attention[0, 2] = torch.tensor([0.2, 0.3, 0.5])
        values = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        result = category_link_metrics(attention, values, [(1, (0,)), (2, (0,))])
        removed = result["removed_mass"]
        counts = result["query_count_by_head"]
        assert isinstance(removed, torch.Tensor)
        assert isinstance(counts, torch.Tensor)
        self.assertAlmostEqual(float(removed[0]), 0.4, places=6)
        self.assertEqual(int(counts[0]), 2)


if __name__ == "__main__":
    unittest.main()
