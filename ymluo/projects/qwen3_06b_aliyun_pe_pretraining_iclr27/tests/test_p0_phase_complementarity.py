from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from optimize_phase_profile import coverage_and_local_penalty  # noqa: E402
from p0_four_machine_plan import assignment, load_plan  # noqa: E402
from pe_strategies import load_strategy, pair_scales  # noqa: E402


class P0PhaseComplementarityTests(unittest.TestCase):
    def test_four_task_mapping(self) -> None:
        payload = load_plan()
        self.assertEqual(assignment(payload, 0)["strategy"], "optimized_phase_complementary")
        self.assertEqual(
            assignment(payload, 1)["strategy"],
            "optimized_phase_complementary_local",
        )
        self.assertEqual(assignment(payload, 2)["strategy"], "native_rope")
        self.assertEqual(assignment(payload, 3)["strategy"], "rnope_every4")

    def test_layer_pair_matrix_returns_exact_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.json"
            payload = {
                "name": "test_matrix",
                "kind": "layer_pair_matrix",
                "layer_pair_scales": [[1.0, 0.5], [0.25, 0.75]],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            strategy = load_strategy(path)
            actual = pair_scales(strategy, 1, 2, 2)
            torch.testing.assert_close(
                actual, torch.tensor([0.25, 0.75], dtype=torch.float64)
            )

    def test_matrix_shape_and_range_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "bad_matrix",
                        "kind": "layer_pair_matrix",
                        "layer_pair_scales": [[1.0, 1.1]],
                    }
                ),
                encoding="utf-8",
            )
            strategy = load_strategy(path)
            with self.assertRaises(ValueError):
                pair_scales(strategy, 0, 1, 2)

    def test_phase_diversity_has_better_unknown_phase_coverage_than_collapse(self) -> None:
        inv_freq = torch.tensor([1.0], dtype=torch.float64)
        remote = torch.tensor([32.0, 64.0, 128.0], dtype=torch.float64)
        probes = torch.arange(32, dtype=torch.float64) * (2.0 * math.pi / 32.0)
        local = torch.tensor([1.0, 2.0], dtype=torch.float64)
        collapsed = torch.ones((4, 1), dtype=torch.float64)
        diverse = torch.tensor([[1.0], [0.75], [0.5], [0.25]], dtype=torch.float64)
        collapsed_coverage, _ = coverage_and_local_penalty(
            collapsed, inv_freq, remote, probes, local, 0.08
        )
        diverse_coverage, _ = coverage_and_local_penalty(
            diverse, inv_freq, remote, probes, local, 0.08
        )
        self.assertGreater(float(diverse_coverage), float(collapsed_coverage))

    def test_optimizer_generates_a_valid_qwen_strategy(self) -> None:
        from transformers import Qwen3Config

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root = root / "model"
            output = root / "optimized.json"
            Qwen3Config(
                vocab_size=128,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                rope_theta=1_000_000.0,
            ).save_pretrained(model_root)
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT / "src" / "optimize_phase_profile.py"),
                    "--model-root",
                    str(model_root),
                    "--output",
                    str(output),
                    "--name",
                    "optimized_test",
                    "--frequency-pairs",
                    "4",
                    "--remote-min",
                    "64",
                    "--remote-max",
                    "1024",
                    "--remote-points",
                    "5",
                    "--content-phase-probes",
                    "8",
                    "--local-max",
                    "32",
                    "--local-points",
                    "5",
                    "--steps",
                    "100",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            strategy = load_strategy(output)
            self.assertEqual(strategy.kind, "layer_pair_matrix")
            self.assertEqual(len(strategy.values["layer_pair_scales"]), 4)
            self.assertEqual(len(strategy.values["layer_pair_scales"][0]), 16)
            self.assertTrue(output.with_suffix(".optimization.json").is_file())


if __name__ == "__main__":
    unittest.main()
