from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from four_machine_plan import assignment, load_plan  # noqa: E402


class FourMachinePlanTests(unittest.TestCase):
    def test_plan_has_four_unique_conditions_and_native_control(self) -> None:
        payload = load_plan()
        rows = [assignment(payload, machine_id) for machine_id in range(4)]
        self.assertEqual(
            [row["strategy"] for row in rows],
            [
                "native_rope",
                "deep_highfreq_drop",
                "uniform_slow_rope",
                "smooth_layer_frequency",
            ],
        )
        self.assertEqual(len({row["strategy"] for row in rows}), 4)

    def test_unknown_machine_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assignment(load_plan(), 4)


if __name__ == "__main__":
    unittest.main()
