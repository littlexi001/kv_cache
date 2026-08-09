from __future__ import annotations

import math
import argparse
import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from sixteen_machine_plan import assignment, load_plan  # noqa: E402
from validate_pretrain_protocol import validate  # noqa: E402


class SixteenMachinePlanTests(unittest.TestCase):
    def test_plan_has_sixteen_unique_documented_conditions(self) -> None:
        payload = load_plan()
        rows = [assignment(payload, task_id) for task_id in range(16)]
        self.assertEqual(rows[0]["strategy"], "native_rope")
        self.assertEqual(len({row["strategy"] for row in rows}), 16)
        self.assertEqual(rows[14]["strategy"], "period_aware_smooth")
        self.assertEqual(rows[15]["strategy"], "phase_diverse_deep")

    def test_100b_schedule_math(self) -> None:
        tokens_per_step = 8192 * 256
        steps = math.ceil(100_000_000_000 / tokens_per_step)
        self.assertEqual(tokens_per_step, 2_097_152)
        self.assertEqual(steps, 47_684)
        self.assertEqual(steps * tokens_per_step, 100_000_595_968)

    def test_invalid_task_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assignment(load_plan(), 16)

    def test_exact_pretraining_contract(self) -> None:
        payload = validate(
            argparse.Namespace(
                gpu_list="0,1,2,3,4,5,6,7",
                initialization="from_scratch",
                sequence_length=8192,
                micro_batch=1,
                gradient_accumulation=32,
                global_batch_size=256,
                target_tokens=100_000_000_000,
                learning_rate=1e-4,
            )
        )
        self.assertEqual(payload["optimizer_steps"], 47_684)

    def test_wrong_gpu_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate(
                argparse.Namespace(
                    gpu_list="0,1,2,3",
                    initialization="from_scratch",
                    sequence_length=8192,
                    micro_batch=1,
                    gradient_accumulation=32,
                    global_batch_size=256,
                    target_tokens=100_000_000_000,
                    learning_rate=1e-4,
                )
            )

    def test_explicit_smoke_override_changes_only_target_tokens(self) -> None:
        payload = validate(
            argparse.Namespace(
                gpu_list="0,1,2,3,4,5,6,7",
                initialization="from_scratch",
                sequence_length=8192,
                micro_batch=1,
                gradient_accumulation=32,
                global_batch_size=256,
                target_tokens=10_000_000,
                learning_rate=1e-4,
                allow_target_token_override=True,
            )
        )
        self.assertEqual(payload["target_tokens"], 10_000_000)
        self.assertEqual(payload["optimizer_steps"], 5)


if __name__ == "__main__":
    unittest.main()
