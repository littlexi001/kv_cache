from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from data_pipeline import build_manifests  # noqa: E402
from evaluate_checkpoint import controlled_case, normalize_answer, token_f1  # noqa: E402
from pe_strategies import (  # noqa: E402
    build_position_embeddings,
    effective_positions,
    load_strategy,
    pair_scales,
    patch_model,
)
from orchestrate import evaluation_complete, event, log_evaluation_to_tensorboard  # noqa: E402
from model_utils import load_model  # noqa: E402
from train_segment import validate_resume_checkpoint  # noqa: E402


class TinyTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [1 + (ord(character) % 251) for character in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(map(str, ids))


class DummyAttention:
    def __init__(self) -> None:
        self.forward = self._forward

    def _forward(self, hidden_states, position_embeddings, attention_mask, **kwargs):
        del hidden_states, attention_mask, kwargs
        return position_embeddings


class DummyLayer:
    def __init__(self) -> None:
        self.self_attn = DummyAttention()


class DummyRotary:
    def __init__(self) -> None:
        self.inv_freq = torch.tensor([1.0, 0.1])
        self.attention_scaling = 1.0


class DummyInner:
    def __init__(self) -> None:
        self.layers = [DummyLayer(), DummyLayer()]
        self.rotary_emb = DummyRotary()


class DummyModel:
    def __init__(self) -> None:
        self.model = DummyInner()


class ContractTests(unittest.TestCase):
    def strategy(self, name: str):
        return load_strategy(PROJECT / "configs" / "strategies" / f"{name}.json")

    def test_native_and_uniform_scales(self) -> None:
        native = pair_scales(self.strategy("native_rope"), 0, 28, 64)
        slow = pair_scales(self.strategy("uniform_slow_rope"), 0, 28, 64)
        self.assertTrue(torch.allclose(native, torch.ones(64, dtype=torch.float64)))
        self.assertTrue(torch.allclose(slow, torch.full((64,), 0.5, dtype=torch.float64)))

    def test_deep_drop_has_declared_boundary(self) -> None:
        strategy = self.strategy("deep_highfreq_drop")
        shallow = pair_scales(strategy, 17, 28, 64)
        deep = pair_scales(strategy, 19, 28, 64)
        self.assertTrue(torch.all(shallow == 1.0))
        self.assertTrue(torch.all(deep[:12] == 0.0))
        self.assertTrue(torch.all(deep[12:] == 1.0))

    def test_smooth_remote_warp_is_exactly_local_identity(self) -> None:
        strategy = self.strategy("smooth_remote_warp")
        positions = torch.tensor([0.0, 1024.0, 2048.0, 8192.0])
        effective = effective_positions(positions, strategy, 27, 28, 64)
        expected = positions[:3, None].double().expand(-1, 64)
        self.assertTrue(torch.allclose(effective[:3], expected))
        self.assertTrue(torch.all(effective[3] <= positions[3]))
        self.assertTrue(torch.any(effective[3] < positions[3]))

    def test_patch_preserves_reference_shape(self) -> None:
        model = DummyModel()
        strategy = self.strategy("uniform_slow_rope")
        self.assertEqual(patch_model(model, strategy), 2)
        hidden = torch.zeros(1, 3, 4)
        reference = (torch.zeros(1, 3, 4), torch.zeros(1, 3, 4))
        output = model.model.layers[0].self_attn.forward(
            hidden_states=hidden,
            position_embeddings=reference,
            attention_mask=None,
            cache_position=torch.arange(3),
        )
        self.assertEqual(tuple(output[0].shape), (1, 3, 4))
        self.assertTrue(torch.isfinite(output[0]).all())

    def test_real_tiny_qwen3_forward_and_cache_for_every_strategy(self) -> None:
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError:
            self.skipTest("installed transformers has no Qwen3")
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
            max_position_embeddings=128,
            rope_theta=1_000_000.0,
        )
        input_ids = torch.randint(0, config.vocab_size, (1, 32))
        for strategy_path in sorted((PROJECT / "configs" / "strategies").glob("*.json")):
            name = strategy_path.stem
            model = Qwen3ForCausalLM(config)
            patch_model(model, self.strategy(name))
            output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            self.assertTrue(torch.isfinite(output.loss), name)
            with torch.no_grad():
                generated = model.generate(input_ids=input_ids, max_new_tokens=2, do_sample=False)
            self.assertEqual(tuple(generated.shape), (1, 34), name)

    def test_from_scratch_initialization_is_seed_reproducible(self) -> None:
        try:
            from transformers import Qwen3Config
        except ImportError:
            self.skipTest("installed transformers has no Qwen3")
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=128,
            rope_theta=1_000_000.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config.save_pretrained(root)
            torch.manual_seed(123)
            first, _ = load_model(
                root, self.strategy("native_rope"), "float32", "sdpa", True,
                initialization="from_scratch",
            )
            first_value = next(first.parameters()).detach().clone()
            torch.manual_seed(123)
            second, _ = load_model(
                root, self.strategy("native_rope"), "float32", "sdpa", True,
                initialization="from_scratch",
            )
            second_value = next(second.parameters()).detach().clone()
            self.assertTrue(torch.equal(first_value, second_value))

    def test_manifest_is_deterministic_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dclm"
            for shard in range(3):
                directory = root / f"global-shard_{shard}" / "local-shard_0"
                directory.mkdir(parents=True)
                for index in range(10):
                    (directory / f"{index}.txt").write_text(f"sample {shard} {index}\n", encoding="utf-8")
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            one = build_manifests(root, first, 8, 3, 1701)
            two = build_manifests(root, second, 8, 3, 1701)
            self.assertEqual(one["manifest_sha256"], two["manifest_sha256"])
            train = set((first / "train_manifest.txt").read_text().splitlines())
            validation = set((first / "validation_manifest.txt").read_text().splitlines())
            self.assertFalse(train & validation)

    def test_controlled_prompt_has_exact_length(self) -> None:
        tokenizer = TinyTokenizer()
        for task in ["niah_single", "niah_multi", "variable_tracking"]:
            prompt, answers, _ = controlled_case(tokenizer, task, 4096, 7)
            self.assertEqual(len(prompt), 4096)
            self.assertTrue(answers[0])

    def test_qa_metrics(self) -> None:
        self.assertEqual(normalize_answer("The Cobalt."), "cobalt")
        self.assertEqual(token_f1("cobalt", "the cobalt"), 1.0)

    def test_event_accepts_a_path_field_without_argument_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            event_path = Path(temporary) / "events.jsonl"
            event(event_path, "prepare_manifest", "started", path="/mnt/workspace/test")
            row = json.loads(event_path.read_text(encoding="utf-8"))
            self.assertEqual(row["path"], "/mnt/workspace/test")
            self.assertEqual(row["stage"], "prepare_manifest")

    def test_unavailable_longbench_is_retried_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "status.json").write_text(json.dumps({"complete": True}), encoding="utf-8")
            (output / "summary.json").write_text(
                json.dumps({"longbench_status": "unavailable"}), encoding="utf-8"
            )
            self.assertTrue(evaluation_complete(output, require_longbench=False))
            self.assertFalse(evaluation_complete(output, require_longbench=True))
            (output / "summary.json").write_text(
                json.dumps({"longbench_status": "complete"}), encoding="utf-8"
            )
            self.assertTrue(evaluation_complete(output, require_longbench=True))

    def test_checkpoint_metrics_are_written_to_tensorboard(self) -> None:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            self.skipTest("tensorboard is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            evaluation = run_dir / "evaluations" / "step_000010"
            evaluation.mkdir(parents=True)
            (evaluation / "summary.json").write_text(
                json.dumps(
                    {
                        "validation_ppl": {"ppl": 3.5},
                        "controlled": [{"qa_f1_percent": 50.0, "exact_match_percent": 40.0, "gold_answer_mean_nll": 1.2}],
                        "longbench": [{"qa_f1_percent": 30.0, "exact_match_percent": 20.0, "gold_answer_mean_nll": 2.3}],
                    }
                ),
                encoding="utf-8",
            )
            log_evaluation_to_tensorboard(run_dir, evaluation, 10)
            accumulator = EventAccumulator(str(run_dir / "tensorboard"))
            accumulator.Reload()
            tags = accumulator.Tags()["scalars"]
            self.assertIn("eval/dclm_ppl", tags)
            self.assertIn("eval/controlled_qa_f1_percent", tags)
            self.assertEqual(accumulator.Scalars("eval/dclm_ppl")[-1].value, 3.5)

    def test_resume_is_restricted_to_complete_local_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            checkpoint = run_dir / "checkpoints" / "checkpoint-50"
            checkpoint.mkdir(parents=True)
            for name in ["model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"]:
                (checkpoint / name).write_bytes(b"local-test")
            evidence = validate_resume_checkpoint(checkpoint, run_dir)
            self.assertTrue(evidence["local_path_validated"])
            outside = Path(temporary) / "outside"
            outside.mkdir()
            for name in ["model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"]:
                (outside / name).write_bytes(b"untrusted-test")
            with self.assertRaises(RuntimeError):
                validate_resume_checkpoint(outside, run_dir)

    def test_result_bundle_and_matched_native_merge(self) -> None:
        def evaluation_payload(label: str, step: int, ppl: float, score: float) -> dict:
            return {
                "label": label,
                "step": step,
                "critical_complete": True,
                "failure_count": 0,
                "elapsed_seconds": 1.0,
                "validation_ppl": {"status": "complete", "ppl": ppl, "mean_nll": 1.0},
                "controlled": [
                    {
                        "qa_f1_percent": score,
                        "exact_match_percent": score,
                        "contains_answer_percent": score,
                        "gold_answer_mean_nll": 1.0,
                    }
                ],
                "longbench": [
                    {
                        "qa_f1_percent": score,
                        "exact_match_percent": score,
                        "contains_answer_percent": score,
                        "gold_answer_mean_nll": 1.0,
                    }
                ],
                "longbench_status": "complete",
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_dir = root / "bundles"
            bundle_dir.mkdir()
            for strategy, ppl, score in [
                ("native_rope", 3.0, 50.0),
                ("smooth_layer_frequency", 3.03, 55.0),
            ]:
                run_dir = root / strategy
                (run_dir / "evaluations" / "base_step0").mkdir(parents=True)
                (run_dir / "evaluations" / "step_000010").mkdir(parents=True)
                (run_dir / "evaluations" / "base_step0" / "summary.json").write_text(
                    json.dumps(evaluation_payload("base", 0, 3.2, 40.0)), encoding="utf-8"
                )
                (run_dir / "evaluations" / "step_000010" / "summary.json").write_text(
                    json.dumps(evaluation_payload("trained", 10, ppl, score)), encoding="utf-8"
                )
                (run_dir / "milestone_000010.json").write_text(
                    json.dumps(
                        {
                            "step": 10,
                            "world_size": 8,
                            "tokens_per_step": 524288,
                            "tokens_seen_nominal": 5242880,
                            "segment_wall_seconds": 5.0,
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "manifest_metadata.json").write_text(
                    json.dumps({"manifest_sha256": "matched-test-manifest"}),
                    encoding="utf-8",
                )
                subprocess.run(
                    [
                        sys.executable,
                        str(PROJECT / "src" / "collect_results.py"),
                        "--run-dir",
                        str(run_dir),
                        "--strategy-name",
                        strategy,
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        sys.executable,
                        str(PROJECT / "src" / "package_results.py"),
                        "--run-dir",
                        str(run_dir),
                        "--strategy-name",
                        strategy,
                    ],
                    check=True,
                )
                for archive in (run_dir / "bundles").glob("*.tar.gz"):
                    shutil.copy2(archive, bundle_dir / archive.name)
            combined = root / "combined"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT / "src" / "merge_result_bundles.py"),
                    "--bundle-dir",
                    str(bundle_dir),
                    "--output-dir",
                    str(combined),
                ],
                check=True,
            )
            payload = json.loads((combined / "combined_summary.json").read_text(encoding="utf-8"))
            smooth = next(row for row in payload["rows"] if row["strategy"] == "smooth_layer_frequency")
            self.assertEqual(smooth["native_control_status"], "matched")
            self.assertAlmostEqual(smooth["controlled_qa_f1_percent_change_vs_native_pp"], 5.0)
            self.assertTrue(smooth["ppl_guardrail_pass"])
            self.assertEqual(payload["manifest_sha256"], "matched-test-manifest")


if __name__ == "__main__":
    unittest.main()
