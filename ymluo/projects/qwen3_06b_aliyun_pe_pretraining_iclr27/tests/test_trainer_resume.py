from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


class TrainerResumeTests(unittest.TestCase):
    def test_two_segment_trainer_resume_on_torch_before_2_6(self) -> None:
        try:
            import torch
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import Whitespace
            from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        except ImportError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root = root / "model"
            model_root.mkdir()
            vocabulary = {
                "[UNK]": 0,
                "[EOS]": 1,
                "alpha": 2,
                "beta": 3,
                "gamma": 4,
                "delta": 5,
                ".": 6,
            }
            backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                unk_token="[UNK]",
                eos_token="[EOS]",
                pad_token="[EOS]",
            )
            tokenizer.save_pretrained(model_root)
            config = Qwen3Config(
                vocab_size=len(vocabulary),
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                max_position_embeddings=128,
                rope_theta=1_000_000.0,
                use_sliding_window=False,
            )
            Qwen3ForCausalLM(config).save_pretrained(model_root, safe_serialization=True)
            text_path = root / "train.txt"
            text_path.write_text(("alpha beta gamma delta.\n" * 200), encoding="utf-8")
            manifest = root / "manifest.txt"
            manifest.write_text(str(text_path) + "\n", encoding="utf-8")
            run_dir = root / "run"
            common = [
                sys.executable,
                str(PROJECT / "src" / "train_segment.py"),
                "--model-root", str(model_root),
                "--strategy", str(PROJECT / "configs" / "strategies" / "native_rope.json"),
                "--train-manifest", str(manifest),
                "--output-dir", str(run_dir),
                "--sequence-length", "32",
                "--micro-batch", "1",
                "--gradient-accumulation", "1",
                "--total-steps", "2",
                "--learning-rate", "1e-4",
                "--warmup-steps", "1",
                "--num-workers", "0",
                "--dtype", "float32",
                "--attention-implementation", "sdpa",
            ]
            subprocess.run(common + ["--stop-after-step", "1"], check=True)
            checkpoint_one = run_dir / "checkpoints" / "checkpoint-1"
            self.assertTrue((checkpoint_one / "optimizer.pt").is_file())
            subprocess.run(
                common
                + [
                    "--stop-after-step", "2",
                    "--resume-from", str(checkpoint_one),
                ],
                check=True,
            )
            checkpoint_two = run_dir / "checkpoints" / "checkpoint-2"
            self.assertTrue((checkpoint_two / "trainer_state.json").is_file())
            state = json.loads((checkpoint_two / "trainer_state.json").read_text(encoding="utf-8"))
            self.assertEqual(int(state["global_step"]), 2)

            scratch_run = root / "scratch_run"
            scratch_command = list(common)
            output_index = scratch_command.index("--output-dir") + 1
            scratch_command[output_index] = str(scratch_run)
            subprocess.run(
                scratch_command
                + [
                    "--stop-after-step", "1",
                    "--initialization", "from_scratch",
                    "--global-batch-size", "1",
                    "--target-tokens", "32",
                    "--tensorboard", "1",
                ],
                check=True,
            )
            scratch_metadata = json.loads(
                (scratch_run / "milestone_000001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(scratch_metadata["initialization"], "from_scratch")
            self.assertEqual(scratch_metadata["global_batch_size_sequences"], 1)
            event_files = list((scratch_run / "tensorboard").glob("events.out.tfevents.*"))
            self.assertTrue(event_files)
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            accumulator = EventAccumulator(str(scratch_run / "tensorboard"))
            accumulator.Reload()
            self.assertIn("progress/tokens_seen", accumulator.Tags()["scalars"])
            self.assertEqual(
                int(accumulator.Scalars("progress/tokens_seen")[-1].value), 32
            )


if __name__ == "__main__":
    unittest.main()
