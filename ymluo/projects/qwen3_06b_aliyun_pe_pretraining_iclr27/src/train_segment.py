from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from data_pipeline import PackedTextDataset
from io_utils import append_jsonl, sha256_file, utc_timestamp, write_json
from model_utils import load_model, load_tokenizer
from pe_strategies import load_strategy, save_strategy_profile


def collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([row["input_ids"] for row in rows]),
        "labels": torch.stack([row["labels"] for row in rows]),
    }


def validate_resume_checkpoint(checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    """Allow torch.load resume only for a complete checkpoint under this run directory."""
    resolved = checkpoint.resolve(strict=True)
    checkpoint_root = (output_dir / "checkpoints").resolve(strict=True)
    if checkpoint_root not in resolved.parents:
        raise RuntimeError(
            f"refusing optimizer resume outside this run's checkpoint directory: {resolved}"
        )
    required = ["model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"]
    missing = [name for name in required if not (resolved / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete resume checkpoint {resolved}: missing {missing}")
    for path in resolved.iterdir():
        if path.is_symlink():
            raise RuntimeError(f"refusing symlink in trusted local checkpoint: {path}")
    integrity_path = resolved / "checkpoint_integrity.json"
    verified = False
    if integrity_path.is_file():
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        for name, expected in integrity.get("sha256", {}).items():
            actual = sha256_file(resolved / name)
            if actual != expected:
                raise RuntimeError(f"checkpoint integrity mismatch for {resolved / name}")
        verified = True
    return {
        "path": str(resolved),
        "local_path_validated": True,
        "hash_manifest_verified": verified,
        "security_boundary": "Only locally generated checkpoints under output_dir/checkpoints are accepted.",
    }


def allow_trusted_numpy_rng_state() -> list[str]:
    """Allow only NumPy RNG container types used by Trainer's local rng_state.pth."""
    import _codecs
    import numpy as np

    if not hasattr(torch.serialization, "add_safe_globals"):
        return []
    safe_types = [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
        _codecs.encode,
    ]
    torch.serialization.add_safe_globals(safe_types)
    return [f"{value.__module__}.{value.__name__}" for value in safe_types]


def main() -> None:
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--stop-after-step", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--data-seed", type=int, default=1701)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument(
        "--initialization", choices=["checkpoint", "from_scratch"], default="checkpoint"
    )
    parser.add_argument("--global-batch-size", type=int, default=0)
    parser.add_argument("--target-tokens", type=int, default=0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--tensorboard", type=int, choices=[0, 1], default=0)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    actual_global_batch = args.micro_batch * args.gradient_accumulation * world_size
    if args.global_batch_size and actual_global_batch != args.global_batch_size:
        raise ValueError(
            "global batch mismatch: "
            f"micro_batch({args.micro_batch}) * gradient_accumulation({args.gradient_accumulation}) "
            f"* world_size({world_size}) = {actual_global_batch}, "
            f"expected {args.global_batch_size}"
        )
    if not 0 < args.stop_after_step <= args.total_steps:
        raise ValueError("stop-after-step must be in (0,total-steps]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resume_evidence = None
    if args.resume_from:
        resume_evidence = validate_resume_checkpoint(args.resume_from, args.output_dir)
        resume_evidence["rng_safe_globals"] = allow_trusted_numpy_rng_state()
        if rank == 0:
            write_json(
                args.output_dir / f"resume_validation_to_{args.stop_after_step:06d}.json",
                resume_evidence,
            )
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    strategy = load_strategy(args.strategy)
    tokenizer = load_tokenizer(args.model_root)
    model, chosen_attention = load_model(
        args.model_root,
        strategy,
        args.dtype,
        args.attention_implementation,
        for_training=True,
        initialization=args.initialization,
    )
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    if rank == 0:
        tokenizer.save_pretrained(args.output_dir / "tokenizer")
        save_strategy_profile(model, strategy, args.output_dir / "strategy_profile.json")

    dataset = PackedTextDataset(
        manifest_path=args.train_manifest,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        rank=rank,
        world_size=world_size,
        infinite=True,
    )
    metrics_path = args.output_dir / "training_metrics.jsonl"
    tokens_per_step = args.sequence_length * actual_global_batch

    class StopAtMilestone(TrainerCallback):
        def on_step_end(self, _args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
            if state.global_step >= args.stop_after_step:
                control.should_training_stop = True
                control.should_save = True
            return control

    class JsonlLogger(TrainerCallback):
        def on_log(self, _args: Any, state: Any, _control: Any, logs: Any = None, **_kwargs: Any) -> None:
            if rank == 0 and logs:
                append_jsonl(
                    metrics_path,
                    {
                        "timestamp": utc_timestamp(),
                        "global_step": int(state.global_step),
                        "tokens_seen_nominal": int(state.global_step) * tokens_per_step,
                        "progress_percent": (
                            100.0 * int(state.global_step) * tokens_per_step / args.target_tokens
                            if args.target_tokens
                            else None
                        ),
                        **{key: float(value) if isinstance(value, (int, float)) else value for key, value in logs.items()},
                    },
                )

    class ProgressTensorBoard(TrainerCallback):
        def __init__(self) -> None:
            self.writer: Any = None

        def on_log(self, _args: Any, state: Any, _control: Any, **_kwargs: Any) -> None:
            if rank != 0 or not args.tensorboard:
                return
            if self.writer is None:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard"))
            seen = int(state.global_step) * tokens_per_step
            self.writer.add_scalar("progress/tokens_seen", seen, int(state.global_step))
            if args.target_tokens:
                self.writer.add_scalar(
                    "progress/percent",
                    100.0 * seen / args.target_tokens,
                    int(state.global_step),
                )
            self.writer.flush()

        def on_train_end(self, _args: Any, _state: Any, _control: Any, **_kwargs: Any) -> None:
            if self.writer is not None:
                self.writer.close()
                self.writer = None

    bf16 = args.dtype.lower() in {"bfloat16", "bf16"}
    fp16 = args.dtype.lower() in {"float16", "fp16"}
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        max_steps=args.total_steps,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.stop_after_step,
        save_total_limit=8,
        save_safetensors=True,
        bf16=bf16,
        fp16=fp16,
        tf32=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["tensorboard"] if args.tensorboard else [],
        logging_dir=str(args.output_dir / "tensorboard"),
        seed=args.seed,
        data_seed=args.data_seed,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate,
        callbacks=[StopAtMilestone(), JsonlLogger(), ProgressTensorBoard()],
    )
    started = time.time()
    result = trainer.train(
        resume_from_checkpoint=str(args.resume_from) if args.resume_from else None
    )
    reached = int(trainer.state.global_step)
    if reached != args.stop_after_step:
        raise RuntimeError(
            f"training stopped at step {reached}, expected exactly {args.stop_after_step}"
        )
    checkpoint = args.output_dir / "checkpoints" / f"checkpoint-{reached}"
    if not checkpoint.is_dir():
        raise RuntimeError(f"milestone checkpoint was not saved: {checkpoint}")
    if rank == 0:
        checkpoint_files = [
            name
            for name in [
                "model.safetensors",
                "optimizer.pt",
                "scheduler.pt",
                "trainer_state.json",
                "rng_state.pth",
            ]
            if (checkpoint / name).is_file()
        ]
        write_json(
            checkpoint / "checkpoint_integrity.json",
            {
                "step": reached,
                "sha256": {
                    name: sha256_file(checkpoint / name) for name in checkpoint_files
                },
            },
        )
        payload = {
            "strategy": strategy.name,
            "step": reached,
            "checkpoint": str(checkpoint),
            "world_size": world_size,
            "tokens_per_step": tokens_per_step,
            "tokens_seen_nominal": reached * tokens_per_step,
            "target_tokens": args.target_tokens or None,
            "global_batch_size_sequences": actual_global_batch,
            "initialization": args.initialization,
            "segment_wall_seconds": time.time() - started,
            "attention_implementation": chosen_attention,
            "train_metrics": result.metrics,
            "resume_validation": resume_evidence,
            "peak_allocated_gib_rank0": (
                torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
            ),
        }
        write_json(args.output_dir / f"milestone_{reached:06d}.json", payload)
        (args.output_dir / f"milestone_{reached:06d}.done").write_text(
            "ok\n", encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
