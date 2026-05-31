from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from qwen15_moe_svd_projected_patch import SvdProjectedRoutingConfig, SvdProjectedRoutingPatch


MOE_0_6B_CONFIG_OVERRIDES: dict[str, Any] = {
    "hidden_size": 768,
    "intermediate_size": 2048,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_key_value_heads": 4,
    "num_experts": 49,
    "num_experts_per_tok": 7,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
}


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="/mnt/workspace/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="qwen15-moe-svd-projected-routing")
    parser.add_argument("--init_from_scratch", type=str2bool, default=True)
    parser.add_argument("--resume_from_checkpoint", default="")
    parser.add_argument("--model_size_preset", choices=["none", "moe_0_6b"], default="moe_0_6b")
    parser.add_argument("--model_config_overrides", default="")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--data_mode", choices=["synthetic", "text"], default="synthetic")
    parser.add_argument("--dataset_path", default="/mnt/workspace/dclm")
    parser.add_argument("--data_files_glob", default="**/*.txt")
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--min_text_chars", type=int, default=20)
    parser.add_argument("--synthetic_vocab_size", type=int, default=4096)
    parser.add_argument("--synthetic_topic_count", type=int, default=16)
    parser.add_argument("--synthetic_entities_per_topic", type=int, default=32)
    parser.add_argument("--synthetic_noise_rate", type=float, default=0.15)

    parser.add_argument("--projection_source", choices=["q", "k", "v", "o"], default="q")
    parser.add_argument("--svd_refresh_interval", type=int, default=100)
    parser.add_argument("--group1_experts", type=int, default=16)
    parser.add_argument("--group2_experts", type=int, default=24)
    parser.add_argument("--group3_experts", type=int, default=8)
    parser.add_argument("--group1_topk", type=int, default=2)
    parser.add_argument("--group2_topk", type=int, default=3)
    parser.add_argument("--group3_topk", type=int, default=1)
    parser.add_argument("--normalize_topk_prob", type=str2bool, default=True)
    parser.add_argument("--load_balance_loss_weight", type=float, default=0.01)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--ddp_find_unused_parameters", type=str2bool, default=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--deepspeed_config", default="")
    parser.add_argument("--report_to", default="tensorboard")
    return parser.parse_args()


def distributed_rank_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


class SyntheticStructuredBlockDataset(IterableDataset):
    def __init__(
        self,
        seq_length: int,
        vocab_size: int,
        topic_count: int,
        entities_per_topic: int,
        noise_rate: float,
        seed: int,
    ) -> None:
        self.seq_length = int(seq_length)
        self.vocab_size = int(vocab_size)
        self.topic_count = int(topic_count)
        self.entities_per_topic = int(entities_per_topic)
        self.noise_rate = float(noise_rate)
        self.seed = int(seed)
        min_vocab = 8 + self.topic_count * self.entities_per_topic
        if self.vocab_size <= min_vocab:
            raise ValueError(f"--synthetic_vocab_size must be > {min_vocab}.")

    def _rng(self) -> random.Random:
        rank, world_size = distributed_rank_info()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        return random.Random(self.seed + 1000003 * (rank * num_workers + worker_id) + 9176 * world_size)

    def __iter__(self):
        rng = self._rng()
        topic_base = 8
        noise_base = topic_base + self.topic_count * self.entities_per_topic
        while True:
            topic = rng.randrange(self.topic_count)
            sequence = []
            for position in range(self.seq_length + 1):
                if position % 8 == 0:
                    topic = rng.randrange(self.topic_count)
                    token = 1 + topic % 7
                elif rng.random() < self.noise_rate:
                    token = rng.randrange(noise_base, self.vocab_size)
                else:
                    entity = (position + rng.randrange(self.entities_per_topic)) % self.entities_per_topic
                    token = topic_base + topic * self.entities_per_topic + entity
                sequence.append(token)
            input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
            labels = torch.tensor(sequence[1:], dtype=torch.long)
            yield {"input_ids": input_ids, "labels": labels}


class RandomTextBlockDataset(IterableDataset):
    def __init__(
        self,
        dataset_path: str,
        data_files_glob: str,
        tokenizer,
        seq_length: int,
        seed: int,
        min_text_chars: int,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.data_files_glob = data_files_glob
        self.tokenizer = tokenizer
        self.seq_length = int(seq_length)
        self.seed = int(seed)
        self.min_text_chars = int(min_text_chars)
        self.files = self._resolve_files()
        self.file_sizes = [max(path.stat().st_size, 1) for path in self.files]

    def _resolve_files(self) -> list[Path]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        files = [path for path in self.dataset_path.glob(self.data_files_glob) if path.is_file()]
        files = [path for path in files if path.suffix.lower() == ".txt"]
        if not files:
            raise FileNotFoundError(f"No .txt files matched {self.data_files_glob} under {self.dataset_path}")
        files.sort()
        return files

    def _rng(self) -> random.Random:
        rank, world_size = distributed_rank_info()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        return random.Random(self.seed + 1000003 * (rank * num_workers + worker_id) + 9176 * world_size)

    def _sample_line(self, rng: random.Random) -> str:
        path = self.files[rng.randrange(len(self.files))]
        offset = rng.randrange(max(path.stat().st_size, 1))
        with path.open("rb") as handle:
            handle.seek(offset)
            if offset > 0:
                handle.readline()
            line = handle.readline()
            if not line:
                handle.seek(0)
                line = handle.readline()
        return line.decode("utf-8", errors="ignore").strip()

    def __iter__(self):
        rng = self._rng()
        buffer: list[int] = []
        while True:
            text = self._sample_line(rng)
            if len(text) < self.min_text_chars:
                continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            buffer.extend(ids)
            while len(buffer) >= self.seq_length + 1:
                chunk = buffer[: self.seq_length + 1]
                del buffer[: self.seq_length + 1]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }


def causal_lm_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in features]) for key in ("input_ids", "labels")}


def apply_model_config_overrides(config: Any, args: argparse.Namespace) -> None:
    overrides: dict[str, Any] = {}
    if args.model_size_preset == "moe_0_6b":
        overrides.update(MOE_0_6B_CONFIG_OVERRIDES)
    if args.model_config_overrides.strip():
        user_overrides = json.loads(args.model_config_overrides)
        if not isinstance(user_overrides, dict):
            raise ValueError("--model_config_overrides must be a JSON object.")
        overrides.update(user_overrides)
    for name, value in overrides.items():
        setattr(config, name, value)


def model_config_summary(config: Any) -> dict[str, Any]:
    names = [
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "num_experts",
        "num_experts_per_tok",
        "decoder_sparse_step",
        "tie_word_embeddings",
    ]
    return {name: getattr(config, name) for name in names if hasattr(config, name)}


def load_model(args: argparse.Namespace, dtype: torch.dtype):
    if args.init_from_scratch:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        apply_model_config_overrides(config, args)
        config._attn_implementation = args.attn_implementation
        config.use_cache = False
        return AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(dtype=dtype)
    return AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )


class SvdProjectedRoutingTrainer(Trainer):
    def __init__(
        self,
        *args,
        routing_patch: SvdProjectedRoutingPatch,
        load_balance_loss_weight: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.routing_patch = routing_patch
        self.load_balance_loss_weight = float(load_balance_loss_weight)
        self._last_aux: dict[str, float] = {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.routing_patch.clear()
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        inputs["use_cache"] = False
        outputs = model(**inputs)
        logits = outputs.logits
        prediction_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            labels.reshape(-1),
            ignore_index=-100,
        )
        loss = prediction_loss
        load_balance = None
        if self.load_balance_loss_weight > 0.0:
            load_balance = self.routing_patch.load_balance_loss()
            if load_balance is not None:
                loss = loss + self.load_balance_loss_weight * load_balance
        outputs.loss = loss

        with torch.no_grad():
            mask = labels.ne(-100)
            pred = logits.argmax(dim=-1)
            correct = (pred.eq(labels) & mask).sum()
            count = mask.sum().clamp_min(1)
            acc = correct.float() / count.float()
        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            self._last_aux = {
                "loss_lm": float(prediction_loss.detach().float().cpu()),
                "accuracy": float(acc.detach().float().cpu()),
                "loss_load_balance": float(load_balance.detach().float().cpu()) if load_balance is not None else 0.0,
            }
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if self._last_aux:
            logs = {**logs, **self._last_aux}
        try:
            return super().log(logs, start_time=start_time)
        except TypeError:
            return super().log(logs)


def build_dataset(args: argparse.Namespace, tokenizer):
    if args.data_mode == "synthetic":
        return SyntheticStructuredBlockDataset(
            seq_length=args.seq_length,
            vocab_size=args.synthetic_vocab_size,
            topic_count=args.synthetic_topic_count,
            entities_per_topic=args.synthetic_entities_per_topic,
            noise_rate=args.synthetic_noise_rate,
            seed=args.seed,
        )
    return RandomTextBlockDataset(
        dataset_path=args.dataset_path,
        data_files_glob=args.data_files_glob,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        seed=args.seed,
        min_text_chars=args.min_text_chars,
    )


def resolve_resume_checkpoint(args: argparse.Namespace) -> str | None:
    value = args.resume_from_checkpoint.strip()
    if not value:
        return None
    if value.lower() == "auto":
        return get_last_checkpoint(args.output_dir)
    return value


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0:
        (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = load_model(args, dtype)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    routing_cfg = SvdProjectedRoutingConfig(
        projection_source=args.projection_source,
        svd_refresh_interval=args.svd_refresh_interval,
        group1_experts=args.group1_experts,
        group2_experts=args.group2_experts,
        group3_experts=args.group3_experts,
        group1_topk=args.group1_topk,
        group2_topk=args.group2_topk,
        group3_topk=args.group3_topk,
        normalize_topk_prob=args.normalize_topk_prob,
    )
    routing_patch = SvdProjectedRoutingPatch(model, routing_cfg)
    routing_patch.apply()

    rank, _world_size = distributed_rank_info()
    if rank == 0:
        total_params = sum(param.numel() for param in model.parameters())
        print(f"patched_moe_layers={routing_patch.num_patched_layers}", flush=True)
        print(f"projection_source={args.projection_source}", flush=True)
        print(f"expert_groups=1+{args.group1_experts}+{args.group2_experts}+{args.group3_experts}", flush=True)
        print(f"active_experts=1+{args.group1_topk}+{args.group2_topk}+{args.group3_topk}", flush=True)
        print(f"model_size_preset={args.model_size_preset}", flush=True)
        print(f"model_config={json.dumps(model_config_summary(model.config), sort_keys=True)}", flush=True)
        print(f"total_parameters={total_params:,}", flush=True)

    train_dataset = build_dataset(args, tokenizer)
    report_to = [] if args.report_to.lower() in {"", "none"} else [args.report_to]
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=report_to,
        logging_dir=str(output_dir / "tensorboard"),
        seed=args.seed,
        data_seed=args.seed,
        deepspeed=args.deepspeed_config or None,
    )
    trainer = SvdProjectedRoutingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=causal_lm_collator,
        routing_patch=routing_patch,
        load_balance_loss_weight=args.load_balance_loss_weight,
    )
    trainer.train(resume_from_checkpoint=resolve_resume_checkpoint(args))
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    routing_patch.close()


if __name__ == "__main__":
    main()
