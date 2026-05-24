from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from qwen15_moe_real_cluster_patch import RealAttentionClusterConfig, RealAttentionClusterPatch


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="/mnt/workspace/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--dataset_path", default="/mnt/workspace/dclm")
    parser.add_argument("--data_files_glob", default="**/*.txt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="qwen15-moe-real-attn-cluster")
    parser.add_argument("--init_from_scratch", type=str2bool, default=True)
    parser.add_argument(
        "--resume_from_checkpoint",
        default="",
        help="Use a checkpoint path, 'auto' to resume the latest checkpoint under output_dir, or empty for no resume.",
    )
    parser.add_argument(
        "--model_size_preset",
        choices=["none", "moe_0_6b"],
        default="none",
        help="Optionally shrink the loaded MoE config before random initialization.",
    )
    parser.add_argument(
        "--model_config_overrides",
        default="",
        help="Optional JSON object of config fields to override after --model_size_preset.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--min_text_chars", type=int, default=20)

    parser.add_argument("--attention_top_ratio", type=float, default=0.10)
    parser.add_argument("--expert_input_top_ratio", type=float, default=0.10)
    parser.add_argument("--include_self", type=str2bool, default=False)
    parser.add_argument("--attention_cluster_weight", type=float, default=0.01)
    parser.add_argument("--attention_cluster_temperature", type=float, default=1.0)
    parser.add_argument("--attention_cluster_detach_attention", type=str2bool, default=True)
    parser.add_argument("--attention_cluster_detach_key_router", type=str2bool, default=False)
    parser.add_argument("--load_balance_loss_weight", type=float, default=0.01)
    parser.add_argument("--load_balance_temperature", type=float, default=1.0)

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
    parser.add_argument("--attn_implementation", choices=["eager"], default="eager")
    parser.add_argument("--ddp_find_unused_parameters", type=str2bool, default=False)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--deepspeed_config", default="")
    parser.add_argument("--report_to", default="tensorboard")
    return parser.parse_args()


def distributed_rank_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


class RandomDclmLineBlockDataset(IterableDataset):
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
        stream_id = rank * num_workers + worker_id
        return random.Random(self.seed + 1000003 * stream_id + 9176 * world_size)

    def _sample_line(self, rng: random.Random) -> str:
        file_idx = rng.randrange(len(self.files))
        path = self.files[file_idx]
        file_size = self.file_sizes[file_idx]
        offset = rng.randrange(file_size)
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
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


def causal_lm_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in features]) for key in ("input_ids", "labels")}


class RealAttentionClusterTrainer(Trainer):
    def __init__(
        self,
        *args,
        cluster_patch: RealAttentionClusterPatch,
        attention_cluster_weight: float,
        load_balance_loss_weight: float,
        router_top_k: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cluster_patch = cluster_patch
        self.attention_cluster_weight = float(attention_cluster_weight)
        self.load_balance_loss_weight = float(load_balance_loss_weight)
        self.router_top_k = int(router_top_k)
        self._last_aux: dict[str, float] = {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.cluster_patch.clear()
        inputs = dict(inputs)
        inputs["output_attentions"] = True
        inputs["use_cache"] = False
        outputs = model(**inputs)
        lm_loss = outputs.loss
        loss = lm_loss

        attn_cluster = None
        if self.attention_cluster_weight > 0.0:
            attn_cluster = self.cluster_patch.attention_cluster_loss()
            if attn_cluster is not None:
                loss = loss + self.attention_cluster_weight * attn_cluster

        load_balance = None
        if self.load_balance_loss_weight > 0.0:
            load_balance = self.cluster_patch.load_balance_loss(top_k=self.router_top_k)
            if load_balance is not None:
                loss = loss + self.load_balance_loss_weight * load_balance

        outputs.loss = loss
        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            self._last_aux = {
                "loss_lm": float(lm_loss.detach().float().cpu()),
                "loss_attn_cluster": float(attn_cluster.detach().float().cpu()) if attn_cluster is not None else 0.0,
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


def infer_router_top_k(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    for name in ("num_experts_per_tok", "num_experts_per_token", "moe_top_k"):
        value = getattr(config, name, None) if config is not None else None
        if value is not None:
            return int(value)
    for module in model.modules():
        value = getattr(module, "top_k", None)
        if value is not None:
            return int(value)
    return 1


def model_config_summary(config: Any) -> dict[str, Any]:
    names = [
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


MOE_0_6B_CONFIG_OVERRIDES: dict[str, Any] = {
    "hidden_size": 768,
    "intermediate_size": 2048,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_key_value_heads": 4,
    "num_experts": 12,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
}


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

    cluster_cfg = RealAttentionClusterConfig(
        attention_top_ratio=args.attention_top_ratio,
        expert_input_top_ratio=args.expert_input_top_ratio,
        include_self=args.include_self,
        attention_cluster_temperature=args.attention_cluster_temperature,
        attention_cluster_detach_attention=args.attention_cluster_detach_attention,
        attention_cluster_detach_key_router=args.attention_cluster_detach_key_router,
        load_balance_temperature=args.load_balance_temperature,
    )
    cluster_patch = RealAttentionClusterPatch(model, cluster_cfg)
    cluster_patch.apply()
    router_top_k = infer_router_top_k(model)

    rank, _world_size = distributed_rank_info()
    if rank == 0:
        total_params = sum(param.numel() for param in model.parameters())
        print(f"patched_moe_layers={cluster_patch.num_patched_layers}", flush=True)
        print(f"router_top_k={router_top_k}", flush=True)
        print(f"model_size_preset={args.model_size_preset}", flush=True)
        print(f"model_config={json.dumps(model_config_summary(model.config), sort_keys=True)}", flush=True)
        print(f"total_parameters={total_params:,}", flush=True)

    train_dataset = RandomDclmLineBlockDataset(
        dataset_path=args.dataset_path,
        data_files_glob=args.data_files_glob,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        seed=args.seed,
        min_text_chars=args.min_text_chars,
    )

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

    trainer = RealAttentionClusterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=causal_lm_collator,
        cluster_patch=cluster_patch,
        attention_cluster_weight=args.attention_cluster_weight,
        load_balance_loss_weight=args.load_balance_loss_weight,
        router_top_k=router_top_k,
    )
    trainer.train(resume_from_checkpoint=resolve_resume_checkpoint(args))
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    cluster_patch.close()


if __name__ == "__main__":
    main()
