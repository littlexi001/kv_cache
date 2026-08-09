from __future__ import annotations

import argparse
import json
import math
import os
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--token-blocks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=153)
    parser.add_argument("--eval-sequences", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--save-steps", type=int, default=40)
    return parser.parse_args()


class PackedTokenDataset(Dataset):
    def __init__(self, path: Path, sequence_length: int, split: str, eval_sequences: int) -> None:
        blocks = np.load(path, mmap_mode="r")
        self.tokens = blocks.reshape(-1)
        self.sequence_length = int(sequence_length)
        total_sequences = len(self.tokens) // self.sequence_length
        if total_sequences <= eval_sequences + 8:
            raise ValueError("token stream is too short")
        if split == "train":
            self.first = 0
            self.count = total_sequences - eval_sequences
        elif split == "eval":
            self.first = total_sequences - eval_sequences
            self.count = eval_sequences
        else:
            raise ValueError(split)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = (self.first + int(index)) * self.sequence_length
        values = np.array(self.tokens[start : start + self.sequence_length], dtype=np.int64, copy=True)
        ids = torch.from_numpy(values)
        return {"input_ids": ids, "labels": ids.clone()}


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in batch]) for key in batch[0]}


def nope_layers(num_layers: int) -> tuple[int, ...]:
    return tuple(range(3, num_layers, 4))


def patch_nope_layers(model: Any) -> tuple[int, ...]:
    selected = frozenset(nope_layers(int(model.config.num_hidden_layers)))
    found = 0
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3Attention":
            continue
        original = module.forward

        def wrapped_forward(
            this: Any,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_value: Any = None,
            cache_position: torch.Tensor | None = None,
            _original: Any = original,
            **kwargs: Any,
        ) -> Any:
            if int(this.layer_idx) in selected:
                cos, sin = position_embeddings
                position_embeddings = (torch.ones_like(cos), torch.zeros_like(sin))
            return _original(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )

        module.forward = types.MethodType(wrapped_forward, module)
        found += 1
    if found != int(model.config.num_hidden_layers):
        raise RuntimeError(f"patched {found} attention modules")
    return tuple(sorted(selected))


def reduce_mean(value: torch.Tensor, world_size: int) -> float:
    detached = value.detach().float()
    if world_size > 1:
        dist.all_reduce(detached, op=dist.ReduceOp.SUM)
        detached /= world_size
    return float(detached.item())


@torch.no_grad()
def evaluate(model: Any, loader: DataLoader, device: torch.device, world_size: int) -> float:
    model.eval()
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        output = model(**batch, use_cache=False)
        total[0] += output.loss.detach().double()
        total[1] += 1
    if world_size > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    if int(total[1].item()) == 0:
        raise RuntimeError("empty evaluation loader")
    return float((total[0] / total[1]).item())


def lr_multiplier(step: int, max_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from bitsandbytes.optim import PagedAdamW8bit
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        lora = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora)
    selected_layers = patch_nope_layers(model)
    trainable, total = model.get_nb_trainable_parameters()

    train_data = PackedTokenDataset(
        args.token_blocks, args.sequence_length, "train", args.eval_sequences
    )
    eval_data = PackedTokenDataset(
        args.token_blocks, args.sequence_length, "eval", args.eval_sequences
    )
    train_sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=local_rank, shuffle=True, seed=args.seed
    )
    eval_sampler = DistributedSampler(
        eval_data, num_replicas=world_size, rank=local_rank, shuffle=False
    )
    train_loader = DataLoader(
        train_data, batch_size=1, sampler=train_sampler, collate_fn=collate,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_data, batch_size=1, sampler=eval_sampler, collate_fn=collate,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = PagedAdamW8bit(
        trainable_parameters, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )
    warmup_steps = max(1, int(round(0.05 * args.max_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, args.max_steps, warmup_steps)
    )
    distributed_model: Any = model
    if world_size > 1:
        distributed_model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=False,
        )

    config = {
        "model": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "token_blocks": str(args.token_blocks),
        "sequence_length": args.sequence_length,
        "max_steps": args.max_steps,
        "world_size": world_size,
        "tokens_scheduled": args.sequence_length * args.max_steps * world_size,
        "eval_sequences": args.eval_sequences,
        "nope_layers": list(selected_layers),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "seed": args.seed,
    }
    if local_rank == 0:
        (args.output_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    before_loss = evaluate(distributed_model, eval_loader, device, world_size)
    if world_size > 1:
        dist.barrier()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    epoch = 0
    log_path = args.output_dir / "train_metrics.jsonl"
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            train_sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        step_started = time.time()
        output = distributed_model(**batch, use_cache=False)
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        mean_loss = reduce_mean(loss, world_size)
        mean_grad = reduce_mean(torch.as_tensor(grad_norm, device=device), world_size)
        if local_rank == 0:
            row = {
                "step": step,
                "loss": mean_loss,
                "ppl": math.exp(min(mean_loss, 30.0)),
                "grad_norm": mean_grad,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "step_seconds": time.time() - step_started,
                "tokens_seen": step * args.sequence_length * world_size,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if step % min(max(1, args.save_steps), args.max_steps) == 0 or step == args.max_steps:
            if world_size > 1:
                dist.barrier()
            if local_rank == 0:
                checkpoint = args.output_dir / "checkpoints" / f"step_{step:06d}"
                checkpoint.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint)
            if world_size > 1:
                dist.barrier()

    after_loss = evaluate(distributed_model, eval_loader, device, world_size)
    final_adapter = args.output_dir / "final_adapter"
    if world_size > 1:
        dist.barrier()
    if local_rank == 0:
        model.save_pretrained(final_adapter)
        tokenizer.save_pretrained(final_adapter)
        metrics = {
            "before_loss": before_loss,
            "after_loss": after_loss,
            "before_ppl": math.exp(min(before_loss, 30.0)),
            "after_ppl": math.exp(min(after_loss, 30.0)),
            "wall_seconds": time.time() - started,
            "peak_allocated_gib_rank0": torch.cuda.max_memory_allocated(device) / (1024**3),
        }
        (args.output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "train.done").write_text("ok\n", encoding="utf-8")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
