"""Shared training implementation for local debug and distributed pretraining."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from transformers import AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup

from models import MyQwen3ForCausalLM
from utils import RandomTokenDataset, TokenizedJSONLData


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="../../dclm/global-shard_01_of_10")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--dataset_type", choices=["dclm", "random"], default="dclm")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--local_batch_size", type=int, default=2)
    parser.add_argument("--global_batch_size", type=int, default=128)
    parser.add_argument("--total_training_steps", type=int, default=100000)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_bf16", type=str_to_bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=str_to_bool, default=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_parameters_billion", type=float, default=2.0)
    parser.add_argument(
        "--architecture",
        choices=["ordinary_moe", "shared_bucket"],
        default="shared_bucket",
    )

    parser.add_argument(
        "--router_input",
        choices=["layer_input", "q", "k", "v"],
        default="k",
        help="Pre-attention representation used by each head router.",
    )
    parser.add_argument("--center_router_input", type=str_to_bool, default=True)
    parser.add_argument("--router_normalization", choices=["none", "l2"], default="l2")
    parser.add_argument("--router_bias", type=str_to_bool, default=False)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--expert_intermediate_size", type=int, default=3072)
    parser.add_argument("--local_window", type=int, default=32)
    parser.add_argument("--sink_tokens", type=int, default=4)

    parser.add_argument("--debug_vocab_size", type=int, default=-1)
    parser.add_argument("--debug_hidden_size", type=int, default=-1)
    parser.add_argument("--debug_num_hidden_layers", type=int, default=-1)
    parser.add_argument("--debug_num_attention_heads", type=int, default=-1)
    parser.add_argument("--debug_num_key_value_heads", type=int, default=-1)
    parser.add_argument("--debug_head_dim", type=int, default=-1)
    parser.add_argument("--debug_random_samples", type=int, default=64)


def set_seed(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args) -> object:
    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    debug_overrides = {
        "vocab_size": args.debug_vocab_size,
        "hidden_size": args.debug_hidden_size,
        "num_hidden_layers": args.debug_num_hidden_layers,
        "num_attention_heads": args.debug_num_attention_heads,
        "num_key_value_heads": args.debug_num_key_value_heads,
        "head_dim": args.debug_head_dim,
    }
    for name, value in debug_overrides.items():
        if value > 0:
            setattr(config, name, value)

    config.inverse_kv_architecture = args.architecture
    config.inverse_kv_router_input = args.router_input
    config.inverse_kv_center_router_input = args.center_router_input
    config.inverse_kv_router_normalization = args.router_normalization
    config.inverse_kv_router_bias = args.router_bias
    config.inverse_kv_num_experts = args.num_experts
    config.inverse_kv_expert_intermediate_size = args.expert_intermediate_size
    config.inverse_kv_local_window = args.local_window
    config.inverse_kv_sink_tokens = args.sink_tokens
    config.use_cache = False
    config._attn_implementation = "eager"
    return config


def build_dataset(args, config):
    if args.dataset_type == "random":
        return RandomTokenDataset(
            num_samples=args.debug_random_samples,
            seq_len=args.seq_len,
            vocab_size=config.vocab_size,
            seed=args.seed,
        ), None

    tokenizer = AutoTokenizer.from_pretrained(args.config_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = TokenizedJSONLData(args.data_dir, args.seq_len, tokenizer, seed=args.seed)
    return dataset, tokenizer.pad_token_id


def count_parameters(model) -> Dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def runtime_config(args, config, parameters) -> Dict:
    effective_router_input = (
        "post_attention_residual" if args.architecture == "ordinary_moe" else args.router_input
    )
    return {
        "architecture": args.architecture,
        "transformers_version_required": "4.51.x",
        "config_dir": args.config_dir,
        "data_dir": args.data_dir,
        "dataset_type": args.dataset_type,
        "seq_len": args.seq_len,
        "global_batch_size": args.global_batch_size,
        "learning_rate": args.learning_rate,
        "router_input": args.router_input,
        "effective_router_input": effective_router_input,
        "center_router_input": args.center_router_input,
        "router_normalization": args.router_normalization,
        "router_bias": args.router_bias,
        "num_experts": args.num_experts,
        "expert_intermediate_size": args.expert_intermediate_size,
        "local_window": args.local_window,
        "sink_tokens": args.sink_tokens,
        "model_config": config.to_dict(),
        "parameters": parameters,
    }


def save_json(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)
    temporary.replace(path)


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int, args) -> None:
    real_model = model.module if isinstance(model, DDP) else model
    payload = {
        "step": step,
        "model": real_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "training_args": vars(args),
    }
    torch.save(payload, path)


def append_jsonl(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    reduced = value.detach().float().clone()
    if world_size > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= world_size
    return reduced


def train(rank: int, world_size: int, device: torch.device, args) -> None:
    set_seed(args.seed, rank)
    run_dir = Path(args.run_dir)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    config = build_config(args)
    dataset, pad_token_id = build_dataset(args, config)
    if isinstance(dataset, IterableDataset):
        dataset.set_distributed(rank, world_size)
        sampler = None
    else:
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.local_batch_size,
        sampler=sampler,
        shuffle=sampler is None and not isinstance(dataset, IterableDataset),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = MyQwen3ForCausalLM(config).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    parameters = count_parameters(model)
    if parameters["total"] > args.max_parameters_billion * 1e9:
        raise ValueError(
            f"Model has {parameters['total'] / 1e9:.3f}B parameters, above "
            f"--max_parameters_billion={args.max_parameters_billion:.3f}."
        )
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    real_model = model.module if isinstance(model, DDP) else model

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_training_steps,
    )

    accumulation_steps = args.global_batch_size // (args.local_batch_size * world_size)
    if accumulation_steps < 1 or args.global_batch_size % (args.local_batch_size * world_size) != 0:
        raise ValueError("global_batch_size must be divisible by local_batch_size * world_size")

    if rank == 0:
        save_json(run_dir / "runtime_config.json", runtime_config(args, config, parameters))
        print(
            f"model_parameters={parameters['total'] / 1e9:.3f}B "
            f"trainable={parameters['trainable'] / 1e9:.3f}B",
            flush=True,
        )

    metrics_path = run_dir / "train_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    epoch = 0
    window_start = time.time()
    window_tokens = 0

    while step < args.total_training_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        if isinstance(dataset, TokenizedJSONLData):
            dataset.set_epoch(epoch)
        for source, target, real_lengths in dataloader:
            source = source.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            real_lengths = torch.as_tensor(real_lengths, device=device)
            positions = torch.arange(source.shape[1], device=device).unsqueeze(0)
            valid_tokens = positions < (real_lengths.unsqueeze(1) - 1).clamp_min(0)
            attention_mask = valid_tokens.long()
            labels = target.masked_fill(~valid_tokens, -100)

            micro_step += 1
            should_step = micro_step % accumulation_steps == 0
            sync_context = contextlib.nullcontext()
            if isinstance(model, DDP) and not should_step:
                sync_context = model.no_sync()
            autocast_enabled = args.use_bf16 and device.type == "cuda"
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    output = model(
                        input_ids=source,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_attentions=False,
                    )
                    logits = output.logits
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        labels.reshape(-1),
                        ignore_index=-100,
                    )
                (loss / accumulation_steps).backward()

            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                correct = ((predictions == target) & valid_tokens).sum()
                valid_count = valid_tokens.sum().clamp_min(1)
                accuracy = correct.float() / valid_count
                window_tokens += int(valid_count.item()) * world_size

            if not should_step:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            mean_loss = reduce_mean(loss, world_size)
            mean_accuracy = reduce_mean(accuracy, world_size)
            if rank == 0 and step % args.log_interval == 0:
                elapsed = max(time.time() - window_start, 1e-6)
                record = {
                    "step": step,
                    "loss": float(mean_loss.item()),
                    "perplexity": float(math.exp(min(mean_loss.item(), 20.0))),
                    "next_token_accuracy": float(mean_accuracy.item()),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "gradient_norm": float(grad_norm),
                    "tokens_per_second": float(window_tokens / elapsed),
                    "time": time.time(),
                }
                record.update(real_model.routing_metrics())
                append_jsonl(metrics_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)
                window_start = time.time()
                window_tokens = 0

            if rank == 0 and step % args.save_interval == 0:
                save_checkpoint(run_dir / f"checkpoint-{step}.pt", model, optimizer, scheduler, step, args)

            if step >= args.total_training_steps:
                break
        epoch += 1

    if rank == 0:
        save_checkpoint(run_dir / f"checkpoint-{step}.pt", model, optimizer, scheduler, step, args)
        save_json(run_dir / "training_complete.json", {"step": step, "completed": True, "time": time.time()})
