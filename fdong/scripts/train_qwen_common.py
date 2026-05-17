import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup

from models import MyQwen3ForCausalLM
from utils import HierarchicalPatternData, TokenizedJSONLData


def parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def add_common_training_args(parser: argparse.ArgumentParser):
    parser.add_argument("--local_batch_size", type=int, default=16)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=1024)

    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--optimizer", type=str, choices=["AdamW", "sgd"], default="AdamW")
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--total_training_steps", type=int, default=1000000)

    parser.add_argument("--data_shuffle", action="store_true", default=True)
    parser.add_argument("--no_data_shuffle", action="store_false", dest="data_shuffle")

    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--no_use_bf16", action="store_false", dest="use_bf16")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--config_dir", type=str, default="../../Qwen3-0.6B")
    parser.add_argument("--data_dir", type=str, default="../../dclm/global-shard_01_of_10")
    parser.add_argument("--ckpt_dir", type=str, default="")

    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["jsonl", "pruned", "synthetic_indexed", "hierarchical_pattern"],
        default="jsonl",
    )
    parser.add_argument("--per", type=float, default=1.0)
    parser.add_argument("--synthetic_num_samples", type=int, default=100000)
    parser.add_argument("--synthetic_block_size", type=int, default=4)
    parser.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    parser.add_argument("--synthetic_content_token_count", type=int, default=2048)
    parser.add_argument("--synthetic_num_units_per_layer", type=int, default=256)
    parser.add_argument("--synthetic_seed", type=int, default=0)
    parser.add_argument("--synthetic_pad_token_id", type=int, default=0)
    parser.add_argument("--synthetic_min_token_id", type=int, default=1)
    parser.add_argument(
        "--synthetic_sampling_distribution",
        type=str,
        choices=["uniform", "zipf"],
        default="uniform",
    )
    parser.add_argument("--synthetic_zipf_alpha", type=float, default=1.0)
    parser.add_argument("--synthetic_zipf_shuffle_ranks", action="store_true", default=True)
    parser.add_argument("--synthetic_no_zipf_shuffle_ranks", action="store_false", dest="synthetic_zipf_shuffle_ranks")

    parser.add_argument("--debug_vocab_size", type=int, default=-1)
    parser.add_argument("--debug_hidden_size", type=int, default=-1)
    parser.add_argument("--debug_intermediate_size", type=int, default=-1)
    parser.add_argument("--debug_num_hidden_layers", type=int, default=-1)
    parser.add_argument("--debug_num_attention_heads", type=int, default=-1)
    parser.add_argument("--debug_num_key_value_heads", type=int, default=-1)
    parser.add_argument("--debug_head_dim", type=int, default=-1)
    parser.add_argument("--debug_max_position_embeddings", type=int, default=-1)

    parser.add_argument("--use_moe", action="store_true", default=False)
    parser.add_argument("--moe_num_unique_experts", type=int, default=4)
    parser.add_argument("--moe_num_experts_per_tok", type=int, default=1)
    parser.add_argument("--moe_intermediate_size", type=int, default=-1)
    parser.add_argument("--moe_use_common_expert", action="store_true", default=False)
    parser.add_argument("--moe_common_intermediate_size", type=int, default=-1)
    parser.add_argument("--moe_router_bias", action="store_true", default=False)
    parser.add_argument("--moe_no_normalize_topk_prob", action="store_true", default=False)
    parser.add_argument(
        "--moe_router_input",
        type=str,
        choices=["hidden", "attention_output"],
        default="hidden",
    )
    parser.add_argument("--moe_head_level", action="store_true", default=False)

    parser.add_argument("--attention_stride_pattern", type=parse_int_list, default=None)
    parser.add_argument("--residual_source_pattern", type=parse_int_list, default=None)


def parse_common_args():
    parser = argparse.ArgumentParser(description="Training configuration")
    add_common_training_args(parser)
    return parser.parse_args()


def resolve_model_patterns(config, args):
    attention_stride_pattern = args.attention_stride_pattern or [1 for _ in range(config.num_hidden_layers)]
    residual_source_pattern = args.residual_source_pattern or [-1 for _ in range(config.num_hidden_layers)]
    return attention_stride_pattern, residual_source_pattern


def apply_debug_model_overrides(config, args):
    overrides = {
        "vocab_size": args.debug_vocab_size,
        "hidden_size": args.debug_hidden_size,
        "intermediate_size": args.debug_intermediate_size,
        "num_hidden_layers": args.debug_num_hidden_layers,
        "num_attention_heads": args.debug_num_attention_heads,
        "num_key_value_heads": args.debug_num_key_value_heads,
        "head_dim": args.debug_head_dim,
        "max_position_embeddings": args.debug_max_position_embeddings,
    }
    for name, value in overrides.items():
        if value != -1:
            setattr(config, name, value)

    if args.debug_vocab_size != -1:
        if getattr(config, "pad_token_id", None) is None or config.pad_token_id >= args.debug_vocab_size:
            config.pad_token_id = 0
        if getattr(config, "bos_token_id", None) is not None and config.bos_token_id >= args.debug_vocab_size:
            config.bos_token_id = 1
        if getattr(config, "eos_token_id", None) is not None and config.eos_token_id >= args.debug_vocab_size:
            config.eos_token_id = 2


def apply_moe_overrides(config, args):
    config.use_moe = bool(args.use_moe)
    config.moe_num_unique_experts = int(args.moe_num_unique_experts)
    config.moe_num_experts_per_tok = int(args.moe_num_experts_per_tok)
    config.moe_intermediate_size = (
        int(args.moe_intermediate_size)
        if args.moe_intermediate_size != -1
        else int(config.intermediate_size)
    )
    config.moe_use_common_expert = bool(args.moe_use_common_expert)
    config.moe_common_intermediate_size = (
        int(args.moe_common_intermediate_size)
        if args.moe_common_intermediate_size != -1
        else int(config.moe_intermediate_size)
    )
    config.moe_router_bias = bool(args.moe_router_bias)
    config.moe_normalize_topk_prob = not bool(args.moe_no_normalize_topk_prob)
    config.moe_router_input = str(args.moe_router_input)
    config.moe_head_level = bool(args.moe_head_level)


def write_runtime_config(ckpt_dir, attention_stride_pattern, residual_source_pattern, config=None):
    if not ckpt_dir:
        return
    runtime_config_path = os.path.join(ckpt_dir, "runtime_config.json")
    runtime_config = {
        "attention_stride_pattern": attention_stride_pattern,
        "residual_source_pattern": residual_source_pattern,
    }
    if config is not None:
        runtime_config.update(
            {
                "use_moe": bool(getattr(config, "use_moe", False)),
                "moe_num_unique_experts": int(getattr(config, "moe_num_unique_experts", 0)),
                "moe_num_experts_per_tok": int(getattr(config, "moe_num_experts_per_tok", 0)),
                "moe_intermediate_size": int(getattr(config, "moe_intermediate_size", 0)),
                "moe_use_common_expert": bool(getattr(config, "moe_use_common_expert", False)),
                "moe_common_intermediate_size": int(getattr(config, "moe_common_intermediate_size", 0)),
                "moe_router_bias": bool(getattr(config, "moe_router_bias", False)),
                "moe_normalize_topk_prob": bool(getattr(config, "moe_normalize_topk_prob", True)),
                "moe_router_input": str(getattr(config, "moe_router_input", "hidden")),
                "moe_head_level": bool(getattr(config, "moe_head_level", False)),
            }
        )
    with open(runtime_config_path, "w", encoding="utf-8") as f:
        json.dump(runtime_config, f, indent=2)


@torch.no_grad()
def prepare_model(local_rank, world_size, device, args):
    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    apply_debug_model_overrides(config, args)
    apply_moe_overrides(config, args)
    config._attn_implementation = "eager"
    config.attention_stride_pattern, config.residual_source_pattern = resolve_model_patterns(config, args)
    if local_rank == 0:
        print(
            "Model size: "
            f"layers={config.num_hidden_layers}, hidden={config.hidden_size}, "
            f"intermediate={config.intermediate_size}, heads={config.num_attention_heads}, "
            f"kv_heads={config.num_key_value_heads}, head_dim={config.head_dim}, "
            f"vocab={config.vocab_size}",
            flush=True,
        )
        print(f"Model attention_stride_pattern: {config.attention_stride_pattern}", flush=True)
        print(f"Model residual_source_pattern: {config.residual_source_pattern}", flush=True)
        print(
            "Model MoE: "
            f"use_moe={config.use_moe}, unique_experts={config.moe_num_unique_experts}, "
            f"topk={config.moe_num_experts_per_tok}, moe_intermediate={config.moe_intermediate_size}, "
            f"use_common={config.moe_use_common_expert}, "
            f"common_intermediate={config.moe_common_intermediate_size}, "
            f"router_input={config.moe_router_input}, head_level={config.moe_head_level}",
            flush=True,
        )
    model = MyQwen3ForCausalLM(config).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[device])

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"rank {local_rank} model ok, params: {trainable_params / 1e9:.2f}B/{total_params / 1e9:.2f}B")
    return model


def prepare_data(local_rank, world_size, args):
    tokenizer = AutoTokenizer.from_pretrained(args.config_dir, trust_remote_code=True)
    if args.dataset_type == "hierarchical_pattern":
        dataset = HierarchicalPatternData(
            max_seq_len=args.seq_len,
            num_samples=args.synthetic_num_samples,
            block_size=args.synthetic_block_size,
            num_hierarchy_layers=args.synthetic_num_hierarchy_layers,
            content_token_count=args.synthetic_content_token_count,
            num_units_per_layer=args.synthetic_num_units_per_layer,
            seed=args.synthetic_seed,
            pad_token_id=args.synthetic_pad_token_id,
            min_token_id=args.synthetic_min_token_id,
            sampling_distribution=args.synthetic_sampling_distribution,
            zipf_alpha=args.synthetic_zipf_alpha,
            zipf_shuffle_ranks=args.synthetic_zipf_shuffle_ranks,
        )
    else:
        dataset = TokenizedJSONLData(args.data_dir, args.seq_len, tokenizer)
    print(f"Construct dataset, total {len(dataset)} samples.")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=args.data_shuffle)
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, num_workers=args.num_workers, sampler=sampler)
    return dataloader


def prepare_loss_optimizer(model, args):
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=151643)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.01)
    else:
        print(f"Unsupported optimizer: {args.optimizer}, using Adam by default.")
        optimizer = torch.optim.Adam(params, lr=args.lr)

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.total_training_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    return token_loss_fn, optimizer, lr_scheduler, scaler


def forward_step(local_rank, device, source, target, model, token_loss_fn, args):
    source, target = source.to(device), target.to(device)
    output = model(source, output_hidden_states=False)
    target = target.reshape(-1)
    loss = token_loss_fn(output.logits.view(-1, output.logits.size(-1)), target)
    return loss


def update_step(optimizer, scheduler):
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def thread_main(local_rank, world_size, device, args):
    print(f"running on device {local_rank}")
    if local_rank == 0 and args.ckpt_dir and not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    dataloader = prepare_data(local_rank, world_size, args)
    model = prepare_model(local_rank, world_size, device, args)
    real_model = model.module if world_size > 1 else model
    if local_rank == 0:
        write_runtime_config(
            args.ckpt_dir,
            real_model.model.attention_stride_pattern,
            real_model.model.residual_source_pattern,
            real_model.config,
        )
    token_loss_fn, optimizer, lr_scheduler, scaler = prepare_loss_optimizer(model, args)
    autocast_enabled = args.use_bf16 and device.type == "cuda"

    gradient_accumulation_steps = args.global_batch_size // args.local_batch_size // world_size
    if gradient_accumulation_steps < 1:
        raise ValueError("global_batch_size must be >= local_batch_size * world_size")

    for local_batch_idx, (source, target, real_lens) in enumerate(dataloader, 1):
        global_batch_idx = local_batch_idx // gradient_accumulation_steps
        start_time = time.time()

        with torch.amp.autocast(dtype=torch.bfloat16, device_type=device.type, enabled=autocast_enabled):
            loss = forward_step(local_rank, device, source, target, model, token_loss_fn, args)

        if world_size == 1:
            (loss / gradient_accumulation_steps).backward()
        else:
            scaler.scale(loss / gradient_accumulation_steps).backward()

        if local_batch_idx % gradient_accumulation_steps == 0:
            if world_size == 1:
                update_step(optimizer, lr_scheduler)
            else:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

            if local_rank == 0 and args.ckpt_dir and global_batch_idx % args.save_interval == 0:
                torch.save(real_model.state_dict(), f"{args.ckpt_dir}/{global_batch_idx}.pth")

        batch_time = time.time() - start_time
        if local_rank == 0:
            print(f"batch: {global_batch_idx}-{local_batch_idx}, loss: {loss:.3f}, batch_time: {batch_time:.3f}", flush=True)

        if global_batch_idx >= args.total_training_steps:
            if local_rank == 0 and args.ckpt_dir:
                torch.save(real_model.state_dict(), f"{args.ckpt_dir}/{global_batch_idx}.pth")
            break
