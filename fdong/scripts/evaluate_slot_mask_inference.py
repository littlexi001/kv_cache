import argparse
import json
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from models import MyQwen3ForCausalLM
from utils import HierarchicalPatternData


def parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def add_args():
    parser = argparse.ArgumentParser(description="Evaluate feature-masked attention on hierarchical synthetic data.")
    parser.add_argument("--config_dir", type=str, default="../Qwen3-0.6B")
    parser.add_argument("--ckpt_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="../experiments/slot_mask_inference.json")

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--synthetic_block_size", type=int, default=4)
    parser.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    parser.add_argument("--synthetic_content_token_count", type=int, default=256)
    parser.add_argument("--synthetic_num_units_per_layer", type=int, default=64)
    parser.add_argument("--synthetic_seed", type=int, default=0)
    parser.add_argument("--synthetic_min_token_id", type=int, default=1)

    parser.add_argument("--debug_vocab_size", type=int, default=257)
    parser.add_argument("--debug_hidden_size", type=int, default=128)
    parser.add_argument("--debug_intermediate_size", type=int, default=256)
    parser.add_argument("--debug_num_hidden_layers", type=int, default=3)
    parser.add_argument("--debug_num_attention_heads", type=int, default=4)
    parser.add_argument("--debug_num_key_value_heads", type=int, default=2)
    parser.add_argument("--debug_head_dim", type=int, default=32)
    parser.add_argument("--debug_max_position_embeddings", type=int, default=256)

    parser.add_argument("--attention_stride_pattern", type=parse_int_list, default=None)
    parser.add_argument("--residual_source_pattern", type=parse_int_list, default=None)

    parser.add_argument("--use_moe", action="store_true", default=True)
    parser.add_argument("--moe_num_unique_experts", type=int, default=4)
    parser.add_argument("--moe_num_experts_per_tok", type=int, default=1)
    parser.add_argument("--moe_intermediate_size", type=int, default=128)
    parser.add_argument("--moe_use_common_expert", action="store_true", default=False)
    parser.add_argument("--moe_common_intermediate_size", type=int, default=128)
    parser.add_argument("--moe_router_bias", action="store_true", default=False)
    parser.add_argument("--moe_no_normalize_topk_prob", action="store_true", default=False)

    parser.add_argument(
        "--modes",
        type=str,
        default="full,same_slot_occurrence,same_slot,same_higher,same_slot_or_higher,random_same_size",
        help="Comma-separated mask modes.",
    )
    parser.add_argument("--slot_layer", type=int, default=0)
    parser.add_argument("--higher_layer", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=123)
    return parser.parse_args()


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def maybe_override(config, name, value):
    if value != -1:
        setattr(config, name, value)


def build_config(args):
    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    maybe_override(config, "vocab_size", args.debug_vocab_size)
    maybe_override(config, "hidden_size", args.debug_hidden_size)
    maybe_override(config, "intermediate_size", args.debug_intermediate_size)
    maybe_override(config, "num_hidden_layers", args.debug_num_hidden_layers)
    maybe_override(config, "num_attention_heads", args.debug_num_attention_heads)
    maybe_override(config, "num_key_value_heads", args.debug_num_key_value_heads)
    maybe_override(config, "head_dim", args.debug_head_dim)
    maybe_override(config, "max_position_embeddings", args.debug_max_position_embeddings)

    if getattr(config, "pad_token_id", None) is None or config.pad_token_id >= config.vocab_size:
        config.pad_token_id = 0
    if getattr(config, "bos_token_id", None) is not None and config.bos_token_id >= config.vocab_size:
        config.bos_token_id = 1
    if getattr(config, "eos_token_id", None) is not None and config.eos_token_id >= config.vocab_size:
        config.eos_token_id = 2

    config._attn_implementation = "eager"
    config.use_cache = False
    config.attention_stride_pattern = args.attention_stride_pattern or [1] * config.num_hidden_layers
    config.residual_source_pattern = args.residual_source_pattern or [-1] * config.num_hidden_layers
    config.use_moe = bool(args.use_moe)
    config.moe_num_unique_experts = int(args.moe_num_unique_experts)
    config.moe_num_experts_per_tok = int(args.moe_num_experts_per_tok)
    config.moe_intermediate_size = int(args.moe_intermediate_size)
    config.moe_use_common_expert = bool(args.moe_use_common_expert)
    config.moe_common_intermediate_size = int(args.moe_common_intermediate_size)
    config.moe_router_bias = bool(args.moe_router_bias)
    config.moe_normalize_topk_prob = not bool(args.moe_no_normalize_topk_prob)
    return config


def load_model(args, device):
    config = build_config(args)
    model = MyQwen3ForCausalLM(config).to(device)
    state = torch.load(args.ckpt_file, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def build_allowed_mask(mode, metadata, slot_layer, higher_layer, random_seed, block_size):
    # metadata: [batch, seq, layers], aligned with source tokens.
    batch, seq_len, _ = metadata.shape
    device = metadata.device
    positions = torch.arange(seq_len, device=device)
    causal = positions[None, :] <= positions[:, None]
    causal = causal[None, :, :].expand(batch, -1, -1)

    if mode == "full":
        return causal

    if mode == "same_slot_occurrence":
        occurrence_ids = positions // int(block_size)
        same_occurrence = occurrence_ids[:, None] == occurrence_ids[None, :]
        return same_occurrence[None, :, :].expand(batch, -1, -1) & causal

    slot_ids = metadata[:, :, slot_layer]
    same_slot = slot_ids[:, :, None] == slot_ids[:, None, :]
    allowed = same_slot & causal & (slot_ids[:, :, None] >= 0) & (slot_ids[:, None, :] >= 0)

    if mode == "same_slot":
        return allowed

    if higher_layer < 0 or higher_layer >= metadata.shape[-1]:
        if mode in {"same_higher", "same_slot_or_higher"}:
            raise ValueError(f"Invalid higher_layer={higher_layer} for metadata shape {tuple(metadata.shape)}.")
    higher_ids = metadata[:, :, higher_layer]
    same_higher = higher_ids[:, :, None] == higher_ids[:, None, :]
    same_higher = same_higher & causal & (higher_ids[:, :, None] >= 0) & (higher_ids[:, None, :] >= 0)

    if mode == "same_higher":
        return same_higher
    if mode == "same_slot_or_higher":
        return allowed | same_higher
    if mode != "random_same_size":
        raise ValueError(f"Unknown mask mode: {mode}")

    random_allowed = torch.zeros_like(allowed)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(random_seed))
    for b in range(batch):
        for i in range(seq_len):
            causal_indices = torch.nonzero(causal[b, i], as_tuple=False).flatten()
            same_count = int(allowed[b, i].sum().item())
            if same_count <= 0:
                continue
            # Keep self visible in the size-matched random baseline, then sample
            # the remaining keys from the valid causal prefix.
            random_allowed[b, i, i] = True
            remaining = max(same_count - 1, 0)
            candidates = causal_indices[causal_indices != i]
            if remaining > 0 and candidates.numel() > 0:
                perm = torch.randperm(candidates.numel(), generator=generator, device=device)
                chosen = candidates[perm[: min(remaining, candidates.numel())]]
                random_allowed[b, i, chosen] = True
    return random_allowed


def build_additive_mask(mode, metadata, slot_layer, higher_layer, random_seed, dtype, block_size):
    allowed = build_allowed_mask(mode, metadata, slot_layer, higher_layer, random_seed, block_size)
    min_dtype = torch.finfo(dtype).min
    mask = torch.zeros(allowed.shape, dtype=dtype, device=metadata.device)
    mask = mask.masked_fill(~allowed, min_dtype)
    return mask[:, None, :, :], allowed


def masked_loss_and_accuracy(logits, target, eval_mask: Optional[torch.Tensor] = None):
    valid = target != 0
    if eval_mask is not None:
        valid = valid & eval_mask
    if valid.sum().item() == 0:
        return {"loss": None, "accuracy": None, "count": 0}
    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(target)
    pred = logits.argmax(dim=-1)
    return {
        "loss": float(per_token_loss[valid].mean().item()),
        "accuracy": float(((pred == target) & valid).sum().item() / valid.sum().item()),
        "count": int(valid.sum().item()),
    }


def init_metric_state():
    return {
        "loss_sum": 0.0,
        "correct": 0,
        "count": 0,
        "internal_loss_sum": 0.0,
        "internal_correct": 0,
        "internal_count": 0,
        "boundary_loss_sum": 0.0,
        "boundary_correct": 0,
        "boundary_count": 0,
        "visible_fraction_sum": 0.0,
        "visible_fraction_count": 0,
    }


def update_metric_state(state, logits, target, metadata, allowed, slot_layer):
    valid = target != 0
    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(target)
    pred = logits.argmax(dim=-1)

    state["loss_sum"] += float(per_token_loss[valid].sum().item())
    state["correct"] += int(((pred == target) & valid).sum().item())
    state["count"] += int(valid.sum().item())

    source_slot = metadata[:, :, slot_layer]
    internal = torch.zeros_like(valid)
    internal[:, :-1] = (source_slot[:, :-1] == source_slot[:, 1:]) & (source_slot[:, :-1] >= 0)
    boundary = torch.zeros_like(valid)
    boundary[:, :-1] = (source_slot[:, :-1] != source_slot[:, 1:]) & (source_slot[:, :-1] >= 0) & (source_slot[:, 1:] >= 0)

    for prefix, mask in (("internal", internal & valid), ("boundary", boundary & valid)):
        if mask.sum().item() == 0:
            continue
        state[f"{prefix}_loss_sum"] += float(per_token_loss[mask].sum().item())
        state[f"{prefix}_correct"] += int(((pred == target) & mask).sum().item())
        state[f"{prefix}_count"] += int(mask.sum().item())

    causal_count = torch.arange(1, target.shape[1] + 1, device=target.device, dtype=torch.float32)
    causal_count = causal_count[None, :].expand(target.shape[0], -1)
    visible_fraction = allowed.sum(dim=-1).to(torch.float32) / causal_count
    state["visible_fraction_sum"] += float(visible_fraction[valid].sum().item())
    state["visible_fraction_count"] += int(valid.sum().item())


def finalize_metric_state(state):
    count = max(state["count"], 1)
    result = {
        "loss": state["loss_sum"] / count,
        "accuracy": state["correct"] / count,
        "count": state["count"],
        "visible_fraction": state["visible_fraction_sum"] / max(state["visible_fraction_count"], 1),
    }
    for prefix in ("internal", "boundary"):
        prefix_count = state[f"{prefix}_count"]
        result[f"{prefix}_loss"] = None if prefix_count == 0 else state[f"{prefix}_loss_sum"] / prefix_count
        result[f"{prefix}_accuracy"] = None if prefix_count == 0 else state[f"{prefix}_correct"] / prefix_count
        result[f"{prefix}_count"] = prefix_count
    return result


@torch.no_grad()
def main():
    args = add_args()
    device = choose_device()
    dataset = HierarchicalPatternData(
        max_seq_len=args.seq_len,
        num_samples=args.num_samples,
        block_size=args.synthetic_block_size,
        num_hierarchy_layers=args.synthetic_num_hierarchy_layers,
        content_token_count=args.synthetic_content_token_count,
        num_units_per_layer=args.synthetic_num_units_per_layer,
        seed=args.synthetic_seed,
        min_token_id=args.synthetic_min_token_id,
        return_metadata=True,
    )
    model = load_model(args, device)
    dtype = next(model.parameters()).dtype

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    states: Dict[str, Dict] = {mode: init_metric_state() for mode in modes}

    for start in range(0, args.num_samples, args.batch_size):
        batch_items = [dataset[i] for i in range(start, min(start + args.batch_size, args.num_samples))]
        source = torch.stack([item[0] for item in batch_items]).to(device)
        target = torch.stack([item[1] for item in batch_items]).to(device)
        metadata = torch.stack([item[3] for item in batch_items]).to(device)

        for mode in modes:
            attention_mask, allowed = build_additive_mask(
                mode,
                metadata,
                slot_layer=args.slot_layer,
                higher_layer=args.higher_layer,
                random_seed=args.random_seed + start,
                dtype=dtype,
                block_size=args.synthetic_block_size,
            )
            outputs = model(
                source,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
            )
            update_metric_state(states[mode], outputs.logits, target, metadata, allowed, args.slot_layer)

    summary = {
        "config": vars(args),
        "device": str(device),
        "results": {mode: finalize_metric_state(state) for mode, state in states.items()},
    }
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
