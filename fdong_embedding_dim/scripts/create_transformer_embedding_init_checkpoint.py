#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoConfig

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FDONG_SCRIPTS = os.path.join(REPO_ROOT, "fdong", "scripts")
if FDONG_SCRIPTS not in sys.path:
    sys.path.insert(0, FDONG_SCRIPTS)

from models import MyQwen3ForCausalLM  # noqa: E402
from train_qwen_common import apply_debug_model_overrides, apply_moe_overrides  # noqa: E402
from analyze_frequency_width_dynamics import make_dataset, make_bucket_ids  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Create a controlled embedding-init checkpoint for fdong transformer runs.")
    p.add_argument("--config_dir", default="fdong/Qwen3-0.6B")
    p.add_argument("--output_path", required=True)
    p.add_argument("--runtime_config_path", default="")
    p.add_argument("--init_mode", choices=["spread", "packed_common", "packed_negative_common"], default="spread")
    p.add_argument("--init_scale", type=float, default=0.2)
    p.add_argument("--init_noise", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--num_samples", type=int, default=2048)
    p.add_argument("--synthetic_block_size", type=int, default=4)
    p.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    p.add_argument("--synthetic_content_token_count", type=int, default=256)
    p.add_argument("--synthetic_num_units_per_layer", type=int, default=64)
    p.add_argument("--synthetic_seed", type=int, default=0)
    p.add_argument("--synthetic_min_token_id", type=int, default=1)
    p.add_argument("--synthetic_zipf_alpha", type=float, default=1.3)
    p.add_argument("--synthetic_zipf_shuffle_ranks", action="store_true", default=True)
    p.add_argument("--eval_sampling_distribution", choices=["uniform", "zipf"], default="uniform")
    p.add_argument("--frequency_feature_layer", type=int, default=1)
    p.add_argument("--head_fraction", type=float, default=0.2)
    p.add_argument("--tail_fraction", type=float, default=0.4)
    p.add_argument("--batch_size", type=int, default=16)

    p.add_argument("--debug_vocab_size", type=int, default=257)
    p.add_argument("--debug_hidden_size", type=int, default=128)
    p.add_argument("--debug_intermediate_size", type=int, default=256)
    p.add_argument("--debug_num_hidden_layers", type=int, default=2)
    p.add_argument("--debug_num_attention_heads", type=int, default=4)
    p.add_argument("--debug_num_key_value_heads", type=int, default=2)
    p.add_argument("--debug_head_dim", type=int, default=32)
    p.add_argument("--debug_max_position_embeddings", type=int, default=256)

    # Minimal arguments expected by apply_moe_overrides.
    p.add_argument("--use_pre_router", action="store_true", default=False)
    p.add_argument("--use_moe", action="store_true", default=False)
    p.add_argument("--moe_head_level", action="store_true", default=False)
    p.add_argument("--router_entropy_floor_loss_weight", type=float, default=0.0)
    p.add_argument("--moe_expert_input_attention_topk", type=int, default=0)
    p.add_argument("--attn_implementation", default="eager")
    p.add_argument("--ground_truth_routing_strategy", default="none")
    p.add_argument("--moe_num_experts_per_tok", type=int, default=1)
    p.add_argument("--ground_truth_routing_mode", default="dispatch")
    p.add_argument("--moe_router_supervision_loss_weight", type=float, default=0.0)
    p.add_argument("--moe_router_input_shape", default="full")
    p.add_argument("--moe_expert_input_shape", default="full")
    p.add_argument("--moe_router_input", default="hidden")
    p.add_argument("--moe_router_input_pos", default="hidden")
    p.add_argument("--moe_expert_input_pos", default="attention_output_residual")
    p.add_argument("--moe_num_unique_experts", type=int, default=4)
    p.add_argument("--moe_intermediate_size", type=int, default=-1)
    p.add_argument("--moe_use_common_expert", action="store_true", default=False)
    p.add_argument("--moe_common_intermediate_size", type=int, default=-1)
    p.add_argument("--moe_router_bias", action="store_true", default=False)
    p.add_argument("--moe_router_type", default="linear")
    p.add_argument("--moe_router_hidden_size", type=int, default=-1)
    p.add_argument("--moe_router_act", default="silu")
    p.add_argument("--moe_no_normalize_topk_prob", action="store_true", default=False)
    p.add_argument("--moe_spectral_band_dims", default=None)
    p.add_argument("--moe_spectral_num_experts_per_band", default=None)
    p.add_argument("--moe_spectral_topk_per_band", default=None)
    p.add_argument("--moe_spectral_intermediate_sizes", default=None)
    p.add_argument("--moe_spectral_update_interval", type=int, default=100)
    p.add_argument("--moe_spectral_warmup_steps", type=int, default=100)
    p.add_argument("--moe_spectral_sample_size", type=int, default=4096)
    p.add_argument("--moe_spectral_basis_momentum", type=float, default=0.0)
    p.add_argument("--moe_spectral_no_include_top_in_router", action="store_true", default=False)
    p.add_argument("--pre_router_input", default="layer_input")
    p.add_argument("--pre_router_controls_attention", action="store_true", default=False)
    p.add_argument("--attention_router_loss_type", default="kl")
    p.add_argument("--attention_router_loss_weight", type=float, default=0.0)
    p.add_argument("--attention_router_rho", type=float, default=0.75)
    p.add_argument("--router_entropy_floor_alpha", type=float, default=0.5)
    p.add_argument("--moe_load_balance_loss_weight", type=float, default=0.0)
    p.add_argument("--moe_router_inhibition_loss_weight", type=float, default=0.0)
    p.add_argument("--moe_router_inhibition_temperature", type=float, default=1.0)
    p.add_argument("--moe_router_supervision_detach_input", action="store_true", default=False)
    p.add_argument("--ground_truth_routing_feature_layer", type=int, default=0)
    p.add_argument("--ground_truth_frequency_estimate_samples", type=int, default=4096)
    return p.parse_args()


def token_bucket_map(args):
    bucket_ids, _ = make_bucket_ids(args, "zipf")
    dataset = make_dataset(args, "uniform", num_samples=args.num_samples)
    token_counts = defaultdict(Counter)
    for idx in range(len(dataset)):
        _, target, _, _ = dataset[idx]
        meta = dataset.get_metadata(idx)["unit_ids_by_layer"][1:]
        feature = meta[:, args.frequency_feature_layer]
        valid = feature.ge(0)
        for tok, feat in zip(target[valid].tolist(), feature[valid].tolist()):
            token_counts[int(tok)][int(bucket_ids[int(feat)].item())] += 1
    token_to_bucket = {}
    for tok in range(args.debug_vocab_size):
        if token_counts[tok]:
            token_to_bucket[tok] = token_counts[tok].most_common(1)[0][0]
        else:
            token_to_bucket[tok] = 1
    return token_to_bucket


def bucket_direction(bucket: int, token_id: int, hidden: int, mode: str, device):
    vec = torch.zeros(hidden, device=device)
    if bucket == 0:
        vec[0] = 1.0
    elif mode == "packed_common":
        vec[0] = 1.0
    elif mode == "packed_negative_common":
        vec[0] = -1.0
    else:
        # Spread middle/tail tokens through residual dimensions.
        residual_dims = max(hidden - 1, 1)
        dim = 1 + (token_id % residual_dims)
        sign = -1.0 if (token_id // max(residual_dims, 1)) % 2 else 1.0
        vec[dim] = sign
    return F.normalize(vec, dim=0)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    apply_debug_model_overrides(config, args)
    apply_moe_overrides(config, args)
    config._attn_implementation = "eager"
    config.attention_stride_pattern = [1 for _ in range(config.num_hidden_layers)]
    config.residual_source_pattern = [-1 for _ in range(config.num_hidden_layers)]
    model = MyQwen3ForCausalLM(config)

    token_to_bucket = token_bucket_map(args)
    embed = model.model.embed_tokens.weight.data
    noise = torch.randn_like(embed) * float(args.init_noise)
    for tok in range(min(embed.shape[0], args.debug_vocab_size)):
        direction = bucket_direction(token_to_bucket[tok], tok, embed.shape[1], args.init_mode, embed.device)
        embed[tok] = float(args.init_scale) * direction + noise[tok]
    if hasattr(model, "lm_head") and model.lm_head.weight.shape == embed.shape:
        model.lm_head.weight.data.copy_(embed)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_path)
    if args.runtime_config_path:
        os.makedirs(os.path.dirname(args.runtime_config_path), exist_ok=True)
        with open(args.runtime_config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "init_mode": args.init_mode,
                    "init_scale": args.init_scale,
                    "init_noise": args.init_noise,
                    "seed": args.seed,
                    "token_bucket_counts": dict(Counter(token_to_bucket.values())),
                },
                f,
                indent=2,
            )
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
