import argparse
import json
import os

import torch
from transformers import AutoConfig

from analyze_moe_variant_selectivity import load_runtime_config
from models import MyQwen3ForCausalLM
from utils import HierarchicalPatternData


def parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def add_args():
    parser = argparse.ArgumentParser(description="Per-layer/head attention mass over synthetic feature units.")
    parser.add_argument("--config_dir", type=str, default="../Qwen3-0.6B")
    parser.add_argument("--ckpt_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="../experiments/feature_attention_heads.json")

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--synthetic_block_size", type=int, default=4)
    parser.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    parser.add_argument("--synthetic_content_token_count", type=int, default=256)
    parser.add_argument("--synthetic_num_units_per_layer", type=int, default=64)
    parser.add_argument("--synthetic_seed", type=int, default=0)
    parser.add_argument("--synthetic_min_token_id", type=int, default=1)
    parser.add_argument("--synthetic_sampling_distribution", choices=["uniform", "zipf"], default="zipf")
    parser.add_argument("--synthetic_zipf_alpha", type=float, default=1.0)
    parser.add_argument("--synthetic_zipf_shuffle_ranks", action="store_true", default=True)
    parser.add_argument("--synthetic_no_zipf_shuffle_ranks", action="store_false", dest="synthetic_zipf_shuffle_ranks")

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


def build_config(args, runtime_config):
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
    config.attention_stride_pattern = runtime_config.get(
        "attention_stride_pattern",
        args.attention_stride_pattern or [1] * config.num_hidden_layers,
    )
    config.residual_source_pattern = runtime_config.get(
        "residual_source_pattern",
        args.residual_source_pattern or [-1] * config.num_hidden_layers,
    )
    config.use_moe = bool(runtime_config.get("use_moe", args.use_moe))
    config.moe_num_unique_experts = int(runtime_config.get("moe_num_unique_experts", args.moe_num_unique_experts))
    config.moe_num_experts_per_tok = int(runtime_config.get("moe_num_experts_per_tok", args.moe_num_experts_per_tok))
    config.moe_intermediate_size = int(runtime_config.get("moe_intermediate_size", args.moe_intermediate_size))
    config.moe_use_common_expert = bool(runtime_config.get("moe_use_common_expert", args.moe_use_common_expert))
    config.moe_common_intermediate_size = int(
        runtime_config.get("moe_common_intermediate_size", args.moe_common_intermediate_size)
    )
    config.moe_router_bias = bool(runtime_config.get("moe_router_bias", args.moe_router_bias))
    config.moe_router_type = str(runtime_config.get("moe_router_type", "linear"))
    config.moe_router_hidden_size = int(runtime_config.get("moe_router_hidden_size", config.hidden_size))
    config.moe_router_act = str(runtime_config.get("moe_router_act", "silu"))
    config.moe_normalize_topk_prob = bool(
        runtime_config.get("moe_normalize_topk_prob", not bool(args.moe_no_normalize_topk_prob))
    )
    config.moe_router_input = str(runtime_config.get("moe_router_input", "hidden"))
    config.moe_head_level = bool(runtime_config.get("moe_head_level", False))
    config.use_pre_router = bool(runtime_config.get("use_pre_router", False))
    config.pre_router_input = str(runtime_config.get("pre_router_input", "layer_input"))
    config.pre_router_controls_attention = bool(runtime_config.get("pre_router_controls_attention", False))
    config.attention_router_loss_type = str(runtime_config.get("attention_router_loss_type", "kl"))
    config.attention_router_loss_weight = float(runtime_config.get("attention_router_loss_weight", 0.0))
    config.attention_router_rho = float(runtime_config.get("attention_router_rho", 0.75))
    config.router_entropy_floor_loss_weight = float(runtime_config.get("router_entropy_floor_loss_weight", 0.0))
    config.router_entropy_floor_alpha = float(runtime_config.get("router_entropy_floor_alpha", 0.5))
    return config


def load_model(args, device):
    runtime_config = load_runtime_config(os.path.dirname(args.ckpt_file))
    model = MyQwen3ForCausalLM(build_config(args, runtime_config)).to(device)
    state = torch.load(args.ckpt_file, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def init_stats(num_layers, num_heads, feature_names):
    stats = {}
    for layer_idx in range(num_layers):
        stats[layer_idx] = {}
        for feature_name in feature_names:
            stats[layer_idx][feature_name] = {
                "include_self_sum": torch.zeros(num_heads, dtype=torch.float64),
                "history_sum": torch.zeros(num_heads, dtype=torch.float64),
                "baseline_include_self_sum": torch.zeros(num_heads, dtype=torch.float64),
                "baseline_history_sum": torch.zeros(num_heads, dtype=torch.float64),
                "include_self_count": 0,
                "history_count": 0,
            }
    return stats


def update_feature_stats(stats, layer_idx, feature_name, attn_layer, metadata_layer):
    # attn_layer: [batch, heads, seq, seq], metadata_layer: [batch, seq]
    batch, heads, seq_len, _ = attn_layer.shape
    device = attn_layer.device
    positions = torch.arange(seq_len, device=device)
    include_self = positions[None, :] <= positions[:, None]
    history = positions[None, :] < positions[:, None]

    same_feature = metadata_layer[:, :, None] == metadata_layer[:, None, :]
    valid_feature = (metadata_layer[:, :, None] >= 0) & (metadata_layer[:, None, :] >= 0)
    include_mask = same_feature & valid_feature & include_self[None, :, :]
    history_mask = same_feature & valid_feature & history[None, :, :]

    include_mass = (attn_layer * include_mask[:, None, :, :].to(attn_layer.dtype)).sum(dim=-1)
    history_mass = (attn_layer * history_mask[:, None, :, :].to(attn_layer.dtype)).sum(dim=-1)

    include_baseline = include_mask.sum(dim=-1).to(torch.float32) / include_self.sum(dim=-1).to(torch.float32)[None, :]
    history_denom = history.sum(dim=-1).clamp_min(1).to(torch.float32)
    history_baseline = history_mask.sum(dim=-1).to(torch.float32) / history_denom[None, :]

    valid_include_rows = include_self.any(dim=-1)[None, :].expand(batch, -1)
    valid_history_rows = history.any(dim=-1)[None, :].expand(batch, -1)

    feature_stats = stats[layer_idx][feature_name]
    feature_stats["include_self_sum"] += include_mass[:, :, valid_include_rows[0]].sum(dim=(0, 2)).cpu().double()
    feature_stats["history_sum"] += history_mass[:, :, valid_history_rows[0]].sum(dim=(0, 2)).cpu().double()
    feature_stats["baseline_include_self_sum"] += (
        include_baseline[:, valid_include_rows[0]].sum(dim=0).sum().cpu().double().repeat(heads)
    )
    feature_stats["baseline_history_sum"] += (
        history_baseline[:, valid_history_rows[0]].sum(dim=0).sum().cpu().double().repeat(heads)
    )
    feature_stats["include_self_count"] += int(valid_include_rows.sum().item())
    feature_stats["history_count"] += int(valid_history_rows.sum().item())


def finalize_stats(stats):
    layers = []
    for layer_idx, layer_stats in stats.items():
        layer_row = {"layer": layer_idx, "features": {}}
        for feature_name, feature_stats in layer_stats.items():
            include_count = max(feature_stats["include_self_count"], 1)
            history_count = max(feature_stats["history_count"], 1)
            include_mass = feature_stats["include_self_sum"] / include_count
            history_mass = feature_stats["history_sum"] / history_count
            include_baseline = feature_stats["baseline_include_self_sum"] / include_count
            history_baseline = feature_stats["baseline_history_sum"] / history_count
            layer_row["features"][feature_name] = {
                "include_self_mass_by_head": [float(x) for x in include_mass.tolist()],
                "history_mass_by_head": [float(x) for x in history_mass.tolist()],
                "include_self_baseline_by_head": [float(x) for x in include_baseline.tolist()],
                "history_baseline_by_head": [float(x) for x in history_baseline.tolist()],
                "include_self_mass_mean": float(include_mass.mean().item()),
                "history_mass_mean": float(history_mass.mean().item()),
                "include_self_baseline_mean": float(include_baseline.mean().item()),
                "history_baseline_mean": float(history_baseline.mean().item()),
            }
        layers.append(layer_row)
    return layers


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
        sampling_distribution=args.synthetic_sampling_distribution,
        zipf_alpha=args.synthetic_zipf_alpha,
        zipf_shuffle_ranks=args.synthetic_zipf_shuffle_ranks,
        return_metadata=True,
    )
    model = load_model(args, device)
    feature_layers = {"local_slot": 0, "higher_unit": 1}
    stats = init_stats(model.config.num_hidden_layers, model.config.num_attention_heads, feature_layers.keys())

    for start in range(0, args.num_samples, args.batch_size):
        batch_items = [dataset[i] for i in range(start, min(start + args.batch_size, args.num_samples))]
        source = torch.stack([item[0] for item in batch_items]).to(device)
        metadata = torch.stack([item[3] for item in batch_items]).to(device)
        outputs = model(source, output_attentions=True, use_cache=False)
        for layer_idx, attn_layer in enumerate(outputs.attentions):
            for feature_name, feature_layer in feature_layers.items():
                if feature_layer < metadata.shape[-1]:
                    update_feature_stats(stats, layer_idx, feature_name, attn_layer.detach(), metadata[:, :, feature_layer])

    summary = {
        "config": vars(args),
        "device": str(device),
        "layers": finalize_stats(stats),
    }
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
