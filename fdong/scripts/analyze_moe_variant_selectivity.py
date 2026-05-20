import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from models import MyQwen3ForCausalLM
from utils import HierarchicalPatternData


DEFAULT_RUNS = ",".join(
    [
        "inverse-kv-local-h128-l3-top1",
        "inverse-kv-attn-output-router",
        "inverse-kv-head-moe-hidden-router",
        "inverse-kv-attn-output-head-moe",
    ]
)


def parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def add_args():
    parser = argparse.ArgumentParser(description="Compare MoE selectivity and attention alignment across variants.")
    parser.add_argument("--config_dir", type=str, default="../Qwen3-0.6B")
    parser.add_argument("--checkpoint_root", type=str, default="../checkpoints")
    parser.add_argument("--runs", type=str, default=DEFAULT_RUNS)
    parser.add_argument("--checkpoint_step", type=int, default=5000)
    parser.add_argument("--output_path", type=str, default="../experiments/moe_variant_selectivity.json")

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--synthetic_block_size", type=int, default=4)
    parser.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    parser.add_argument("--synthetic_content_token_count", type=int, default=256)
    parser.add_argument("--synthetic_num_units_per_layer", type=int, default=64)
    parser.add_argument("--synthetic_seed", type=int, default=0)
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
    parser.add_argument("--modes", type=str, default="full,same_slot_occurrence,same_slot,same_higher,random_same_size")
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


def legacy_run_dir(run_dir: str) -> str:
    parent, name = os.path.split(run_dir)
    if name.startswith("ground-truth"):
        legacy_name = ("or" + "acle") + name[len("ground-truth") :]
        return os.path.join(parent, legacy_name)
    return run_dir


def load_runtime_config(run_dir: str) -> Dict:
    if not os.path.exists(run_dir):
        run_dir = legacy_run_dir(run_dir)
    path = os.path.join(run_dir, "runtime_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def runtime_get(runtime_config: Dict, key: str, default=None):
    if key in runtime_config:
        return runtime_config[key]
    legacy_key = key.replace("ground_truth", "or" + "acle")
    return runtime_config.get(legacy_key, default)


def find_checkpoint(run_dir: str, preferred_step: int) -> Tuple[Optional[str], Optional[int]]:
    candidates = []
    if not os.path.isdir(run_dir):
        run_dir = legacy_run_dir(run_dir)
    if not os.path.isdir(run_dir):
        return None, None
    for name in os.listdir(run_dir):
        if not name.endswith(".pth"):
            continue
        step_text = name[:-4]
        if not step_text.isdigit():
            continue
        step = int(step_text)
        if step <= preferred_step:
            candidates.append((step, os.path.join(run_dir, name)))
    if not candidates:
        return None, None
    step, path = max(candidates, key=lambda item: item[0])
    return path, step


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
    config.use_moe = bool(runtime_config.get("use_moe", True))
    config.moe_num_unique_experts = int(runtime_config.get("moe_num_unique_experts", 4))
    config.moe_num_experts_per_tok = int(runtime_config.get("moe_num_experts_per_tok", 1))
    config.moe_intermediate_size = int(runtime_config.get("moe_intermediate_size", 128))
    config.moe_use_common_expert = bool(runtime_config.get("moe_use_common_expert", False))
    config.moe_common_intermediate_size = int(runtime_config.get("moe_common_intermediate_size", 128))
    config.moe_router_bias = bool(runtime_config.get("moe_router_bias", False))
    config.moe_router_type = str(runtime_config.get("moe_router_type", "linear"))
    config.moe_router_hidden_size = int(runtime_config.get("moe_router_hidden_size", config.hidden_size))
    config.moe_router_act = str(runtime_config.get("moe_router_act", "silu"))
    config.moe_normalize_topk_prob = bool(runtime_config.get("moe_normalize_topk_prob", True))
    config.moe_router_input = str(runtime_config.get("moe_router_input", "hidden"))
    config.moe_head_level = bool(runtime_config.get("moe_head_level", False))
    config.ground_truth_routing_mode = str(runtime_get(runtime_config, "ground_truth_routing_mode", "dispatch"))
    config.moe_router_supervision_loss_weight = float(runtime_config.get("moe_router_supervision_loss_weight", 0.0))
    config.ground_truth_routing_strategy = str(runtime_get(runtime_config, "ground_truth_routing_strategy", "none"))
    config.ground_truth_routing_feature_layer = int(runtime_get(runtime_config, "ground_truth_routing_feature_layer", 0))
    config.ground_truth_frequency_estimate_samples = int(runtime_get(runtime_config, "ground_truth_frequency_estimate_samples", 0))
    return config


def load_model(args, runtime_config, ckpt_path, device):
    model = MyQwen3ForCausalLM(build_config(args, runtime_config)).to(device)
    state = torch.load(ckpt_path, map_location=device)
    # Older linear-router checkpoints were saved before MyQwen3Router wrapped the
    # projection as `router.net`. Remap those keys for analysis compatibility.
    if any(".mlp.router.weight" in key for key in state):
        remapped = {}
        for key, value in state.items():
            key = key.replace(".mlp.router.weight", ".mlp.router.net.weight")
            key = key.replace(".mlp.router.bias", ".mlp.router.net.bias")
            remapped[key] = value
        state = remapped
    model.load_state_dict(state)
    model.eval()
    return model


def entropy_from_counts(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [count / total for count in counts if count > 0]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def mutual_information(x, y):
    pairs = list(zip(x, y))
    total = len(pairs)
    if total == 0:
        return 0.0
    x_counts = Counter(x)
    y_counts = Counter(y)
    xy_counts = Counter(pairs)
    mi = 0.0
    for (xi, yi), count in xy_counts.items():
        pxy = count / total
        px = x_counts[xi] / total
        py = y_counts[yi] / total
        mi += pxy * math.log((pxy + 1e-12) / (px * py + 1e-12))
    return float(mi)


def mapping_purity(source, target):
    grouped = {}
    for s, t in zip(source, target):
        grouped.setdefault(int(s), Counter())[int(t)] += 1
    total = sum(sum(counter.values()) for counter in grouped.values())
    if total == 0:
        return 0.0
    return float(sum(max(counter.values()) for counter in grouped.values()) / total)


def pairwise_same_bucket_given_same_feature(bucket_ids, feature_ids):
    same_feature = feature_ids[:, :, None] == feature_ids[:, None, :]
    valid = same_feature & (feature_ids[:, :, None] >= 0) & (feature_ids[:, None, :] >= 0)
    not_self = ~torch.eye(feature_ids.size(1), dtype=torch.bool, device=feature_ids.device)[None, :, :]
    valid = valid & not_self
    same_bucket = bucket_ids[:, :, None] == bucket_ids[:, None, :]
    return float((same_bucket & valid).float().sum().item() / valid.sum().clamp_min(1).item())


def attention_feature_metrics(attn, feature_ids):
    # attn: [batch, heads, seq, seq], feature_ids: [batch, seq]
    batch, heads, seq_len, _ = attn.shape
    device = attn.device
    positions = torch.arange(seq_len, device=device)
    include_self = positions[None, :] <= positions[:, None]
    history = positions[None, :] < positions[:, None]
    same_feature = feature_ids[:, :, None] == feature_ids[:, None, :]
    valid_feature = (feature_ids[:, :, None] >= 0) & (feature_ids[:, None, :] >= 0)
    same_include = same_feature & valid_feature & include_self[None, :, :]
    diff_include = (~same_feature) & valid_feature & include_self[None, :, :]
    same_history = same_feature & valid_feature & history[None, :, :]
    diff_history = (~same_feature) & valid_feature & history[None, :, :]

    same_include_mask = same_include[:, None, :, :]
    diff_include_mask = diff_include[:, None, :, :]
    same_history_mask = same_history[:, None, :, :]
    diff_history_mask = diff_history[:, None, :, :]

    same_include_mass = (attn * same_include_mask.to(attn.dtype)).sum(dim=-1)
    same_history_mass = (attn * same_history_mask.to(attn.dtype)).sum(dim=-1)
    include_mass = (attn * include_self[None, None, :, :].to(attn.dtype)).sum(dim=-1).clamp_min(1e-12)
    history_mass = (attn * history[None, None, :, :].to(attn.dtype)).sum(dim=-1).clamp_min(1e-12)

    same_include_mass_per_head = (same_include_mass / include_mass).mean(dim=(0, 2))
    same_history_mass_per_head = (same_history_mass / history_mass).mean(dim=(0, 2))
    same_include_mean = attn.masked_select(same_include_mask).mean().item() if same_include_mask.any() else 0.0
    diff_include_mean = attn.masked_select(diff_include_mask).mean().item() if diff_include_mask.any() else 0.0
    same_history_mean = attn.masked_select(same_history_mask).mean().item() if same_history_mask.any() else 0.0
    diff_history_mean = attn.masked_select(diff_history_mask).mean().item() if diff_history_mask.any() else 0.0
    return {
        "include_self_mass_by_head": [float(x) for x in same_include_mass_per_head.cpu().tolist()],
        "include_self_mass_mean": float(same_include_mass_per_head.mean().item()),
        "include_self_pair_score_same": float(same_include_mean),
        "include_self_pair_score_diff": float(diff_include_mean),
        "include_self_pair_score_lift": float(same_include_mean / max(diff_include_mean, 1e-12)),
        "history_mass_by_head": [float(x) for x in same_history_mass_per_head.cpu().tolist()],
        "history_mass_mean": float(same_history_mass_per_head.mean().item()),
        "history_pair_score_same": float(same_history_mean),
        "history_pair_score_diff": float(diff_history_mean),
        "history_pair_score_lift": float(same_history_mean / max(diff_history_mean, 1e-12)),
    }


def attention_expert_alignment(attn, expert_labels):
    # expert_labels: [batch, seq] for token MoE, or [batch, seq, heads] for head-level MoE.
    batch, heads, seq_len, _ = attn.shape
    device = attn.device
    include_self = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=0)
    history = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=-1)
    if expert_labels.dim() == 2:
        same_expert = expert_labels[:, :, None] == expert_labels[:, None, :]
        same_expert = same_expert[:, None, :, :].expand(-1, heads, -1, -1)
    else:
        same_expert = expert_labels.permute(0, 2, 1)
        same_expert = same_expert[:, :, :, None] == same_expert[:, :, None, :]
    same_include_mask = same_expert & include_self[None, None, :, :]
    diff_include_mask = (~same_expert) & include_self[None, None, :, :]
    same_history_mask = same_expert & history[None, None, :, :]
    diff_history_mask = (~same_expert) & history[None, None, :, :]
    same_include_mass = (attn * same_include_mask.to(attn.dtype)).sum(dim=-1)
    same_history_mass = (attn * same_history_mask.to(attn.dtype)).sum(dim=-1)
    include_mass = (attn * include_self[None, None, :, :].to(attn.dtype)).sum(dim=-1).clamp_min(1e-12)
    history_mass = (attn * history[None, None, :, :].to(attn.dtype)).sum(dim=-1).clamp_min(1e-12)
    include_mass_by_head = (same_include_mass / include_mass).mean(dim=(0, 2))
    history_mass_by_head = (same_history_mass / history_mass).mean(dim=(0, 2))
    same_include_mean = attn.masked_select(same_include_mask).mean().item() if same_include_mask.any() else 0.0
    diff_include_mean = attn.masked_select(diff_include_mask).mean().item() if diff_include_mask.any() else 0.0
    same_history_mean = attn.masked_select(same_history_mask).mean().item() if same_history_mask.any() else 0.0
    diff_history_mean = attn.masked_select(diff_history_mask).mean().item() if diff_history_mask.any() else 0.0
    return {
        "same_expert_include_self_mass_by_head": [float(x) for x in include_mass_by_head.cpu().tolist()],
        "same_expert_include_self_mass_mean": float(include_mass_by_head.mean().item()),
        "same_expert_include_self_pair_score": float(same_include_mean),
        "diff_expert_include_self_pair_score": float(diff_include_mean),
        "same_expert_include_self_pair_score_lift": float(same_include_mean / max(diff_include_mean, 1e-12)),
        "same_expert_history_mass_by_head": [float(x) for x in history_mass_by_head.cpu().tolist()],
        "same_expert_history_mass_mean": float(history_mass_by_head.mean().item()),
        "same_expert_history_pair_score": float(same_history_mean),
        "diff_expert_history_pair_score": float(diff_history_mean),
        "same_expert_history_pair_score_lift": float(same_history_mean / max(diff_history_mean, 1e-12)),
    }


def build_allowed_mask(mode, metadata, random_seed, block_size):
    batch, seq_len, _ = metadata.shape
    device = metadata.device
    positions = torch.arange(seq_len, device=device)
    causal = positions[None, :] <= positions[:, None]
    causal = causal[None, :, :].expand(batch, -1, -1)
    if mode == "full":
        return causal
    if mode == "same_slot_occurrence":
        occurrence_ids = positions // int(block_size)
        return (occurrence_ids[:, None] == occurrence_ids[None, :])[None, :, :].expand(batch, -1, -1) & causal
    slot_ids = metadata[:, :, 0]
    same_slot = slot_ids[:, :, None] == slot_ids[:, None, :]
    same_slot = same_slot & causal & (slot_ids[:, :, None] >= 0) & (slot_ids[:, None, :] >= 0)
    if mode == "same_slot":
        return same_slot
    higher_ids = metadata[:, :, 1]
    same_higher = higher_ids[:, :, None] == higher_ids[:, None, :]
    same_higher = same_higher & causal & (higher_ids[:, :, None] >= 0) & (higher_ids[:, None, :] >= 0)
    if mode == "same_higher":
        return same_higher
    if mode == "same_slot_or_higher":
        return same_slot | same_higher
    if mode != "random_same_size":
        raise ValueError(f"Unknown mode: {mode}")
    random_allowed = torch.zeros_like(same_slot)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(random_seed))
    for b in range(batch):
        for i in range(seq_len):
            causal_indices = torch.nonzero(causal[b, i], as_tuple=False).flatten()
            keep_count = int(same_slot[b, i].sum().item())
            if keep_count <= 0:
                continue
            random_allowed[b, i, i] = True
            candidates = causal_indices[causal_indices != i]
            remaining = keep_count - 1
            if remaining > 0 and candidates.numel() > 0:
                perm = torch.randperm(candidates.numel(), generator=generator, device=device)
                random_allowed[b, i, candidates[perm[: min(remaining, candidates.numel())]]] = True
    return random_allowed


def additive_mask_from_allowed(allowed, dtype):
    min_dtype = torch.finfo(dtype).min
    mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    return mask.masked_fill(~allowed, min_dtype)[:, None, :, :]


def init_sparse_states(modes):
    return {mode: {"loss_sum": 0.0, "correct": 0, "count": 0, "visible_sum": 0.0} for mode in modes}


def update_sparse_state(state, logits, target, allowed):
    valid = target != 0
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none").reshape_as(target)
    pred = logits.argmax(dim=-1)
    state["loss_sum"] += float(loss[valid].sum().item())
    state["correct"] += int(((pred == target) & valid).sum().item())
    state["count"] += int(valid.sum().item())
    seq_len = target.shape[1]
    causal_count = torch.arange(1, seq_len + 1, device=target.device, dtype=torch.float32)[None, :]
    visible = allowed.sum(dim=-1).to(torch.float32) / causal_count
    state["visible_sum"] += float(visible[valid].sum().item())


def finalize_sparse_states(states):
    result = {}
    for mode, state in states.items():
        count = max(state["count"], 1)
        result[mode] = {
            "loss": state["loss_sum"] / count,
            "accuracy": state["correct"] / count,
            "visible_fraction": state["visible_sum"] / count,
            "count": state["count"],
        }
    return result


def init_accumulators(config):
    return {
        "loss_sum": 0.0,
        "correct": 0,
        "count": 0,
        "feature_ids": {"local_slot": [], "higher_unit": []},
        "experts": [[] for _ in range(config.num_hidden_layers)],
        "attention_features": [[] for _ in range(config.num_hidden_layers)],
        "attention_expert": [[] for _ in range(config.num_hidden_layers)],
    }


def primary_expert_labels(expert_labels):
    # token MoE: [B,S,K] -> [B,S]; head MoE: [B,S,H,K] -> [B,S,H]
    return expert_labels[..., 0]


def build_ground_truth_expert_mapping(dataset, args, runtime_config):
    strategy = str(runtime_get(runtime_config, "ground_truth_routing_strategy", "none"))
    if strategy == "none":
        return None
    feature_layer = int(runtime_get(runtime_config, "ground_truth_routing_feature_layer", 0))
    num_features = int(args.synthetic_num_units_per_layer)
    num_experts = int(runtime_config.get("moe_num_unique_experts", 4))
    if strategy == "hash":
        return torch.remainder(torch.arange(num_features, dtype=torch.long), num_experts), feature_layer
    if strategy != "frequency_balanced":
        raise ValueError(f"Unsupported ground-truth routing strategy: {strategy}")

    estimate_samples = min(int(runtime_get(runtime_config, "ground_truth_frequency_estimate_samples", 4096)), len(dataset))
    requested_estimate_samples = int(runtime_get(runtime_config, "ground_truth_frequency_estimate_samples", 4096))
    if requested_estimate_samples > len(dataset):
        mapping_dataset = HierarchicalPatternData(
            max_seq_len=args.seq_len,
            num_samples=requested_estimate_samples,
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
        estimate_samples = requested_estimate_samples
    else:
        mapping_dataset = dataset

    counts = torch.zeros(num_features, dtype=torch.float64)
    for idx in range(estimate_samples):
        metadata = mapping_dataset.get_metadata(idx)["unit_ids_by_layer"][:-1, feature_layer]
        valid = metadata >= 0
        if valid.any():
            counts += torch.bincount(metadata[valid], minlength=num_features).to(counts.dtype)

    expert_loads = torch.zeros(num_experts, dtype=torch.float64)
    mapping = torch.empty(num_features, dtype=torch.long)
    for feature_id in torch.argsort(counts, descending=True).tolist():
        expert_id = int(torch.argmin(expert_loads).item())
        mapping[feature_id] = expert_id
        expert_loads[expert_id] += counts[feature_id]
    return mapping, feature_layer


def make_ground_truth_expert_ids(metadata, ground_truth_mapping_and_layer, device):
    if ground_truth_mapping_and_layer is None:
        return None
    mapping, feature_layer = ground_truth_mapping_and_layer
    feature_ids = metadata[:, :, feature_layer].to(device)
    mapping = mapping.to(device)
    safe_feature_ids = feature_ids.clamp_min(0)
    ground_truth_expert_ids = mapping[safe_feature_ids]
    return ground_truth_expert_ids.masked_fill(feature_ids < 0, 0)


@torch.no_grad()
def analyze_run(args, run_name, dataset, device):
    run_dir = os.path.join(args.checkpoint_root, run_name)
    runtime_config = load_runtime_config(run_dir)
    ckpt_path, step = find_checkpoint(run_dir, args.checkpoint_step)
    if ckpt_path is None:
        return {"run": run_name, "error": "no checkpoint found", "checkpoint_dir": run_dir}

    model = load_model(args, runtime_config, ckpt_path, device)
    dtype = next(model.parameters()).dtype
    ground_truth_mapping_and_layer = build_ground_truth_expert_mapping(dataset, args, runtime_config)
    use_ground_truth_dispatch = str(runtime_get(runtime_config, "ground_truth_routing_mode", "dispatch")) == "dispatch"
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    sparse_states = init_sparse_states(modes)
    acc = init_accumulators(model.config)

    for start in range(0, args.num_samples, args.batch_size):
        batch_items = [dataset[i] for i in range(start, min(start + args.batch_size, args.num_samples))]
        source = torch.stack([item[0] for item in batch_items]).to(device)
        target = torch.stack([item[1] for item in batch_items]).to(device)
        metadata = torch.stack([item[3] for item in batch_items]).to(device)
        ground_truth_expert_ids = make_ground_truth_expert_ids(metadata, ground_truth_mapping_and_layer, device)
        dispatch_ground_truth_expert_ids = ground_truth_expert_ids if use_ground_truth_dispatch else None

        outputs = model(
            source,
            output_attentions=True,
            output_expert_labels=True,
            use_cache=False,
            ground_truth_expert_ids=dispatch_ground_truth_expert_ids,
        )
        loss = F.cross_entropy(outputs.logits.reshape(-1, outputs.logits.size(-1)), target.reshape(-1), reduction="none")
        loss = loss.reshape_as(target)
        pred = outputs.logits.argmax(dim=-1)
        valid = target != 0
        acc["loss_sum"] += float(loss[valid].sum().item())
        acc["correct"] += int(((pred == target) & valid).sum().item())
        acc["count"] += int(valid.sum().item())
        acc["feature_ids"]["local_slot"].append(metadata[:, :, 0].cpu())
        acc["feature_ids"]["higher_unit"].append(metadata[:, :, 1].cpu())

        for layer_idx, labels in enumerate(outputs.expert_labels):
            primary = primary_expert_labels(labels).detach()
            acc["experts"][layer_idx].append(primary.cpu())
            layer_attention = outputs.attentions[layer_idx].detach()
            feature_rows = {}
            for feature_name, feature_layer in (("local_slot", 0), ("higher_unit", 1)):
                feature_rows[feature_name] = attention_feature_metrics(layer_attention, metadata[:, :, feature_layer])
            acc["attention_features"][layer_idx].append(feature_rows)
            acc["attention_expert"][layer_idx].append(attention_expert_alignment(layer_attention, primary))

        for mode in modes:
            allowed = build_allowed_mask(mode, metadata, args.random_seed + start, args.synthetic_block_size)
            outputs_sparse = model(
                source,
                attention_mask=additive_mask_from_allowed(allowed, dtype),
                output_attentions=False,
                output_expert_labels=False,
                use_cache=False,
                ground_truth_expert_ids=dispatch_ground_truth_expert_ids,
            )
            update_sparse_state(sparse_states[mode], outputs_sparse.logits, target, allowed)

    features_all = {
        name: torch.cat(chunks, dim=0)
        for name, chunks in acc["feature_ids"].items()
    }
    layers = []
    for layer_idx, expert_chunks in enumerate(acc["experts"]):
        expert_tensor = torch.cat(expert_chunks, dim=0)
        layer_row = {
            "layer": layer_idx,
            "expert_shape": list(expert_tensor.shape),
            "feature_selectivity": {},
            "attention_feature_metrics": {},
            "attention_expert_alignment": {},
        }
        if expert_tensor.dim() == 2:
            flat_experts = expert_tensor.reshape(-1).tolist()
            layer_row["expert_load"] = dict(Counter(flat_experts))
            layer_row["expert_entropy"] = entropy_from_counts(Counter(flat_experts).values())
            for feature_name, feature_tensor in features_all.items():
                flat_feature = feature_tensor.reshape(-1).tolist()
                layer_row["feature_selectivity"][feature_name] = {
                    "feature_expert_mi": mutual_information(flat_feature, flat_experts),
                    "feature_to_expert_purity": mapping_purity(flat_feature, flat_experts),
                    "expert_to_feature_purity": mapping_purity(flat_experts, flat_feature),
                    "same_feature_same_expert_rate": pairwise_same_bucket_given_same_feature(
                        expert_tensor, feature_tensor
                    ),
                }
        else:
            num_heads = expert_tensor.shape[2]
            for feature_name, feature_tensor in features_all.items():
                per_head = []
                for head_idx in range(num_heads):
                    head_expert = expert_tensor[:, :, head_idx]
                    flat_feature = feature_tensor.reshape(-1).tolist()
                    flat_expert = head_expert.reshape(-1).tolist()
                    per_head.append(
                        {
                            "head": head_idx,
                            "feature_expert_mi": mutual_information(flat_feature, flat_expert),
                            "feature_to_expert_purity": mapping_purity(flat_feature, flat_expert),
                            "expert_to_feature_purity": mapping_purity(flat_expert, flat_feature),
                            "same_feature_same_expert_rate": pairwise_same_bucket_given_same_feature(
                                head_expert, feature_tensor
                            ),
                            "expert_load": dict(Counter(flat_expert)),
                        }
                    )
                combined_expert = expert_tensor.reshape(expert_tensor.shape[0], expert_tensor.shape[1], -1)
                layer_row["feature_selectivity"][feature_name] = {
                    "per_head": per_head,
                    "mean_feature_expert_mi": float(sum(row["feature_expert_mi"] for row in per_head) / len(per_head)),
                    "mean_same_feature_same_expert_rate": float(
                        sum(row["same_feature_same_expert_rate"] for row in per_head) / len(per_head)
                    ),
                    "combined_head_expert_note": "combined_head_expert treats each head expert as a separate bucket",
                }

        for feature_name in ("local_slot", "higher_unit"):
            rows = [row[feature_name] for row in acc["attention_features"][layer_idx]]
            keys = rows[0].keys()
            layer_row["attention_feature_metrics"][feature_name] = {
                key: (
                    [float(sum(row[key][i] for row in rows) / len(rows)) for i in range(len(rows[0][key]))]
                    if isinstance(rows[0][key], list)
                    else float(sum(row[key] for row in rows) / len(rows))
                )
                for key in keys
            }
        rows = acc["attention_expert"][layer_idx]
        layer_row["attention_expert_alignment"] = {
            key: (
                [float(sum(row[key][i] for row in rows) / len(rows)) for i in range(len(rows[0][key]))]
                if isinstance(rows[0][key], list)
                else float(sum(row[key] for row in rows) / len(rows))
            )
            for key in rows[0].keys()
        }
        layers.append(layer_row)

    return {
        "run": run_name,
        "checkpoint_step": step,
        "checkpoint_path": ckpt_path,
        "runtime_config": runtime_config,
        "loss": acc["loss_sum"] / max(acc["count"], 1),
        "accuracy": acc["correct"] / max(acc["count"], 1),
        "sparse_attention_inference": finalize_sparse_states(sparse_states),
        "layers": layers,
    }


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
    runs = [run.strip() for run in args.runs.split(",") if run.strip()]
    summary = {
        "config": vars(args),
        "device": str(device),
        "runs": [analyze_run(args, run, dataset, device) for run in runs],
    }
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
