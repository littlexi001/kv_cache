import argparse
import copy
import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from analyze_moe_variant_selectivity import (
    attention_expert_alignment,
    attention_feature_metrics,
    find_checkpoint,
    load_model,
    load_runtime_config,
    mapping_purity,
    mutual_information,
    pairwise_same_bucket_given_same_feature,
)
from utils import ControlledReusedTokenData


DEFAULT_RUNS = [
    "round6-controlled-dense-h32",
    "round6-controlled-dense-h64",
    "round6-controlled-dense-h128",
    "round6-controlled-moe-hidden-full-top1",
    "round6-controlled-moe-hidden-full-top2",
    "round6-controlled-moe-hidden-full-common",
    "round6-controlled-moe-hidden-full-lb001",
    "round6-controlled-moe-rfull-attn-eresid",
    "round6-controlled-moe-rfull-q-eresid",
    "round6-controlled-moe-rfull-k-eresid",
    "round6-controlled-moe-rfull-v-eresid",
    "round6-controlled-moe-rfull-layerin-eresid",
    "round6-controlled-moe-rfull-hidden-eresid",
    "round6-controlled-moe-rhead-attn-eresid",
    "round6-controlled-moe-rhead-q-eresid",
    "round6-controlled-moe-rhead-k-eresid",
    "round6-controlled-moe-rhead-v-eresid",
    "round6-controlled-moe-rhead-layerin-eresid",
    "round6-controlled-moe-rhead-hidden-eresid",
    "round6-controlled-moe-k-head-ne4-topk1",
    "round6-controlled-moe-k-head-ne4-topk2",
    "round6-controlled-moe-k-head-ne8-topk1",
    "round6-controlled-moe-k-head-ne8-topk2",
    "round6-controlled-moe-k-head-common",
    "round6-controlled-moe-k-head-lb001",
    "round6-controlled-moe-k-head-lb01",
    "round6-controlled-moe-k-head-eattn",
    "round6-controlled-moe-k-head-ehidden",
    "round6-controlled-moe-k-head-elayerin",
    "round6-controlled-moe-k-head-eq",
    "round6-controlled-moe-k-head-ek",
    "round6-controlled-moe-k-head-ev",
    "round6-controlled-moe-headhead-k-eresid",
    "round6-controlled-moe-headhead-k-eattn",
    "round6-controlled-moe-headhead-k-eq",
    "round6-controlled-moe-headhead-k-ek",
    "round6-controlled-moe-headhead-k-ev",
    "round6-controlled-moe-headhead-k-elayerin",
    "round6-controlled-moe-headhead-k-ehidden",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Round6 controlled reused-token experiments.")
    parser.add_argument("--config_dir", type=str, default="../Qwen3-0.6B")
    parser.add_argument("--checkpoint_root", type=str, default="../checkpoints")
    parser.add_argument("--log_dir", type=str, default="../logs")
    parser.add_argument("--runs", type=str, default=",".join(DEFAULT_RUNS))
    parser.add_argument("--checkpoint_step", type=int, default=2000)
    parser.add_argument("--output_json", type=str, default="../experiments/round6_controlled_reuse_analysis.json")
    parser.add_argument("--output_md", type=str, default="../experiments/round6_controlled_reuse_analysis.md")

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--synthetic_block_size", type=int, default=4)
    parser.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    parser.add_argument("--synthetic_content_token_count", type=int, default=512)
    parser.add_argument("--synthetic_num_units_per_layer", type=int, default=512)
    parser.add_argument("--synthetic_seed", type=int, default=0)
    parser.add_argument("--synthetic_min_token_id", type=int, default=1)
    parser.add_argument("--controlled_same_input_diff_output_rate", type=float, default=0.1)
    parser.add_argument("--controlled_same_input_diff_output_size", type=int, default=4)
    parser.add_argument("--controlled_same_input_diff_output_distribution", choices=["uniform", "zipf"], default="zipf")
    parser.add_argument("--controlled_same_input_diff_output_zipf_alpha", type=float, default=1.0)
    parser.add_argument("--controlled_diff_input_same_output_rate", type=float, default=0.1)
    parser.add_argument("--controlled_diff_input_same_output_size", type=int, default=4)
    parser.add_argument("--controlled_diff_input_same_output_distribution", choices=["uniform", "zipf"], default="zipf")
    parser.add_argument("--controlled_diff_input_same_output_zipf_alpha", type=float, default=1.0)
    parser.add_argument("--controlled_top_sampling_distribution", choices=["uniform", "zipf"], default="uniform")
    parser.add_argument("--controlled_top_sampling_zipf_alpha", type=float, default=1.0)

    parser.add_argument("--debug_vocab_size", type=int, default=513)
    parser.add_argument("--debug_hidden_size", type=int, default=64)
    parser.add_argument("--debug_intermediate_size", type=int, default=128)
    parser.add_argument("--debug_num_hidden_layers", type=int, default=2)
    parser.add_argument("--debug_num_attention_heads", type=int, default=4)
    parser.add_argument("--debug_num_key_value_heads", type=int, default=2)
    parser.add_argument("--debug_head_dim", type=int, default=16)
    parser.add_argument("--debug_max_position_embeddings", type=int, default=256)
    parser.add_argument("--attention_stride_pattern", type=str, default=None)
    parser.add_argument("--residual_source_pattern", type=str, default=None)
    return parser.parse_args()


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataset(args):
    return ControlledReusedTokenData(
        max_seq_len=args.seq_len,
        num_samples=args.num_samples,
        slot_size=args.synthetic_block_size,
        num_hierarchy_layers=args.synthetic_num_hierarchy_layers,
        content_token_count=args.synthetic_content_token_count,
        num_units_per_layer=args.synthetic_num_units_per_layer,
        seed=args.synthetic_seed,
        min_token_id=args.synthetic_min_token_id,
        same_input_diff_output_rate=args.controlled_same_input_diff_output_rate,
        same_input_diff_output_size=args.controlled_same_input_diff_output_size,
        same_input_diff_output_distribution=args.controlled_same_input_diff_output_distribution,
        same_input_diff_output_zipf_alpha=args.controlled_same_input_diff_output_zipf_alpha,
        diff_input_same_output_rate=args.controlled_diff_input_same_output_rate,
        diff_input_same_output_size=args.controlled_diff_input_same_output_size,
        diff_input_same_output_distribution=args.controlled_diff_input_same_output_distribution,
        diff_input_same_output_zipf_alpha=args.controlled_diff_input_same_output_zipf_alpha,
        top_sampling_distribution=args.controlled_top_sampling_distribution,
        top_sampling_zipf_alpha=args.controlled_top_sampling_zipf_alpha,
        return_metadata=True,
    )


def infer_debug_shape_from_checkpoint(args, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    inferred = copy.copy(args)

    embed = state.get("model.embed_tokens.weight")
    if embed is not None:
        inferred.debug_vocab_size = int(embed.shape[0])
        inferred.debug_hidden_size = int(embed.shape[1])

    layer_indices = []
    for key in state:
        match = re.match(r"model\.layers\.(\d+)\.", key)
        if match:
            layer_indices.append(int(match.group(1)))
    if layer_indices:
        inferred.debug_num_hidden_layers = max(layer_indices) + 1

    q_proj = state.get("model.layers.0.self_attn.q_proj.weight")
    k_proj = state.get("model.layers.0.self_attn.k_proj.weight")
    if q_proj is not None and k_proj is not None:
        hidden = int(q_proj.shape[1])
        inferred.debug_hidden_size = hidden
        kv_out = int(k_proj.shape[0])
        num_kv_heads = int(args.debug_num_key_value_heads)
        if kv_out % num_kv_heads == 0:
            inferred.debug_head_dim = kv_out // num_kv_heads
            if inferred.debug_head_dim > 0 and int(q_proj.shape[0]) % inferred.debug_head_dim == 0:
                inferred.debug_num_attention_heads = int(q_proj.shape[0]) // inferred.debug_head_dim

    dense_gate = state.get("model.layers.0.mlp.gate_proj.weight")
    if dense_gate is not None:
        inferred.debug_intermediate_size = int(dense_gate.shape[0])

    return inferred


def new_counter():
    return {"loss_sum": 0.0, "correct": 0, "count": 0}


def update_counter(counter, losses, pred, target, mask):
    if mask.sum().item() == 0:
        return
    counter["loss_sum"] += float(losses[mask].sum().item())
    counter["correct"] += int(((pred == target) & mask).sum().item())
    counter["count"] += int(mask.sum().item())


def finish_counter(counter):
    count = max(counter["count"], 1)
    return {
        "loss": counter["loss_sum"] / count,
        "accuracy": counter["correct"] / count,
        "count": counter["count"],
    }


def entropy_from_counter(counter: Counter):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log((count / total) + 1e-12) for count in counter.values() if count > 0)


def effective_experts(labels: torch.Tensor) -> float:
    values = [int(x) for x in labels.reshape(-1).tolist() if int(x) >= 0]
    if not values:
        return 0.0
    return float(math.exp(entropy_from_counter(Counter(values))))


def normalize_expert_labels(primary: torch.Tensor, runtime_config: Dict, num_heads: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
    router_shape = str(runtime_config.get("moe_router_input_shape", "full"))
    if primary.dim() == 2:
        return "token", primary, primary
    if primary.dim() != 3:
        raise ValueError(f"Unsupported expert label shape: {tuple(primary.shape)}")
    if router_shape == "head" and primary.shape[-1] == num_heads:
        return "head", primary, primary.reshape(primary.shape[0], primary.shape[1], -1)
    if router_shape == "spectral":
        labels = primary.clone()
        combined = torch.zeros(labels.shape[:2], dtype=torch.long)
        multiplier = 1
        for idx in range(labels.shape[-1]):
            value = labels[:, :, idx].clamp_min(0).long()
            combined = combined + value * multiplier
            max_value = int(value.max().item()) if value.numel() else 0
            multiplier *= max(max_value + 1, 1)
        return "spectral", labels, combined
    return "multi", primary, primary.reshape(primary.shape[0], primary.shape[1], -1)


def flatten_masked(feature: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
    if mask is None:
        mask = feature >= 0
    else:
        mask = mask & (feature >= 0)
    return feature[mask].reshape(-1).tolist(), labels[mask].reshape(-1).tolist()


def masked_pairwise_same(labels: torch.Tensor, feature: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    masked_feature = feature.clone()
    if mask is not None:
        masked_feature = masked_feature.masked_fill(~mask, -1)
    if labels.dim() == 2:
        return pairwise_same_bucket_given_same_feature(labels, masked_feature)
    rates = []
    for idx in range(labels.shape[-1]):
        rates.append(pairwise_same_bucket_given_same_feature(labels[:, :, idx], masked_feature))
    return float(sum(rates) / len(rates)) if rates else 0.0


def feature_selectivity(labels: torch.Tensor, combined: torch.Tensor, feature: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict:
    source, target = flatten_masked(feature, combined, mask)
    if mask is None:
        effective_label_tensor = combined
    elif combined.dim() == mask.dim():
        effective_label_tensor = combined.masked_fill(~mask, -1)
    else:
        effective_label_tensor = combined.masked_fill(~mask.unsqueeze(-1), -1)
    return {
        "feature_expert_mi": mutual_information(source, target),
        "feature_to_expert_purity": mapping_purity(source, target),
        "expert_to_feature_purity": mapping_purity(target, source),
        "same_feature_same_expert_rate": masked_pairwise_same(labels, feature, mask),
        "effective_experts": effective_experts(effective_label_tensor),
        "count": len(source),
    }


def average_dicts(rows: List[Dict], keys: List[str]) -> Dict:
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def masked_attention_metrics(attn: torch.Tensor, feature: torch.Tensor, mask: Optional[torch.Tensor] = None):
    feature_for_attention = feature.clone()
    if mask is not None:
        feature_for_attention = feature_for_attention.masked_fill(~mask, -1)
    return attention_feature_metrics(attn, feature_for_attention)


def attention_expert_summary(attn: torch.Tensor, labels: torch.Tensor, combined: torch.Tensor, label_kind: str) -> Dict:
    if label_kind == "head":
        return attention_expert_alignment(attn, labels)
    return attention_expert_alignment(attn, combined)


def parse_train_loss(log_dir: str, run: str):
    path = os.path.join(log_dir, f"{run}.train.log")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    matches = re.findall(r"batch:\s*(\d+)-\d+,\s*loss:\s*([0-9.]+),\s*token_loss:\s*([0-9.]+)", text)
    if not matches:
        return None
    step, loss, token_loss = matches[-1]
    return {"step": int(step), "loss": float(loss), "token_loss": float(token_loss)}


def get_meta_batch(items, key):
    return torch.stack([item[3][key] for item in items])


def prediction_masks(source_meta, target_meta, target, slot_size):
    source_pos = source_meta["position_in_base_slot"]
    source_slot_type = source_meta["slot_type_ids_by_layer"][:, :, 0]
    target_slot_type = target_meta["slot_type_ids_by_layer"][:, :, 0]
    valid = target != 0
    masks = {
        "overall": valid,
        "predict_prefix_token": valid & (source_pos < slot_size - 2),
        "predict_output_token": valid & (source_pos == slot_size - 2),
        "predict_next_slot_first_token": valid & (source_pos == slot_size - 1),
    }
    type_names = {
        "normal": ControlledReusedTokenData.SLOT_TYPE_NORMAL,
        "same_input_diff_output": ControlledReusedTokenData.SLOT_TYPE_SAME_INPUT_DIFF_OUTPUT,
        "diff_input_same_output": ControlledReusedTokenData.SLOT_TYPE_DIFF_INPUT_SAME_OUTPUT,
    }
    for name, value in type_names.items():
        masks[f"current_base_{name}"] = valid & (source_slot_type == value)
        masks[f"predict_output_from_base_{name}"] = valid & (source_pos == slot_size - 2) & (source_slot_type == value)
        masks[f"target_base_{name}"] = valid & (target_slot_type == value)
    return masks


@torch.no_grad()
def analyze_run(args, run: str, dataset, device):
    run_dir = os.path.join(args.checkpoint_root, run)
    runtime_config = load_runtime_config(run_dir)
    ckpt_path, step = find_checkpoint(run_dir, args.checkpoint_step)
    if ckpt_path is None:
        return {"run": run, "error": "missing checkpoint"}

    model_args = infer_debug_shape_from_checkpoint(args, ckpt_path)
    model = load_model(model_args, runtime_config, ckpt_path, device)
    use_moe = bool(runtime_config.get("use_moe", getattr(model.config, "use_moe", False)))
    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)

    counters = {}
    layer_experts = [[] for _ in range(num_layers)]
    layer_attention_feature_rows = [[] for _ in range(num_layers)]
    layer_attention_expert_rows = [[] for _ in range(num_layers)]
    feature_batches = []

    for start in range(0, args.num_samples, args.batch_size):
        end = min(start + args.batch_size, args.num_samples)
        items = [dataset[idx] for idx in range(start, end)]
        source = torch.stack([item[0] for item in items]).to(device)
        target = torch.stack([item[1] for item in items]).to(device)
        source_meta_cpu = {
            "unit_ids_by_layer": get_meta_batch(items, "unit_ids_by_layer"),
            "slot_type_ids_by_layer": get_meta_batch(items, "slot_type_ids_by_layer"),
            "group_ids_by_layer": get_meta_batch(items, "group_ids_by_layer"),
            "position_in_base_slot": get_meta_batch(items, "position_in_base_slot"),
        }
        target_meta_cpu = {
            key: torch.stack([dataset.get_metadata(idx)[key][1:] for idx in range(start, end)])
            for key in source_meta_cpu
        }
        source_meta = {key: value.to(device) for key, value in source_meta_cpu.items()}
        target_meta = {key: value.to(device) for key, value in target_meta_cpu.items()}

        output = model(
            source,
            output_attentions=True,
            output_expert_labels=use_moe,
            use_cache=False,
        )
        loss = F.cross_entropy(output.logits.reshape(-1, output.logits.size(-1)), target.reshape(-1), reduction="none")
        loss = loss.reshape_as(target)
        pred = output.logits.argmax(dim=-1)

        for name, mask in prediction_masks(source_meta, target_meta, target, args.synthetic_block_size).items():
            counters.setdefault(name, new_counter())
            update_counter(counters[name], loss, pred, target, mask)

        feature_batches.append({
            "base_unit": source_meta_cpu["unit_ids_by_layer"][:, :, 0],
            "high_unit": source_meta_cpu["unit_ids_by_layer"][:, :, 1] if args.synthetic_num_hierarchy_layers > 1 else source_meta_cpu["unit_ids_by_layer"][:, :, 0],
            "base_group": source_meta_cpu["group_ids_by_layer"][:, :, 0],
            "high_group": source_meta_cpu["group_ids_by_layer"][:, :, 1] if args.synthetic_num_hierarchy_layers > 1 else source_meta_cpu["group_ids_by_layer"][:, :, 0],
            "base_type": source_meta_cpu["slot_type_ids_by_layer"][:, :, 0],
            "high_type": source_meta_cpu["slot_type_ids_by_layer"][:, :, 1] if args.synthetic_num_hierarchy_layers > 1 else source_meta_cpu["slot_type_ids_by_layer"][:, :, 0],
            "token_id": source.cpu(),
            "target_token": target.cpu(),
        })

        for layer_idx, attn_tensor in enumerate(output.attentions):
            attn = attn_tensor.detach().cpu()
            rows = {
                "base_unit": masked_attention_metrics(attn, source_meta_cpu["unit_ids_by_layer"][:, :, 0]),
                "high_unit": masked_attention_metrics(attn, feature_batches[-1]["high_unit"]),
                "base_same_input_diff_output_group": masked_attention_metrics(
                    attn,
                    source_meta_cpu["group_ids_by_layer"][:, :, 0],
                    source_meta_cpu["slot_type_ids_by_layer"][:, :, 0] == ControlledReusedTokenData.SLOT_TYPE_SAME_INPUT_DIFF_OUTPUT,
                ),
                "base_diff_input_same_output_group": masked_attention_metrics(
                    attn,
                    source_meta_cpu["group_ids_by_layer"][:, :, 0],
                    source_meta_cpu["slot_type_ids_by_layer"][:, :, 0] == ControlledReusedTokenData.SLOT_TYPE_DIFF_INPUT_SAME_OUTPUT,
                ),
            }
            layer_attention_feature_rows[layer_idx].append(rows)

        if use_moe:
            for layer_idx, labels in enumerate(output.expert_labels):
                primary = labels[..., 0].detach().cpu()
                layer_experts[layer_idx].append(primary)
                label_kind, label_for_alignment, combined = normalize_expert_labels(primary, runtime_config, num_heads)
                layer_attention_expert_rows[layer_idx].append(
                    attention_expert_summary(output.attentions[layer_idx].detach().cpu(), label_for_alignment, combined, label_kind)
                )

    metrics = {key: finish_counter(value) for key, value in counters.items()}
    features = {
        key: torch.cat([batch[key] for batch in feature_batches], dim=0)
        for key in feature_batches[0]
    }

    layers = []
    if use_moe:
        for layer_idx in range(num_layers):
            primary = torch.cat(layer_experts[layer_idx], dim=0)
            label_kind, labels, combined = normalize_expert_labels(primary, runtime_config, num_heads)
            layer_row = {
                "layer": layer_idx,
                "label_kind": label_kind,
                "expert_shape": list(primary.shape),
                "effective_experts": effective_experts(combined),
                "token_id": feature_selectivity(labels, combined, features["token_id"]),
                "target_token": feature_selectivity(labels, combined, features["target_token"]),
                "base_unit": feature_selectivity(labels, combined, features["base_unit"]),
                "high_unit": feature_selectivity(labels, combined, features["high_unit"]),
                "base_same_input_diff_output_group": feature_selectivity(
                    labels,
                    combined,
                    features["base_group"],
                    features["base_type"] == ControlledReusedTokenData.SLOT_TYPE_SAME_INPUT_DIFF_OUTPUT,
                ),
                "base_diff_input_same_output_group": feature_selectivity(
                    labels,
                    combined,
                    features["base_group"],
                    features["base_type"] == ControlledReusedTokenData.SLOT_TYPE_DIFF_INPUT_SAME_OUTPUT,
                ),
                "base_normal_group": feature_selectivity(
                    labels,
                    combined,
                    features["base_group"],
                    features["base_type"] == ControlledReusedTokenData.SLOT_TYPE_NORMAL,
                ),
                "high_same_input_diff_output_group": feature_selectivity(
                    labels,
                    combined,
                    features["high_group"],
                    features["high_type"] == ControlledReusedTokenData.SLOT_TYPE_SAME_INPUT_DIFF_OUTPUT,
                ),
                "high_diff_input_same_output_group": feature_selectivity(
                    labels,
                    combined,
                    features["high_group"],
                    features["high_type"] == ControlledReusedTokenData.SLOT_TYPE_DIFF_INPUT_SAME_OUTPUT,
                ),
            }
            rows = layer_attention_expert_rows[layer_idx]
            layer_row["attention_expert"] = average_dicts(rows, [
                "same_expert_include_self_mass_mean",
                "same_expert_history_mass_mean",
                "same_expert_include_self_pair_score_lift",
                "same_expert_history_pair_score_lift",
            ])
            layers.append(layer_row)

    attention_layers = []
    for layer_idx in range(num_layers):
        row = {"layer": layer_idx}
        for key in [
            "base_unit",
            "high_unit",
            "base_same_input_diff_output_group",
            "base_diff_input_same_output_group",
        ]:
            row[key] = average_dicts([item[key] for item in layer_attention_feature_rows[layer_idx]], [
                "include_self_mass_mean",
                "history_mass_mean",
                "include_self_pair_score_lift",
                "history_pair_score_lift",
            ])
        attention_layers.append(row)

    mean_layers = {}
    if use_moe and layers:
        for key in [
            "token_id",
            "target_token",
            "base_unit",
            "high_unit",
            "base_same_input_diff_output_group",
            "base_diff_input_same_output_group",
            "base_normal_group",
            "high_same_input_diff_output_group",
            "high_diff_input_same_output_group",
        ]:
            mean_layers[key] = {
                "same_feature_same_expert_rate": sum(layer[key]["same_feature_same_expert_rate"] for layer in layers) / len(layers),
                "feature_to_expert_purity": sum(layer[key]["feature_to_expert_purity"] for layer in layers) / len(layers),
                "expert_to_feature_purity": sum(layer[key]["expert_to_feature_purity"] for layer in layers) / len(layers),
                "feature_expert_mi": sum(layer[key]["feature_expert_mi"] for layer in layers) / len(layers),
            }
        mean_layers["effective_experts"] = sum(layer["effective_experts"] for layer in layers) / len(layers)
        mean_layers["attention_expert_mass"] = sum(layer["attention_expert"]["same_expert_include_self_mass_mean"] for layer in layers) / len(layers)

    mean_attention = {}
    for key in [
        "base_unit",
        "high_unit",
        "base_same_input_diff_output_group",
        "base_diff_input_same_output_group",
    ]:
        mean_attention[key] = {
            "include_self_mass_mean": sum(layer[key]["include_self_mass_mean"] for layer in attention_layers) / len(attention_layers),
            "history_mass_mean": sum(layer[key]["history_mass_mean"] for layer in attention_layers) / len(attention_layers),
            "include_self_pair_score_lift": sum(layer[key]["include_self_pair_score_lift"] for layer in attention_layers) / len(attention_layers),
            "history_pair_score_lift": sum(layer[key]["history_pair_score_lift"] for layer in attention_layers) / len(attention_layers),
        }

    return {
        "run": run,
        "checkpoint_step": step,
        "checkpoint_path": ckpt_path,
        "train": parse_train_loss(args.log_dir, run),
        "runtime_config": runtime_config,
        "use_moe": use_moe,
        "metrics": metrics,
        "mean_layers": mean_layers,
        "mean_attention": mean_attention,
        "layers": layers,
        "attention_layers": attention_layers,
    }


def pct(value):
    return f"{100.0 * value:.2f}%"


def fmt(value):
    if value is None:
        return "-"
    return f"{value:.4f}"


def get_nested(row, *keys, default=None):
    cur = row
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def write_markdown(args, results):
    lines = []
    lines.append("# Round6 Controlled Reused-token Analysis")
    lines.append("")
    lines.append("数据设定：controlled reused-token；默认 `uniform` 顶层采样；same-input-diff-output 与 diff-input-same-output 各占 `10%`。")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    headers = [
        "Run",
        "Type",
        "Train loss",
        "Eval loss",
        "NTP acc",
        "Output-token acc",
        "A output acc",
        "B output acc",
        "Token same-expert",
        "Target same-expert",
        "Base unit same-expert",
        "A group same-expert",
        "B group same-expert",
        "Attn base mass",
        "Attn high mass",
        "Attn A mass",
        "Attn B mass",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in results:
        if "error" in row:
            lines.append(f"| `{row['run']}` | ERROR | | | | | | | | | | | | | | |")
            continue
        train_loss = row["train"]["loss"] if row["train"] else None
        model_type = "moe" if row.get("use_moe") else "dense"
        mean = row.get("mean_layers", {})
        attn = row.get("mean_attention", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['run']}`",
                    model_type,
                    fmt(train_loss),
                    fmt(row["metrics"]["overall"]["loss"]),
                    pct(row["metrics"]["overall"]["accuracy"]),
                    pct(row["metrics"]["predict_output_token"]["accuracy"]),
                    pct(row["metrics"]["predict_output_from_base_same_input_diff_output"]["accuracy"]),
                    pct(row["metrics"]["predict_output_from_base_diff_input_same_output"]["accuracy"]),
                    pct(get_nested(mean, "token_id", "same_feature_same_expert_rate", default=0.0)),
                    pct(get_nested(mean, "target_token", "same_feature_same_expert_rate", default=0.0)),
                    pct(get_nested(mean, "base_unit", "same_feature_same_expert_rate", default=0.0)),
                    pct(get_nested(mean, "base_same_input_diff_output_group", "same_feature_same_expert_rate", default=0.0)),
                    pct(get_nested(mean, "base_diff_input_same_output_group", "same_feature_same_expert_rate", default=0.0)),
                    pct(get_nested(attn, "base_unit", "include_self_mass_mean", default=0.0)),
                    pct(get_nested(attn, "high_unit", "include_self_mass_mean", default=0.0)),
                    pct(get_nested(attn, "base_same_input_diff_output_group", "include_self_mass_mean", default=0.0)),
                    pct(get_nested(attn, "base_diff_input_same_output_group", "include_self_mass_mean", default=0.0)),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Metric Notes")
    lines.append("")
    lines.append("- `A` 表示 same-input-different-output slot；`B` 表示 different-input-same-output slot。")
    lines.append("- `Output-token acc` 只统计 slot 中倒数第二个位置预测最后一个 output symbol 的准确率。")
    lines.append("- `A/B group same-expert` 是同一结构 group 内 token 被分到同一 expert bucket 的比例；dense 模型没有该项。")
    lines.append("- `Attn A/B mass` 是 attention mass 落在同一 A/B group 内 token 上的比例，包含 self attention。")
    lines.append("")
    lines.append("## Full JSON")
    lines.append("")
    lines.append(f"See `{args.output_json}` for per-layer details.")
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    device = choose_device()
    dataset = make_dataset(args)
    runs = [run.strip() for run in args.runs.split(",") if run.strip()]
    results = [analyze_run(args, run, dataset, device) for run in runs]
    output = {"config": vars(args), "device": str(device), "runs": results}
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    write_markdown(args, results)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
