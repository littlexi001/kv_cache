import argparse
import json
import math
import os
from collections import Counter

import torch
import torch.nn.functional as F

from analyze_moe_variant_selectivity import (
    attention_expert_alignment,
    build_config,
    choose_device,
    entropy_from_counts,
    find_checkpoint,
    load_model,
    load_runtime_config,
    mapping_purity,
    mutual_information,
    pairwise_same_bucket_given_same_feature,
    primary_expert_labels,
)
from utils import HierarchicalPatternData


def parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def add_args():
    parser = argparse.ArgumentParser(description="Analyze pre-router attention-derived routing experiments.")
    parser.add_argument("--config_dir", type=str, default="../Qwen3-0.6B")
    parser.add_argument("--checkpoint_root", type=str, default="../checkpoints")
    parser.add_argument("--runs", type=str, default="pre-router-zipf-layer_input-kl,pre-router-zipf-layer_input-pairwise,pre-router-zipf-layer_input-topk_logits")
    parser.add_argument("--checkpoint_step", type=int, default=5000)
    parser.add_argument("--output_path", type=str, default="../experiments/pre_router_metrics_step5000.json")

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
    parser.add_argument("--rho", type=float, default=0.75)
    return parser.parse_args()


def top_mass_attention(attn, rho):
    sorted_vals, sorted_idx = torch.sort(attn, dim=-1, descending=True)
    cumsum = sorted_vals.cumsum(dim=-1)
    keep_sorted = cumsum <= float(rho)
    first_over = (cumsum > float(rho)).to(torch.int64).argmax(dim=-1, keepdim=True)
    keep_sorted.scatter_(-1, first_over, True)
    keep_sorted = keep_sorted & (sorted_vals > 0)
    keep = torch.zeros_like(keep_sorted)
    keep.scatter_(-1, sorted_idx, keep_sorted)
    selected = attn * keep.to(attn.dtype)
    return selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-8), keep


def selected_attention_same_expert(attn_weights, expert_labels, rho):
    attn = attn_weights.float().mean(dim=1)
    batch, seq_len, _ = attn.shape
    eye = torch.eye(seq_len, device=attn.device, dtype=torch.bool)
    attn = attn.masked_fill(eye.unsqueeze(0), 0.0)
    row_sum = attn.sum(dim=-1, keepdim=True)
    valid_rows = row_sum.squeeze(-1) > 0
    attn = attn / row_sum.clamp_min(1e-8)
    selected, keep = top_mass_attention(attn, rho)
    same_expert = expert_labels[:, :, None] == expert_labels[:, None, :]
    valid = valid_rows.to(attn.dtype)
    mass = (selected * same_expert.to(selected.dtype)).sum(dim=-1)
    count_rate = ((keep & same_expert).float().sum(dim=-1) / keep.float().sum(dim=-1).clamp_min(1)).masked_fill(~valid_rows, 0.0)
    return {
        "topmass_same_expert_mass": float((mass * valid).sum().item() / valid.sum().clamp_min(1).item()),
        "topmass_same_expert_token_rate": float((count_rate * valid).sum().item() / valid.sum().clamp_min(1).item()),
        "topmass_mean_selected_tokens": float((keep.float().sum(dim=-1) * valid).sum().item() / valid.sum().clamp_min(1).item()),
    }


def routed_kv_retention_stats(expert_labels):
    if expert_labels.dim() == 2:
        expert_labels = expert_labels.unsqueeze(-1)
    batch, seq_len, _ = expert_labels.shape
    overlap = expert_labels[:, :, None, :, None] == expert_labels[:, None, :, None, :]
    same_bucket = overlap.any(dim=(-1, -2))
    history = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=expert_labels.device), diagonal=-1)
    causal_with_self = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=expert_labels.device))
    allowed_history = same_bucket & history.unsqueeze(0)
    allowed_with_self = same_bucket & causal_with_self.unsqueeze(0)

    history_count = history.sum().item() * batch
    causal_count = causal_with_self.sum().item() * batch
    allowed_history_count = allowed_history.sum().item()
    allowed_with_self_count = allowed_with_self.sum().item()
    per_query_history = history.sum(dim=-1).clamp_min(1).to(torch.float32)
    per_query_allowed_history = allowed_history.float().sum(dim=-1)
    valid_query = history.sum(dim=-1) > 0
    history_fraction_by_query = per_query_allowed_history[:, valid_query] / per_query_history[valid_query]

    return {
        "allowed_history_token_fraction": float(allowed_history_count / max(history_count, 1)),
        "kv_compression_ratio_history": float(history_count / max(allowed_history_count, 1)),
        "allowed_history_tokens_per_query": float(allowed_history_count / max(batch * max(seq_len - 1, 1), 1)),
        "allowed_history_fraction_per_query_mean": float(history_fraction_by_query.mean().item())
        if history_fraction_by_query.numel() > 0
        else 0.0,
        "allowed_with_self_token_fraction": float(allowed_with_self_count / max(causal_count, 1)),
        "kv_compression_ratio_with_self": float(causal_count / max(allowed_with_self_count, 1)),
    }


def expert_stats(expert_tensor, feature_tensors):
    flat_experts = expert_tensor.reshape(-1).tolist()
    counts = Counter(flat_experts)
    total = sum(counts.values())
    entropy = entropy_from_counts(counts.values())
    effective = math.exp(entropy)
    result = {
        "expert_load": dict(counts),
        "expert_entropy": entropy,
        "effective_expert_count": effective,
        "max_expert_fraction": max(counts.values()) / max(total, 1),
        "feature_selectivity": {},
    }
    for feature_name, feature_tensor in feature_tensors.items():
        flat_feature = feature_tensor.reshape(-1).tolist()
        result["feature_selectivity"][feature_name] = {
            "feature_expert_mi": mutual_information(flat_feature, flat_experts),
            "feature_to_expert_purity": mapping_purity(flat_feature, flat_experts),
            "expert_to_feature_purity": mapping_purity(flat_experts, flat_feature),
            "same_feature_same_expert_rate": pairwise_same_bucket_given_same_feature(expert_tensor, feature_tensor),
        }
    return result


def update_ntp(state, logits, target):
    valid = target != 0
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none").reshape_as(target)
    pred = logits.argmax(dim=-1)
    state["loss_sum"] += float(loss[valid].sum().item())
    state["correct"] += int(((pred == target) & valid).sum().item())
    state["count"] += int(valid.sum().item())


def finalize_ntp(state):
    count = max(state["count"], 1)
    return {
        "loss": state["loss_sum"] / count,
        "accuracy": state["correct"] / count,
        "count": state["count"],
    }


def average_metric_dict(rows):
    keys = rows[0].keys()
    return {
        key: (
            [float(sum(row[key][i] for row in rows) / len(rows)) for i in range(len(rows[0][key]))]
            if isinstance(rows[0][key], list)
            else float(sum(row[key] for row in rows) / len(rows))
        )
        for key in keys
    }


@torch.no_grad()
def analyze_run(args, run_name, dataset, device):
    run_dir = os.path.join(args.checkpoint_root, run_name)
    runtime_config = load_runtime_config(run_dir)
    ckpt_path, step = find_checkpoint(run_dir, args.checkpoint_step)
    if ckpt_path is None:
        return {"run": run_name, "error": "no checkpoint found", "checkpoint_dir": run_dir}

    model = load_model(args, runtime_config, ckpt_path, device)
    if not bool(getattr(model.config, "use_pre_router", False)):
        return {"run": run_name, "error": "checkpoint config does not enable use_pre_router", "checkpoint_path": ckpt_path}

    full_ntp = {"loss_sum": 0.0, "correct": 0, "count": 0}
    routed_ntp = {"loss_sum": 0.0, "correct": 0, "count": 0}
    feature_chunks = {"local_slot": [], "higher_unit": []}
    expert_chunks = [[] for _ in range(model.config.num_hidden_layers)]
    topmass_rows = [[] for _ in range(model.config.num_hidden_layers)]
    same_expert_attention_rows = [[] for _ in range(model.config.num_hidden_layers)]
    kv_retention_rows = [[] for _ in range(model.config.num_hidden_layers)]
    routed_expert_chunks = [[] for _ in range(model.config.num_hidden_layers)]

    for start in range(0, args.num_samples, args.batch_size):
        items = [dataset[i] for i in range(start, min(start + args.batch_size, args.num_samples))]
        source = torch.stack([item[0] for item in items]).to(device)
        target = torch.stack([item[1] for item in items]).to(device)
        metadata = torch.stack([item[3] for item in items]).to(device)
        feature_chunks["local_slot"].append(metadata[:, :, 0].cpu())
        feature_chunks["higher_unit"].append(metadata[:, :, 1].cpu())

        outputs = model(
            source,
            output_attentions=True,
            output_expert_labels=True,
            use_cache=False,
        )
        update_ntp(full_ntp, outputs.logits, target)
        for layer_idx, labels in enumerate(outputs.expert_labels):
            primary = primary_expert_labels(labels).detach()
            expert_chunks[layer_idx].append(primary.cpu())
            kv_retention_rows[layer_idx].append(routed_kv_retention_stats(labels.detach()))
            topmass_rows[layer_idx].append(
                selected_attention_same_expert(outputs.attentions[layer_idx].detach(), primary, args.rho)
            )
            same_expert_attention_rows[layer_idx].append(
                attention_expert_alignment(outputs.attentions[layer_idx].detach(), primary)
            )

        routed_outputs = model(
            source,
            output_attentions=False,
            output_expert_labels=True,
            use_cache=False,
            use_pre_router_kv_mask=True,
        )
        update_ntp(routed_ntp, routed_outputs.logits, target)
        for layer_idx, labels in enumerate(routed_outputs.expert_labels):
            routed_expert_chunks[layer_idx].append(primary_expert_labels(labels).detach().cpu())

    feature_tensors = {name: torch.cat(chunks, dim=0) for name, chunks in feature_chunks.items()}
    layers = []
    for layer_idx in range(model.config.num_hidden_layers):
        expert_tensor = torch.cat(expert_chunks[layer_idx], dim=0)
        routed_expert_tensor = torch.cat(routed_expert_chunks[layer_idx], dim=0)
        layer_stats = expert_stats(expert_tensor, feature_tensors)
        routed_stats = expert_stats(routed_expert_tensor, feature_tensors)
        layer_stats.update(
            {
                "layer": layer_idx,
                "attention_topmass_same_expert": average_metric_dict(topmass_rows[layer_idx]),
                "same_expert_attention_alignment": average_metric_dict(same_expert_attention_rows[layer_idx]),
                "routed_kv_retention": average_metric_dict(kv_retention_rows[layer_idx]),
                "routed_attention_expert_stats": routed_stats,
            }
        )
        layers.append(layer_stats)

    normal_forward_ntp = finalize_ntp(full_ntp)
    pre_router_kv_mask_ntp = finalize_ntp(routed_ntp)
    pre_router_controls_attention = bool(runtime_config.get("pre_router_controls_attention", False))
    return {
        "run": run_name,
        "checkpoint_step": step,
        "checkpoint_path": ckpt_path,
        "runtime_config": runtime_config,
        "normal_forward_attention_mode": (
            "trained_routed_attention" if pre_router_controls_attention else "full_attention"
        ),
        "normal_forward_ntp": normal_forward_ntp,
        "full_attention_ntp": normal_forward_ntp,
        "pre_router_kv_mask_ntp": pre_router_kv_mask_ntp,
        "pre_router_kv_mask_accuracy_drop": normal_forward_ntp["accuracy"] - pre_router_kv_mask_ntp["accuracy"],
        "pre_router_kv_mask_note": (
            "redundant_with_training_forward_when_pre_router_controls_attention_is_true"
            if pre_router_controls_attention
            else "eval_only_routed_attention"
        ),
        "layers": layers,
    }


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
        "metrics": {
            "normal_forward_ntp": "next-token prediction under the checkpoint's normal forward path; for pre_router_controls_attention checkpoints this is trained routed attention",
            "full_attention_ntp": "legacy alias of normal_forward_ntp",
            "expert_entropy/effective_expert_count": "collapse check for pre-router MoE dispatch",
            "attention_topmass_same_expert": "whether high-attention positive set maps to same pre-router expert",
            "feature_selectivity": "whether pre-router indirectly aligns to local/high-level hierarchy features",
            "routed_kv_retention": "fraction of causal historical KV retained by pre-router cluster overlap",
            "pre_router_kv_mask_ntp": "eval-only routed attention using pre-router expert buckets as KV mask",
        },
        "runs": [analyze_run(args, run, dataset, device) for run in runs],
    }
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
