import argparse
import json
import math
import os
from collections import Counter

import torch
import torch.nn.functional as F

from analyze_moe_variant_selectivity import (
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
    parser = argparse.ArgumentParser(description="Diagnose what MoE routing aligns with.")
    parser.add_argument("--config_dir", type=str, default="../Qwen3-0.6B")
    parser.add_argument("--checkpoint_root", type=str, default="../checkpoints")
    parser.add_argument("--runs", type=str, default="inverse-kv-zipf-baseline")
    parser.add_argument("--checkpoint_step", type=int, default=5000)
    parser.add_argument("--output_path", type=str, default="../experiments/baseline_routing_feature_diagnostic.json")

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
    parser.add_argument("--attention_rho", type=float, default=0.75)
    parser.add_argument("--cluster_sample_limit", type=int, default=8192)
    parser.add_argument("--top_candidates", type=int, default=8)
    return parser.parse_args()


def nmi(feature, expert):
    feature_list = [int(x) for x in feature.reshape(-1).tolist()]
    expert_list = [int(x) for x in expert.reshape(-1).tolist()]
    mi = mutual_information(feature_list, expert_list)
    h_feature = entropy_from_counts(Counter(feature_list).values())
    h_expert = entropy_from_counts(Counter(expert_list).values())
    denom = math.sqrt(max(h_feature * h_expert, 1e-12))
    return float(mi / denom), float(mi), float(h_feature)


def feature_expert_metrics(feature_tensor, expert_tensor):
    original_feature_tensor = feature_tensor
    if expert_tensor.dim() == feature_tensor.dim() + 1:
        feature_tensor = feature_tensor.unsqueeze(-1).expand_as(expert_tensor)
    if feature_tensor.shape != expert_tensor.shape:
        raise ValueError(f"feature/expert shape mismatch: {tuple(feature_tensor.shape)} vs {tuple(expert_tensor.shape)}")
    valid = feature_tensor >= 0
    if valid.sum().item() == 0:
        return {
            "num_valid_tokens": 0,
            "num_feature_values": 0,
            "feature_expert_mi": 0.0,
            "feature_expert_nmi": 0.0,
            "feature_entropy": 0.0,
            "feature_to_expert_purity": 0.0,
            "expert_to_feature_purity": 0.0,
            "same_feature_same_expert_rate": 0.0,
        }
    feature_valid = feature_tensor.masked_fill(~valid, -1)
    expert_valid = expert_tensor.masked_fill(~valid, -1)
    flat_feature = feature_valid[valid].reshape(-1).tolist()
    flat_expert = expert_valid[valid].reshape(-1).tolist()
    nmi_value, mi_value, feature_entropy = nmi(feature_valid[valid], expert_valid[valid])
    if expert_tensor.dim() == 3 and original_feature_tensor.dim() == 2:
        same_feature_same_expert = float(
            sum(
                pairwise_same_bucket_given_same_feature(expert_tensor[:, :, head_idx], original_feature_tensor)
                for head_idx in range(expert_tensor.shape[-1])
            )
            / expert_tensor.shape[-1]
        )
    else:
        same_feature_same_expert = pairwise_same_bucket_given_same_feature(expert_valid, feature_valid)
    return {
        "num_valid_tokens": int(valid.sum().item()),
        "num_feature_values": int(len(set(int(x) for x in flat_feature))),
        "feature_expert_mi": mi_value,
        "feature_expert_nmi": nmi_value,
        "feature_entropy": feature_entropy,
        "feature_to_expert_purity": mapping_purity(flat_feature, flat_expert),
        "expert_to_feature_purity": mapping_purity(flat_expert, flat_feature),
        "same_feature_same_expert_rate": same_feature_same_expert,
    }


def zipf_rank_features(dataset, metadata, num_units):
    if dataset.top_unit_sample_weights is None:
        rank = torch.full_like(metadata[:, :, 1], -1)
        bucket = torch.full_like(metadata[:, :, 1], -1)
        return rank, bucket
    weights = torch.tensor(dataset.top_unit_sample_weights, dtype=torch.float32)
    order = torch.argsort(weights, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order), dtype=torch.long)
    high = metadata[:, :, 1].clamp_min(0)
    rank = ranks[high].masked_fill(metadata[:, :, 1] < 0, -1)
    bucket_count = min(8, max(1, num_units))
    bucket = torch.div(rank * bucket_count, max(num_units, 1), rounding_mode="floor")
    bucket = bucket.clamp(max=bucket_count - 1).masked_fill(rank < 0, -1)
    return rank, bucket


def build_static_features(dataset, source, target, metadata, args):
    batch, seq_len = source.shape
    positions = torch.arange(seq_len, device=source.device)[None, :].expand(batch, -1)
    local_pos = positions % int(args.synthetic_block_size)
    local_boundary = torch.zeros_like(local_pos)
    local_boundary = local_boundary.masked_fill(local_pos == 0, 1)
    local_boundary = local_boundary.masked_fill(local_pos == int(args.synthetic_block_size) - 1, 2)
    high_span = int(args.synthetic_block_size) ** int(args.synthetic_num_hierarchy_layers)
    high_pos = positions % max(high_span, 1)
    high_boundary = torch.zeros_like(high_pos)
    high_boundary = high_boundary.masked_fill(high_pos == 0, 1)
    high_boundary = high_boundary.masked_fill(high_pos == high_span - 1, 2)
    zipf_rank, zipf_rank_bucket = zipf_rank_features(dataset, metadata.cpu(), args.synthetic_num_units_per_layer)
    return {
        "local_slot": metadata[:, :, 0].cpu(),
        "higher_unit": metadata[:, :, 1].cpu(),
        "token_id": source.cpu(),
        "target_token": target.cpu(),
        "slot_position": local_pos.cpu(),
        "local_boundary": local_boundary.cpu(),
        "higher_position": high_pos.cpu(),
        "higher_boundary": high_boundary.cpu(),
        "zipf_rank": zipf_rank.cpu(),
        "zipf_rank_bucket": zipf_rank_bucket.cpu(),
    }


def top_mass_majority_feature(attn, feature_ids, rho):
    # attn: [B, H, S, S], feature_ids: [B, S]. Uses mean attention over heads and history-only keys.
    attn = attn.float().mean(dim=1)
    batch, seq_len, _ = attn.shape
    device = attn.device
    history = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=-1)
    attn = attn.masked_fill(~history[None, :, :], 0.0)
    row_sum = attn.sum(dim=-1, keepdim=True)
    attn = attn / row_sum.clamp_min(1e-8)
    sorted_vals, sorted_idx = torch.sort(attn, dim=-1, descending=True)
    cumsum = sorted_vals.cumsum(dim=-1)
    keep_sorted = cumsum <= float(rho)
    first_over = (cumsum > float(rho)).to(torch.int64).argmax(dim=-1, keepdim=True)
    keep_sorted.scatter_(-1, first_over, True)
    keep_sorted = keep_sorted & (sorted_vals > 0)
    keep = torch.zeros_like(keep_sorted)
    keep.scatter_(-1, sorted_idx, keep_sorted)

    majority = torch.full((batch, seq_len), -1, dtype=torch.long, device=device)
    for b in range(batch):
        for q in range(1, seq_len):
            key_idx = torch.nonzero(keep[b, q], as_tuple=False).flatten()
            if key_idx.numel() == 0:
                continue
            values = feature_ids[b, key_idx]
            values = values[values >= 0]
            if values.numel() == 0:
                continue
            unique, counts = torch.unique(values, return_counts=True)
            majority[b, q] = unique[torch.argmax(counts)]
    return majority.cpu()


def top1_key_feature(attn, feature_ids):
    attn = attn.float().mean(dim=1)
    batch, seq_len, _ = attn.shape
    device = attn.device
    history = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=-1)
    masked = attn.masked_fill(~history[None, :, :], -1.0)
    key_idx = masked.argmax(dim=-1)
    result = torch.gather(feature_ids, 1, key_idx)
    result[:, 0] = -1
    return result.cpu()


def collect_attention_features(attn, metadata, rho):
    return {
        "attn_top1_key_local_slot": top1_key_feature(attn, metadata[:, :, 0]),
        "attn_top1_key_higher_unit": top1_key_feature(attn, metadata[:, :, 1]),
        "attn_topmass_majority_local_slot": top_mass_majority_feature(attn, metadata[:, :, 0], rho),
        "attn_topmass_majority_higher_unit": top_mass_majority_feature(attn, metadata[:, :, 1], rho),
    }


def kmeans_labels(x, num_clusters, sample_limit, iterations=25, seed=0):
    if x.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    x = x.float()
    n = x.shape[0]
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed))
    if n > sample_limit:
        sample_idx = torch.randperm(n, generator=generator, device=x.device)[:sample_limit]
        fit_x = x[sample_idx]
    else:
        fit_x = x
    init_idx = torch.randperm(fit_x.shape[0], generator=generator, device=x.device)[:num_clusters]
    centers = fit_x[init_idx].clone()
    for _ in range(iterations):
        dist = torch.cdist(fit_x, centers)
        labels = dist.argmin(dim=-1)
        new_centers = []
        for cluster_idx in range(num_clusters):
            mask = labels == cluster_idx
            if mask.any():
                new_centers.append(fit_x[mask].mean(dim=0))
            else:
                new_centers.append(centers[cluster_idx])
        new_centers = torch.stack(new_centers, dim=0)
        if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-5):
            centers = new_centers
            break
        centers = new_centers
    return torch.cdist(x, centers).argmin(dim=-1).cpu()


def flatten_batches(chunks):
    return torch.cat([chunk.reshape(-1, chunk.shape[-1]) for chunk in chunks], dim=0)


def flatten_feature_chunks(chunks):
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def analyze_run(args, run_name, dataset, device):
    run_dir = os.path.join(args.checkpoint_root, run_name)
    runtime_config = load_runtime_config(run_dir)
    ckpt_path, step = find_checkpoint(run_dir, args.checkpoint_step)
    if ckpt_path is None:
        return {"run": run_name, "error": "no checkpoint found", "checkpoint_dir": run_dir}

    model = load_model(args, runtime_config, ckpt_path, device)
    is_head_level = bool(getattr(model.config, "moe_head_level", False))

    attn_output_cache = {}
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def hook(_module, _inputs, output):
                attn_output_cache[idx] = output[0].detach().cpu()
            return hook
        handles.append(layer.self_attn.register_forward_hook(make_hook(layer_idx)))

    ntp = {"loss_sum": 0.0, "correct": 0, "count": 0}
    static_feature_chunks = {}
    attention_feature_chunks = [
        {
            "attn_top1_key_local_slot": [],
            "attn_top1_key_higher_unit": [],
            "attn_topmass_majority_local_slot": [],
            "attn_topmass_majority_higher_unit": [],
        }
        for _ in range(model.config.num_hidden_layers)
    ]
    expert_chunks = [[] for _ in range(model.config.num_hidden_layers)]
    hidden_rep_chunks = [[] for _ in range(model.config.num_hidden_layers)]
    attn_rep_chunks = [[] for _ in range(model.config.num_hidden_layers)]

    try:
        for start in range(0, args.num_samples, args.batch_size):
            batch_items = [dataset[i] for i in range(start, min(start + args.batch_size, args.num_samples))]
            source = torch.stack([item[0] for item in batch_items]).to(device)
            target = torch.stack([item[1] for item in batch_items]).to(device)
            metadata = torch.stack([item[3] for item in batch_items]).to(device)

            outputs = model(
                source,
                output_attentions=True,
                output_expert_labels=True,
                output_hidden_states=True,
                use_cache=False,
            )
            loss = F.cross_entropy(
                outputs.logits.reshape(-1, outputs.logits.size(-1)),
                target.reshape(-1),
                reduction="none",
            ).reshape_as(target)
            pred = outputs.logits.argmax(dim=-1)
            valid = target != 0
            ntp["loss_sum"] += float(loss[valid].sum().item())
            ntp["correct"] += int(((pred == target) & valid).sum().item())
            ntp["count"] += int(valid.sum().item())

            static_features = build_static_features(dataset, source, target, metadata, args)
            for name, tensor in static_features.items():
                static_feature_chunks.setdefault(name, []).append(tensor)

            for layer_idx, labels in enumerate(outputs.expert_labels):
                expert_chunks[layer_idx].append(primary_expert_labels(labels).detach().cpu())
                hidden_rep_chunks[layer_idx].append(outputs.hidden_states[layer_idx + 1].detach().cpu())
                attn_rep_chunks[layer_idx].append(attn_output_cache[layer_idx])
                attn_features = collect_attention_features(outputs.attentions[layer_idx].detach(), metadata, args.attention_rho)
                for name, tensor in attn_features.items():
                    attention_feature_chunks[layer_idx][name].append(tensor)
    finally:
        for handle in handles:
            handle.remove()

    static_features_all = {
        name: flatten_feature_chunks(chunks)
        for name, chunks in static_feature_chunks.items()
    }

    layers = []
    for layer_idx in range(model.config.num_hidden_layers):
        expert_tensor = flatten_feature_chunks(expert_chunks[layer_idx])
        flat_experts = expert_tensor.reshape(-1).tolist()
        expert_counts = Counter(flat_experts)
        layer_features = dict(static_features_all)
        for name, chunks in attention_feature_chunks[layer_idx].items():
            layer_features[name] = flatten_feature_chunks(chunks)

        num_experts = int(getattr(model.config, "moe_num_unique_experts", 4))
        hidden_reps = flatten_batches(hidden_rep_chunks[layer_idx])
        attn_reps = flatten_batches(attn_rep_chunks[layer_idx])
        layer_features["hidden_kmeans_cluster"] = kmeans_labels(
            hidden_reps,
            num_clusters=num_experts,
            sample_limit=args.cluster_sample_limit,
            seed=1000 + layer_idx,
        ).reshape(expert_tensor.shape[0], expert_tensor.shape[1])
        layer_features["attention_output_kmeans_cluster"] = kmeans_labels(
            attn_reps,
            num_clusters=num_experts,
            sample_limit=args.cluster_sample_limit,
            seed=2000 + layer_idx,
        ).reshape(expert_tensor.shape[0], expert_tensor.shape[1])

        candidate_metrics = {}
        for feature_name, feature_tensor in layer_features.items():
            candidate_metrics[feature_name] = feature_expert_metrics(feature_tensor.long(), expert_tensor.long())

        per_head_top_by_nmi = None
        if is_head_level:
            per_head_top_by_nmi = []
            num_heads = expert_tensor.shape[-1]
            for head_idx in range(num_heads):
                head_experts = expert_tensor[:, :, head_idx]
                head_rows = []
                for feature_name, feature_tensor in layer_features.items():
                    metrics = feature_expert_metrics(feature_tensor.long(), head_experts.long())
                    head_rows.append({"feature": feature_name, **metrics})
                head_rows.sort(
                    key=lambda row: (row["feature_expert_nmi"], row["feature_to_expert_purity"]),
                    reverse=True,
                )
                per_head_top_by_nmi.append({"head": head_idx, "top_by_nmi": head_rows[: args.top_candidates]})

        ranked_by_nmi = sorted(
            (
                {"feature": name, **metrics}
                for name, metrics in candidate_metrics.items()
            ),
            key=lambda row: (row["feature_expert_nmi"], row["feature_to_expert_purity"]),
            reverse=True,
        )
        ranked_by_purity = sorted(
            (
                {"feature": name, **metrics}
                for name, metrics in candidate_metrics.items()
            ),
            key=lambda row: (row["feature_to_expert_purity"], row["feature_expert_nmi"]),
            reverse=True,
        )

        layers.append(
            {
                "layer": layer_idx,
                "expert_load": dict(expert_counts),
                "expert_entropy": entropy_from_counts(expert_counts.values()),
                "effective_expert_count": math.exp(entropy_from_counts(expert_counts.values())),
                "max_expert_fraction": max(expert_counts.values()) / max(sum(expert_counts.values()), 1),
                "is_head_level_moe": is_head_level,
                "candidate_metrics": candidate_metrics,
                "top_by_nmi": ranked_by_nmi[: args.top_candidates],
                "top_by_feature_to_expert_purity": ranked_by_purity[: args.top_candidates],
                "per_head_top_by_nmi": per_head_top_by_nmi,
            }
        )

    return {
        "run": run_name,
        "checkpoint_step": step,
        "checkpoint_path": ckpt_path,
        "runtime_config": runtime_config,
        "ntp": {
            "loss": ntp["loss_sum"] / max(ntp["count"], 1),
            "accuracy": ntp["correct"] / max(ntp["count"], 1),
            "count": ntp["count"],
        },
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
        "metric_notes": {
            "feature_to_expert_purity": "For each feature value, fraction covered by its dominant expert.",
            "expert_to_feature_purity": "For each expert, fraction covered by its dominant feature value.",
            "same_feature_same_expert_rate": "Pairwise probability that two tokens with the same feature share expert.",
            "feature_expert_nmi": "Normalized MI between feature id and expert id; robust against trivial purity inflation.",
        },
        "runs": [analyze_run(args, run, dataset, device) for run in runs],
    }
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
