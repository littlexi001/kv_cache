import argparse
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from analyze_moe_variant_selectivity import (
    build_ground_truth_expert_mapping,
    choose_device,
    find_checkpoint,
    load_model,
    load_runtime_config,
    make_ground_truth_expert_ids,
    parse_int_list,
)
from utils import HierarchicalPatternData


DEFAULT_RUNS = ",".join(
    [
        "inverse-kv-supervised-gate-zipf-high-hash",
        "inverse-kv-mlp-gate-hidden-supervised-zipf-high-hash",
        "inverse-kv-mlp-gate-attention_output-supervised-zipf-high-hash",
    ]
)


def add_args():
    parser = argparse.ArgumentParser(
        description="Analyze where supervised-gate routing errors occur relative to hierarchy boundaries."
    )
    parser.add_argument("--config_dir", type=str, default="fdong/Qwen3-0.6B")
    parser.add_argument("--checkpoint_root", type=str, default="fdong/checkpoints")
    parser.add_argument("--runs", type=str, default=DEFAULT_RUNS)
    parser.add_argument("--checkpoint_step", type=int, default=5000)
    parser.add_argument("--output_path", type=str, default="fdong/experiments/ground_truth_routing_error_locations_step5000.json")

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--synthetic_block_size", type=int, default=4)
    parser.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    parser.add_argument("--synthetic_content_token_count", type=int, default=256)
    parser.add_argument("--synthetic_num_units_per_layer", type=int, default=64)
    parser.add_argument("--synthetic_seed", type=int, default=0)
    parser.add_argument("--synthetic_min_token_id", type=int, default=1)
    parser.add_argument("--synthetic_sampling_distribution", type=str, choices=["uniform", "zipf"], default="zipf")
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

    parser.add_argument("--ground_truth_routing_strategy", type=str, choices=["hash", "frequency_balanced"], default="hash")
    parser.add_argument("--ground_truth_routing_feature_layer", type=int, default=1)
    parser.add_argument("--ground_truth_frequency_estimate_samples", type=int, default=4096)
    return parser.parse_args()


def build_dataset(args):
    return HierarchicalPatternData(
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


def probe_runtime_config(args, runtime_config):
    merged = dict(runtime_config)
    merged["ground_truth_routing_strategy"] = args.ground_truth_routing_strategy
    merged["ground_truth_routing_feature_layer"] = int(args.ground_truth_routing_feature_layer)
    merged["ground_truth_frequency_estimate_samples"] = int(args.ground_truth_frequency_estimate_samples)
    return merged


def primary_expert_labels(labels):
    return labels[..., 0]


def make_position_masks(metadata, target_metadata):
    local = metadata[:, :, 0]
    high = metadata[:, :, 1]
    target_local = target_metadata[:, :, 0]
    target_high = target_metadata[:, :, 1]
    batch, seq_len = local.shape
    device = local.device
    pos = torch.arange(seq_len, device=device)[None, :].expand(batch, -1)

    prev_local = torch.cat([local[:, :1], local[:, :-1]], dim=1)
    prev_high = torch.cat([high[:, :1], high[:, :-1]], dim=1)
    next_local = torch.cat([local[:, 1:], target_local[:, -1:]], dim=1)
    next_high = torch.cat([high[:, 1:], target_high[:, -1:]], dim=1)

    masks = {
        "all": torch.ones_like(local, dtype=torch.bool),
        "current_local_start": (pos == 0) | (local != prev_local),
        "current_local_end": local != next_local,
        "current_high_start": (pos == 0) | (high != prev_high),
        "current_high_end": high != next_high,
        "predict_next_local_boundary": local != target_local,
        "predict_next_high_boundary": high != target_high,
        "inside_same_local_next": local == target_local,
        "inside_same_high_next": high == target_high,
    }
    masks["boundary_any"] = (
        masks["current_local_start"]
        | masks["current_local_end"]
        | masks["current_high_start"]
        | masks["current_high_end"]
        | masks["predict_next_local_boundary"]
        | masks["predict_next_high_boundary"]
    )
    masks["boundary_predict_any"] = masks["predict_next_local_boundary"] | masks["predict_next_high_boundary"]
    masks["non_boundary"] = ~masks["boundary_any"]
    return masks


def init_counter():
    return {
        "tokens": 0,
        "routing_errors": 0,
        "prediction_errors": 0,
        "routing_and_prediction_errors": 0,
        "loss_sum": 0.0,
    }


def update_counter(counter, mask, route_error, pred_error, token_loss):
    mask = mask.bool()
    count = int(mask.sum().item())
    counter["tokens"] += count
    if count == 0:
        return
    counter["routing_errors"] += int((route_error & mask).sum().item())
    counter["prediction_errors"] += int((pred_error & mask).sum().item())
    counter["routing_and_prediction_errors"] += int((route_error & pred_error & mask).sum().item())
    counter["loss_sum"] += float(token_loss[mask].sum().item())


def finalize_counter(counter, total_route_errors):
    tokens = max(counter["tokens"], 1)
    routing_errors = counter["routing_errors"]
    prediction_errors = counter["prediction_errors"]
    return {
        **counter,
        "routing_error_rate": routing_errors / tokens,
        "prediction_error_rate": prediction_errors / tokens,
        "avg_loss": counter["loss_sum"] / tokens,
        "share_of_all_routing_errors": routing_errors / max(total_route_errors, 1),
        "prediction_error_given_routing_error": counter["routing_and_prediction_errors"] / max(routing_errors, 1),
    }


@torch.no_grad()
def analyze_run(args, run_name, dataset, device):
    run_dir = os.path.join(args.checkpoint_root, run_name)
    runtime_config = load_runtime_config(run_dir)
    ckpt_path, step = find_checkpoint(run_dir, args.checkpoint_step)
    if ckpt_path is None:
        return {"run": run_name, "error": "no checkpoint found", "checkpoint_dir": run_dir}

    model = load_model(args, runtime_config, ckpt_path, device)
    model.eval()

    target_config = probe_runtime_config(args, runtime_config)
    ground_truth_mapping = build_ground_truth_expert_mapping(dataset, args, target_config)
    layer_stats = [
        defaultdict(init_counter)
        for _ in range(model.config.num_hidden_layers)
    ]
    position_stats = [
        defaultdict(init_counter)
        for _ in range(model.config.num_hidden_layers)
    ]

    for start in range(0, args.num_samples, args.batch_size):
        batch_items = [dataset[i] for i in range(start, min(start + args.batch_size, args.num_samples))]
        source = torch.stack([item[0] for item in batch_items]).to(device)
        target = torch.stack([item[1] for item in batch_items]).to(device)
        source_metadata = torch.stack([item[3] for item in batch_items]).to(device)
        target_metadata = torch.stack([dataset.get_metadata(start + idx)["unit_ids_by_layer"][1:] for idx in range(len(batch_items))]).to(device)
        target_expert_ids = make_ground_truth_expert_ids(source_metadata, ground_truth_mapping, device)

        outputs = model(
            source,
            output_expert_labels=True,
            use_cache=False,
        )
        token_loss = F.cross_entropy(
            outputs.logits.reshape(-1, outputs.logits.size(-1)),
            target.reshape(-1),
            reduction="none",
        ).reshape_as(target)
        pred_error = outputs.logits.argmax(dim=-1) != target
        masks = make_position_masks(source_metadata, target_metadata)

        for layer_idx, labels in enumerate(outputs.expert_labels):
            primary = primary_expert_labels(labels)
            route_error = primary != target_expert_ids
            for name, mask in masks.items():
                update_counter(layer_stats[layer_idx][name], mask, route_error, pred_error, token_loss)
            pos_mod = torch.arange(source.shape[1], device=device)[None, :].expand_as(source) % int(args.synthetic_block_size)
            for mod in range(int(args.synthetic_block_size)):
                update_counter(position_stats[layer_idx][f"position_mod_{mod}"], pos_mod == mod, route_error, pred_error, token_loss)

    layers = []
    for layer_idx in range(model.config.num_hidden_layers):
        total_errors = layer_stats[layer_idx]["all"]["routing_errors"]
        layers.append(
            {
                "layer": layer_idx,
                "categories": {
                    name: finalize_counter(counter, total_errors)
                    for name, counter in sorted(layer_stats[layer_idx].items())
                },
                "position_mod": {
                    name: finalize_counter(counter, total_errors)
                    for name, counter in sorted(position_stats[layer_idx].items())
                },
            }
        )

    return {
        "run": run_name,
        "checkpoint_step": step,
        "checkpoint_path": ckpt_path,
        "ground_truth_target": {
            "ground_truth_routing_strategy": args.ground_truth_routing_strategy,
            "ground_truth_routing_feature_layer": int(args.ground_truth_routing_feature_layer),
        },
        "layers": layers,
    }


def main():
    args = add_args()
    device = choose_device()
    dataset = build_dataset(args)
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
