import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from analyze_moe_variant_selectivity import (
    build_ground_truth_expert_mapping,
    find_checkpoint,
    load_model,
    load_runtime_config,
    make_ground_truth_expert_ids,
    runtime_get,
)
from utils import HierarchicalPatternData


DEFAULT_RUNS = ",".join(
    [
        "inverse-kv-zipf-baseline",
        "ground-truth-zipf-higher-hash",
        "inverse-kv-supervised-gate-zipf-high-hash",
    ]
)


def parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def add_args():
    parser = argparse.ArgumentParser(
        description="Offline linear probe for ground-truth expert-id separability in frozen model router inputs."
    )
    parser.add_argument("--config_dir", type=str, default="fdong/Qwen3-0.6B")
    parser.add_argument("--checkpoint_root", type=str, default="fdong/checkpoints")
    parser.add_argument("--runs", type=str, default=DEFAULT_RUNS)
    parser.add_argument("--checkpoint_step", type=int, default=5000)
    parser.add_argument("--output_path", type=str, default="fdong/experiments/offline_ground_truth_gate_probe.json")

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_train_samples", type=int, default=512)
    parser.add_argument("--num_test_samples", type=int, default=256)
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
    parser.add_argument(
        "--probe_input_source",
        type=str,
        choices=["actual_router_input", "hidden", "attention_output"],
        default="actual_router_input",
        help="Which frozen representation to probe. actual_router_input matches the model's MoE router input.",
    )

    parser.add_argument("--probe_epochs", type=int, default=120)
    parser.add_argument("--probe_lr", type=float, default=0.05)
    parser.add_argument("--probe_batch_size", type=int, default=4096)
    parser.add_argument("--probe_weight_decay", type=float, default=0.0)
    parser.add_argument("--max_probe_tokens", type=int, default=65536)
    parser.add_argument("--probe_seed", type=int, default=123)
    return parser.parse_args()


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataset(args, num_samples):
    return HierarchicalPatternData(
        max_seq_len=args.seq_len,
        num_samples=num_samples,
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


def ground_truth_probe_runtime_config(args, runtime_config):
    merged = dict(runtime_config)
    merged["ground_truth_routing_strategy"] = args.ground_truth_routing_strategy
    merged["ground_truth_routing_feature_layer"] = int(args.ground_truth_routing_feature_layer)
    merged["ground_truth_frequency_estimate_samples"] = int(args.ground_truth_frequency_estimate_samples)
    return merged


def should_use_ground_truth_dispatch(runtime_config):
    strategy = str(runtime_get(runtime_config, "ground_truth_routing_strategy", "none"))
    mode = str(runtime_get(runtime_config, "ground_truth_routing_mode", "dispatch"))
    return strategy != "none" and mode == "dispatch"


@torch.no_grad()
def collect_router_inputs(model, dataset, args, runtime_config, probe_runtime_config, start_index, num_samples, device):
    captured: Dict[int, List[torch.Tensor]] = {idx: [] for idx in range(model.config.num_hidden_layers)}

    def make_mlp_hook(layer_idx):
        def hook(_module, inputs, kwargs):
            if args.probe_input_source == "actual_router_input":
                states = kwargs.get("router_hidden_states")
                if states is None:
                    states = inputs[0]
            elif args.probe_input_source == "hidden":
                states = inputs[0]
            else:
                return
            captured[layer_idx].append(states.detach().cpu())

        return hook

    def make_attention_hook(layer_idx):
        def hook(_module, _inputs, output):
            if args.probe_input_source != "attention_output":
                return
            states = model.model.layers[layer_idx].post_attention_layernorm(output[0])
            captured[layer_idx].append(states.detach().cpu())

        return hook

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.register_forward_pre_hook(make_mlp_hook(layer_idx), with_kwargs=True))
        hooks.append(layer.self_attn.register_forward_hook(make_attention_hook(layer_idx)))

    labels = []
    ground_truth_mapping = build_ground_truth_expert_mapping(dataset, args, probe_runtime_config)
    model_ground_truth_mapping = build_ground_truth_expert_mapping(dataset, args, runtime_config)
    use_dispatch = should_use_ground_truth_dispatch(runtime_config)

    try:
        for start in range(start_index, start_index + num_samples, args.batch_size):
            items = [
                dataset[i]
                for i in range(start, min(start + args.batch_size, start_index + num_samples))
            ]
            source = torch.stack([item[0] for item in items]).to(device)
            metadata = torch.stack([item[3] for item in items]).to(device)
            target_ground_truth_ids = make_ground_truth_expert_ids(metadata, ground_truth_mapping, device)
            dispatch_ground_truth_ids = None
            if use_dispatch:
                dispatch_ground_truth_ids = make_ground_truth_expert_ids(metadata, model_ground_truth_mapping, device)
            model(
                source,
                use_cache=False,
                output_hidden_states=False,
                logits_to_keep=1,
                ground_truth_expert_ids=dispatch_ground_truth_ids,
            )
            labels.append(target_ground_truth_ids.detach().cpu())
    finally:
        for hook in hooks:
            hook.remove()

    features = {
        layer_idx: torch.cat(chunks, dim=0)
        for layer_idx, chunks in captured.items()
    }
    return features, torch.cat(labels, dim=0)


def flatten_tokens(tensor):
    return tensor.reshape(-1, tensor.shape[-1])


def flatten_labels(tensor):
    return tensor.reshape(-1)


def majority_accuracy(labels):
    valid = labels >= 0
    if valid.sum().item() == 0:
        return 0.0
    counts = torch.bincount(labels[valid].to(torch.long))
    return float(counts.max().item() / valid.sum().item())


def train_probe(train_x, train_y, test_x, test_y, num_classes, args):
    train_mask = train_y >= 0
    test_mask = test_y >= 0
    train_x = train_x[train_mask].to(torch.float32)
    train_y = train_y[train_mask].to(torch.long)
    test_x = test_x[test_mask].to(torch.float32)
    test_y = test_y[test_mask].to(torch.long)

    if args.max_probe_tokens > 0 and train_x.shape[0] > args.max_probe_tokens:
        generator = torch.Generator()
        generator.manual_seed(int(args.probe_seed))
        perm = torch.randperm(train_x.shape[0], generator=generator)[: args.max_probe_tokens]
        train_x = train_x[perm]
        train_y = train_y[perm]

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    torch.manual_seed(int(args.probe_seed))
    classifier = nn.Linear(train_x.shape[-1], num_classes)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=float(args.probe_lr),
        weight_decay=float(args.probe_weight_decay),
    )
    batch_size = min(int(args.probe_batch_size), train_x.shape[0])
    for _ in range(int(args.probe_epochs)):
        perm = torch.randperm(train_x.shape[0])
        for start in range(0, train_x.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            loss = F.cross_entropy(classifier(train_x[idx]), train_y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        train_logits = classifier(train_x)
        test_logits = classifier(test_x)
        train_pred = train_logits.argmax(dim=-1)
        test_pred = test_logits.argmax(dim=-1)
        return {
            "train_accuracy": float((train_pred == train_y).float().mean().item()),
            "test_accuracy": float((test_pred == test_y).float().mean().item()),
            "train_loss": float(F.cross_entropy(train_logits, train_y).item()),
            "test_loss": float(F.cross_entropy(test_logits, test_y).item()),
            "train_count": int(train_y.numel()),
            "test_count": int(test_y.numel()),
        }


def analyze_run(args, run_name, dataset, device):
    run_dir = os.path.join(args.checkpoint_root, run_name)
    runtime_config = load_runtime_config(run_dir)
    ckpt_path, step = find_checkpoint(run_dir, args.checkpoint_step)
    if ckpt_path is None:
        return {"run": run_name, "error": "no checkpoint found", "checkpoint_dir": run_dir}

    model = load_model(args, runtime_config, ckpt_path, device)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    probe_runtime_config = ground_truth_probe_runtime_config(args, runtime_config)
    train_features, train_labels = collect_router_inputs(
        model,
        dataset,
        args,
        runtime_config,
        probe_runtime_config,
        start_index=0,
        num_samples=args.num_train_samples,
        device=device,
    )
    test_features, test_labels = collect_router_inputs(
        model,
        dataset,
        args,
        runtime_config,
        probe_runtime_config,
        start_index=args.num_train_samples,
        num_samples=args.num_test_samples,
        device=device,
    )

    flat_train_labels = flatten_labels(train_labels)
    flat_test_labels = flatten_labels(test_labels)
    layer_results = []
    for layer_idx in range(model.config.num_hidden_layers):
        layer_results.append(
            {
                "layer": layer_idx,
                "router_input_shape": list(train_features[layer_idx].shape),
                "probe": train_probe(
                    flatten_tokens(train_features[layer_idx]),
                    flat_train_labels,
                    flatten_tokens(test_features[layer_idx]),
                    flat_test_labels,
                    int(runtime_config.get("moe_num_unique_experts", 4)),
                    args,
                ),
            }
        )

    return {
        "run": run_name,
        "checkpoint_step": step,
        "checkpoint_path": ckpt_path,
        "runtime_config": runtime_config,
        "ground_truth_probe_target": {
            "ground_truth_routing_strategy": args.ground_truth_routing_strategy,
            "ground_truth_routing_feature_layer": int(args.ground_truth_routing_feature_layer),
            "ground_truth_frequency_estimate_samples": int(args.ground_truth_frequency_estimate_samples),
        },
        "model_forward_uses_ground_truth_dispatch": should_use_ground_truth_dispatch(runtime_config),
        "majority_baseline": {
            "train": majority_accuracy(flat_train_labels),
            "test": majority_accuracy(flat_test_labels),
        },
        "layers": layer_results,
    }


def main():
    args = add_args()
    device = choose_device()
    dataset = build_dataset(args, args.num_train_samples + args.num_test_samples)
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
