import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig

from models import MyQwen3ForCausalLM
from utils import HierarchicalPatternData


def parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def add_args():
    parser = argparse.ArgumentParser(description="Linear probes for feature readability in attention outputs.")
    parser.add_argument("--config_dir", type=str, default="../Qwen3-0.6B")
    parser.add_argument("--ckpt_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="../experiments/attention_output_probe.json")

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_train_samples", type=int, default=256)
    parser.add_argument("--num_test_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--synthetic_block_size", type=int, default=4)
    parser.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    parser.add_argument("--synthetic_content_token_count", type=int, default=256)
    parser.add_argument("--synthetic_num_units_per_layer", type=int, default=64)
    parser.add_argument("--synthetic_seed", type=int, default=0)
    parser.add_argument("--synthetic_min_token_id", type=int, default=1)
    parser.add_argument("--synthetic_sampling_distribution", type=str, default="uniform", choices=["uniform", "zipf"])
    parser.add_argument("--synthetic_zipf_alpha", type=float, default=1.0)

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
    parser.add_argument("--moe_router_input", type=str, default="hidden", choices=["hidden", "attention_output"])
    parser.add_argument("--moe_head_level", action="store_true", default=False)

    parser.add_argument("--probe_epochs", type=int, default=80)
    parser.add_argument("--probe_lr", type=float, default=0.05)
    parser.add_argument("--probe_batch_size", type=int, default=4096)
    parser.add_argument("--probe_weight_decay", type=float, default=0.0)
    parser.add_argument("--max_probe_tokens", type=int, default=32768)
    parser.add_argument("--probe_seed", type=int, default=123)
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
    config.moe_router_input = args.moe_router_input
    config.moe_head_level = bool(args.moe_head_level)
    return config


def load_model(args, device):
    config = build_config(args)
    model = MyQwen3ForCausalLM(config).to(device)
    state = torch.load(args.ckpt_file, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def build_dataset(args, num_samples, offset):
    return HierarchicalPatternData(
        max_seq_len=args.seq_len,
        num_samples=num_samples + offset,
        block_size=args.synthetic_block_size,
        num_hierarchy_layers=args.synthetic_num_hierarchy_layers,
        content_token_count=args.synthetic_content_token_count,
        num_units_per_layer=args.synthetic_num_units_per_layer,
        seed=args.synthetic_seed,
        min_token_id=args.synthetic_min_token_id,
        sampling_distribution=args.synthetic_sampling_distribution,
        zipf_alpha=args.synthetic_zipf_alpha,
        return_metadata=True,
    )


def majority_accuracy(labels: torch.Tensor) -> float:
    valid = labels >= 0
    if valid.sum().item() == 0:
        return 0.0
    counts = torch.bincount(labels[valid].to(torch.long))
    return float(counts.max().item() / valid.sum().item())


def train_linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    args,
):
    train_mask = train_y >= 0
    test_mask = test_y >= 0
    train_x = train_x[train_mask].to(torch.float32)
    train_y = train_y[train_mask].to(torch.long)
    test_x = test_x[test_mask].to(torch.float32)
    test_y = test_y[test_mask].to(torch.long)

    if args.max_probe_tokens > 0 and train_x.shape[0] > args.max_probe_tokens:
        generator = torch.Generator(device=train_x.device)
        generator.manual_seed(int(args.probe_seed))
        perm = torch.randperm(train_x.shape[0], generator=generator, device=train_x.device)[: args.max_probe_tokens]
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
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
    )
    batch_size = min(args.probe_batch_size, train_x.shape[0])
    for _ in range(args.probe_epochs):
        perm = torch.randperm(train_x.shape[0])
        for start in range(0, train_x.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            logits = classifier(train_x[idx])
            loss = nn.functional.cross_entropy(logits, train_y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        train_pred = classifier(train_x).argmax(dim=-1)
        test_pred = classifier(test_x).argmax(dim=-1)
        train_acc = float((train_pred == train_y).float().mean().item())
        test_acc = float((test_pred == test_y).float().mean().item())
    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_count": int(train_y.numel()),
        "test_count": int(test_y.numel()),
        "num_classes": int(num_classes),
    }


@torch.no_grad()
def collect_features(model, dataset, start_index, num_samples, batch_size, device):
    captured: Dict[int, Dict[str, List[torch.Tensor]]] = {
        layer_idx: {"attn_output": [], "head_attn_output": []}
        for layer_idx in range(model.config.num_hidden_layers)
    }

    def make_hook(layer_idx):
        def hook(_module, _inputs, output):
            captured[layer_idx]["attn_output"].append(output[0].detach().cpu())
            captured[layer_idx]["head_attn_output"].append(output[2].detach().cpu())
        return hook

    hooks = [
        layer.self_attn.register_forward_hook(make_hook(layer_idx))
        for layer_idx, layer in enumerate(model.model.layers)
    ]

    hidden_inputs = [[] for _ in range(model.config.num_hidden_layers)]
    hidden_outputs = [[] for _ in range(model.config.num_hidden_layers)]
    labels_by_layer = [[] for _ in range(dataset.num_hierarchy_layers)]

    try:
        for start in range(start_index, start_index + num_samples, batch_size):
            batch_items = [
                dataset[i]
                for i in range(start, min(start + batch_size, start_index + num_samples))
            ]
            source = torch.stack([item[0] for item in batch_items]).to(device)
            metadata = torch.stack([item[3] for item in batch_items])

            outputs = model(
                source,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                logits_to_keep=1,
            )
            hidden_states = outputs.hidden_states
            for layer_idx in range(model.config.num_hidden_layers):
                hidden_inputs[layer_idx].append(hidden_states[layer_idx].detach().cpu())
                hidden_outputs[layer_idx].append(hidden_states[layer_idx + 1].detach().cpu())
            for feature_layer in range(dataset.num_hierarchy_layers):
                labels_by_layer[feature_layer].append(metadata[:, :, feature_layer].detach().cpu())
    finally:
        for hook in hooks:
            hook.remove()

    features = {}
    for layer_idx in range(model.config.num_hidden_layers):
        layer_features = {
            "layer_input": torch.cat(hidden_inputs[layer_idx], dim=0),
            "layer_output": torch.cat(hidden_outputs[layer_idx], dim=0),
            "attn_output_wo_residual": torch.cat(captured[layer_idx]["attn_output"], dim=0),
        }
        head_outputs = torch.cat(captured[layer_idx]["head_attn_output"], dim=0)
        layer_features["head_attn_output_flat"] = head_outputs.reshape(
            head_outputs.shape[0],
            head_outputs.shape[1],
            -1,
        )
        for head_idx in range(head_outputs.shape[2]):
            layer_features[f"head_{head_idx}_attn_output"] = head_outputs[:, :, head_idx, :]
        features[layer_idx] = layer_features

    labels = {
        feature_layer: torch.cat(values, dim=0)
        for feature_layer, values in enumerate(labels_by_layer)
    }
    return features, labels


def flatten_tokens(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1, tensor.shape[-1])


def main():
    args = add_args()
    device = choose_device()
    model = load_model(args, device)
    dataset = build_dataset(args, args.num_train_samples + args.num_test_samples, 0)

    train_features, train_labels = collect_features(
        model,
        dataset,
        start_index=0,
        num_samples=args.num_train_samples,
        batch_size=args.batch_size,
        device=device,
    )
    test_features, test_labels = collect_features(
        model,
        dataset,
        start_index=args.num_train_samples,
        num_samples=args.num_test_samples,
        batch_size=args.batch_size,
        device=device,
    )

    feature_targets = {
        "local_slot": 0,
        "higher_unit": 1 if args.synthetic_num_hierarchy_layers > 1 else 0,
    }
    summary = {
        "config": vars(args),
        "device": str(device),
        "majority_baselines": {},
        "layers": [],
    }

    for target_name, feature_layer in feature_targets.items():
        summary["majority_baselines"][target_name] = {
            "train": majority_accuracy(train_labels[feature_layer].reshape(-1)),
            "test": majority_accuracy(test_labels[feature_layer].reshape(-1)),
        }

    for layer_idx in range(model.config.num_hidden_layers):
        layer_row = {"layer": layer_idx, "probes": {}}
        for source_name, train_tensor in train_features[layer_idx].items():
            test_tensor = test_features[layer_idx][source_name]
            source_result = {}
            for target_name, feature_layer in feature_targets.items():
                num_classes = int(args.synthetic_num_units_per_layer)
                source_result[target_name] = train_linear_probe(
                    flatten_tokens(train_tensor),
                    train_labels[feature_layer].reshape(-1),
                    flatten_tokens(test_tensor),
                    test_labels[feature_layer].reshape(-1),
                    num_classes,
                    args,
                )
            layer_row["probes"][source_name] = source_result
        summary["layers"].append(layer_row)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
