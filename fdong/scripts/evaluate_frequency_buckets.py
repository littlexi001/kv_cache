import argparse
import json
import math
import os
import re

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from models import MyQwen3ForCausalLM
from utils import HierarchicalPatternData


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate synthetic NTP loss by training-frequency bucket.")
    parser.add_argument("--config_dir", type=str, default="../Qwen3-0.6B")
    parser.add_argument("--checkpoint_root", type=str, default="../checkpoints")
    parser.add_argument("--runs", type=str, required=True)
    parser.add_argument("--checkpoint_step", type=int, default=1000)
    parser.add_argument("--output_path", type=str, default="../experiments/frequency_bucket_eval.json")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--synthetic_block_size", type=int, default=4)
    parser.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    parser.add_argument("--synthetic_content_token_count", type=int, default=256)
    parser.add_argument("--synthetic_num_units_per_layer", type=int, default=64)
    parser.add_argument("--synthetic_seed", type=int, default=0)
    parser.add_argument("--synthetic_min_token_id", type=int, default=1)
    parser.add_argument("--train_sampling_distribution", choices=["uniform", "zipf"], default="zipf")
    parser.add_argument("--eval_sampling_distribution", choices=["uniform", "zipf"], default="uniform")
    parser.add_argument("--synthetic_zipf_alpha", type=float, default=1.3)
    parser.add_argument("--synthetic_zipf_shuffle_ranks", action="store_true", default=True)
    parser.add_argument("--synthetic_no_zipf_shuffle_ranks", action="store_false", dest="synthetic_zipf_shuffle_ranks")
    parser.add_argument("--frequency_feature_layer", type=int, default=1)
    parser.add_argument("--head_fraction", type=float, default=0.2)
    parser.add_argument("--tail_fraction", type=float, default=0.4)
    return parser.parse_args()


def load_runtime_config(run_dir):
    path = os.path.join(run_dir, "runtime_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_debug_config_from_state(config, state):
    embed = state["model.embed_tokens.weight"]
    config.vocab_size = int(embed.shape[0])
    config.hidden_size = int(embed.shape[1])
    layer_ids = []
    for key in state:
        match = re.match(r"model\.layers\.(\d+)\.", key)
        if match:
            layer_ids.append(int(match.group(1)))
    config.num_hidden_layers = max(layer_ids) + 1

    q_proj = state["model.layers.0.self_attn.q_proj.weight"]
    k_proj = state["model.layers.0.self_attn.k_proj.weight"]
    q_norm = state["model.layers.0.self_attn.q_norm.weight"]
    config.head_dim = int(q_norm.shape[0])
    config.num_attention_heads = int(q_proj.shape[0] // config.head_dim)
    config.num_key_value_heads = int(k_proj.shape[0] // config.head_dim)
    if "model.layers.0.mlp.gate_proj.weight" in state:
        config.intermediate_size = int(state["model.layers.0.mlp.gate_proj.weight"].shape[0])
    config.max_position_embeddings = max(int(getattr(config, "max_position_embeddings", 0)), 256)


def apply_runtime_config(config, runtime_config):
    config._attn_implementation = "eager"
    config.use_cache = False
    config.attention_stride_pattern = runtime_config.get("attention_stride_pattern", [1] * config.num_hidden_layers)
    config.residual_source_pattern = runtime_config.get("residual_source_pattern", [-1] * config.num_hidden_layers)
    config.use_moe = bool(runtime_config.get("use_moe", False))
    config.moe_num_unique_experts = int(runtime_config.get("moe_num_unique_experts", 4))
    config.moe_num_experts_per_tok = int(runtime_config.get("moe_num_experts_per_tok", 1))
    config.moe_intermediate_size = int(runtime_config.get("moe_intermediate_size", config.intermediate_size))
    config.moe_use_common_expert = bool(runtime_config.get("moe_use_common_expert", False))
    config.moe_common_intermediate_size = int(runtime_config.get("moe_common_intermediate_size", config.moe_intermediate_size))
    config.moe_router_bias = bool(runtime_config.get("moe_router_bias", False))
    config.moe_router_type = str(runtime_config.get("moe_router_type", "linear"))
    config.moe_router_hidden_size = int(runtime_config.get("moe_router_hidden_size", config.hidden_size))
    config.moe_router_act = str(runtime_config.get("moe_router_act", "silu"))
    config.moe_normalize_topk_prob = bool(runtime_config.get("moe_normalize_topk_prob", True))
    config.moe_router_input = str(runtime_config.get("moe_router_input", "hidden"))
    config.moe_router_input_pos = str(runtime_config.get("moe_router_input_pos", config.moe_router_input))
    config.moe_router_input_shape = str(runtime_config.get("moe_router_input_shape", "full"))
    config.moe_expert_input_pos = str(runtime_config.get("moe_expert_input_pos", "attention_output_residual"))
    config.moe_expert_input_shape = str(runtime_config.get("moe_expert_input_shape", "full"))
    config.moe_head_level = bool(runtime_config.get("moe_head_level", False))
    config.use_pre_router = bool(runtime_config.get("use_pre_router", False))
    config.pre_router_input = str(runtime_config.get("pre_router_input", "layer_input"))
    config.pre_router_controls_attention = bool(runtime_config.get("pre_router_controls_attention", False))


def make_dataset(args, sampling_distribution):
    return HierarchicalPatternData(
        max_seq_len=args.seq_len,
        num_samples=args.num_samples,
        block_size=args.synthetic_block_size,
        num_hierarchy_layers=args.synthetic_num_hierarchy_layers,
        content_token_count=args.synthetic_content_token_count,
        num_units_per_layer=args.synthetic_num_units_per_layer,
        seed=args.synthetic_seed,
        min_token_id=args.synthetic_min_token_id,
        sampling_distribution=sampling_distribution,
        zipf_alpha=args.synthetic_zipf_alpha,
        zipf_shuffle_ranks=args.synthetic_zipf_shuffle_ranks,
        return_metadata=True,
    )


def frequency_scores(args):
    if args.frequency_feature_layer != args.synthetic_num_hierarchy_layers - 1:
        return None
    train_dataset = make_dataset(args, args.train_sampling_distribution)
    if train_dataset.top_unit_sample_weights is None:
        return torch.ones(args.synthetic_num_units_per_layer, dtype=torch.float64)
    return torch.tensor(train_dataset.top_unit_sample_weights, dtype=torch.float64)


def make_bucket_ids(args):
    scores = frequency_scores(args)
    if scores is None:
        raise ValueError("This evaluator currently buckets the top hierarchy layer; use --frequency_feature_layer 1.")
    num_units = int(scores.numel())
    head_n = max(1, int(round(num_units * args.head_fraction)))
    tail_n = max(1, int(round(num_units * args.tail_fraction)))
    if head_n + tail_n >= num_units:
        raise ValueError("head_fraction + tail_fraction must leave at least one middle unit.")

    sorted_ids = torch.argsort(scores, descending=True)
    bucket_ids = torch.full((num_units,), 1, dtype=torch.long)
    bucket_ids[sorted_ids[:head_n]] = 0
    bucket_ids[sorted_ids[-tail_n:]] = 2
    bucket_names = ["head", "middle", "tail"]
    bucket_meta = {}
    for idx, name in enumerate(bucket_names):
        ids = torch.nonzero(bucket_ids == idx, as_tuple=False).flatten()
        bucket_meta[name] = {
            "num_units": int(ids.numel()),
            "unit_ids": [int(x) for x in ids.tolist()],
            "score_sum": float(scores[ids].sum().item()),
            "score_mean": float(scores[ids].mean().item()),
        }
    return bucket_ids, bucket_meta


def new_counter():
    return {"loss_sum": 0.0, "correct": 0, "count": 0}


def update(counter, loss, pred, target, mask):
    if mask.sum().item() == 0:
        return
    counter["loss_sum"] += float(loss[mask].sum().item())
    counter["correct"] += int((pred[mask] == target[mask]).sum().item())
    counter["count"] += int(mask.sum().item())


def finish(counter):
    if counter["count"] == 0:
        return {"loss": None, "perplexity": None, "accuracy": None, "count": 0}
    loss = counter["loss_sum"] / counter["count"]
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 50.0)),
        "accuracy": counter["correct"] / counter["count"],
        "count": counter["count"],
    }


@torch.no_grad()
def evaluate_run(args, run, bucket_ids, bucket_meta):
    run_dir = os.path.join(args.checkpoint_root, run)
    ckpt_path = os.path.join(run_dir, f"{args.checkpoint_step}.pth")
    state = torch.load(ckpt_path, map_location="cpu")
    runtime_config = load_runtime_config(run_dir)
    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    infer_debug_config_from_state(config, state)
    apply_runtime_config(config, runtime_config)
    model = MyQwen3ForCausalLM(config).eval()
    model.load_state_dict(state, strict=True)

    dataset = make_dataset(args, args.eval_sampling_distribution)
    counters = {
        "overall": new_counter(),
        "inside_high": new_counter(),
        "high_boundary": new_counter(),
    }
    for bucket in bucket_meta:
        counters[f"{bucket}_all"] = new_counter()
        counters[f"{bucket}_inside_high"] = new_counter()
        counters[f"{bucket}_high_boundary"] = new_counter()

    for start in range(0, args.num_samples, args.batch_size):
        end = min(start + args.batch_size, args.num_samples)
        items = [dataset[idx] for idx in range(start, end)]
        source = torch.stack([item[0] for item in items])
        target = torch.stack([item[1] for item in items])
        source_meta = torch.stack([item[3] for item in items])
        target_meta = torch.stack([dataset.get_metadata(idx)["unit_ids_by_layer"][1:] for idx in range(start, end)])

        logits = model(source).logits
        per_token_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none")
        per_token_loss = per_token_loss.reshape_as(target)
        pred = logits.argmax(dim=-1)
        valid = target != 0

        source_feature = source_meta[:, :, args.frequency_feature_layer]
        target_feature = target_meta[:, :, args.frequency_feature_layer]
        high_same = (source_feature == target_feature) & (source_feature >= 0)
        high_boundary = ~high_same
        safe_target_feature = target_feature.clamp_min(0)
        target_bucket = bucket_ids[safe_target_feature]

        update(counters["overall"], per_token_loss, pred, target, valid)
        update(counters["inside_high"], per_token_loss, pred, target, valid & high_same)
        update(counters["high_boundary"], per_token_loss, pred, target, valid & high_boundary)
        for bucket_idx, bucket_name in enumerate(bucket_meta):
            bucket_mask = target_feature.ge(0) & (target_bucket == bucket_idx)
            update(counters[f"{bucket_name}_all"], per_token_loss, pred, target, valid & bucket_mask)
            update(counters[f"{bucket_name}_inside_high"], per_token_loss, pred, target, valid & bucket_mask & high_same)
            update(counters[f"{bucket_name}_high_boundary"], per_token_loss, pred, target, valid & bucket_mask & high_boundary)

    return {
        "run": run,
        "checkpoint": ckpt_path,
        "model": {
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "head_dim": config.head_dim,
            "use_moe": bool(getattr(config, "use_moe", False)),
        },
        "metrics": {name: finish(counter) for name, counter in counters.items()},
    }


def main():
    args = parse_args()
    bucket_ids, bucket_meta = make_bucket_ids(args)
    runs = [item.strip() for item in args.runs.split(",") if item.strip()]
    results = {
        "config": vars(args),
        "bucket_meta": bucket_meta,
        "runs": [evaluate_run(args, run, bucket_ids, bucket_meta) for run in runs],
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    for row in results["runs"]:
        metrics = row["metrics"]
        parts = [
            row["run"],
            f"overall_loss={metrics['overall']['loss']:.4f}",
            f"overall_acc={metrics['overall']['accuracy']:.4f}",
        ]
        for bucket in bucket_meta:
            all_metrics = metrics[f"{bucket}_all"]
            boundary_metrics = metrics[f"{bucket}_high_boundary"]
            parts.append(f"{bucket}_loss={all_metrics['loss']:.4f}")
            parts.append(f"{bucket}_acc={all_metrics['accuracy']:.4f}")
            parts.append(f"{bucket}_boundary_loss={boundary_metrics['loss']:.4f}")
            parts.append(f"{bucket}_boundary_acc={boundary_metrics['accuracy']:.4f}")
        print(" ".join(parts), flush=True)


if __name__ == "__main__":
    main()
