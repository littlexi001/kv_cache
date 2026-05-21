import argparse
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from analyze_moe_variant_selectivity import (
    choose_device,
    find_checkpoint,
    load_model,
    load_runtime_config,
    parse_int_list,
)
from models.myqwen import MyQwen3MoE, apply_rotary_pos_emb, eager_attention_forward, repeat_kv
from utils import HierarchicalPatternData


DEFAULT_RUNS = ",".join(
    [
        "inverse-kv-supervised-gate-zipf-high-hash",
        "inverse-kv-mlp-gate-hidden-supervised-zipf-high-hash",
        "inverse-kv-mlp-gate-attention_output-supervised-zipf-high-hash",
    ]
)


def add_args():
    parser = argparse.ArgumentParser(description="Evaluate expert-indexed inverse KV attention.")
    parser.add_argument("--config_dir", type=str, default="fdong/Qwen3-0.6B")
    parser.add_argument("--checkpoint_root", type=str, default="fdong/checkpoints")
    parser.add_argument("--runs", type=str, default=DEFAULT_RUNS)
    parser.add_argument("--checkpoint_step", type=int, default=5000)
    parser.add_argument("--output_path", type=str, default="fdong/experiments/inverse_kv_indexing_step5000.json")
    parser.add_argument("--top_m_values", type=parse_int_list, default=[1, 2, 3, 4])

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=256)
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


def make_causal_mask(batch_size, seq_len, dtype, device):
    min_dtype = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_dtype, dtype=dtype, device=device)
    causal = torch.arange(seq_len, device=device)[None, :] <= torch.arange(seq_len, device=device)[:, None]
    mask = mask.masked_fill(causal, 0)
    return mask[None, None, :, :].expand(batch_size, 1, -1, -1)


def router_logits(moe, router_input):
    return moe.router(router_input.reshape(-1, router_input.shape[-1])).reshape(
        *router_input.shape[:-1],
        moe.num_unique_experts,
    )


def topm_from_logits(logits, top_m):
    return torch.topk(logits, k=min(int(top_m), logits.shape[-1]), dim=-1).indices


def topk_expert_labels(moe, router_input):
    logits = router_logits(moe, router_input)
    routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
    return torch.topk(routing_weights, k=moe.num_experts_per_tok, dim=-1).indices, logits


def apply_moe(layer, mlp_input, router_input):
    outputs = layer.mlp(
        mlp_input,
        output_expert_labels=True,
        router_hidden_states=router_input if layer.moe_router_input == "attention_output" else None,
    )
    return outputs[0], outputs[1]


@torch.no_grad()
def dense_reference(model, input_ids):
    model.eval()
    qwen = model.model
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    hidden_states = qwen.embed_tokens(input_ids)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    position_embeddings = qwen.rotary_emb(hidden_states, position_ids)
    causal_mask = make_causal_mask(batch_size, seq_len, hidden_states.dtype, device)

    layer_hidden_states = []
    expert_labels = []
    router_inputs = []
    attention_outputs = []

    for layer_idx, layer in enumerate(qwen.layers):
        residual_source_idx = qwen.residual_source_pattern[layer_idx]
        residual_source = hidden_states if residual_source_idx == -1 else layer_hidden_states[residual_source_idx]
        normed = layer.input_layernorm(hidden_states)
        attn_output, _, _ = layer.self_attn(
            hidden_states=normed,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        hidden_after_attn = residual_source + attn_output
        mlp_input = layer.post_attention_layernorm(hidden_after_attn)
        if not isinstance(layer.mlp, MyQwen3MoE):
            raise ValueError("This inverse-KV evaluator currently supports token-level MoE checkpoints only.")
        router_input = layer.post_attention_layernorm(attn_output) if layer.moe_router_input == "attention_output" else mlp_input
        mlp_output, labels = apply_moe(layer, mlp_input, router_input)
        hidden_states = hidden_after_attn + mlp_output

        layer_hidden_states.append(hidden_states)
        expert_labels.append(labels.detach())
        router_inputs.append(router_input.detach())
        attention_outputs.append(attn_output.detach())

    final_hidden = qwen.norm(hidden_states)
    logits = model.lm_head(final_hidden)
    return {
        "logits": logits,
        "expert_labels": expert_labels,
        "router_inputs": router_inputs,
        "attention_outputs": attention_outputs,
    }


def proxy_router_input(layer, residual_source, value_states):
    repeated_values = repeat_kv(value_states, layer.self_attn.num_key_value_groups)
    value_heads = repeated_values.transpose(1, 2).contiguous()
    value_flat = value_heads.reshape(value_heads.shape[0], value_heads.shape[1], -1)
    projected_value = layer.self_attn.o_proj(value_flat)
    if layer.moe_router_input == "attention_output":
        return layer.post_attention_layernorm(projected_value)
    return layer.post_attention_layernorm(residual_source + projected_value)


def build_inverse_kv_mask(key_expert_labels, query_topm, dtype):
    # key_expert_labels: [B, S, K], query_topm: [B, S, M]
    batch_size, seq_len, _ = key_expert_labels.shape
    device = key_expert_labels.device
    key_labels = key_expert_labels[:, None, :, :, None]
    query_labels = query_topm[:, :, None, None, :]
    expert_overlap = (key_labels == query_labels).any(dim=(-1, -2))
    causal = torch.arange(seq_len, device=device)[None, :] <= torch.arange(seq_len, device=device)[:, None]
    causal = causal[None, :, :]
    self_mask = torch.eye(seq_len, dtype=torch.bool, device=device)[None, :, :]
    allowed = (expert_overlap & causal) | self_mask
    min_dtype = torch.finfo(dtype).min
    additive = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=dtype, device=device)
    additive = additive.masked_fill(~allowed[:, None, :, :], min_dtype)
    history = torch.arange(seq_len, device=device)[None, :] < torch.arange(seq_len, device=device)[:, None]
    history = history[None, :, :]
    selected_history = (expert_overlap & history).sum().item()
    total_history = history.expand(batch_size, -1, -1).sum().item()
    return additive, float(selected_history / max(total_history, 1))


def topm_overlap_metrics(proxy_topm, true_topm):
    proxy = proxy_topm[:, :, :, None]
    true = true_topm[:, :, None, :]
    intersection = (proxy == true).any(dim=-1).sum(dim=-1).to(torch.float32)
    union = proxy_topm.shape[-1] + true_topm.shape[-1] - intersection
    return {
        "any_overlap": float((intersection > 0).float().mean().item()),
        "mean_jaccard": float((intersection / union.clamp_min(1)).mean().item()),
        "exact_set_match": float(
            (
                torch.sort(proxy_topm, dim=-1).values == torch.sort(true_topm, dim=-1).values
            ).all(dim=-1).float().mean().item()
        ),
    }


def cosine_mean(a, b):
    return float(F.cosine_similarity(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), dim=-1).mean().item())


@torch.no_grad()
def inverse_forward(model, input_ids, dense_ref, mode, top_m):
    qwen = model.model
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    hidden_states = qwen.embed_tokens(input_ids)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    position_embeddings = qwen.rotary_emb(hidden_states, position_ids)
    layer_hidden_states = []
    metrics = {
        "kv_selected_fraction_sum": 0.0,
        "proxy_true_cosine_sum": 0.0,
        "attention_output_cosine_sum": 0.0,
        "attention_output_mse_sum": 0.0,
        "topm_any_overlap_sum": 0.0,
        "topm_jaccard_sum": 0.0,
        "topm_exact_match_sum": 0.0,
        "layers": 0,
    }

    for layer_idx, layer in enumerate(qwen.layers):
        residual_source_idx = qwen.residual_source_pattern[layer_idx]
        residual_source = hidden_states if residual_source_idx == -1 else layer_hidden_states[residual_source_idx]
        normed = layer.input_layernorm(hidden_states)
        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_states = layer.self_attn.q_norm(layer.self_attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
        key_states = layer.self_attn.k_norm(layer.self_attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
        value_states = layer.self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        true_logits = router_logits(layer.mlp, dense_ref["router_inputs"][layer_idx])
        true_topm = topm_from_logits(true_logits, top_m)
        if mode == "true_router":
            query_topm = true_topm
            proxy_input = dense_ref["router_inputs"][layer_idx]
        elif mode == "early_proxy":
            proxy_input = proxy_router_input(layer, residual_source, value_states)
            proxy_logits = router_logits(layer.mlp, proxy_input)
            query_topm = topm_from_logits(proxy_logits, top_m)
            overlap = topm_overlap_metrics(query_topm, true_topm)
            metrics["topm_any_overlap_sum"] += overlap["any_overlap"]
            metrics["topm_jaccard_sum"] += overlap["mean_jaccard"]
            metrics["topm_exact_match_sum"] += overlap["exact_set_match"]
        else:
            raise ValueError(f"Unknown inverse mode: {mode}")

        mask, selected_fraction = build_inverse_kv_mask(
            dense_ref["expert_labels"][layer_idx],
            query_topm,
            dtype=hidden_states.dtype,
        )
        head_attn_output, _ = eager_attention_forward(
            layer.self_attn,
            query_states,
            key_states,
            value_states,
            mask,
            scaling=layer.self_attn.scaling,
        )
        attn_output = head_attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = layer.self_attn.o_proj(attn_output)
        hidden_after_attn = residual_source + attn_output
        mlp_input = layer.post_attention_layernorm(hidden_after_attn)
        router_input = layer.post_attention_layernorm(attn_output) if layer.moe_router_input == "attention_output" else mlp_input
        mlp_output, _ = apply_moe(layer, mlp_input, router_input)
        hidden_states = hidden_after_attn + mlp_output
        layer_hidden_states.append(hidden_states)

        metrics["kv_selected_fraction_sum"] += selected_fraction
        metrics["proxy_true_cosine_sum"] += cosine_mean(proxy_input, dense_ref["router_inputs"][layer_idx])
        metrics["attention_output_cosine_sum"] += cosine_mean(attn_output, dense_ref["attention_outputs"][layer_idx])
        metrics["attention_output_mse_sum"] += float(F.mse_loss(attn_output, dense_ref["attention_outputs"][layer_idx]).item())
        metrics["layers"] += 1

    final_hidden = qwen.norm(hidden_states)
    logits = model.lm_head(final_hidden)
    layer_count = max(metrics.pop("layers"), 1)
    return logits, {
        "mean_kv_selected_fraction": metrics["kv_selected_fraction_sum"] / layer_count,
        "mean_proxy_true_router_input_cosine": metrics["proxy_true_cosine_sum"] / layer_count,
        "mean_attention_output_cosine_vs_dense": metrics["attention_output_cosine_sum"] / layer_count,
        "mean_attention_output_mse_vs_dense": metrics["attention_output_mse_sum"] / layer_count,
        "mean_topm_any_overlap_with_true": metrics["topm_any_overlap_sum"] / layer_count if mode == "early_proxy" else 1.0,
        "mean_topm_jaccard_with_true": metrics["topm_jaccard_sum"] / layer_count if mode == "early_proxy" else 1.0,
        "mean_topm_exact_match_with_true": metrics["topm_exact_match_sum"] / layer_count if mode == "early_proxy" else 1.0,
    }


def init_eval():
    return {"loss_sum": 0.0, "correct": 0, "count": 0}


def update_eval(state, logits, target):
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none").reshape_as(target)
    pred = logits.argmax(dim=-1)
    state["loss_sum"] += float(loss.sum().item())
    state["correct"] += int((pred == target).sum().item())
    state["count"] += int(target.numel())


def finalize_eval(state):
    count = max(state["count"], 1)
    return {
        "loss": state["loss_sum"] / count,
        "accuracy": state["correct"] / count,
        "count": state["count"],
    }


def init_metric_state():
    return defaultdict(float)


def update_metric_state(state, metrics):
    state["batches"] += 1
    for key, value in metrics.items():
        state[key] += float(value)


def finalize_metric_state(state):
    batches = max(int(state["batches"]), 1)
    return {key: value / batches for key, value in state.items() if key != "batches"}


@torch.no_grad()
def analyze_run(args, run_name, dataset, device):
    run_dir = os.path.join(args.checkpoint_root, run_name)
    runtime_config = load_runtime_config(run_dir)
    ckpt_path, step = find_checkpoint(run_dir, args.checkpoint_step)
    if ckpt_path is None:
        return {"run": run_name, "error": "no checkpoint found", "checkpoint_dir": run_dir}
    model = load_model(args, runtime_config, ckpt_path, device)
    model.eval()

    full_eval = init_eval()
    inverse_eval = {}
    inverse_metrics = {}
    for top_m in args.top_m_values:
        for mode in ("true_router", "early_proxy"):
            key = f"{mode}_top{top_m}"
            inverse_eval[key] = init_eval()
            inverse_metrics[key] = init_metric_state()

    for start in range(0, args.num_samples, args.batch_size):
        batch_items = [dataset[i] for i in range(start, min(start + args.batch_size, args.num_samples))]
        source = torch.stack([item[0] for item in batch_items]).to(device)
        target = torch.stack([item[1] for item in batch_items]).to(device)
        dense_ref = dense_reference(model, source)
        update_eval(full_eval, dense_ref["logits"], target)
        for top_m in args.top_m_values:
            for mode in ("true_router", "early_proxy"):
                key = f"{mode}_top{top_m}"
                logits, metrics = inverse_forward(model, source, dense_ref, mode, top_m)
                update_eval(inverse_eval[key], logits, target)
                update_metric_state(inverse_metrics[key], metrics)

    return {
        "run": run_name,
        "checkpoint_step": step,
        "checkpoint_path": ckpt_path,
        "runtime_config": runtime_config,
        "full_attention": finalize_eval(full_eval),
        "inverse_kv": {
            key: {
                **finalize_eval(inverse_eval[key]),
                **finalize_metric_state(inverse_metrics[key]),
            }
            for key in sorted(inverse_eval.keys())
        },
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
