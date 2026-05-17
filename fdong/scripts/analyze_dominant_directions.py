import argparse
import json
import math
import os
import re
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from models import MyQwen3ForCausalLM
from utils import HierarchicalPatternData


DEFAULT_RUN_SPECS = ",".join(
    [
        "uniform_baseline:inverse-kv-local-h128-l3-top1:uniform",
        "zipf_baseline:inverse-kv-zipf-baseline:zipf",
    ]
)

ATTN_RE = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$")
EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(.+)\.(gate_proj|up_proj|down_proj)\.weight$")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze what dominant singular directions align with.")
    parser.add_argument("--config_dir", type=str, default="../Qwen3-0.6B")
    parser.add_argument("--checkpoint_root", type=str, default="../checkpoints")
    parser.add_argument("--run_specs", type=str, default=DEFAULT_RUN_SPECS)
    parser.add_argument("--checkpoint_step", type=int, default=5000)
    parser.add_argument("--output_path", type=str, default="../experiments/dominant_directions_baselines_step5000.json")

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--synthetic_block_size", type=int, default=4)
    parser.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    parser.add_argument("--synthetic_content_token_count", type=int, default=256)
    parser.add_argument("--synthetic_num_units_per_layer", type=int, default=64)
    parser.add_argument("--synthetic_seed", type=int, default=0)
    parser.add_argument("--synthetic_min_token_id", type=int, default=1)
    parser.add_argument("--synthetic_zipf_alpha", type=float, default=1.1)
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
    parser.add_argument("--top_tokens", type=int, default=8)
    return parser.parse_args()


def parse_run_specs(text: str) -> List[Tuple[str, str, str]]:
    specs = []
    for item in text.split(","):
        if not item.strip():
            continue
        alias, run_name, distribution = item.split(":")
        specs.append((alias, run_name, distribution))
    return specs


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_checkpoint(run_dir: str, preferred_step: int):
    candidates = []
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


def load_runtime_config(run_dir: str):
    path = os.path.join(run_dir, "runtime_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    config.attention_stride_pattern = runtime_config.get("attention_stride_pattern", [1] * config.num_hidden_layers)
    config.residual_source_pattern = runtime_config.get("residual_source_pattern", [-1] * config.num_hidden_layers)
    config.use_moe = bool(runtime_config.get("use_moe", True))
    config.moe_num_unique_experts = int(runtime_config.get("moe_num_unique_experts", 4))
    config.moe_num_experts_per_tok = int(runtime_config.get("moe_num_experts_per_tok", 1))
    config.moe_intermediate_size = int(runtime_config.get("moe_intermediate_size", 128))
    config.moe_use_common_expert = bool(runtime_config.get("moe_use_common_expert", False))
    config.moe_common_intermediate_size = int(runtime_config.get("moe_common_intermediate_size", 128))
    config.moe_router_bias = bool(runtime_config.get("moe_router_bias", False))
    config.moe_normalize_topk_prob = bool(runtime_config.get("moe_normalize_topk_prob", True))
    config.moe_router_input = str(runtime_config.get("moe_router_input", "hidden"))
    config.moe_head_level = bool(runtime_config.get("moe_head_level", False))
    return config


def build_dataset(args, distribution):
    return HierarchicalPatternData(
        max_seq_len=args.seq_len,
        num_samples=args.num_samples,
        block_size=args.synthetic_block_size,
        num_hierarchy_layers=args.synthetic_num_hierarchy_layers,
        content_token_count=args.synthetic_content_token_count,
        num_units_per_layer=args.synthetic_num_units_per_layer,
        seed=args.synthetic_seed,
        min_token_id=args.synthetic_min_token_id,
        sampling_distribution=distribution,
        zipf_alpha=args.synthetic_zipf_alpha,
        zipf_shuffle_ranks=args.synthetic_zipf_shuffle_ranks,
        return_metadata=True,
    )


def unit(x):
    x = x.float()
    return x / torch.linalg.vector_norm(x).clamp_min(1e-12)


def abs_cos(a, b):
    return float(torch.abs((unit(a) * unit(b)).sum()).item())


def signed_cos(a, b):
    return float((unit(a) * unit(b)).sum().item())


def top_token_matches(direction, embedding, token_ids, k):
    direction = unit(direction)
    emb = F.normalize(embedding[token_ids].float(), dim=-1)
    sims = emb @ direction
    abs_sims = sims.abs()
    values, indices = torch.topk(abs_sims, k=min(k, len(token_ids)))
    rows = []
    for value, idx in zip(values.tolist(), indices.tolist()):
        token_id = int(token_ids[idx])
        rows.append(
            {
                "token_id": token_id,
                "abs_cos": float(value),
                "signed_cos": float(sims[idx].item()),
            }
        )
    return rows


def raw_and_centered_top_directions(x):
    x = x.float()
    raw_u, raw_s, raw_vh = torch.linalg.svd(x, full_matrices=False)
    centered = x - x.mean(dim=0, keepdim=True)
    centered_u, centered_s, centered_vh = torch.linalg.svd(centered, full_matrices=False)
    raw_energy = raw_s.square()
    centered_energy = centered_s.square()
    return {
        "raw_v1": raw_vh[0],
        "centered_v1": centered_vh[0],
        "raw_top1_energy": float(raw_energy[0].item() / raw_energy.sum().clamp_min(1e-12).item()),
        "centered_top1_energy": float(centered_energy[0].item() / centered_energy.sum().clamp_min(1e-12).item()),
        "mean": x.mean(dim=0),
    }


@torch.no_grad()
def collect_hidden_states(args, model, dataset, device):
    chunks = None
    for start in range(0, len(dataset), args.batch_size):
        batch = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
        source = torch.stack([item[0] for item in batch]).to(device)
        output = model(source, output_hidden_states=True)
        if chunks is None:
            chunks = [[] for _ in output.hidden_states]
        for idx, states in enumerate(output.hidden_states):
            chunks[idx].append(states.detach().float().cpu().reshape(-1, states.shape[-1]))
    return [torch.cat(items, dim=0) for items in chunks]


def representation_name(index: int, num_reps: int):
    if index == 0:
        return "embedding"
    if index == num_reps - 1:
        return "final_norm"
    return f"after_layer_{index - 1}"


def matrix_group(name: str):
    match = ATTN_RE.match(name)
    if match:
        return f"attention.{match.group(2)}", int(match.group(1))
    match = EXPERT_RE.match(name)
    if match:
        return f"moe_expert.{match.group(3)}", int(match.group(1))
    return None, None


def top_right_direction(weight):
    _, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    energy = s.square()
    return vh[0], float(energy[0].item() / energy.sum().clamp_min(1e-12).item())


def analyze_matrix_directions(model, embedding, content_token_ids, embedding_mean, embedding_centered_v1, top_tokens):
    rows = []
    for name, tensor in model.state_dict().items():
        group, layer = matrix_group(name)
        if group is None:
            continue
        weight = tensor.detach().float().cpu()
        if weight.ndim != 2:
            continue
        v1, top1_energy = top_right_direction(weight)
        # Only compare directions that live in hidden-size token embedding space.
        comparable_to_embedding = v1.numel() == embedding.shape[-1]
        row = {
            "name": name,
            "group": group,
            "layer": layer,
            "input_dim": int(v1.numel()),
            "top1_energy": top1_energy,
            "comparable_to_embedding": comparable_to_embedding,
        }
        if comparable_to_embedding:
            row.update(
                {
                    "abs_cos_embedding_mean": abs_cos(v1, embedding_mean),
                    "abs_cos_embedding_centered_v1": abs_cos(v1, embedding_centered_v1),
                    "top_embedding_tokens": top_token_matches(v1, embedding, content_token_ids, top_tokens),
                }
            )
        rows.append(row)
    return rows


def summarize_matrix_rows(rows):
    summary = {}
    for group in sorted(set(row["group"] for row in rows)):
        group_rows = [row for row in rows if row["group"] == group and row.get("comparable_to_embedding")]
        if not group_rows:
            continue
        summary[group] = {
            "num_matrices": len(group_rows),
            "mean_abs_cos_embedding_mean": float(sum(row["abs_cos_embedding_mean"] for row in group_rows) / len(group_rows)),
            "mean_abs_cos_embedding_centered_v1": float(
                sum(row["abs_cos_embedding_centered_v1"] for row in group_rows) / len(group_rows)
            ),
            "mean_top1_energy": float(sum(row["top1_energy"] for row in group_rows) / len(group_rows)),
            "max_abs_cos_embedding_mean": max(row["abs_cos_embedding_mean"] for row in group_rows),
        }
    return summary


@torch.no_grad()
def analyze_run(args, alias, run_name, distribution, device):
    run_dir = os.path.join(args.checkpoint_root, run_name)
    ckpt_path, step = find_checkpoint(run_dir, args.checkpoint_step)
    if ckpt_path is None:
        return {"alias": alias, "run": run_name, "error": f"No checkpoint <= {args.checkpoint_step}"}
    runtime_config = load_runtime_config(run_dir)
    model = MyQwen3ForCausalLM(build_config(args, runtime_config)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    embedding = model.model.embed_tokens.weight.detach().float().cpu()
    content_token_ids = torch.arange(
        args.synthetic_min_token_id,
        args.synthetic_min_token_id + args.synthetic_content_token_count,
        dtype=torch.long,
    )
    content_embedding = embedding[content_token_ids]
    embedding_mean = content_embedding.mean(dim=0)
    emb_dirs = raw_and_centered_top_directions(content_embedding)

    dataset = build_dataset(args, distribution)
    reps = collect_hidden_states(args, model, dataset, device)
    rep_rows = []
    for idx, rep in enumerate(reps):
        dirs = raw_and_centered_top_directions(rep)
        rep_rows.append(
            {
                "name": representation_name(idx, len(reps)),
                "raw_top1_energy": dirs["raw_top1_energy"],
                "centered_top1_energy": dirs["centered_top1_energy"],
                "abs_cos_raw_v1_rep_mean": abs_cos(dirs["raw_v1"], dirs["mean"]),
                "abs_cos_raw_v1_embedding_mean": abs_cos(dirs["raw_v1"], embedding_mean),
                "abs_cos_centered_v1_embedding_mean": abs_cos(dirs["centered_v1"], embedding_mean),
                "abs_cos_raw_v1_embedding_centered_v1": abs_cos(dirs["raw_v1"], emb_dirs["centered_v1"]),
                "abs_cos_centered_v1_embedding_centered_v1": abs_cos(dirs["centered_v1"], emb_dirs["centered_v1"]),
                "raw_v1_top_embedding_tokens": top_token_matches(dirs["raw_v1"], embedding, content_token_ids, args.top_tokens),
                "centered_v1_top_embedding_tokens": top_token_matches(dirs["centered_v1"], embedding, content_token_ids, args.top_tokens),
            }
        )

    matrix_rows = analyze_matrix_directions(
        model=model.cpu(),
        embedding=embedding,
        content_token_ids=content_token_ids,
        embedding_mean=embedding_mean,
        embedding_centered_v1=emb_dirs["centered_v1"],
        top_tokens=args.top_tokens,
    )
    return {
        "alias": alias,
        "run": run_name,
        "distribution": distribution,
        "checkpoint_step": step,
        "checkpoint_path": ckpt_path,
        "embedding": {
            "content_mean_norm": float(torch.linalg.vector_norm(embedding_mean).item()),
            "content_embedding_norm_mean": float(torch.linalg.vector_norm(content_embedding, dim=-1).mean().item()),
            "raw_top1_energy": emb_dirs["raw_top1_energy"],
            "centered_top1_energy": emb_dirs["centered_top1_energy"],
            "abs_cos_raw_v1_embedding_mean": abs_cos(emb_dirs["raw_v1"], embedding_mean),
            "centered_v1_top_embedding_tokens": top_token_matches(emb_dirs["centered_v1"], embedding, content_token_ids, args.top_tokens),
        },
        "representations": rep_rows,
        "matrix_summary": summarize_matrix_rows(matrix_rows),
        "matrices": matrix_rows,
    }


def main():
    args = parse_args()
    device = choose_device()
    output = {
        "config": vars(args),
        "device": str(device),
        "runs": [
            analyze_run(args, alias, run_name, distribution, device)
            for alias, run_name, distribution in parse_run_specs(args.run_specs)
        ],
    }
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
