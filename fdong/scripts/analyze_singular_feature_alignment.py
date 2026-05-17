import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from models.myqwen import MyQwen3ForCausalLM, MyQwen3HeadMoE, MyQwen3MoE
from utils import HierarchicalPatternData


DEFAULT_RUN_SPECS = ",".join(
    [
        "uniform_baseline:inverse-kv-local-h128-l3-top1:uniform",
        "zipf_baseline:inverse-kv-zipf-baseline:zipf",
    ]
)

ATTN_LINEAR_RE = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")


def parse_args():
    parser = argparse.ArgumentParser(description="Align weight singular directions with ground-truth feature centroids.")
    parser.add_argument("--config_dir", type=str, default="../Qwen3-0.6B")
    parser.add_argument("--checkpoint_root", type=str, default="../checkpoints")
    parser.add_argument("--run_specs", type=str, default=DEFAULT_RUN_SPECS)
    parser.add_argument("--checkpoint_step", type=int, default=5000)
    parser.add_argument("--output_path", type=str, default="../experiments/singular_feature_alignment_baselines_step5000.json")

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

    parser.add_argument("--topk_dims", type=str, default="1,2,5,10,20")
    parser.add_argument("--min_feature_count", type=int, default=8)
    return parser.parse_args()


def parse_run_specs(text: str) -> List[Tuple[str, str, str]]:
    specs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError("Each run spec must be alias:run_name:distribution")
        alias, run_name, distribution = parts
        if distribution not in ("uniform", "zipf"):
            raise ValueError("distribution must be uniform or zipf")
        specs.append((alias, run_name, distribution))
    return specs


def parse_int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


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


def pearson(xs: List[float], ys: List[float]):
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))


class FeatureAccumulator:
    def __init__(self, dim: int):
        self.dim = dim
        self.total_count = 0
        self.total_sum = torch.zeros(dim)
        self.feature_sums = {
            "token_id": defaultdict(lambda: torch.zeros(dim)),
            "local_slot": defaultdict(lambda: torch.zeros(dim)),
            "higher_unit": defaultdict(lambda: torch.zeros(dim)),
        }
        self.feature_counts = {
            "token_id": Counter(),
            "local_slot": Counter(),
            "higher_unit": Counter(),
        }

    def add(self, vectors: torch.Tensor, labels: Dict[str, torch.Tensor]):
        vectors = vectors.detach().float().cpu().reshape(-1, self.dim)
        count = vectors.shape[0]
        if count == 0:
            return
        self.total_count += count
        self.total_sum += vectors.sum(dim=0)
        for feature_name, feature_labels in labels.items():
            feature_labels = feature_labels.detach().cpu().reshape(-1)
            for feature_id in torch.unique(feature_labels).tolist():
                mask = feature_labels == feature_id
                if int(mask.sum()) == 0:
                    continue
                self.feature_sums[feature_name][int(feature_id)] += vectors[mask].sum(dim=0)
                self.feature_counts[feature_name][int(feature_id)] += int(mask.sum())

    def global_mean(self):
        if self.total_count == 0:
            return torch.zeros(self.dim)
        return self.total_sum / self.total_count


class AlignmentCollector:
    def __init__(self):
        self.current_labels = None
        self.accumulators: Dict[str, FeatureAccumulator] = {}
        self.weights: Dict[str, torch.Tensor] = {}
        self.info: Dict[str, Dict] = {}

    def ensure(self, key: str, weight: torch.Tensor, info: Dict, dim: int):
        if key not in self.accumulators:
            self.accumulators[key] = FeatureAccumulator(dim)
            self.weights[key] = weight.detach().float().cpu()
            self.info[key] = dict(info)

    def labels_for_all_tokens(self):
        return {
            "token_id": self.current_labels["token_id"].reshape(-1),
            "local_slot": self.current_labels["local_slot"].reshape(-1),
            "higher_unit": self.current_labels["higher_unit"].reshape(-1),
        }

    def add_attention(self, key: str, weight: torch.Tensor, info: Dict, vectors: torch.Tensor):
        dim = vectors.shape[-1]
        self.ensure(key, weight, info, dim)
        self.accumulators[key].add(vectors.reshape(-1, dim), self.labels_for_all_tokens())

    def add_selected(self, key: str, weight: torch.Tensor, info: Dict, vectors: torch.Tensor, token_indices: torch.Tensor):
        dim = vectors.shape[-1]
        self.ensure(key, weight, info, dim)
        flat_labels = self.labels_for_all_tokens()
        selected_labels = {
            name: value[token_indices.detach().cpu()]
            for name, value in flat_labels.items()
        }
        self.accumulators[key].add(vectors, selected_labels)


def make_attention_pre_hook(collector: AlignmentCollector, module_name: str, module):
    match = ATTN_LINEAR_RE.match(module_name)
    if not match:
        raise ValueError(f"Unexpected attention module name: {module_name}")
    layer_idx, subtype = match.groups()
    key = module_name
    info = {
        "family": "attention",
        "group": f"attention.{subtype}",
        "layer": int(layer_idx),
        "subtype": subtype,
        "input_space": f"layer_{layer_idx}_{subtype}_input",
    }

    def hook(_module, inputs):
        if collector.current_labels is None:
            return
        collector.add_attention(key, _module.weight, info, inputs[0])

    return hook


def make_token_moe_hook(collector: AlignmentCollector, module_name: str, module: MyQwen3MoE):
    layer_idx = int(module_name.split(".")[2])

    def hook(_module, inputs, output):
        if collector.current_labels is None or not isinstance(output, tuple):
            return
        hidden_states = inputs[0].detach()
        expert_labels = output[1].detach()
        flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_experts = expert_labels.reshape(-1, expert_labels.shape[-1])[:, 0]
        for expert_idx, expert in enumerate(_module.experts):
            token_idx = torch.where(flat_experts == expert_idx)[0]
            if token_idx.numel() == 0:
                continue
            expert_input = flat_states[token_idx]
            prefix = f"{module_name}.experts.{expert_idx}"
            base_info = {
                "family": "moe_expert",
                "layer": layer_idx,
                "expert": expert_idx,
                "head": None,
            }
            collector.add_selected(
                f"{prefix}.gate_proj",
                expert.gate_proj.weight,
                {**base_info, "group": "moe_expert.gate_proj", "subtype": "gate_proj", "input_space": "expert_input"},
                expert_input,
                token_idx,
            )
            collector.add_selected(
                f"{prefix}.up_proj",
                expert.up_proj.weight,
                {**base_info, "group": "moe_expert.up_proj", "subtype": "up_proj", "input_space": "expert_input"},
                expert_input,
                token_idx,
            )
            intermediate = expert.act_fn(expert.gate_proj(expert_input)) * expert.up_proj(expert_input)
            collector.add_selected(
                f"{prefix}.down_proj",
                expert.down_proj.weight,
                {**base_info, "group": "moe_expert.down_proj", "subtype": "down_proj", "input_space": "expert_intermediate"},
                intermediate,
                token_idx,
            )

    return hook


def make_head_moe_hook(collector: AlignmentCollector, module_name: str, module: MyQwen3HeadMoE):
    layer_idx = int(module_name.split(".")[2])

    def hook(_module, inputs, output):
        if collector.current_labels is None or not isinstance(output, tuple):
            return
        hidden_states = inputs[0].detach()
        expert_labels = output[1].detach()
        batch, seq_len, num_heads, head_dim = hidden_states.shape
        flat_token_indices = torch.arange(batch * seq_len).repeat_interleave(1)
        for head_idx in range(num_heads):
            head_states = hidden_states[:, :, head_idx, :].reshape(-1, head_dim)
            head_experts = expert_labels[:, :, head_idx, :].reshape(-1, expert_labels.shape[-1])[:, 0]
            for expert_idx, expert in enumerate(_module.experts[head_idx]):
                token_idx = torch.where(head_experts == expert_idx)[0]
                if token_idx.numel() == 0:
                    continue
                expert_input = head_states[token_idx]
                prefix = f"{module_name}.experts.{head_idx}.{expert_idx}"
                base_info = {
                    "family": "moe_expert",
                    "layer": layer_idx,
                    "expert": expert_idx,
                    "head": head_idx,
                }
                collector.add_selected(
                    f"{prefix}.gate_proj",
                    expert.gate_proj.weight,
                    {**base_info, "group": "moe_expert.gate_proj", "subtype": "gate_proj", "input_space": "head_expert_input"},
                    expert_input,
                    flat_token_indices[token_idx],
                )
                collector.add_selected(
                    f"{prefix}.up_proj",
                    expert.up_proj.weight,
                    {**base_info, "group": "moe_expert.up_proj", "subtype": "up_proj", "input_space": "head_expert_input"},
                    expert_input,
                    flat_token_indices[token_idx],
                )
                intermediate = expert.act_fn(expert.gate_proj(expert_input)) * expert.up_proj(expert_input)
                collector.add_selected(
                    f"{prefix}.down_proj",
                    expert.down_proj.weight,
                    {**base_info, "group": "moe_expert.down_proj", "subtype": "down_proj", "input_space": "head_expert_intermediate"},
                    intermediate,
                    flat_token_indices[token_idx],
                )

    return hook


def register_hooks(model, collector: AlignmentCollector):
    handles = []
    for module_name, module in model.named_modules():
        if ATTN_LINEAR_RE.match(module_name):
            handles.append(module.register_forward_pre_hook(make_attention_pre_hook(collector, module_name, module)))
        elif isinstance(module, MyQwen3MoE):
            handles.append(module.register_forward_hook(make_token_moe_hook(collector, module_name, module)))
        elif isinstance(module, MyQwen3HeadMoE):
            handles.append(module.register_forward_hook(make_head_moe_hook(collector, module_name, module)))
    return handles


def singular_input_directions(weight: torch.Tensor, remove_direction: Optional[torch.Tensor] = None):
    # weight is [out_dim, in_dim]; right singular vectors live in the input space.
    weight = weight.float()
    if remove_direction is not None:
        direction = remove_direction.float()
        norm = torch.linalg.vector_norm(direction).item()
        if norm > 1e-12:
            unit = direction / norm
            weight = weight - (weight @ unit).unsqueeze(-1) @ unit.unsqueeze(0)
    _, _, vh = torch.linalg.svd(weight, full_matrices=False)
    return vh


def projection_energy(direction: torch.Tensor, vh: torch.Tensor, topk: int):
    direction = direction.float()
    norm = torch.linalg.vector_norm(direction).item()
    if norm <= 1e-12:
        return 0.0
    unit = direction / norm
    kk = min(topk, vh.shape[0])
    return float(((vh[:kk] @ unit) ** 2).sum().item())


def analyze_accumulator(acc: FeatureAccumulator, weight: torch.Tensor, topk_dims: List[int], min_feature_count: int):
    vh = singular_input_directions(weight)
    global_mean = acc.global_mean()
    mean_removed_vh = singular_input_directions(weight, remove_direction=global_mean)
    result = {
        "num_vectors": acc.total_count,
        "input_dim": acc.dim,
        "rank": int(vh.shape[0]),
        "activation_mean_norm": float(torch.linalg.vector_norm(global_mean).item()),
        "weight_top1_alignment_to_activation_mean": projection_energy(global_mean, vh, 1),
        "mean_removed_rank": int(mean_removed_vh.shape[0]),
        "features": {},
    }
    for feature_name, sums in acc.feature_sums.items():
        counts = acc.feature_counts[feature_name]
        rows = []
        for feature_id, feature_sum in sums.items():
            count = counts[feature_id]
            if count < min_feature_count:
                continue
            centroid = feature_sum / count
            centered = centroid - global_mean
            row = {
                "feature_id": int(feature_id),
                "count": int(count),
                "log_count": float(math.log(count + 1.0)),
            }
            for topk in topk_dims:
                row[f"raw_top{topk}_alignment"] = projection_energy(centroid, vh, topk)
                row[f"centered_top{topk}_alignment"] = projection_energy(centered, vh, topk)
                row[f"mean_removed_centered_top{topk}_alignment"] = projection_energy(centered, mean_removed_vh, topk)
            rows.append(row)
        feature_summary = {
            "num_features": len(rows),
            "top_by_count": sorted(rows, key=lambda row: row["count"], reverse=True)[:8],
        }
        for topk in topk_dims:
            raw_values = [row[f"raw_top{topk}_alignment"] for row in rows]
            centered_values = [row[f"centered_top{topk}_alignment"] for row in rows]
            mean_removed_values = [row[f"mean_removed_centered_top{topk}_alignment"] for row in rows]
            log_counts = [row["log_count"] for row in rows]
            feature_summary[f"mean_raw_top{topk}_alignment"] = float(sum(raw_values) / len(raw_values)) if raw_values else None
            feature_summary[f"mean_centered_top{topk}_alignment"] = (
                float(sum(centered_values) / len(centered_values)) if centered_values else None
            )
            feature_summary[f"mean_mean_removed_centered_top{topk}_alignment"] = (
                float(sum(mean_removed_values) / len(mean_removed_values)) if mean_removed_values else None
            )
            feature_summary[f"corr_log_count_raw_top{topk}_alignment"] = pearson(log_counts, raw_values)
            feature_summary[f"corr_log_count_centered_top{topk}_alignment"] = pearson(log_counts, centered_values)
            feature_summary[f"corr_log_count_mean_removed_centered_top{topk}_alignment"] = pearson(
                log_counts,
                mean_removed_values,
            )
        result["features"][feature_name] = feature_summary
    return result


@torch.no_grad()
def collect_run(args, alias, run_name, distribution, device):
    run_dir = os.path.join(args.checkpoint_root, run_name)
    ckpt_path, step = find_checkpoint(run_dir, args.checkpoint_step)
    if ckpt_path is None:
        return {"alias": alias, "run": run_name, "error": f"No checkpoint <= {args.checkpoint_step}"}

    runtime_config = load_runtime_config(run_dir)
    model = MyQwen3ForCausalLM(build_config(args, runtime_config)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    dataset = build_dataset(args, distribution)
    collector = AlignmentCollector()
    handles = register_hooks(model, collector)
    for start in range(0, len(dataset), args.batch_size):
        batch = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
        source = torch.stack([item[0] for item in batch]).to(device)
        metadata = torch.stack([item[3] for item in batch])
        collector.current_labels = {
            "token_id": source.detach().cpu(),
            "local_slot": metadata[:, :, 0],
            "higher_unit": metadata[:, :, 1],
        }
        model(source, output_expert_labels=True)
    collector.current_labels = None
    for handle in handles:
        handle.remove()

    topk_dims = parse_int_list(args.topk_dims)
    matrices = []
    for key in sorted(collector.accumulators.keys()):
        matrix_result = analyze_accumulator(
            collector.accumulators[key],
            collector.weights[key],
            topk_dims=topk_dims,
            min_feature_count=args.min_feature_count,
        )
        matrices.append(
            {
                "name": key,
                **collector.info[key],
                **matrix_result,
            }
        )

    return {
        "alias": alias,
        "run": run_name,
        "distribution": distribution,
        "checkpoint_step": step,
        "checkpoint_path": ckpt_path,
        "runtime_config": runtime_config,
        "num_matrices": len(matrices),
        "summary": summarize_matrices(matrices, topk_dims),
        "matrices": matrices,
    }


def summarize_matrices(matrices, topk_dims):
    summary = {}
    for group in sorted(set(matrix["group"] for matrix in matrices)):
        group_mats = [matrix for matrix in matrices if matrix["group"] == group]
        group_summary = {"num_matrices": len(group_mats)}
        for feature_name in ["token_id", "local_slot", "higher_unit"]:
            rows = [matrix["features"].get(feature_name, {}) for matrix in group_mats]
            feature_summary = {}
            for topk in topk_dims:
                for mode in ["raw", "centered"]:
                    key = f"mean_{mode}_top{topk}_alignment"
                    values = [row.get(key) for row in rows if row.get(key) is not None]
                    feature_summary[key] = float(sum(values) / len(values)) if values else None
                    corr_key = f"corr_log_count_{mode}_top{topk}_alignment"
                    corr_values = [row.get(corr_key) for row in rows if row.get(corr_key) is not None]
                    feature_summary[corr_key] = float(sum(corr_values) / len(corr_values)) if corr_values else None
                key = f"mean_mean_removed_centered_top{topk}_alignment"
                values = [row.get(key) for row in rows if row.get(key) is not None]
                feature_summary[key] = float(sum(values) / len(values)) if values else None
                corr_key = f"corr_log_count_mean_removed_centered_top{topk}_alignment"
                corr_values = [row.get(corr_key) for row in rows if row.get(corr_key) is not None]
                feature_summary[corr_key] = float(sum(corr_values) / len(corr_values)) if corr_values else None
            group_summary[feature_name] = feature_summary
        summary[group] = group_summary
    return summary


def main():
    args = parse_args()
    device = choose_device()
    run_specs = parse_run_specs(args.run_specs)
    output = {
        "config": vars(args),
        "device": str(device),
        "runs": [
            collect_run(args, alias, run_name, distribution, device)
            for alias, run_name, distribution in run_specs
        ],
    }
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
