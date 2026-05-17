import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

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


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze whether ground-truth features appear in representation space.")
    parser.add_argument("--config_dir", type=str, default="../Qwen3-0.6B")
    parser.add_argument("--checkpoint_root", type=str, default="../checkpoints")
    parser.add_argument("--run_specs", type=str, default=DEFAULT_RUN_SPECS)
    parser.add_argument("--checkpoint_step", type=int, default=5000)
    parser.add_argument("--output_path", type=str, default="../experiments/feature_representations_baselines.json")

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

    parser.add_argument("--pair_samples", type=int, default=200000)
    parser.add_argument("--probe_train_fraction", type=float, default=0.7)
    parser.add_argument("--pca_dims", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=123)
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


@torch.no_grad()
def collect_representations(args, model, dataset, device):
    rep_chunks = None
    local_chunks, high_chunks, token_chunks = [], [], []
    for start in range(0, len(dataset), args.batch_size):
        batch = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
        source = torch.stack([item[0] for item in batch]).to(device)
        metadata = torch.stack([item[3] for item in batch])
        output = model(source, output_hidden_states=True)
        hidden_states = output.hidden_states
        if rep_chunks is None:
            rep_chunks = [[] for _ in hidden_states]
        for idx, states in enumerate(hidden_states):
            rep_chunks[idx].append(states.detach().float().cpu().reshape(-1, states.shape[-1]))
        local_chunks.append(metadata[:, :, 0].reshape(-1))
        high_chunks.append(metadata[:, :, 1].reshape(-1))
        token_chunks.append(source.detach().cpu().reshape(-1))

    reps = [torch.cat(chunks, dim=0) for chunks in rep_chunks]
    labels = {
        "token_id": torch.cat(token_chunks, dim=0),
        "local_slot": torch.cat(local_chunks, dim=0),
        "higher_unit": torch.cat(high_chunks, dim=0),
    }
    return reps, labels


def majority_accuracy(labels: torch.Tensor) -> float:
    counts = Counter(labels.tolist())
    return max(counts.values()) / max(len(labels), 1)


def split_indices(num_items: int, train_fraction: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_items, generator=generator)
    train_len = int(num_items * train_fraction)
    return perm[:train_len], perm[train_len:]


def centroid_probe(x: torch.Tensor, labels: torch.Tensor, train_idx: torch.Tensor, test_idx: torch.Tensor):
    x_norm = F.normalize(x, dim=-1)
    train_x = x_norm[train_idx]
    train_y = labels[train_idx]
    test_x = x_norm[test_idx]
    test_y = labels[test_idx]
    classes = sorted(set(train_y.tolist()))
    centroids = []
    kept_classes = []
    for cls in classes:
        mask = train_y == cls
        if int(mask.sum()) == 0:
            continue
        centroids.append(F.normalize(train_x[mask].mean(dim=0, keepdim=True), dim=-1)[0])
        kept_classes.append(cls)
    centroid_matrix = torch.stack(centroids, dim=0)
    class_tensor = torch.tensor(kept_classes, dtype=labels.dtype)
    logits = test_x @ centroid_matrix.T
    pred = class_tensor[logits.argmax(dim=-1)]
    valid = torch.isin(test_y, class_tensor)
    if int(valid.sum()) == 0:
        return {"accuracy": 0.0, "coverage": 0.0}
    return {
        "accuracy": float((pred[valid] == test_y[valid]).float().mean().item()),
        "coverage": float(valid.float().mean().item()),
        "majority_baseline": majority_accuracy(test_y),
    }


def pair_cosine_metrics(x: torch.Tensor, labels: torch.Tensor, pair_samples: int, seed: int):
    num_items = x.shape[0]
    generator = torch.Generator().manual_seed(seed)
    left = torch.randint(0, num_items, (pair_samples,), generator=generator)
    right = torch.randint(0, num_items, (pair_samples,), generator=generator)
    x_norm = F.normalize(x, dim=-1)
    sims = (x_norm[left] * x_norm[right]).sum(dim=-1)
    same = labels[left] == labels[right]
    if int(same.sum()) == 0 or int((~same).sum()) == 0:
        return {}
    same_mean = float(sims[same].mean().item())
    diff_mean = float(sims[~same].mean().item())
    return {
        "same_pair_cosine": same_mean,
        "diff_pair_cosine": diff_mean,
        "cosine_gap": same_mean - diff_mean,
        "same_pair_fraction": float(same.float().mean().item()),
    }


def pca_summary(x: torch.Tensor, labels_by_feature: Dict[str, torch.Tensor], max_dims: int):
    x = x.float()
    centered = x - x.mean(dim=0, keepdim=True)
    _, singular_values, v = torch.pca_lowrank(centered, q=min(max_dims, centered.shape[-1]), center=False)
    energy = singular_values.square()
    total_energy = float(centered.square().sum().item())
    pca_x = centered @ v

    def topk_energy(k):
        if total_energy <= 0:
            return 0.0
        return float(energy[: min(k, len(energy))].sum().item() / total_energy)

    feature_r2 = {}
    for feature_name, labels in labels_by_feature.items():
        feature_r2[feature_name] = {}
        for k in [1, 2, 5, 10, 20, max_dims]:
            kk = min(k, pca_x.shape[-1])
            z = pca_x[:, :kk]
            total = float(((z - z.mean(dim=0, keepdim=True)) ** 2).sum().item())
            if total <= 0:
                feature_r2[feature_name][f"top{k}_between_class_variance_ratio"] = 0.0
                continue
            between = 0.0
            for cls in torch.unique(labels):
                mask = labels == cls
                if int(mask.sum()) == 0:
                    continue
                cls_mean = z[mask].mean(dim=0)
                between += int(mask.sum()) * float((cls_mean ** 2).sum().item())
            feature_r2[feature_name][f"top{k}_between_class_variance_ratio"] = between / total

    return {
        "top1_energy": topk_energy(1),
        "top2_energy": topk_energy(2),
        "top5_energy": topk_energy(5),
        "top10_energy": topk_energy(10),
        "top20_energy": topk_energy(20),
        "singular_values": [float(x) for x in singular_values.tolist()],
        "feature_between_class_variance": feature_r2,
        "pca_projection": pca_x,
    }


def pearson(xs: List[float], ys: List[float]):
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))


def frequency_metrics(x: torch.Tensor, high_labels: torch.Tensor, pc_projection: torch.Tensor):
    counts = Counter(high_labels.tolist())
    rows = []
    x_norms = x.norm(dim=-1)
    for cls, count in sorted(counts.items()):
        mask = high_labels == cls
        rows.append(
            {
                "higher_unit": int(cls),
                "count": int(count),
                "log_count": float(math.log(count + 1.0)),
                "mean_norm": float(x_norms[mask].mean().item()),
                "mean_pc1": float(pc_projection[mask, 0].mean().item()) if pc_projection.shape[-1] >= 1 else 0.0,
                "mean_pc2": float(pc_projection[mask, 1].mean().item()) if pc_projection.shape[-1] >= 2 else 0.0,
                "within_variance": float(((x[mask] - x[mask].mean(dim=0, keepdim=True)) ** 2).sum(dim=-1).mean().item()),
            }
        )
    return {
        "num_higher_units": len(rows),
        "top_higher_units": sorted(rows, key=lambda row: row["count"], reverse=True)[:8],
        "log_count_vs_mean_norm": pearson([row["log_count"] for row in rows], [row["mean_norm"] for row in rows]),
        "log_count_vs_mean_pc1": pearson([row["log_count"] for row in rows], [row["mean_pc1"] for row in rows]),
        "log_count_vs_mean_pc2": pearson([row["log_count"] for row in rows], [row["mean_pc2"] for row in rows]),
        "log_count_vs_within_variance": pearson([row["log_count"] for row in rows], [row["within_variance"] for row in rows]),
    }


def representation_name(index: int, num_reps: int):
    if index == 0:
        return "embedding"
    if index == num_reps - 1:
        return "final_norm"
    return f"after_layer_{index - 1}"


def analyze_representations(args, reps, labels):
    train_idx, test_idx = split_indices(reps[0].shape[0], args.probe_train_fraction, args.random_seed)
    results = []
    for idx, x in enumerate(reps):
        labels_for_pca = {
            "local_slot": labels["local_slot"],
            "higher_unit": labels["higher_unit"],
        }
        pca = pca_summary(x, labels_for_pca, args.pca_dims)
        rep_result = {
            "name": representation_name(idx, len(reps)),
            "norm_mean": float(x.norm(dim=-1).mean().item()),
            "norm_std": float(x.norm(dim=-1).std().item()),
            "pca": {key: value for key, value in pca.items() if key != "pca_projection"},
            "features": {},
            "frequency": frequency_metrics(x, labels["higher_unit"], pca["pca_projection"]),
        }
        for feature_name in ["token_id", "local_slot", "higher_unit"]:
            y = labels[feature_name]
            rep_result["features"][feature_name] = {
                "centroid_probe": centroid_probe(x, y, train_idx, test_idx),
                "pair_cosine": pair_cosine_metrics(
                    x,
                    y,
                    pair_samples=args.pair_samples,
                    seed=args.random_seed + idx * 101 + len(feature_name),
                ),
            }
        results.append(rep_result)
    return results


def summarize_best(representations):
    summary = {}
    for feature in ["token_id", "local_slot", "higher_unit"]:
        best_probe = max(
            representations,
            key=lambda row: row["features"][feature]["centroid_probe"]["accuracy"],
        )
        best_gap = max(
            representations,
            key=lambda row: row["features"][feature]["pair_cosine"].get("cosine_gap", -999),
        )
        summary[feature] = {
            "best_centroid_probe_rep": best_probe["name"],
            "best_centroid_probe_accuracy": best_probe["features"][feature]["centroid_probe"]["accuracy"],
            "best_pair_gap_rep": best_gap["name"],
            "best_pair_cosine_gap": best_gap["features"][feature]["pair_cosine"].get("cosine_gap"),
        }
    best_freq_norm = max(
        representations,
        key=lambda row: abs(row["frequency"]["log_count_vs_mean_norm"] or 0.0),
    )
    best_freq_pc1 = max(
        representations,
        key=lambda row: abs(row["frequency"]["log_count_vs_mean_pc1"] or 0.0),
    )
    summary["frequency"] = {
        "strongest_log_count_norm_rep": best_freq_norm["name"],
        "strongest_log_count_norm_corr": best_freq_norm["frequency"]["log_count_vs_mean_norm"],
        "strongest_log_count_pc1_rep": best_freq_pc1["name"],
        "strongest_log_count_pc1_corr": best_freq_pc1["frequency"]["log_count_vs_mean_pc1"],
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
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset = build_dataset(args, distribution)
    reps, labels = collect_representations(args, model, dataset, device)
    representations = analyze_representations(args, reps, labels)
    return {
        "alias": alias,
        "run": run_name,
        "distribution": distribution,
        "checkpoint_step": step,
        "checkpoint_path": ckpt_path,
        "runtime_config": runtime_config,
        "label_majority_baselines": {
            name: majority_accuracy(value)
            for name, value in labels.items()
        },
        "summary": summarize_best(representations),
        "representations": representations,
    }


def main():
    args = parse_args()
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    device = choose_device()
    run_specs = parse_run_specs(args.run_specs)
    output = {
        "config": vars(args),
        "device": str(device),
        "runs": [
            analyze_run(args, alias, run_name, distribution, device)
            for alias, run_name, distribution in run_specs
        ],
    }
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
