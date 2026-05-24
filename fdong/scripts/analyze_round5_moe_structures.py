import argparse
import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from analyze_moe_variant_selectivity import (
    attention_expert_alignment,
    attention_feature_metrics,
    build_config,
    find_checkpoint,
    load_model,
    load_runtime_config,
    mapping_purity,
    mutual_information,
    pairwise_same_bucket_given_same_feature,
)
from utils import HierarchicalPatternData


ROUND5_RUNS = [
    "round5-full-attn-output-exp-resid",
    "round5-full-q-exp-resid",
    "round5-full-k-exp-resid",
    "round5-full-v-exp-resid",
    "round5-full-layer-input-exp-resid",
    "round5-full-hidden-exp-resid",
    "round5-head-attn-output-exp-resid",
    "round5-head-q-exp-resid",
    "round5-head-k-exp-resid",
    "round5-head-v-exp-resid",
    "round5-head-layer-input-exp-resid",
    "round5-head-hidden-exp-resid",
    "round5-spectral-attn-output-exp-resid",
    "round5-spectral-q-exp-resid",
    "round5-spectral-k-exp-resid",
    "round5-spectral-v-exp-resid",
    "round5-spectral-layer-input-exp-resid",
    "round5-spectral-hidden-exp-resid",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Round5 MoE structure sweep.")
    parser.add_argument("--config_dir", type=str, default="../Qwen3-0.6B")
    parser.add_argument("--checkpoint_root", type=str, default="../checkpoints")
    parser.add_argument("--log_dir", type=str, default="../logs")
    parser.add_argument("--runs", type=str, default=",".join(ROUND5_RUNS))
    parser.add_argument("--checkpoint_step", type=int, default=1000)
    parser.add_argument("--output_json", type=str, default="../experiments/round5_moe_structure_analysis.json")
    parser.add_argument("--output_md", type=str, default="../experiments/round5_moe_structure_analysis.md")

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--synthetic_block_size", type=int, default=4)
    parser.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    parser.add_argument("--synthetic_content_token_count", type=int, default=512)
    parser.add_argument("--synthetic_num_units_per_layer", type=int, default=512)
    parser.add_argument("--synthetic_seed", type=int, default=0)
    parser.add_argument("--synthetic_min_token_id", type=int, default=1)
    parser.add_argument("--synthetic_sampling_distribution", choices=["uniform", "zipf"], default="zipf")
    parser.add_argument("--synthetic_zipf_alpha", type=float, default=1.0)
    parser.add_argument("--synthetic_zipf_shuffle_ranks", action="store_true", default=True)
    parser.add_argument("--synthetic_no_zipf_shuffle_ranks", action="store_false", dest="synthetic_zipf_shuffle_ranks")

    parser.add_argument("--debug_vocab_size", type=int, default=513)
    parser.add_argument("--debug_hidden_size", type=int, default=64)
    parser.add_argument("--debug_intermediate_size", type=int, default=128)
    parser.add_argument("--debug_num_hidden_layers", type=int, default=2)
    parser.add_argument("--debug_num_attention_heads", type=int, default=4)
    parser.add_argument("--debug_num_key_value_heads", type=int, default=2)
    parser.add_argument("--debug_head_dim", type=int, default=16)
    parser.add_argument("--debug_max_position_embeddings", type=int, default=256)
    parser.add_argument("--attention_stride_pattern", type=str, default=None)
    parser.add_argument("--residual_source_pattern", type=str, default=None)
    return parser.parse_args()


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def new_counter():
    return {"loss_sum": 0.0, "correct": 0, "count": 0}


def update_counter(counter, losses, pred, target, mask):
    if mask.sum().item() == 0:
        return
    counter["loss_sum"] += float(losses[mask].sum().item())
    counter["correct"] += int(((pred == target) & mask).sum().item())
    counter["count"] += int(mask.sum().item())


def finish_counter(counter):
    count = max(counter["count"], 1)
    return {
        "loss": counter["loss_sum"] / count,
        "accuracy": counter["correct"] / count,
        "count": counter["count"],
    }


def entropy_from_counter(counter: Counter):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log((count / total) + 1e-12) for count in counter.values() if count > 0)


def effective_experts(labels: torch.Tensor) -> float:
    values = [int(x) for x in labels.reshape(-1).tolist() if int(x) >= 0]
    if not values:
        return 0.0
    return float(math.exp(entropy_from_counter(Counter(values))))


def normalize_expert_labels(primary: torch.Tensor, runtime_config: Dict, num_heads: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
    router_shape = str(runtime_config.get("moe_router_input_shape", "full"))
    if primary.dim() == 2:
        return "token", primary, primary
    if primary.dim() != 3:
        raise ValueError(f"Unsupported expert label shape: {tuple(primary.shape)}")
    if router_shape == "head" and primary.shape[-1] == num_heads:
        combined = primary.reshape(primary.shape[0], primary.shape[1], -1)
        return "head", primary, combined
    if router_shape == "spectral":
        labels = primary.clone()
        combined = torch.zeros(labels.shape[:2], dtype=torch.long)
        multiplier = 1
        for idx in range(labels.shape[-1]):
            value = labels[:, :, idx].clamp_min(0).long()
            combined = combined + value * multiplier
            max_value = int(value.max().item()) if value.numel() else 0
            multiplier *= max(max_value + 1, 1)
        return "spectral", labels, combined
    return "multi", primary, primary.reshape(primary.shape[0], primary.shape[1], -1)


def same_feature_same_expert_for_labels(labels: torch.Tensor, feature: torch.Tensor) -> float:
    if labels.dim() == 2:
        return pairwise_same_bucket_given_same_feature(labels, feature)
    rates = []
    for idx in range(labels.shape[-1]):
        one = labels[:, :, idx]
        valid = one >= 0
        if valid.any():
            rates.append(pairwise_same_bucket_given_same_feature(one, feature))
    return float(sum(rates) / len(rates)) if rates else 0.0


def feature_selectivity(labels: torch.Tensor, combined: torch.Tensor, feature: torch.Tensor) -> Dict:
    flat_feature = feature.reshape(-1).tolist()
    flat_combined = combined.reshape(-1).tolist()
    return {
        "feature_expert_mi": mutual_information(flat_feature, flat_combined),
        "feature_to_expert_purity": mapping_purity(flat_feature, flat_combined),
        "expert_to_feature_purity": mapping_purity(flat_combined, flat_feature),
        "same_feature_same_expert_rate": same_feature_same_expert_for_labels(labels, feature),
        "effective_experts": effective_experts(combined),
    }


def attention_expert_summary(attn: torch.Tensor, labels: torch.Tensor, combined: torch.Tensor, label_kind: str) -> Dict:
    if label_kind == "head":
        return attention_expert_alignment(attn, labels)
    return attention_expert_alignment(attn, combined)


def parse_train_loss(log_dir: str, run: str):
    path = os.path.join(log_dir, f"{run}.train.log")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    matches = re.findall(r"batch:\s*(\d+)-\d+,\s*loss:\s*([0-9.]+),\s*token_loss:\s*([0-9.]+)", text)
    if not matches:
        return None
    step, loss, token_loss = matches[-1]
    return {"step": int(step), "loss": float(loss), "token_loss": float(token_loss)}


def make_dataset(args):
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


@torch.no_grad()
def analyze_run(args, run: str, dataset, device):
    run_dir = os.path.join(args.checkpoint_root, run)
    runtime_config = load_runtime_config(run_dir)
    ckpt_path, step = find_checkpoint(run_dir, args.checkpoint_step)
    if ckpt_path is None:
        return {"run": run, "error": "missing checkpoint"}

    model = load_model(args, runtime_config, ckpt_path, device)
    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    counters = {
        "overall": new_counter(),
        "inside_local": new_counter(),
        "local_boundary": new_counter(),
        "local_boundary_not_high": new_counter(),
        "high_boundary": new_counter(),
    }
    features = {"local_slot": [], "higher_unit": []}
    expert_by_layer = [[] for _ in range(num_layers)]
    attention_feature_by_layer = [[] for _ in range(num_layers)]
    attention_expert_by_layer = [[] for _ in range(num_layers)]

    for start in range(0, args.num_samples, args.batch_size):
        end = min(start + args.batch_size, args.num_samples)
        items = [dataset[idx] for idx in range(start, end)]
        source = torch.stack([item[0] for item in items]).to(device)
        target = torch.stack([item[1] for item in items]).to(device)
        source_meta = torch.stack([item[3] for item in items]).to(device)
        target_meta = torch.stack([dataset.get_metadata(idx)["unit_ids_by_layer"][1:] for idx in range(start, end)]).to(device)

        output = model(source, output_attentions=True, output_expert_labels=True, use_cache=False)
        loss = F.cross_entropy(output.logits.reshape(-1, output.logits.size(-1)), target.reshape(-1), reduction="none")
        loss = loss.reshape_as(target)
        pred = output.logits.argmax(dim=-1)
        valid = target != 0
        local_same = (source_meta[:, :, 0] == target_meta[:, :, 0]) & (source_meta[:, :, 0] >= 0)
        high_same = (source_meta[:, :, 1] == target_meta[:, :, 1]) & (source_meta[:, :, 1] >= 0)
        local_boundary = ~local_same
        high_boundary = ~high_same
        masks = {
            "overall": valid,
            "inside_local": valid & local_same,
            "local_boundary": valid & local_boundary,
            "local_boundary_not_high": valid & local_boundary & ~high_boundary,
            "high_boundary": valid & high_boundary,
        }
        for name, mask in masks.items():
            update_counter(counters[name], loss, pred, target, mask)

        features["local_slot"].append(source_meta[:, :, 0].cpu())
        features["higher_unit"].append(source_meta[:, :, 1].cpu())

        for layer_idx, labels in enumerate(output.expert_labels):
            primary = labels[..., 0].detach().cpu()
            expert_by_layer[layer_idx].append(primary)
            attn = output.attentions[layer_idx].detach().cpu()
            batch_feature_rows = {}
            for feature_name, feature_layer in (("local_slot", 0), ("higher_unit", 1)):
                batch_feature_rows[feature_name] = attention_feature_metrics(attn, source_meta[:, :, feature_layer].cpu())
            attention_feature_by_layer[layer_idx].append(batch_feature_rows)

            label_kind, label_for_alignment, combined = normalize_expert_labels(primary, runtime_config, num_heads)
            attention_expert_by_layer[layer_idx].append(
                attention_expert_summary(attn, label_for_alignment, combined, label_kind)
            )

    features_all = {key: torch.cat(value, dim=0) for key, value in features.items()}
    layers = []
    for layer_idx in range(num_layers):
        primary = torch.cat(expert_by_layer[layer_idx], dim=0)
        label_kind, labels, combined = normalize_expert_labels(primary, runtime_config, num_heads)
        row = {
            "layer": layer_idx,
            "label_kind": label_kind,
            "expert_shape": list(primary.shape),
            "effective_experts": effective_experts(combined),
            "local_slot": feature_selectivity(labels, combined, features_all["local_slot"]),
            "higher_unit": feature_selectivity(labels, combined, features_all["higher_unit"]),
        }
        for feature_name in ("local_slot", "higher_unit"):
            rows = [item[feature_name] for item in attention_feature_by_layer[layer_idx]]
            row[f"attention_{feature_name}"] = {
                "include_self_mass_mean": sum(x["include_self_mass_mean"] for x in rows) / len(rows),
                "history_mass_mean": sum(x["history_mass_mean"] for x in rows) / len(rows),
                "include_self_pair_score_lift": sum(x["include_self_pair_score_lift"] for x in rows) / len(rows),
                "history_pair_score_lift": sum(x["history_pair_score_lift"] for x in rows) / len(rows),
            }
        rows = attention_expert_by_layer[layer_idx]
        row["attention_expert"] = {
            "same_expert_include_self_mass_mean": sum(x["same_expert_include_self_mass_mean"] for x in rows) / len(rows),
            "same_expert_history_mass_mean": sum(x["same_expert_history_mass_mean"] for x in rows) / len(rows),
            "same_expert_include_self_pair_score_lift": sum(x["same_expert_include_self_pair_score_lift"] for x in rows) / len(rows),
            "same_expert_history_pair_score_lift": sum(x["same_expert_history_pair_score_lift"] for x in rows) / len(rows),
        }
        layers.append(row)

    mean_layers = {}
    for feature in ("local_slot", "higher_unit"):
        mean_layers[feature] = {
            "same_feature_same_expert_rate": sum(x[feature]["same_feature_same_expert_rate"] for x in layers) / len(layers),
            "feature_to_expert_purity": sum(x[feature]["feature_to_expert_purity"] for x in layers) / len(layers),
            "feature_expert_mi": sum(x[feature]["feature_expert_mi"] for x in layers) / len(layers),
        }
    mean_layers["effective_experts"] = sum(x["effective_experts"] for x in layers) / len(layers)
    mean_layers["attention_local_include_self_mass"] = sum(x["attention_local_slot"]["include_self_mass_mean"] for x in layers) / len(layers)
    mean_layers["attention_high_include_self_mass"] = sum(x["attention_higher_unit"]["include_self_mass_mean"] for x in layers) / len(layers)
    mean_layers["attention_expert_mass"] = sum(x["attention_expert"]["same_expert_include_self_mass_mean"] for x in layers) / len(layers)
    return {
        "run": run,
        "checkpoint_step": step,
        "checkpoint_path": ckpt_path,
        "train": parse_train_loss(args.log_dir, run),
        "runtime_config": runtime_config,
        "metrics": {key: finish_counter(value) for key, value in counters.items()},
        "mean_layers": mean_layers,
        "layers": layers,
    }


def pct(value):
    return f"{100.0 * value:.2f}%"


def format_float(value):
    return f"{value:.4f}"


def write_markdown(args, results):
    lines = []
    lines.append("# Round5 MoE Structure Analysis")
    lines.append("")
    lines.append("数据设定：`b4-u512-vocab512-zipf1.0`；模型设定：`2-layer h64`；checkpoint step: `1000`。")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    headers = [
        "Run",
        "Router",
        "Train loss",
        "Eval loss",
        "NTP acc",
        "Inside-local",
        "Local-boundary-not-high",
        "High-boundary",
        "Local same-expert",
        "High same-expert",
        "Eff. experts",
        "Attn-local mass",
        "Attn-high mass",
        "Attn-expert mass",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in results:
        if "error" in row:
            lines.append(f"| `{row['run']}` | ERROR | | | | | | | | | | | | |")
            continue
        cfg = row["runtime_config"]
        router = f"{cfg.get('moe_router_input_pos')}/{cfg.get('moe_router_input_shape')}"
        train_loss = row["train"]["loss"] if row["train"] else float("nan")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['run']}`",
                    f"`{router}`",
                    format_float(train_loss),
                    format_float(row["metrics"]["overall"]["loss"]),
                    pct(row["metrics"]["overall"]["accuracy"]),
                    pct(row["metrics"]["inside_local"]["accuracy"]),
                    pct(row["metrics"]["local_boundary_not_high"]["accuracy"]),
                    pct(row["metrics"]["high_boundary"]["accuracy"]),
                    pct(row["mean_layers"]["local_slot"]["same_feature_same_expert_rate"]),
                    pct(row["mean_layers"]["higher_unit"]["same_feature_same_expert_rate"]),
                    format_float(row["mean_layers"]["effective_experts"]),
                    pct(row["mean_layers"]["attention_local_include_self_mass"]),
                    pct(row["mean_layers"]["attention_high_include_self_mass"]),
                    pct(row["mean_layers"]["attention_expert_mass"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `Local same-expert` / `High same-expert` 是同一 feature 内 token 被分到同一 expert bucket 的平均比例；head-router 按 head 平均，spectral 按 routed band 平均。")
    lines.append("- `Attn-local/high mass` 是 attention score 落在同 local/high slot 的比例，包含 self attention。")
    lines.append("- `Attn-expert mass` 是 attention score 落在同 expert bucket token 上的比例，表示 routing bucket 与 attention retrieval bucket 的重合程度。")
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    device = choose_device()
    dataset = make_dataset(args)
    runs = [x.strip() for x in args.runs.split(",") if x.strip()]
    results = [analyze_run(args, run, dataset, device) for run in runs]
    output = {"config": vars(args), "device": str(device), "runs": results}
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    write_markdown(args, results)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
