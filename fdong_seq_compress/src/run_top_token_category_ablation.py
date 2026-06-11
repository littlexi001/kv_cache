from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from geometry_metrics import select_indices  # noqa: E402
from model_loader import load_model_and_tokenizer  # noqa: E402


EPS = 1e-12
CATEGORIES = ("answer", "front", "end", "other")


def parse_float_list(value: str) -> List[float]:
    result = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not result or any(item <= 0 or item >= 1 for item in result):
        raise ValueError("Ratios must be comma-separated values in (0, 1).")
    return result


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_sample(path: Path, sample_id: str) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if str(item.get("sample_id")) == sample_id:
                return item
    raise ValueError(f"Sample not found: {sample_id} in {path}")


def build_prompt(sample: Dict) -> str:
    return (
        "Use the context to answer the question. Answer with the exact phrase when possible.\n\n"
        f"Context:\n{sample['context']}\n\n"
        f"Question: {sample['question']}\n"
        "Answer:"
    )


def answer_token_indices(prompt: str, answer: str, offsets: Sequence[Tuple[int, int]]) -> set[int]:
    result: set[int] = set()
    cursor = 0
    while answer:
        char_start = prompt.find(answer, cursor)
        if char_start < 0:
            break
        char_end = char_start + len(answer)
        for token_idx, (token_start, token_end) in enumerate(offsets):
            if token_start < char_end and token_end > char_start:
                result.add(token_idx)
        cursor = char_end
    return result


def compute_layer_qkv(model, hidden_states: torch.Tensor, layer_idx: int, position_embeddings):
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    with torch.no_grad():
        x = layer.input_layernorm(hidden_states)
        shape = (*x.shape[:-1], -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(x).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(shape)).transpose(1, 2)
        v = attn.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, position_embeddings[0], position_embeddings[1])
    return q[0], k[0], v[0]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= EPS:
        return float("nan")
    return float(torch.dot(a, b).div(denom).item())


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(EPS)).item())


def classify_indices(
    indices: torch.Tensor,
    visible_count: int,
    answer_indices: set[int],
    front_ratio: float,
    end_ratio: float,
) -> Dict[str, torch.Tensor]:
    front_count = max(1, math.ceil(front_ratio * visible_count))
    end_count = max(1, math.ceil(end_ratio * visible_count))
    buckets: Dict[str, List[int]] = {category: [] for category in CATEGORIES}
    for index in indices.tolist():
        if index in answer_indices:
            category = "answer"
        elif index < front_count:
            category = "front"
        elif index >= visible_count - end_count:
            category = "end"
        else:
            category = "other"
        buckets[category].append(index)
    return {category: torch.tensor(values, dtype=torch.long) for category, values in buckets.items()}


def subset_output(scores: torch.Tensor, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor | None:
    if indices.numel() == 0:
        return None
    selected_scores = scores[indices]
    selected_probs = torch.softmax(selected_scores, dim=-1)
    return selected_probs @ values[indices]


def aggregate(rows: Sequence[Dict], keys: Sequence[str], values: Sequence[str]) -> List[Dict]:
    groups: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group_key, members in groups.items():
        result = {key: value for key, value in zip(keys, group_key)}
        result["sample_count"] = len(members)
        for value in values:
            finite = [float(member[value]) for member in members if math.isfinite(float(member[value]))]
            result[value] = sum(finite) / len(finite) if finite else float("nan")
        output.append(result)
    return output


def analyze_query_head(
    scores: torch.Tensor,
    values: torch.Tensor,
    answer_indices: set[int],
    score_ratios: Sequence[float],
    position_ratios: Sequence[float],
    identity: Dict,
) -> Tuple[List[Dict], List[Dict]]:
    visible_count = scores.numel()
    full_output = torch.softmax(scores, dim=-1) @ values
    composition_rows: List[Dict] = []
    ablation_rows: List[Dict] = []

    for score_ratio in score_ratios:
        keep_count = max(1, math.ceil(score_ratio * visible_count))
        top_indices = torch.topk(scores, k=keep_count, largest=True, sorted=False).indices.cpu()
        top_output = subset_output(scores, values, top_indices)
        if top_output is None:
            continue
        full_probs = torch.softmax(scores, dim=-1)
        top_full_mass = float(full_probs[top_indices].sum().item())

        for position_ratio in position_ratios:
            buckets = classify_indices(
                top_indices,
                visible_count,
                answer_indices,
                position_ratio,
                position_ratio,
            )
            visible_buckets = classify_indices(
                torch.arange(visible_count, dtype=torch.long),
                visible_count,
                answer_indices,
                position_ratio,
                position_ratio,
            )
            for category in CATEGORIES:
                indices = buckets[category]
                category_visible_count = int(visible_buckets[category].numel())
                category_recall = (
                    float(indices.numel() / category_visible_count)
                    if category_visible_count > 0
                    else float("nan")
                )
                composition_rows.append(
                    {
                        **identity,
                        "score_ratio": score_ratio,
                        "position_front_ratio": position_ratio,
                        "position_end_ratio": position_ratio,
                        "category": category,
                        "category_count": int(indices.numel()),
                        "category_fraction_within_score_top": float(indices.numel() / keep_count),
                        "category_visible_count": category_visible_count,
                        "category_recall_in_score_top": category_recall,
                        "category_top_enrichment": category_recall / score_ratio,
                        "category_full_softmax_mass": float(full_probs[indices].sum().item()) if indices.numel() else 0.0,
                        "score_top_count": keep_count,
                        "score_top_full_softmax_mass": top_full_mass,
                    }
                )

            modes: Dict[str, torch.Tensor] = {"top_all": top_indices}
            for category in CATEGORIES:
                category_indices = buckets[category]
                modes[f"only_{category}"] = category_indices
                modes[f"without_{category}"] = torch.cat(
                    [buckets[other] for other in CATEGORIES if other != category]
                )

            for mode, selected_indices in modes.items():
                output = subset_output(scores, values, selected_indices)
                valid = output is not None
                ablation_rows.append(
                    {
                        **identity,
                        "score_ratio": score_ratio,
                        "position_front_ratio": position_ratio,
                        "position_end_ratio": position_ratio,
                        "mode": mode,
                        "selected_count": int(selected_indices.numel()),
                        "selected_fraction_of_visible": float(selected_indices.numel() / visible_count),
                        "valid": int(valid),
                        "cosine_to_score_top_all": cosine(output, top_output) if valid else float("nan"),
                        "relative_l2_to_score_top_all": relative_l2(output, top_output) if valid else float("nan"),
                        "cosine_to_full": cosine(output, full_output) if valid else float("nan"),
                        "relative_l2_to_full": relative_l2(output, full_output) if valid else float("nan"),
                        "norm_ratio_to_score_top_all": (
                            float(
                                (
                                    torch.linalg.vector_norm(output)
                                    / torch.linalg.vector_norm(top_output).clamp_min(EPS)
                                ).item()
                            )
                            if valid
                            else float("nan")
                        ),
                    }
                )
    return composition_rows, ablation_rows


def plot_composition(rows: Sequence[Dict], output_dir: Path) -> None:
    summary = aggregate(
        rows,
        ["layer", "score_ratio", "position_front_ratio", "category"],
        ["category_fraction_within_score_top", "category_recall_in_score_top"],
    )
    colors = {"answer": "#b42318", "front": "#35618d", "end": "#4e8b5e", "other": "#8b6f47"}
    configs = sorted({(row["score_ratio"], row["position_front_ratio"]) for row in summary})
    for score_ratio, position_ratio in configs:
        subset = [
            row
            for row in summary
            if row["score_ratio"] == score_ratio and row["position_front_ratio"] == position_ratio
        ]
        layers = sorted({int(row["layer"]) for row in subset})
        bottom = [0.0] * len(layers)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
        for category in CATEGORIES:
            by_layer = {int(row["layer"]): row["category_fraction_within_score_top"] for row in subset if row["category"] == category}
            values = [by_layer.get(layer, 0.0) for layer in layers]
            axes[0].bar(layers, values, bottom=bottom, label=category, color=colors[category])
            bottom = [left + right for left, right in zip(bottom, values)]
            recall_by_layer = {
                int(row["layer"]): row["category_recall_in_score_top"]
                for row in subset
                if row["category"] == category
            }
            axes[1].plot(
                layers,
                [recall_by_layer.get(layer, float("nan")) for layer in layers],
                marker="o",
                markersize=3,
                label=category,
                color=colors[category],
            )
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("P(category | score-top)")
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("P(score-top | category)")
        for axis in axes:
            axis.set_xlabel("Layer")
            axis.grid(alpha=0.25)
        axes[0].legend(ncol=4)
        axes[1].legend(ncol=2)
        fig.suptitle(f"Score top {100*score_ratio:g}%, position front/end {100*position_ratio:g}%")
        fig.tight_layout()
        score_slug = f"{100*score_ratio:g}".replace(".", "p")
        position_slug = f"{100*position_ratio:g}".replace(".", "p")
        name = f"composition_score{score_slug}_position{position_slug}.png"
        fig.savefig(output_dir / name, dpi=180)
        plt.close(fig)


def plot_ablation(rows: Sequence[Dict], output_dir: Path) -> None:
    summary = aggregate(
        rows,
        ["layer", "score_ratio", "position_front_ratio", "mode"],
        ["cosine_to_score_top_all", "relative_l2_to_score_top_all", "valid"],
    )
    configs = sorted({(row["score_ratio"], row["position_front_ratio"]) for row in summary})
    modes = [f"without_{category}" for category in CATEGORIES] + [f"only_{category}" for category in CATEGORIES]
    colors = ["#b42318", "#35618d", "#4e8b5e", "#8b6f47", "#d96c5f", "#6c91b5", "#78a985", "#aa916d"]
    for score_ratio, position_ratio in configs:
        subset = [
            row
            for row in summary
            if row["score_ratio"] == score_ratio and row["position_front_ratio"] == position_ratio
        ]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
        for mode, color in zip(modes, colors):
            mode_rows = sorted((row for row in subset if row["mode"] == mode), key=lambda row: row["layer"])
            axes[0].plot(
                [row["layer"] for row in mode_rows],
                [row["cosine_to_score_top_all"] for row in mode_rows],
                marker="o",
                markersize=3,
                label=mode,
                color=color,
            )
            axes[1].plot(
                [row["layer"] for row in mode_rows],
                [row["relative_l2_to_score_top_all"] for row in mode_rows],
                marker="o",
                markersize=3,
                label=mode,
                color=color,
            )
        axes[0].set_ylabel("Cosine to score-top-all output")
        axes[1].set_ylabel("Relative L2 to score-top-all output")
        for axis in axes:
            axis.set_xlabel("Layer")
            axis.grid(alpha=0.25)
        axes[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"Category ablation: score top {100*score_ratio:g}%, position front/end {100*position_ratio:g}%")
        fig.tight_layout()
        score_slug = f"{100*score_ratio:g}".replace(".", "p")
        position_slug = f"{100*position_ratio:g}".replace(".", "p")
        name = f"ablation_score{score_slug}_position{position_slug}.png"
        fig.savefig(output_dir / name, dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile and ablate semantic/position categories inside score-top attention tokens.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument(
        "--data-path",
        default="ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl",
    )
    parser.add_argument("--sample-id", default="niah_len2000_depth25")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--q-heads", default="all")
    parser.add_argument("--query-last-tokens", type=int, default=10)
    parser.add_argument("--score-ratios", default="0.01,0.02,0.04")
    parser.add_argument("--position-ratios", default="0.01,0.05,0.10")
    args = parser.parse_args()

    score_ratios = parse_float_list(args.score_ratios)
    position_ratios = parse_float_list(args.position_ratios)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"fdong_seq_compress/outputs/top_token_category_ablation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_sample(Path(args.data_path), args.sample_id)
    prompt = build_prompt(sample)
    tokenizer, model, device = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="eager",
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded.input_ids[0]
    offsets = [(int(start), int(end)) for start, end in encoded.offset_mapping[0].tolist()]
    seq_len = int(input_ids.numel())
    model_max = getattr(model.config, "max_position_embeddings", None)
    if model_max is not None and seq_len > int(model_max):
        raise ValueError(f"seq_len={seq_len} exceeds model max_position_embeddings={model_max}")
    answer_indices = answer_token_indices(prompt, str(sample["expected_answer"]), offsets)
    if not answer_indices:
        raise ValueError("Could not locate expected_answer in the tokenized prompt.")
    query_start = max(1, seq_len - args.query_last_tokens)
    query_indices = list(range(query_start, seq_len))

    print(
        f"device={device} sample={args.sample_id} seq_len={seq_len} "
        f"answer_tokens={len(answer_indices)} queries={query_indices}",
        flush=True,
    )
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids[None, :].to(device),
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden_states[0], position_ids)

    num_layers = int(model.config.num_hidden_layers)
    num_q_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    num_groups = num_q_heads // num_kv_heads
    layers = select_indices(num_layers, args.layers)
    q_heads = select_indices(num_q_heads, args.q_heads)

    composition_rows: List[Dict] = []
    ablation_rows: List[Dict] = []
    token_rows = [
        {
            "token_index": idx,
            "token_id": int(input_ids[idx]),
            "token_text": tokenizer.decode([int(input_ids[idx])], clean_up_tokenization_spaces=False).replace("\n", "\\n"),
            "is_answer": int(idx in answer_indices),
            "is_query": int(idx in query_indices),
        }
        for idx in range(seq_len)
    ]

    for layer_idx in layers:
        q, k, v = compute_layer_qkv(model, hidden_states[layer_idx], layer_idx, position_embeddings)
        q = q.detach().cpu().to(dtype=torch.float32)
        k = k.detach().cpu().to(dtype=torch.float32)
        v = v.detach().cpu().to(dtype=torch.float32)
        scaling = float(model.model.layers[layer_idx].self_attn.scaling)
        for q_head_idx in q_heads:
            kv_head_idx = q_head_idx // num_groups
            for query_idx in query_indices:
                scores = (q[q_head_idx, query_idx] @ k[kv_head_idx, : query_idx + 1].T) * scaling
                identity = {
                    "sample_id": args.sample_id,
                    "layer": layer_idx,
                    "q_head": q_head_idx,
                    "kv_head": kv_head_idx,
                    "query_index": query_idx,
                    "query_token_text": token_rows[query_idx]["token_text"],
                    "visible_count": query_idx + 1,
                }
                current_composition, current_ablation = analyze_query_head(
                    scores,
                    v[kv_head_idx, : query_idx + 1],
                    answer_indices,
                    score_ratios,
                    position_ratios,
                    identity,
                )
                composition_rows.extend(current_composition)
                ablation_rows.extend(current_ablation)
        print(f"finished layer={layer_idx}", flush=True)

    composition_summary = aggregate(
        composition_rows,
        ["layer", "score_ratio", "position_front_ratio", "position_end_ratio", "category"],
        [
            "category_count",
            "category_fraction_within_score_top",
            "category_visible_count",
            "category_recall_in_score_top",
            "category_top_enrichment",
            "category_full_softmax_mass",
            "score_top_full_softmax_mass",
        ],
    )
    ablation_summary = aggregate(
        ablation_rows,
        ["layer", "score_ratio", "position_front_ratio", "position_end_ratio", "mode"],
        [
            "selected_count",
            "selected_fraction_of_visible",
            "valid",
            "cosine_to_score_top_all",
            "relative_l2_to_score_top_all",
            "cosine_to_full",
            "relative_l2_to_full",
            "norm_ratio_to_score_top_all",
        ],
    )
    write_csv(output_dir / "tokens.csv", token_rows)
    write_csv(output_dir / "composition_by_query_head.csv", composition_rows)
    write_csv(output_dir / "composition_by_layer.csv", composition_summary)
    write_csv(output_dir / "category_ablation_by_query_head.csv", ablation_rows)
    write_csv(output_dir / "category_ablation_by_layer.csv", ablation_summary)
    plot_composition(composition_rows, output_dir)
    plot_ablation(ablation_rows, output_dir)

    metadata = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "sample_id": args.sample_id,
        "device": str(device),
        "seq_len": seq_len,
        "model_max_position_embeddings": model_max,
        "answer": sample["expected_answer"],
        "answer_token_indices": sorted(answer_indices),
        "query_indices": query_indices,
        "layers": layers,
        "q_heads": q_heads,
        "score_ratios": score_ratios,
        "position_ratios": position_ratios,
        "category_priority": ["answer", "front", "end", "other"],
        "claim_boundary": (
            "Category ablations measure immediate per-layer attention-output necessity/sufficiency. "
            "They do not yet prove end-to-end answer-loss causality."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
