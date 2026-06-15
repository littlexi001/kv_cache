from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from geometry_metrics import select_indices  # noqa: E402
from model_loader import load_model_and_tokenizer  # noqa: E402
from text_loader import load_tokenized_text  # noqa: E402


EPS = 1e-12


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


def finite_mean(values: Iterable[float]) -> float:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(valid) / len(valid)) if valid else float("nan")


def summarize(values: Sequence[float]) -> Dict[str, float]:
    valid = torch.tensor(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=torch.float64,
    )
    if valid.numel() == 0:
        return {"count": 0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "count": int(valid.numel()),
        "mean": float(valid.mean().item()),
        "p50": float(torch.quantile(valid, 0.5).item()),
        "p90": float(torch.quantile(valid, 0.9).item()),
    }


def cosine_matrix_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=EPS)


def compute_layer_qk(
    model,
    hidden_states: torch.Tensor,
    layer_idx: int,
    position_embeddings,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    with torch.no_grad():
        x = layer.input_layernorm(hidden_states)
        hidden_shape = (*x.shape[:-1], -1, attn.head_dim)
        q_pre = attn.q_norm(attn.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k_pre = attn.k_norm(attn.k_proj(x).view(hidden_shape)).transpose(1, 2)
        q_post, k_post = apply_rotary_pos_emb(
            q_pre,
            k_pre,
            position_embeddings[0],
            position_embeddings[1],
        )
    return q_pre[0], k_pre[0], q_post[0], k_post[0]


def weight_kernel_metrics(model, layer_idx: int, q_head_idx: int, kv_head_idx: int, seed: int) -> Dict:
    attn = model.model.layers[layer_idx].self_attn
    head_dim = int(attn.head_dim)
    q_weight = attn.q_proj.weight.detach().cpu().float()[
        q_head_idx * head_dim : (q_head_idx + 1) * head_dim
    ]
    k_weight = attn.k_proj.weight.detach().cpu().float()[
        kv_head_idx * head_dim : (kv_head_idx + 1) * head_dim
    ]
    kernel = q_weight.T @ k_weight
    kernel_t = kernel.T
    kernel_norm = torch.linalg.vector_norm(kernel).clamp_min(EPS)
    symmetry_cosine = float((kernel * kernel_t).sum().div(kernel_norm.pow(2)).item())
    skew_ratio = float(torch.linalg.vector_norm(kernel - kernel_t).div(kernel_norm).item())

    q_basis = torch.linalg.qr(q_weight.T, mode="reduced").Q
    k_basis = torch.linalg.qr(k_weight.T, mode="reduced").Q
    principal_cosines = torch.linalg.svdvals(q_basis.T @ k_basis)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    probes = F.normalize(torch.randn(256, kernel.shape[0], generator=generator), dim=-1)
    quadratic = torch.einsum("bi,ij,bj->b", probes, kernel, probes)
    negative = quadratic[quadratic < 0]
    positive = quadratic[quadratic > 0]
    negative_energy = negative.abs().sum()
    total_energy = negative.abs().sum() + positive.abs().sum()

    return {
        "layer": layer_idx,
        "q_head": q_head_idx,
        "kv_head": kv_head_idx,
        "kernel_symmetry_cosine": symmetry_cosine,
        "kernel_skew_ratio": skew_ratio,
        "negative_quadratic_fraction": float((quadratic < 0).float().mean().item()),
        "negative_quadratic_energy_fraction": float((negative_energy / total_energy.clamp_min(EPS)).item()),
        "rowspace_principal_cosine_mean": float(principal_cosines.mean().item()),
        "rowspace_principal_cosine_min": float(principal_cosines.min().item()),
        "rowspace_principal_cosine_max": float(principal_cosines.max().item()),
    }


def relation_size(history_len: int, top_k: int, top_ratio: float) -> int:
    if top_ratio > 0.0:
        return min(history_len, max(1, int(math.ceil(history_len * top_ratio))))
    return min(top_k, history_len)


def strict_history_topk(scores: torch.Tensor, top_k: int, top_ratio: float) -> List[torch.Tensor]:
    seq_len = scores.shape[0]
    result: List[torch.Tensor] = []
    for query_idx in range(seq_len):
        if query_idx == 0:
            result.append(torch.empty(0, dtype=torch.long))
            continue
        k = relation_size(query_idx, top_k, top_ratio)
        result.append(torch.topk(scores[query_idx, :query_idx], k=k).indices.cpu())
    return result


def relation_threshold_metrics(
    scores: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    top_indices: Sequence[torch.Tensor],
    query_indices: Sequence[int],
    identity: Dict,
) -> Dict:
    threshold_scores: List[float] = []
    threshold_z: List[float] = []
    threshold_cosines: List[float] = []
    selected_mass: List[float] = []
    selected_ratios: List[float] = []
    for query_idx in query_indices:
        selected = top_indices[query_idx]
        if selected.numel() == 0:
            continue
        row = scores[query_idx, :query_idx]
        row_mean = row.mean()
        row_std = row.std(unbiased=False).clamp_min(EPS)
        selected_scores = row[selected]
        threshold = selected_scores.min()
        threshold_scores.append(float(threshold.item()))
        threshold_z.append(float(((threshold - row_mean) / row_std).item()))
        q_vec = q[query_idx]
        k_selected = k[selected]
        cosines = F.cosine_similarity(q_vec[None, :], k_selected, dim=-1)
        threshold_cosines.append(float(cosines.min().item()))
        probs = torch.softmax(row, dim=-1)
        selected_mass.append(float(probs[selected].sum().item()))
        selected_ratios.append(float(selected.numel() / query_idx))
    return {
        **identity,
        "relation_score_threshold_mean": finite_mean(threshold_scores),
        "relation_score_threshold_z_mean": finite_mean(threshold_z),
        "relation_cosine_threshold_mean": finite_mean(threshold_cosines),
        "relation_selected_mass_mean": finite_mean(selected_mass),
        "relation_selected_ratio_mean": finite_mean(selected_ratios),
    }


def log_distance_bucket(distance: int) -> int:
    return int(math.floor(math.log2(max(1, distance))))


def distance_matched_candidates(parent_idx: int, endpoint_idx: int) -> List[int]:
    bucket = log_distance_bucket(parent_idx - endpoint_idx)
    low = 1 << bucket
    high = (1 << (bucket + 1)) - 1
    start = max(0, parent_idx - high)
    stop = parent_idx - low
    return list(range(start, stop + 1)) if stop >= start else []


def indegree_buckets(top_indices: Sequence[torch.Tensor], num_buckets: int = 5) -> Tuple[List[int], List[int]]:
    indegree = [0 for _ in top_indices]
    for neighbors in top_indices:
        for token_idx in neighbors.tolist():
            indegree[token_idx] += 1
    values = torch.tensor(indegree, dtype=torch.float32)
    boundaries = torch.quantile(values, torch.linspace(0.0, 1.0, num_buckets + 1)[1:-1]).tolist()
    buckets = [sum(value > boundary for boundary in boundaries) for value in indegree]
    return indegree, buckets


def percentile_in_history(scores: torch.Tensor, query_idx: int, token_idx: int) -> float:
    row = scores[query_idx, :query_idx]
    if row.numel() == 0:
        return float("nan")
    return float((row <= row[token_idx]).float().mean().item())


def analyze_two_hop_closure(
    scores: torch.Tensor,
    top_indices: List[torch.Tensor],
    query_indices: Sequence[int],
    distance_samples: int,
    rng: random.Random,
    identity: Dict,
) -> Tuple[Dict, List[Dict]]:
    closure: List[float] = []
    uniform_baselines: List[float] = []
    distance_baselines: List[float] = []
    distance_popularity_baselines: List[float] = []
    endpoint_percentiles: List[float] = []
    endpoint_probs: List[float] = []
    path_rows: List[Dict] = []

    probs_cache: Dict[int, torch.Tensor] = {}
    indegree, popularity_bucket = indegree_buckets(top_indices)
    for later_idx in query_indices:
        later_top = set(top_indices[later_idx].tolist())
        if not later_top:
            continue
        if later_idx not in probs_cache:
            probs_cache[later_idx] = torch.softmax(scores[later_idx, :later_idx], dim=-1)
        later_probs = probs_cache[later_idx]
        for middle_idx in top_indices[later_idx].tolist():
            if middle_idx <= 0:
                continue
            middle_top = top_indices[middle_idx].tolist()
            if not middle_top:
                continue
            uniform_baseline = len([idx for idx in later_top if idx < middle_idx]) / float(middle_idx)
            for endpoint_idx in middle_top:
                is_closed = 1.0 if endpoint_idx in later_top else 0.0
                closure.append(is_closed)
                uniform_baselines.append(uniform_baseline)
                endpoint_percentiles.append(percentile_in_history(scores, later_idx, endpoint_idx))
                endpoint_probs.append(float(later_probs[endpoint_idx].item()))

                candidates = distance_matched_candidates(middle_idx, endpoint_idx)
                candidates = [candidate for candidate in candidates if candidate != endpoint_idx]
                matched_values: List[float] = []
                if candidates:
                    for _ in range(distance_samples):
                        matched_values.append(1.0 if rng.choice(candidates) in later_top else 0.0)
                distance_baselines.append(finite_mean(matched_values))

                popularity_candidates = [
                    candidate
                    for candidate in candidates
                    if popularity_bucket[candidate] == popularity_bucket[endpoint_idx]
                ]
                popularity_values: List[float] = []
                if popularity_candidates:
                    for _ in range(distance_samples):
                        popularity_values.append(1.0 if rng.choice(popularity_candidates) in later_top else 0.0)
                distance_popularity_baselines.append(finite_mean(popularity_values))

                if len(path_rows) < 200:
                    path_rows.append(
                        {
                            **identity,
                            "later_index": later_idx,
                            "middle_index": middle_idx,
                            "endpoint_index": endpoint_idx,
                            "closed": is_closed,
                            "endpoint_percentile_in_later_row": endpoint_percentiles[-1],
                            "endpoint_attention_probability": endpoint_probs[-1],
                            "uniform_baseline": uniform_baseline,
                            "distance_matched_baseline": distance_baselines[-1],
                            "distance_popularity_matched_baseline": distance_popularity_baselines[-1],
                            "endpoint_indegree": indegree[endpoint_idx],
                        }
                    )

    closure_rate = finite_mean(closure)
    uniform_rate = finite_mean(uniform_baselines)
    distance_rate = finite_mean(distance_baselines)
    distance_popularity_rate = finite_mean(distance_popularity_baselines)
    return (
        {
            **identity,
            "num_two_hop_paths": len(closure),
            "closure_rate": closure_rate,
            "uniform_baseline": uniform_rate,
            "distance_matched_baseline": distance_rate,
            "distance_popularity_matched_baseline": distance_popularity_rate,
            "closure_lift_vs_uniform": closure_rate / max(uniform_rate, EPS),
            "closure_lift_vs_distance": closure_rate / max(distance_rate, EPS),
            "closure_lift_vs_distance_popularity": closure_rate / max(distance_popularity_rate, EPS),
            "endpoint_percentile_mean": finite_mean(endpoint_percentiles),
            "endpoint_attention_probability_mean": finite_mean(endpoint_probs),
        },
        path_rows,
    )


def jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    return len(a & b) / float(len(union)) if union else float("nan")


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.double().clamp_min(EPS)
    q = q.double().clamp_min(EPS)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float((0.5 * (p * (p.log() - m.log())).sum() + 0.5 * (q * (q.log() - m.log())).sum()).item())


def retrieval_pair_metrics(
    scores: torch.Tensor,
    first_idx: int,
    second_idx: int,
    top_k: int,
    top_ratio: float,
) -> Tuple[float, float]:
    common_len = min(first_idx, second_idx)
    if common_len <= 1:
        return float("nan"), float("nan")
    k = relation_size(common_len, top_k, top_ratio)
    first_scores = scores[first_idx, :common_len]
    second_scores = scores[second_idx, :common_len]
    first_set = set(torch.topk(first_scores, k=k).indices.tolist())
    second_set = set(torch.topk(second_scores, k=k).indices.tolist())
    return jaccard(first_set, second_set), js_divergence(torch.softmax(first_scores, -1), torch.softmax(second_scores, -1))


def analyze_query_stability(
    q: torch.Tensor,
    scores: torch.Tensor,
    query_indices: Sequence[int],
    top_k: int,
    top_ratio: float,
    rng: random.Random,
    identity: Dict,
) -> Dict:
    q_norm = cosine_matrix_rows(q.cpu().float())
    nearest_cosines: List[float] = []
    nearest_jaccards: List[float] = []
    nearest_js: List[float] = []
    random_jaccards: List[float] = []
    random_js: List[float] = []

    for query_idx in query_indices:
        if query_idx < 2:
            continue
        similarities = q_norm[:query_idx] @ q_norm[query_idx]
        nearest_idx = int(torch.argmax(similarities).item())
        nearest_cosines.append(float(similarities[nearest_idx].item()))
        near_jaccard, near_js = retrieval_pair_metrics(scores, nearest_idx, query_idx, top_k, top_ratio)
        nearest_jaccards.append(near_jaccard)
        nearest_js.append(near_js)

        distance = query_idx - nearest_idx
        bucket = log_distance_bucket(distance)
        low = 1 << bucket
        high = min(query_idx, (1 << (bucket + 1)) - 1)
        candidates = list(range(max(1, query_idx - high), max(1, query_idx - low + 1)))
        candidates = [candidate for candidate in candidates if candidate != nearest_idx]
        if not candidates:
            candidates = [candidate for candidate in range(1, query_idx) if candidate != nearest_idx]
        if candidates:
            random_idx = rng.choice(candidates)
            random_jaccard, random_js_value = retrieval_pair_metrics(scores, random_idx, query_idx, top_k, top_ratio)
            random_jaccards.append(random_jaccard)
            random_js.append(random_js_value)

    return {
        **identity,
        "num_query_pairs": len(nearest_jaccards),
        "nearest_query_cosine_mean": finite_mean(nearest_cosines),
        "nearest_query_retrieval_jaccard": finite_mean(nearest_jaccards),
        "distance_matched_random_jaccard": finite_mean(random_jaccards),
        "retrieval_jaccard_lift": finite_mean(nearest_jaccards) / max(finite_mean(random_jaccards), EPS),
        "nearest_query_attention_js": finite_mean(nearest_js),
        "distance_matched_random_attention_js": finite_mean(random_js),
    }


def plot_attention_heatmap(scores: torch.Tensor, output_path: Path, title: str, window: int) -> None:
    size = min(window, scores.shape[0])
    sub = scores[-size:, -size:].clone()
    causal = torch.tril(torch.ones_like(sub, dtype=torch.bool))
    sub = sub.masked_fill(~causal, float("nan"))
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(sub.numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set_title(title)
    axis.set_xlabel("Key position in final window")
    axis.set_ylabel("Query position in final window")
    fig.colorbar(image, ax=axis, label="scaled QK score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def aggregate_by_layer(rows: Sequence[Dict], metric_names: Sequence[str]) -> List[Dict]:
    layers = sorted({int(row["layer"]) for row in rows})
    result: List[Dict] = []
    for layer in layers:
        layer_rows = [row for row in rows if int(row["layer"]) == layer]
        aggregate = {"layer": layer, "num_heads": len(layer_rows)}
        for metric in metric_names:
            aggregate[metric] = finite_mean([float(row[metric]) for row in layer_rows])
        result.append(aggregate)
    return result


def render_results(
    path: Path,
    config: Dict,
    weight_summary: List[Dict],
    closure_summary: List[Dict],
    stability_summary: List[Dict],
) -> None:
    lines = [
        "# QK Attention Transitivity: Results",
        "",
        "## Setup",
        "",
        f"- Model: `{config['model_path']}`",
        f"- Text: `{config['text_path']}`",
        f"- Sequence length: `{config['seq_len']}`",
        f"- Layers: `{config['layers']}`",
        f"- Query heads: `{config['q_heads']}`",
        f"- Strict-history top-k: `{config['top_k']}`",
        "",
        "## Weight-kernel diagnostics",
        "",
        "| layer | symmetry cosine | skew ratio | negative quadratic fraction | row-space cosine |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in weight_summary:
        lines.append(
            f"| {row['layer']} | {row['kernel_symmetry_cosine']:.3f} | {row['kernel_skew_ratio']:.3f} | "
            f"{row['negative_quadratic_fraction']:.3f} | {row['rowspace_principal_cosine_mean']:.3f} |"
        )
    lines += [
        "",
        "## Directed two-hop closure",
        "",
        "| layer | closure | uniform baseline | distance baseline | distance+popularity baseline | lift vs strongest baseline | endpoint percentile |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in closure_summary:
        lines.append(
            f"| {row['layer']} | {row['closure_rate']:.3f} | {row['uniform_baseline']:.3f} | "
            f"{row['distance_matched_baseline']:.3f} | {row['distance_popularity_matched_baseline']:.3f} | "
            f"{row['closure_lift_vs_distance_popularity']:.2f} | "
            f"{row['endpoint_percentile_mean']:.3f} |"
        )
    lines += [
        "",
        "## Similar-query retrieval stability",
        "",
        "| layer | nearest query cosine | retrieval Jaccard | random Jaccard | Jaccard lift | attention JS | random JS |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stability_summary:
        lines.append(
            f"| {row['layer']} | {row['nearest_query_cosine_mean']:.3f} | "
            f"{row['nearest_query_retrieval_jaccard']:.3f} | {row['distance_matched_random_jaccard']:.3f} | "
            f"{row['retrieval_jaccard_lift']:.2f} | {row['nearest_query_attention_js']:.3f} | "
            f"{row['distance_matched_random_attention_js']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation contract",
        "",
        "- Closure lift must be judged against the distance-matched baseline, not against zero.",
        "- A block-shaped heatmap alone does not prove semantic transitivity.",
        "- Positive results support local QK bucket geometry; they do not prove that Attention and MoE must share one partition.",
        "- Inspect per-head CSV files before making a model-wide claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure directed QK-attention transitivity in Qwen3.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument("--text-path", default="fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--layers", default="0,13,27")
    parser.add_argument("--q-heads", default="all")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--top-ratio", type=float, default=0.0)
    parser.add_argument("--min-query-index", type=int, default=128)
    parser.add_argument("--query-stride", type=int, default=8)
    parser.add_argument("--distance-samples", type=int, default=4)
    parser.add_argument("--heatmap-window", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"fdong_seq_compress/outputs/qk_attention_transitivity_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="eager",
    )
    _, input_ids = load_tokenized_text(tokenizer, args.text_path, args.max_tokens)
    seq_len = int(input_ids.numel())
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
    query_indices = list(range(max(args.min_query_index, args.top_k + 2), seq_len, max(1, args.query_stride)))
    rng = random.Random(args.seed)

    weight_rows: List[Dict] = []
    closure_rows: List[Dict] = []
    stability_rows: List[Dict] = []
    path_rows: List[Dict] = []

    print(f"device={device} seq_len={seq_len} layers={layers} q_heads={q_heads}", flush=True)
    for layer_idx in layers:
        q_pre, k_pre, q_post, k_post = compute_layer_qk(
            model,
            hidden_states[layer_idx],
            layer_idx,
            position_embeddings,
        )
        q_post = q_post.detach().cpu().float()
        k_post = k_post.detach().cpu().float()
        scaling = float(model.model.layers[layer_idx].self_attn.scaling)

        for q_head_idx in q_heads:
            kv_head_idx = q_head_idx // num_groups
            identity = {"layer": layer_idx, "q_head": q_head_idx, "kv_head": kv_head_idx}
            weight_rows.append(
                weight_kernel_metrics(
                    model,
                    layer_idx,
                    q_head_idx,
                    kv_head_idx,
                    args.seed + layer_idx * 1000 + q_head_idx,
                )
            )
            scores = (q_post[q_head_idx] @ k_post[kv_head_idx].T) * scaling
            top_indices = strict_history_topk(scores, args.top_k, args.top_ratio)
            closure_row, representative_paths = analyze_two_hop_closure(
                scores,
                top_indices,
                query_indices,
                args.distance_samples,
                rng,
                identity,
            )
            closure_rows.append(closure_row)
            closure_row.update(
                relation_threshold_metrics(
                    scores,
                    q_post[q_head_idx],
                    k_post[kv_head_idx],
                    top_indices,
                    query_indices,
                    identity,
                )
            )
            path_rows.extend(representative_paths)
            stability_rows.append(
                analyze_query_stability(
                    q_post[q_head_idx],
                    scores,
                    query_indices,
                    args.top_k,
                    args.top_ratio,
                    rng,
                    identity,
                )
            )
            if q_head_idx == q_heads[0]:
                plot_attention_heatmap(
                    scores,
                    output_dir / f"attention_heatmap_layer{layer_idx}_head{q_head_idx}.png",
                    f"Qwen3-0.6B layer {layer_idx}, query head {q_head_idx}",
                    args.heatmap_window,
                )
        print(f"finished layer={layer_idx}", flush=True)

    weight_metrics = [
        "kernel_symmetry_cosine",
        "kernel_skew_ratio",
        "negative_quadratic_fraction",
        "negative_quadratic_energy_fraction",
        "rowspace_principal_cosine_mean",
    ]
    closure_metrics = [
        "closure_rate",
        "uniform_baseline",
        "distance_matched_baseline",
        "distance_popularity_matched_baseline",
        "closure_lift_vs_uniform",
        "closure_lift_vs_distance",
        "closure_lift_vs_distance_popularity",
        "endpoint_percentile_mean",
        "endpoint_attention_probability_mean",
        "relation_score_threshold_mean",
        "relation_score_threshold_z_mean",
        "relation_cosine_threshold_mean",
        "relation_selected_mass_mean",
        "relation_selected_ratio_mean",
    ]
    stability_metrics = [
        "nearest_query_cosine_mean",
        "nearest_query_retrieval_jaccard",
        "distance_matched_random_jaccard",
        "retrieval_jaccard_lift",
        "nearest_query_attention_js",
        "distance_matched_random_attention_js",
    ]
    weight_summary = aggregate_by_layer(weight_rows, weight_metrics)
    closure_summary = aggregate_by_layer(closure_rows, closure_metrics)
    stability_summary = aggregate_by_layer(stability_rows, stability_metrics)

    config = {
        "model_path": args.model_path,
        "text_path": args.text_path,
        "seq_len": seq_len,
        "layers": layers,
        "q_heads": q_heads,
        "top_k": args.top_k,
        "top_ratio": args.top_ratio,
        "min_query_index": args.min_query_index,
        "query_stride": args.query_stride,
        "distance_samples": args.distance_samples,
        "device": str(device),
    }
    write_csv(output_dir / "weight_kernel_metrics.csv", weight_rows)
    write_csv(output_dir / "transitivity_metrics.csv", closure_rows)
    write_csv(output_dir / "query_stability_metrics.csv", stability_rows)
    write_csv(output_dir / "representative_paths.csv", path_rows)
    write_csv(output_dir / "weight_kernel_summary_by_layer.csv", weight_summary)
    write_csv(output_dir / "transitivity_summary_by_layer.csv", closure_summary)
    write_csv(output_dir / "query_stability_summary_by_layer.csv", stability_summary)
    summary = {
        "config": config,
        "weight_kernel_summary_by_layer": weight_summary,
        "transitivity_summary_by_layer": closure_summary,
        "query_stability_summary_by_layer": stability_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_results(
        output_dir / "visualization_results.md",
        config,
        weight_summary,
        closure_summary,
        stability_summary,
    )
    print(f"saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
