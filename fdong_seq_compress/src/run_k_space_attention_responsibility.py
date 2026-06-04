from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from geometry_metrics import select_indices  # noqa: E402
from model_loader import load_model_and_tokenizer  # noqa: E402
from text_loader import decode_tokens, load_tokenized_text  # noqa: E402


EPS = 1e-12


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: Sequence[float], prefix: str) -> Dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
        }
    x = torch.tensor(finite, dtype=torch.float64)
    q = torch.quantile(x, torch.tensor([0.50, 0.90], dtype=torch.float64))
    return {
        f"{prefix}_count": int(x.numel()),
        f"{prefix}_mean": float(x.mean().item()),
        f"{prefix}_p50": float(q[0].item()),
        f"{prefix}_p90": float(q[1].item()),
    }


def parse_methods(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def selected_query_positions(seq_len: int, query_start: int, query_stride: int, max_queries: int) -> List[int]:
    start = max(1, min(query_start, seq_len - 1))
    positions = list(range(start, seq_len, max(1, query_stride)))
    if max_queries > 0:
        positions = positions[:max_queries]
    return positions


def run_forward(model, input_ids: torch.Tensor, device: torch.device):
    with torch.no_grad():
        return model.model(
            input_ids=input_ids[None, :].to(device),
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )


def compute_layer_qk(model, hidden_states: torch.Tensor, layer_idx: int, position_embeddings) -> Tuple[torch.Tensor, torch.Tensor]:
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    with torch.no_grad():
        x = layer.input_layernorm(hidden_states)
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(hidden_shape)).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, position_embeddings[0], position_embeddings[1])
    return q[0], k[0]


def kmeans(x: torch.Tensor, num_clusters: int, steps: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.detach().cpu().to(dtype=torch.float32)
    n = x.shape[0]
    k = min(max(1, num_clusters), n)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=gen)[:k]
    centers = x[perm].clone()
    assignment = torch.zeros(n, dtype=torch.long)
    for _ in range(max(1, steps)):
        distances = torch.cdist(x, centers, p=2)
        assignment = torch.argmin(distances, dim=1)
        for cid in range(k):
            members = x[assignment == cid]
            if members.numel() > 0:
                centers[cid] = members.mean(dim=0)
    distances = torch.cdist(x, centers, p=2)
    assignment = torch.argmin(distances, dim=1)
    return centers, assignment


def local_candidates(t: int, local_window: int) -> Set[int]:
    return set(range(max(0, t - max(1, local_window)), t))


def q_l2_nearest(q_t: torch.Tensor, k_past: torch.Tensor, max_candidates: int) -> Set[int]:
    distances = torch.linalg.vector_norm(k_past - q_t[None, :], dim=1)
    kk = min(max_candidates, distances.numel())
    return set(torch.topk(-distances, k=kk).indices.tolist())


def q_dot_nearest(q_t: torch.Tensor, k_past: torch.Tensor, max_candidates: int) -> Set[int]:
    scores = k_past @ q_t
    kk = min(max_candidates, scores.numel())
    return set(torch.topk(scores, k=kk).indices.tolist())


def cluster_candidates(
    q_t: torch.Tensor,
    centers: torch.Tensor,
    assignment: torch.Tensor,
    top_clusters: int,
    max_candidates: int,
    method: str,
) -> Tuple[Set[int], List[int]]:
    if method == "cluster_l2_topn":
        scores = -torch.linalg.vector_norm(centers - q_t[None, :], dim=1)
    elif method == "cluster_dot_topn":
        scores = centers @ q_t
    else:
        raise ValueError(f"Unsupported cluster method: {method}")
    kk = min(max(1, top_clusters), centers.shape[0])
    selected_clusters = torch.topk(scores, k=kk).indices.tolist()
    members: List[int] = []
    selected_set = set(selected_clusters)
    for idx, cid in enumerate(assignment.tolist()):
        if cid in selected_set:
            members.append(idx)
    if max_candidates > 0 and len(members) > max_candidates:
        # Keep a deterministic position-spread sample to avoid measuring only the earliest tokens.
        if max_candidates == 1:
            members = [members[-1]]
        else:
            step = (len(members) - 1) / float(max_candidates - 1)
            members = [members[int(round(i * step))] for i in range(max_candidates)]
    return set(members), selected_clusters


def attention_metrics(
    probs: torch.Tensor,
    candidates: Set[int],
    top_attention_ks: Sequence[int],
) -> Dict[str, float]:
    if probs.numel() == 0:
        return {}
    idx = torch.tensor(sorted(candidates), dtype=torch.long)
    mass = float(probs[idx].sum().item()) if idx.numel() else 0.0
    row = {
        "candidate_count": len(candidates),
        "candidate_ratio": len(candidates) / float(probs.numel()),
        "attention_mass_recall": mass,
        "missed_attention_mass": 1.0 - mass,
    }
    for k in top_attention_ks:
        kk = min(int(k), probs.numel())
        top_idx = set(torch.topk(probs, k=kk).indices.tolist())
        overlap = len(top_idx & candidates)
        row[f"top{k}_token_recall"] = overlap / float(kk)
        row[f"top{k}_all_in_candidate"] = 1.0 if overlap == kk else 0.0
    return row


def cluster_purity_metrics(probs: torch.Tensor, assignment: torch.Tensor, num_clusters: int) -> Dict[str, float]:
    cluster_mass = torch.zeros(num_clusters, dtype=torch.float32)
    cluster_mass.scatter_add_(0, assignment, probs.detach().cpu().to(dtype=torch.float32))
    sorted_mass, sorted_idx = torch.sort(cluster_mass, descending=True)
    entropy = -float((cluster_mass * torch.log(cluster_mass.clamp_min(EPS))).sum().item())
    effective_clusters = float(math.exp(entropy))
    return {
        "best_cluster_mass": float(sorted_mass[0].item()) if sorted_mass.numel() else float("nan"),
        "top2_cluster_mass": float(sorted_mass[:2].sum().item()) if sorted_mass.numel() else float("nan"),
        "top4_cluster_mass": float(sorted_mass[:4].sum().item()) if sorted_mass.numel() else float("nan"),
        "attention_effective_clusters": effective_clusters,
        "best_attention_cluster": int(sorted_idx[0].item()) if sorted_idx.numel() else -1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether K-space distance/clusters predict future qK attention responsibility.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument("--text-path", default="fdong_seq_compress/data/synthetic_texts/biomed_long_range_facts_hard_compact.txt")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--query-start", type=int, default=2500)
    parser.add_argument("--query-stride", type=int, default=16)
    parser.add_argument("--max-queries", type=int, default=32)
    parser.add_argument("--layers", default="0,13,27")
    parser.add_argument("--q-heads", default="0,4,8,12")
    parser.add_argument("--methods", default="local,q_l2_nearest,q_dot_nearest,cluster_l2_topn,cluster_dot_topn")
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--num-clusters", type=int, default=20)
    parser.add_argument("--top-clusters", type=int, default=2)
    parser.add_argument("--kmeans-steps", type=int, default=5)
    parser.add_argument("--top-attention-ks", default="1,5,10")
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"fdong_seq_compress/outputs/k_space_attention_responsibility_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    _, input_ids = load_tokenized_text(tokenizer, args.text_path, args.max_tokens)
    seq_len = int(input_ids.numel())
    outputs = run_forward(model, input_ids, device)
    hidden_states = outputs.hidden_states
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden_states[0], position_ids)

    num_layers = int(model.config.num_hidden_layers)
    num_q_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    num_groups = num_q_heads // num_kv_heads
    scaling = float(model.model.layers[0].self_attn.scaling)
    layer_indices = select_indices(num_layers, args.layers)
    q_head_indices = select_indices(num_q_heads, args.q_heads)
    query_positions = selected_query_positions(seq_len, args.query_start, args.query_stride, args.max_queries)
    methods = parse_methods(args.methods)
    top_attention_ks = [int(part) for part in args.top_attention_ks.split(",") if part.strip()]

    write_csv(output_dir / "tokens.csv", decode_tokens(tokenizer, input_ids))

    query_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    purity_rows: List[Dict] = []
    timing_rows: List[Dict] = []

    for layer_idx in layer_indices:
        layer_start = time.perf_counter()
        q, k = compute_layer_qk(model, hidden_states[layer_idx], layer_idx, position_embeddings)
        q_cpu = q.detach().cpu().to(dtype=torch.float32)
        k_cpu = k.detach().cpu().to(dtype=torch.float32)
        timing_rows.append({"event": "compute_layer_qk", "layer": layer_idx, "seconds": time.perf_counter() - layer_start})

        cluster_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for q_head_idx in q_head_indices:
            kv_head_idx = q_head_idx // num_groups
            if kv_head_idx >= num_kv_heads:
                continue
            if kv_head_idx not in cluster_cache:
                cluster_start = time.perf_counter()
                prefix_len = min(args.query_start, seq_len - 1)
                centers, assignment = kmeans(
                    k_cpu[kv_head_idx, :prefix_len],
                    args.num_clusters,
                    args.kmeans_steps,
                    args.random_seed + layer_idx * 1000 + kv_head_idx,
                )
                cluster_cache[kv_head_idx] = (centers, assignment)
                timing_rows.append(
                    {
                        "event": "kmeans",
                        "layer": layer_idx,
                        "kv_head": kv_head_idx,
                        "prefix_len": prefix_len,
                        "seconds": time.perf_counter() - cluster_start,
                    }
                )
            centers, assignment = cluster_cache[kv_head_idx]

            method_metrics: Dict[str, List[Dict[str, float]]] = {method: [] for method in methods}
            purity_metrics_all: List[Dict[str, float]] = []
            for t in query_positions:
                prefix_len = min(args.query_start, t)
                k_past = k_cpu[kv_head_idx, :prefix_len]
                q_t = q_cpu[q_head_idx, t]
                scores = (q_t @ k_past.T) * scaling
                probs = torch.softmax(scores, dim=-1)
                purity = cluster_purity_metrics(probs, assignment[:prefix_len], centers.shape[0])
                purity_row = {
                    "layer": layer_idx,
                    "q_head": q_head_idx,
                    "kv_head": kv_head_idx,
                    "query_pos": t,
                    "prefix_len": prefix_len,
                    "num_clusters": args.num_clusters,
                    **purity,
                }
                purity_rows.append(purity_row)
                purity_metrics_all.append(purity)

                for method in methods:
                    selected_clusters: List[int] = []
                    if method == "local":
                        candidates = {idx for idx in local_candidates(t, args.local_window) if idx < prefix_len}
                    elif method == "q_l2_nearest":
                        candidates = q_l2_nearest(q_t, k_past, args.max_candidates)
                    elif method == "q_dot_nearest":
                        candidates = q_dot_nearest(q_t, k_past, args.max_candidates)
                    elif method in {"cluster_l2_topn", "cluster_dot_topn"}:
                        candidates, selected_clusters = cluster_candidates(
                            q_t,
                            centers,
                            assignment[:prefix_len],
                            args.top_clusters,
                            args.max_candidates,
                            method,
                        )
                    else:
                        raise ValueError(f"Unsupported method: {method}")
                    metrics = attention_metrics(probs, candidates, top_attention_ks)
                    row = {
                        "layer": layer_idx,
                        "q_head": q_head_idx,
                        "kv_head": kv_head_idx,
                        "query_pos": t,
                        "prefix_len": prefix_len,
                        "method": method,
                        "seq_len": seq_len,
                        "local_window": args.local_window,
                        "max_candidates": args.max_candidates,
                        "num_clusters": args.num_clusters,
                        "top_clusters": args.top_clusters,
                        "selected_clusters": " ".join(str(x) for x in selected_clusters),
                        **metrics,
                    }
                    query_rows.append(row)
                    method_metrics[method].append(metrics)

            for method, rows in method_metrics.items():
                base = {
                    "layer": layer_idx,
                    "q_head": q_head_idx,
                    "kv_head": kv_head_idx,
                    "method": method,
                    "num_queries": len(rows),
                    "seq_len": seq_len,
                    "query_start": args.query_start,
                    "query_stride": args.query_stride,
                    "max_candidates": args.max_candidates,
                    "num_clusters": args.num_clusters,
                    "top_clusters": args.top_clusters,
                }
                for key in ["candidate_count", "candidate_ratio", "attention_mass_recall", "missed_attention_mass"]:
                    base.update(summarize([row.get(key, float("nan")) for row in rows], key))
                for top_k in top_attention_ks:
                    key = f"top{top_k}_token_recall"
                    base.update(summarize([row.get(key, float("nan")) for row in rows], key))
                    all_key = f"top{top_k}_all_in_candidate"
                    base.update(summarize([row.get(all_key, float("nan")) for row in rows], all_key))
                summary_rows.append(base)
                print(
                    f"layer={layer_idx} q_head={q_head_idx} method={method} "
                    f"mass={base['attention_mass_recall_mean']:.4f} top10={base.get('top10_token_recall_mean', float('nan')):.4f}",
                    flush=True,
                )

            purity_base = {
                "layer": layer_idx,
                "q_head": q_head_idx,
                "kv_head": kv_head_idx,
                "num_queries": len(purity_metrics_all),
                "num_clusters": args.num_clusters,
            }
            for key in ["best_cluster_mass", "top2_cluster_mass", "top4_cluster_mass", "attention_effective_clusters"]:
                purity_base.update(summarize([row.get(key, float("nan")) for row in purity_metrics_all], key))
            purity_base["method"] = "oracle_attention_cluster_purity"
            summary_rows.append(purity_base)

    write_csv(output_dir / "query_responsibility_by_method.csv", query_rows)
    write_csv(output_dir / "cluster_purity_by_query.csv", purity_rows)
    write_csv(output_dir / "summary_by_layer_head_method.csv", summary_rows)
    write_csv(output_dir / "timing.csv", timing_rows)
    summary = {
        "model_path": args.model_path,
        "text_path": args.text_path,
        "output_dir": str(output_dir),
        "device": str(device),
        "dtype": args.dtype,
        "seq_len": seq_len,
        "query_start": args.query_start,
        "query_stride": args.query_stride,
        "max_queries": args.max_queries,
        "layers": layer_indices,
        "q_heads": q_head_indices,
        "methods": methods,
        "max_candidates": args.max_candidates,
        "num_clusters": args.num_clusters,
        "top_clusters": args.top_clusters,
        "kmeans_steps": args.kmeans_steps,
        "num_query_rows": len(query_rows),
        "num_summary_rows": len(summary_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
