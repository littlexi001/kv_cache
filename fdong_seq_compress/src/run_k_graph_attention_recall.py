from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch
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
            f"{prefix}_p95": float("nan"),
        }
    x = torch.tensor(finite, dtype=torch.float64)
    q = torch.quantile(x, torch.tensor([0.50, 0.90, 0.95], dtype=torch.float64))
    return {
        f"{prefix}_count": int(x.numel()),
        f"{prefix}_mean": float(x.mean().item()),
        f"{prefix}_p50": float(q[0].item()),
        f"{prefix}_p90": float(q[1].item()),
        f"{prefix}_p95": float(q[2].item()),
    }


def parse_methods(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_positions(value: str) -> Set[int]:
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def parse_position_spec(value: str) -> Set[int]:
    positions: Set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            left, right = [int(x.strip()) for x in part.split(":", 1)]
            if right < left:
                raise ValueError(f"Invalid position range: {part}")
            positions.update(range(left, right))
        else:
            positions.add(int(part))
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


def selected_query_positions(seq_len: int, decode_start: int, query_stride: int, max_queries: int) -> List[int]:
    start = max(1, min(decode_start, seq_len - 1))
    positions = list(range(start, seq_len, max(1, query_stride)))
    if max_queries > 0:
        positions = positions[:max_queries]
    return positions


def local_candidates(t: int, window: int) -> Set[int]:
    left = max(0, t - max(1, window))
    return set(range(left, t))


def build_k_graph(kh: torch.Tensor, graph_top_k: int, similarity: str) -> Tuple[List[List[int]], List[List[int]]]:
    n = kh.shape[0]
    if n < 2:
        return [[] for _ in range(n)], [[] for _ in range(n)]

    x = kh.detach().cpu().to(dtype=torch.float32)
    if similarity == "cos":
        x = x - x.mean(dim=0, keepdim=True)
        x = torch.nn.functional.normalize(x, p=2, dim=1, eps=EPS)
        scores = x @ x.T
        larger_is_better = True
    elif similarity == "l2":
        sq_norm = x.square().sum(dim=1, keepdim=True)
        scores = (sq_norm + sq_norm.T - 2.0 * (x @ x.T)).clamp_min(0.0).sqrt()
        larger_is_better = False
    else:
        raise ValueError(f"Unsupported similarity: {similarity}")

    indices = torch.arange(n)
    invalid = indices[None, :] >= indices[:, None]
    scores = scores.masked_fill(invalid, -float("inf") if larger_is_better else float("inf"))

    actual_k = min(max(1, graph_top_k), n - 1)
    if larger_is_better:
        values, neighbor_idx = torch.topk(scores, k=actual_k, dim=1)
    else:
        neg_values, neighbor_idx = torch.topk(-scores, k=actual_k, dim=1)
        values = -neg_values

    out_neighbors: List[List[int]] = [[] for _ in range(n)]
    in_neighbors: List[List[int]] = [[] for _ in range(n)]
    for src in range(n):
        for rank in range(actual_k):
            value = float(values[src, rank].item())
            if not math.isfinite(value):
                continue
            dst = int(neighbor_idx[src, rank].item())
            out_neighbors[src].append(dst)
            in_neighbors[dst].append(src)
    return out_neighbors, in_neighbors


def expand_graph(
    seeds: Iterable[int],
    t: int,
    out_neighbors: List[List[int]],
    in_neighbors: List[List[int]],
    hops: int,
    direction: str,
    max_nodes: int,
) -> Set[int]:
    visited: Set[int] = {node for node in seeds if 0 <= node < t}
    frontier: Set[int] = set(visited)
    for _ in range(max(0, hops)):
        next_frontier: Set[int] = set()
        for node in frontier:
            neighbors: List[int] = []
            if direction in {"out", "both"}:
                neighbors.extend(out_neighbors[node])
            if direction in {"in", "both"}:
                neighbors.extend(in_neighbors[node])
            for nb in neighbors:
                if 0 <= nb < t and nb not in visited:
                    next_frontier.add(nb)
        visited.update(next_frontier)
        frontier = next_frontier
        if max_nodes > 0 and len(visited) >= max_nodes:
            break
        if not frontier:
            break
    if max_nodes > 0 and len(visited) > max_nodes:
        return set(sorted(visited)[:max_nodes])
    return visited


def qk_attention(q_t: torch.Tensor, k_past: torch.Tensor, scaling: float) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = (q_t.detach().cpu().to(dtype=torch.float32) @ k_past.detach().cpu().to(dtype=torch.float32).T) * scaling
    probs = torch.softmax(scores, dim=-1)
    return scores, probs


def topq_seeds(scores: torch.Tensor, t: int, window: int, seed_count: int) -> Set[int]:
    local = sorted(local_candidates(t, window))
    if not local:
        return set()
    local_scores = scores[torch.tensor(local, dtype=torch.long)]
    k = min(max(1, seed_count), len(local))
    top_local = torch.topk(local_scores, k=k).indices.tolist()
    return {local[idx] for idx in top_local}


def candidate_set_for_method(
    method: str,
    t: int,
    scores: torch.Tensor,
    local_window: int,
    seed_count: int,
    out_neighbors: List[List[int]],
    in_neighbors: List[List[int]],
    graph_hops: int,
    graph_direction: str,
    max_candidates: int,
    always_include_positions: Set[int],
    rng: random.Random,
) -> Set[int]:
    local = local_candidates(t, local_window)
    always = {idx for idx in always_include_positions if 0 <= idx < t}
    if method == "local":
        candidates = local
    elif method == "random_local_size":
        size = min(len(local), t)
        candidates = set(rng.sample(range(t), k=size)) if size > 0 else set()
    elif method == "random_max_candidates":
        size = min(max_candidates if max_candidates > 0 else len(local), t)
        candidates = set(rng.sample(range(t), k=size)) if size > 0 else set()
    elif method == "local_graph_all":
        candidates = expand_graph(local, t, out_neighbors, in_neighbors, graph_hops, graph_direction, 0)
    elif method == "local_topq_graph":
        seeds = topq_seeds(scores, t, local_window, seed_count)
        candidates = local | expand_graph(seeds, t, out_neighbors, in_neighbors, graph_hops, graph_direction, 0)
    elif method == "topq_local":
        candidates = topq_seeds(scores, t, local_window, seed_count)
    else:
        raise ValueError(f"Unsupported method: {method}")

    candidates.update(always)
    if max_candidates > 0 and len(candidates) > max_candidates:
        always_keep = sorted(always & candidates)
        local_keep = sorted((local & candidates) - set(always_keep))
        remaining = sorted(candidates - set(always_keep) - set(local_keep))
        budget_after_always = max(0, max_candidates - len(always_keep))
        trimmed = always_keep + local_keep[:budget_after_always]
        if len(trimmed) < max_candidates:
            trimmed += remaining[: max(0, max_candidates - len(trimmed))]
        candidates = set(trimmed[:max_candidates])
    return {idx for idx in candidates if 0 <= idx < t}


def evaluate_candidates(probs: torch.Tensor, candidates: Set[int], top_ks: Sequence[int], sink_positions: Set[int]) -> Dict[str, float]:
    if probs.numel() == 0:
        return {}
    idx = torch.tensor(sorted(candidates), dtype=torch.long)
    mass = float(probs[idx].sum().item()) if idx.numel() else 0.0
    valid_sink_positions = {pos for pos in sink_positions if 0 <= pos < probs.numel()}
    sink_idx = torch.tensor(sorted(valid_sink_positions), dtype=torch.long)
    sink_mass = float(probs[sink_idx].sum().item()) if sink_idx.numel() else 0.0
    nonsink_denom = max(EPS, 1.0 - sink_mass)
    nonsink_candidates = sorted(candidates - valid_sink_positions)
    nonsink_idx = torch.tensor(nonsink_candidates, dtype=torch.long)
    nonsink_mass = float(probs[nonsink_idx].sum().item()) if nonsink_idx.numel() else 0.0
    row = {
        "candidate_count": len(candidates),
        "candidate_ratio": len(candidates) / float(probs.numel()),
        "attention_mass_recall": mass,
        "missed_attention_mass": 1.0 - mass,
        "sink_attention_mass": sink_mass,
        "sink_in_candidate": 1.0 if bool(valid_sink_positions & candidates) else 0.0,
        "nonsink_attention_mass_recall": nonsink_mass / nonsink_denom,
        "nonsink_missed_attention_mass": 1.0 - (nonsink_mass / nonsink_denom),
    }
    for k in top_ks:
        kk = min(int(k), probs.numel())
        top_idx = set(torch.topk(probs, k=kk).indices.tolist())
        overlap = len(top_idx & candidates)
        row[f"top{k}_token_recall"] = overlap / float(kk)
        row[f"top{k}_all_in_candidate"] = 1.0 if overlap == kk else 0.0
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether K graph candidates recall full qK attention.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument("--text-path", default="fdong_seq_compress/data/synthetic_texts/long_textbook_distributed_systems.txt")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--decode-start", type=int, default=1000)
    parser.add_argument("--query-stride", type=int, default=8)
    parser.add_argument("--max-queries", type=int, default=128)
    parser.add_argument("--layers", default="27")
    parser.add_argument("--q-heads", default="0")
    parser.add_argument("--similarity", choices=["cos", "l2"], default="l2")
    parser.add_argument("--graph-top-k", type=int, default=10)
    parser.add_argument("--graph-hops", type=int, default=1)
    parser.add_argument("--graph-direction", choices=["out", "in", "both"], default="both")
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--methods", default="local,random_local_size,random_max_candidates,local_topq_graph,local_graph_all")
    parser.add_argument("--top-attention-ks", default="1,5,10")
    parser.add_argument("--always-include-positions", default="0:10")
    parser.add_argument("--sink-positions", default="0")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--save-examples", type=int, default=20)
    parser.add_argument("--allow-longer-than-model-max", action="store_true")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"fdong_seq_compress/outputs/k_graph_attention_recall_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    _, input_ids = load_tokenized_text(tokenizer, args.text_path, args.max_tokens)
    seq_len = int(input_ids.numel())
    model_max_position_embeddings = getattr(model.config, "max_position_embeddings", None)
    if model_max_position_embeddings is not None and seq_len > int(model_max_position_embeddings):
        message = (
            f"Tokenized sequence length ({seq_len}) exceeds model.config.max_position_embeddings "
            f"({model_max_position_embeddings})."
        )
        if not args.allow_longer_than_model_max:
            raise ValueError(message + " Use shorter --max-tokens or pass --allow-longer-than-model-max intentionally.")
        print(f"WARNING: {message}", flush=True)

    outputs = run_forward(model, input_ids, device)
    hidden_states = outputs.hidden_states
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden_states[0], position_ids)

    num_layers = int(getattr(model.config, "num_hidden_layers"))
    num_q_heads = int(getattr(model.config, "num_attention_heads"))
    num_kv_heads = int(getattr(model.config, "num_key_value_heads"))
    num_groups = num_q_heads // num_kv_heads
    scaling = float(getattr(model.model.layers[0].self_attn, "scaling"))

    layer_indices = select_indices(num_layers, args.layers)
    q_head_indices = select_indices(num_q_heads, args.q_heads)
    query_positions = selected_query_positions(seq_len, args.decode_start, args.query_stride, args.max_queries)
    methods = parse_methods(args.methods)
    top_attention_ks = [int(part) for part in args.top_attention_ks.split(",") if part.strip()]
    always_include_positions = parse_position_spec(args.always_include_positions)
    sink_positions = parse_positions(args.sink_positions)
    rng = random.Random(args.random_seed)

    write_csv(output_dir / "tokens.csv", decode_tokens(tokenizer, input_ids))

    query_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    example_rows: List[Dict] = []

    for layer_idx in layer_indices:
        q, k = compute_layer_qk(model, hidden_states[layer_idx], layer_idx, position_embeddings)
        graph_cache: Dict[int, Tuple[List[List[int]], List[List[int]]]] = {}
        for q_head_idx in q_head_indices:
            kv_head_idx = q_head_idx // num_groups
            if kv_head_idx >= num_kv_heads:
                continue
            if kv_head_idx not in graph_cache:
                graph_cache[kv_head_idx] = build_k_graph(k[kv_head_idx], args.graph_top_k, args.similarity)
            out_neighbors, in_neighbors = graph_cache[kv_head_idx]

            method_rows: Dict[str, List[Dict[str, float]]] = {method: [] for method in methods}
            for t in query_positions:
                scores, probs = qk_attention(q[q_head_idx, t], k[kv_head_idx, :t], scaling)
                full_top = torch.topk(probs, k=min(max(top_attention_ks), probs.numel())).indices.tolist()
                for method in methods:
                    candidates = candidate_set_for_method(
                        method,
                        t,
                        scores,
                        args.local_window,
                        args.seed_count,
                        out_neighbors,
                        in_neighbors,
                        args.graph_hops,
                        args.graph_direction,
                        args.max_candidates,
                        always_include_positions,
                        rng,
                    )
                    metrics = evaluate_candidates(probs, candidates, top_attention_ks, sink_positions)
                    row = {
                        "layer": layer_idx,
                        "q_head": q_head_idx,
                        "kv_head": kv_head_idx,
                        "query_pos": t,
                        "method": method,
                        "seq_len": seq_len,
                        "decode_start": args.decode_start,
                        "local_window": args.local_window,
                        "seed_count": args.seed_count,
                        "graph_top_k": args.graph_top_k,
                        "graph_hops": args.graph_hops,
                        "graph_direction": args.graph_direction,
                        "similarity": args.similarity,
                        **metrics,
                    }
                    query_rows.append(row)
                    method_rows[method].append(metrics)
                    if len(example_rows) < args.save_examples:
                        example_rows.append(
                            {
                                **row,
                                "query_token": tokenizer.decode([int(input_ids[t].item())], clean_up_tokenization_spaces=False).replace("\n", "\\n"),
                                "full_top_positions": " ".join(str(x) for x in full_top),
                                "candidate_positions": " ".join(str(x) for x in sorted(candidates)[:80]),
                            }
                        )

            for method, rows in method_rows.items():
                base = {
                    "layer": layer_idx,
                    "q_head": q_head_idx,
                    "kv_head": kv_head_idx,
                    "method": method,
                    "num_queries": len(rows),
                    "seq_len": seq_len,
                    "decode_start": args.decode_start,
                    "local_window": args.local_window,
                    "seed_count": args.seed_count,
                    "graph_top_k": args.graph_top_k,
                    "graph_hops": args.graph_hops,
                    "graph_direction": args.graph_direction,
                    "similarity": args.similarity,
                        "max_candidates": args.max_candidates,
                        "always_include_positions": args.always_include_positions,
                    }
                for key in [
                    "candidate_count",
                    "candidate_ratio",
                    "attention_mass_recall",
                    "missed_attention_mass",
                    "sink_attention_mass",
                    "sink_in_candidate",
                    "nonsink_attention_mass_recall",
                    "nonsink_missed_attention_mass",
                ]:
                    base.update(summarize([row.get(key, float("nan")) for row in rows], key))
                for top_k in top_attention_ks:
                    key = f"top{top_k}_token_recall"
                    base.update(summarize([row.get(key, float("nan")) for row in rows], key))
                    all_key = f"top{top_k}_all_in_candidate"
                    base.update(summarize([row.get(all_key, float("nan")) for row in rows], all_key))
                summary_rows.append(base)
                print(
                    f"layer={layer_idx} q_head={q_head_idx} method={method} "
                    f"mass={base['attention_mass_recall_mean']:.4f} "
                    f"cand={base['candidate_count_mean']:.1f}",
                    flush=True,
                )

    write_csv(output_dir / "query_recall_by_method.csv", query_rows)
    write_csv(output_dir / "summary_by_layer_head_method.csv", summary_rows)
    write_csv(output_dir / "examples.csv", example_rows)

    summary = {
        "model_path": args.model_path,
        "text_path": args.text_path,
        "output_dir": str(output_dir),
        "device": str(device),
        "dtype": args.dtype,
        "seq_len": seq_len,
        "model_max_position_embeddings": model_max_position_embeddings,
        "seq_len_within_model_max_position_embeddings": (
            None if model_max_position_embeddings is None else seq_len <= int(model_max_position_embeddings)
        ),
        "decode_start": args.decode_start,
        "query_stride": args.query_stride,
        "max_queries": args.max_queries,
        "layers": layer_indices,
        "q_heads": q_head_indices,
        "similarity": args.similarity,
        "graph_top_k": args.graph_top_k,
        "graph_hops": args.graph_hops,
        "graph_direction": args.graph_direction,
        "local_window": args.local_window,
        "seed_count": args.seed_count,
        "max_candidates": args.max_candidates,
        "methods": methods,
        "top_attention_ks": top_attention_ks,
        "always_include_positions": sorted(always_include_positions),
        "sink_positions": sorted(sink_positions),
        "num_query_rows": len(query_rows),
        "num_summary_rows": len(summary_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
