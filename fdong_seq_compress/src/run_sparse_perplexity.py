from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model_loader import load_model_and_tokenizer  # noqa: E402
from text_loader import load_tokenized_text  # noqa: E402


EPS = 1e-12


def parse_methods(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


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


def write_timing_svg(path: Path, timing_rows: List[Dict]) -> None:
    layer_rows = [
        row
        for row in timing_rows
        if row.get("event") == "sparse_layer" and "layer_total_seconds" in row
    ]
    if not layer_rows:
        return
    methods = sorted({str(row["method"]) for row in layer_rows})
    layers = sorted({int(row["layer"]) for row in layer_rows})
    max_time = max(float(row["layer_total_seconds"]) for row in layer_rows)
    max_time = max(max_time, 1e-6)
    width = 980
    height = 420
    left = 62
    right = 24
    top = 24
    bottom = 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]

    def x_for_layer(layer: int) -> float:
        if len(layers) == 1:
            return left + plot_w / 2
        return left + (layer - min(layers)) * plot_w / max(1, (max(layers) - min(layers)))

    def y_for_time(value: float) -> float:
        return top + plot_h - (value / max_time) * plot_h

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1"/>',
        f'<text x="{left}" y="18" font-family="monospace" font-size="13" fill="#111827">Sparse forward layer time</text>',
        f'<text x="{left + plot_w / 2 - 40}" y="{height - 12}" font-family="monospace" font-size="12" fill="#374151">layer</text>',
        f'<text x="6" y="{top + 16}" font-family="monospace" font-size="12" fill="#374151">sec</text>',
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h - frac * plot_h
        value = frac * max_time
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        pieces.append(f'<text x="12" y="{y + 4:.1f}" font-family="monospace" font-size="10" fill="#6b7280">{value:.2f}</text>')
    for layer in layers:
        if layer % 5 == 0 or layer == layers[-1]:
            x = x_for_layer(layer)
            pieces.append(f'<text x="{x - 6:.1f}" y="{top + plot_h + 18}" font-family="monospace" font-size="10" fill="#6b7280">{layer}</text>')

    for mi, method in enumerate(methods):
        rows = [row for row in layer_rows if str(row["method"]) == method]
        rows.sort(key=lambda row: int(row["layer"]))
        points = " ".join(
            f'{x_for_layer(int(row["layer"])):.1f},{y_for_time(float(row["layer_total_seconds"])):.1f}'
            for row in rows
        )
        color = colors[mi % len(colors)]
        pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_y = top + 18 + mi * 18
        pieces.append(f'<line x1="{left + plot_w - 180}" y1="{legend_y - 4}" x2="{left + plot_w - 160}" y2="{legend_y - 4}" stroke="{color}" stroke-width="2"/>')
        pieces.append(f'<text x="{left + plot_w - 154}" y="{legend_y}" font-family="monospace" font-size="11" fill="#111827">{method}</text>')
    pieces.append("</svg>")
    path.write_text("\n".join(pieces), encoding="utf-8")


def causal_full_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()


def local_candidates(t: int, window: int) -> Set[int]:
    return set(range(max(0, t - max(1, window)), t))


def topq_seeds(scores: torch.Tensor, t: int, window: int, seed_count: int) -> Set[int]:
    local = sorted(local_candidates(t, window))
    if not local:
        return set()
    idx = torch.tensor(local, dtype=torch.long, device=scores.device)
    local_scores = scores[idx]
    k = min(max(1, seed_count), len(local))
    top_local = torch.topk(local_scores, k=k).indices.detach().cpu().tolist()
    return {local[i] for i in top_local}


def build_k_graph(kh: torch.Tensor, graph_top_k: int, similarity: str) -> Tuple[List[List[int]], List[List[int]]]:
    n = kh.shape[0]
    if n < 2:
        return [[] for _ in range(n)], [[] for _ in range(n)]

    x = kh.detach().cpu().to(dtype=torch.float32)
    if similarity == "cos":
        x = x - x.mean(dim=0, keepdim=True)
        x = F.normalize(x, p=2, dim=1, eps=EPS)
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
        if not frontier:
            break
    return visited


def trim_candidates(candidates: Set[int], always: Set[int], local: Set[int], max_candidates: int) -> Set[int]:
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return candidates
    always_keep = sorted(always & candidates)
    local_keep = sorted((local & candidates) - set(always_keep))
    remaining = sorted(candidates - set(always_keep) - set(local_keep))
    budget_after_always = max(0, max_candidates - len(always_keep))
    trimmed = always_keep + local_keep[:budget_after_always]
    if len(trimmed) < max_candidates:
        trimmed += remaining[: max(0, max_candidates - len(trimmed))]
    return set(trimmed[:max_candidates])


def kmeans_prefill(x: torch.Tensor, num_clusters: int, steps: int, seed: int) -> Tuple[torch.Tensor, List[int]]:
    n = x.shape[0]
    k = min(max(1, num_clusters), n)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
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
    return centers, assignment.tolist()


def sampled_pairwise_l2_median(x: torch.Tensor, max_pairs: int, seed: int) -> float:
    n = x.shape[0]
    if n < 2:
        return 1.0
    pair_count = n * (n - 1) // 2
    if pair_count <= max_pairs:
        distances = torch.pdist(x, p=2)
    else:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        i = torch.randint(0, n, (max_pairs,), generator=gen)
        j = torch.randint(0, n - 1, (max_pairs,), generator=gen)
        j = j + (j >= i).long()
        distances = torch.linalg.vector_norm(x[i] - x[j], dim=1)
    return float(torch.median(distances).clamp_min(EPS).item())


def build_prefill_cluster_blocks(
    kh: torch.Tensor,
    decode_start: int,
    num_clusters: int,
    kmeans_steps: int,
    seed: int,
) -> Tuple[torch.Tensor, List[Set[int]], torch.Tensor]:
    seq_len = kh.shape[0]
    prefill = max(1, min(decode_start, seq_len))
    x = kh.detach().cpu().to(dtype=torch.float32)
    centers, assignment_list = kmeans_prefill(x[:prefill], num_clusters, kmeans_steps, seed)
    assignment = torch.tensor(assignment_list, dtype=torch.long)
    members: List[Set[int]] = [set() for _ in range(centers.shape[0])]
    for idx, cid in enumerate(assignment.tolist()):
        members[cid].add(idx)
    return centers, members, assignment


def build_threshold_cluster_candidates(
    qh: torch.Tensor,
    kh: torch.Tensor,
    decode_start: int,
    num_clusters: int,
    cluster_threshold: float,
    kmeans_steps: int,
    scale_sample_pairs: int,
    min_selected_clusters: int,
    seed: int,
) -> Tuple[List[Set[int]], Dict[str, float]]:
    seq_len = kh.shape[0]
    prefill = max(1, min(decode_start, seq_len))
    x = kh.detach().cpu().to(dtype=torch.float32)
    centers, members, assignment = build_prefill_cluster_blocks(kh, decode_start, num_clusters, kmeans_steps, seed)
    scale = sampled_pairwise_l2_median(x[:prefill], scale_sample_pairs, seed + 17)

    q_cpu = qh.detach().cpu().to(dtype=torch.float32)
    candidates: List[Set[int]] = [set() for _ in range(seq_len)]
    selected_cluster_counts: List[int] = []
    selected_token_counts: List[int] = []
    min_norm_distances: List[float] = []
    for t in range(prefill, seq_len):
        norm_distances = torch.linalg.vector_norm(centers - q_cpu[t][None, :], dim=1) / max(scale, EPS)
        selected = set(torch.nonzero(norm_distances <= cluster_threshold, as_tuple=False).flatten().tolist())
        if len(selected) < min_selected_clusters:
            kk = min(max(1, min_selected_clusters), centers.shape[0])
            selected.update(torch.topk(-norm_distances, k=kk).indices.tolist())
        token_set: Set[int] = set()
        for cid in selected:
            token_set.update(members[cid])
        candidates[t] = token_set
        selected_cluster_counts.append(len(selected))
        selected_token_counts.append(len(token_set))
        min_norm_distances.append(float(norm_distances.min().item()))

    nonempty = [len(m) for m in members if m]
    stats = {
        "cluster_count": float(centers.shape[0]),
        "cluster_size_min": float(min(nonempty) if nonempty else 0),
        "cluster_size_max": float(max(nonempty) if nonempty else 0),
        "cluster_size_mean": float(sum(nonempty) / max(1, len(nonempty))),
        "cluster_threshold": float(cluster_threshold),
        "cluster_distance_scale_median_pairwise_l2": float(scale),
        "selected_cluster_count_mean": float(sum(selected_cluster_counts) / max(1, len(selected_cluster_counts))),
        "selected_cluster_count_min": float(min(selected_cluster_counts) if selected_cluster_counts else 0),
        "selected_cluster_count_max": float(max(selected_cluster_counts) if selected_cluster_counts else 0),
        "selected_cluster_token_count_mean": float(sum(selected_token_counts) / max(1, len(selected_token_counts))),
        "min_norm_q_center_distance_mean": float(sum(min_norm_distances) / max(1, len(min_norm_distances))),
    }
    return candidates, stats


def build_online_cluster_candidates(
    qh: torch.Tensor,
    kh: torch.Tensor,
    decode_start: int,
    num_clusters: int,
    top_clusters: int,
    kmeans_steps: int,
    seed: int,
) -> Tuple[List[Set[int]], Dict[str, float]]:
    seq_len = kh.shape[0]
    prefill = max(1, min(decode_start, seq_len))
    x = kh.detach().cpu().to(dtype=torch.float32)
    centers, prefill_assignment = kmeans_prefill(x[:prefill], num_clusters, kmeans_steps, seed)
    k = centers.shape[0]

    members: List[Set[int]] = [set() for _ in range(k)]
    sums = torch.zeros_like(centers)
    counts = torch.zeros(k, dtype=torch.long)
    for idx, cid in enumerate(prefill_assignment):
        members[cid].add(idx)
        sums[cid] += x[idx]
        counts[cid] += 1
    for cid in range(k):
        if counts[cid] > 0:
            centers[cid] = sums[cid] / counts[cid].to(dtype=sums.dtype)

    q_cpu = qh.detach().cpu().to(dtype=torch.float32)
    candidates: List[Set[int]] = [set() for _ in range(seq_len)]
    cluster_sizes: List[int] = []
    selected_cluster_counts: List[int] = []
    for t in range(1, seq_len):
        cluster_scores = q_cpu[t] @ centers.T
        kk = min(max(1, top_clusters), k)
        selected = torch.topk(cluster_scores, k=kk).indices.tolist()
        selected_cluster_counts.append(len(selected))
        token_set: Set[int] = set()
        for cid in selected:
            token_set.update(idx for idx in members[cid] if idx < t)
        candidates[t] = token_set

        distances = torch.cdist(x[t : t + 1], centers, p=2)[0]
        cid = int(torch.argmin(distances).item())
        members[cid].add(t)
        sums[cid] += x[t]
        counts[cid] += 1
        centers[cid] = sums[cid] / counts[cid].to(dtype=sums.dtype)
        cluster_sizes.append(int(counts[cid].item()))

    nonempty = [int(c.item()) for c in counts if int(c.item()) > 0]
    stats = {
        "cluster_count": float(k),
        "cluster_size_min": float(min(nonempty) if nonempty else 0),
        "cluster_size_max": float(max(nonempty) if nonempty else 0),
        "cluster_size_mean": float(sum(nonempty) / max(1, len(nonempty))),
        "selected_cluster_count_mean": float(sum(selected_cluster_counts) / max(1, len(selected_cluster_counts))),
    }
    return candidates, stats


def build_method_masks_for_kv_head(
    method: str,
    q_for_kv: torch.Tensor,
    kh: torch.Tensor,
    scaling: float,
    decode_start: int,
    local_window: int,
    seed_count: int,
    max_candidates: int,
    always_positions: Set[int],
    graph_top_k: int,
    graph_hops: int,
    graph_direction: str,
    similarity: str,
    num_clusters: int,
    top_clusters: int,
    cluster_threshold: float,
    kmeans_steps: int,
    cluster_scale_sample_pairs: int,
    min_selected_clusters: int,
    random_seed: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    seq_len = kh.shape[0]
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
    stats: Dict[str, float] = {}
    causal_prefill_end = min(max(1, decode_start), seq_len)
    mask[:causal_prefill_end] = torch.ones((causal_prefill_end, seq_len), dtype=torch.bool).tril()

    out_neighbors: List[List[int]] = []
    in_neighbors: List[List[int]] = []
    if method in {"local_topq_graph", "local_graph_all"}:
        out_neighbors, in_neighbors = build_k_graph(kh, graph_top_k, similarity)

    cluster_candidates: Optional[List[Set[int]]] = None
    if method in {"cluster_topn", "local_cluster_topn"}:
        cluster_candidates, stats = build_online_cluster_candidates(
            q_for_kv,
            kh,
            decode_start,
            num_clusters,
            top_clusters,
            kmeans_steps,
            random_seed,
        )
    elif method in {"cluster_threshold", "local_cluster_threshold"}:
        cluster_candidates, stats = build_threshold_cluster_candidates(
            q_for_kv,
            kh,
            decode_start,
            num_clusters,
            cluster_threshold,
            kmeans_steps,
            cluster_scale_sample_pairs,
            min_selected_clusters,
            random_seed,
        )

    q_cpu = q_for_kv.detach().cpu().to(dtype=torch.float32)
    k_cpu = kh.detach().cpu().to(dtype=torch.float32)
    candidate_counts: List[int] = []
    for t in range(causal_prefill_end, seq_len):
        always = {idx for idx in always_positions if 0 <= idx < t}
        local = local_candidates(t, local_window)
        if method == "full":
            candidates = set(range(t))
        elif method == "local":
            candidates = set(local)
        elif method == "local_topq_graph":
            scores = (q_cpu[t] @ k_cpu[:t].T) * scaling
            seeds = topq_seeds(scores, t, local_window, seed_count)
            candidates = local | expand_graph(seeds, t, out_neighbors, in_neighbors, graph_hops, graph_direction)
        elif method == "local_graph_all":
            candidates = expand_graph(local, t, out_neighbors, in_neighbors, graph_hops, graph_direction)
        elif method == "cluster_topn":
            assert cluster_candidates is not None
            candidates = set(cluster_candidates[t])
        elif method == "local_cluster_topn":
            assert cluster_candidates is not None
            candidates = local | set(cluster_candidates[t])
        elif method == "cluster_threshold":
            assert cluster_candidates is not None
            candidates = set(cluster_candidates[t])
        elif method == "local_cluster_threshold":
            assert cluster_candidates is not None
            candidates = local | set(cluster_candidates[t])
        else:
            raise ValueError(f"Unsupported method: {method}")
        candidates.update(always)
        candidates = {idx for idx in candidates if 0 <= idx < t}
        candidates = trim_candidates(candidates, always, local, max_candidates)
        candidates.add(t)
        if candidates:
            mask[t, torch.tensor(sorted(candidates), dtype=torch.long)] = True
        candidate_counts.append(len(candidates))

    if candidate_counts:
        counts = torch.tensor(candidate_counts, dtype=torch.float32)
        stats.update(
            {
                "candidate_count_mean": float(counts.mean().item()),
                "candidate_count_min": float(counts.min().item()),
                "candidate_count_max": float(counts.max().item()),
                "candidate_ratio_mean": float((counts / max(1, seq_len)).mean().item()),
            }
        )
    return mask, stats


def compute_loss(logits: torch.Tensor, input_ids: torch.Tensor, eval_start: int) -> Tuple[float, float, int]:
    shift_logits = logits[:-1].float()
    shift_labels = input_ids[1:].to(shift_logits.device)
    positions = torch.arange(1, input_ids.numel(), device=shift_logits.device)
    keep = positions >= max(1, eval_start)
    selected_logits = shift_logits[keep]
    selected_labels = shift_labels[keep]
    loss = F.cross_entropy(selected_logits, selected_labels, reduction="mean")
    return float(loss.item()), float(torch.exp(loss).item()), int(selected_labels.numel())


def summarize_float_values(values: Sequence[float], prefix: str) -> Dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_count": 0.0,
        }
    x = torch.tensor(finite, dtype=torch.float64)
    q = torch.quantile(x, torch.tensor([0.50, 0.90], dtype=torch.float64))
    return {
        f"{prefix}_mean": float(x.mean().item()),
        f"{prefix}_p50": float(q[0].item()),
        f"{prefix}_p90": float(q[1].item()),
        f"{prefix}_count": float(x.numel()),
    }


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    masks_by_kv_head: Sequence[torch.Tensor],
    scaling: float,
    num_groups: int,
    decode_start: int,
    sink_positions: Set[int],
    top_attention_ks: Sequence[int],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    num_q_heads, seq_len, head_dim = q.shape
    outputs: List[torch.Tensor] = []
    neg_inf = torch.finfo(q.dtype).min
    causal = torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device).tril()
    eval_start = max(1, min(decode_start, seq_len))
    eval_slice = slice(eval_start, seq_len)
    attention_mass_values: List[float] = []
    nonsink_mass_values: List[float] = []
    top_recall_values: Dict[int, List[float]] = {int(k): [] for k in top_attention_ks}

    for q_head_idx in range(num_q_heads):
        kv_head_idx = q_head_idx // num_groups
        scores = (q[q_head_idx] @ k[kv_head_idx].T) * scaling
        mask = masks_by_kv_head[kv_head_idx].to(device=scores.device)
        full_scores = scores.masked_fill(~causal, neg_inf)
        full_probs = torch.softmax(full_scores.float(), dim=-1)

        if eval_start < seq_len:
            eval_probs = full_probs[eval_slice]
            eval_mask = mask[eval_slice]
            mass = (eval_probs * eval_mask.float()).sum(dim=-1)
            attention_mass_values.extend(mass.detach().cpu().tolist())

            valid_sink_positions = [pos for pos in sorted(sink_positions) if 0 <= pos < seq_len]
            nonsink_mask = eval_mask.clone()
            if valid_sink_positions:
                nonsink_mask[:, torch.tensor(valid_sink_positions, dtype=torch.long, device=nonsink_mask.device)] = False
                sink_mass = eval_probs[:, torch.tensor(valid_sink_positions, dtype=torch.long, device=eval_probs.device)].sum(dim=-1)
            else:
                sink_mass = torch.zeros(eval_probs.shape[0], dtype=eval_probs.dtype, device=eval_probs.device)
            nonsink_mass = (eval_probs * nonsink_mask.float()).sum(dim=-1) / (1.0 - sink_mass).clamp_min(EPS)
            nonsink_mass_values.extend(nonsink_mass.detach().cpu().tolist())

            for top_k in top_attention_ks:
                kk = min(int(top_k), seq_len)
                if kk <= 0:
                    continue
                top_idx = torch.topk(eval_probs, k=kk, dim=-1).indices
                top_in_candidate = eval_mask.gather(1, top_idx).float().mean(dim=-1)
                top_recall_values[int(top_k)].extend(top_in_candidate.detach().cpu().tolist())

        sparse_scores = scores.masked_fill(~mask.to(device=scores.device), neg_inf)
        probs = torch.softmax(sparse_scores.float(), dim=-1).to(dtype=v.dtype)
        outputs.append(probs @ v[kv_head_idx])

    metrics = {}
    metrics.update(summarize_float_values(attention_mass_values, "attention_mass_recall"))
    metrics.update(summarize_float_values(nonsink_mass_values, "nonsink_attention_mass_recall"))
    for top_k, values in top_recall_values.items():
        metrics.update(summarize_float_values(values, f"top{top_k}_token_recall"))
    return torch.stack(outputs, dim=0).transpose(0, 1).reshape(seq_len, num_q_heads * head_dim), metrics


def run_sparse_forward(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    method: str,
    decode_start: int,
    local_window: int,
    seed_count: int,
    max_candidates: int,
    always_positions: Set[int],
    graph_top_k: int,
    graph_hops: int,
    graph_direction: str,
    similarity: str,
    num_clusters: int,
    top_clusters: int,
    cluster_threshold: float,
    kmeans_steps: int,
    cluster_scale_sample_pairs: int,
    min_selected_clusters: int,
    random_seed: int,
    sink_positions: Set[int],
    top_attention_ks: Sequence[int],
) -> Tuple[torch.Tensor, List[Dict[str, float]], Dict[str, float], List[Dict[str, float]]]:
    seq_len = int(input_ids.numel())
    input_ids = input_ids.to(device)
    hidden_states = model.model.embed_tokens(input_ids)[None, :, :]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

    num_q_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    num_groups = num_q_heads // num_kv_heads
    layer_stats: List[Dict[str, float]] = []
    layer_attention_metrics: List[Dict[str, float]] = []
    timing_rows: List[Dict[str, float]] = []

    with torch.no_grad():
        for layer_idx, layer in enumerate(model.model.layers):
            layer_start = time.perf_counter()
            attn = layer.self_attn
            residual = hidden_states
            qkv_start = time.perf_counter()
            x = layer.input_layernorm(hidden_states)
            input_shape = x.shape[:-1]
            hidden_shape = (*input_shape, -1, attn.head_dim)
            q = attn.q_norm(attn.q_proj(x).view(hidden_shape)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(x).view(hidden_shape)).transpose(1, 2)
            v = attn.v_proj(x).view(hidden_shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, position_embeddings[0], position_embeddings[1])
            q0, k0, v0 = q[0], k[0], v[0]
            qkv_seconds = time.perf_counter() - qkv_start

            candidate_start = time.perf_counter()
            masks_by_kv_head: List[torch.Tensor] = []
            stats_by_kv: List[Dict[str, float]] = []
            for kv_head_idx in range(num_kv_heads):
                q_start = kv_head_idx * num_groups
                q_end = min(num_q_heads, q_start + num_groups)
                q_for_kv = q0[q_start:q_end].mean(dim=0)
                mask, stats = build_method_masks_for_kv_head(
                    method,
                    q_for_kv,
                    k0[kv_head_idx],
                    float(attn.scaling),
                    decode_start,
                    local_window,
                    seed_count,
                    max_candidates,
                    always_positions,
                    graph_top_k,
                    graph_hops,
                    graph_direction,
                    similarity,
                    num_clusters,
                    top_clusters,
                    cluster_threshold,
                    kmeans_steps,
                    cluster_scale_sample_pairs,
                    min_selected_clusters,
                    random_seed + layer_idx * 1000 + kv_head_idx,
                )
                masks_by_kv_head.append(mask)
                stats_by_kv.append(stats)
            candidate_seconds = time.perf_counter() - candidate_start

            attention_start = time.perf_counter()
            attn_output, attn_metrics = sparse_attention(
                q0,
                k0,
                v0,
                masks_by_kv_head,
                float(attn.scaling),
                num_groups,
                decode_start,
                sink_positions,
                top_attention_ks,
            )
            attn_output = attn.o_proj(attn_output)[None, :, :]
            hidden_states = residual + attn_output
            attention_seconds = time.perf_counter() - attention_start

            mlp_start = time.perf_counter()
            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))
            mlp_seconds = time.perf_counter() - mlp_start
            layer_total_seconds = time.perf_counter() - layer_start

            merged: Dict[str, float] = {"layer": float(layer_idx)}
            for key in sorted({k for row in stats_by_kv for k in row.keys()}):
                values = [row[key] for row in stats_by_kv if key in row and math.isfinite(float(row[key]))]
                if values:
                    merged[f"{key}_mean_over_kv_heads"] = float(sum(values) / len(values))
            merged.update(attn_metrics)
            layer_stats.append(merged)
            layer_attention_metrics.append(attn_metrics)
            timing_rows.append(
                {
                    "event": "sparse_layer",
                    "method": method,
                    "layer": float(layer_idx),
                    "seq_len": float(seq_len),
                    "qkv_seconds": qkv_seconds,
                    "candidate_seconds": candidate_seconds,
                    "attention_seconds": attention_seconds,
                    "mlp_seconds": mlp_seconds,
                    "layer_total_seconds": layer_total_seconds,
                    "candidate_count_mean": merged.get("candidate_count_mean_mean_over_kv_heads", float("nan")),
                    "attention_mass_recall_mean": merged.get("attention_mass_recall_mean", float("nan")),
                }
            )
            print(
                f"method={method} layer={layer_idx} "
                f"cand={merged.get('candidate_count_mean_mean_over_kv_heads', float('nan')):.1f} "
                f"mass={merged.get('attention_mass_recall_mean', float('nan')):.4f} "
                f"time={layer_total_seconds:.3f}s",
                flush=True,
            )

        hidden_states = model.model.norm(hidden_states)
        logits = model.lm_head(hidden_states)[0]

    aggregate_attention_metrics: Dict[str, float] = {}
    metric_keys = sorted({key for row in layer_attention_metrics for key in row.keys()})
    for key in metric_keys:
        values = [row[key] for row in layer_attention_metrics if key in row and math.isfinite(float(row[key]))]
        if values:
            aggregate_attention_metrics[key] = float(sum(values) / len(values))
    return logits, layer_stats, aggregate_attention_metrics, timing_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sparse KV candidate methods by next-token loss/perplexity.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument("--text-path", default="fdong_seq_compress/data/synthetic_texts/long_textbook_distributed_systems.txt")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--decode-start", type=int, default=1000)
    parser.add_argument("--methods", default="local,local_topq_graph,cluster_topn,local_cluster_topn")
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--always-include-positions", default="0:10")
    parser.add_argument("--sink-positions", default="0")
    parser.add_argument("--top-attention-ks", default="1,5,10")
    parser.add_argument("--graph-top-k", type=int, default=10)
    parser.add_argument("--graph-hops", type=int, default=1)
    parser.add_argument("--graph-direction", choices=["out", "in", "both"], default="both")
    parser.add_argument("--similarity", choices=["cos", "l2"], default="l2")
    parser.add_argument("--num-clusters", type=int, default=10)
    parser.add_argument("--top-clusters", type=int, default=2)
    parser.add_argument("--cluster-threshold", type=float, default=0.75)
    parser.add_argument("--cluster-scale-sample-pairs", type=int, default=20000)
    parser.add_argument("--min-selected-clusters", type=int, default=1)
    parser.add_argument("--kmeans-steps", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--allow-longer-than-model-max", action="store_true")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"fdong_seq_compress/outputs/sparse_perplexity_{timestamp}")
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

    methods = parse_methods(args.methods)
    always_positions = parse_position_spec(args.always_include_positions)
    sink_positions = parse_position_spec(args.sink_positions)
    top_attention_ks = [int(part.strip()) for part in args.top_attention_ks.split(",") if part.strip()]
    result_rows: List[Dict] = []
    timing_rows: List[Dict] = []

    print("Running full-attention baseline...", flush=True)
    baseline_start = time.perf_counter()
    with torch.no_grad():
        baseline_logits = model(input_ids[None, :].to(device), use_cache=False, return_dict=True).logits[0]
    baseline_seconds = time.perf_counter() - baseline_start
    full_loss, full_ppl, num_eval_tokens = compute_loss(baseline_logits, input_ids, args.decode_start)
    timing_rows.append(
        {
            "event": "full_forward",
            "method": "full",
            "seq_len": seq_len,
            "total_seconds": baseline_seconds,
            "seconds_per_token": baseline_seconds / max(1, seq_len),
        }
    )
    result_rows.append(
        {
            "method": "full",
            "loss": full_loss,
            "ppl": full_ppl,
            "delta_loss_vs_full": 0.0,
            "ppl_ratio_vs_full": 1.0,
            "num_eval_tokens": num_eval_tokens,
        }
    )
    print(
        f"full loss={full_loss:.6f} ppl={full_ppl:.4f} "
        f"eval_tokens={num_eval_tokens} time={baseline_seconds:.3f}s",
        flush=True,
    )

    for method in methods:
        print(f"Running sparse method: {method}", flush=True)
        method_start = time.perf_counter()
        logits, layer_stats, attention_metrics, method_timing_rows = run_sparse_forward(
            model,
            input_ids,
            device,
            method,
            args.decode_start,
            args.local_window,
            args.seed_count,
            args.max_candidates,
            always_positions,
            args.graph_top_k,
            args.graph_hops,
            args.graph_direction,
            args.similarity,
            args.num_clusters,
            args.top_clusters,
            args.cluster_threshold,
            args.kmeans_steps,
            args.cluster_scale_sample_pairs,
            args.min_selected_clusters,
            args.random_seed,
            sink_positions,
            top_attention_ks,
        )
        method_seconds = time.perf_counter() - method_start
        loss, ppl, _ = compute_loss(logits, input_ids, args.decode_start)
        row = {
            "method": method,
            "loss": loss,
            "ppl": ppl,
            "delta_loss_vs_full": loss - full_loss,
            "ppl_ratio_vs_full": ppl / full_ppl,
            "num_eval_tokens": num_eval_tokens,
            "local_window": args.local_window,
            "seed_count": args.seed_count,
            "max_candidates": args.max_candidates,
            "graph_top_k": args.graph_top_k,
            "graph_hops": args.graph_hops,
            "graph_direction": args.graph_direction,
            "similarity": args.similarity,
            "num_clusters": args.num_clusters,
            "top_clusters": args.top_clusters,
            "cluster_threshold": args.cluster_threshold,
            "cluster_scale_sample_pairs": args.cluster_scale_sample_pairs,
            "min_selected_clusters": args.min_selected_clusters,
            "kmeans_steps": args.kmeans_steps,
            **attention_metrics,
        }
        result_rows.append(row)
        timing_rows.extend(method_timing_rows)
        timing_rows.append(
            {
                "event": "sparse_method_total",
                "method": method,
                "seq_len": seq_len,
                "total_seconds": method_seconds,
                "seconds_per_token": method_seconds / max(1, seq_len),
            }
        )
        write_csv(output_dir / f"layer_stats_{method}.csv", layer_stats)
        print(
            f"{method} loss={loss:.6f} ppl={ppl:.4f} "
            f"delta_loss={loss - full_loss:.6f} ppl_ratio={ppl / full_ppl:.4f} "
            f"time={method_seconds:.3f}s",
            flush=True,
        )

    write_csv(output_dir / "perplexity_by_method.csv", result_rows)
    write_csv(output_dir / "timing_by_method_layer.csv", timing_rows)
    write_timing_svg(output_dir / "timing_by_layer.svg", timing_rows)
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
        "methods": methods,
        "sink_positions": sorted(sink_positions),
        "top_attention_ks": top_attention_ks,
        "num_eval_tokens": num_eval_tokens,
        "full_loss": full_loss,
        "full_ppl": full_ppl,
        "full_forward_seconds": baseline_seconds,
        "args": vars(args),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
