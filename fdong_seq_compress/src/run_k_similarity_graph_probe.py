from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from geometry_metrics import extract_cache_tensor, select_indices  # noqa: E402
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


def parse_bins(value: str) -> List[float]:
    if ":" in value:
        left, right, step = [float(part) for part in value.split(":")]
        if step <= 0:
            raise ValueError("--hist-bins step must be positive.")
        bins = []
        current = left
        while current <= right + step * 0.5:
            bins.append(round(current, 10))
            current += step
        return bins
    bins = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(bins) < 2:
        raise ValueError("Need at least two histogram bin edges.")
    return bins


def auto_bins(values: torch.Tensor, similarity: str, num_bins: int = 40) -> List[float]:
    if values.numel() == 0:
        return [0.0, 1.0]
    if similarity == "cos":
        return parse_bins("-1.0:1.0:0.05")

    values = values.detach().to(dtype=torch.float64)
    left = float(torch.quantile(values, 0.01).item())
    right = float(torch.quantile(values, 0.99).item())
    if not math.isfinite(left) or not math.isfinite(right) or abs(right - left) <= EPS:
        center = float(values.mean().item()) if values.numel() else 0.0
        width = max(abs(center) * 0.1, 1.0)
        left, right = center - width, center + width
    if similarity == "l2":
        left = max(0.0, left)
    step = (right - left) / num_bins
    return [left + step * idx for idx in range(num_bins + 1)]


def bins_for_values(values: torch.Tensor, similarity: str, hist_bins: str) -> List[float]:
    if hist_bins == "auto":
        return auto_bins(values, similarity)
    return parse_bins(hist_bins)


def run_forward(model, input_ids: torch.Tensor, device: torch.device):
    with torch.no_grad():
        return model(
            input_ids=input_ids[None, :].to(device),
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )


def token_level_matrix(cache: torch.Tensor, seq_len: int) -> torch.Tensor:
    # cache: [kv_heads, seq_len, head_dim] -> [seq_len, kv_heads * head_dim]
    x = cache[:, :seq_len, :].detach().to(device="cpu", dtype=torch.float32)
    return x.transpose(0, 1).contiguous().reshape(seq_len, -1)


def head_level_matrices(cache: torch.Tensor, seq_len: int, heads: Iterable[int]) -> List[Tuple[int, torch.Tensor]]:
    cache = cache.detach().to(device="cpu", dtype=torch.float32)
    return [(head_idx, cache[head_idx, :seq_len, :].contiguous()) for head_idx in heads]


def transform_keys(x: torch.Tensor, key_transform: str, pc_remove_count: int) -> torch.Tensor:
    if key_transform == "raw":
        return x

    centered = x - x.mean(dim=0, keepdim=True)
    if key_transform == "center":
        return centered

    if key_transform == "remove_pc":
        if pc_remove_count <= 0:
            return centered
        _, _, vh = torch.linalg.svd(centered.to(dtype=torch.float32), full_matrices=False)
        rank = min(pc_remove_count, vh.shape[0])
        basis = vh[:rank]
        return centered - (centered @ basis.T) @ basis

    if key_transform == "whiten":
        _, singular, vh = torch.linalg.svd(centered.to(dtype=torch.float32), full_matrices=False)
        keep = singular > EPS
        if not bool(keep.any()):
            return centered
        basis = vh[keep]
        scale = singular[keep].clamp_min(EPS)
        # Return whitened coordinates in PCA space. The sqrt factor keeps the
        # average squared coordinate comparable across sequence lengths.
        return (centered @ basis.T) * (math.sqrt(max(1, x.shape[0] - 1)) / scale)

    raise ValueError(f"Unsupported key transform: {key_transform}")


def causal_topk_similarity(
    x: torch.Tensor,
    top_k: int,
    similarity: str,
) -> Tuple[List[Dict], torch.Tensor]:
    n = x.shape[0]
    if n < 2:
        return [], torch.empty(0, dtype=torch.float32)

    if similarity == "cos":
        x_score = torch.nn.functional.normalize(x, p=2, dim=1, eps=EPS)
        scores = x_score @ x_score.T
        larger_is_better = True
    elif similarity == "dot":
        x_score = x
        scores = x_score @ x_score.T
        larger_is_better = True
    elif similarity == "l2":
        x_score = x.to(dtype=torch.float32)
        sq_norm = x_score.square().sum(dim=1, keepdim=True)
        scores = (sq_norm + sq_norm.T - 2.0 * (x_score @ x_score.T)).clamp_min(0.0).sqrt()
        larger_is_better = False
    else:
        raise ValueError(f"Unsupported similarity: {similarity}")

    indices = torch.arange(n)
    invalid = indices[None, :] >= indices[:, None]
    scores = scores.masked_fill(invalid, -float("inf") if larger_is_better else float("inf"))

    actual_k = min(top_k, n - 1)
    if larger_is_better:
        values, neighbor_indices = torch.topk(scores, k=actual_k, dim=1)
    else:
        neg_values, neighbor_indices = torch.topk(-scores, k=actual_k, dim=1)
        values = -neg_values
    rows: List[Dict] = []
    kept_values = []
    for token_idx in range(n):
        for rank_idx in range(actual_k):
            value = float(values[token_idx, rank_idx].item())
            if not math.isfinite(value):
                continue
            rows.append(
                {
                    "token_index": token_idx,
                    "rank": rank_idx + 1,
                    "neighbor_index": int(neighbor_indices[token_idx, rank_idx].item()),
                    "similarity": value,
                    "token_distance": token_idx - int(neighbor_indices[token_idx, rank_idx].item()),
                }
            )
            kept_values.append(value)

    return rows, torch.tensor(kept_values, dtype=torch.float32)


def causal_radius_similarity(
    x: torch.Tensor,
    similarity: str,
    radius_threshold: float,
    max_radius_neighbors: int,
) -> Tuple[List[Dict], torch.Tensor]:
    n = x.shape[0]
    if n < 2:
        return [], torch.empty(0, dtype=torch.float32)

    if similarity == "cos":
        x_score = torch.nn.functional.normalize(x, p=2, dim=1, eps=EPS)
        scores = x_score @ x_score.T
        larger_is_better = True
    elif similarity == "dot":
        x_score = x
        scores = x_score @ x_score.T
        larger_is_better = True
    elif similarity == "l2":
        x_score = x.to(dtype=torch.float32)
        sq_norm = x_score.square().sum(dim=1, keepdim=True)
        scores = (sq_norm + sq_norm.T - 2.0 * (x_score @ x_score.T)).clamp_min(0.0).sqrt()
        larger_is_better = False
    else:
        raise ValueError(f"Unsupported similarity: {similarity}")

    indices = torch.arange(n)
    invalid = indices[None, :] >= indices[:, None]
    if larger_is_better:
        keep = (scores >= radius_threshold) & ~invalid
    else:
        keep = (scores <= radius_threshold) & ~invalid

    rows: List[Dict] = []
    kept_values = []
    for token_idx in range(n):
        neighbor_indices = torch.nonzero(keep[token_idx], as_tuple=False).flatten()
        if neighbor_indices.numel() == 0:
            continue
        values = scores[token_idx, neighbor_indices]
        order = torch.argsort(values, descending=larger_is_better)
        if max_radius_neighbors > 0:
            order = order[:max_radius_neighbors]
        for rank_idx, order_idx in enumerate(order.tolist()):
            neighbor_idx = int(neighbor_indices[order_idx].item())
            value = float(values[order_idx].item())
            if not math.isfinite(value):
                continue
            rows.append(
                {
                    "token_index": token_idx,
                    "rank": rank_idx + 1,
                    "neighbor_index": neighbor_idx,
                    "similarity": value,
                    "token_distance": token_idx - neighbor_idx,
                }
            )
            kept_values.append(value)

    return rows, torch.tensor(kept_values, dtype=torch.float32)


def summarize_values(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    if values.numel() == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    q = torch.quantile(values, torch.tensor([0.05, 0.50, 0.95], dtype=values.dtype))
    return {
        f"{prefix}_count": int(values.numel()),
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        f"{prefix}_p05": float(q[0].item()),
        f"{prefix}_p50": float(q[1].item()),
        f"{prefix}_p95": float(q[2].item()),
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
    }


def histogram_rows(values: torch.Tensor, bins: List[float], base: Dict) -> List[Dict]:
    rows = []
    if values.numel() == 0:
        return rows
    for left, right in zip(bins[:-1], bins[1:]):
        if right == bins[-1]:
            count = int(((values >= left) & (values <= right)).sum().item())
        else:
            count = int(((values >= left) & (values < right)).sum().item())
        rows.append(
            {
                **base,
                "bin_left": left,
                "bin_right": right,
                "count": count,
            }
        )
    return rows


def distance_bins(max_distance: int) -> List[Tuple[int, int, str]]:
    bins = [
        (1, 1, "1"),
        (2, 2, "2"),
        (3, 4, "3-4"),
        (5, 8, "5-8"),
        (9, 16, "9-16"),
        (17, 32, "17-32"),
        (33, 64, "33-64"),
        (65, 128, "65-128"),
        (129, 256, "129-256"),
        (257, 512, "257-512"),
    ]
    if max_distance > 512:
        bins.append((513, max_distance, "513+"))
    return bins


def binned_count_rows(values: List[int], bins: List[Tuple[int, int, str]], base: Dict, value_name: str) -> List[Dict]:
    rows = []
    for left, right, label in bins:
        count = sum(1 for value in values if left <= value <= right)
        rows.append(
            {
                **base,
                f"{value_name}_bin": label,
                "bin_left": left,
                "bin_right": right,
                "count": count,
            }
        )
    return rows


def summarize_distances(
    distances: List[int],
    prefix: str,
    thresholds: Iterable[int],
    artifact_periods: Iterable[int] = (1024, 2048, 4096),
    artifact_near_width: int = 2,
) -> Dict[str, float]:
    if not distances:
        result = {
            f"{prefix}_count": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        }
        for threshold in thresholds:
            result[f"{prefix}_frac_le_{threshold}"] = float("nan")
            result[f"{prefix}_frac_ge_{threshold}"] = float("nan")
        for period in artifact_periods:
            result[f"{prefix}_frac_eq_{period}"] = float("nan")
            result[f"{prefix}_frac_near_{period}_pm{artifact_near_width}"] = float("nan")
            result[f"{prefix}_frac_near_multiple_{period}_pm{artifact_near_width}"] = float("nan")
        return result

    values = torch.tensor(distances, dtype=torch.float32)
    int_values = torch.tensor(distances, dtype=torch.int64)
    result = {
        f"{prefix}_count": int(values.numel()),
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_p50": float(torch.quantile(values, 0.50).item()),
        f"{prefix}_p90": float(torch.quantile(values, 0.90).item()),
        f"{prefix}_p95": float(torch.quantile(values, 0.95).item()),
        f"{prefix}_max": float(values.max().item()),
    }
    for threshold in thresholds:
        result[f"{prefix}_frac_le_{threshold}"] = float((values <= threshold).to(torch.float32).mean().item())
        result[f"{prefix}_frac_ge_{threshold}"] = float((values >= threshold).to(torch.float32).mean().item())
    for period in artifact_periods:
        if period <= 0:
            continue
        exact = int_values == period
        near = (int_values - period).abs() <= artifact_near_width
        remainder = int_values.remainder(period)
        near_multiple = (int_values >= period) & (
            (remainder <= artifact_near_width) | (remainder >= period - artifact_near_width)
        )
        result[f"{prefix}_frac_eq_{period}"] = float(exact.to(torch.float32).mean().item())
        result[f"{prefix}_frac_near_{period}_pm{artifact_near_width}"] = float(near.to(torch.float32).mean().item())
        result[f"{prefix}_frac_near_multiple_{period}_pm{artifact_near_width}"] = float(
            near_multiple.to(torch.float32).mean().item()
        )
    return result


def indegree_values(neighbor_rows: List[Dict], seq_len: int) -> List[int]:
    counts = [0 for _ in range(seq_len)]
    for row in neighbor_rows:
        counts[int(row["neighbor_index"])] += 1
    return counts


def summarize_indegrees(values: List[int], prefix: str) -> Dict[str, float]:
    if not values:
        return {
            f"{prefix}_nodes": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_p99": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_zero_frac": float("nan"),
            f"{prefix}_top1pct_edge_frac": float("nan"),
            f"{prefix}_top5pct_edge_frac": float("nan"),
        }
    x = torch.tensor(values, dtype=torch.float32)
    total_edges = x.sum().clamp_min(1.0)
    sorted_x = torch.sort(x, descending=True).values
    top1_count = max(1, math.ceil(0.01 * len(values)))
    top5_count = max(1, math.ceil(0.05 * len(values)))
    return {
        f"{prefix}_nodes": len(values),
        f"{prefix}_mean": float(x.mean().item()),
        f"{prefix}_p50": float(torch.quantile(x, 0.50).item()),
        f"{prefix}_p90": float(torch.quantile(x, 0.90).item()),
        f"{prefix}_p95": float(torch.quantile(x, 0.95).item()),
        f"{prefix}_p99": float(torch.quantile(x, 0.99).item()),
        f"{prefix}_max": float(x.max().item()),
        f"{prefix}_zero_frac": float((x == 0).to(torch.float32).mean().item()),
        f"{prefix}_top1pct_edge_frac": float((sorted_x[:top1_count].sum() / total_edges).item()),
        f"{prefix}_top5pct_edge_frac": float((sorted_x[:top5_count].sum() / total_edges).item()),
    }


def summarize_graph_structure(neighbor_rows: List[Dict], seq_len: int, prefix: str) -> Dict[str, float]:
    if seq_len <= 0:
        return {
            f"{prefix}_nodes": 0,
            f"{prefix}_edges": 0,
            f"{prefix}_avg_outdegree": float("nan"),
            f"{prefix}_largest_component_frac": float("nan"),
            f"{prefix}_component_count": 0,
            f"{prefix}_isolated_frac": float("nan"),
            f"{prefix}_local_le_8_edge_frac": float("nan"),
            f"{prefix}_long_ge_128_edge_frac": float("nan"),
            f"{prefix}_long_ge_256_edge_frac": float("nan"),
        }

    parent = list(range(seq_len))
    size = [1 for _ in range(seq_len)]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    distances = []
    outdegree = [0 for _ in range(seq_len)]
    for row in neighbor_rows:
        src = int(row["token_index"])
        dst = int(row["neighbor_index"])
        if 0 <= src < seq_len and 0 <= dst < seq_len:
            union(src, dst)
            outdegree[src] += 1
            distances.append(int(row["token_distance"]))

    component_sizes: Dict[int, int] = {}
    for idx in range(seq_len):
        root = find(idx)
        component_sizes[root] = component_sizes.get(root, 0) + 1

    edge_count = len(neighbor_rows)
    if edge_count:
        distance_tensor = torch.tensor(distances, dtype=torch.float32)
        local_le_8 = float((distance_tensor <= 8).to(torch.float32).mean().item())
        long_ge_128 = float((distance_tensor >= 128).to(torch.float32).mean().item())
        long_ge_256 = float((distance_tensor >= 256).to(torch.float32).mean().item())
    else:
        local_le_8 = long_ge_128 = long_ge_256 = float("nan")

    largest_component = max(component_sizes.values()) if component_sizes else 0
    return {
        f"{prefix}_nodes": seq_len,
        f"{prefix}_edges": edge_count,
        f"{prefix}_avg_outdegree": float(sum(outdegree) / max(1, seq_len)),
        f"{prefix}_largest_component_frac": float(largest_component / max(1, seq_len)),
        f"{prefix}_component_count": len(component_sizes),
        f"{prefix}_isolated_frac": float(sum(1 for value in outdegree if value == 0) / max(1, seq_len)),
        f"{prefix}_local_le_8_edge_frac": local_le_8,
        f"{prefix}_long_ge_128_edge_frac": long_ge_128,
        f"{prefix}_long_ge_256_edge_frac": long_ge_256,
    }


def indegree_bins(max_indegree: int) -> List[Tuple[int, int, str]]:
    bins = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 8, "5-8")]
    upper = 16
    left = 9
    while left <= max_indegree:
        right = min(upper, max_indegree)
        bins.append((left, right, f"{left}-{right}" if right < max_indegree else f"{left}+"))
        left = upper + 1
        upper *= 2
    return bins


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_histogram_svg(
    path: Path,
    rows: List[Dict],
    title: str,
    width: int = 1000,
    height: int = 560,
    x_label: str = "metric bucket",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n", encoding="utf-8")
        return

    bins = [(float(row["bin_left"]), float(row["bin_right"])) for row in rows]
    counts = [int(row["count"]) for row in rows]
    max_count = max(max(counts), 1)
    margin_left, margin_right, margin_top, margin_bottom = 70, 24, 60, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    bar_gap = 2
    bar_w = max(1, plot_w / len(rows) - bar_gap)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" fill="#202124">{svg_escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1"/>',
    ]

    for idx, ((left, right), count) in enumerate(zip(bins, counts)):
        x = margin_left + idx * (plot_w / len(rows)) + bar_gap / 2
        bar_h = plot_h * count / max_count
        y = height - margin_bottom - bar_h
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#4c78a8"/>'
        )
        if idx % max(1, len(rows) // 10) == 0:
            parts.append(
                f'<text x="{x:.2f}" y="{height - margin_bottom + 20}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="11" fill="#444">{left:g}</text>'
            )

    for tick_idx in range(6):
        count = max_count * tick_idx / 5
        y = height - margin_bottom - plot_h * tick_idx / 5
        parts.append(f'<line x1="{margin_left - 5}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" stroke="#333"/>')
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="11" fill="#444">{int(count)}</text>'
        )

    parts.extend(
        [
            f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#333">{svg_escape(x_label)}</text>',
            f'<text x="18" y="{height / 2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#333" transform="rotate(-90 18 {height / 2})">count</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_bar_svg(
    path: Path,
    rows: List[Dict],
    label_key: str,
    title: str,
    x_label: str,
    y_label: str = "count",
    width: int = 1000,
    height: int = 560,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n", encoding="utf-8")
        return

    labels = [str(row[label_key]) for row in rows]
    counts = [int(row["count"]) for row in rows]
    max_count = max(max(counts), 1)
    margin_left, margin_right, margin_top, margin_bottom = 72, 24, 60, 86
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    slot_w = plot_w / len(rows)
    bar_w = max(1, slot_w * 0.75)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" fill="#202124">{svg_escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1"/>',
    ]
    for idx, (label, count) in enumerate(zip(labels, counts)):
        x = margin_left + idx * slot_w + (slot_w - bar_w) / 2
        bar_h = plot_h * count / max_count
        y = height - margin_bottom - bar_h
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#59a14f"/>')
        parts.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{height - margin_bottom + 18}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11" fill="#444" transform="rotate(35 {x + bar_w / 2:.2f} {height - margin_bottom + 18})">{svg_escape(label)}</text>'
        )

    for tick_idx in range(6):
        count = max_count * tick_idx / 5
        y = height - margin_bottom - plot_h * tick_idx / 5
        parts.append(f'<line x1="{margin_left - 5}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" stroke="#333"/>')
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="11" fill="#444">{int(count)}</text>'
        )

    parts.extend(
        [
            f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#333">{svg_escape(x_label)}</text>',
            f'<text x="18" y="{height / 2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#333" transform="rotate(-90 18 {height / 2})">{svg_escape(y_label)}</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def safe_name(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe causal top-k K-cache similarity distributions.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument("--text-path", default="fdong_seq_compress/data/synthetic_texts/long_english_article_01.txt")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--analysis-level", choices=["token", "head", "both"], default="token")
    parser.add_argument("--graph-mode", choices=["topk", "radius"], default="topk")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--radius-threshold", type=float, default=float("nan"))
    parser.add_argument("--max-radius-neighbors", type=int, default=200)
    parser.add_argument("--similarity", choices=["cos", "dot", "l2"], default="cos")
    parser.add_argument("--center-tokens", action="store_true")
    parser.add_argument("--key-transform", choices=["raw", "center", "remove_pc", "whiten"], default="")
    parser.add_argument("--pc-remove-count", type=int, default=0)
    parser.add_argument("--hist-bins", default="auto")
    parser.add_argument("--save-neighbors", action="store_true")
    parser.add_argument("--allow-longer-than-model-max", action="store_true")
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1.")
    if args.graph_mode == "radius" and not math.isfinite(args.radius_threshold):
        raise ValueError("--radius-threshold must be set when --graph-mode=radius.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"fdong_seq_compress/outputs/k_similarity_probe_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    _, input_ids = load_tokenized_text(tokenizer, args.text_path, args.max_tokens)
    seq_len = int(input_ids.numel())
    if seq_len < 2:
        raise ValueError("Need at least two tokens for causal top-k analysis.")
    model_max_position_embeddings = getattr(model.config, "max_position_embeddings", None)
    if model_max_position_embeddings is not None and seq_len > int(model_max_position_embeddings):
        message = (
            f"Tokenized sequence length ({seq_len}) exceeds model.config.max_position_embeddings "
            f"({model_max_position_embeddings}). This can make K-cache geometry unreliable due to "
            "position/RoPE handling. Use a shorter sequence or pass --allow-longer-than-model-max "
            "only if you intentionally want to test extrapolation."
        )
        if not args.allow_longer_than_model_max:
            raise ValueError(message)
        print(f"WARNING: {message}", flush=True)

    write_csv(output_dir / "tokens.csv", decode_tokens(tokenizer, input_ids))
    outputs = run_forward(model, input_ids, device)

    num_layers = int(getattr(model.config, "num_hidden_layers"))
    layer_indices = select_indices(num_layers, args.layers)

    all_neighbor_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    hist_rows: List[Dict] = []
    distance_summary_rows: List[Dict] = []
    distance_hist_rows: List[Dict] = []
    indegree_summary_rows: List[Dict] = []
    indegree_hist_rows: List[Dict] = []
    graph_summary_rows: List[Dict] = []
    global_values: List[torch.Tensor] = []
    global_distances: List[int] = []
    global_indegrees: List[int] = []
    global_neighbor_rows: List[Dict] = []

    key_transform = args.key_transform or ("center" if args.center_tokens else "raw")

    for layer_idx in layer_indices:
        cache = extract_cache_tensor(outputs.past_key_values, layer_idx, "K")
        head_indices = select_indices(cache.shape[0], args.heads)
        targets: List[Tuple[str, int, torch.Tensor]] = []

        if args.analysis_level in {"token", "both"}:
            targets.append(("token", -1, token_level_matrix(cache, seq_len)))
        if args.analysis_level in {"head", "both"}:
            targets.extend(("head", head_idx, x) for head_idx, x in head_level_matrices(cache, seq_len, head_indices))

        for level, head_idx, x in targets:
            x = transform_keys(x, key_transform, args.pc_remove_count)
            if args.graph_mode == "topk":
                neighbor_rows, values = causal_topk_similarity(x, args.top_k, args.similarity)
            else:
                neighbor_rows, values = causal_radius_similarity(
                    x,
                    args.similarity,
                    args.radius_threshold,
                    args.max_radius_neighbors,
                )
            base = {
                "layer": layer_idx,
                "analysis_level": level,
                "head": "all" if head_idx < 0 else head_idx,
                "graph_mode": args.graph_mode,
                "similarity": args.similarity,
                "center_tokens": bool(args.center_tokens),
                "key_transform": key_transform,
                "pc_remove_count": args.pc_remove_count,
                "top_k": args.top_k,
                "radius_threshold": args.radius_threshold if math.isfinite(args.radius_threshold) else "",
                "max_radius_neighbors": args.max_radius_neighbors,
                "seq_len": seq_len,
                "feature_dim": int(x.shape[1]),
            }
            global_values.append(values)

            summary = dict(base)
            summary.update(summarize_values(values, "topk"))
            max_rank = args.top_k if args.graph_mode == "topk" else min(args.max_radius_neighbors, 20)
            for rank in range(1, max_rank + 1):
                rank_values = values.new_tensor([row["similarity"] for row in neighbor_rows if row["rank"] == rank])
                summary.update(summarize_values(rank_values, f"rank{rank}"))
            summary_rows.append(summary)
            target_bins = bins_for_values(values, args.similarity, args.hist_bins)
            target_hist_rows = histogram_rows(values, target_bins, base)
            hist_rows.extend(target_hist_rows)

            plot_name = (
                f"layer_{layer_idx:02d}_{level}_"
                f"head_{safe_name(base['head'])}_{args.similarity}_"
                f"top{args.top_k}_centered_{int(args.center_tokens)}.svg"
            )
            write_histogram_svg(
                output_dir / "plots" / plot_name,
                target_hist_rows,
                (
                    f"Layer {layer_idx} {level} head={base['head']} "
                    f"causal {args.graph_mode} {args.similarity} "
                    f"(transform={key_transform})"
                ),
                x_label=f"{args.similarity} bucket",
            )

            if args.save_neighbors:
                for row in neighbor_rows:
                    all_neighbor_rows.append({**base, **row})
            global_neighbor_rows.extend(neighbor_rows)

            distances = [int(row["token_distance"]) for row in neighbor_rows]
            global_distances.extend(distances)
            dist_summary = dict(base)
            dist_summary.update(summarize_distances(distances, "distance", thresholds=[8, 32, 128, 256]))
            distance_summary_rows.append(dist_summary)
            dist_rows = binned_count_rows(distances, distance_bins(seq_len - 1), base, "distance")
            distance_hist_rows.extend(dist_rows)
            write_bar_svg(
                output_dir / "plots" / plot_name.replace(".svg", "_distance.svg"),
                dist_rows,
                "distance_bin",
                (
                    f"Layer {layer_idx} {level} head={base['head']} "
                    f"{args.graph_mode} neighbor token distance"
                ),
                "token distance bucket",
            )

            indegrees = indegree_values(neighbor_rows, seq_len)
            global_indegrees.extend(indegrees)
            indeg_summary = dict(base)
            indeg_summary.update(summarize_indegrees(indegrees, "indegree"))
            indegree_summary_rows.append(indeg_summary)
            indeg_rows = binned_count_rows(indegrees, indegree_bins(max(indegrees) if indegrees else 0), base, "indegree")
            indegree_hist_rows.extend(indeg_rows)
            write_bar_svg(
                output_dir / "plots" / plot_name.replace(".svg", "_indegree.svg"),
                indeg_rows,
                "indegree_bin",
                (
                    f"Layer {layer_idx} {level} head={base['head']} "
                    f"{args.graph_mode} in-degree"
                ),
                "in-degree bucket",
            )

            graph_summary = dict(base)
            graph_summary.update(summarize_graph_structure(neighbor_rows, seq_len, "graph"))
            graph_summary_rows.append(graph_summary)

            print(
                f"layer={layer_idx} level={level} head={base['head']} "
                f"count={int(values.numel())} mean={summary['topk_mean']:.4f} p95={summary['topk_p95']:.4f}",
                flush=True,
            )

    write_csv(output_dir / "summary_by_layer.csv", summary_rows)
    write_csv(output_dir / "histograms.csv", hist_rows)
    write_csv(output_dir / "distance_summary_by_layer.csv", distance_summary_rows)
    write_csv(output_dir / "distance_histograms.csv", distance_hist_rows)
    write_csv(output_dir / "indegree_summary_by_layer.csv", indegree_summary_rows)
    write_csv(output_dir / "indegree_histograms.csv", indegree_hist_rows)
    write_csv(output_dir / "graph_structure_summary_by_layer.csv", graph_summary_rows)
    if args.save_neighbors:
        write_csv(output_dir / "topk_neighbors.csv", all_neighbor_rows)

    if global_values:
        values = torch.cat([v for v in global_values if v.numel() > 0])
    else:
        values = torch.empty(0)
    global_base = {
        "layer": "all",
        "analysis_level": args.analysis_level,
        "head": "all",
        "graph_mode": args.graph_mode,
        "similarity": args.similarity,
        "center_tokens": bool(args.center_tokens),
        "key_transform": key_transform,
        "pc_remove_count": args.pc_remove_count,
        "top_k": args.top_k,
        "radius_threshold": args.radius_threshold if math.isfinite(args.radius_threshold) else "",
        "max_radius_neighbors": args.max_radius_neighbors,
        "seq_len": seq_len,
        "feature_dim": "mixed",
    }
    bins = bins_for_values(values, args.similarity, args.hist_bins)
    global_hist = histogram_rows(values, bins, global_base)
    write_csv(output_dir / "histogram_global.csv", global_hist)
    write_histogram_svg(
        output_dir / "histogram_global.svg",
        global_hist,
        f"K-cache causal {args.graph_mode} {args.similarity} ({args.analysis_level}, transform={key_transform})",
        x_label=f"{args.similarity} bucket",
    )
    global_distance_base = {
        "layer": "all",
        "analysis_level": args.analysis_level,
        "head": "all",
        "graph_mode": args.graph_mode,
        "similarity": args.similarity,
        "center_tokens": bool(args.center_tokens),
        "key_transform": key_transform,
        "pc_remove_count": args.pc_remove_count,
        "top_k": args.top_k,
        "radius_threshold": args.radius_threshold if math.isfinite(args.radius_threshold) else "",
        "max_radius_neighbors": args.max_radius_neighbors,
        "seq_len": seq_len,
        "feature_dim": "mixed",
    }
    global_distance_hist = binned_count_rows(global_distances, distance_bins(seq_len - 1), global_distance_base, "distance")
    write_csv(output_dir / "distance_histogram_global.csv", global_distance_hist)
    write_bar_svg(
        output_dir / "distance_histogram_global.svg",
        global_distance_hist,
        "distance_bin",
        f"K-cache causal {args.graph_mode} neighbor distance ({args.analysis_level}, transform={key_transform})",
        "token distance bucket",
    )

    global_indegree_base = dict(global_distance_base)
    global_indegree_hist = binned_count_rows(
        global_indegrees,
        indegree_bins(max(global_indegrees) if global_indegrees else 0),
        global_indegree_base,
        "indegree",
    )
    write_csv(output_dir / "indegree_histogram_global.csv", global_indegree_hist)
    write_bar_svg(
        output_dir / "indegree_histogram_global.svg",
        global_indegree_hist,
        "indegree_bin",
        f"K-cache causal {args.graph_mode} in-degree ({args.analysis_level}, transform={key_transform})",
        "in-degree bucket",
    )

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
        "layers": layer_indices,
        "heads": args.heads,
        "analysis_level": args.analysis_level,
        "graph_mode": args.graph_mode,
        "top_k": args.top_k,
        "radius_threshold": args.radius_threshold if math.isfinite(args.radius_threshold) else None,
        "max_radius_neighbors": args.max_radius_neighbors,
        "similarity": args.similarity,
        "center_tokens": bool(args.center_tokens),
        "key_transform": key_transform,
        "pc_remove_count": args.pc_remove_count,
        "hist_bins": bins,
        "hist_bins_arg": args.hist_bins,
        "num_summary_rows": len(summary_rows),
        "global": summarize_values(values, "topk"),
        "global_distance": summarize_distances(global_distances, "distance", thresholds=[8, 32, 128, 256]),
        "global_indegree": summarize_indegrees(global_indegrees, "indegree"),
        "global_graph": summarize_graph_structure(global_neighbor_rows, seq_len, "graph"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
