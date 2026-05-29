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


def transform_keys(x: torch.Tensor, center_tokens: bool) -> torch.Tensor:
    if center_tokens:
        return x - x.mean(dim=0, keepdim=True)
    return x


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
    elif similarity == "dot":
        x_score = x
    else:
        raise ValueError(f"Unsupported similarity: {similarity}")

    sim = x_score @ x_score.T
    indices = torch.arange(n)
    invalid = indices[None, :] >= indices[:, None]
    sim = sim.masked_fill(invalid, -float("inf"))

    actual_k = min(top_k, n - 1)
    values, neighbor_indices = torch.topk(sim, k=actual_k, dim=1)
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


def summarize_values(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    if values.numel() == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p95": float("nan"),
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


def summarize_distances(distances: List[int], prefix: str, thresholds: Iterable[int]) -> Dict[str, float]:
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
        return result

    values = torch.tensor(distances, dtype=torch.float32)
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


def write_histogram_svg(path: Path, rows: List[Dict], title: str, width: int = 1000, height: int = 560) -> None:
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
            f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#333">similarity bucket</text>',
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
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--similarity", choices=["cos", "dot"], default="cos")
    parser.add_argument("--center-tokens", action="store_true")
    parser.add_argument("--hist-bins", default="-1.0:1.0:0.05")
    parser.add_argument("--save-neighbors", action="store_true")
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1.")

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

    write_csv(output_dir / "tokens.csv", decode_tokens(tokenizer, input_ids))
    outputs = run_forward(model, input_ids, device)

    num_layers = int(getattr(model.config, "num_hidden_layers"))
    layer_indices = select_indices(num_layers, args.layers)
    bins = parse_bins(args.hist_bins)

    all_neighbor_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    hist_rows: List[Dict] = []
    distance_summary_rows: List[Dict] = []
    distance_hist_rows: List[Dict] = []
    indegree_summary_rows: List[Dict] = []
    indegree_hist_rows: List[Dict] = []
    global_values: List[torch.Tensor] = []
    global_distances: List[int] = []
    global_indegrees: List[int] = []

    for layer_idx in layer_indices:
        cache = extract_cache_tensor(outputs.past_key_values, layer_idx, "K")
        head_indices = select_indices(cache.shape[0], args.heads)
        targets: List[Tuple[str, int, torch.Tensor]] = []

        if args.analysis_level in {"token", "both"}:
            targets.append(("token", -1, token_level_matrix(cache, seq_len)))
        if args.analysis_level in {"head", "both"}:
            targets.extend(("head", head_idx, x) for head_idx, x in head_level_matrices(cache, seq_len, head_indices))

        for level, head_idx, x in targets:
            x = transform_keys(x, args.center_tokens)
            neighbor_rows, values = causal_topk_similarity(x, args.top_k, args.similarity)
            base = {
                "layer": layer_idx,
                "analysis_level": level,
                "head": "all" if head_idx < 0 else head_idx,
                "similarity": args.similarity,
                "center_tokens": bool(args.center_tokens),
                "top_k": args.top_k,
                "seq_len": seq_len,
                "feature_dim": int(x.shape[1]),
            }
            global_values.append(values)

            summary = dict(base)
            summary.update(summarize_values(values, "topk"))
            for rank in range(1, args.top_k + 1):
                rank_values = values.new_tensor([row["similarity"] for row in neighbor_rows if row["rank"] == rank])
                summary.update(summarize_values(rank_values, f"rank{rank}"))
            summary_rows.append(summary)
            target_hist_rows = histogram_rows(values, bins, base)
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
                    f"causal top-{args.top_k} {args.similarity} "
                    f"(centered={args.center_tokens})"
                ),
            )

            if args.save_neighbors:
                for row in neighbor_rows:
                    all_neighbor_rows.append({**base, **row})

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
                    f"top-{args.top_k} neighbor token distance"
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
                    f"top-{args.top_k} in-degree"
                ),
                "in-degree bucket",
            )

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
        "similarity": args.similarity,
        "center_tokens": bool(args.center_tokens),
        "top_k": args.top_k,
        "seq_len": seq_len,
        "feature_dim": "mixed",
    }
    global_hist = histogram_rows(values, bins, global_base)
    write_csv(output_dir / "histogram_global.csv", global_hist)
    write_histogram_svg(
        output_dir / "histogram_global.svg",
        global_hist,
        f"K-cache causal top-{args.top_k} {args.similarity} similarity ({args.analysis_level}, centered={args.center_tokens})",
    )
    global_distance_base = {
        "layer": "all",
        "analysis_level": args.analysis_level,
        "head": "all",
        "similarity": args.similarity,
        "center_tokens": bool(args.center_tokens),
        "top_k": args.top_k,
        "seq_len": seq_len,
        "feature_dim": "mixed",
    }
    global_distance_hist = binned_count_rows(global_distances, distance_bins(seq_len - 1), global_distance_base, "distance")
    write_csv(output_dir / "distance_histogram_global.csv", global_distance_hist)
    write_bar_svg(
        output_dir / "distance_histogram_global.svg",
        global_distance_hist,
        "distance_bin",
        f"K-cache causal top-{args.top_k} neighbor distance ({args.analysis_level}, centered={args.center_tokens})",
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
        f"K-cache causal top-{args.top_k} in-degree ({args.analysis_level}, centered={args.center_tokens})",
        "in-degree bucket",
    )

    summary = {
        "model_path": args.model_path,
        "text_path": args.text_path,
        "output_dir": str(output_dir),
        "device": str(device),
        "dtype": args.dtype,
        "seq_len": seq_len,
        "layers": layer_indices,
        "heads": args.heads,
        "analysis_level": args.analysis_level,
        "top_k": args.top_k,
        "similarity": args.similarity,
        "center_tokens": bool(args.center_tokens),
        "hist_bins": bins,
        "num_summary_rows": len(summary_rows),
        "global": summarize_values(values, "topk"),
        "global_distance": summarize_distances(global_distances, "distance", thresholds=[8, 32, 128, 256]),
        "global_indegree": summarize_indegrees(global_indegrees, "indegree"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
