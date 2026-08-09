from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from analyze_state_pointer_query_manifold import pointer_token_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure hubness, bucket overlap, and cross-head intersection selectivity "
            "of train-only query-prototype block postings."
        )
    )
    parser.add_argument("--posting_path", required=True)
    parser.add_argument("--prototype_path", required=True)
    parser.add_argument("--step_profile", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--max_heads", type=int, default=8)
    parser.add_argument("--head_thresholds", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--pair_samples", type=int, default=20000)
    parser.add_argument("--route_batch", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def parse_set(spec: str) -> set[str]:
    return {item.strip() for item in spec.split(",") if item.strip()}


def parse_ints(spec: str) -> list[int]:
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    total = values.sum()
    if total <= 0:
        return 0.0
    count = len(values)
    weights = np.arange(1, count + 1, dtype=np.float64)
    return float((2.0 * np.dot(weights, values) / (count * total)) - (count + 1) / count)


@torch.inference_mode()
def route_all_pointer_vectors(
    *,
    step_payload: dict,
    test_indices: list[int],
    pointer_by_index: dict[int, list[int]],
    selected_heads: list[tuple[int, int]],
    centers: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[list[list[np.ndarray]], list[dict[str, float]]]:
    layers = [int(item) for item in step_payload["layers"]]
    layer_to_index = {layer: index for index, layer in enumerate(layers)}
    routes_by_head: list[list[np.ndarray]] = []
    route_stats = []
    for head_index, (layer, query_head) in enumerate(selected_heads):
        vectors = []
        lengths = []
        for index in test_indices:
            query = step_payload["svd_q"][
                index,
                pointer_by_index[index],
                layer_to_index[layer],
                query_head,
            ].float()
            vectors.append(query)
            lengths.append(len(query))
        matrix = F.normalize(torch.cat(vectors), dim=-1)
        head_centers = F.normalize(centers[head_index].float(), dim=-1).to(device)
        assignments = []
        nearest = []
        for start in range(0, len(matrix), batch_size):
            similarity = matrix[start : start + batch_size].to(device) @ head_centers.T
            values, indices = similarity.max(dim=1)
            assignments.append(indices.cpu())
            nearest.append(values.cpu())
        assignments_numpy = torch.cat(assignments).numpy()
        nearest_numpy = torch.cat(nearest).numpy()
        per_step = []
        offset = 0
        unique_counts = []
        for length in lengths:
            routed = np.unique(assignments_numpy[offset : offset + length])
            per_step.append(routed)
            unique_counts.append(len(routed))
            offset += length
        routes_by_head.append(per_step)
        route_stats.append(
            {
                "layer": layer,
                "query_head": query_head,
                "pointer_vectors": int(len(matrix)),
                "nearest_cosine_mean": float(nearest_numpy.mean()),
                "unique_prototypes_per_step_mean": mean(unique_counts),
            }
        )
    return routes_by_head, route_stats


def sample_pair_overlap(
    posting_ids: np.ndarray, *, samples: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    heads, prototypes, depth = posting_ids.shape
    sorted_ids = np.sort(posting_ids, axis=-1)
    within = []
    across = []
    for _ in range(samples):
        head = int(rng.integers(heads))
        left, right = rng.integers(prototypes, size=2)
        while right == left:
            right = int(rng.integers(prototypes))
        intersection = np.intersect1d(
            sorted_ids[head, left], sorted_ids[head, right], assume_unique=True
        ).size
        within.append(intersection / (2 * depth - intersection))

        left_head, right_head = rng.choice(heads, size=2, replace=False)
        left_proto, right_proto = rng.integers(prototypes, size=2)
        intersection = np.intersect1d(
            sorted_ids[left_head, left_proto],
            sorted_ids[right_head, right_proto],
            assume_unique=True,
        ).size
        across.append(intersection / (2 * depth - intersection))
    return {
        "samples": samples,
        "within_head_jaccard_mean": mean(within),
        "within_head_jaccard_p95": float(np.quantile(within, 0.95)),
        "across_head_jaccard_mean": mean(across),
        "across_head_jaccard_p95": float(np.quantile(across, 0.95)),
    }


def main() -> None:
    args = parse_args()
    postings = torch.load(args.posting_path, map_location="cpu", weights_only=False)
    prototypes = torch.load(args.prototype_path, map_location="cpu", weights_only=False)
    step_payload = torch.load(args.step_profile, map_location="cpu", weights_only=False)
    posting_ids = postings["ids"].long().numpy()
    selected_heads = [tuple(int(item) for item in pair) for pair in postings["selected_heads"]]
    centers = prototypes["centers"]
    if args.max_heads <= 0 or args.max_heads > len(selected_heads):
        raise ValueError("max_heads must be within indexed heads")
    selected_heads = selected_heads[: args.max_heads]
    posting_ids = posting_ids[: args.max_heads]
    centers = centers[: args.max_heads]
    num_blocks = int(postings["num_blocks"])
    splits = parse_set(args.splits)
    thresholds = parse_ints(args.head_thresholds)
    if min(thresholds) <= 0 or max(thresholds) > args.max_heads:
        raise ValueError("head thresholds must be within [1, max_heads]")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    test_indices = [
        index
        for index, step in enumerate(step_payload["steps"])
        if str(step["split"]) in splits
    ]
    pointer_by_index = {
        index: pointer_token_indices(
            tokenizer=tokenizer,
            step=step_payload["steps"][index],
            token_positions=step_payload["token_positions"][index],
        )
        for index in test_indices
    }
    routes_by_head, route_stats = route_all_pointer_vectors(
        step_payload=step_payload,
        test_indices=test_indices,
        pointer_by_index=pointer_by_index,
        selected_heads=selected_heads,
        centers=centers,
        device=torch.device(args.device),
        batch_size=args.route_batch,
    )

    flat_ids = posting_ids.reshape(-1)
    global_df = np.bincount(flat_ids, minlength=num_blocks)
    per_head_df = np.stack(
        [np.bincount(posting_ids[head].reshape(-1), minlength=num_blocks) for head in range(args.max_heads)]
    )
    total_postings = len(flat_ids)
    top1_count = max(1, int(math.ceil(0.01 * num_blocks)))
    top1_share = float(np.sort(global_df)[-top1_count:].sum() / total_postings)

    rows = []
    threshold_stats = {
        (step_index, threshold): {"blocks": [], "hits": []}
        for step_index in (0, 1)
        for threshold in thresholds
    }
    gold_head_counts = {0: [], 1: []}
    for local_step, profile_index in enumerate(test_indices):
        step = step_payload["steps"][profile_index]
        step_index = int(step["step_index"])
        gold = {int(item) for item in step["target_block_ids"]}
        head_count = np.zeros(num_blocks, dtype=np.uint8)
        for head_index in range(args.max_heads):
            routed_prototypes = routes_by_head[head_index][local_step]
            blocks = np.unique(posting_ids[head_index, routed_prototypes].reshape(-1))
            head_count[blocks] += 1
        gold_count = max(int(head_count[block_id]) for block_id in gold)
        gold_head_counts[step_index].append(gold_count)
        row = {
            "query_id": int(step["query_id"]),
            "step_index": step_index,
            "gold_head_count": gold_count,
            "thresholds": {},
        }
        for threshold in thresholds:
            candidate_count = int((head_count >= threshold).sum())
            hit = gold_count >= threshold
            threshold_stats[(step_index, threshold)]["blocks"].append(candidate_count)
            threshold_stats[(step_index, threshold)]["hits"].append(hit)
            row["thresholds"][str(threshold)] = {
                "candidate_blocks": candidate_count,
                "gold_hit": hit,
            }
        rows.append(row)

    concurrence = []
    for step_index in (0, 1):
        for threshold in thresholds:
            values = threshold_stats[(step_index, threshold)]
            candidate_fraction = mean(values["blocks"]) / num_blocks
            gold_recall = mean(values["hits"])
            concurrence.append(
                {
                    "step_index": step_index,
                    "minimum_supporting_heads": threshold,
                    "mean_candidate_blocks": mean(values["blocks"]),
                    "median_candidate_blocks": float(np.median(values["blocks"])),
                    "candidate_block_distribution": distribution(
                        np.asarray(values["blocks"])
                    ),
                    "mean_candidate_fraction": candidate_fraction,
                    "gold_recall": gold_recall,
                    "gold_enrichment_over_random_block": (
                        gold_recall / candidate_fraction
                        if candidate_fraction > 0
                        else None
                    ),
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "real-K train-only prototype posting hubness and head concurrence",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "gold_usage": "evaluation of routed posting intersections only",
        "posting_path": args.posting_path,
        "prototype_path": args.prototype_path,
        "step_profile": args.step_profile,
        "num_blocks": num_blocks,
        "heads": args.max_heads,
        "prototypes_per_head": int(posting_ids.shape[1]),
        "posting_depth": int(posting_ids.shape[2]),
        "total_posting_entries": total_postings,
        "posting_index_bytes": int(Path(args.posting_path).stat().st_size),
        "global_block_document_frequency": {
            **distribution(global_df),
            "gini": gini(global_df),
            "top1pct_block_posting_share": top1_share,
            "unseen_block_rate": float((global_df == 0).mean()),
        },
        "per_head_document_frequency_macro": {
            "mean": mean(row.mean() for row in per_head_df),
            "p95_mean": mean(np.quantile(row, 0.95) for row in per_head_df),
            "max_mean": mean(row.max() for row in per_head_df),
            "gini_mean": mean(gini(row) for row in per_head_df),
        },
        "random_posting_overlap": sample_pair_overlap(
            posting_ids, samples=args.pair_samples, seed=args.seed
        ),
        "route_stats": route_stats,
        "gold_supporting_head_count": {
            str(step_index): distribution(np.asarray(gold_head_counts[step_index]))
            for step_index in (0, 1)
        },
        "head_concurrence": concurrence,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
