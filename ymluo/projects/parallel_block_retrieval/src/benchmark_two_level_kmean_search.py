from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_hierarchical_kmean_search import build_vectors, group_vectors


PARENT_FRACTIONS = (0.05, 0.10, 0.20, 0.50)
BLOCK_FRACTIONS = (0.005, 0.01, 0.02, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-level parent/leaf K-centroid search for seeded KV expansion."
    )
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--query_blocks", type=int, default=500)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--parent_size", type=int, default=64)
    parser.add_argument("--leaf_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.parent_size % args.leaf_size:
        raise ValueError("parent_size must be divisible by leaf_size")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    vectors = build_vectors(Path(args.index_dir), device)
    leaf_vectors, leaf_count = group_vectors(vectors, args.leaf_size)
    parent_vectors, parent_count = group_vectors(vectors, args.parent_size)
    leaves_per_parent = args.parent_size // args.leaf_size

    rng = np.random.default_rng(args.seed)
    query_count = min(args.query_blocks, len(vectors))
    query_ids = np.sort(rng.choice(len(vectors), size=query_count, replace=False))
    query_ids_tensor = torch.from_numpy(query_ids).to(device=device, dtype=torch.long)
    query = vectors.index_select(0, query_ids_tensor)
    exact_scores = query @ vectors.transpose(0, 1)
    exact_scores[torch.arange(query_count, device=device), query_ids_tensor] = -torch.inf
    exact_neighbors = torch.topk(
        exact_scores, k=args.neighbors, dim=1, largest=True, sorted=False
    ).indices
    parent_scores = query @ parent_vectors.transpose(0, 1)

    experiments: list[dict[str, Any]] = []
    for parent_fraction in PARENT_FRACTIONS:
        selected_parent_count = min(
            parent_count, max(1, math.ceil(parent_count * parent_fraction))
        )
        selected_parents = torch.topk(
            parent_scores,
            k=selected_parent_count,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        for block_fraction in BLOCK_FRACTIONS:
            target_leaf_count = max(
                1, math.ceil(len(vectors) * block_fraction / args.leaf_size)
            )
            recalls = []
            actual_scan = []
            actual_leaf = []
            for index in range(query_count):
                parents = selected_parents[index]
                leaf_offsets = torch.arange(leaves_per_parent, device=device)[None, :]
                candidate_leaves = (
                    parents[:, None] * leaves_per_parent + leaf_offsets
                ).reshape(-1)
                candidate_leaves = candidate_leaves[candidate_leaves < leaf_count].unique()
                leaf_scores = leaf_vectors.index_select(0, candidate_leaves) @ query[index]
                take_leaf = min(target_leaf_count, len(candidate_leaves))
                selected_leaves = candidate_leaves.index_select(
                    0, torch.topk(leaf_scores, k=take_leaf, largest=True).indices
                )
                starts = selected_leaves * args.leaf_size
                block_offsets = torch.arange(args.leaf_size, device=device)[None, :]
                candidates = (starts[:, None] + block_offsets).reshape(-1)
                candidates = candidates[candidates < len(vectors)].unique()
                candidates = candidates[candidates != int(query_ids[index])]
                scores = vectors.index_select(0, candidates) @ query[index]
                take = min(args.neighbors, len(candidates))
                found = candidates.index_select(
                    0, torch.topk(scores, k=take, largest=True).indices
                )
                recalls.append(
                    len(
                        set(found.cpu().tolist()).intersection(
                            exact_neighbors[index].cpu().tolist()
                        )
                    )
                    / args.neighbors
                )
                actual_scan.append(len(candidates) / len(vectors))
                actual_leaf.append(len(candidate_leaves) / leaf_count)

            dot_fraction = (
                1.0 / args.parent_size
                + statistics.fmean(actual_leaf) / args.leaf_size
                + statistics.fmean(actual_scan)
            )
            experiments.append(
                {
                    "parent_fraction": parent_fraction,
                    "requested_block_scan_fraction": block_fraction,
                    "selected_parents": selected_parent_count,
                    "mean_candidate_leaf_fraction": statistics.fmean(actual_leaf),
                    "mean_actual_block_scan_fraction": statistics.fmean(actual_scan),
                    "mean_exact_neighbor_recall": statistics.fmean(recalls),
                    "estimated_dot_product_fraction_vs_full_scan": dot_fraction,
                    "estimated_dot_product_speedup": 1.0 / dot_fraction,
                }
            )

    payload = {
        "source": "two-level hierarchical real K-centroid neighbor search",
        "contains_synthetic_vectors": False,
        "index_dir": args.index_dir,
        "blocks": len(vectors),
        "query_blocks": query_count,
        "neighbors": args.neighbors,
        "parent_size": args.parent_size,
        "leaf_size": args.leaf_size,
        "experiments": experiments,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
