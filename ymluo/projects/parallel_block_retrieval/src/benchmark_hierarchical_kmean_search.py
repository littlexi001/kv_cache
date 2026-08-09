from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


GROUP_SIZES = (8, 16, 32, 64, 128)
SCAN_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark contiguous superblock search for KV-neighbor expansion."
    )
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--query_blocks", type=int, default=1000)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def build_vectors(index_dir: Path, device: torch.device) -> torch.Tensor:
    values = torch.from_numpy(
        np.asarray(np.load(index_dir / "pre_k_mean.npy", mmap_mode="r"), dtype=np.float32)
    ).to(device)
    values = values - values.mean(dim=0, keepdim=True)
    values = F.normalize(values, dim=-1)
    return F.normalize(values.flatten(1), dim=-1)


def group_vectors(vectors: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    group_count = math.ceil(len(vectors) / group_size)
    padded_count = group_count * group_size
    if padded_count > len(vectors):
        padding = torch.zeros(
            padded_count - len(vectors),
            vectors.shape[1],
            dtype=vectors.dtype,
            device=vectors.device,
        )
        padded = torch.cat([vectors, padding], dim=0)
        mask = torch.cat(
            [
                torch.ones(len(vectors), device=vectors.device),
                torch.zeros(len(padding), device=vectors.device),
            ]
        )
    else:
        padded = vectors
        mask = torch.ones(len(vectors), device=vectors.device)
    grouped = padded.reshape(group_count, group_size, -1)
    grouped_mask = mask.reshape(group_count, group_size, 1)
    means = (grouped * grouped_mask).sum(dim=1) / grouped_mask.sum(dim=1).clamp_min(1)
    return F.normalize(means, dim=-1), group_count


def main() -> None:
    args = parse_args()
    if args.query_blocks <= 0 or args.neighbors <= 0:
        raise ValueError("query_blocks and neighbors must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    index_dir = Path(args.index_dir)
    vectors = build_vectors(index_dir, device)
    rng = np.random.default_rng(args.seed)
    query_count = min(args.query_blocks, len(vectors))
    query_ids = np.sort(rng.choice(len(vectors), size=query_count, replace=False))
    query = vectors.index_select(
        0, torch.from_numpy(query_ids).to(device=device, dtype=torch.long)
    )

    exact_started = time.perf_counter()
    exact_scores = query @ vectors.transpose(0, 1)
    exact_scores[
        torch.arange(query_count, device=device),
        torch.from_numpy(query_ids).to(device=device, dtype=torch.long),
    ] = -torch.inf
    exact_neighbors = torch.topk(
        exact_scores, k=args.neighbors, dim=1, largest=True, sorted=False
    ).indices
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    exact_seconds = time.perf_counter() - exact_started

    experiments: list[dict[str, Any]] = []
    for group_size in GROUP_SIZES:
        groups, group_count = group_vectors(vectors, group_size)
        group_scores = query @ groups.transpose(0, 1)
        query_group = torch.from_numpy(query_ids // group_size).to(device)
        same_group_recall = []
        for index in range(query_count):
            start = int(query_group[index].item()) * group_size
            end = min(start + group_size, len(vectors))
            same_group_recall.append(
                float(
                    (
                        (exact_neighbors[index] >= start)
                        & (exact_neighbors[index] < end)
                    )
                    .float()
                    .mean()
                    .item()
                )
            )
        for scan_fraction in SCAN_FRACTIONS:
            target_blocks = max(group_size, math.ceil(len(vectors) * scan_fraction))
            selected_groups = min(
                group_count, math.ceil(target_blocks / group_size)
            )
            top_groups = torch.topk(
                group_scores, k=selected_groups, dim=1, largest=True, sorted=False
            ).indices
            recalls = []
            scanned = []
            search_started = time.perf_counter()
            for index in range(query_count):
                group_ids = top_groups[index]
                starts = group_ids * group_size
                offsets = torch.arange(group_size, device=device)[None, :]
                candidate_ids = (starts[:, None] + offsets).reshape(-1)
                candidate_ids = candidate_ids[candidate_ids < len(vectors)].unique()
                candidate_ids = candidate_ids[candidate_ids != int(query_ids[index])]
                candidate_scores = vectors.index_select(0, candidate_ids) @ query[index]
                take = min(args.neighbors, len(candidate_ids))
                found = candidate_ids.index_select(
                    0,
                    torch.topk(candidate_scores, k=take, largest=True).indices,
                )
                recalls.append(
                    len(set(found.cpu().tolist()).intersection(exact_neighbors[index].cpu().tolist()))
                    / args.neighbors
                )
                scanned.append(len(candidate_ids) / len(vectors))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            search_seconds = time.perf_counter() - search_started
            experiments.append(
                {
                    "group_size": group_size,
                    "requested_scan_fraction": scan_fraction,
                    "selected_groups": selected_groups,
                    "mean_actual_scan_fraction": statistics.fmean(scanned),
                    "mean_exact_neighbor_recall": statistics.fmean(recalls),
                    "same_contiguous_group_neighbor_recall": statistics.fmean(
                        same_group_recall
                    ),
                    "search_seconds": search_seconds,
                }
            )

    payload = {
        "source": "hierarchical search over global-centered real K block centroids",
        "contains_synthetic_vectors": False,
        "index_dir": str(index_dir),
        "blocks": len(vectors),
        "profiles": 4,
        "combined_dimension": int(vectors.shape[1]),
        "query_blocks": query_count,
        "neighbors": args.neighbors,
        "exact_seconds": exact_seconds,
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
