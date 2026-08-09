from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_hierarchical_kmean_search import build_vectors


SCAN_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search real K neighbors through original document record centroids."
    )
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--records_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--query_blocks", type=int, default=500)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    vectors = build_vectors(Path(args.index_dir), device)
    records = read_jsonl(Path(args.records_path))
    record_vectors = []
    block_to_record = torch.full(
        (len(vectors),), -1, dtype=torch.long, device=device
    )
    for record_index, record in enumerate(records):
        start = int(record["block_start"])
        count = int(record["block_count"])
        end = min(start + count, len(vectors))
        if end <= start:
            raise ValueError("record has no indexed blocks")
        record_vectors.append(vectors[start:end].mean(dim=0))
        block_to_record[start:end] = record_index
    if bool((block_to_record < 0).any()):
        raise ValueError("records do not cover every indexed block")
    record_vectors_tensor = F.normalize(torch.stack(record_vectors), dim=-1)

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
    record_scores = query @ record_vectors_tensor.transpose(0, 1)
    ranked_records = torch.argsort(record_scores, dim=1, descending=True)

    own_record_recall = []
    for index in range(query_count):
        record_id = int(block_to_record[query_ids_tensor[index]].item())
        start = int(records[record_id]["block_start"])
        end = start + int(records[record_id]["block_count"])
        own_record_recall.append(
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

    experiments = []
    for fraction in SCAN_FRACTIONS:
        target_blocks = math.ceil(len(vectors) * fraction)
        recalls = []
        scans = []
        selected_record_counts = []
        for index in range(query_count):
            selected = []
            accumulated = 0
            for record_id_value in ranked_records[index].cpu().tolist():
                record = records[int(record_id_value)]
                selected.append(int(record_id_value))
                accumulated += int(record["block_count"])
                if accumulated >= target_blocks:
                    break
            candidate_parts = []
            for record_id in selected:
                start = int(records[record_id]["block_start"])
                count = int(records[record_id]["block_count"])
                candidate_parts.append(
                    torch.arange(start, min(start + count, len(vectors)), device=device)
                )
            candidates = torch.cat(candidate_parts).unique()
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
            scans.append(len(candidates) / len(vectors))
            selected_record_counts.append(len(selected))
        dot_fraction = len(records) / len(vectors) + statistics.fmean(scans)
        experiments.append(
            {
                "requested_scan_fraction": fraction,
                "mean_actual_scan_fraction": statistics.fmean(scans),
                "mean_selected_records": statistics.fmean(selected_record_counts),
                "mean_exact_neighbor_recall": statistics.fmean(recalls),
                "estimated_dot_product_fraction_vs_full_scan": dot_fraction,
                "estimated_dot_product_speedup": 1.0 / dot_fraction,
            }
        )

    payload = {
        "source": "record-aware hierarchy over global-centered real K block centroids",
        "contains_synthetic_vectors": False,
        "index_dir": args.index_dir,
        "records_path": args.records_path,
        "blocks": len(vectors),
        "records": len(records),
        "query_blocks": query_count,
        "neighbors": args.neighbors,
        "mean_own_record_exact_neighbor_recall": statistics.fmean(own_record_recall),
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
