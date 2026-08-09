from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from benchmark_selected_head_debiased_retrieval import read_selection
from run_all_head_prior_debiased_retrieval import read_jsonl, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how many 10M blocks survive mathematically safe "
            "center-radius support-function bounds."
        )
    )
    parser.add_argument("--bound_dir", required=True)
    parser.add_argument("--query_profiles", required=True)
    parser.add_argument("--selection_csv", required=True)
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument("--full_raw_reference_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gate_feature", default="raw_top1_block_diversity")
    parser.add_argument("--heads_per_fold", type=int, default=16)
    parser.add_argument("--query_batch", type=int, default=4)
    parser.add_argument("--safety_tolerance", type=float, default=1e-4)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.percentile(values.astype(np.float64), quantile))


def main() -> None:
    args = parse_args()
    if args.query_batch <= 0 or args.safety_tolerance < 0:
        raise ValueError("query_batch must be positive and tolerance nonnegative")
    rank, world_size, _local_rank, device = setup_distributed()
    bound_dir = Path(args.bound_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    bound = json.loads((bound_dir / "summary.json").read_text(encoding="utf-8"))
    packed_dir = Path(bound["packed_profile_dir"])
    packed = json.loads((packed_dir / "summary.json").read_text(encoding="utf-8"))
    segments_list = [int(item) for item in bound["segments"]]
    num_blocks = int(bound["num_blocks"])
    retained_tokens = int(bound["retained_tokens"])
    num_query_heads = int(bound["num_query_heads"])
    num_kv_heads = int(bound["num_kv_heads"])
    repeat_groups = num_query_heads // num_kv_heads
    selected_by_fold = read_selection(
        Path(args.selection_csv), args.gate_feature, args.heads_per_fold
    )
    union_flat_heads = sorted(
        {head for heads in selected_by_fold.values() for head in heads}
    )
    local_flat_heads = union_flat_heads[rank::world_size]
    queries = read_jsonl(Path(args.queries_jsonl))
    query_payload = torch.load(
        Path(args.query_profiles), map_location="cpu", weights_only=False
    )
    query_vectors = query_payload["svd_q"]
    query_mask = query_payload["mask"]
    with np.load(Path(args.full_raw_reference_npz)) as reference:
        reference_ids = reference["block_ids"]
        reference_scores = reference["scores"]
        fold_ids = reference["fold_ids"].astype(np.int64)
        reference_layers = reference["layers"].astype(np.int64)
    if len(queries) != len(fold_ids):
        raise ValueError("query metadata and reference counts differ")

    shard_by_rank = {int(shard["rank"]): shard for shard in packed["shards"]}
    files_by_layer: dict[int, list[dict[str, Any]]] = {}
    bytes_by_segments = {segments: 0 for segments in segments_list}
    for result in bound["files"]:
        layer = int(result["layer"])
        shard_rank = int(result["rank"])
        item = {
            "rank": shard_rank,
            "block_start": int(shard_by_rank[shard_rank]["block_start"]),
            "block_end": int(shard_by_rank[shard_rank]["block_end"]),
            "segments": {
                int(file["segments"]): file for file in result["files"]
            },
        }
        files_by_layer.setdefault(layer, []).append(item)
        for file in result["files"]:
            bytes_by_segments[int(file["segments"])] += int(file["bytes"])
    for items in files_by_layer.values():
        items.sort(key=lambda row: row["block_start"])

    local_rows: list[dict[str, Any]] = []
    local_head_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for flat_head in local_flat_heads:
        layer_index, query_head = divmod(flat_head, num_query_heads)
        layer = int(reference_layers[layer_index])
        original_kv_head = query_head // repeat_groups
        packed_kv_heads = [
            int(item)
            for item in bound["selected_kv_heads_by_layer"][str(layer)]
        ]
        packed_kv_index = packed_kv_heads.index(original_kv_head)
        active_queries = np.asarray(
            [
                query_index
                for query_index, fold in enumerate(fold_ids)
                if flat_head in selected_by_fold[int(fold)]
            ],
            dtype=np.int64,
        )
        if active_queries.size == 0:
            continue
        layer_queries = query_vectors[active_queries, :, layer_index, query_head]
        layer_masks = query_mask[active_queries]
        thresholds = reference_scores[
            active_queries, layer_index, query_head, -1
        ].astype(np.float32)
        exact_ids = reference_ids[active_queries, layer_index, query_head]
        exact_scores = reference_scores[active_queries, layer_index, query_head]
        candidate_counts = {
            segments: np.zeros(len(active_queries), dtype=np.int32)
            for segments in segments_list
        }
        min_slacks = {
            segments: np.full(len(active_queries), np.inf, dtype=np.float32)
            for segments in segments_list
        }
        head_started = time.perf_counter()
        for shard in files_by_layer[layer]:
            block_start = int(shard["block_start"])
            block_end = int(shard["block_end"])
            for segments in segments_list:
                file = shard["segments"][segments]
                centers_np = np.load(
                    bound_dir / file["center_path"], mmap_mode="r"
                )[:, packed_kv_index]
                radii_np = np.load(
                    bound_dir / file["radius_path"], mmap_mode="r"
                )[:, packed_kv_index]
                centers = torch.from_numpy(np.array(centers_np, copy=True)).to(
                    device=device, non_blocking=True
                )
                radii = torch.from_numpy(np.array(radii_np, copy=True)).to(
                    device=device, non_blocking=True
                )
                for query_start in range(0, len(active_queries), args.query_batch):
                    query_end = min(
                        len(active_queries), query_start + args.query_batch
                    )
                    q = layer_queries[query_start:query_end].to(
                        device=device, dtype=torch.float32, non_blocking=True
                    )
                    mask = layer_masks[query_start:query_end].to(
                        device=device, dtype=torch.float32, non_blocking=True
                    )
                    q_norm = torch.linalg.vector_norm(q, dim=-1)
                    upper_tokens = torch.einsum(
                        "qtd,bsd->qtbs", q, centers
                    ) + q_norm[:, :, None, None] * radii[None, None]
                    upper = upper_tokens.amax(dim=-1)
                    valid = mask.sum(dim=1).clamp_min(1)
                    upper_score = (upper * mask[:, :, None]).sum(dim=1) / valid[
                        :, None
                    ]
                    upper_cpu = upper_score.cpu().numpy()
                    threshold = thresholds[query_start:query_end, None]
                    candidate_counts[segments][query_start:query_end] += np.sum(
                        upper_cpu >= threshold - args.safety_tolerance,
                        axis=1,
                    ).astype(np.int32)
                    for batch_index in range(query_end - query_start):
                        output_index = query_start + batch_index
                        ids = exact_ids[output_index]
                        inside = (ids >= block_start) & (ids < block_end)
                        if not np.any(inside):
                            continue
                        local_ids = ids[inside] - block_start
                        slack = (
                            upper_cpu[batch_index, local_ids]
                            - exact_scores[output_index, inside]
                        )
                        min_slacks[segments][output_index] = min(
                            min_slacks[segments][output_index],
                            float(np.min(slack)),
                        )
                del centers, radii

        torch.cuda.synchronize(device)
        head_seconds = time.perf_counter() - head_started
        for output_index, query_index in enumerate(active_queries):
            for segments in segments_list:
                fraction = candidate_counts[segments][output_index] / num_blocks
                cost_ratio = segments / retained_tokens + fraction
                local_rows.append(
                    {
                        "query_index": int(query_index),
                        "dataset": str(queries[int(query_index)]["dataset"]),
                        "fold": int(fold_ids[query_index]),
                        "flat_head": flat_head,
                        "layer": layer,
                        "query_head": query_head,
                        "segments": segments,
                        "candidate_blocks": int(
                            candidate_counts[segments][output_index]
                        ),
                        "candidate_fraction": fraction,
                        "estimated_dot_cost_ratio": cost_ratio,
                        "estimated_dot_speedup": 1.0 / cost_ratio,
                        "min_exact_top16_bound_slack": float(
                            min_slacks[segments][output_index]
                        ),
                        "safety_violation": int(
                            min_slacks[segments][output_index]
                            < -args.safety_tolerance
                        ),
                    }
                )
        local_head_rows.append(
            {
                "flat_head": flat_head,
                "layer": layer,
                "query_head": query_head,
                "active_queries": len(active_queries),
                "seconds_all_segments": head_seconds,
            }
        )
        print(json.dumps(local_head_rows[-1]), flush=True)

    if world_size > 1:
        gathered_rows: list[list[dict[str, Any]] | None] = [
            None for _ in range(world_size)
        ]
        gathered_heads: list[list[dict[str, Any]] | None] = [
            None for _ in range(world_size)
        ]
        dist.all_gather_object(gathered_rows, local_rows)
        dist.all_gather_object(gathered_heads, local_head_rows)
        rows = [row for part in gathered_rows if part for row in part]
        head_rows = [row for part in gathered_heads if part for row in part]
    else:
        rows = local_rows
        head_rows = local_head_rows

    if rank == 0:
        rows.sort(key=lambda row: (row["segments"], row["query_index"], row["flat_head"]))
        head_rows.sort(key=lambda row: row["flat_head"])
        write_csv(output_dir / "query_head_results.csv", rows)
        write_csv(output_dir / "head_runtime.csv", head_rows)
        summary_rows: list[dict[str, Any]] = []
        one_billion_scale = 1_000_000_000 / (
            int(bound["num_blocks"]) * 256
        )
        for segments in segments_list:
            subset = [row for row in rows if row["segments"] == segments]
            fractions = np.asarray(
                [row["candidate_fraction"] for row in subset], dtype=np.float64
            )
            cost_ratios = np.asarray(
                [row["estimated_dot_cost_ratio"] for row in subset],
                dtype=np.float64,
            )
            slacks = np.asarray(
                [row["min_exact_top16_bound_slack"] for row in subset],
                dtype=np.float64,
            )
            summary_rows.append(
                {
                    "segments": segments,
                    "query_head_pairs": len(subset),
                    "mean_candidate_fraction": float(fractions.mean()),
                    "median_candidate_fraction": percentile(fractions, 50),
                    "p95_candidate_fraction": percentile(fractions, 95),
                    "mean_estimated_dot_cost_ratio": float(cost_ratios.mean()),
                    "estimated_dot_speedup_from_mean_cost": float(
                        1.0 / cost_ratios.mean()
                    ),
                    "safety_violations": int(
                        np.sum(slacks < -args.safety_tolerance)
                    ),
                    "min_exact_top16_bound_slack": float(slacks.min()),
                    "index_bytes_10m": int(bytes_by_segments[segments]),
                    "index_bytes_1b_linear_projection": int(
                        bytes_by_segments[segments] * one_billion_scale
                    ),
                }
            )
        write_csv(output_dir / "summary.csv", summary_rows)
        summary = {
            "experiment": "safe_center_radius_support_bound_pruning",
            "contains_synthetic_vectors": False,
            "selection_uses_gold": False,
            "gold_used_for_threshold": False,
            "threshold_source": "exact raw Top16 score from frozen QK ranking",
            "bound_is_mathematically_safe_in_svd32": True,
            "bound_formula": bound["bound"],
            "queries": len(queries),
            "active_query_head_pairs": len(rows) // len(segments_list),
            "blocks": num_blocks,
            "retained_tokens_per_block": retained_tokens,
            "segments": segments_list,
            "world_size": world_size,
            "safety_tolerance": args.safety_tolerance,
            "summary": summary_rows,
            "total_wall_seconds": time.perf_counter() - started,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
