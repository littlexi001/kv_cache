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

from run_all_head_prior_debiased_retrieval import (
    METHODS,
    deterministic_merge_distributed_topk,
    deterministic_update_topk,
    empty_topk,
    read_jsonl,
    setup_distributed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a real sparse scan over the union of train-only heads "
            "selected by all leave-one-dataset-out folds."
        )
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--query_profiles", required=True)
    parser.add_argument("--selection_csv", required=True)
    parser.add_argument("--full_reference_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gate_feature", default="raw_top1_block_diversity")
    parser.add_argument("--heads_per_fold", type=int, default=16)
    parser.add_argument("--top_per_head", type=int, default=16)
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--query_batch", type=int, default=8)
    parser.add_argument("--block_chunk", type=int, default=64)
    parser.add_argument("--exclude_block_prefix_tokens", type=int, default=16)
    parser.add_argument("--std_epsilon", type=float, default=1e-4)
    return parser.parse_args()


def read_selection(
    path: Path, feature: str, heads_per_fold: int
) -> dict[int, list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[int, list[tuple[int, int]]] = {}
    for row in rows:
        if row["feature"] != feature or int(row["head_count"]) != heads_per_fold:
            continue
        fold = int(row["fold"])
        selected.setdefault(fold, []).append(
            (int(row["selected_rank"]), int(row["flat_head"]))
        )
    result = {
        fold: [flat_head for _rank, flat_head in sorted(values)]
        for fold, values in selected.items()
    }
    if not result or any(len(heads) != heads_per_fold for heads in result.values()):
        raise ValueError("selection file does not contain one complete head set per fold")
    return result


def score_selected_layer(
    *,
    layer: int,
    shards: list[dict[str, Any]],
    profile_dir: Path,
    queries: torch.Tensor,
    query_mask: torch.Tensor,
    fold_ids: torch.Tensor,
    selected_query_heads: list[int],
    packed_kv_heads: list[int] | None,
    original_query_heads: int,
    num_kv_heads: int,
    top_per_head: int,
    query_batch: int,
    block_chunk: int,
    exclude_block_prefix_tokens: int,
    std_epsilon: float,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    query_count, query_tokens, _query_heads, rank_dim = queries.shape
    repeat_groups = original_query_heads // num_kv_heads
    if original_query_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    selected = torch.as_tensor(
        selected_query_heads, dtype=torch.long, device=device
    )
    selected_queries = queries.index_select(2, selected)
    selected_count = len(selected_query_heads)
    original_kv_indices = [
        head // repeat_groups for head in selected_query_heads
    ]
    if packed_kv_heads is None:
        stored_kv_indices = original_kv_indices
    else:
        packed_position = {
            original_head: index
            for index, original_head in enumerate(packed_kv_heads)
        }
        if any(head not in packed_position for head in original_kv_indices):
            raise ValueError(f"packed profile is missing a KV head for layer {layer}")
        stored_kv_indices = [
            packed_position[head] for head in original_kv_indices
        ]
    rankings = {
        method: empty_topk(query_count, selected_count, top_per_head, device)
        for method in METHODS
    }
    fold_splits: list[tuple[torch.Tensor, torch.Tensor]] = []
    for fold in torch.unique(fold_ids, sorted=True).tolist():
        test_indices = torch.nonzero(fold_ids == fold, as_tuple=False).flatten()
        train_indices = torch.nonzero(fold_ids != fold, as_tuple=False).flatten()
        if test_indices.numel() > 0:
            if train_indices.numel() == 0:
                raise ValueError(f"fold {fold} has no calibration queries")
            fold_splits.append((train_indices, test_indices))

    for shard in shards:
        path = profile_dir / Path(shard["layer_k_paths"][str(layer)]).name
        array = np.load(path, mmap_mode="r")
        shard_start = int(shard["block_start"])
        for offset in range(0, int(array.shape[0]), block_chunk):
            count = min(block_chunk, int(array.shape[0]) - offset)
            key_array = np.take(
                array[offset : offset + count],
                np.asarray(stored_kv_indices, dtype=np.int64),
                axis=2,
            )
            keys = torch.from_numpy(np.array(key_array, copy=True)).to(
                device=device, non_blocking=True
            )
            keys = keys[:, exclude_block_prefix_tokens:]
            block_ids = torch.arange(
                shard_start + offset,
                shard_start + offset + count,
                device=device,
                dtype=torch.long,
            )
            chunk_scores = torch.empty(
                (query_count, selected_count, count),
                dtype=torch.float32,
                device=device,
            )
            for query_start in range(0, query_count, query_batch):
                query_end = min(query_count, query_start + query_batch)
                query_part = selected_queries[query_start:query_end]
                similarities = torch.einsum(
                    "qihd,bthd->qihbt", query_part, keys
                )
                per_query_token = similarities.amax(dim=-1).float()
                mask = query_mask[query_start:query_end, :, None, None]
                valid = (
                    query_mask[query_start:query_end]
                    .sum(dim=1)
                    .clamp_min(1)
                    .float()
                )
                scores = (per_query_token * mask).sum(dim=1) / valid[
                    :, None, None
                ]
                chunk_scores[query_start:query_end] = scores

            raw_scores, raw_ids = rankings["raw"]
            rankings["raw"] = deterministic_update_topk(
                raw_scores, raw_ids, chunk_scores, block_ids, top_per_head
            )
            centered_scores = torch.empty_like(chunk_scores)
            zscore_scores = torch.empty_like(chunk_scores)
            for train_indices, test_indices in fold_splits:
                train_scores = chunk_scores.index_select(0, train_indices)
                train_mean = train_scores.mean(dim=0)
                train_var = (
                    train_scores.square().mean(dim=0) - train_mean.square()
                ).clamp_min(0)
                test_scores = chunk_scores.index_select(0, test_indices)
                centered = test_scores - train_mean.unsqueeze(0)
                centered_scores[test_indices] = centered
                zscore_scores[test_indices] = centered / train_var.sqrt().clamp_min(
                    std_epsilon
                ).unsqueeze(0)
            for method, method_scores in (
                ("centered", centered_scores),
                ("zscore", zscore_scores),
            ):
                best_scores, best_ids = rankings[method]
                rankings[method] = deterministic_update_topk(
                    best_scores,
                    best_ids,
                    method_scores,
                    block_ids,
                    top_per_head,
                )
    return rankings


def rrf_ranking(
    ids: np.ndarray, target_blocks: int, num_blocks: int, constant: float = 60.0
) -> np.ndarray:
    depth = ids.shape[1]
    weights = np.tile(
        1.0 / (constant + np.arange(1, depth + 1, dtype=np.float64)),
        ids.shape[0],
    )
    flat = ids.reshape(-1)
    scores = np.bincount(flat, weights=weights, minlength=num_blocks)
    nominated = np.flatnonzero(scores)
    return nominated[
        np.lexsort((nominated, -scores[nominated]))[:target_blocks]
    ]


def main() -> None:
    args = parse_args()
    rank, world_size, _local_rank, device = setup_distributed()
    corpus_dir = Path(args.corpus_dir)
    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    profile_summary = json.loads(
        (profile_dir / "summary.json").read_text(encoding="utf-8")
    )
    packed_kv_heads_by_layer = {
        int(layer): [int(item) for item in heads]
        for layer, heads in profile_summary.get(
            "selected_kv_heads_by_layer", {}
        ).items()
    }
    layers = [int(item) for item in profile_summary["layers"]]
    num_query_heads = int(profile_summary["num_query_heads"])
    num_kv_heads = int(profile_summary["num_kv_heads"])
    num_blocks = int(profile_summary["num_blocks"])
    selected_by_fold = read_selection(
        Path(args.selection_csv), args.gate_feature, args.heads_per_fold
    )
    union_flat_heads = sorted(
        {head for heads in selected_by_fold.values() for head in heads}
    )
    flat_to_output = {head: index for index, head in enumerate(union_flat_heads)}
    selected_by_layer: dict[int, list[int]] = {}
    for flat_head in union_flat_heads:
        layer_index, query_head = divmod(flat_head, num_query_heads)
        selected_by_layer.setdefault(layer_index, []).append(query_head)

    query_payload = torch.load(
        Path(args.query_profiles), map_location="cpu", weights_only=False
    )
    query_vectors = query_payload["svd_q"]
    query_mask = query_payload["mask"]
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    query_count = int(query_vectors.shape[0])
    if len(queries) != query_count:
        raise ValueError("query profile and query metadata counts differ")
    with np.load(Path(args.full_reference_npz)) as reference:
        fold_ids_np = reference["fold_ids"].astype(np.int64)
        reference_ids = reference["block_ids"]
        reference_scores = reference["scores"]
        reference_layers = reference["layers"].astype(np.int64)
    if not np.array_equal(reference_layers, np.asarray(layers, dtype=np.int64)):
        raise ValueError("reference layers differ from profile layers")
    if set(int(item) for item in np.unique(fold_ids_np)) != set(selected_by_fold):
        raise ValueError("selection folds differ from reference folds")
    fold_ids = torch.from_numpy(fold_ids_np).to(device=device)
    local_shards = [
        shard
        for shard_index, shard in enumerate(profile_summary["shards"])
        if shard_index % world_size == rank
    ]
    sparse_scores = {
        method: np.empty(
            (query_count, len(union_flat_heads), args.top_per_head),
            dtype=np.float32,
        )
        if rank == 0
        else None
        for method in METHODS
    }
    sparse_ids = {
        method: np.empty(
            (query_count, len(union_flat_heads), args.top_per_head),
            dtype=np.int32,
        )
        if rank == 0
        else None
        for method in METHODS
    }

    started = time.perf_counter()
    layer_rows: list[dict[str, Any]] = []
    layer_mask = query_mask.to(device=device, non_blocking=True)
    for layer_index in sorted(selected_by_layer):
        layer_started = time.perf_counter()
        layer = layers[layer_index]
        query_heads = sorted(selected_by_layer[layer_index])
        layer_queries = query_vectors[:, :, layer_index].to(
            device=device, non_blocking=True
        )
        local_rankings = score_selected_layer(
            layer=layer,
            shards=local_shards,
            profile_dir=profile_dir,
            queries=layer_queries,
            query_mask=layer_mask,
            fold_ids=fold_ids,
            selected_query_heads=query_heads,
            packed_kv_heads=packed_kv_heads_by_layer.get(layer),
            original_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            top_per_head=args.top_per_head,
            query_batch=args.query_batch,
            block_chunk=args.block_chunk,
            exclude_block_prefix_tokens=args.exclude_block_prefix_tokens,
            std_epsilon=args.std_epsilon,
            device=device,
        )
        output_indices = [
            flat_to_output[layer_index * num_query_heads + query_head]
            for query_head in query_heads
        ]
        for method in METHODS:
            merged_scores, merged_ids = deterministic_merge_distributed_topk(
                *local_rankings[method], args.top_per_head, world_size
            )
            if rank == 0:
                assert sparse_scores[method] is not None
                assert sparse_ids[method] is not None
                sparse_scores[method][:, output_indices] = merged_scores.cpu().numpy()
                sparse_ids[method][:, output_indices] = (
                    merged_ids.cpu().numpy().astype(np.int32)
                )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - layer_started
        if rank == 0:
            layer_rows.append(
                {
                    "layer": layer,
                    "layer_index": layer_index,
                    "selected_query_heads": query_heads,
                    "selected_head_count": len(query_heads),
                    "seconds": elapsed,
                }
            )
            print(json.dumps(layer_rows[-1]), flush=True)
    total_seconds = time.perf_counter() - started

    if rank == 0:
        for method in METHODS:
            assert sparse_scores[method] is not None
            assert sparse_ids[method] is not None
            np.savez_compressed(
                output_dir / f"{method}_selected_topk.npz",
                scores=sparse_scores[method],
                block_ids=sparse_ids[method],
                flat_heads=np.asarray(union_flat_heads, dtype=np.int32),
                fold_ids=fold_ids_np.astype(np.int32),
            )

        assert sparse_scores["zscore"] is not None
        assert sparse_ids["zscore"] is not None
        max_score_error = 0.0
        id_mismatches = 0
        compared_slots = 0
        for sparse_index, flat_head in enumerate(union_flat_heads):
            layer_index, query_head = divmod(flat_head, num_query_heads)
            expected_scores = reference_scores[:, layer_index, query_head]
            expected_ids = reference_ids[:, layer_index, query_head]
            max_score_error = max(
                max_score_error,
                float(
                    np.max(
                        np.abs(
                            sparse_scores["zscore"][:, sparse_index].astype(np.float64)
                            - expected_scores.astype(np.float64)
                        )
                    )
                ),
            )
            id_mismatches += int(
                np.sum(sparse_ids["zscore"][:, sparse_index] != expected_ids)
            )
            compared_slots += int(expected_ids.size)

        sparse_hits = np.zeros(query_count, dtype=bool)
        reference_hits = np.zeros(query_count, dtype=bool)
        for query_index, query in enumerate(queries):
            fold = int(fold_ids_np[query_index])
            heads = selected_by_fold[fold]
            sparse_positions = [flat_to_output[head] for head in heads]
            sparse_ranking = rrf_ranking(
                sparse_ids["zscore"][query_index, sparse_positions],
                args.target_blocks,
                num_blocks,
            )
            reference_head_ids = np.stack(
                [
                    reference_ids[
                        query_index,
                        head // num_query_heads,
                        head % num_query_heads,
                    ]
                    for head in heads
                ]
            )
            reference_ranking = rrf_ranking(
                reference_head_ids, args.target_blocks, num_blocks
            )
            gold = np.asarray(query.get("gold_block_ids", []), dtype=np.int64)
            sparse_hits[query_index] = np.isin(gold, sparse_ranking).any()
            reference_hits[query_index] = np.isin(gold, reference_ranking).any()

        selected_kv_channels = {
            (
                flat_head // num_query_heads,
                (flat_head % num_query_heads)
                // (num_query_heads // num_kv_heads),
            )
            for flat_head in union_flat_heads
        }
        summary = {
            "experiment": "real_selected_head_lodo_debiased_scan",
            "selection_uses_gold": False,
            "selection_uses_heldout_queries": False,
            "gold_used_only_for_final_recall": True,
            "queries": query_count,
            "blocks": num_blocks,
            "full_query_head_channels": len(layers) * num_query_heads,
            "selected_union_query_head_channels": len(union_flat_heads),
            "query_head_channel_reduction": (
                len(layers) * num_query_heads / len(union_flat_heads)
            ),
            "full_layer_kv_channels": len(layers) * num_kv_heads,
            "selected_union_layer_kv_channels": len(selected_kv_channels),
            "selected_layers": len(selected_by_layer),
            "heads_per_fold": args.heads_per_fold,
            "union_flat_heads": union_flat_heads,
            "world_size": world_size,
            "profile_layout": (
                "packed_selected_kv"
                if packed_kv_heads_by_layer
                else "original_interleaved_kv"
            ),
            "profile_dir": str(profile_dir),
            "profile_bytes": int(
                profile_summary.get(
                    "packed_profile_bytes",
                    profile_summary.get("profile_bytes", 0),
                )
            ),
            "total_seconds": total_seconds,
            "layer_timings": layer_rows,
            "reference_alignment": {
                "zscore_max_abs_error": max_score_error,
                "id_mismatch_slots": id_mismatches,
                "id_mismatch_fraction": id_mismatches / compared_slots,
                "sparse_rrf39_recall": float(sparse_hits.mean()),
                "reference_rrf39_recall": float(reference_hits.mean()),
                "rrf39_hit_disagreements": int(
                    np.sum(sparse_hits != reference_hits)
                ),
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
