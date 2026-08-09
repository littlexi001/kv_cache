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

from profile_real_qk import read_jsonl, setup_distributed


METHODS = ("raw", "centered", "zscore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether query-invariant per-head block priors can be separated "
            "from real QK relevance scores without using gold labels."
        )
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--query_profiles", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_per_head", type=int, default=16)
    parser.add_argument("--query_batch", type=int, default=8)
    parser.add_argument("--block_chunk", type=int, default=64)
    parser.add_argument("--exclude_block_prefix_tokens", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--fold_strategy",
        choices=("dataset_stratified", "dataset_leave_one_out"),
        default="dataset_stratified",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--std_epsilon", type=float, default=1e-4)
    parser.add_argument("--max_queries", type=int, default=0)
    return parser.parse_args()


def stratified_fold_ids(
    queries: list[dict[str, Any]], folds: int, seed: int
) -> np.ndarray:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    rng = np.random.default_rng(seed)
    fold_ids = np.full(len(queries), -1, dtype=np.int64)
    datasets = sorted({str(query["dataset"]) for query in queries})
    for dataset in datasets:
        indices = np.asarray(
            [
                index
                for index, query in enumerate(queries)
                if str(query["dataset"]) == dataset
            ],
            dtype=np.int64,
        )
        rng.shuffle(indices)
        fold_ids[indices] = np.arange(len(indices), dtype=np.int64) % folds
    if np.any(fold_ids < 0):
        raise RuntimeError("failed to assign every query to a fold")
    return fold_ids


def dataset_leave_one_out_fold_ids(
    queries: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[int, str]]:
    datasets = sorted({str(query["dataset"]) for query in queries})
    dataset_to_fold = {dataset: fold for fold, dataset in enumerate(datasets)}
    fold_ids = np.asarray(
        [dataset_to_fold[str(query["dataset"])] for query in queries],
        dtype=np.int64,
    )
    return fold_ids, {fold: dataset for dataset, fold in dataset_to_fold.items()}


def empty_topk(
    query_count: int,
    num_query_heads: int,
    top_per_head: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.full(
        (query_count, num_query_heads, top_per_head),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    ids = torch.full(
        (query_count, num_query_heads, top_per_head),
        -1,
        dtype=torch.long,
        device=device,
    )
    return scores, ids


def deterministic_select_topk(
    scores: torch.Tensor, ids: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort by score descending, then block ID ascending for exact ties."""
    keep = min(k, int(scores.shape[2]))
    id_order = torch.argsort(ids, dim=2, descending=False, stable=True)
    ids_by_id = torch.gather(ids, dim=2, index=id_order)
    scores_by_id = torch.gather(scores, dim=2, index=id_order)
    score_order = torch.argsort(
        scores_by_id, dim=2, descending=True, stable=True
    )[:, :, :keep]
    return (
        torch.gather(scores_by_id, dim=2, index=score_order),
        torch.gather(ids_by_id, dim=2, index=score_order),
    )


def deterministic_update_topk(
    current_scores: torch.Tensor,
    current_ids: torch.Tensor,
    new_scores: torch.Tensor,
    new_ids: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded_ids = new_ids[None, None, :].expand(
        new_scores.shape[0], new_scores.shape[1], new_scores.shape[2]
    )
    return deterministic_select_topk(
        torch.cat([current_scores, new_scores], dim=2),
        torch.cat([current_ids, expanded_ids], dim=2),
        k,
    )


def deterministic_merge_distributed_topk(
    local_scores: torch.Tensor,
    local_ids: torch.Tensor,
    top_per_head: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if world_size == 1:
        return local_scores, local_ids
    score_parts = [torch.empty_like(local_scores) for _ in range(world_size)]
    id_parts = [torch.empty_like(local_ids) for _ in range(world_size)]
    dist.all_gather(score_parts, local_scores.contiguous())
    dist.all_gather(id_parts, local_ids.contiguous())
    return deterministic_select_topk(
        torch.cat(score_parts, dim=2),
        torch.cat(id_parts, dim=2),
        top_per_head,
    )


def score_layer_shards_debiased(
    *,
    layer: int,
    shards: list[dict[str, Any]],
    profile_dir: Path,
    queries: torch.Tensor,
    query_mask: torch.Tensor,
    fold_ids: torch.Tensor,
    num_kv_heads: int,
    top_per_head: int,
    query_batch: int,
    block_chunk: int,
    exclude_block_prefix_tokens: int,
    std_epsilon: float,
    device: torch.device,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    query_count, query_tokens, num_query_heads, rank_dim = queries.shape
    repeat_groups = num_query_heads // num_kv_heads
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")

    rankings = {
        method: empty_topk(query_count, num_query_heads, top_per_head, device)
        for method in METHODS
    }
    # Per head: sum(E_q[s]), sum(E_q[s^2]), sum(E_q[s]^2),
    # sum(Var_q[s]), and number of blocks. Weighting by block keeps values stable.
    diagnostics = torch.zeros(
        (num_query_heads, 5), dtype=torch.float32, device=device
    )
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
            key_array = np.array(array[offset : offset + count], copy=True)
            keys = torch.from_numpy(key_array).to(device=device, non_blocking=True)
            keys = keys[:, exclude_block_prefix_tokens:]
            block_ids = torch.arange(
                shard_start + offset,
                shard_start + offset + count,
                device=device,
                dtype=torch.long,
            )
            chunk_scores = torch.empty(
                (query_count, num_query_heads, count),
                dtype=torch.float32,
                device=device,
            )

            for query_start in range(0, query_count, query_batch):
                query_end = min(query_count, query_start + query_batch)
                query_part = queries[query_start:query_end]
                batch = query_end - query_start
                grouped_queries = query_part.reshape(
                    batch,
                    query_tokens,
                    num_kv_heads,
                    repeat_groups,
                    rank_dim,
                )
                similarities = torch.einsum(
                    "qigpd,btgd->qigpbt", grouped_queries, keys
                )
                per_query_token = similarities.amax(dim=-1).float()
                mask = query_mask[query_start:query_end, :, None, None, None]
                valid = (
                    query_mask[query_start:query_end]
                    .sum(dim=1)
                    .clamp_min(1)
                    .float()
                )
                scores = (per_query_token * mask).sum(dim=1) / valid[
                    :, None, None, None
                ]
                scores = scores.reshape(
                    batch, num_query_heads, count
                )
                chunk_scores[query_start:query_end] = scores

            raw_scores, raw_ids = rankings["raw"]
            rankings["raw"] = deterministic_update_topk(
                raw_scores, raw_ids, chunk_scores, block_ids, top_per_head
            )

            block_means = chunk_scores.mean(dim=0)
            block_second_moments = chunk_scores.square().mean(dim=0)
            block_variances = (block_second_moments - block_means.square()).clamp_min(0)
            diagnostics[:, 0] += block_means.sum(dim=1)
            diagnostics[:, 1] += block_second_moments.sum(dim=1)
            diagnostics[:, 2] += block_means.square().sum(dim=1)
            diagnostics[:, 3] += block_variances.sum(dim=1)
            diagnostics[:, 4] += count

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

    return rankings, diagnostics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.top_per_head <= 0 or args.query_batch <= 0 or args.block_chunk <= 0:
        raise ValueError("top_per_head, query_batch, and block_chunk must be positive")
    if args.std_epsilon <= 0:
        raise ValueError("std_epsilon must be positive")

    rank, world_size, _local_rank, device = setup_distributed()
    corpus_dir = Path(args.corpus_dir)
    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        for method in METHODS:
            (output_dir / method).mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    profile_summary = json.loads(
        (profile_dir / "summary.json").read_text(encoding="utf-8")
    )
    if profile_summary.get("contains_synthetic_vectors"):
        raise ValueError("prior decomposition requires real Q/K profiles")
    layers = [int(item) for item in profile_summary["layers"]]
    num_query_heads = int(profile_summary["num_query_heads"])
    num_kv_heads = int(profile_summary["num_kv_heads"])
    block_count = int(profile_summary["num_blocks"])

    query_payload = torch.load(
        Path(args.query_profiles), map_location="cpu", weights_only=False
    )
    query_vectors = query_payload["svd_q"]
    query_mask = query_payload["mask"]
    query_count = int(query_vectors.shape[0])
    if args.max_queries > 0:
        query_count = min(query_count, args.max_queries)
        query_vectors = query_vectors[:query_count]
        query_mask = query_mask[:query_count]
    queries = read_jsonl(corpus_dir / "queries.jsonl")[:query_count]
    if len(queries) != query_count:
        raise ValueError("query profile and query metadata counts do not match")
    if args.fold_strategy == "dataset_leave_one_out":
        fold_ids_np, fold_names = dataset_leave_one_out_fold_ids(queries)
    else:
        fold_ids_np = stratified_fold_ids(queries, args.folds, args.seed)
        fold_names = {
            fold: f"stratified_fold_{fold}"
            for fold in sorted(int(item) for item in np.unique(fold_ids_np))
        }
    effective_folds = len(fold_names)
    fold_ids = torch.from_numpy(fold_ids_np).to(device=device)

    profile_shards = list(profile_summary["shards"])
    local_shards = [
        shard
        for shard_index, shard in enumerate(profile_shards)
        if shard_index % world_size == rank
    ]
    all_scores = {
        method: np.empty(
            (query_count, len(layers), num_query_heads, args.top_per_head),
            dtype=np.float32,
        )
        if rank == 0
        else None
        for method in METHODS
    }
    all_ids = {
        method: np.empty(
            (query_count, len(layers), num_query_heads, args.top_per_head),
            dtype=np.int32,
        )
        if rank == 0
        else None
        for method in METHODS
    }

    started = time.perf_counter()
    layer_seconds: list[float] = []
    diagnostic_rows: list[dict[str, Any]] = []
    layer_mask = query_mask.to(device=device, non_blocking=True)
    for layer_index, layer in enumerate(layers):
        layer_started = time.perf_counter()
        layer_queries = query_vectors[:, :, layer_index].to(
            device=device, non_blocking=True
        )
        local_rankings, diagnostics = score_layer_shards_debiased(
            layer=layer,
            shards=local_shards,
            profile_dir=profile_dir,
            queries=layer_queries,
            query_mask=layer_mask,
            fold_ids=fold_ids,
            num_kv_heads=num_kv_heads,
            top_per_head=args.top_per_head,
            query_batch=args.query_batch,
            block_chunk=args.block_chunk,
            exclude_block_prefix_tokens=args.exclude_block_prefix_tokens,
            std_epsilon=args.std_epsilon,
            device=device,
        )
        if world_size > 1:
            dist.all_reduce(diagnostics, op=dist.ReduceOp.SUM)

        for method in METHODS:
            merged_scores, merged_ids = deterministic_merge_distributed_topk(
                *local_rankings[method], args.top_per_head, world_size
            )
            if rank == 0:
                assert all_scores[method] is not None and all_ids[method] is not None
                all_scores[method][:, layer_index] = merged_scores.cpu().numpy()
                all_ids[method][:, layer_index] = (
                    merged_ids.cpu().numpy().astype(np.int32)
                )

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - layer_started
        layer_seconds.append(elapsed)
        if rank == 0:
            values = diagnostics.cpu().numpy().astype(np.float64)
            for head in range(num_query_heads):
                observed_blocks = values[head, 4]
                score_mean = values[head, 0] / observed_blocks
                total_variance = max(
                    0.0, values[head, 1] / observed_blocks - score_mean**2
                )
                between_block_variance = max(
                    0.0, values[head, 2] / observed_blocks - score_mean**2
                )
                within_block_variance = max(
                    0.0, values[head, 3] / observed_blocks
                )
                diagnostic_rows.append(
                    {
                        "layer": layer,
                        "query_head": head,
                        "blocks": int(round(observed_blocks)),
                        "score_mean": score_mean,
                        "total_variance": total_variance,
                        "between_block_prior_variance": between_block_variance,
                        "within_block_query_variance": within_block_variance,
                        "prior_variance_fraction": (
                            between_block_variance / total_variance
                            if total_variance > 0
                            else 0.0
                        ),
                    }
                )
            print(
                json.dumps(
                    {
                        "layer": layer,
                        "layer_index": layer_index,
                        "layers": len(layers),
                        "seconds": elapsed,
                    }
                ),
                flush=True,
            )

    total_seconds = time.perf_counter() - started
    if rank == 0:
        for method in METHODS:
            assert all_scores[method] is not None and all_ids[method] is not None
            np.savez_compressed(
                output_dir / method / "per_head_topk.npz",
                scores=all_scores[method],
                block_ids=all_ids[method],
                layers=np.asarray(layers, dtype=np.int32),
                fold_ids=fold_ids_np.astype(np.int32),
            )
        write_csv(
            output_dir / "fold_assignments.csv",
            [
                {
                    "query_index": index,
                    "dataset": str(query["dataset"]),
                    "fold": int(fold_ids_np[index]),
                    "fold_name": fold_names[int(fold_ids_np[index])],
                }
                for index, query in enumerate(queries)
            ],
        )
        write_csv(output_dir / "variance_diagnostics.csv", diagnostic_rows)
        summary = {
            "experiment": "cross_fitted_query_invariant_block_prior_debiasing",
            "contains_synthetic_vectors": False,
            "selection_uses_gold_for_rankings": False,
            "prior_calibration_uses_gold": False,
            "prior_calibration_uses_held_out_queries": False,
            "score_model": "s_lh(q,b) = mu_lh(b) + delta_lh(q,b)",
            "tie_breaking": "score descending, then block_id ascending",
            "methods": {
                "raw": "s_lh(q,b)",
                "centered": "s_lh(q,b) - mean_train_fold[s_lh(q,b)]",
                "zscore": (
                    "(s_lh(q,b) - mean_train_fold[s_lh(q,b)]) / "
                    "max(std_train_fold[s_lh(q,b)], epsilon)"
                ),
            },
            "folds": effective_folds,
            "fold_strategy": args.fold_strategy,
            "fold_names": fold_names,
            "seed": args.seed,
            "queries": query_count,
            "blocks": block_count,
            "layers": layers,
            "query_heads": num_query_heads,
            "top_per_head": args.top_per_head,
            "std_epsilon": args.std_epsilon,
            "world_size": world_size,
            "total_seconds": total_seconds,
            "mean_layer_seconds": float(np.mean(layer_seconds)),
            "max_layer_seconds": float(np.max(layer_seconds)),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
