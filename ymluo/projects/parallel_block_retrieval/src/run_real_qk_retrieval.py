from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed block retrieval over real Qwen Q/K and a true K-SVD index."
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_tokens", type=int, default=10_000)
    parser.add_argument("--candidate_fraction", type=float, default=0.02)
    parser.add_argument("--qabs_dims", type=int, default=8)
    parser.add_argument("--methods", default="full128,svd32,svd64,svd32_rerank,qabs8")
    parser.add_argument("--query_batch", type=int, default=16)
    parser.add_argument("--block_chunk", type=int, default=256)
    parser.add_argument("--exclude_block_prefix_tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Real retrieval benchmark requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, device


def sync(device: torch.device, world_size: int) -> None:
    torch.cuda.synchronize(device)
    if world_size > 1:
        dist.barrier()
    torch.cuda.synchronize(device)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_index(
    profile_dir: Path,
    profile_summary: dict[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    shards = profile_summary["shards"]
    assigned_indices = [index for index in range(len(shards)) if index % world_size == rank]
    if not assigned_indices:
        raise RuntimeError(f"Runtime rank {rank} received no profile shards")
    assigned = [shards[index] for index in assigned_indices]
    local_blocks = sum(int(shard["local_blocks"]) for shard in assigned)
    block_tokens = int(profile_summary["block_tokens"])
    profile_count = len(profile_summary["pair_specs"])
    head_dim = int(profile_summary["head_dim"])
    svd_rank = int(profile_summary["svd_rank"])

    raw = torch.empty(
        local_blocks,
        block_tokens,
        profile_count,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    svd = torch.empty(
        local_blocks,
        block_tokens,
        profile_count,
        svd_rank,
        dtype=torch.float16,
        device=device,
    )
    block_id_parts: list[torch.Tensor] = []
    cursor = 0
    for shard in assigned:
        raw_path = profile_dir / Path(shard["raw_k_path"]).name
        svd_path = profile_dir / Path(shard["svd_k_path"]).name
        raw_array = np.load(raw_path, mmap_mode="c")
        svd_array = np.load(svd_path, mmap_mode="c")
        count = int(raw_array.shape[0])
        raw[cursor : cursor + count].copy_(torch.from_numpy(raw_array))
        svd[cursor : cursor + count].copy_(torch.from_numpy(svd_array))
        block_id_parts.append(
            torch.arange(
                int(shard["block_start"]),
                int(shard["block_end"]),
                dtype=torch.long,
                device=device,
            )
        )
        cursor += count
    block_ids = torch.cat(block_id_parts)
    if not bool(torch.all(block_ids[1:] > block_ids[:-1]).item()):
        order = torch.argsort(block_ids)
        block_ids = block_ids.index_select(0, order)
        raw = raw.index_select(0, order)
        svd = svd.index_select(0, order)
    return raw, svd, block_ids, assigned_indices


def score_dense_blocks(
    keys: torch.Tensor,
    queries: torch.Tensor,
    query_mask: torch.Tensor,
    *,
    query_batch: int,
    block_chunk: int,
    exclude_block_prefix_tokens: int,
    scale: float,
) -> torch.Tensor:
    query_count, _, profile_count, _ = queries.shape
    block_count, block_tokens, _, dim = keys.shape
    output = torch.empty(query_count, block_count, dtype=torch.float32, device=keys.device)
    for query_start in range(0, query_count, query_batch):
        query_end = min(query_count, query_start + query_batch)
        q_batch = queries[query_start:query_end]
        mask_batch = query_mask[query_start:query_end]
        batch_count = query_end - query_start
        for block_start in range(0, block_count, block_chunk):
            block_end = min(block_count, block_start + block_chunk)
            chunk = keys[block_start:block_end, exclude_block_prefix_tokens:]
            chunk_count = block_end - block_start
            scored_tokens = chunk.shape[1]
            best = torch.full(
                (batch_count, chunk_count),
                -torch.inf,
                dtype=torch.float32,
                device=keys.device,
            )
            for profile in range(profile_count):
                query_part = q_batch[:, :, profile, :]
                key_part = chunk[:, :, profile, :].reshape(chunk_count * scored_tokens, dim)
                scores = torch.matmul(query_part, key_part.transpose(0, 1))
                scores = scores.masked_fill(~mask_batch[:, :, None], -torch.inf)
                profile_best = scores.reshape(
                    batch_count, query_part.shape[1], chunk_count, scored_tokens
                ).amax(dim=(1, 3))
                best = torch.maximum(best, profile_best.float())
            output[query_start:query_end, block_start:block_end] = best * scale
    return output


def score_colbert_blocks(
    keys: torch.Tensor,
    queries: torch.Tensor,
    query_mask: torch.Tensor,
    *,
    query_batch: int,
    block_chunk: int,
    exclude_block_prefix_tokens: int,
) -> torch.Tensor:
    query_count, _, profile_count, _ = queries.shape
    block_count, _, _, dim = keys.shape
    query_norms = queries.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    normalized_queries = queries / query_norms
    output = torch.empty(query_count, block_count, dtype=torch.float32, device=keys.device)
    for query_start in range(0, query_count, query_batch):
        query_end = min(query_count, query_start + query_batch)
        q_batch = normalized_queries[query_start:query_end]
        mask_batch = query_mask[query_start:query_end]
        valid_counts = mask_batch.sum(dim=1).clamp_min(1).to(torch.float32)
        batch_count = query_end - query_start
        for block_start in range(0, block_count, block_chunk):
            block_end = min(block_count, block_start + block_chunk)
            chunk = keys[block_start:block_end, exclude_block_prefix_tokens:]
            chunk = chunk / chunk.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            chunk_count = block_end - block_start
            scored_tokens = chunk.shape[1]
            combined = torch.zeros(
                (batch_count, chunk_count), dtype=torch.float32, device=keys.device
            )
            for profile in range(profile_count):
                query_part = q_batch[:, :, profile, :]
                key_part = chunk[:, :, profile, :].reshape(chunk_count * scored_tokens, dim)
                scores = torch.matmul(query_part, key_part.transpose(0, 1))
                per_query = scores.reshape(
                    batch_count, query_part.shape[1], chunk_count, scored_tokens
                ).amax(dim=3)
                per_query = per_query.masked_fill(~mask_batch[:, :, None], 0.0)
                combined += per_query.float().sum(dim=1) / valid_counts[:, None]
            output[query_start:query_end, block_start:block_end] = combined / profile_count
    return output


def score_qabs_blocks(
    keys: torch.Tensor,
    queries: torch.Tensor,
    query_mask: torch.Tensor,
    *,
    dims: int,
    query_batch: int,
    block_chunk: int,
    exclude_block_prefix_tokens: int,
    scale: float,
) -> torch.Tensor:
    query_count, query_vectors, profile_count, head_dim = queries.shape
    block_count, block_tokens, _, _ = keys.shape
    dims = min(max(1, dims), head_dim)
    output = torch.empty(query_count, block_count, dtype=torch.float32, device=keys.device)
    indices = torch.topk(queries.abs(), k=dims, dim=-1, largest=True).indices
    selected_queries = torch.gather(queries, dim=-1, index=indices)
    for query_start in range(0, query_count, query_batch):
        query_end = min(query_count, query_start + query_batch)
        batch_count = query_end - query_start
        mask_batch = query_mask[query_start:query_end]
        for block_start in range(0, block_count, block_chunk):
            block_end = min(block_count, block_start + block_chunk)
            chunk = keys[block_start:block_end, exclude_block_prefix_tokens:]
            chunk_count = block_end - block_start
            scored_tokens = chunk.shape[1]
            best = torch.full(
                (batch_count, chunk_count),
                -torch.inf,
                dtype=torch.float32,
                device=keys.device,
            )
            for profile in range(profile_count):
                query_indices = indices[query_start:query_end, :, profile, :]
                query_values = selected_queries[query_start:query_end, :, profile, :]
                key_part = chunk[:, :, profile, :]
                expanded_keys = key_part[None, None, :, :, :].expand(
                    batch_count, query_vectors, chunk_count, scored_tokens, head_dim
                )
                gather_index = query_indices[:, :, None, None, :].expand(
                    batch_count, query_vectors, chunk_count, scored_tokens, dims
                )
                selected_keys = torch.gather(expanded_keys, dim=-1, index=gather_index)
                scores = (selected_keys * query_values[:, :, None, None, :]).sum(dim=-1)
                scores = scores.masked_fill(~mask_batch[:, :, None, None], -torch.inf)
                best = torch.maximum(best, scores.amax(dim=(1, 3)).float())
            output[query_start:query_end, block_start:block_end] = best * scale
    return output


def global_topk(
    local_scores: torch.Tensor,
    local_block_ids: torch.Tensor,
    k: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_k = min(k, local_scores.shape[1])
    adjusted_scores = local_scores.to(torch.float64) - local_block_ids[None, :].to(torch.float64) * 1.0e-12
    values, positions = torch.topk(adjusted_scores, k=local_k, dim=1, largest=True)
    ids = local_block_ids[positions]
    if world_size > 1:
        values_parts = [torch.empty_like(values) for _ in range(world_size)]
        ids_parts = [torch.empty_like(ids) for _ in range(world_size)]
        dist.all_gather(values_parts, values.contiguous())
        dist.all_gather(ids_parts, ids.contiguous())
        values = torch.cat(values_parts, dim=1)
        ids = torch.cat(ids_parts, dim=1)
    final_values, final_positions = torch.topk(values, k=min(k, values.shape[1]), dim=1, largest=True)
    final_ids = torch.gather(ids, dim=1, index=final_positions)
    return final_values, final_ids


def candidate_exact_scores(
    raw_keys: torch.Tensor,
    raw_queries: torch.Tensor,
    query_mask: torch.Tensor,
    candidate_ids: torch.Tensor,
    local_block_ids: torch.Tensor,
    exclude_block_prefix_tokens: int,
    scale: float,
    world_size: int,
) -> torch.Tensor:
    query_count, candidate_count = candidate_ids.shape
    profile_count = raw_queries.shape[2]
    scores_out = torch.full(
        (query_count, candidate_count),
        -torch.inf,
        dtype=torch.float32,
        device=raw_keys.device,
    )
    for query_index in range(query_count):
        ids = candidate_ids[query_index]
        positions = torch.searchsorted(local_block_ids, ids)
        valid = positions < local_block_ids.numel()
        safe_positions = positions.clamp(max=max(0, local_block_ids.numel() - 1))
        valid &= local_block_ids[safe_positions] == ids
        if not bool(valid.any().item()):
            continue
        destination = valid.nonzero(as_tuple=False).flatten()
        owned_positions = safe_positions.index_select(0, destination)
        owned_keys = raw_keys.index_select(0, owned_positions)[:, exclude_block_prefix_tokens:]
        best = torch.full(
            (owned_keys.shape[0],),
            -torch.inf,
            dtype=torch.float32,
            device=raw_keys.device,
        )
        valid_query_positions = query_mask[query_index].nonzero(as_tuple=False).flatten()
        for profile in range(profile_count):
            q = raw_queries[query_index, valid_query_positions, profile, :]
            k = owned_keys[:, :, profile, :]
            profile_scores = torch.einsum("sd,mtd->smt", q, k).amax(dim=(0, 2)).float()
            best = torch.maximum(best, profile_scores)
        scores_out[query_index, destination] = best * scale
    if world_size > 1:
        dist.all_reduce(scores_out, op=dist.ReduceOp.MAX)
    return scores_out


def candidate_colbert_scores(
    keys: torch.Tensor,
    queries: torch.Tensor,
    query_mask: torch.Tensor,
    candidate_ids: torch.Tensor,
    local_block_ids: torch.Tensor,
    exclude_block_prefix_tokens: int,
    world_size: int,
) -> torch.Tensor:
    query_count, candidate_count = candidate_ids.shape
    profile_count = queries.shape[2]
    scores_out = torch.full(
        (query_count, candidate_count),
        -torch.inf,
        dtype=torch.float32,
        device=keys.device,
    )
    for query_index in range(query_count):
        ids = candidate_ids[query_index]
        positions = torch.searchsorted(local_block_ids, ids)
        valid = positions < local_block_ids.numel()
        safe_positions = positions.clamp(max=max(0, local_block_ids.numel() - 1))
        valid &= local_block_ids[safe_positions] == ids
        if not bool(valid.any().item()):
            continue
        destination = valid.nonzero(as_tuple=False).flatten()
        owned_positions = safe_positions.index_select(0, destination)
        owned_keys = keys.index_select(0, owned_positions)[:, exclude_block_prefix_tokens:]
        owned_keys = owned_keys / owned_keys.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        valid_query_positions = query_mask[query_index].nonzero(as_tuple=False).flatten()
        combined = torch.zeros(
            owned_keys.shape[0], dtype=torch.float32, device=keys.device
        )
        for profile in range(profile_count):
            query_part = queries[query_index, valid_query_positions, profile, :]
            query_part = query_part / query_part.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            key_part = owned_keys[:, :, profile, :]
            profile_scores = torch.einsum("sd,mtd->smt", query_part, key_part)
            combined += profile_scores.amax(dim=2).float().mean(dim=0)
        scores_out[query_index, destination] = combined / profile_count
    if world_size > 1:
        dist.all_reduce(scores_out, op=dist.ReduceOp.MAX)
    return scores_out


def distributed_lookup_scores(
    local_scores: torch.Tensor,
    local_block_ids: torch.Tensor,
    requested_ids: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    query_count, requested_count = requested_ids.shape
    output = torch.full(
        (query_count, requested_count),
        -torch.inf,
        dtype=torch.float32,
        device=local_scores.device,
    )
    for query_index in range(query_count):
        ids = requested_ids[query_index]
        positions = torch.searchsorted(local_block_ids, ids)
        valid = positions < local_block_ids.numel()
        safe = positions.clamp(max=max(0, local_block_ids.numel() - 1))
        valid &= local_block_ids[safe] == ids
        if bool(valid.any().item()):
            destination = valid.nonzero(as_tuple=False).flatten()
            output[query_index, destination] = local_scores[query_index, safe[destination]]
    if world_size > 1:
        dist.all_reduce(output, op=dist.ReduceOp.MAX)
    return output


def exact_mass_denominator(local_exact_scores: torch.Tensor, world_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    local_max = local_exact_scores.amax(dim=1)
    global_max = local_max.clone()
    if world_size > 1:
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
    local_sum = torch.exp(local_exact_scores - global_max[:, None]).sum(dim=1)
    global_sum = local_sum.clone()
    if world_size > 1:
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
    return global_max, global_sum


def evaluate_ids(
    *,
    method: str,
    selected_ids: torch.Tensor,
    oracle_ids: torch.Tensor,
    selected_exact_scores: torch.Tensor,
    global_exact_max: torch.Tensor,
    global_exact_sum: torch.Tensor,
    queries: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        selected = [int(item) for item in selected_ids[query_index].tolist()]
        oracle = set(int(item) for item in oracle_ids[query_index].tolist())
        gold = set(int(item) for item in query.get("gold_block_ids", []))
        oracle_overlap = len(oracle & set(selected)) / max(1, len(oracle))
        gold_ranks = [rank + 1 for rank, block_id in enumerate(selected) if block_id in gold]
        selected_mass = float(
            (
                torch.exp(selected_exact_scores[query_index] - global_exact_max[query_index]).sum()
                / global_exact_sum[query_index].clamp_min(1.0e-30)
            ).item()
        )
        rows.append(
            {
                "method": method,
                "query_id": int(query["query_id"]),
                "dataset": query["dataset"],
                "oracle_block_recall": oracle_overlap,
                "exact_block_mass_recall": selected_mass,
                "answer_block_recall": float(bool(gold_ranks)),
                "answer_block_mrr": 1.0 / min(gold_ranks) if gold_ranks else 0.0,
                "gold_block_count": len(gold),
                "selected_block_ids": selected,
            }
        )
    summary = {
        "method": method,
        "queries": len(rows),
        "oracle_block_recall": statistics.fmean(row["oracle_block_recall"] for row in rows),
        "exact_block_mass_recall": statistics.fmean(row["exact_block_mass_recall"] for row in rows),
        "answer_block_recall": statistics.fmean(row["answer_block_recall"] for row in rows),
        "answer_block_mrr": statistics.fmean(row["answer_block_mrr"] for row in rows),
    }
    return summary, rows


def aggregate_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["dataset"])].append(row)
    output: list[dict[str, Any]] = []
    for (method, dataset), group in sorted(grouped.items()):
        output.append(
            {
                "method": method,
                "dataset": dataset,
                "queries": len(group),
                "oracle_block_recall": statistics.fmean(row["oracle_block_recall"] for row in group),
                "exact_block_mass_recall": statistics.fmean(row["exact_block_mass_recall"] for row in group),
                "answer_block_recall": statistics.fmean(row["answer_block_recall"] for row in group),
                "answer_block_mrr": statistics.fmean(row["answer_block_mrr"] for row in group),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = setup_distributed()
    profile_dir = Path(args.profile_dir)
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    profile_summary = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    if profile_summary.get("contains_synthetic_vectors", True):
        raise RuntimeError("Refusing to run the real-QK benchmark on a synthetic profile")
    if profile_summary.get("profile_space") not in {
        "pre_rope_qk",
        "pre_rope_block_qk",
        "pre_rope_record_qk",
        "post_rope_record_qk",
    }:
        raise RuntimeError(f"Unsupported profile space: {profile_summary.get('profile_space')}")
    profile_shards = int(profile_summary["profile_world_size"])
    if world_size > profile_shards:
        raise ValueError(f"Runtime world_size={world_size} exceeds profile shards={profile_shards}")

    load_started = time.perf_counter()
    raw_keys, svd_keys, local_block_ids, assigned_shards = load_index(
        profile_dir, profile_summary, rank, world_size, device
    )
    query_payload = torch.load(profile_dir / "query_profiles.pt", map_location="cpu", weights_only=False)
    raw_queries = query_payload["raw_q"].to(device=device, dtype=torch.float16)
    svd_queries = query_payload["svd_q"].to(device=device, dtype=torch.float16)
    query_mask = query_payload["mask"].to(device=device)
    queries = query_payload["queries"]
    load_seconds = time.perf_counter() - load_started

    block_tokens = int(profile_summary["block_tokens"])
    if not 0 <= args.exclude_block_prefix_tokens < block_tokens:
        raise ValueError(
            f"exclude_block_prefix_tokens must be in [0, {block_tokens}); "
            f"got {args.exclude_block_prefix_tokens}"
        )
    num_blocks = int(profile_summary["num_blocks"])
    num_tokens = int(profile_summary["num_tokens"])
    head_dim = int(profile_summary["head_dim"])
    budget_blocks = max(1, args.target_tokens // block_tokens)
    candidate_blocks = max(budget_blocks, math.ceil(args.candidate_fraction * num_blocks))
    scale = 1.0 / math.sqrt(head_dim)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]

    sync(device, world_size)
    oracle_started = time.perf_counter()
    local_exact_scores = score_dense_blocks(
        raw_keys,
        raw_queries,
        query_mask,
        query_batch=args.query_batch,
        block_chunk=args.block_chunk,
        exclude_block_prefix_tokens=args.exclude_block_prefix_tokens,
        scale=scale,
    )
    _, oracle_ids = global_topk(local_exact_scores, local_block_ids, budget_blocks, world_size)
    global_exact_max, global_exact_sum = exact_mass_denominator(local_exact_scores, world_size)
    sync(device, world_size)
    oracle_precompute_seconds = time.perf_counter() - oracle_started

    def run_method(method: str) -> torch.Tensor:
        if method == "full128":
            scores = score_dense_blocks(
                raw_keys,
                raw_queries,
                query_mask,
                query_batch=args.query_batch,
                block_chunk=args.block_chunk,
                exclude_block_prefix_tokens=args.exclude_block_prefix_tokens,
                scale=scale,
            )
            return global_topk(scores, local_block_ids, budget_blocks, world_size)[1]
        svd_match = re.fullmatch(r"svd(\d+)(_rerank)?", method)
        if svd_match is not None:
            requested_rank = int(svd_match.group(1))
            if requested_rank > svd_keys.shape[-1]:
                raise ValueError(
                    f"Method {method} requires rank {requested_rank}, but the profile stores "
                    f"rank {svd_keys.shape[-1]}"
                )
            scores = score_dense_blocks(
                svd_keys[..., :requested_rank],
                svd_queries[..., :requested_rank],
                query_mask,
                query_batch=args.query_batch,
                block_chunk=args.block_chunk,
                exclude_block_prefix_tokens=args.exclude_block_prefix_tokens,
                scale=scale,
            )
            if svd_match.group(2) is None:
                return global_topk(scores, local_block_ids, budget_blocks, world_size)[1]
            _, candidates = global_topk(scores, local_block_ids, candidate_blocks, world_size)
            exact_candidates = candidate_exact_scores(
                raw_keys,
                raw_queries,
                query_mask,
                candidates,
                local_block_ids,
                args.exclude_block_prefix_tokens,
                scale,
                world_size,
            )
            adjusted_candidates = (
                exact_candidates.to(torch.float64) - candidates.to(torch.float64) * 1.0e-12
            )
            positions = torch.topk(
                adjusted_candidates,
                k=min(budget_blocks, adjusted_candidates.shape[1]),
                dim=1,
                largest=True,
            ).indices
            return torch.gather(candidates, dim=1, index=positions)
        colbert_match = re.fullmatch(r"colbert(32|64|128)(_rerank)?", method)
        if colbert_match is not None:
            requested_rank = int(colbert_match.group(1))
            if requested_rank == 128:
                method_keys = raw_keys
                method_queries = raw_queries
            else:
                if requested_rank > svd_keys.shape[-1]:
                    raise ValueError(
                        f"Method {method} requires rank {requested_rank}, but the profile stores "
                        f"rank {svd_keys.shape[-1]}"
                    )
                method_keys = svd_keys[..., :requested_rank]
                method_queries = svd_queries[..., :requested_rank]
            scores = score_colbert_blocks(
                method_keys,
                method_queries,
                query_mask,
                query_batch=args.query_batch,
                block_chunk=args.block_chunk,
                exclude_block_prefix_tokens=args.exclude_block_prefix_tokens,
            )
            if colbert_match.group(2) is None:
                return global_topk(scores, local_block_ids, budget_blocks, world_size)[1]
            _, candidates = global_topk(scores, local_block_ids, candidate_blocks, world_size)
            exact_candidates = candidate_colbert_scores(
                raw_keys,
                raw_queries,
                query_mask,
                candidates,
                local_block_ids,
                args.exclude_block_prefix_tokens,
                world_size,
            )
            adjusted_candidates = (
                exact_candidates.to(torch.float64) - candidates.to(torch.float64) * 1.0e-12
            )
            positions = torch.topk(
                adjusted_candidates,
                k=min(budget_blocks, adjusted_candidates.shape[1]),
                dim=1,
                largest=True,
            ).indices
            return torch.gather(candidates, dim=1, index=positions)
        if method == "qabs8":
            scores = score_qabs_blocks(
                raw_keys,
                raw_queries,
                query_mask,
                dims=args.qabs_dims,
                query_batch=args.query_batch,
                block_chunk=args.block_chunk,
                exclude_block_prefix_tokens=args.exclude_block_prefix_tokens,
                scale=scale,
            )
            return global_topk(scores, local_block_ids, budget_blocks, world_size)[1]
        raise ValueError(f"Unknown method: {method}")

    all_query_rows: list[dict[str, Any]] = []
    method_summaries: list[dict[str, Any]] = []
    for method in methods:
        for _ in range(args.warmup):
            run_method(method)
        timings: list[float] = []
        selected_ids: torch.Tensor | None = None
        for _ in range(args.repeats):
            sync(device, world_size)
            started = time.perf_counter()
            selected_ids = run_method(method)
            sync(device, world_size)
            elapsed = time.perf_counter() - started
            elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
            if world_size > 1:
                dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            timings.append(float(elapsed_tensor.item()))
        if selected_ids is None:
            raise RuntimeError(f"No output produced for method {method}")
        selected_exact_scores = distributed_lookup_scores(
            local_exact_scores, local_block_ids, selected_ids, world_size
        )
        summary, query_rows = evaluate_ids(
            method=method,
            selected_ids=selected_ids,
            oracle_ids=oracle_ids,
            selected_exact_scores=selected_exact_scores,
            global_exact_max=global_exact_max,
            global_exact_sum=global_exact_sum,
            queries=queries,
        )
        median_seconds = statistics.median(timings)
        summary.update(
            {
                "world_size": world_size,
                "median_seconds": median_seconds,
                "min_seconds": min(timings),
                "max_seconds": max(timings),
                "queries_per_second": len(queries) / median_seconds,
                "scanned_tokens_per_second": len(queries) * num_tokens / median_seconds,
                "candidate_blocks": candidate_blocks if method.endswith("_rerank") else "",
            }
        )
        all_query_rows.extend(query_rows)
        method_summaries.append(summary)
        if rank == 0:
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    if rank == 0:
        dataset_rows = aggregate_by_dataset(all_query_rows)
        query_csv_rows = [
            {**row, "selected_block_ids": json.dumps(row["selected_block_ids"])} for row in all_query_rows
        ]
        write_csv(
            out_dir / "query_results.csv",
            query_csv_rows,
            [
                "method",
                "query_id",
                "dataset",
                "oracle_block_recall",
                "exact_block_mass_recall",
                "answer_block_recall",
                "answer_block_mrr",
                "gold_block_count",
                "selected_block_ids",
            ],
        )
        write_csv(
            out_dir / "method_summary.csv",
            method_summaries,
            [
                "method",
                "queries",
                "oracle_block_recall",
                "exact_block_mass_recall",
                "answer_block_recall",
                "answer_block_mrr",
                "world_size",
                "median_seconds",
                "min_seconds",
                "max_seconds",
                "queries_per_second",
                "scanned_tokens_per_second",
                "candidate_blocks",
            ],
        )
        write_csv(
            out_dir / "dataset_summary.csv",
            dataset_rows,
            [
                "method",
                "dataset",
                "queries",
                "oracle_block_recall",
                "exact_block_mass_recall",
                "answer_block_recall",
                "answer_block_mrr",
            ],
        )
        output = {
            "source": "real LongBench text profiled by a real Qwen3 forward pass",
            "contains_synthetic_vectors": False,
            "profile_space": profile_summary["profile_space"],
            "world_size": world_size,
            "runtime_rank": rank,
            "assigned_profile_shards_rank0": assigned_shards,
            "num_blocks": num_blocks,
            "num_tokens": num_tokens,
            "block_tokens": block_tokens,
            "target_tokens": args.target_tokens,
            "budget_blocks": budget_blocks,
            "candidate_fraction": args.candidate_fraction,
            "candidate_blocks": candidate_blocks,
            "num_queries": len(queries),
            "pair_specs": profile_summary["pair_specs"],
            "svd_rank": profile_summary["svd_rank"],
            "qabs_dims": args.qabs_dims,
            "exclude_block_prefix_tokens": args.exclude_block_prefix_tokens,
            "index_load_seconds": load_seconds,
            "oracle_precompute_seconds": oracle_precompute_seconds,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "methods": method_summaries,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
