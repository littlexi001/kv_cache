from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from analyze_state_pointer_query_manifold import pointer_token_indices
from profile_real_qk import barrier, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact SVD QK reranking inside prototype-routed block candidates."
    )
    parser.add_argument("--base_profile_dir", required=True)
    parser.add_argument("--step_profile", required=True)
    parser.add_argument("--candidate_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--vote_depth", type=int, default=16)
    parser.add_argument("--final_blocks", type=int, default=39)
    parser.add_argument("--rrf_constant", type=int, default=60)
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def locate_shards(block_ids: np.ndarray, shards: list[dict]) -> np.ndarray:
    ends = np.array([int(shard["block_end"]) for shard in shards], dtype=np.int64)
    indices = np.searchsorted(ends, block_ids, side="right")
    if np.any(indices >= len(shards)):
        raise ValueError("candidate block id is outside K-index shards")
    return indices


def gather_candidate_k(
    *,
    block_ids: np.ndarray,
    layer: int,
    kv_head: int,
    shards: list[dict],
    profile_dir: Path,
    array_cache: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    shard_indices = locate_shards(block_ids, shards)
    sample_shard = int(shard_indices[0])
    sample_key = (sample_shard, layer)
    if sample_key not in array_cache:
        source = Path(shards[sample_shard]["layer_k_paths"][str(layer)])
        array_cache[sample_key] = np.load(profile_dir / source.name, mmap_mode="r")
    sample = array_cache[sample_key]
    output = np.empty(
        (len(block_ids), sample.shape[1], sample.shape[-1]), dtype=np.float16
    )
    for shard_index in np.unique(shard_indices):
        shard_index = int(shard_index)
        cache_key = (shard_index, layer)
        if cache_key not in array_cache:
            source = Path(shards[shard_index]["layer_k_paths"][str(layer)])
            array_cache[cache_key] = np.load(profile_dir / source.name, mmap_mode="r")
        positions = np.flatnonzero(shard_indices == shard_index)
        local_ids = block_ids[positions] - int(shards[shard_index]["block_start"])
        output[positions] = np.asarray(
            array_cache[cache_key][local_ids, :, kv_head], dtype=np.float16
        )
    return output


def rrf(top_ids: list[np.ndarray], *, depth: int, target: int, constant: int) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for ids in top_ids:
        for rank, block_id in enumerate(ids[:depth], start=1):
            scores[int(block_id)] += 1.0 / (constant + rank)
    return [
        block_id
        for block_id, _score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:target]
    ]


@torch.inference_mode()
def rerank_local(
    *,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    step_payload: dict,
    candidate_payload: dict,
    profile_summary: dict,
    output_dir: Path,
) -> tuple[Path, float]:
    profile_dir = Path(args.base_profile_dir)
    shards = profile_summary["shards"]
    layers = [int(item) for item in step_payload["layers"]]
    layer_to_index = {layer: index for index, layer in enumerate(layers)}
    num_query_heads = int(profile_summary["num_query_heads"])
    num_kv_heads = int(profile_summary["num_kv_heads"])
    repeat_groups = num_query_heads // num_kv_heads
    selected_heads = [
        tuple(int(item) for item in pair) for pair in candidate_payload["selected_heads"]
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    candidate_ids = candidate_payload["candidate_ids"].long().numpy()
    profile_indices = [int(item) for item in candidate_payload["step_profile_indices"]]
    array_cache: dict[tuple[int, int], np.ndarray] = {}
    rows = []
    started = time.perf_counter()
    local_ordinals = list(range(rank, len(profile_indices), world_size))
    for local_count, ordinal in enumerate(local_ordinals, start=1):
        profile_index = profile_indices[ordinal]
        step = step_payload["steps"][profile_index]
        blocks = candidate_ids[ordinal]
        pointer_indices = pointer_token_indices(
            tokenizer=tokenizer,
            step=step,
            token_positions=step_payload["token_positions"][profile_index],
        )
        ranked_lists = []
        scoring_started = time.perf_counter()
        for layer, query_head in selected_heads:
            query = step_payload["svd_q"][
                profile_index,
                pointer_indices,
                layer_to_index[layer],
                query_head,
            ].to(device=device, dtype=torch.float16)
            key_numpy = gather_candidate_k(
                block_ids=blocks,
                layer=layer,
                kv_head=query_head // repeat_groups,
                shards=shards,
                profile_dir=profile_dir,
                array_cache=array_cache,
            )
            key = torch.from_numpy(key_numpy).to(device=device, dtype=torch.float16)
            scores = torch.einsum("ud,ctd->uct", query, key).amax(dim=-1)
            depth = min(args.vote_depth, len(blocks))
            positions = torch.topk(scores, k=depth, dim=1).indices.cpu().numpy()
            ranked_lists.extend(blocks[row] for row in positions)
        final_blocks = rrf(
            ranked_lists,
            depth=args.vote_depth,
            target=args.final_blocks,
            constant=args.rrf_constant,
        )
        gold = {int(item) for item in step["target_block_ids"]}
        rows.append(
            {
                "candidate_ordinal": ordinal,
                "step_profile_index": profile_index,
                "query_id": int(step["query_id"]),
                "step_index": int(step["step_index"]),
                "candidate_hit": bool(gold & set(int(item) for item in blocks)),
                "rerank_hit39": bool(gold & set(final_blocks)),
                "gold_block_ids": sorted(gold),
                "final_blocks": final_blocks,
                "scoring_seconds": time.perf_counter() - scoring_started,
            }
        )
        if local_count % 20 == 0:
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "local_steps": local_count,
                        "assigned_steps": len(local_ordinals),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    local_path = output_dir / f"rerank_rows_rank{rank:03d}.pt"
    torch.save(rows, local_path)
    return local_path, time.perf_counter() - started


def summarize(
    *,
    args: argparse.Namespace,
    world_size: int,
    wall_seconds: float,
    candidate_payload: dict,
    profile_summary: dict,
    output_dir: Path,
) -> dict:
    rows = []
    for rank in range(world_size):
        rows.extend(
            torch.load(
                output_dir / f"rerank_rows_rank{rank:03d}.pt",
                map_location="cpu",
                weights_only=False,
            )
        )
    rows.sort(key=lambda row: row["candidate_ordinal"])
    cutoffs = [1, 3, 4, 8, 16, args.final_blocks]
    step_metrics = {}
    for step_index in (0, 1):
        subset = [row for row in rows if row["step_index"] == step_index]
        candidate_hits = sum(row["candidate_hit"] for row in subset)
        rerank_hits = sum(row["rerank_hit39"] for row in subset)
        step_metrics[str(step_index)] = {
            "steps": len(subset),
            "candidate_gold_recall": candidate_hits / len(subset),
            "reranked_gold_hit39": rerank_hits / len(subset),
            "rerank_retention_given_candidate": rerank_hits / max(candidate_hits, 1),
            "hit_rate_by_blocks": {
                str(cutoff): mean(
                    bool(
                        set(row["gold_block_ids"])
                        & set(row["final_blocks"][:cutoff])
                    )
                    for row in subset
                )
                for cutoff in cutoffs
            },
        }
    by_query = defaultdict(dict)
    for row in rows:
        by_query[row["query_id"]][row["step_index"]] = row
    both_candidate = mean(
        query[0]["candidate_hit"] and query[1]["candidate_hit"]
        for query in by_query.values()
    )
    both_reranked = mean(
        query[0]["rerank_hit39"] and query[1]["rerank_hit39"]
        for query in by_query.values()
    )
    both_by_cutoff = {
        str(cutoff): mean(
            all(
                bool(
                    set(query[step_index]["gold_block_ids"])
                    & set(query[step_index]["final_blocks"][:cutoff])
                )
                for step_index in (0, 1)
            )
            for query in by_query.values()
        )
        for cutoff in cutoffs
    }
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    selected_heads = candidate_payload["selected_heads"]
    candidate_budget = int(candidate_payload["candidate_budget"])
    estimated_k_bytes = (
        len(rows)
        * candidate_budget
        * int(profile_summary["block_tokens"])
        * int(profile_summary["svd_rank"])
        * 2
        * len(selected_heads)
    )
    return {
        "source": "exact SVD32 QK reranking inside prototype-routed candidates",
        "contains_synthetic_vectors": False,
        "selection_uses_test_gold": False,
        "base_profile_dir": args.base_profile_dir,
        "step_profile": args.step_profile,
        "candidate_path": args.candidate_path,
        "steps": len(rows),
        "queries": len(by_query),
        "selected_heads": selected_heads,
        "candidate_budget": candidate_budget,
        "final_blocks": args.final_blocks,
        "vote_depth": args.vote_depth,
        "step_metrics": step_metrics,
        "both_steps_candidate_recall": both_candidate,
        "both_steps_reranked_hit39": both_reranked,
        "both_steps_hit_rate_by_blocks": both_by_cutoff,
        "world_size": world_size,
        "wall_seconds": wall_seconds,
        "mean_step_scoring_seconds": mean(row["scoring_seconds"] for row in rows),
        "estimated_svd_k_bytes_read": estimated_k_bytes,
    }


def main() -> None:
    args = parse_args()
    rank, world_size, _local_rank, device = setup_distributed()
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)
    step_payload = torch.load(args.step_profile, map_location="cpu", weights_only=False)
    candidate_payload = torch.load(args.candidate_path, map_location="cpu", weights_only=False)
    profile_summary = json.loads(
        (Path(args.base_profile_dir) / "summary.json").read_text()
    )
    _local_path, local_seconds = rerank_local(
        args=args,
        rank=rank,
        world_size=world_size,
        device=device,
        step_payload=step_payload,
        candidate_payload=candidate_payload,
        profile_summary=profile_summary,
        output_dir=output_dir,
    )
    wall = torch.tensor(local_seconds, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(wall, op=dist.ReduceOp.MAX)
    barrier(world_size)
    if rank == 0:
        summary = summarize(
            args=args,
            world_size=world_size,
            wall_seconds=float(wall.item()),
            candidate_payload=candidate_payload,
            profile_summary=profile_summary,
            output_dir=output_dir,
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    barrier(world_size)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
