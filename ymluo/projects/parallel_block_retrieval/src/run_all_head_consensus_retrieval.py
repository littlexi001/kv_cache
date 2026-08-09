from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from profile_real_qk import read_jsonl, setup_distributed
from run_lexical_block_retrieval import evaluate_selection, group_for_context, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve per query-head independently, then recall blocks supported across heads."
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument(
        "--query_profiles",
        default="",
        help="Optional query_profiles.pt from a query-only holdout profile.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--top_per_head", type=int, default=16)
    parser.add_argument("--query_batch", type=int, default=8)
    parser.add_argument("--block_chunk", type=int, default=64)
    parser.add_argument("--exclude_block_prefix_tokens", type=int, default=16)
    parser.add_argument("--rrf_constant", type=float, default=60.0)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--reference_pairs", default="3:10,21:8,6:7,16:14")
    return parser.parse_args()


def parse_pairs(spec: str) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for item in spec.split(","):
        layer, head = item.strip().split(":", maxsplit=1)
        output.append((int(layer), int(head)))
    return output


def update_topk(
    current_scores: torch.Tensor,
    current_ids: torch.Tensor,
    new_scores: torch.Tensor,
    new_ids: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded_ids = new_ids[None, None, :].expand(
        new_scores.shape[0], new_scores.shape[1], new_scores.shape[2]
    )
    scores = torch.cat([current_scores, new_scores], dim=2)
    ids = torch.cat([current_ids, expanded_ids], dim=2)
    keep = min(k, int(scores.shape[2]))
    values, positions = torch.topk(scores, k=keep, dim=2, largest=True, sorted=True)
    return values, torch.gather(ids, dim=2, index=positions)


def score_layer_shards(
    *,
    layer: int,
    layer_index: int,
    shards: list[dict[str, Any]],
    profile_dir: Path,
    queries: torch.Tensor,
    query_mask: torch.Tensor,
    num_kv_heads: int,
    top_per_head: int,
    query_batch: int,
    block_chunk: int,
    exclude_block_prefix_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_count, query_tokens, num_query_heads, rank_dim = queries.shape
    repeat_groups = num_query_heads // num_kv_heads
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    best_scores = torch.full(
        (query_count, num_query_heads, top_per_head),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    best_ids = torch.full(
        (query_count, num_query_heads, top_per_head),
        -1,
        dtype=torch.long,
        device=device,
    )
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
                valid = query_mask[query_start:query_end].sum(dim=1).clamp_min(1).float()
                scores = (per_query_token * mask).sum(dim=1) / valid[:, None, None, None]
                scores = scores.reshape(batch, num_query_heads, count)
                values, ids = update_topk(
                    best_scores[query_start:query_end],
                    best_ids[query_start:query_end],
                    scores,
                    block_ids,
                    top_per_head,
                )
                best_scores[query_start:query_end] = values
                best_ids[query_start:query_end] = ids
    return best_scores, best_ids


def merge_distributed_topk(
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
    scores = torch.cat(score_parts, dim=2)
    ids = torch.cat(id_parts, dim=2)
    values, positions = torch.topk(scores, k=top_per_head, dim=2, largest=True, sorted=True)
    return values, torch.gather(ids, dim=2, index=positions)


def block_consensus(
    head_ids: np.ndarray,
    *,
    rrf_constant: float,
    rank_limit: int,
    selected_pairs: set[tuple[int, int]] | None = None,
    layers: list[int],
) -> dict[int, dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {}
    for layer_index, layer in enumerate(layers):
        for query_head in range(head_ids.shape[1]):
            if selected_pairs is not None and (layer, query_head) not in selected_pairs:
                continue
            for rank in range(min(rank_limit, head_ids.shape[2])):
                block_id = int(head_ids[layer_index, query_head, rank])
                if block_id < 0:
                    continue
                item = stats.setdefault(
                    block_id,
                    {
                        "layers": set(),
                        "head_votes": 0,
                        "rrf": 0.0,
                        "best_rank": rank + 1,
                    },
                )
                item["layers"].add(layer)
                item["head_votes"] += 1
                item["rrf"] += 1.0 / (rrf_constant + rank + 1)
                item["best_rank"] = min(int(item["best_rank"]), rank + 1)
    return stats


def rank_consensus(
    stats: dict[int, dict[str, Any]],
    mode: str,
) -> list[int]:
    def key(item: tuple[int, dict[str, Any]]) -> tuple[Any, ...]:
        block_id, values = item
        layers = len(values["layers"])
        heads = int(values["head_votes"])
        rrf = float(values["rrf"])
        best = int(values["best_rank"])
        if mode == "layer_consensus":
            return (-layers, -heads, -rrf, best, block_id)
        if mode == "head_vote":
            return (-heads, -layers, -rrf, best, block_id)
        if mode == "rrf":
            return (-rrf, -layers, -heads, best, block_id)
        raise ValueError(f"unknown consensus mode: {mode}")

    return [block_id for block_id, _values in sorted(stats.items(), key=key)]


def summarize_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        output.append(
            {
                "method": method,
                "queries": len(group),
                "source_record_recall": statistics.fmean(
                    float(row["source_record_recall"]) for row in group
                ),
                "record_top1_recall": statistics.fmean(
                    float(row["record_top1_recall"]) for row in group
                ),
                "answer_block_recall": statistics.fmean(
                    float(row["answer_block_recall"]) for row in group
                ),
                "answer_block_mrr": statistics.fmean(
                    float(row["answer_block_mrr"]) for row in group
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.target_blocks <= 0 or args.top_per_head <= 0:
        raise ValueError("target_blocks and top_per_head must be positive")
    rank, world_size, _local_rank, device = setup_distributed()
    corpus_dir = Path(args.corpus_dir)
    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    profile_summary = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    if profile_summary.get("contains_synthetic_vectors"):
        raise ValueError("all-head retrieval requires real Q/K profiles")
    layers = [int(item) for item in profile_summary["layers"]]
    num_query_heads = int(profile_summary["num_query_heads"])
    num_kv_heads = int(profile_summary["num_kv_heads"])
    query_profile_path = (
        Path(args.query_profiles)
        if args.query_profiles
        else profile_dir / Path(profile_summary["query_profiles_path"]).name
    )
    query_payload = torch.load(
        query_profile_path,
        map_location="cpu",
        weights_only=False,
    )
    query_vectors = query_payload["svd_q"]
    query_mask = query_payload["mask"]
    query_count = int(query_vectors.shape[0])
    if args.max_queries > 0:
        query_count = min(query_count, args.max_queries)
        query_vectors = query_vectors[:query_count]
        query_mask = query_mask[:query_count]
    queries = read_jsonl(corpus_dir / "queries.jsonl")[:query_count]
    records = read_jsonl(corpus_dir / "records.jsonl")

    profile_shards = list(profile_summary["shards"])
    local_shards = [
        shard for shard_index, shard in enumerate(profile_shards) if shard_index % world_size == rank
    ]
    all_scores = (
        np.empty(
            (query_count, len(layers), num_query_heads, args.top_per_head),
            dtype=np.float32,
        )
        if rank == 0
        else None
    )
    all_ids = (
        np.empty(
            (query_count, len(layers), num_query_heads, args.top_per_head),
            dtype=np.int32,
        )
        if rank == 0
        else None
    )

    started = time.perf_counter()
    layer_seconds: list[float] = []
    for layer_index, layer in enumerate(layers):
        layer_started = time.perf_counter()
        layer_queries = query_vectors[:, :, layer_index].to(device=device, non_blocking=True)
        layer_mask = query_mask.to(device=device, non_blocking=True)
        local_scores, local_ids = score_layer_shards(
            layer=layer,
            layer_index=layer_index,
            shards=local_shards,
            profile_dir=profile_dir,
            queries=layer_queries,
            query_mask=layer_mask,
            num_kv_heads=num_kv_heads,
            top_per_head=args.top_per_head,
            query_batch=args.query_batch,
            block_chunk=args.block_chunk,
            exclude_block_prefix_tokens=args.exclude_block_prefix_tokens,
            device=device,
        )
        merged_scores, merged_ids = merge_distributed_topk(
            local_scores, local_ids, args.top_per_head, world_size
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - layer_started
        layer_seconds.append(elapsed)
        if rank == 0:
            assert all_scores is not None and all_ids is not None
            all_scores[:, layer_index] = merged_scores.cpu().numpy()
            all_ids[:, layer_index] = merged_ids.cpu().numpy().astype(np.int32)
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
        assert all_scores is not None and all_ids is not None
        np.savez_compressed(
            output_dir / "per_head_topk.npz",
            scores=all_scores,
            block_ids=all_ids,
            layers=np.asarray(layers, dtype=np.int32),
        )
        block_count = int(profile_summary["num_blocks"])
        block_to_record = np.empty(block_count, dtype=np.int32)
        source_record_by_start: dict[int, int] = {}
        for record_id, record in enumerate(records):
            block_start = int(record["block_start"])
            block_end = block_start + int(record["block_count"])
            block_to_record[block_start:block_end] = record_id
            source_record_by_start[block_start] = record_id

        reference_pairs = {
            (layer, head)
            for layer, head in parse_pairs(args.reference_pairs)
            if layer in layers and 0 <= head < num_query_heads
        }
        if not reference_pairs:
            reference_pairs = {(layers[0], 0)}
        rows: list[dict[str, Any]] = []
        selected_stats_rows: list[dict[str, Any]] = []
        limits = sorted({min(item, args.top_per_head) for item in [1, 2, 4, 8, args.top_per_head]})
        oracle_counts = {f"all_heads_top{limit}": 0 for limit in limits}
        oracle_counts.update(
            {f"selected4_top{limit}": 0 for limit in limits}
        )
        union_sizes: list[int] = []
        for query_index, query in enumerate(queries):
            head_ids = all_ids[query_index]
            gold = set(int(item) for item in query.get("gold_block_ids", []))
            for limit in limits:
                all_union = set(int(item) for item in head_ids[:, :, :limit].reshape(-1))
                all_union.discard(-1)
                oracle_counts[f"all_heads_top{limit}"] += int(bool(gold & all_union))
                selected_union = {
                    int(head_ids[layers.index(layer), head, head_rank])
                    for layer, head in reference_pairs
                    for head_rank in range(limit)
                }
                selected_union.discard(-1)
                oracle_counts[f"selected4_top{limit}"] += int(bool(gold & selected_union))

            stats = block_consensus(
                head_ids,
                rrf_constant=args.rrf_constant,
                rank_limit=args.top_per_head,
                layers=layers,
            )
            union_sizes.append(len(stats))
            selected4_stats = block_consensus(
                head_ids,
                rrf_constant=args.rrf_constant,
                rank_limit=args.top_per_head,
                selected_pairs=reference_pairs,
                layers=layers,
            )
            method_rankings = {
                "allhead_layer_consensus": rank_consensus(stats, "layer_consensus"),
                "allhead_head_vote": rank_consensus(stats, "head_vote"),
                "allhead_rrf": rank_consensus(stats, "rrf"),
                "selected4_independent_rrf": rank_consensus(selected4_stats, "rrf"),
            }
            source_record = source_record_by_start[int(query["block_start"])]
            for method, full_ranking in method_rankings.items():
                ranked = full_ranking[: args.target_blocks]
                predicted_record = int(block_to_record[ranked[0]])
                rows.append(
                    evaluate_selection(
                        method=method,
                        query=query,
                        ranked_ids=ranked,
                        context_ids=group_for_context(ranked, block_to_record),
                        predicted_record=predicted_record,
                        source_record=source_record,
                        record_margin=0.0,
                    )
                )
            primary = method_rankings["allhead_layer_consensus"][: args.target_blocks]
            for selected_rank, block_id in enumerate(primary, start=1):
                values = stats[block_id]
                selected_stats_rows.append(
                    {
                        "query_id": query_index,
                        "rank": selected_rank,
                        "block_id": block_id,
                        "layer_votes": len(values["layers"]),
                        "head_votes": int(values["head_votes"]),
                        "rrf": float(values["rrf"]),
                        "best_head_rank": int(values["best_rank"]),
                        "is_gold_block": float(block_id in gold),
                    }
                )

        method_summaries = summarize_methods(rows)
        write_csv(output_dir / "query_results.csv", rows, list(rows[0]))
        write_csv(
            output_dir / "method_summary.csv", method_summaries, list(method_summaries[0])
        )
        write_csv(
            output_dir / "selected_consensus_blocks.csv",
            selected_stats_rows,
            list(selected_stats_rows[0]),
        )

        head_rows: list[dict[str, Any]] = []
        layer_rows: list[dict[str, Any]] = []
        for layer_index, layer in enumerate(layers):
            layer_entry: dict[str, Any] = {"layer": layer, "queries": query_count}
            for limit in limits:
                layer_hits = 0
                for query_index, query in enumerate(queries):
                    gold = set(int(item) for item in query.get("gold_block_ids", []))
                    ids = set(
                        int(item)
                        for item in all_ids[query_index, layer_index, :, :limit].reshape(-1)
                    )
                    layer_hits += int(bool(gold & ids))
                layer_entry[f"oracle_recall_top{limit}_per_head"] = layer_hits / query_count
            layer_rows.append(layer_entry)
            for query_head in range(num_query_heads):
                head_entry: dict[str, Any] = {
                    "layer": layer,
                    "query_head": query_head,
                    "kv_head": query_head // (num_query_heads // num_kv_heads),
                    "queries": query_count,
                    "unique_top1_blocks": len(
                        set(int(item) for item in all_ids[:, layer_index, query_head, 0])
                    ),
                }
                for limit in limits:
                    hits = 0
                    for query_index, query in enumerate(queries):
                        gold = set(int(item) for item in query.get("gold_block_ids", []))
                        ids = set(
                            int(item)
                            for item in all_ids[query_index, layer_index, query_head, :limit]
                        )
                        hits += int(bool(gold & ids))
                    head_entry[f"answer_recall_top{limit}"] = hits / query_count
                head_rows.append(head_entry)
        write_csv(output_dir / "per_head_summary.csv", head_rows, list(head_rows[0]))
        write_csv(output_dir / "per_layer_summary.csv", layer_rows, list(layer_rows[0]))

        summary = {
            "source": "independent all-layer/all-query-head retrieval followed by rank consensus",
            "contains_synthetic_vectors": False,
            "queries": query_count,
            "layers": layers,
            "num_query_heads": num_query_heads,
            "total_query_heads": len(layers) * num_query_heads,
            "top_per_head": args.top_per_head,
            "target_blocks": args.target_blocks,
            "rrf_constant": args.rrf_constant,
            "retrieval_world_size": world_size,
            "total_seconds": total_seconds,
            "mean_layer_seconds": statistics.fmean(layer_seconds),
            "mean_union_blocks": statistics.fmean(union_sizes),
            "oracle_recall": {
                key: value / query_count for key, value in oracle_counts.items()
            },
            "methods": method_summaries,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
