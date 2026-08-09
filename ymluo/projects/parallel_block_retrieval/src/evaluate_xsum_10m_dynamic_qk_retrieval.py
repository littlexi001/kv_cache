from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from profile_real_qk import QKCapture, captured_qk, resolve_dtype, run_base_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dynamic SVD32 QK retrieval on nested XSum memories."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prefix_tokens", default="8,16,32,64")
    parser.add_argument("--topks", default="8,64,512")
    parser.add_argument("--ranking_depth", type=int, default=512)
    parser.add_argument("--query_q_tokens", type=int, default=16)
    parser.add_argument("--score_chunk_blocks", type=int, default=2048)
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def parse_ints(spec: str) -> list[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) <= 0:
        raise ValueError("integer list must contain positive values")
    return values


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def top_indices(scores: np.ndarray, depth: int) -> list[int]:
    scores = np.asarray(scores, dtype=np.float64)
    depth = min(depth, len(scores))
    if depth == len(scores):
        candidates = np.arange(len(scores), dtype=np.int64)
    else:
        candidates = np.argpartition(-scores, depth - 1)[:depth]
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order].astype(np.int64).tolist()


def reciprocal_rank_fusion(
    rankings: list[list[int]], *, depth: int, rrf_k: float
) -> list[int]:
    scores: dict[int, float] = {}
    best: dict[int, int] = {}
    for ranking in rankings:
        for rank, block_id in enumerate(ranking, start=1):
            scores[block_id] = scores.get(block_id, 0.0) + 1.0 / (rrf_k + rank)
            best[block_id] = min(best.get(block_id, rank), rank)
    return sorted(scores, key=lambda item: (-scores[item], best[item], item))[:depth]


def gold_metrics(
    ranking: list[int], base_blocks: int, source_blocks: int, topks: list[int]
) -> dict[str, Any]:
    gold = set(range(base_blocks, base_blocks + source_blocks))
    result: dict[str, Any] = {}
    for topk in topks:
        selected = set(ranking[:topk])
        result[f"source_recall_at_{topk}"] = len(selected & gold) / source_blocks
        result[f"source_any_at_{topk}"] = bool(selected & gold)
        result[f"source_last_at_{topk}"] = base_blocks + source_blocks - 1 in selected
    return result


def exact_gold_ranks(
    scores: np.ndarray, base_blocks: int, source_blocks: int
) -> tuple[int, int]:
    ids = np.arange(len(scores), dtype=np.int64)
    ranks = []
    for gold_id in range(base_blocks, base_blocks + source_blocks):
        value = scores[gold_id]
        rank = 1 + int((scores > value).sum()) + int(((scores == value) & (ids < gold_id)).sum())
        ranks.append(rank)
    return min(ranks), ranks[-1]


def load_base_index(
    profile_summary: dict[str, Any], device: torch.device
) -> torch.Tensor:
    total_blocks = int(profile_summary["base_blocks"])
    first = np.load(profile_summary["shards"][0]["path"], mmap_mode="r")
    output = torch.empty(
        total_blocks,
        first.shape[1],
        first.shape[2],
        first.shape[3],
        dtype=torch.float16,
        device=device,
    )
    for shard in profile_summary["shards"]:
        array = np.load(shard["path"], mmap_mode="r")
        start = int(shard["block_start"])
        end = int(shard["block_end"])
        output[start:end].copy_(torch.from_numpy(np.array(array, copy=True)).to(device))
    return output


@torch.inference_mode()
def per_head_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    chunk_blocks: int,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(keys), chunk_blocks):
        chunk = keys[start : start + chunk_blocks]
        similarity = torch.einsum(
            "qpd,btpd->qbpt", query.float(), chunk.float()
        )
        parts.append(similarity.amax(dim=(0, 3)).transpose(0, 1).cpu())
    return torch.cat(parts, dim=1)


def method_rankings(
    per_head: np.ndarray,
    *,
    depth: int,
    rrf_k: float,
) -> tuple[list[int], list[int], list[list[int]]]:
    fused_scores = per_head.max(axis=0)
    max_ranking = top_indices(fused_scores, depth)
    head_rankings = [top_indices(scores, depth) for scores in per_head]
    rrf_ranking = reciprocal_rank_fusion(head_rankings, depth=depth, rrf_k=rrf_k)
    return max_ranking, rrf_ranking, head_rankings


def summarize(rows: list[dict[str, Any]], topks: list[int]) -> dict[str, Any]:
    quality = []
    keys = sorted(
        {
            (int(row["memory_tokens"]), int(row["prefix_tokens"]), str(row["method"]))
            for row in rows
        }
    )
    for memory_tokens, prefix_tokens, method in keys:
        group = [
            row
            for row in rows
            if int(row["memory_tokens"]) == memory_tokens
            and int(row["prefix_tokens"]) == prefix_tokens
            and str(row["method"]) == method
        ]
        item: dict[str, Any] = {
            "memory_tokens": memory_tokens,
            "prefix_tokens": prefix_tokens,
            "method": method,
            "queries": len(group),
            "mean_query_capture_seconds": mean(
                float(row["query_capture_seconds"]) for row in group
            ),
            "mean_score_seconds": mean(float(row["score_seconds"]) for row in group),
            "mean_gold_supporting_heads_at_512": mean(
                float(row["gold_supporting_heads_at_512"]) for row in group
            ),
        }
        for topk in topks:
            item[f"mean_source_recall_at_{topk}"] = mean(
                float(row[f"source_recall_at_{topk}"]) for row in group
            )
            item[f"source_any_at_{topk}"] = mean(
                bool(row[f"source_any_at_{topk}"]) for row in group
            )
            item[f"source_last_at_{topk}"] = mean(
                bool(row[f"source_last_at_{topk}"]) for row in group
            )
        quality.append(item)

    stability = []
    for memory_tokens in sorted({int(row["memory_tokens"]) for row in rows}):
        for method in sorted({str(row["method"]) for row in rows}):
            lookup = {
                (int(row["query_id"]), int(row["prefix_tokens"])): row
                for row in rows
                if int(row["memory_tokens"]) == memory_tokens
                and str(row["method"]) == method
            }
            prefixes = sorted({prefix for _, prefix in lookup})
            for left, right in zip(prefixes, prefixes[1:]):
                values = []
                for query_id in sorted({query_id for query_id, _ in lookup}):
                    left_ids = set(lookup[(query_id, left)]["top_block_ids"][:8])
                    right_ids = set(lookup[(query_id, right)]["top_block_ids"][:8])
                    values.append(len(left_ids & right_ids) / len(left_ids | right_ids))
                stability.append(
                    {
                        "memory_tokens": memory_tokens,
                        "method": method,
                        "prefix_transition": f"{left}->{right}",
                        "top8_jaccard": mean(values),
                    }
                )
    return {"retrieval_quality": quality, "prefix_stability": stability}


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, _, device = setup_distributed()
    data_dir = Path(args.data_dir)
    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)

    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    profile_summary = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    source_blocks = int(data_summary["source_blocks"])
    prefix_tokens = parse_ints(args.prefix_tokens)
    topks = parse_ints(args.topks)
    if max(topks) > args.ranking_depth:
        raise ValueError("ranking_depth must cover all top-k values")

    dtype = resolve_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    pair_specs = profile_summary["pair_specs"]
    capture = QKCapture(model, sorted({int(item["layer"]) for item in pair_specs}))
    basis_payload = torch.load(profile_dir / "basis.pt", map_location="cpu", weights_only=False)
    basis = basis_payload["basis"].to(device=device, dtype=dtype)
    projected_mean = torch.einsum(
        "pd,pdr->pr", basis_payload["mean"].to(device).float(), basis.float()
    ).to(dtype)

    load_started = time.perf_counter()
    base_raw = load_base_index(profile_summary, device)
    source_array = np.load(profile_summary["source_index_path"], mmap_mode="r")
    source_raw = torch.from_numpy(np.array(source_array, copy=True)).to(device).reshape(
        len(queries), source_blocks, source_array.shape[1], source_array.shape[2], source_array.shape[3]
    )
    base_centered = F.normalize(
        base_raw - projected_mean[None, None, :, :], dim=-1
    )
    source_centered = F.normalize(
        source_raw - projected_mean[None, None, None, :, :], dim=-1
    )
    torch.cuda.synchronize(device)
    index_load_seconds = time.perf_counter() - load_started

    local_query_ids = [item for item in range(len(queries)) if item % world_size == rank]
    rows: list[dict[str, Any]] = []
    for local_index, query_id in enumerate(local_query_ids, start=1):
        for prefix in prefix_tokens:
            input_ids = torch.from_numpy(
                np.asarray(queries[query_id, :prefix], dtype=np.int64)
            )[None, :].to(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            run_base_model(model, capture, input_ids)
            query_raw, _ = captured_qk(model, capture, pair_specs, "pre_rope_block_qk")
            keep = min(args.query_q_tokens, int(query_raw.shape[1]))
            query_raw = query_raw[0, -keep:]
            query_projected = torch.einsum("qpd,pdr->qpr", query_raw, basis)
            query_centered = F.normalize(query_projected.float(), dim=-1).to(dtype)
            torch.cuda.synchronize(device)
            capture_seconds = time.perf_counter() - started

            torch.cuda.synchronize(device)
            started = time.perf_counter()
            raw_base_scores = per_head_scores(
                query_projected, base_raw, args.score_chunk_blocks
            )
            raw_source_scores = per_head_scores(
                query_projected, source_raw[query_id], args.score_chunk_blocks
            )
            centered_base_scores = per_head_scores(
                query_centered, base_centered, args.score_chunk_blocks
            )
            centered_source_scores = per_head_scores(
                query_centered, source_centered[query_id], args.score_chunk_blocks
            )
            torch.cuda.synchronize(device)
            score_seconds = time.perf_counter() - started

            for memory_tokens in data_summary["memory_scales_tokens"]:
                total_blocks = int(memory_tokens) // int(data_summary["block_tokens"])
                base_count = total_blocks - source_blocks
                score_sets = {
                    "qk_raw": np.concatenate(
                        [raw_base_scores[:, :base_count].numpy(), raw_source_scores.numpy()], axis=1
                    ),
                    "qk_kcentered_cosine": np.concatenate(
                        [
                            centered_base_scores[:, :base_count].numpy(),
                            centered_source_scores.numpy(),
                        ],
                        axis=1,
                    ),
                }
                for family, per_head in score_sets.items():
                    max_ranking, rrf_ranking, head_rankings = method_rankings(
                        per_head, depth=args.ranking_depth, rrf_k=args.rrf_k
                    )
                    gold = set(range(base_count, base_count + source_blocks))
                    supporting_heads = sum(
                        bool(set(ranking[:512]) & gold) for ranking in head_rankings
                    )
                    for suffix, ranking, exact_scores in (
                        ("max", max_ranking, per_head.max(axis=0)),
                        ("head_rrf", rrf_ranking, None),
                    ):
                        row: dict[str, Any] = {
                            "query_id": query_id,
                            "memory_tokens": int(memory_tokens),
                            "memory_blocks": total_blocks,
                            "base_blocks": base_count,
                            "prefix_tokens": prefix,
                            "method": f"{family}_{suffix}",
                            "query_capture_seconds": capture_seconds,
                            "score_seconds": score_seconds,
                            "gold_supporting_heads_at_512": supporting_heads,
                            "top_block_ids": ranking[: max(topks)],
                            "selection_uses_target": False,
                            **gold_metrics(ranking, base_count, source_blocks, topks),
                        }
                        if exact_scores is not None:
                            best_rank, last_rank = exact_gold_ranks(
                                exact_scores, base_count, source_blocks
                            )
                            row["best_gold_rank"] = best_rank
                            row["last_source_rank"] = last_rank
                        rows.append(row)
        print(
            json.dumps(
                {
                    "rank": rank,
                    "query": local_index,
                    "queries": len(local_query_ids),
                    "query_id": query_id,
                }
            ),
            flush=True,
        )

    shard_path = output_dir / f"rows_rank{rank:03d}.jsonl"
    with shard_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / f"runtime_rank{rank:03d}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "queries": len(local_query_ids),
                "index_load_seconds": index_load_seconds,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    barrier(world_size)
    if rank == 0:
        all_rows = [
            row
            for shard in range(world_size)
            for row in read_jsonl(output_dir / f"rows_rank{shard:03d}.jsonl")
        ]
        all_rows.sort(
            key=lambda row: (
                int(row["query_id"]),
                int(row["memory_tokens"]),
                int(row["prefix_tokens"]),
                str(row["method"]),
            )
        )
        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        runtime = [
            json.loads(
                (output_dir / f"runtime_rank{item:03d}.json").read_text(
                    encoding="utf-8"
                )
            )
            for item in range(world_size)
        ]
        summary = {
            "source": "real XSum dynamic SVD32 QK retrieval",
            "data_summary": data_summary,
            "profile_summary": profile_summary,
            "methods": [
                "qk_raw_max",
                "qk_raw_head_rrf",
                "qk_kcentered_cosine_max",
                "qk_kcentered_cosine_head_rrf",
            ],
            "prefix_tokens": prefix_tokens,
            "topks": topks,
            "query_q_tokens": args.query_q_tokens,
            "centered_cosine_definition": (
                "normalize projected Q; subtract the train-K projected mean from K, "
                "then normalize K"
            ),
            "ranking_depth": args.ranking_depth,
            "world_size": world_size,
            "runtime": runtime,
            "contains_synthetic_vectors": False,
            "selection_uses_target": False,
            **summarize(all_rows, topks),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    barrier(world_size)
    capture.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
