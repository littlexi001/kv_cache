from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy import sparse
from transformers import AutoModel, AutoTokenizer

from evaluate_xsum_news_ppl_retrieval import encode_e5
from run_iterative_condition_retrieval import BM25Index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure nested-scale and prefix-evolution retrieval on the shared XSum "
            "continuation memory using BM25 and E5."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--embedding_model_name_or_path", default="intfloat/e5-base-v2")
    parser.add_argument("--prefix_tokens", default="8,16,32,64")
    parser.add_argument("--topks", default="8,64,512")
    parser.add_argument("--fusion_depth", type=int, default=4096)
    parser.add_argument("--embedding_batch_size", type=int, default=128)
    parser.add_argument("--embedding_max_length", type=int, default=96)
    parser.add_argument("--decode_batch_size", type=int, default=2048)
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


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


def decode_blocks(
    tokenizer: Any, blocks: np.ndarray, batch_size: int
) -> list[str]:
    texts: list[str] = []
    for start in range(0, len(blocks), batch_size):
        batch = np.asarray(blocks[start : start + batch_size], dtype=np.int64)
        texts.extend(tokenizer.batch_decode(batch.tolist(), skip_special_tokens=True))
    return texts


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


def bm25_external_scores(
    index: BM25Index, documents: list[str], query: str, *, k1: float, b: float
) -> np.ndarray:
    counts = index.vectorizer.transform(documents).tocsr().astype(np.float32)
    if counts.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    row_ids = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
    frequencies = counts.data.copy()
    denominator = frequencies + k1 * (
        1.0 - b + b * lengths[row_ids] / max(index.average_length, 1.0e-6)
    )
    counts.data = (
        index.inverse_document_frequency[counts.indices]
        * frequencies
        * (k1 + 1.0)
        / denominator
    )
    query_counts = index.vectorizer.transform([query]).tocsr().astype(np.float32)
    query_counts.data.fill(1.0)
    result = query_counts @ counts.transpose()
    if sparse.issparse(result):
        result = result.toarray()
    return np.asarray(result).reshape(-1).astype(np.float32, copy=False)


def gold_metrics(
    ranking: list[int], base_blocks: int, source_blocks: int, topks: list[int]
) -> dict[str, Any]:
    gold = set(range(base_blocks, base_blocks + source_blocks))
    positions = {block_id: rank for rank, block_id in enumerate(ranking, start=1)}
    result: dict[str, Any] = {
        "best_gold_rank_within_fusion_depth": min(
            (positions[item] for item in gold if item in positions), default=None
        ),
        "last_source_rank_within_fusion_depth": positions.get(
            base_blocks + source_blocks - 1
        ),
    }
    for topk in topks:
        selected = set(ranking[:topk])
        result[f"source_recall_at_{topk}"] = len(selected & gold) / len(gold)
        result[f"source_any_at_{topk}"] = bool(selected & gold)
        result[f"source_last_at_{topk}"] = base_blocks + source_blocks - 1 in selected
    return result


def exact_gold_ranks(
    scores: np.ndarray, base_blocks: int, source_blocks: int
) -> tuple[int, int]:
    scores = np.asarray(scores)
    ids = np.arange(len(scores), dtype=np.int64)
    ranks = []
    for gold_id in range(base_blocks, base_blocks + source_blocks):
        value = scores[gold_id]
        rank = 1 + int((scores > value).sum()) + int(((scores == value) & (ids < gold_id)).sum())
        ranks.append(rank)
    return min(ranks), ranks[-1]


def summarize(rows: list[dict[str, Any]], topks: list[int]) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
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
            "mean_query_seconds": mean(float(row["query_seconds"]) for row in group),
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
        groups.append(item)

    stability = []
    for memory_tokens in sorted({int(row["memory_tokens"]) for row in rows}):
        for method in sorted({str(row["method"]) for row in rows}):
            method_rows = {
                (int(row["query_id"]), int(row["prefix_tokens"])): row
                for row in rows
                if int(row["memory_tokens"]) == memory_tokens
                and str(row["method"]) == method
            }
            prefixes = sorted({prefix for _, prefix in method_rows})
            for left, right in zip(prefixes, prefixes[1:]):
                jaccards = []
                for query_id in sorted({query_id for query_id, _ in method_rows}):
                    left_set = set(method_rows[(query_id, left)]["top_block_ids"][:8])
                    right_set = set(method_rows[(query_id, right)]["top_block_ids"][:8])
                    jaccards.append(len(left_set & right_set) / len(left_set | right_set))
                stability.append(
                    {
                        "memory_tokens": memory_tokens,
                        "method": method,
                        "prefix_transition": f"{left}->{right}",
                        "top8_jaccard": mean(jaccards),
                    }
                )
    return {"retrieval_quality": groups, "prefix_stability": stability}


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    metadata = read_jsonl(data_dir / "metadata.jsonl")
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    source_blocks = np.load(data_dir / "source_blocks.npy", mmap_mode="r")
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    prefix_tokens = parse_ints(args.prefix_tokens)
    topks = parse_ints(args.topks)
    if max(prefix_tokens) > queries.shape[1]:
        raise ValueError("prefix length exceeds stored query")
    if max(topks) > args.fusion_depth:
        raise ValueError("fusion_depth must cover every requested top-k")

    qwen_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    base_texts = decode_blocks(qwen_tokenizer, base_blocks, args.decode_batch_size)
    source_texts = [
        decode_blocks(qwen_tokenizer, source_blocks[index], args.decode_batch_size)
        for index in range(len(source_blocks))
    ]
    query_texts = {
        (query_id, prefix): qwen_tokenizer.decode(
            np.asarray(queries[query_id, :prefix], dtype=np.int64).tolist(),
            skip_special_tokens=True,
        )
        for query_id in range(len(queries))
        for prefix in prefix_tokens
    }

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    e5_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name_or_path, use_fast=True)
    e5_model = AutoModel.from_pretrained(
        args.embedding_model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    e5_model.eval()

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    base_embeddings = encode_e5(
        e5_model,
        e5_tokenizer,
        base_texts,
        prefix="passage: ",
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        device=device,
    ).to(torch.float16)
    flat_sources = [text for group in source_texts for text in group]
    source_embeddings = encode_e5(
        e5_model,
        e5_tokenizer,
        flat_sources,
        prefix="passage: ",
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        device=device,
    ).reshape(len(source_texts), int(data_summary["source_blocks"]), -1).to(
        torch.float16
    )
    torch.cuda.synchronize(device)
    e5_passage_index_seconds = time.perf_counter() - started

    ordered_query_keys = sorted(query_texts)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    query_embeddings = encode_e5(
        e5_model,
        e5_tokenizer,
        [query_texts[key] for key in ordered_query_keys],
        prefix="query: ",
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        device=device,
    ).to(torch.float16)
    query_embedding_by_key = {
        key: query_embeddings[index] for index, key in enumerate(ordered_query_keys)
    }
    torch.cuda.synchronize(device)
    e5_query_embedding_batch_seconds = time.perf_counter() - started
    e5_query_embedding_seconds = e5_query_embedding_batch_seconds / len(
        ordered_query_keys
    )
    np.save(
        output_dir / "e5_base_embeddings_f16.npy",
        base_embeddings.cpu().numpy(),
    )

    rows: list[dict[str, Any]] = []
    index_rows = []
    block_tokens = int(data_summary["block_tokens"])
    source_block_count = int(data_summary["source_blocks"])
    for memory_tokens in data_summary["memory_scales_tokens"]:
        total_blocks = int(memory_tokens) // block_tokens
        base_count = total_blocks - source_block_count
        started = time.perf_counter()
        bm25 = BM25Index(
            base_texts[:base_count], min_df=1, max_df=1.0, k1=1.2, b=0.75
        )
        bm25_index_seconds = time.perf_counter() - started
        index_rows.append(
            {
                "memory_tokens": int(memory_tokens),
                "base_blocks": base_count,
                "bm25_index_seconds": bm25_index_seconds,
            }
        )
        for query_id in range(len(queries)):
            for prefix in prefix_tokens:
                query_text = query_texts[(query_id, prefix)]
                started = time.perf_counter()
                bm25_base_scores = bm25.score_postings([query_text])[0]
                bm25_source_scores = bm25_external_scores(
                    bm25,
                    source_texts[query_id],
                    query_text,
                    k1=1.2,
                    b=0.75,
                )
                bm25_scores = np.concatenate([bm25_base_scores, bm25_source_scores])
                bm25_ranking = top_indices(bm25_scores, args.fusion_depth)
                bm25_seconds = time.perf_counter() - started

                torch.cuda.synchronize(device)
                started = time.perf_counter()
                query_embedding = query_embedding_by_key[(query_id, prefix)]
                e5_base_scores = (
                    base_embeddings[:base_count].float() @ query_embedding.float()
                )
                e5_source_scores = (
                    source_embeddings[query_id].float() @ query_embedding.float()
                )
                e5_scores = torch.cat([e5_base_scores, e5_source_scores]).cpu().numpy()
                e5_ranking = top_indices(e5_scores, args.fusion_depth)
                torch.cuda.synchronize(device)
                e5_seconds = (
                    time.perf_counter() - started + e5_query_embedding_seconds
                )

                started = time.perf_counter()
                hybrid_ranking = reciprocal_rank_fusion(
                    [bm25_ranking, e5_ranking],
                    depth=args.fusion_depth,
                    rrf_k=args.rrf_k,
                )
                hybrid_seconds = time.perf_counter() - started
                for method, ranking, seconds, scores in (
                    ("bm25", bm25_ranking, bm25_seconds, bm25_scores),
                    ("e5", e5_ranking, e5_seconds, e5_scores),
                    (
                        "bm25_e5_rrf",
                        hybrid_ranking,
                        bm25_seconds + e5_seconds + hybrid_seconds,
                        None,
                    ),
                ):
                    row: dict[str, Any] = {
                        "query_id": query_id,
                        "memory_tokens": int(memory_tokens),
                        "memory_blocks": total_blocks,
                        "base_blocks": base_count,
                        "prefix_tokens": prefix,
                        "method": method,
                        "query_seconds": seconds,
                        "top_block_ids": ranking[: max(topks)],
                        "selection_uses_target": False,
                        **gold_metrics(
                            ranking, base_count, source_block_count, topks
                        ),
                    }
                    if scores is not None:
                        best_rank, last_rank = exact_gold_ranks(
                            scores, base_count, source_block_count
                        )
                        row["best_gold_rank"] = best_rank
                        row["last_source_rank"] = last_rank
                    rows.append(row)
        del bm25

    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": f"{data_summary['source']} dynamic text retrieval",
        "data_summary": data_summary,
        "methods": ["bm25", "e5", "bm25_e5_rrf"],
        "prefix_tokens": prefix_tokens,
        "topks": topks,
        "fusion_depth": args.fusion_depth,
        "e5_passage_index_seconds": e5_passage_index_seconds,
        "e5_query_embedding_batch_seconds": e5_query_embedding_batch_seconds,
        "e5_query_embedding_amortized_seconds": e5_query_embedding_seconds,
        "timing_note": "E5 query encoding is batched; per-query encoding is batch throughput, not batch-1 latency",
        "index_rows": index_rows,
        "contains_synthetic_text": False,
        "selection_uses_target": False,
        **summarize(rows, topks),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
