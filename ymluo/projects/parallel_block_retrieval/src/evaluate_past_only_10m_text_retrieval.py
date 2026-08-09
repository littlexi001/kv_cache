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
from transformers import AutoModel, AutoTokenizer

from evaluate_xsum_10m_dynamic_text_retrieval import (
    decode_blocks,
    reciprocal_rank_fusion,
    top_indices,
)
from evaluate_xsum_news_ppl_retrieval import encode_e5
from run_iterative_condition_retrieval import BM25Index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BM25/E5 retrieval from a real 10M past-only PG19 memory."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--embedding_model_name_or_path", default="intfloat/e5-base-v2")
    parser.add_argument("--state_suffix_tokens", default="64,128,256,512")
    parser.add_argument("--topks", default="8,64,512")
    parser.add_argument("--fusion_depth", type=int, default=512)
    parser.add_argument("--embedding_batch_size", type=int, default=128)
    parser.add_argument("--embedding_max_length", type=int, default=512)
    parser.add_argument("--decode_batch_size", type=int, default=2048)
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    return parser.parse_args()


def parse_ints(spec: str) -> list[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) <= 0:
        raise ValueError("integer list must contain positive values")
    return values


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def scope_metrics(
    ranking: list[int],
    *,
    query_scope: int,
    local_start: int,
    block_scope_ids: np.ndarray,
    block_original_centers: np.ndarray,
    topks: list[int],
) -> dict[str, Any]:
    same_positions = [
        rank
        for rank, block_id in enumerate(ranking, start=1)
        if int(block_scope_ids[block_id]) == query_scope
    ]
    result: dict[str, Any] = {
        "best_same_scope_rank": min(same_positions) if same_positions else None
    }
    for topk in topks:
        selected = np.asarray(ranking[:topk], dtype=np.int64)
        scopes = np.asarray(block_scope_ids[selected], dtype=np.int64)
        positions = np.asarray(block_original_centers[selected], dtype=np.int64)
        same = scopes == query_scope
        if np.any(same & (positions >= local_start)):
            raise RuntimeError("past-only invariant violated in retrieved blocks")
        distances = local_start - positions
        result[f"same_scope_any_at_{topk}"] = bool(np.any(same))
        result[f"same_scope_fraction_at_{topk}"] = float(np.mean(same))
        for threshold, label in ((4096, "4k"), (16384, "16k")):
            result[f"same_scope_within_{label}_any_at_{topk}"] = bool(
                np.any(same & (distances >= 0) & (distances <= threshold))
            )
    return result


def summarize(rows: list[dict[str, Any]], topks: list[int]) -> list[dict[str, Any]]:
    output = []
    for suffix in sorted({int(row["prefix_tokens"]) for row in rows}):
        for method in sorted({str(row["method"]) for row in rows}):
            group = [
                row
                for row in rows
                if int(row["prefix_tokens"]) == suffix and str(row["method"]) == method
            ]
            item: dict[str, Any] = {
                "state_suffix_tokens": suffix,
                "method": method,
                "queries": len(group),
                "mean_query_seconds": mean(float(row["query_seconds"]) for row in group),
                "mean_best_same_scope_rank": mean(
                    float(row["best_same_scope_rank"])
                    for row in group
                    if row["best_same_scope_rank"] is not None
                ),
            }
            for topk in topks:
                for metric in (
                    "same_scope_any",
                    "same_scope_fraction",
                    "same_scope_within_4k_any",
                    "same_scope_within_16k_any",
                ):
                    key = f"{metric}_at_{topk}"
                    item[key] = mean(float(row[key]) for row in group)
            output.append(item)
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    if not data_summary.get("past_only") or data_summary.get("source_blocks") != 0:
        raise ValueError("this evaluator requires a past-only dataset without source blocks")
    metadata = {
        int(row["query_id"]): row for row in read_jsonl(data_dir / "metadata.jsonl")
    }
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    block_scope_ids = np.load(data_dir / "base_block_scope_ids.npy", mmap_mode="r")
    block_original_centers = np.load(
        data_dir / "base_block_original_centers.npy", mmap_mode="r"
    )
    suffixes = parse_ints(args.state_suffix_tokens)
    topks = parse_ints(args.topks)
    if max(suffixes) > queries.shape[1]:
        raise ValueError("state suffix exceeds stored local context")
    if max(topks) > args.fusion_depth:
        raise ValueError("fusion_depth must cover every top-k")

    qwen_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    base_texts = decode_blocks(qwen_tokenizer, base_blocks, args.decode_batch_size)
    query_texts = {
        (query_id, suffix): qwen_tokenizer.decode(
            np.asarray(queries[query_id, -suffix:], dtype=np.int64).tolist(),
            skip_special_tokens=True,
        )
        for query_id in range(len(queries))
        for suffix in suffixes
    }

    started = time.perf_counter()
    bm25 = BM25Index(base_texts, min_df=1, max_df=1.0, k1=1.2, b=0.75)
    bm25_index_seconds = time.perf_counter() - started

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
        max_length=96,
        device=device,
    ).to(torch.float16)
    torch.cuda.synchronize(device)
    e5_passage_index_seconds = time.perf_counter() - started

    ordered_keys = sorted(query_texts)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    query_embeddings = encode_e5(
        e5_model,
        e5_tokenizer,
        [query_texts[key] for key in ordered_keys],
        prefix="query: ",
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        device=device,
    ).to(torch.float16)
    torch.cuda.synchronize(device)
    query_embedding_seconds = (time.perf_counter() - started) / len(ordered_keys)
    query_embedding_by_key = {
        key: query_embeddings[index] for index, key in enumerate(ordered_keys)
    }
    np.save(output_dir / "e5_base_embeddings_f16.npy", base_embeddings.cpu().numpy())

    rows = []
    for query_id in range(len(queries)):
        query_scope = int(metadata[query_id]["book_index"])
        local_start = int(metadata[query_id]["local_context_start_token"])
        for suffix in suffixes:
            key = (query_id, suffix)
            query_text = query_texts[key]
            started = time.perf_counter()
            bm25_scores = bm25.score_postings([query_text])[0]
            bm25_ranking = top_indices(bm25_scores, args.fusion_depth)
            bm25_seconds = time.perf_counter() - started

            torch.cuda.synchronize(device)
            started = time.perf_counter()
            e5_scores = base_embeddings.float() @ query_embedding_by_key[key].float()
            e5_ranking = top_indices(e5_scores.cpu().numpy(), args.fusion_depth)
            torch.cuda.synchronize(device)
            e5_seconds = time.perf_counter() - started + query_embedding_seconds

            started = time.perf_counter()
            hybrid_ranking = reciprocal_rank_fusion(
                [bm25_ranking, e5_ranking], depth=args.fusion_depth, rrf_k=args.rrf_k
            )
            hybrid_seconds = time.perf_counter() - started
            for method, ranking, seconds in (
                ("bm25", bm25_ranking, bm25_seconds),
                ("e5", e5_ranking, e5_seconds),
                ("bm25_e5_rrf", hybrid_ranking, bm25_seconds + e5_seconds + hybrid_seconds),
            ):
                rows.append(
                    {
                        "query_id": query_id,
                        "memory_tokens": int(data_summary["memory_tokens"]),
                        "memory_blocks": len(base_blocks),
                        "prefix_tokens": suffix,
                        "state_uses_recent_suffix": True,
                        "method": method,
                        "query_seconds": seconds,
                        "top_block_ids": ranking[: max(topks)],
                        "selection_uses_target": False,
                        **scope_metrics(
                            ranking,
                            query_scope=query_scope,
                            local_start=local_start,
                            block_scope_ids=block_scope_ids,
                            block_original_centers=block_original_centers,
                            topks=topks,
                        ),
                    }
                )

    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": f"{data_summary['source']} dynamic text retrieval",
        "data_summary": data_summary,
        "methods": ["bm25", "e5", "bm25_e5_rrf"],
        "state_suffix_tokens": suffixes,
        "topks": topks,
        "fusion_depth": args.fusion_depth,
        "bm25_index_seconds": bm25_index_seconds,
        "e5_passage_index_seconds": e5_passage_index_seconds,
        "e5_query_embedding_amortized_seconds": query_embedding_seconds,
        "contains_synthetic_text": False,
        "selection_uses_target": False,
        "past_only": True,
        "retrieval_quality": summarize(rows, topks),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
