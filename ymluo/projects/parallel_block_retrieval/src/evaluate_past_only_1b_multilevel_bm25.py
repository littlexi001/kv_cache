from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from evaluate_past_only_100m_hierarchical_bm25 import (
    CompactBM25,
    decode_blocks,
    parse_ints,
    rank_candidates,
    read_jsonl,
    scope_metrics,
    scope_score_geometry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a real large PG19 book -> segment -> block BM25 index "
            "with a bounded final working set."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--state_suffix_tokens", default="64,128,256,512")
    parser.add_argument("--book_depths", default="8,32,128,512,1024")
    parser.add_argument("--segment_depths", default="8,32,128,512")
    parser.add_argument("--flat_book_depths", default="8,32")
    parser.add_argument("--topks", default="8,64,512")
    parser.add_argument("--segment_blocks", type=int, default=64)
    parser.add_argument("--decode_batch_size", type=int, default=4096)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_df", type=float, default=0.98)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--query_limit", type=int, default=0)
    return parser.parse_args()


def build_hierarchy(
    block_texts: list[str],
    block_scope_ids: np.ndarray,
    *,
    segment_blocks: int,
) -> tuple[
    list[int],
    dict[int, int],
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    list[np.ndarray],
    list[str],
    list[str],
]:
    valid_scopes = sorted({int(scope) for scope in block_scope_ids if int(scope) >= 0})
    scope_to_row = {scope: row for row, scope in enumerate(valid_scopes)}
    scope_blocks: list[list[int]] = [[] for _ in valid_scopes]
    for block_id, scope in enumerate(block_scope_ids):
        row = scope_to_row.get(int(scope))
        if row is not None:
            scope_blocks[row].append(block_id)

    segment_block_lists: list[np.ndarray] = []
    segment_scope_rows: list[int] = []
    segment_texts: list[str] = []
    scope_segment_lists: list[list[int]] = [[] for _ in valid_scopes]
    book_parts: list[list[str]] = [[] for _ in valid_scopes]
    run_start = 0
    while run_start < len(block_scope_ids):
        scope = int(block_scope_ids[run_start])
        run_end = run_start + 1
        while run_end < len(block_scope_ids) and int(block_scope_ids[run_end]) == scope:
            run_end += 1
        scope_row = scope_to_row.get(scope)
        if scope_row is not None:
            for start in range(run_start, run_end, segment_blocks):
                block_ids = np.arange(
                    start, min(start + segment_blocks, run_end), dtype=np.int64
                )
                segment_id = len(segment_block_lists)
                text = " ".join(block_texts[block_id] for block_id in block_ids)
                segment_block_lists.append(block_ids)
                segment_scope_rows.append(scope_row)
                segment_texts.append(text)
                scope_segment_lists[scope_row].append(segment_id)
                book_parts[scope_row].append(text)
        run_start = run_end
    scope_blocks_arrays = [np.asarray(ids, dtype=np.int64) for ids in scope_blocks]
    scope_segment_arrays = [
        np.asarray(ids, dtype=np.int64) for ids in scope_segment_lists
    ]
    return (
        valid_scopes,
        scope_to_row,
        scope_blocks_arrays,
        scope_segment_arrays,
        np.asarray(segment_scope_rows, dtype=np.int32),
        segment_block_lists,
        segment_texts,
        [" ".join(parts) for parts in book_parts],
    )


def hierarchy_metrics(
    selected_books: np.ndarray,
    selected_segments: list[int],
    *,
    query_scope_row: int,
    segment_scope_rows: np.ndarray,
    segment_block_lists: list[np.ndarray],
    block_original_centers: np.ndarray,
    local_start: int,
) -> dict[str, Any]:
    selected_segment_ids = np.asarray(selected_segments, dtype=np.int64)
    segment_book_hit = bool(
        len(selected_segment_ids)
        and np.any(segment_scope_rows[selected_segment_ids] == query_scope_row)
    )
    within_4k = False
    within_16k = False
    for segment_id in selected_segment_ids:
        if int(segment_scope_rows[segment_id]) != query_scope_row:
            continue
        centers = block_original_centers[segment_block_lists[segment_id]]
        if np.any(np.abs(centers - local_start) <= 4096):
            within_4k = True
        if np.any(np.abs(centers - local_start) <= 16384):
            within_16k = True
    return {
        "book_router_hit": bool(query_scope_row in selected_books),
        "segment_router_same_book_hit": segment_book_hit,
        "segment_router_same_book_within_4k_hit": within_4k,
        "segment_router_same_book_within_16k_hit": within_16k,
    }


def summarize(rows: list[dict[str, Any]], topks: list[int]) -> list[dict[str, Any]]:
    output = []
    for method in sorted({str(row["method"]) for row in rows}):
        for suffix in sorted({int(row["prefix_tokens"]) for row in rows}):
            group = [
                row
                for row in rows
                if row["method"] == method and int(row["prefix_tokens"]) == suffix
            ]
            if not group:
                continue
            item: dict[str, Any] = {
                "method": method,
                "state_suffix_tokens": suffix,
                "queries": len(group),
                "mean_query_seconds": mean(float(row["query_seconds"]) for row in group),
                "mean_candidate_books": mean(
                    float(row["candidate_books"]) for row in group
                ),
                "mean_candidate_segments": mean(
                    float(row["candidate_segments"]) for row in group
                ),
                "mean_candidate_blocks": mean(
                    float(row["candidate_blocks"]) for row in group
                ),
                "book_router_recall": mean(
                    float(row["book_router_hit"]) for row in group
                ),
                "segment_same_book_recall": mean(
                    float(row["segment_router_same_book_hit"]) for row in group
                ),
                "segment_within_4k_recall": mean(
                    float(row["segment_router_same_book_within_4k_hit"])
                    for row in group
                ),
                "segment_within_16k_recall": mean(
                    float(row["segment_router_same_book_within_16k_hit"])
                    for row in group
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


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    if not data_summary.get("past_only") or data_summary.get("source_blocks") != 0:
        raise ValueError("requires strict past-only data without predefined source")
    if data_summary.get("contains_synthetic_text") or data_summary.get(
        "contains_repeated_distractor_text"
    ):
        raise ValueError("requires non-repeated real text")

    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    block_scope_ids = np.asarray(
        np.load(data_dir / "base_block_scope_ids.npy", mmap_mode="r"), dtype=np.int32
    )
    block_original_centers = np.load(
        data_dir / "base_block_original_centers.npy", mmap_mode="r"
    )
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    metadata = {row["query_id"]: row for row in read_jsonl(data_dir / "metadata.jsonl")}
    suffixes = parse_ints(args.state_suffix_tokens)
    book_depths = parse_ints(args.book_depths)
    segment_depths = parse_ints(args.segment_depths)
    flat_book_depths = set(parse_ints(args.flat_book_depths))
    topks = parse_ints(args.topks)
    query_count = min(len(queries), args.query_limit) if args.query_limit else len(queries)
    max_topk = max(topks)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    print(f"decoding {len(base_blocks):,} blocks", flush=True)
    started = time.perf_counter()
    block_texts = decode_blocks(tokenizer, base_blocks, args.decode_batch_size)
    decode_seconds = time.perf_counter() - started

    print("building block BM25", flush=True)
    started = time.perf_counter()
    block_index = CompactBM25(
        block_texts,
        min_df=args.min_df,
        max_df=args.max_df,
        k1=args.k1,
        b=args.b,
    )
    block_index_seconds = time.perf_counter() - started

    print("building book and segment containers", flush=True)
    started = time.perf_counter()
    (
        valid_scopes,
        scope_to_row,
        scope_blocks,
        scope_segments,
        segment_scope_rows,
        segment_block_lists,
        segment_texts,
        book_texts,
    ) = build_hierarchy(
        block_texts,
        block_scope_ids,
        segment_blocks=args.segment_blocks,
    )
    hierarchy_build_seconds = time.perf_counter() - started
    del block_texts
    gc.collect()

    print(f"building {len(segment_texts):,}-segment BM25", flush=True)
    started = time.perf_counter()
    segment_index = CompactBM25(
        segment_texts,
        min_df=1,
        max_df=1.0,
        k1=args.k1,
        b=args.b,
    )
    segment_index_seconds = time.perf_counter() - started
    del segment_texts
    gc.collect()

    print(f"building {len(book_texts):,}-book BM25", flush=True)
    started = time.perf_counter()
    book_index = CompactBM25(
        book_texts,
        min_df=1,
        max_df=1.0,
        k1=args.k1,
        b=args.b,
    )
    book_index_seconds = time.perf_counter() - started
    del book_texts
    gc.collect()

    query_texts = {
        (query_id, suffix): tokenizer.decode(
            np.asarray(queries[query_id, -suffix:], dtype=np.int64).tolist(),
            skip_special_tokens=True,
        )
        for query_id in range(query_count)
        for suffix in suffixes
    }
    all_book_ids = np.arange(len(valid_scopes), dtype=np.int64)
    all_block_ids = np.arange(len(base_blocks), dtype=np.int64)
    rows: list[dict[str, Any]] = []

    for query_id in range(query_count):
        query_scope = int(metadata[query_id]["book_index"])
        query_scope_row = scope_to_row[query_scope]
        local_start = int(metadata[query_id]["local_context_start_token"])
        for suffix in suffixes:
            query_text = query_texts[(query_id, suffix)]
            block_query = block_index.query_vector(query_text)
            segment_query = segment_index.query_vector(query_text)
            book_query = book_index.query_vector(query_text)

            started = time.perf_counter()
            global_scores = block_index.score_postings(block_query)
            global_ranking = rank_candidates(all_block_ids, global_scores, max_topk)
            global_seconds = time.perf_counter() - started
            rows.append(
                {
                    "query_id": query_id,
                    "prefix_tokens": suffix,
                    "method": "global_bm25_unigram",
                    "query_seconds": global_seconds,
                    "candidate_books": len(valid_scopes),
                    "candidate_segments": len(segment_block_lists),
                    "candidate_blocks": len(base_blocks),
                    "book_router_hit": True,
                    "segment_router_same_book_hit": True,
                    "segment_router_same_book_within_4k_hit": True,
                    "segment_router_same_book_within_16k_hit": True,
                    "top_block_ids": global_ranking,
                    "selection_uses_target": False,
                    **scope_metrics(
                        global_ranking,
                        query_scope=query_scope,
                        local_start=local_start,
                        block_scope_ids=block_scope_ids,
                        block_original_centers=block_original_centers,
                        topks=topks,
                    ),
                }
            )

            started = time.perf_counter()
            book_scores = book_index.score_postings(book_query)
            book_ranking, book_geometry = scope_score_geometry(
                all_book_ids,
                book_scores,
                query_scope_row=query_scope_row,
                query_features=int(book_query.nnz),
            )
            book_score_seconds = time.perf_counter() - started

            for book_depth in book_depths:
                selected_books = book_ranking[: min(book_depth, len(book_ranking))]
                candidate_segment_ids = np.concatenate(
                    [scope_segments[book_id] for book_id in selected_books]
                )
                started = time.perf_counter()
                candidate_segment_scores = segment_index.score_candidates(
                    segment_query, candidate_segment_ids
                )
                max_segment_depth = min(max(segment_depths), len(candidate_segment_ids))
                segment_ranking = rank_candidates(
                    candidate_segment_ids,
                    candidate_segment_scores,
                    max_segment_depth,
                )
                segment_score_seconds = time.perf_counter() - started

                for segment_depth in segment_depths:
                    selected_segments = segment_ranking[:segment_depth]
                    candidate_block_ids = np.concatenate(
                        [segment_block_lists[segment_id] for segment_id in selected_segments]
                    )
                    started = time.perf_counter()
                    candidate_block_scores = block_index.score_candidates(
                        block_query, candidate_block_ids
                    )
                    ranking = rank_candidates(
                        candidate_block_ids, candidate_block_scores, max_topk
                    )
                    block_score_seconds = time.perf_counter() - started
                    rows.append(
                        {
                            "query_id": query_id,
                            "prefix_tokens": suffix,
                            "method": (
                                f"multilevel_bm25_book{book_depth}_segment{segment_depth}"
                            ),
                            "query_seconds": (
                                book_score_seconds
                                + segment_score_seconds
                                + block_score_seconds
                            ),
                            "candidate_books": len(selected_books),
                            "candidate_segments": len(candidate_segment_ids),
                            "selected_segments": len(selected_segments),
                            "candidate_blocks": len(candidate_block_ids),
                            "top_block_ids": ranking,
                            "selection_uses_target": False,
                            **book_geometry,
                            **hierarchy_metrics(
                                selected_books,
                                selected_segments,
                                query_scope_row=query_scope_row,
                                segment_scope_rows=segment_scope_rows,
                                segment_block_lists=segment_block_lists,
                                block_original_centers=block_original_centers,
                                local_start=local_start,
                            ),
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

                if book_depth in flat_book_depths:
                    candidate_block_ids = np.concatenate(
                        [scope_blocks[book_id] for book_id in selected_books]
                    )
                    started = time.perf_counter()
                    candidate_block_scores = block_index.score_candidates(
                        block_query, candidate_block_ids
                    )
                    ranking = rank_candidates(
                        candidate_block_ids, candidate_block_scores, max_topk
                    )
                    block_score_seconds = time.perf_counter() - started
                    rows.append(
                        {
                            "query_id": query_id,
                            "prefix_tokens": suffix,
                            "method": f"flat_book_bm25_depth{book_depth}",
                            "query_seconds": book_score_seconds + block_score_seconds,
                            "candidate_books": len(selected_books),
                            "candidate_segments": len(candidate_segment_ids),
                            "candidate_blocks": len(candidate_block_ids),
                            "book_router_hit": bool(query_scope_row in selected_books),
                            "segment_router_same_book_hit": bool(
                                query_scope_row in selected_books
                            ),
                            "segment_router_same_book_within_4k_hit": bool(
                                query_scope_row in selected_books
                            ),
                            "segment_router_same_book_within_16k_hit": bool(
                                query_scope_row in selected_books
                            ),
                            "top_block_ids": ranking,
                            "selection_uses_target": False,
                            **book_geometry,
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
            print(f"completed query={query_id} suffix={suffix}", flush=True)

    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "real strict past-only PG19 multilevel BM25",
        "data_summary": data_summary,
        "protocol": {
            "contains_synthetic_text": False,
            "contains_repeated_distractor_text": False,
            "past_only": True,
            "predefined_source": False,
            "selection_uses_target": False,
            "hierarchy": "book -> fixed contiguous segment -> block",
            "segment_blocks": args.segment_blocks,
            "segment_tokens": args.segment_blocks * int(data_summary["block_tokens"]),
            "final_working_set_tokens_at_top8": 8
            * int(data_summary["block_tokens"]),
        },
        "memory_tokens": int(data_summary["memory_tokens"]),
        "memory_blocks": len(base_blocks),
        "books": len(valid_scopes),
        "segments": len(segment_block_lists),
        "segment_noncontiguous_violations": sum(
            int(len(ids) > 1 and np.any(np.diff(ids) != 1))
            for ids in segment_block_lists
        ),
        "state_suffix_tokens": suffixes,
        "book_depths": book_depths,
        "segment_depths": segment_depths,
        "flat_book_depths": sorted(flat_book_depths),
        "topks": topks,
        "query_count": query_count,
        "decode_seconds": decode_seconds,
        "hierarchy_build_seconds": hierarchy_build_seconds,
        "block_index_seconds": block_index_seconds,
        "segment_index_seconds": segment_index_seconds,
        "book_index_seconds": book_index_seconds,
        "block_index_bytes": block_index.storage_bytes(),
        "segment_index_bytes": segment_index.storage_bytes(),
        "book_index_bytes": book_index.storage_bytes(),
        "retrieval_quality": summarize(rows, topks),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
