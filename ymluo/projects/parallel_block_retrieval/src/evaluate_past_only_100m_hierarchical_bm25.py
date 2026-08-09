from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer

from evaluate_past_only_10m_text_retrieval import scope_metrics
from evaluate_xsum_10m_dynamic_text_retrieval import decode_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate global and hierarchical BM25 on nested past-only memories."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--state_suffix_tokens", default="64,128,256,512")
    parser.add_argument(
        "--query_end_offset_tokens",
        type=int,
        default=0,
        help="Exclude this many observed tokens at the end of each state from retrieval query text.",
    )
    parser.add_argument("--memory_scales_tokens", default="9900032,20000000,50000000,100000000")
    parser.add_argument("--scope_depths", default="1,3,8")
    parser.add_argument("--topks", default="8,64,512")
    parser.add_argument("--decode_batch_size", type=int, default=4096)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_df", type=float, default=0.995)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


def parse_ints(spec: str) -> list[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) <= 0:
        raise ValueError("integer list must contain positive values")
    return values


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


class CompactBM25:
    def __init__(
        self,
        documents: list[str],
        *,
        min_df: int,
        max_df: float,
        k1: float,
        b: float,
    ) -> None:
        self.vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 1),
            min_df=min_df,
            max_df=max_df,
            dtype=np.float32,
        )
        counts = self.vectorizer.fit_transform(documents).tocsr().astype(np.float32)
        document_count = int(counts.shape[0])
        document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        inverse_document_frequency = np.log1p(
            (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        ).astype(np.float32)
        lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        average_length = max(float(lengths.mean()), 1.0e-6)
        row_ids = np.repeat(np.arange(document_count), np.diff(counts.indptr))
        frequencies = counts.data
        denominator = frequencies + k1 * (
            1.0 - b + b * lengths[row_ids] / average_length
        )
        counts.data = (
            inverse_document_frequency[counts.indices]
            * frequencies
            * (k1 + 1.0)
            / denominator
        )
        self.weighted_documents = counts
        self.weighted_documents_csc = counts.tocsc()
        self.document_count = document_count

    def query_vector(self, text: str) -> sparse.csr_matrix:
        query = self.vectorizer.transform([text]).tocsr().astype(np.float32)
        query.data.fill(1.0)
        return query

    def score_postings(self, query: sparse.csr_matrix) -> np.ndarray:
        scores = np.zeros(self.document_count, dtype=np.float32)
        postings = self.weighted_documents_csc
        for feature_id in query.indices:
            start = postings.indptr[feature_id]
            end = postings.indptr[feature_id + 1]
            scores[postings.indices[start:end]] += postings.data[start:end]
        return scores

    def score_candidates(
        self, query: sparse.csr_matrix, candidate_ids: np.ndarray
    ) -> np.ndarray:
        if not len(candidate_ids):
            return np.empty(0, dtype=np.float32)
        return np.asarray(
            (query @ self.weighted_documents[candidate_ids].transpose()).toarray()[0],
            dtype=np.float32,
        )

    def storage_bytes(self) -> int:
        arrays = (
            self.weighted_documents.data,
            self.weighted_documents.indices,
            self.weighted_documents.indptr,
            self.weighted_documents_csc.data,
            self.weighted_documents_csc.indices,
            self.weighted_documents_csc.indptr,
        )
        return sum(int(item.nbytes) for item in arrays)


def rank_candidates(
    candidate_ids: np.ndarray, scores: np.ndarray, depth: int
) -> list[int]:
    if not len(candidate_ids):
        return []
    take = min(depth, len(candidate_ids))
    if take == len(candidate_ids):
        local = np.arange(len(candidate_ids), dtype=np.int64)
    else:
        local = np.argpartition(scores, -take)[-take:]
    local = local[np.lexsort((candidate_ids[local], -scores[local]))]
    return candidate_ids[local].astype(np.int64).tolist()


def scope_score_geometry(
    active_scope_ids: np.ndarray,
    active_scores: np.ndarray,
    *,
    query_scope_row: int,
    query_features: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    order = np.lexsort((active_scope_ids, -active_scores))
    ranked_scope_rows = active_scope_ids[order].astype(np.int64)
    ranked_scores = np.asarray(active_scores[order], dtype=np.float64)
    true_positions = np.flatnonzero(ranked_scope_rows == query_scope_row)
    true_scope_rank = int(true_positions[0]) + 1 if len(true_positions) else None
    positive = ranked_scores[ranked_scores > 0]
    positive_sum = max(float(positive.sum()), 1.0e-12)
    probabilities = positive / positive_sum if len(positive) else np.empty(0)
    if len(probabilities) > 1:
        normalized_entropy = float(
            -np.sum(probabilities * np.log(probabilities + 1.0e-30))
            / np.log(len(probabilities))
        )
    else:
        normalized_entropy = 0.0

    def score_at(rank: int) -> float:
        return float(ranked_scores[rank - 1]) if len(ranked_scores) >= rank else 0.0

    top1 = score_at(1)
    geometry = {
        "true_scope_rank": true_scope_rank,
        "active_scopes": len(active_scope_ids),
        "positive_scope_scores": len(positive),
        "scope_query_features": query_features,
        "scope_top1_score": top1,
        "scope_margin_1_2": top1 - score_at(2),
        "scope_margin_3_4": score_at(3) - score_at(4),
        "scope_margin_8_9": score_at(8) - score_at(9),
        "scope_margin_16_17": score_at(16) - score_at(17),
        "scope_margin_32_33": score_at(32) - score_at(33),
        "scope_normalized_margin_1_2": (top1 - score_at(2)) / max(abs(top1), 1.0e-12),
        "scope_top1_positive_share": float(positive[:1].sum() / positive_sum),
        "scope_top3_positive_share": float(positive[:3].sum() / positive_sum),
        "scope_top8_positive_share": float(positive[:8].sum() / positive_sum),
        "scope_top16_positive_share": float(positive[:16].sum() / positive_sum),
        "scope_top32_positive_share": float(positive[:32].sum() / positive_sum),
        "scope_score_normalized_entropy": normalized_entropy,
        "scope_score_hhi": float(np.sum(probabilities**2)) if len(probabilities) else 0.0,
        "scope_top1_z": float(
            (top1 - positive.mean()) / max(float(positive.std()), 1.0e-12)
        )
        if len(positive)
        else 0.0,
        "scope_top64_rows": ranked_scope_rows[:64].tolist(),
        "scope_top64_scores": ranked_scores[:64].astype(float).tolist(),
    }
    return ranked_scope_rows, geometry


def summarize(rows: list[dict[str, Any]], topks: list[int]) -> list[dict[str, Any]]:
    output = []
    keys = sorted(
        {
            (int(row["memory_tokens"]), int(row["prefix_tokens"]), str(row["method"]))
            for row in rows
        }
    )
    for memory_tokens, suffix, method in keys:
        group = [
            row
            for row in rows
            if int(row["memory_tokens"]) == memory_tokens
            and int(row["prefix_tokens"]) == suffix
            and str(row["method"]) == method
        ]
        item: dict[str, Any] = {
            "memory_tokens": memory_tokens,
            "state_suffix_tokens": suffix,
            "method": method,
            "queries": len(group),
            "mean_query_seconds": mean(float(row["query_seconds"]) for row in group),
            "mean_candidate_blocks": mean(float(row["candidate_blocks"]) for row in group),
            "mean_candidate_fraction": mean(
                float(row["candidate_fraction"]) for row in group
            ),
            "scope_router_recall": mean(float(row["scope_router_hit"]) for row in group),
        }
        for topk in topks:
            for metric in ("same_scope_any", "same_scope_fraction"):
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
        raise ValueError("requires past-only data without predefined source blocks")

    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    block_scope_ids = np.asarray(
        np.load(data_dir / "base_block_scope_ids.npy", mmap_mode="r"), dtype=np.int64
    )
    block_original_centers = np.load(
        data_dir / "base_block_original_centers.npy", mmap_mode="r"
    )
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    metadata = {row["query_id"]: row for row in read_jsonl(data_dir / "metadata.jsonl")}
    suffixes = parse_ints(args.state_suffix_tokens)
    scales = parse_ints(args.memory_scales_tokens)
    scope_depths = parse_ints(args.scope_depths)
    topks = parse_ints(args.topks)
    block_tokens = int(data_summary["block_tokens"])
    if max(scales) > len(base_blocks) * block_tokens:
        raise ValueError("requested scale exceeds stored memory")
    if any(scale % block_tokens for scale in scales):
        raise ValueError("every memory scale must be block aligned")
    if args.query_end_offset_tokens < 0:
        raise ValueError("query_end_offset_tokens cannot be negative")
    if min(suffixes) <= args.query_end_offset_tokens:
        raise ValueError("every state suffix must exceed query_end_offset_tokens")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    print(f"decoding {len(base_blocks):,} blocks", flush=True)
    started = time.perf_counter()
    block_texts = decode_blocks(tokenizer, base_blocks, args.decode_batch_size)
    decode_seconds = time.perf_counter() - started

    print("building 100M block BM25", flush=True)
    started = time.perf_counter()
    block_index = CompactBM25(
        block_texts,
        min_df=args.min_df,
        max_df=args.max_df,
        k1=args.k1,
        b=args.b,
    )
    block_index_seconds = time.perf_counter() - started

    valid_scopes = sorted({int(scope) for scope in block_scope_ids if int(scope) >= 0})
    scope_to_row = {scope: row for row, scope in enumerate(valid_scopes)}
    scope_parts: list[list[str]] = [[] for _ in valid_scopes]
    scope_block_lists: list[list[int]] = [[] for _ in valid_scopes]
    for block_id, (scope, text) in enumerate(zip(block_scope_ids, block_texts)):
        row = scope_to_row.get(int(scope))
        if row is not None:
            scope_parts[row].append(text)
            scope_block_lists[row].append(block_id)
    scope_texts = [" ".join(parts) for parts in scope_parts]
    scope_blocks = [np.asarray(items, dtype=np.int64) for items in scope_block_lists]
    del scope_parts, scope_block_lists, block_texts
    gc.collect()

    print(f"building {len(scope_texts):,}-scope BM25", flush=True)
    started = time.perf_counter()
    scope_index = CompactBM25(
        scope_texts,
        min_df=1,
        max_df=1.0,
        k1=args.k1,
        b=args.b,
    )
    scope_index_seconds = time.perf_counter() - started
    del scope_texts
    gc.collect()

    query_end = -args.query_end_offset_tokens if args.query_end_offset_tokens else None
    query_texts = {
        (query_id, suffix): tokenizer.decode(
            np.asarray(
                queries[query_id, -suffix:query_end], dtype=np.int64
            ).tolist(),
            skip_special_tokens=True,
        )
        for query_id in range(len(queries))
        for suffix in suffixes
    }
    max_topk = max(topks)
    rows = []
    for query_id in range(len(queries)):
        query_scope = int(metadata[query_id]["book_index"])
        local_start = int(metadata[query_id]["local_context_start_token"])
        query_scope_row = scope_to_row[query_scope]
        for suffix in suffixes:
            query_text = query_texts[(query_id, suffix)]
            query_vector = block_index.query_vector(query_text)
            scope_query_vector = scope_index.query_vector(query_text)

            started = time.perf_counter()
            global_scores = block_index.score_postings(query_vector)
            global_score_seconds = time.perf_counter() - started
            started = time.perf_counter()
            scope_scores = scope_index.score_postings(scope_query_vector)
            scope_score_seconds = time.perf_counter() - started

            for scale in scales:
                scale_blocks = scale // block_tokens
                active_scope_mask = np.asarray(
                    [len(ids) > 0 and int(ids[0]) < scale_blocks for ids in scope_blocks],
                    dtype=bool,
                )
                active_scope_ids = np.flatnonzero(active_scope_mask)
                active_scores = scope_scores[active_scope_ids]
                ranked_scope_rows, scope_geometry = scope_score_geometry(
                    active_scope_ids,
                    active_scores,
                    query_scope_row=query_scope_row,
                    query_features=int(scope_query_vector.nnz),
                )

                started = time.perf_counter()
                global_ranking = rank_candidates(
                    np.arange(scale_blocks, dtype=np.int64),
                    global_scores[:scale_blocks],
                    max_topk,
                )
                global_seconds = global_score_seconds + time.perf_counter() - started
                rows.append(
                    {
                        "query_id": query_id,
                        "memory_tokens": scale,
                        "memory_blocks": scale_blocks,
                        "prefix_tokens": suffix,
                        "query_end_offset_tokens": args.query_end_offset_tokens,
                        "query_tokens_used": suffix - args.query_end_offset_tokens,
                        "method": "global_bm25_unigram",
                        "query_seconds": global_seconds,
                        "candidate_blocks": scale_blocks,
                        "candidate_fraction": 1.0,
                        "scope_router_hit": True,
                        "selected_scope_rows": [],
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

                for scope_depth in scope_depths:
                    take_scopes = min(scope_depth, len(active_scope_ids))
                    selected_scope_rows = ranked_scope_rows[:take_scopes]
                    candidate_ids = np.concatenate(
                        [scope_blocks[row] for row in selected_scope_rows]
                    )
                    candidate_ids = candidate_ids[candidate_ids < scale_blocks]
                    started = time.perf_counter()
                    candidate_scores = block_index.score_candidates(
                        query_vector, candidate_ids
                    )
                    ranking = rank_candidates(candidate_ids, candidate_scores, max_topk)
                    query_seconds = scope_score_seconds + time.perf_counter() - started
                    rows.append(
                        {
                            "query_id": query_id,
                            "memory_tokens": scale,
                            "memory_blocks": scale_blocks,
                            "prefix_tokens": suffix,
                            "query_end_offset_tokens": args.query_end_offset_tokens,
                            "query_tokens_used": suffix - args.query_end_offset_tokens,
                            "method": f"hier_bm25_scope{scope_depth}",
                            "query_seconds": query_seconds,
                            "candidate_blocks": len(candidate_ids),
                            "candidate_fraction": len(candidate_ids) / scale_blocks,
                            "scope_router_hit": query_scope_row in selected_scope_rows,
                            "selected_scope_rows": selected_scope_rows.tolist(),
                            "top_block_ids": ranking,
                            "selection_uses_target": False,
                            **scope_geometry,
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

                oracle_ids = scope_blocks[query_scope_row]
                oracle_ids = oracle_ids[oracle_ids < scale_blocks]
                started = time.perf_counter()
                oracle_scores = block_index.score_candidates(query_vector, oracle_ids)
                oracle_ranking = rank_candidates(oracle_ids, oracle_scores, max_topk)
                oracle_seconds = time.perf_counter() - started
                rows.append(
                    {
                        "query_id": query_id,
                        "memory_tokens": scale,
                        "memory_blocks": scale_blocks,
                        "prefix_tokens": suffix,
                        "query_end_offset_tokens": args.query_end_offset_tokens,
                        "query_tokens_used": suffix - args.query_end_offset_tokens,
                        "method": "oracle_scope_bm25",
                        "query_seconds": oracle_seconds,
                        "candidate_blocks": len(oracle_ids),
                        "candidate_fraction": len(oracle_ids) / scale_blocks,
                        "scope_router_hit": True,
                        "selected_scope_rows": [query_scope_row],
                        "top_block_ids": oracle_ranking,
                        "selection_uses_target": False,
                        **scope_metrics(
                            oracle_ranking,
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
        "source": "strict past-only PG19 nested-scale hierarchical BM25",
        "data_summary": data_summary,
        "protocol": {
            "contains_synthetic_text": False,
            "contains_repeated_distractor_text": False,
            "query_and_added_distractor_splits_disjoint": True,
            "selection_uses_target": False,
            "past_only": True,
            "predefined_source": False,
            "nested_scales_use_fixed_100m_idf": True,
            "query_end_offset_tokens": args.query_end_offset_tokens,
            "scope_type": "book",
            "final_working_set_tokens_at_top8": 8 * block_tokens,
        },
        "memory_scales_tokens": scales,
        "state_suffix_tokens": suffixes,
        "scope_depths": scope_depths,
        "topks": topks,
        "decode_seconds": decode_seconds,
        "block_index_seconds": block_index_seconds,
        "scope_index_seconds": scope_index_seconds,
        "block_index_bytes": block_index.storage_bytes(),
        "scope_index_bytes": scope_index.storage_bytes(),
        "block_features": len(block_index.vectorizer.vocabulary_),
        "scope_features": len(scope_index.vectorizer.vocabulary_),
        "retrieval_quality": summarize(rows, topks),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
