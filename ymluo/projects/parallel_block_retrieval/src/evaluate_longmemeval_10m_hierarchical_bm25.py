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
from transformers import AutoTokenizer

from evaluate_past_only_100m_hierarchical_bm25 import CompactBM25, rank_candidates
from evaluate_xsum_10m_dynamic_text_retrieval import decode_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate block, owner, session, and owner-to-session BM25 routing on "
            "the shared LongMemEval 10M memory."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--owner_depths", default="1,3,8")
    parser.add_argument("--session_depths", default="1,3,8,16,32")
    parser.add_argument("--semantic_recency_pools", default="8,16,32")
    parser.add_argument("--topks", default="8,32,128")
    parser.add_argument("--decode_batch_size", type=int, default=4096)
    parser.add_argument("--min_df", type=int, default=1)
    parser.add_argument("--max_df", type=float, default=0.995)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
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


def interval_blocks(start: int, end: int, block_tokens: int) -> np.ndarray:
    if end <= start:
        return np.empty(0, dtype=np.int64)
    return np.arange(start // block_tokens, (end - 1) // block_tokens + 1, dtype=np.int64)


def quota_union(primary: list[int], secondary: list[int], depth: int) -> list[int]:
    primary_quota = (depth + 1) // 2
    secondary_quota = depth - primary_quota
    output = []
    seen = set()
    for source, quota in ((primary, primary_quota), (secondary, secondary_quota)):
        if quota <= 0:
            continue
        added = 0
        for item in source:
            if item in seen:
                continue
            seen.add(item)
            output.append(item)
            added += 1
            if added >= quota:
                break
    for source in (primary, secondary):
        for item in source:
            if len(output) >= depth:
                return output
            if item in seen:
                continue
            seen.add(item)
            output.append(item)
            if len(output) >= depth:
                return output
    return output


def selection_metrics(
    ranking: list[int],
    *,
    query: dict[str, Any],
    block_session_rows: np.ndarray,
    block_owner_ids: np.ndarray,
    topks: list[int],
) -> dict[str, Any]:
    positives = set(map(int, query["positive_block_ids"]))
    latest_positives = set(map(int, query["latest_positive_block_ids"]))
    positive_sessions = set(map(int, query["positive_session_rows"]))
    hard_negatives = set(map(int, query["hard_negative_block_ids"]))
    hard_negative_sessions = set(map(int, query["hard_negative_session_rows"]))
    output: dict[str, Any] = {}
    for topk in topks:
        selected = ranking[:topk]
        selected_set = set(selected)
        selected_sessions = {
            int(block_session_rows[block_id])
            for block_id in selected
            if int(block_session_rows[block_id]) >= 0
        }
        output[f"owner_any_at_{topk}"] = any(
            int(block_owner_ids[block_id]) == int(query["owner_row"])
            for block_id in selected
        )
        output[f"owner_fraction_at_{topk}"] = mean(
            int(block_owner_ids[block_id]) == int(query["owner_row"])
            for block_id in selected
        )
        if positives:
            output[f"exact_block_any_at_{topk}"] = bool(positives & selected_set)
            output[f"exact_block_recall_at_{topk}"] = len(
                positives & selected_set
            ) / len(positives)
            output[f"latest_exact_block_any_at_{topk}"] = bool(
                latest_positives & selected_set
            )
            output[f"evidence_session_any_at_{topk}"] = bool(
                positive_sessions & selected_sessions
            )
            output[f"evidence_session_recall_at_{topk}"] = len(
                positive_sessions & selected_sessions
            ) / len(positive_sessions)
            output[f"all_evidence_sessions_at_{topk}"] = positive_sessions.issubset(
                selected_sessions
            )
        else:
            for name in (
                "exact_block_any",
                "exact_block_recall",
                "latest_exact_block_any",
                "evidence_session_any",
                "evidence_session_recall",
                "all_evidence_sessions",
            ):
                output[f"{name}_at_{topk}"] = None
        if hard_negatives:
            output[f"hard_negative_block_any_at_{topk}"] = bool(
                hard_negatives & selected_set
            )
            output[f"hard_negative_session_any_at_{topk}"] = bool(
                hard_negative_sessions & selected_sessions
            )
        else:
            output[f"hard_negative_block_any_at_{topk}"] = None
            output[f"hard_negative_session_any_at_{topk}"] = None
    return output


def summarize_group(
    group: list[dict[str, Any]], *, topks: list[int], block_tokens: int
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "queries": len(group),
        "mean_query_seconds": mean(float(row["query_seconds"]) for row in group),
        "mean_candidate_blocks": mean(float(row["candidate_blocks"]) for row in group),
        "mean_candidate_fraction": mean(
            float(row["candidate_fraction"]) for row in group
        ),
        "final_working_set_tokens_at_smallest_topk": min(topks) * block_tokens,
    }
    positive = [row for row in group if not row["is_abstention"]]
    abstention = [row for row in group if row["is_abstention"]]
    for topk in topks:
        for metric in (
            "owner_any",
            "owner_fraction",
            "exact_block_any",
            "exact_block_recall",
            "latest_exact_block_any",
            "evidence_session_any",
            "evidence_session_recall",
            "all_evidence_sessions",
        ):
            source = group if metric.startswith("owner_") else positive
            key = f"{metric}_at_{topk}"
            item[key] = mean(float(row[key]) for row in source) if source else None
        for metric in ("hard_negative_block_any", "hard_negative_session_any"):
            key = f"{metric}_at_{topk}"
            item[key] = mean(float(row[key]) for row in abstention) if abstention else None
    return item


def summarize(
    rows: list[dict[str, Any]], *, topks: list[int], block_tokens: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quality = []
    by_type = []
    for method in sorted({str(row["method"]) for row in rows}):
        group = [row for row in rows if row["method"] == method]
        quality.append(
            {
                "method": method,
                **summarize_group(group, topks=topks, block_tokens=block_tokens),
            }
        )
        for question_type in sorted({str(row["question_type"]) for row in group}):
            typed = [row for row in group if row["question_type"] == question_type]
            by_type.append(
                {
                    "method": method,
                    "question_type": question_type,
                    **summarize_group(
                        typed, topks=topks, block_tokens=block_tokens
                    ),
                }
            )
    return quality, by_type


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    queries = read_jsonl(data_dir / "queries.jsonl")
    sessions = read_jsonl(data_dir / "session_manifest.jsonl")
    owners = read_jsonl(data_dir / "owner_manifest.jsonl")
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    block_owner_ids = np.asarray(
        np.load(data_dir / "base_block_owner_ids.npy", mmap_mode="r"), dtype=np.int64
    )
    block_session_rows = np.asarray(
        np.load(data_dir / "base_block_session_rows.npy", mmap_mode="r"),
        dtype=np.int64,
    )
    block_tokens = int(data_summary["block_tokens"])
    owner_depths = parse_ints(args.owner_depths)
    session_depths = parse_ints(args.session_depths)
    semantic_recency_pools = parse_ints(args.semantic_recency_pools)
    topks = parse_ints(args.topks)
    max_topk = max(topks)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    print(f"decoding {len(base_blocks):,} LongMemEval blocks", flush=True)
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

    owner_ids = [int(row["owner_row"]) for row in owners]
    owner_to_index = {owner_id: index for index, owner_id in enumerate(owner_ids)}
    owner_blocks = []
    owner_texts = []
    for owner in owners:
        block_ids = interval_blocks(
            int(owner["start_token"]), int(owner["end_token"]), block_tokens
        )
        owner_blocks.append(block_ids)
        owner_texts.append(" ".join(block_texts[int(item)] for item in block_ids))

    print(f"building {len(owner_texts):,}-owner BM25", flush=True)
    started = time.perf_counter()
    owner_index = CompactBM25(
        owner_texts, min_df=1, max_df=1.0, k1=args.k1, b=args.b
    )
    owner_index_seconds = time.perf_counter() - started

    session_blocks = []
    session_texts = []
    session_owner_indices = np.empty(len(sessions), dtype=np.int64)
    session_date_minutes = np.empty(len(sessions), dtype=np.int64)
    for session in sessions:
        block_ids = interval_blocks(
            int(session["start_token"]), int(session["end_token"]), block_tokens
        )
        session_blocks.append(block_ids)
        session_texts.append(" ".join(block_texts[int(item)] for item in block_ids))
        session_owner_indices[int(session["session_row"])] = owner_to_index[
            int(session["owner_row"])
        ]
        session_date_minutes[int(session["session_row"])] = int(
            session["date_minutes"]
        )
    del block_texts
    gc.collect()

    print(f"building {len(session_texts):,}-session BM25", flush=True)
    started = time.perf_counter()
    session_index = CompactBM25(
        session_texts, min_df=1, max_df=1.0, k1=args.k1, b=args.b
    )
    session_index_seconds = time.perf_counter() - started
    del session_texts, owner_texts
    gc.collect()

    all_block_ids = np.arange(len(base_blocks), dtype=np.int64)
    all_owner_indices = np.arange(len(owners), dtype=np.int64)
    all_session_rows = np.arange(len(sessions), dtype=np.int64)
    sessions_by_owner = [
        np.flatnonzero(session_owner_indices == owner_index).astype(np.int64)
        for owner_index in range(len(owners))
    ]
    rows = []

    def append_result(
        query: dict[str, Any],
        *,
        method: str,
        candidate_ids: np.ndarray,
        block_query: Any,
        base_query_seconds: float,
        precomputed_scores: np.ndarray | None = None,
        selected_owner_indices: list[int] | None = None,
        selected_session_rows: list[int] | None = None,
    ) -> None:
        candidate_ids = np.unique(candidate_ids)
        started_at = time.perf_counter()
        if precomputed_scores is None:
            candidate_scores = block_index.score_candidates(block_query, candidate_ids)
        else:
            candidate_scores = precomputed_scores[candidate_ids]
        ranking = rank_candidates(candidate_ids, candidate_scores, max_topk)
        query_seconds = base_query_seconds + time.perf_counter() - started_at
        rows.append(
            {
                "query_id": int(query["query_id"]),
                "question_id": str(query["question_id"]),
                "question_type": str(query["question_type"]),
                "is_abstention": bool(query["is_abstention"]),
                "method": method,
                "query_seconds": query_seconds,
                "candidate_blocks": len(candidate_ids),
                "candidate_fraction": len(candidate_ids) / len(base_blocks),
                "selected_owner_indices": selected_owner_indices or [],
                "selected_session_rows": selected_session_rows or [],
                "top_block_ids": ranking,
                "selection_uses_answer": False,
                **selection_metrics(
                    ranking,
                    query=query,
                    block_session_rows=block_session_rows,
                    block_owner_ids=block_owner_ids,
                    topks=topks,
                ),
            }
        )

    for query in queries:
        query_text = str(query["question"])
        block_query = block_index.query_vector(query_text)
        owner_query = owner_index.query_vector(query_text)
        session_query = session_index.query_vector(query_text)
        started = time.perf_counter()
        owner_scores = owner_index.score_postings(owner_query)
        owner_score_seconds = time.perf_counter() - started
        started = time.perf_counter()
        session_scores = session_index.score_postings(session_query)
        session_score_seconds = time.perf_counter() - started

        started = time.perf_counter()
        global_block_scores = block_index.score_postings(block_query)
        block_score_seconds = time.perf_counter() - started
        append_result(
            query,
            method="global_block_bm25",
            candidate_ids=all_block_ids,
            block_query=block_query,
            base_query_seconds=block_score_seconds,
            precomputed_scores=global_block_scores,
        )

        owner_index_true = owner_to_index[int(query["owner_row"])]
        append_result(
            query,
            method="owner_metadata_block_bm25",
            candidate_ids=owner_blocks[owner_index_true],
            block_query=block_query,
            base_query_seconds=0.0,
            selected_owner_indices=[owner_index_true],
        )

        ranked_owners = rank_candidates(
            all_owner_indices, owner_scores, max(owner_depths)
        )
        ranked_sessions = rank_candidates(
            all_session_rows, session_scores, max(session_depths)
        )
        owner_sessions = sessions_by_owner[owner_index_true]
        started = time.perf_counter()
        owner_session_scores = session_index.score_candidates(
            session_query, owner_sessions
        )
        owner_session_score_seconds = time.perf_counter() - started
        owner_session_ranking = rank_candidates(
            owner_sessions, owner_session_scores, max(session_depths)
        )
        eligible_recent_sessions = owner_sessions[
            session_date_minutes[owner_sessions]
            <= int(query["question_date_minutes"])
        ]
        recent_order = np.lexsort(
            (
                eligible_recent_sessions,
                -session_date_minutes[eligible_recent_sessions],
            )
        )
        recent_session_ranking = eligible_recent_sessions[recent_order].tolist()
        for owner_depth in owner_depths:
            selected_owners = ranked_owners[:owner_depth]
            candidates = np.concatenate(
                [owner_blocks[index] for index in selected_owners]
            )
            append_result(
                query,
                method=f"owner_router{owner_depth}_block_bm25",
                candidate_ids=candidates,
                block_query=block_query,
                base_query_seconds=owner_score_seconds,
                selected_owner_indices=selected_owners,
            )

        for session_depth in session_depths:
            selected_sessions = ranked_sessions[:session_depth]
            candidates = np.concatenate(
                [session_blocks[index] for index in selected_sessions]
            )
            append_result(
                query,
                method=f"session_router{session_depth}_block_bm25",
                candidate_ids=candidates,
                block_query=block_query,
                base_query_seconds=session_score_seconds,
                selected_session_rows=selected_sessions,
            )

            selected_owner_sessions = owner_session_ranking[:session_depth]
            owner_candidates = np.concatenate(
                [session_blocks[index] for index in selected_owner_sessions]
            )
            append_result(
                query,
                method=f"owner_metadata_session{session_depth}_block_bm25",
                candidate_ids=owner_candidates,
                block_query=block_query,
                base_query_seconds=owner_session_score_seconds,
                selected_owner_indices=[owner_index_true],
                selected_session_rows=selected_owner_sessions,
            )

            selected_recent_sessions = recent_session_ranking[:session_depth]
            recent_candidates = (
                np.concatenate(
                    [session_blocks[index] for index in selected_recent_sessions]
                )
                if selected_recent_sessions
                else np.empty(0, dtype=np.int64)
            )
            append_result(
                query,
                method=f"owner_metadata_recent{session_depth}_block_bm25",
                candidate_ids=recent_candidates,
                block_query=block_query,
                base_query_seconds=0.0,
                selected_owner_indices=[owner_index_true],
                selected_session_rows=selected_recent_sessions,
            )

            selected_hybrid_sessions = quota_union(
                owner_session_ranking, recent_session_ranking, session_depth
            )
            hybrid_candidates = np.concatenate(
                [session_blocks[index] for index in selected_hybrid_sessions]
            )
            append_result(
                query,
                method=f"owner_metadata_hybrid{session_depth}_block_bm25",
                candidate_ids=hybrid_candidates,
                block_query=block_query,
                base_query_seconds=owner_session_score_seconds,
                selected_owner_indices=[owner_index_true],
                selected_session_rows=selected_hybrid_sessions,
            )

        for pool_depth in semantic_recency_pools:
            semantic_pool = owner_session_ranking[:pool_depth]
            semantic_recent = sorted(
                semantic_pool,
                key=lambda item: (-session_date_minutes[item], item),
            )
            for session_depth in session_depths:
                if session_depth > pool_depth:
                    continue
                selected_temporal_sessions = semantic_recent[:session_depth]
                temporal_candidates = np.concatenate(
                    [session_blocks[index] for index in selected_temporal_sessions]
                )
                append_result(
                    query,
                    method=(
                        f"owner_metadata_semantic{pool_depth}_"
                        f"recent{session_depth}_block_bm25"
                    ),
                    candidate_ids=temporal_candidates,
                    block_query=block_query,
                    base_query_seconds=owner_session_score_seconds,
                    selected_owner_indices=[owner_index_true],
                    selected_session_rows=selected_temporal_sessions,
                )

        for owner_depth in owner_depths:
            selected_owners = ranked_owners[:owner_depth]
            routed_sessions = np.concatenate(
                [sessions_by_owner[index] for index in selected_owners]
            )
            started = time.perf_counter()
            routed_session_scores = session_index.score_candidates(
                session_query, routed_sessions
            )
            routed_session_score_seconds = time.perf_counter() - started
            routed_session_ranking = rank_candidates(
                routed_sessions, routed_session_scores, max(session_depths)
            )
            for session_depth in session_depths:
                selected_sessions = routed_session_ranking[:session_depth]
                candidates = np.concatenate(
                    [session_blocks[index] for index in selected_sessions]
                )
                append_result(
                    query,
                    method=(
                        f"owner_router{owner_depth}_session{session_depth}_block_bm25"
                    ),
                    candidate_ids=candidates,
                    block_query=block_query,
                    base_query_seconds=(
                        owner_score_seconds + routed_session_score_seconds
                    ),
                    selected_owner_indices=selected_owners,
                    selected_session_rows=selected_sessions,
                )
        print(f"completed query={query['query_id']}", flush=True)

    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    quality, by_type = summarize(rows, topks=topks, block_tokens=block_tokens)
    summary = {
        "source": "LongMemEval shared-10M hierarchical BM25 property study",
        "data_summary": data_summary,
        "protocol": {
            "selection_uses_answer": False,
            "positive_labels_used_only_for_metrics": True,
            "owner_metadata_is_tenant_scope_not_evidence_oracle": True,
            "abstention_answer_sessions_treated_as_hard_negatives": True,
            "final_working_set_tokens_at_top8": 8 * block_tokens,
        },
        "owner_depths": owner_depths,
        "session_depths": session_depths,
        "semantic_recency_pools": semantic_recency_pools,
        "topks": topks,
        "decode_seconds": decode_seconds,
        "block_index_seconds": block_index_seconds,
        "owner_index_seconds": owner_index_seconds,
        "session_index_seconds": session_index_seconds,
        "block_index_bytes": block_index.storage_bytes(),
        "owner_index_bytes": owner_index.storage_bytes(),
        "session_index_bytes": session_index.storage_bytes(),
        "quality": quality,
        "quality_by_question_type": by_type,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
