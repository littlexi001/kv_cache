from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import binomtest
from transformers import AutoTokenizer

from evaluate_longmemeval_10m_hierarchical_bm25 import (
    interval_blocks,
    quota_union,
    selection_metrics,
)
from evaluate_past_only_100m_hierarchical_bm25 import CompactBM25, rank_candidates
from evaluate_xsum_10m_dynamic_text_retrieval import decode_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate question-prefix and generated-plan retrieval trajectories on a "
            "shared LongMemEval 10M memory."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--decode_tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--plan_tokenizer", default="Qwen/Qwen3-8B")
    parser.add_argument("--question_fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--plan_lengths", default="8,16,32,64")
    parser.add_argument("--owner_depth", type=int, default=8)
    parser.add_argument("--session_depth", type=int, default=3)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--rank_depth", type=int, default=40)
    parser.add_argument("--decode_batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def parse_floats(spec: str) -> list[float]:
    values = sorted({float(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) <= 0 or max(values) > 1:
        raise ValueError("question fractions must be in (0, 1]")
    return values


def parse_ints(spec: str) -> list[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) <= 0:
        raise ValueError("integer lists must contain positive values")
    return values


def question_prefix(question: str, fraction: float) -> str:
    words = question.split()
    take = max(1, int(math.ceil(len(words) * fraction)))
    return " ".join(words[:take])


def jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def paired_binary(
    baseline: list[dict[str, Any]], treatment: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    base = {int(row["query_id"]): row for row in baseline}
    new = {int(row["query_id"]): row for row in treatment}
    ids = sorted(set(base) & set(new))
    wins = sum(not bool(base[qid][metric]) and bool(new[qid][metric]) for qid in ids)
    losses = sum(bool(base[qid][metric]) and not bool(new[qid][metric]) for qid in ids)
    return {
        "queries": len(ids),
        "wins": wins,
        "losses": losses,
        "ties": len(ids) - wins - losses,
        "two_sided_binomial_p": (
            float(binomtest(wins, wins + losses, 0.5).pvalue)
            if wins + losses
            else 1.0
        ),
    }


def summarize(rows: list[dict[str, Any]], topk: int) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(str(row["method"]), str(row["state"])) for row in rows})
    for method, state in keys:
        raw_group = [
            row for row in rows if row["method"] == method and row["state"] == state
        ]
        excluded = sum(bool(row["answer_overlap_posthoc"]) for row in raw_group)
        group = [row for row in raw_group if not row["answer_overlap_posthoc"]]
        positive = [row for row in group if not row["is_abstention"]]
        abstention = [row for row in group if row["is_abstention"]]
        output.append(
            {
                "method": method,
                "state": state,
                "state_order": int(raw_group[0]["state_order"]),
                "queries": len(group),
                "positive_queries": len(positive),
                "answer_overlap_excluded_queries": excluded,
                "mean_candidate_blocks": mean(
                    float(row["candidate_blocks"]) for row in group
                ),
                "mean_candidate_tokens": mean(
                    float(row["candidate_tokens"]) for row in group
                ),
                "mean_query_seconds": mean(float(row["query_seconds"]) for row in group),
                f"exact_block_any_at_{topk}": mean(
                    float(row[f"exact_block_any_at_{topk}"]) for row in positive
                ),
                f"latest_exact_block_any_at_{topk}": mean(
                    float(row[f"latest_exact_block_any_at_{topk}"])
                    for row in positive
                ),
                f"evidence_session_recall_at_{topk}": mean(
                    float(row[f"evidence_session_recall_at_{topk}"])
                    for row in positive
                ),
                f"all_evidence_sessions_at_{topk}": mean(
                    float(row[f"all_evidence_sessions_at_{topk}"])
                    for row in positive
                ),
                f"hard_negative_block_any_at_{topk}": mean(
                    float(row[f"hard_negative_block_any_at_{topk}"])
                    for row in abstention
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if min(args.owner_depth, args.session_depth, args.topk) <= 0:
        raise ValueError("depths and topk must be positive")
    if args.rank_depth < args.topk:
        raise ValueError("rank_depth must be at least topk")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    queries = read_jsonl(data_dir / "queries.jsonl")
    trajectory_rows = read_jsonl(args.trajectories)
    trajectory_by_query = {int(row["query_id"]): row for row in trajectory_rows}
    if {int(row["query_id"]) for row in queries} != set(trajectory_by_query):
        raise ValueError("query ids in data and trajectories do not align")
    sessions = read_jsonl(data_dir / "session_manifest.jsonl")
    owners = read_jsonl(data_dir / "owner_manifest.jsonl")
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    block_owner_ids = np.asarray(
        np.load(data_dir / "base_block_owner_ids.npy", mmap_mode="r"), dtype=np.int64
    )
    block_session_rows = np.asarray(
        np.load(data_dir / "base_block_session_rows.npy", mmap_mode="r"), dtype=np.int64
    )
    block_tokens = int(data_summary["block_tokens"])
    question_fractions = parse_floats(args.question_fractions)
    plan_lengths = parse_ints(args.plan_lengths)

    decode_tokenizer = AutoTokenizer.from_pretrained(args.decode_tokenizer, use_fast=True)
    plan_tokenizer = AutoTokenizer.from_pretrained(args.plan_tokenizer, use_fast=True)
    print(f"decoding {len(base_blocks):,} blocks", flush=True)
    started = time.perf_counter()
    block_texts = decode_blocks(decode_tokenizer, base_blocks, args.decode_batch_size)
    decode_seconds = time.perf_counter() - started
    print("building block BM25", flush=True)
    started = time.perf_counter()
    block_index = CompactBM25(
        block_texts, min_df=1, max_df=0.995, k1=1.2, b=0.75
    )
    block_index_seconds = time.perf_counter() - started

    owner_ids = [int(row["owner_row"]) for row in owners]
    owner_to_index = {owner_id: index for index, owner_id in enumerate(owner_ids)}
    owner_blocks = []
    owner_texts = []
    for owner in owners:
        ids = interval_blocks(
            int(owner["start_token"]), int(owner["end_token"]), block_tokens
        )
        owner_blocks.append(ids)
        owner_texts.append(" ".join(block_texts[int(item)] for item in ids))
    started = time.perf_counter()
    owner_index = CompactBM25(owner_texts, min_df=1, max_df=1.0, k1=1.2, b=0.75)
    owner_index_seconds = time.perf_counter() - started

    session_blocks = []
    session_texts = []
    session_owner_indices = np.empty(len(sessions), dtype=np.int64)
    session_date_minutes = np.empty(len(sessions), dtype=np.int64)
    for session in sessions:
        ids = interval_blocks(
            int(session["start_token"]), int(session["end_token"]), block_tokens
        )
        session_blocks.append(ids)
        session_texts.append(" ".join(block_texts[int(item)] for item in ids))
        row = int(session["session_row"])
        session_owner_indices[row] = owner_to_index[int(session["owner_row"])]
        session_date_minutes[row] = int(session["date_minutes"])
    del block_texts, owner_texts
    gc.collect()
    print("building session BM25", flush=True)
    started = time.perf_counter()
    session_index = CompactBM25(
        session_texts, min_df=1, max_df=1.0, k1=1.2, b=0.75
    )
    session_index_seconds = time.perf_counter() - started
    del session_texts
    gc.collect()

    all_blocks = np.arange(len(base_blocks), dtype=np.int64)
    all_owners = np.arange(len(owners), dtype=np.int64)
    all_sessions = np.arange(len(sessions), dtype=np.int64)
    sessions_by_owner = [
        np.flatnonzero(session_owner_indices == index).astype(np.int64)
        for index in range(len(owners))
    ]
    states_by_query: dict[int, list[tuple[str, int, str]]] = {}
    for query in queries:
        qid = int(query["query_id"])
        question = str(query["question"])
        states = []
        order = 0
        for fraction in question_fractions:
            label = f"question_{int(round(fraction * 100)):03d}pct"
            states.append((label, order, question_prefix(question, fraction)))
            order += 1
        token_ids = list(map(int, trajectory_by_query[qid]["generated_token_ids"]))
        for length in plan_lengths:
            prefix = plan_tokenizer.decode(
                token_ids[:length], skip_special_tokens=True
            ).strip()
            states.append((f"question_full_plan_{length:03d}", order, f"{question}\n{prefix}"))
            order += 1
        states_by_query[qid] = states

    rows: list[dict[str, Any]] = []
    max_top = args.rank_depth

    def append_result(
        query: dict[str, Any],
        trajectory: dict[str, Any],
        state: str,
        state_order: int,
        method: str,
        candidates: np.ndarray,
        block_query: Any,
        base_seconds: float,
        *,
        selected_owners: list[int] | None = None,
        selected_sessions: list[int] | None = None,
        precomputed_scores: np.ndarray | None = None,
    ) -> None:
        candidates = np.unique(candidates)
        started_at = time.perf_counter()
        scores = (
            precomputed_scores[candidates]
            if precomputed_scores is not None
            else block_index.score_candidates(block_query, candidates)
        )
        ranking = rank_candidates(candidates, scores, max_top)
        elapsed = base_seconds + time.perf_counter() - started_at
        rows.append(
            {
                "query_id": int(query["query_id"]),
                "question_id": str(query["question_id"]),
                "question_type": str(query["question_type"]),
                "is_abstention": bool(query["is_abstention"]),
                "answer_overlap_posthoc": bool(trajectory["answer_overlap_posthoc"]),
                "state": state,
                "state_order": state_order,
                "method": method,
                "candidate_blocks": len(candidates),
                "candidate_tokens": len(candidates) * block_tokens,
                "query_seconds": elapsed,
                "selected_owner_indices": selected_owners or [],
                "selected_session_rows": selected_sessions or [],
                "top_block_ids": ranking,
                "selection_uses_answer": False,
                **selection_metrics(
                    ranking,
                    query=query,
                    block_session_rows=block_session_rows,
                    block_owner_ids=block_owner_ids,
                    topks=[args.topk],
                ),
            }
        )

    for query_index, query in enumerate(queries):
        qid = int(query["query_id"])
        trajectory = trajectory_by_query[qid]
        owner_true = owner_to_index[int(query["owner_row"])]
        owner_sessions = sessions_by_owner[owner_true]
        eligible_recent = owner_sessions[
            session_date_minutes[owner_sessions] <= int(query["question_date_minutes"])
        ]
        recent_order = np.lexsort((eligible_recent, -session_date_minutes[eligible_recent]))
        recent_ranking = eligible_recent[recent_order].tolist()
        for state, state_order, query_text in states_by_query[qid]:
            block_query = block_index.query_vector(query_text)
            owner_query = owner_index.query_vector(query_text)
            session_query = session_index.query_vector(query_text)

            started = time.perf_counter()
            global_scores = block_index.score_postings(block_query)
            global_seconds = time.perf_counter() - started
            append_result(
                query,
                trajectory,
                state,
                state_order,
                "global_block_bm25",
                all_blocks,
                block_query,
                global_seconds,
                precomputed_scores=global_scores,
            )

            started = time.perf_counter()
            global_session_scores = session_index.score_postings(session_query)
            global_session_seconds = time.perf_counter() - started
            global_sessions = rank_candidates(
                all_sessions, global_session_scores, args.session_depth
            )
            global_candidates = np.concatenate(
                [session_blocks[item] for item in global_sessions]
            )
            append_result(
                query,
                trajectory,
                state,
                state_order,
                f"global_session{args.session_depth}_block_bm25",
                global_candidates,
                block_query,
                global_session_seconds,
                selected_sessions=global_sessions,
            )

            started = time.perf_counter()
            owner_scores = owner_index.score_postings(owner_query)
            owner_seconds = time.perf_counter() - started
            routed_owners = rank_candidates(all_owners, owner_scores, args.owner_depth)
            routed_sessions = np.concatenate(
                [sessions_by_owner[item] for item in routed_owners]
            )
            started = time.perf_counter()
            routed_scores = session_index.score_candidates(session_query, routed_sessions)
            routed_seconds = time.perf_counter() - started
            selected_routed_sessions = rank_candidates(
                routed_sessions, routed_scores, args.session_depth
            )
            routed_candidates = np.concatenate(
                [session_blocks[item] for item in selected_routed_sessions]
            )
            append_result(
                query,
                trajectory,
                state,
                state_order,
                f"owner_router{args.owner_depth}_session{args.session_depth}_block_bm25",
                routed_candidates,
                block_query,
                owner_seconds + routed_seconds,
                selected_owners=routed_owners,
                selected_sessions=selected_routed_sessions,
            )

            started = time.perf_counter()
            owner_session_scores = session_index.score_candidates(
                session_query, owner_sessions
            )
            owner_session_seconds = time.perf_counter() - started
            semantic_sessions = rank_candidates(
                owner_sessions, owner_session_scores, args.session_depth
            )
            semantic_candidates = np.concatenate(
                [session_blocks[item] for item in semantic_sessions]
            )
            append_result(
                query,
                trajectory,
                state,
                state_order,
                f"owner_metadata_session{args.session_depth}_block_bm25",
                semantic_candidates,
                block_query,
                owner_session_seconds,
                selected_owners=[owner_true],
                selected_sessions=semantic_sessions,
            )

            hybrid_sessions = quota_union(
                semantic_sessions, recent_ranking, args.session_depth
            )
            hybrid_candidates = np.concatenate(
                [session_blocks[item] for item in hybrid_sessions]
            )
            append_result(
                query,
                trajectory,
                state,
                state_order,
                f"owner_metadata_hybrid{args.session_depth}_block_bm25",
                hybrid_candidates,
                block_query,
                owner_session_seconds,
                selected_owners=[owner_true],
                selected_sessions=hybrid_sessions,
            )
        print(f"completed query={qid} ({query_index + 1}/{len(queries)})", flush=True)

    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    positive_rows = [
        row
        for row in rows
        if not row["is_abstention"] and not row["answer_overlap_posthoc"]
    ]
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in positive_rows:
        by_key[(str(row["method"]), str(row["state"]))].append(row)
    full_state = "question_100pct"
    comparisons = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        baseline = by_key[(method, full_state)]
        for length in plan_lengths:
            state = f"question_full_plan_{length:03d}"
            treatment = by_key[(method, state)]
            base_map = {int(row["query_id"]): row for row in baseline}
            new_map = {int(row["query_id"]): row for row in treatment}
            ids = sorted(set(base_map) & set(new_map))
            comparisons.append(
                {
                    "method": method,
                    "baseline_state": full_state,
                    "treatment_state": state,
                    "mean_top_block_jaccard": mean(
                        jaccard(
                            base_map[qid]["top_block_ids"][: args.topk],
                            new_map[qid]["top_block_ids"][: args.topk],
                        )
                        for qid in ids
                    ),
                    f"paired_exact_block_any_at_{args.topk}": paired_binary(
                        baseline, treatment, f"exact_block_any_at_{args.topk}"
                    ),
                    f"paired_all_evidence_sessions_at_{args.topk}": paired_binary(
                        baseline, treatment, f"all_evidence_sessions_at_{args.topk}"
                    ),
                }
            )

    frontier = []
    trajectory_states = [full_state] + [
        f"question_full_plan_{length:03d}" for length in plan_lengths
    ]
    for method in methods:
        for query in queries:
            if query["is_abstention"]:
                continue
            qid = int(query["query_id"])
            if bool(trajectory_by_query[qid]["answer_overlap_posthoc"]):
                continue
            selected_rows = [
                by_key[(method, state)][
                    next(
                        index
                        for index, row in enumerate(by_key[(method, state)])
                        if int(row["query_id"]) == qid
                    )
                ]
                for state in trajectory_states
            ]
            union = sorted(
                {
                    int(block_id)
                    for row in selected_rows
                    for block_id in row["top_block_ids"][: args.topk]
                }
            )
            positives = set(map(int, query["positive_block_ids"]))
            positive_sessions = set(map(int, query["positive_session_rows"]))
            union_sessions = {
                int(block_session_rows[block_id])
                for block_id in union
                if int(block_session_rows[block_id]) >= 0
            }
            static_ranking = list(map(int, selected_rows[0]["top_block_ids"]))
            matched_k = min(len(union), len(static_ranking))
            matched_static = set(static_ranking[:matched_k])
            matched_static_sessions = {
                int(block_session_rows[block_id])
                for block_id in matched_static
                if int(block_session_rows[block_id]) >= 0
            }
            fixed16 = set(static_ranking[: min(16, len(static_ranking))])
            fixed16_sessions = {
                int(block_session_rows[block_id])
                for block_id in fixed16
                if int(block_session_rows[block_id]) >= 0
            }
            frontier.append(
                {
                    "method": method,
                    "query_id": qid,
                    "states": len(trajectory_states),
                    "unique_blocks": len(union),
                    "working_set_tokens": len(union) * block_tokens,
                    "exact_block_any": bool(positives & set(union)),
                    "all_evidence_sessions": positive_sessions.issubset(union_sessions),
                    "static_exact_block_any": bool(
                        selected_rows[0][f"exact_block_any_at_{args.topk}"]
                    ),
                    "static_all_evidence_sessions": bool(
                        selected_rows[0][f"all_evidence_sessions_at_{args.topk}"]
                    ),
                    "matched_static_blocks": matched_k,
                    "matched_static_exact_block_any": bool(
                        positives & matched_static
                    ),
                    "matched_static_all_evidence_sessions": positive_sessions.issubset(
                        matched_static_sessions
                    ),
                    "static_top16_exact_block_any": bool(positives & fixed16),
                    "static_top16_all_evidence_sessions": positive_sessions.issubset(
                        fixed16_sessions
                    ),
                }
            )
    frontier_summary = []
    for method in methods:
        group = [row for row in frontier if row["method"] == method]
        frontier_summary.append(
            {
                "method": method,
                "queries": len(group),
                "mean_unique_blocks": mean(float(row["unique_blocks"]) for row in group),
                "mean_working_set_tokens": mean(
                    float(row["working_set_tokens"]) for row in group
                ),
                "static_exact_block_any": mean(
                    float(row["static_exact_block_any"]) for row in group
                ),
                "trajectory_union_exact_block_any": mean(
                    float(row["exact_block_any"]) for row in group
                ),
                "static_all_evidence_sessions": mean(
                    float(row["static_all_evidence_sessions"]) for row in group
                ),
                "trajectory_union_all_evidence_sessions": mean(
                    float(row["all_evidence_sessions"]) for row in group
                ),
                "matched_static_exact_block_any": mean(
                    float(row["matched_static_exact_block_any"]) for row in group
                ),
                "matched_static_all_evidence_sessions": mean(
                    float(row["matched_static_all_evidence_sessions"]) for row in group
                ),
                "static_top16_exact_block_any": mean(
                    float(row["static_top16_exact_block_any"]) for row in group
                ),
                "static_top16_all_evidence_sessions": mean(
                    float(row["static_top16_all_evidence_sessions"]) for row in group
                ),
            }
        )

    with (output_dir / "frontier_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in frontier:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "LongMemEval shared-10M generated-state retrieval trajectory",
        "data_summary": data_summary,
        "protocol": {
            "memory_tokens": int(data_summary["memory_tokens"]),
            "selection_uses_answer": False,
            "answer_used_only_for_posthoc_overlap_exclusion": True,
            "owner_metadata_is_tenant_scope_not_evidence_oracle": True,
            "question_fractions": question_fractions,
            "plan_lengths": plan_lengths,
            "owner_depth": args.owner_depth,
            "session_depth": args.session_depth,
            "final_topk": args.topk,
            "stored_rank_depth": args.rank_depth,
            "per_state_working_set_tokens": args.topk * block_tokens,
        },
        "answer_overlap_queries": sum(
            bool(row["answer_overlap_posthoc"]) for row in trajectory_rows
        ),
        "decode_seconds": decode_seconds,
        "block_index_seconds": block_index_seconds,
        "owner_index_seconds": owner_index_seconds,
        "session_index_seconds": session_index_seconds,
        "block_index_bytes": block_index.storage_bytes(),
        "owner_index_bytes": owner_index.storage_bytes(),
        "session_index_bytes": session_index.storage_bytes(),
        "quality": summarize(rows, args.topk),
        "plan_vs_full_question": comparisons,
        "trajectory_frontier": frontier_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
