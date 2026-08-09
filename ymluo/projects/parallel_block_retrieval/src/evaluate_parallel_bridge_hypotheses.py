from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from transformers import AutoTokenizer

from analyze_branch_transition_verifier import choose_branch
from evaluate_global_step_hybrid_candidates import stable_rank
from prepare_verified_chained_answer_steps import (
    clean_generated_state,
    rewrite_answer_step_with_generated_state,
)
from profile_step_state_q import step_state_text
from run_iterative_condition_retrieval import BM25Index
from run_lexical_block_retrieval import decode_blocks
from run_single_query_dynamic_kv_generation import answer_hit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty-preserving parallel bridge-state retrieval."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--bridge_generation_rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--per_hypothesis_limit", type=int, default=16)
    parser.add_argument("--rrf_k", type=float, default=60.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rank_target(ranking: Sequence[int], target: int) -> int:
    try:
        return ranking.index(target) + 1
    except ValueError:
        return 0


def round_robin_fusion(
    rankings: Sequence[Sequence[int]], branch_order: Sequence[int]
) -> list[int]:
    fused: list[int] = []
    seen: set[int] = set()
    depth = max((len(ranking) for ranking in rankings), default=0)
    for rank in range(depth):
        for branch_index in branch_order:
            ranking = rankings[branch_index]
            if rank >= len(ranking):
                continue
            block_id = int(ranking[rank])
            if block_id not in seen:
                seen.add(block_id)
                fused.append(block_id)
    return fused


def equal_rrf_fusion(rankings: Sequence[Sequence[int]], rrf_k: float) -> list[int]:
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for ranking in rankings:
        for rank, value in enumerate(ranking, start=1):
            block_id = int(value)
            scores[block_id] = scores.get(block_id, 0.0) + 1.0 / (rrf_k + rank)
            best_rank[block_id] = min(best_rank.get(block_id, rank), rank)
    return sorted(scores, key=lambda item: (-scores[item], best_rank[item], item))


def summarize(rows: Sequence[dict[str, Any]], budgets: Sequence[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "queries": len(rows),
        "selected_bridge_hit_rate": statistics.fmean(
            bool(row["selected_bridge_hit"]) for row in rows
        ),
        "any_bridge_hit_rate": statistics.fmean(
            bool(row["any_bridge_hit"]) for row in rows
        ),
    }
    for method in ("selected", "round_robin", "rrf"):
        for budget in budgets:
            summary[f"{method}_recall_at_{budget}"] = statistics.fmean(
                0 < int(row[f"{method}_rank"]) <= budget for row in rows
            )
    for budget in (1, 3, 16):
        summary[f"any_hypothesis_recall_at_{budget}_each"] = statistics.fmean(
            0 < int(row["best_hypothesis_rank"]) <= budget for row in rows
        )
    for bridge_group in (False, True):
        group = [row for row in rows if bool(row["any_bridge_hit"]) == bridge_group]
        label = "any_bridge_hit" if bridge_group else "no_bridge_hit"
        summary[f"{label}_queries"] = len(group)
        for method in ("selected", "round_robin", "rrf"):
            summary[f"{label}_{method}_recall_at_3"] = (
                statistics.fmean(
                    0 < int(row[f"{method}_rank"]) <= 3 for row in group
                )
                if group
                else math.nan
            )
    return summary


def main() -> None:
    args = parse_args()
    if args.per_hypothesis_limit <= 0:
        raise ValueError("per_hypothesis_limit must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_steps = read_jsonl(Path(args.step_queries_path))
    bridge_steps = {
        int(row["query_id"]): row
        for row in all_steps
        if str(row["split"]) == args.split and row["step_type"] == "resolve_bridge"
    }
    answer_steps = {
        int(row["query_id"]): row
        for row in all_steps
        if str(row["split"]) == args.split
        and row["step_type"] == "resolve_answer_from_bridge"
    }
    generation_rows = [
        row
        for row in read_jsonl(Path(args.bridge_generation_rows_path))
        if str(row["split"]) == args.split and row["step_type"] == "resolve_bridge"
    ]
    generation_rows.sort(key=lambda row: int(row["query_id"]))
    if args.max_queries:
        generation_rows = generation_rows[: args.max_queries]

    hypothesis_steps: list[dict[str, Any]] = []
    query_metadata: list[dict[str, Any]] = []
    for generation in generation_rows:
        query_id = int(generation["query_id"])
        bridge_step = bridge_steps[query_id]
        answer_step = answer_steps[query_id]
        selected_index, score_trace = choose_branch(bridge_step, generation["branches"])
        branch_scores = [float(item["score"]) for item in score_trace]
        branch_order = sorted(
            range(len(branch_scores)), key=lambda index: (-branch_scores[index], index)
        )
        target_bridge = str(bridge_step["target_output"])
        states = []
        bridge_hits = []
        for branch_index, branch in enumerate(generation["branches"]):
            generated_state = clean_generated_state(str(branch["generated_text"]))
            if not generated_state:
                generated_state = "(empty generation)"
            rewritten = rewrite_answer_step_with_generated_state(
                answer_step, generated_state
            )
            hypothesis_steps.append(rewritten)
            states.append(generated_state)
            bridge_hits.append(answer_hit(generated_state, [target_bridge]))
        query_metadata.append(
            {
                "query_id": query_id,
                "target_block_id": int(answer_step["target_block_ids"][0]),
                "selected_index": selected_index,
                "branch_order": branch_order,
                "branch_scores": branch_scores,
                "score_trace": score_trace,
                "generated_states": states,
                "bridge_hits": bridge_hits,
            }
        )

    corpus_dir = Path(args.corpus_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    build_started = time.perf_counter()
    block_texts = decode_blocks(tokenizer, blocks)
    bm25 = BM25Index(block_texts, min_df=1, max_df=1.0, k1=1.2, b=0.75)
    build_seconds = time.perf_counter() - build_started
    query_texts = [step_state_text(step) for step in hypothesis_steps]
    search_started = time.perf_counter()
    score_matrix = bm25.score_postings(query_texts)
    search_seconds = time.perf_counter() - search_started

    branch_count = len(generation_rows[0]["branches"]) if generation_rows else 0
    rows = []
    for query_offset, metadata in enumerate(query_metadata):
        rankings = []
        for branch_index in range(branch_count):
            score_index = query_offset * branch_count + branch_index
            rankings.append(
                stable_rank(
                    score_matrix[score_index], args.per_hypothesis_limit
                )
            )
        target = int(metadata["target_block_id"])
        selected = rankings[int(metadata["selected_index"])]
        round_robin = round_robin_fusion(rankings, metadata["branch_order"])
        rrf = equal_rrf_fusion(rankings, args.rrf_k)
        union_bm25_scores = []
        for branch_index in range(branch_count):
            score_index = query_offset * branch_count + branch_index
            union_bm25_scores.append(
                [float(score_matrix[score_index, block_id]) for block_id in round_robin]
            )
        hypothesis_ranks = [rank_target(ranking, target) for ranking in rankings]
        positive_hypothesis_ranks = [rank for rank in hypothesis_ranks if rank > 0]
        rows.append(
            {
                **metadata,
                "selected_bridge_hit": bool(
                    metadata["bridge_hits"][int(metadata["selected_index"])]
                ),
                "any_bridge_hit": any(metadata["bridge_hits"]),
                "hypothesis_ranks": hypothesis_ranks,
                "hypothesis_candidates": rankings,
                "hypothesis_union_bm25_scores": union_bm25_scores,
                "best_hypothesis_rank": (
                    min(positive_hypothesis_ranks) if positive_hypothesis_ranks else 0
                ),
                "selected_rank": rank_target(selected, target),
                "round_robin_rank": rank_target(round_robin, target),
                "rrf_rank": rank_target(rrf, target),
                "selected_candidates": selected,
                "round_robin_candidates": round_robin,
                "rrf_candidates": rrf,
            }
        )

    budgets = [1, 3, 6, 9, 16]
    summary = {
        "source": "parallel retrieval over all generated bridge hypotheses",
        "selection_uses_gold": False,
        "num_blocks": int(blocks.shape[0]),
        "hypotheses_per_query": branch_count,
        "per_hypothesis_limit": args.per_hypothesis_limit,
        "bm25_build_seconds": build_seconds,
        "bm25_search_seconds": search_seconds,
        "mean_search_ms_per_query": 1000.0 * search_seconds / max(1, len(rows)),
        **summarize(rows, budgets),
    }
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
