from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_selected_head_debiased_retrieval import (
    read_selection,
    rrf_ranking,
)
from run_all_head_prior_debiased_retrieval import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify sparse selected-head rankings against a full scan."
    )
    parser.add_argument("--sparse_npz", required=True)
    parser.add_argument("--full_npz", required=True)
    parser.add_argument("--selection_csv", required=True)
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--gate_feature", default="raw_top1_block_diversity")
    parser.add_argument("--heads_per_fold", type=int, default=16)
    parser.add_argument("--query_heads_per_layer", type=int, default=16)
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--num_blocks", type=int, default=39062)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main() -> None:
    args = parse_args()
    selected_by_fold = read_selection(
        Path(args.selection_csv), args.gate_feature, args.heads_per_fold
    )
    queries = read_jsonl(Path(args.queries_jsonl))
    with np.load(Path(args.sparse_npz)) as sparse:
        sparse_ids = sparse["block_ids"]
        sparse_scores = sparse["scores"]
        flat_heads = sparse["flat_heads"].astype(np.int64)
        fold_ids = sparse["fold_ids"].astype(np.int64)
    with np.load(Path(args.full_npz)) as full:
        full_ids = full["block_ids"]
        full_scores = full["scores"]
        full_fold_ids = full["fold_ids"].astype(np.int64)
    if not np.array_equal(fold_ids, full_fold_ids):
        raise ValueError("fold assignments differ")
    if len(queries) != len(fold_ids):
        raise ValueError("query counts differ")
    sparse_position = {
        int(flat_head): index for index, flat_head in enumerate(flat_heads)
    }

    candidate_jaccards: list[float] = []
    ranking_overlaps: list[float] = []
    ranking_exact = 0
    ranking_set_exact = 0
    union_exact = 0
    sparse_hits = np.zeros(len(queries), dtype=bool)
    full_hits = np.zeros(len(queries), dtype=bool)
    for query_index, query in enumerate(queries):
        heads = selected_by_fold[int(fold_ids[query_index])]
        sparse_positions = [sparse_position[head] for head in heads]
        sparse_candidates = sparse_ids[query_index, sparse_positions]
        full_candidates = np.stack(
            [
                full_ids[
                    query_index,
                    head // args.query_heads_per_layer,
                    head % args.query_heads_per_layer,
                ]
                for head in heads
            ]
        )
        sparse_union = set(int(item) for item in sparse_candidates.reshape(-1))
        full_union = set(int(item) for item in full_candidates.reshape(-1))
        union_exact += sparse_union == full_union
        candidate_jaccards.append(
            len(sparse_union & full_union) / len(sparse_union | full_union)
        )
        sparse_rank = rrf_ranking(
            sparse_candidates, args.target_blocks, args.num_blocks
        )
        full_rank = rrf_ranking(
            full_candidates, args.target_blocks, args.num_blocks
        )
        ranking_exact += bool(np.array_equal(sparse_rank, full_rank))
        sparse_rank_set = set(int(item) for item in sparse_rank)
        full_rank_set = set(int(item) for item in full_rank)
        ranking_set_exact += sparse_rank_set == full_rank_set
        ranking_overlaps.append(
            len(sparse_rank_set & full_rank_set) / args.target_blocks
        )
        gold = np.asarray(query.get("gold_block_ids", []), dtype=np.int64)
        sparse_hits[query_index] = np.isin(gold, sparse_rank).any()
        full_hits[query_index] = np.isin(gold, full_rank).any()

    score_errors: list[float] = []
    id_mismatches = 0
    slots = 0
    for sparse_index, flat_head in enumerate(flat_heads):
        layer_index, query_head = divmod(
            int(flat_head), args.query_heads_per_layer
        )
        expected_scores = full_scores[:, layer_index, query_head]
        expected_ids = full_ids[:, layer_index, query_head]
        score_errors.append(
            float(
                np.max(
                    np.abs(
                        sparse_scores[:, sparse_index].astype(np.float64)
                        - expected_scores.astype(np.float64)
                    )
                )
            )
        )
        id_mismatches += int(
            np.sum(sparse_ids[:, sparse_index] != expected_ids)
        )
        slots += int(expected_ids.size)

    result: dict[str, Any] = {
        "queries": len(queries),
        "selected_union_heads": len(flat_heads),
        "per_head_topk_alignment": {
            "max_abs_score_error": max(score_errors),
            "id_mismatch_slots": id_mismatches,
            "id_mismatch_fraction": id_mismatches / slots,
        },
        "candidate_union_alignment": {
            "exact_queries": union_exact,
            "mean_jaccard": mean(candidate_jaccards),
            "min_jaccard": min(candidate_jaccards),
        },
        "rrf39_alignment": {
            "exact_order_queries": ranking_exact,
            "exact_set_queries": ranking_set_exact,
            "mean_set_overlap": mean(ranking_overlaps),
            "min_set_overlap": min(ranking_overlaps),
            "sparse_recall": float(sparse_hits.mean()),
            "full_recall": float(full_hits.mean()),
            "hit_disagreements": int(np.sum(sparse_hits != full_hits)),
        },
    }
    Path(args.output_json).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
