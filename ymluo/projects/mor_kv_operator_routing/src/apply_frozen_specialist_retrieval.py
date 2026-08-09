from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from run_lodo_natural_specialist_retrieval import (
    SpecialistAction,
    build_block_to_record,
    combine_rankings,
    compile_specialists,
    group_for_context,
    load_bm25_rankings,
    parse_int_list,
    query_metrics,
    read_jsonl,
    specialist_ranking,
    tune_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a specialist-head policy on calibration datasets and apply it once "
            "to a disjoint-query holdout sharing the same frozen KV block index."
        )
    )
    parser.add_argument("--calibration_per_head_topk", required=True)
    parser.add_argument("--calibration_queries_jsonl", required=True)
    parser.add_argument("--calibration_bm25_query_results", required=True)
    parser.add_argument("--target_per_head_topk", required=True)
    parser.add_argument("--target_queries_jsonl", required=True)
    parser.add_argument("--target_records_jsonl", required=True)
    parser.add_argument("--target_bm25_query_results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--gqa_group_size", type=int, default=2)
    parser.add_argument("--head_counts", default="4,8,16,32,64")
    parser.add_argument("--depths", default="1,2,4,8,16")
    parser.add_argument("--bm25_quotas", default="8,16,24,32")
    return parser.parse_args()


def validate_contiguous_queries(queries: list[dict[str, Any]]) -> None:
    queries.sort(key=lambda row: int(row["query_id"]))
    if [int(row["query_id"]) for row in queries] != list(range(len(queries))):
        raise ValueError("queries must have contiguous IDs")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_queries = read_jsonl(Path(args.calibration_queries_jsonl))
    target_queries = read_jsonl(Path(args.target_queries_jsonl))
    validate_contiguous_queries(calibration_queries)
    validate_contiguous_queries(target_queries)
    calibration_blocks = np.asarray(
        np.load(args.calibration_per_head_topk)["block_ids"], dtype=np.int32
    )
    target_blocks = np.asarray(
        np.load(args.target_per_head_topk)["block_ids"], dtype=np.int32
    )
    if calibration_blocks.shape[0] != len(calibration_queries):
        raise ValueError("calibration rankings and queries disagree")
    if target_blocks.shape[0] != len(target_queries):
        raise ValueError("target rankings and queries disagree")
    if calibration_blocks.shape[1:] != target_blocks.shape[1:]:
        raise ValueError("calibration and target head ranking shapes disagree")
    calibration_uids = {str(query["record_uid"]) for query in calibration_queries}
    target_uids = {str(query["record_uid"]) for query in target_queries}
    overlap = calibration_uids & target_uids
    if overlap:
        raise ValueError(f"calibration/target query overlap: {len(overlap)}")

    calibration_datasets = [str(query["dataset"]) for query in calibration_queries]
    all_calibration = list(range(len(calibration_queries)))
    inner_folds: list[
        tuple[Sequence[int], Sequence[tuple[float, int, int]]]
    ] = []
    for heldout in sorted(set(calibration_datasets)):
        inner_train = [
            index for index in all_calibration if calibration_datasets[index] != heldout
        ]
        inner_test = [
            index for index in all_calibration if calibration_datasets[index] == heldout
        ]
        inner_folds.append(
            (
                inner_test,
                compile_specialists(
                    calibration_blocks,
                    calibration_queries,
                    inner_train,
                    args.gqa_group_size,
                ),
            )
        )
    calibration_bm25 = load_bm25_rankings(
        Path(args.calibration_bm25_query_results)
    )
    head_counts = parse_int_list(args.head_counts)
    depths = parse_int_list(args.depths)
    quotas = parse_int_list(args.bm25_quotas)
    candidates = [
        SpecialistAction(head_count, depth, aggregation, quota)
        for head_count in head_counts
        for depth in depths
        for aggregation in ["weighted_rrf", "minority_max"]
        for quota in quotas
    ]
    action, inner_metrics = tune_action(
        candidates,
        inner_folds,
        calibration_blocks,
        calibration_queries,
        calibration_bm25,
        args.target_blocks,
    )
    specialists = compile_specialists(
        calibration_blocks,
        calibration_queries,
        all_calibration,
        args.gqa_group_size,
    )
    target_bm25 = load_bm25_rankings(Path(args.target_bm25_query_results))
    records = read_jsonl(Path(args.target_records_jsonl))
    num_blocks = max(
        int(record["block_start"]) + int(record["block_count"]) for record in records
    )
    block_to_record = build_block_to_record(records, num_blocks)
    method = f"frozen_specialist_hybrid{args.target_blocks}"
    rows: list[dict[str, Any]] = []
    metrics: list[dict[str, float]] = []
    for query_index, query in enumerate(target_queries):
        specialist = specialist_ranking(
            query_index, target_blocks, specialists, action
        )
        ranked = combine_rankings(
            specialist,
            target_bm25[int(query["query_id"])],
            args.target_blocks,
            action.bm25_quota,
        )
        context = group_for_context(ranked, block_to_record)
        query_result = query_metrics(query, ranked)
        metrics.append(query_result)
        block_start = int(query["block_start"])
        block_end = block_start + int(query["block_count"])
        rows.append(
            {
                "method": method,
                "query_id": int(query["query_id"]),
                "dataset": query["dataset"],
                "source_record_recall": float(
                    any(block_start <= block_id < block_end for block_id in context)
                ),
                "record_top1_recall": 0.0,
                "answer_block_recall": query_result["answer_block_recall"],
                "answer_block_mrr": query_result["answer_block_mrr"],
                "gold_block_count": len(query.get("gold_block_ids", [])),
                "record_margin": 0.0,
                "selected_block_ids": json.dumps(context),
                "ranked_block_ids": json.dumps(ranked),
            }
        )
    with (output_dir / "query_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": "frozen calibration-only specialist-head policy on disjoint-query holdout",
        "calibration_queries": len(calibration_queries),
        "target_queries": len(target_queries),
        "query_uid_overlap": 0,
        "action": asdict(action),
        "inner_lodo_metrics": inner_metrics,
        "top_specialists": [
            {"utility": utility, "layer": layer, "query_head": head}
            for utility, layer, head in specialists[: action.head_count]
        ],
        "target_metrics": {
            key: statistics.fmean(row[key] for row in metrics) for key in metrics[0]
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
