from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SpecialistAction:
    head_count: int
    depth: int
    aggregation: str
    bm25_quota: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile cross-dataset specialist heads and generate equal-budget natural "
            "LongBench KV block selections."
        )
    )
    parser.add_argument("--per_head_topk", required=True)
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument("--records_jsonl", required=True)
    parser.add_argument("--bm25_query_results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--gqa_group_size", type=int, default=2)
    parser.add_argument("--head_counts", default="4,8,16,32,64")
    parser.add_argument("--depths", default="1,2,4,8,16")
    parser.add_argument("--bm25_quotas", default="8,16,24,32")
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_bm25_rankings(path: Path) -> dict[int, list[int]]:
    rankings: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "bm25_record39":
                continue
            rankings[int(row["query_id"])] = [
                int(item) for item in json.loads(row["ranked_block_ids"])
            ]
    return rankings


def reciprocal_gold_rank(ranked: Sequence[int], gold: set[int]) -> float:
    for rank, block_id in enumerate(ranked, start=1):
        if int(block_id) in gold:
            return 1.0 / rank
    return 0.0


def compile_specialists(
    block_ids: np.ndarray,
    queries: Sequence[dict[str, Any]],
    train_indices: Sequence[int],
    gqa_group_size: int,
) -> list[tuple[float, int, int]]:
    if block_ids.ndim != 4:
        raise ValueError(f"expected [query, layer, head, depth], got {block_ids.shape}")
    layers, heads = block_ids.shape[1:3]
    utility = np.zeros((layers, heads), dtype=np.float64)
    for query_index in train_indices:
        gold = {int(item) for item in queries[query_index].get("gold_block_ids", [])}
        if not gold:
            continue
        for layer in range(layers):
            for head in range(heads):
                utility[layer, head] += reciprocal_gold_rank(
                    block_ids[query_index, layer, head].tolist(), gold
                )
    utility /= max(len(train_indices), 1)
    deduplicated: list[tuple[float, int, int]] = []
    for layer in range(layers):
        for first_head in range(0, heads, gqa_group_size):
            group = list(range(first_head, min(first_head + gqa_group_size, heads)))
            chosen = max(group, key=lambda head: (utility[layer, head], -head))
            deduplicated.append((float(utility[layer, chosen]), layer, chosen))
    deduplicated.sort(key=lambda item: (-item[0], item[1], item[2]))
    return deduplicated


def specialist_ranking(
    query_index: int,
    block_ids: np.ndarray,
    specialists: Sequence[tuple[float, int, int]],
    action: SpecialistAction,
) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    selected_heads = specialists[: action.head_count]
    for utility, layer, head in selected_heads:
        weight = max(float(utility), 1.0e-8)
        for rank, block_id in enumerate(
            block_ids[query_index, layer, head, : action.depth].tolist(), start=1
        ):
            block_id = int(block_id)
            contribution = (
                weight / (60.0 + rank)
                if action.aggregation == "weighted_rrf"
                else weight / rank
            )
            if action.aggregation == "weighted_rrf":
                scores[block_id] += contribution
            elif action.aggregation == "minority_max":
                scores[block_id] = max(scores[block_id], contribution)
            else:
                raise ValueError(f"unknown aggregation {action.aggregation}")
    return sorted(scores, key=lambda block_id: (-scores[block_id], block_id))


def combine_rankings(
    specialist: Sequence[int],
    bm25: Sequence[int],
    target_blocks: int,
    bm25_quota: int,
) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()

    def extend(source: Iterable[int], limit: int) -> None:
        for raw_block_id in source:
            if len(selected) >= limit:
                break
            block_id = int(raw_block_id)
            if block_id not in seen:
                seen.add(block_id)
                selected.append(block_id)

    extend(bm25, min(bm25_quota, target_blocks))
    extend(specialist, target_blocks)
    extend(bm25, target_blocks)
    return selected


def group_for_context(ranked: Sequence[int], block_to_record: np.ndarray) -> list[int]:
    groups: dict[int, list[int]] = defaultdict(list)
    record_order: list[int] = []
    for raw_block_id in ranked:
        block_id = int(raw_block_id)
        record_id = int(block_to_record[block_id])
        if record_id not in groups:
            record_order.append(record_id)
        groups[record_id].append(block_id)
    output: list[int] = []
    for record_id in record_order:
        output.extend(sorted(groups[record_id]))
    return output


def query_metrics(query: dict[str, Any], ranked: Sequence[int]) -> dict[str, float]:
    gold = {int(item) for item in query.get("gold_block_ids", [])}
    hits = [rank for rank, block_id in enumerate(ranked, start=1) if int(block_id) in gold]
    return {
        "answer_block_recall": float(bool(hits)),
        "evidence_fraction": len(gold & set(int(item) for item in ranked)) / max(len(gold), 1),
        "answer_block_mrr": 1.0 / min(hits) if hits else 0.0,
    }


def evaluate_action(
    action: SpecialistAction,
    validation_indices: Sequence[int],
    specialists: Sequence[tuple[float, int, int]],
    block_ids: np.ndarray,
    queries: Sequence[dict[str, Any]],
    bm25: dict[int, list[int]],
    target_blocks: int,
) -> dict[str, float]:
    metrics: list[dict[str, float]] = []
    for query_index in validation_indices:
        specialist = specialist_ranking(query_index, block_ids, specialists, action)
        ranked = combine_rankings(
            specialist,
            bm25[int(queries[query_index]["query_id"])],
            target_blocks,
            action.bm25_quota,
        )
        metrics.append(query_metrics(queries[query_index], ranked))
    return {
        key: statistics.fmean(row[key] for row in metrics)
        for key in metrics[0]
    }


def tune_action(
    candidates: Sequence[SpecialistAction],
    inner_folds: Sequence[tuple[Sequence[int], Sequence[tuple[float, int, int]]]],
    block_ids: np.ndarray,
    queries: Sequence[dict[str, Any]],
    bm25: dict[int, list[int]],
    target_blocks: int,
) -> tuple[SpecialistAction, dict[str, float]]:
    candidate_scores: list[tuple[tuple[float, ...], SpecialistAction, dict[str, float]]] = []
    for action in candidates:
        fold_metrics: list[dict[str, float]] = []
        for inner_test, specialists in inner_folds:
            fold_metrics.append(
                evaluate_action(
                    action,
                    inner_test,
                    specialists,
                    block_ids,
                    queries,
                    bm25,
                    target_blocks,
                )
            )
        mean_metrics = {
            key: statistics.fmean(row[key] for row in fold_metrics)
            for key in fold_metrics[0]
        }
        key = (
            -mean_metrics["evidence_fraction"],
            -mean_metrics["answer_block_recall"],
            -mean_metrics["answer_block_mrr"],
            action.bm25_quota,
            action.head_count,
            action.depth,
            0.0 if action.aggregation == "minority_max" else 1.0,
        )
        candidate_scores.append((key, action, mean_metrics))
    _, best_action, best_metrics = min(candidate_scores, key=lambda item: item[0])
    return best_action, best_metrics


def build_block_to_record(records: Sequence[dict[str, Any]], num_blocks: int) -> np.ndarray:
    mapping = np.full(num_blocks, -1, dtype=np.int32)
    for record_id, record in enumerate(records):
        start = int(record["block_start"])
        end = start + int(record["block_count"])
        mapping[start:end] = record_id
    if np.any(mapping < 0):
        raise ValueError("records do not cover every block")
    return mapping


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = read_jsonl(Path(args.queries_jsonl))
    queries.sort(key=lambda row: int(row["query_id"]))
    if [int(row["query_id"]) for row in queries] != list(range(len(queries))):
        raise ValueError("queries must have contiguous query IDs")
    records = read_jsonl(Path(args.records_jsonl))
    bm25 = load_bm25_rankings(Path(args.bm25_query_results))
    payload = np.load(args.per_head_topk)
    block_ids = np.asarray(payload["block_ids"], dtype=np.int32)
    if block_ids.shape[0] != len(queries):
        raise ValueError("per-head rankings and queries disagree")
    num_blocks = max(
        int(record["block_start"]) + int(record["block_count"]) for record in records
    )
    block_to_record = build_block_to_record(records, num_blocks)
    datasets = [str(query["dataset"]) for query in queries]
    head_counts = parse_int_list(args.head_counts)
    depths = parse_int_list(args.depths)
    quotas = parse_int_list(args.bm25_quotas)
    if max(depths) > block_ids.shape[-1]:
        raise ValueError("requested depth exceeds per-head Top-k")

    rrf_candidates = [
        SpecialistAction(head_count, depth, "weighted_rrf", 0)
        for head_count in head_counts
        for depth in depths
    ]
    max_candidates = [
        SpecialistAction(head_count, depth, "minority_max", 0)
        for head_count in head_counts
        for depth in depths
    ]
    hybrid_candidates = [
        SpecialistAction(head_count, depth, aggregation, quota)
        for head_count in head_counts
        for depth in depths
        for aggregation in ["weighted_rrf", "minority_max"]
        for quota in quotas
    ]
    families = {
        f"lodo_specialist_rrf{args.target_blocks}": rrf_candidates,
        f"lodo_specialist_max{args.target_blocks}": max_candidates,
        f"lodo_specialist_hybrid{args.target_blocks}": hybrid_candidates,
    }
    output_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for heldout in sorted(set(datasets)):
        outer_train = [index for index, dataset in enumerate(datasets) if dataset != heldout]
        outer_test = [index for index, dataset in enumerate(datasets) if dataset == heldout]
        specialists = compile_specialists(
            block_ids, queries, outer_train, args.gqa_group_size
        )
        inner_folds: list[tuple[Sequence[int], Sequence[tuple[float, int, int]]]] = []
        for inner_heldout in sorted({datasets[index] for index in outer_train}):
            inner_train = [
                index for index in outer_train if datasets[index] != inner_heldout
            ]
            inner_test = [
                index for index in outer_train if datasets[index] == inner_heldout
            ]
            inner_folds.append(
                (
                    inner_test,
                    compile_specialists(
                        block_ids, queries, inner_train, args.gqa_group_size
                    ),
                )
            )
        for method, candidates in families.items():
            action, inner_metrics = tune_action(
                candidates,
                inner_folds,
                block_ids,
                queries,
                bm25,
                args.target_blocks,
            )
            policy_rows.append(
                {
                    "heldout_dataset": heldout,
                    "method": method,
                    **asdict(action),
                    **{f"inner_{key}": value for key, value in inner_metrics.items()},
                    "top_specialists": json.dumps(
                        [
                            {"utility": utility, "layer": layer, "query_head": head}
                            for utility, layer, head in specialists[: action.head_count]
                        ]
                    ),
                }
            )
            for query_index in outer_test:
                query = queries[query_index]
                specialist = specialist_ranking(
                    query_index, block_ids, specialists, action
                )
                ranked = combine_rankings(
                    specialist,
                    bm25[int(query["query_id"])],
                    args.target_blocks,
                    action.bm25_quota,
                )
                context = group_for_context(ranked, block_to_record)
                metrics = query_metrics(query, ranked)
                block_start = int(query["block_start"])
                block_end = block_start + int(query["block_count"])
                output_rows.append(
                    {
                        "method": method,
                        "query_id": int(query["query_id"]),
                        "dataset": query["dataset"],
                        "source_record_recall": float(
                            any(block_start <= block_id < block_end for block_id in context)
                        ),
                        "record_top1_recall": 0.0,
                        "answer_block_recall": metrics["answer_block_recall"],
                        "answer_block_mrr": metrics["answer_block_mrr"],
                        "gold_block_count": len(query.get("gold_block_ids", [])),
                        "record_margin": 0.0,
                        "selected_block_ids": json.dumps(context),
                        "ranked_block_ids": json.dumps(ranked),
                    }
                )

    output_rows.sort(key=lambda row: (row["method"], int(row["query_id"])))
    fields = list(output_rows[0])
    with (output_dir / "query_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    policy_rows.sort(key=lambda row: (row["method"], row["heldout_dataset"]))
    with (output_dir / "lodo_policies.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(policy_rows[0]))
        writer.writeheader()
        writer.writerows(policy_rows)
    summaries: list[dict[str, Any]] = []
    for method in families:
        rows = [row for row in output_rows if row["method"] == method]
        summaries.append(
            {
                "method": method,
                "queries": len(rows),
                "source_record_recall": statistics.fmean(
                    float(row["source_record_recall"]) for row in rows
                ),
                "answer_block_recall": statistics.fmean(
                    float(row["answer_block_recall"]) for row in rows
                ),
                "answer_block_mrr": statistics.fmean(
                    float(row["answer_block_mrr"]) for row in rows
                ),
            }
        )
    summary = {
        "source": "nested leave-one-dataset-out GQA-deduplicated specialist-head retrieval",
        "queries": len(queries),
        "datasets": sorted(set(datasets)),
        "target_blocks": args.target_blocks,
        "gqa_group_size": args.gqa_group_size,
        "calibration_rule": (
            "Outer held-out dataset is never used to compile heads or tune actions; "
            "action hyperparameters are selected by inner leave-one-dataset-out retrieval."
        ),
        "methods": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
