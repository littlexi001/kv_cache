from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import fisher_exact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure document-level routing structure in PG19 utility candidates."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--text_rows", required=True)
    parser.add_argument("--base_block_book_ids", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--metadata_scope_key", default="book_index")
    parser.add_argument("--scope_label", default="book")
    parser.add_argument("--target_scopes_are_shortest", action="store_true")
    parser.add_argument("--memory_tokens", type=int, default=9_900_032)
    parser.add_argument("--prefix_tokens", type=int, default=64)
    parser.add_argument("--routing_depth", type=int, default=512)
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--bootstrap_samples", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def selection_quality(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    improvements = [float(row["delta_nll_b"]) for row in rows]
    nll = mean(float(row["mean_nll_b"]) for row in rows)
    return {
        "queries": len(rows),
        "mean_nll": nll,
        "ppl": math.exp(min(nll, 20.0)),
        "mean_nll_improvement": mean(improvements),
        "improvement_bootstrap95": bootstrap_ci(improvements, samples, seed),
        "positive_utility_rate": mean(value > 0 for value in improvements),
    }


def main() -> None:
    args = parse_args()
    metadata = read_jsonl(Path(args.data_dir) / "metadata.jsonl")
    rows = read_jsonl(args.rows)
    text_rows = [
        row
        for row in read_jsonl(args.text_rows)
        if int(row["memory_tokens"]) == args.memory_tokens
        and int(row["prefix_tokens"]) == args.prefix_tokens
    ]
    block_book_ids = np.load(args.base_block_book_ids, mmap_mode="r")
    base_count = len(block_book_ids)
    query_book = {
        int(row["query_id"]): int(row[args.metadata_scope_key]) for row in metadata
    }
    retrieval_methods = {"bm25", "e5", "bm25_e5_rrf"}
    for row in rows:
        query_id = int(row["query_id"])
        books = set()
        for block_id in row["block_ids"]:
            block_id = int(block_id)
            if block_id < base_count:
                book_id = int(block_book_ids[block_id])
                if book_id >= 0:
                    books.add(book_id)
            else:
                books.add(query_book[query_id])
        row["window_book_ids"] = sorted(books)
        row["same_book_as_query"] = query_book[query_id] in books

    query_ids = sorted(query_book)
    retrieval_by_query = {
        query_id: [
            row
            for row in rows
            if int(row["query_id"]) == query_id
            and retrieval_methods & set(row["origins"])
        ]
        for query_id in query_ids
    }
    random_by_query = {
        query_id: [
            row
            for row in rows
            if int(row["query_id"]) == query_id and "random" in row["origins"]
        ]
        for query_id in query_ids
    }

    candidate_same_rates = []
    random_same_rates = []
    event_same = 0
    event_other = 0
    nonevent_same = 0
    nonevent_other = 0
    per_query_same_utility_advantage = []
    best_a_same = []
    best_a_other = []
    best_a_all = []
    for query_id in query_ids:
        retrieval = retrieval_by_query[query_id]
        random_group = random_by_query[query_id]
        threshold = float(
            np.quantile([float(row["delta_nll_b"]) for row in random_group], 0.95)
        )
        candidate_same_rates.append(mean(bool(row["same_book_as_query"]) for row in retrieval))
        random_same_rates.append(mean(bool(row["same_book_as_query"]) for row in random_group))
        for row in retrieval:
            event = float(row["delta_nll_b"]) > threshold
            same = bool(row["same_book_as_query"])
            if event and same:
                event_same += 1
            elif event:
                event_other += 1
            elif same:
                nonevent_same += 1
            else:
                nonevent_other += 1
        same_group = [row for row in retrieval if bool(row["same_book_as_query"])]
        other_group = [row for row in retrieval if not bool(row["same_book_as_query"])]
        if same_group and other_group:
            per_query_same_utility_advantage.append(
                mean(float(row["delta_nll_b"]) for row in same_group)
                - mean(float(row["delta_nll_b"]) for row in other_group)
            )
            best_a_same.append(max(same_group, key=lambda row: float(row["delta_nll_a"])))
            best_a_other.append(max(other_group, key=lambda row: float(row["delta_nll_a"])))
        best_a_all.append(max(retrieval, key=lambda row: float(row["delta_nll_a"])))

    fisher = fisher_exact(
        [[event_same, event_other], [nonevent_same, nonevent_other]],
        alternative="greater",
    )
    paired_best_advantage = [
        float(other["mean_nll_b"]) - float(same["mean_nll_b"])
        for same, other in zip(best_a_same, best_a_other)
    ]
    method_top1 = []
    for method in sorted(retrieval_methods):
        chosen = []
        for query_id in query_ids:
            group = [row for row in retrieval_by_query[query_id] if method in row["origins"]]
            chosen.append(min(group, key=lambda row: int(row["origins"][method])))
        method_top1.append(
            {
                "method": method,
                "same_book_rate": mean(bool(row["same_book_as_query"]) for row in chosen),
                "quality": selection_quality(
                    chosen, args.bootstrap_samples, args.seed + len(method_top1)
                ),
            }
        )

    text_lookup = {
        (int(row["query_id"]), str(row["method"])): row for row in text_rows
    }
    valid_books, valid_counts = np.unique(
        np.asarray(block_book_ids)[np.asarray(block_book_ids) >= 0],
        return_counts=True,
    )
    book_block_counts = {
        int(book_id): int(count)
        for book_id, count in zip(valid_books, valid_counts)
    }
    source_count = int(
        json.loads((Path(args.data_dir) / "summary.json").read_text(encoding="utf-8"))[
            "source_blocks"
        ]
    )
    for scope_id in query_book.values():
        book_block_counts.setdefault(scope_id, source_count)
    document_routing = []
    for method in sorted(retrieval_methods):
        ranks_by_score = {
            "max reciprocal block-rank evidence": [],
            "sum reciprocal block-rank mass": [],
            "length-normalized reciprocal mass": [],
        }
        routed_blocks_by_score = {
            name: {1: [], 3: [], 5: []} for name in ranks_by_score
        }
        routed_book_blocks = []
        for query_id in query_ids:
            sum_scores: dict[int, float] = {}
            max_scores: dict[int, float] = {}
            ranking = text_lookup[(query_id, method)]["top_block_ids"][
                : args.routing_depth
            ]
            for rank, block_id in enumerate(ranking, start=1):
                block_id = int(block_id)
                book_id = (
                    int(block_book_ids[block_id])
                    if block_id < base_count
                    else query_book[query_id]
                )
                if book_id >= 0:
                    value = 1.0 / (args.rrf_k + rank)
                    sum_scores[book_id] = sum_scores.get(book_id, 0.0) + value
                    max_scores[book_id] = max(max_scores.get(book_id, 0.0), value)
            score_variants = {
                "max reciprocal block-rank evidence": max_scores,
                "sum reciprocal block-rank mass": sum_scores,
                "length-normalized reciprocal mass": {
                    book_id: value / math.sqrt(book_block_counts[book_id])
                    for book_id, value in sum_scores.items()
                },
            }
            for score_name, scores in score_variants.items():
                ordered = sorted(scores, key=lambda item: (-scores[item], item))
                ranks_by_score[score_name].append(
                    ordered.index(query_book[query_id]) + 1
                    if query_book[query_id] in scores
                    else len(book_block_counts) + 1
                )
                for document_budget in (1, 3, 5):
                    routed_blocks_by_score[score_name][document_budget].append(
                        sum(
                            book_block_counts[book_id]
                            for book_id in ordered[:document_budget]
                        )
                    )
            routed_book_blocks.append(book_block_counts[query_book[query_id]])
        for score_name, document_ranks in ranks_by_score.items():
            document_routing.append(
                {
                    "method": method,
                    "block_ranking_depth": args.routing_depth,
                    "document_score": score_name,
                    "document_top1_recall": mean(rank <= 1 for rank in document_ranks),
                    "document_top3_recall": mean(rank <= 3 for rank in document_ranks),
                    "document_top5_recall": mean(rank <= 5 for rank in document_ranks),
                    "mean_document_rank": mean(document_ranks),
                    "mean_blocks_in_target_book": mean(routed_book_blocks),
                    "mean_search_domain_reduction_if_target_book_known": (
                        base_count / mean(routed_book_blocks)
                    ),
                    "mean_routed_blocks_top1_document": mean(
                        routed_blocks_by_score[score_name][1]
                    ),
                    "mean_routed_blocks_top3_documents": mean(
                        routed_blocks_by_score[score_name][3]
                    ),
                    "mean_routed_blocks_top5_documents": mean(
                        routed_blocks_by_score[score_name][5]
                    ),
                    "mean_domain_reduction_top3_documents": (
                        base_count / mean(routed_blocks_by_score[score_name][3])
                    ),
                    "target_scopes_are_shortest_eligible_scopes": (
                        args.target_scopes_are_shortest
                    ),
                }
            )

    hierarchical_block_retrieval = []
    gold = set(range(base_count, base_count + source_count))
    for method in sorted(retrieval_methods):
        global_any = []
        global_recall = []
        routed_any = []
        routed_recall = []
        routed_last = []
        routed_domains = []
        for query_id in query_ids:
            ranking = [
                int(item)
                for item in text_lookup[(query_id, method)]["top_block_ids"]
            ]
            sum_scores: dict[int, float] = {}
            for rank, block_id in enumerate(ranking[: args.routing_depth], start=1):
                book_id = (
                    int(block_book_ids[block_id])
                    if block_id < base_count
                    else query_book[query_id]
                )
                if book_id >= 0:
                    sum_scores[book_id] = sum_scores.get(book_id, 0.0) + 1.0 / (
                        args.rrf_k + rank
                    )
            normalized = {
                book_id: value / math.sqrt(book_block_counts[book_id])
                for book_id, value in sum_scores.items()
            }
            routed_books = sorted(
                normalized, key=lambda item: (-normalized[item], item)
            )[:3]
            routed_set = set(routed_books)
            filtered = []
            for block_id in ranking:
                book_id = (
                    int(block_book_ids[block_id])
                    if block_id < base_count
                    else query_book[query_id]
                )
                if book_id in routed_set:
                    filtered.append(block_id)
                    if len(filtered) == 8:
                        break
            global_selected = set(ranking[:8])
            routed_selected = set(filtered)
            global_any.append(bool(global_selected & gold))
            global_recall.append(len(global_selected & gold) / source_count)
            routed_any.append(bool(routed_selected & gold))
            routed_recall.append(len(routed_selected & gold) / source_count)
            routed_last.append(base_count + source_count - 1 in routed_selected)
            routed_domains.append(sum(book_block_counts[item] for item in routed_books))
        hierarchical_block_retrieval.append(
            {
                "method": method,
                "document_router": "top3 length-normalized reciprocal mass from top512 blocks",
                "within_document_selection": "first top8 blocks after filtering original ranking",
                "global_top8_any_hit": mean(global_any),
                "routed_top8_any_hit": mean(routed_any),
                "global_top8_source_recall": mean(global_recall),
                "routed_top8_source_recall": mean(routed_recall),
                "routed_top8_last_source_hit": mean(routed_last),
                "mean_routed_domain_blocks": mean(routed_domains),
                "mean_domain_reduction": base_count / mean(routed_domains),
                "ranking_truncated_at_stored_fusion_depth": len(
                    text_lookup[(query_ids[0], method)]["top_block_ids"]
                ),
            }
        )

    output = {
        "source": f"real {args.scope_label}-level candidate provenance",
        "scope_label": args.scope_label,
        "queries": len(query_ids),
        "contains_synthetic_text": False,
        "candidate_retrieval_uses_target": False,
        "mean_same_book_rate_in_retrieval_candidates": mean(candidate_same_rates),
        "mean_same_book_rate_in_random_candidates": mean(random_same_rates),
        "mean_same_scope_rate_in_retrieval_candidates": mean(candidate_same_rates),
        "mean_same_scope_rate_in_random_candidates": mean(random_same_rates),
        "future_utility_event_above_random95": {
            "same_book_events": event_same,
            "other_book_events": event_other,
            "same_book_nonevents": nonevent_same,
            "other_book_nonevents": nonevent_other,
            "same_book_event_rate": event_same / (event_same + nonevent_same),
            "other_book_event_rate": event_other / (event_other + nonevent_other),
            "fisher_odds_ratio": float(fisher.statistic),
            "fisher_one_sided_p": float(fisher.pvalue),
            "same_scope_event_rate": event_same / (event_same + nonevent_same),
            "other_scope_event_rate": event_other / (event_other + nonevent_other),
        },
        "same_book_mean_utility_advantage": {
            "mean_delta_nll_advantage": mean(per_query_same_utility_advantage),
            "bootstrap95": bootstrap_ci(
                per_query_same_utility_advantage,
                args.bootstrap_samples,
                args.seed + 100,
            ),
        },
        "best_A_selection": {
            "unconstrained": selection_quality(
                best_a_all, args.bootstrap_samples, args.seed + 200
            ),
            "same_book_only": selection_quality(
                best_a_same, args.bootstrap_samples, args.seed + 201
            ),
            "other_book_only": selection_quality(
                best_a_other, args.bootstrap_samples, args.seed + 202
            ),
            "same_book_improvement_over_other_book": {
                "mean_nll_improvement": mean(paired_best_advantage),
                "bootstrap95": bootstrap_ci(
                    paired_best_advantage,
                    args.bootstrap_samples,
                    args.seed + 203,
                ),
                "same_book_wins": sum(value > 0 for value in paired_best_advantage),
                "other_book_wins": sum(value < 0 for value in paired_best_advantage),
            },
        },
        "scope_interpretation": {
            "same_book_fields_mean_same_scope": True,
            "scope_label": args.scope_label,
        },
        "static_top1_by_method": method_top1,
        "document_level_routing": document_routing,
        "hierarchical_top3_document_then_top8_block": hierarchical_block_retrieval,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
