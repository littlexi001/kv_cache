from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


TOPKS = (8, 64, 512)
HUB_ALPHAS = (0.0, 1.0, 3.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure incremental retrieval frontiers, hierarchical trajectory "
            "stability, and query-independent retrieval hubs on real 10M memories."
        )
    )
    parser.add_argument("--xsum_rows", required=True)
    parser.add_argument("--xsum_qk_rows", required=True)
    parser.add_argument("--pg19_rows", required=True)
    parser.add_argument("--code_rows", required=True)
    parser.add_argument("--code_scope_ids", required=True)
    parser.add_argument("--code_metadata", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--output_rows", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def summarize_values(
    values: list[float], *, samples: int, rng: np.random.Generator
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "query_bootstrap95": interval(means),
    }


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def filter_largest_memory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    memory_tokens = max(int(row["memory_tokens"]) for row in rows)
    selected = [row for row in rows if int(row["memory_tokens"]) == memory_tokens]
    if any(bool(row["selection_uses_target"]) for row in selected):
        raise ValueError("retrieval rows use the target")
    return selected


def row_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    return {
        (int(row["query_id"]), int(row["prefix_tokens"]), str(row["method"])): row
        for row in rows
    }


def incremental_rows(
    dataset: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    lookup = row_lookup(rows)
    query_ids = sorted({int(row["query_id"]) for row in rows})
    prefixes = sorted({int(row["prefix_tokens"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    output = []
    for method in methods:
        for prefix_index in range(len(prefixes) - 1):
            previous = prefixes[prefix_index]
            current = prefixes[prefix_index + 1]
            for query_id in query_ids:
                left = lookup[(query_id, previous, method)]
                right = lookup[(query_id, current, method)]
                left_ranking = [int(item) for item in left["top_block_ids"]]
                right_ranking = [int(item) for item in right["top_block_ids"]]
                next_top8 = set(right_ranking[:8])
                history = set()
                for earlier in prefixes[: prefix_index + 1]:
                    history.update(
                        int(item)
                        for item in lookup[(query_id, earlier, method)][
                            "top_block_ids"
                        ][:512]
                    )
                base_blocks = int(left["base_blocks"])
                memory_blocks = int(left["memory_blocks"])
                source = set(range(base_blocks, memory_blocks))
                previous_frontier = set(left_ranking[:512])
                previous_rescored = [
                    block_id
                    for block_id in right_ranking[:512]
                    if block_id in previous_frontier
                ]
                history_rescored = [
                    block_id for block_id in right_ranking[:512] if block_id in history
                ]
                previous_auditable = len(previous_rescored) >= 8
                history_auditable = len(history_rescored) >= 8
                previous_rescored_top8 = (
                    set(previous_rescored[:8]) if previous_auditable else None
                )
                history_rescored_top8 = (
                    set(history_rescored[:8]) if history_auditable else None
                )
                row = {
                    "dataset": dataset,
                    "query_id": query_id,
                    "method": method,
                    "prefix_transition": f"{previous}->{current}",
                    "previous_prefix_tokens": previous,
                    "current_prefix_tokens": current,
                    "top8_jaccard": jaccard(
                        set(left_ranking[:8]), set(right_ranking[:8])
                    ),
                    "history_frontier_blocks": len(history),
                    "history_frontier_recall_of_next_top8": len(history & next_top8)
                    / 8.0,
                    "next_top8_source_any": bool(next_top8 & source),
                    "next_top8_source_recall": len(next_top8 & source) / len(source),
                    "previous_top512_source_any": bool(
                        set(left_ranking[:512]) & source
                    ),
                    "history_frontier_source_any": bool(history & source),
                    "source_emerges_in_next_top8_outside_previous_top512": bool(
                        next_top8 & source
                    )
                    and not bool(previous_frontier & source),
                    "previous_frontier_intersection_current_top512": len(
                        previous_rescored
                    ),
                    "previous_frontier_rescore_exactly_auditable": previous_auditable,
                    "history_frontier_rescore_exactly_auditable": history_auditable,
                    "previous_top512_rescored_top8_jaccard_with_full": (
                        jaccard(previous_rescored_top8, next_top8)
                        if previous_rescored_top8 is not None
                        else None
                    ),
                    "history_frontier_rescored_top8_jaccard_with_full": (
                        jaccard(history_rescored_top8, next_top8)
                        if history_rescored_top8 is not None
                        else None
                    ),
                    "previous_top512_rescored_source_any_at_8": (
                        bool(previous_rescored_top8 & source)
                        if previous_rescored_top8 is not None
                        else None
                    ),
                    "history_frontier_rescored_source_any_at_8": (
                        bool(history_rescored_top8 & source)
                        if history_rescored_top8 is not None
                        else None
                    ),
                    "previous_top512_rescored_source_any_delta_vs_full": (
                        float(bool(previous_rescored_top8 & source))
                        - float(bool(next_top8 & source))
                        if previous_rescored_top8 is not None
                        else None
                    ),
                    "history_frontier_rescored_source_any_delta_vs_full": (
                        float(bool(history_rescored_top8 & source))
                        - float(bool(next_top8 & source))
                        if history_rescored_top8 is not None
                        else None
                    ),
                    "selection_uses_target": False,
                }
                for topk in TOPKS:
                    previous_set = set(left_ranking[:topk])
                    row[f"previous_top{topk}_recall_of_next_top8"] = (
                        len(previous_set & next_top8) / 8.0
                    )
                output.append(row)
    return output


def summarize_incremental(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"], row["prefix_transition"])].append(
            row
        )
    metrics = [
        "top8_jaccard",
        "previous_top8_recall_of_next_top8",
        "previous_top64_recall_of_next_top8",
        "previous_top512_recall_of_next_top8",
        "history_frontier_recall_of_next_top8",
        "next_top8_source_any",
        "previous_top512_source_any",
        "history_frontier_source_any",
        "source_emerges_in_next_top8_outside_previous_top512",
        "history_frontier_blocks",
        "previous_frontier_intersection_current_top512",
        "previous_frontier_rescore_exactly_auditable",
        "history_frontier_rescore_exactly_auditable",
    ]
    optional_metrics = [
        "previous_top512_rescored_top8_jaccard_with_full",
        "history_frontier_rescored_top8_jaccard_with_full",
        "previous_top512_rescored_source_any_at_8",
        "history_frontier_rescored_source_any_at_8",
        "previous_top512_rescored_source_any_delta_vs_full",
        "history_frontier_rescored_source_any_delta_vs_full",
    ]
    output = []
    for key, group in sorted(groups.items()):
        summary = {
            "dataset": key[0],
            "method": key[1],
            "prefix_transition": key[2],
            "queries": len(group),
        }
        for metric in metrics:
            summary[metric] = summarize_values(
                [float(row[metric]) for row in group], samples=samples, rng=rng
            )
        for metric in optional_metrics:
            observed = [
                float(row[metric]) for row in group if row[metric] is not None
            ]
            summary[metric] = {
                **summarize_values(observed, samples=samples, rng=rng),
                "audited_queries": len(observed),
            }
        full_hits = np.asarray(
            [float(row["next_top8_source_any"]) for row in group], dtype=np.float64
        )
        preserved = np.asarray(
            [
                float(row["next_top8_source_any"])
                * float(row["previous_top512_source_any"])
                for row in group
            ],
            dtype=np.float64,
        )
        indices = rng.integers(0, len(group), size=(samples, len(group)))
        denominator = full_hits[indices].sum(axis=1)
        numerator = preserved[indices].sum(axis=1)
        valid = denominator > 0
        ratios = numerator[valid] / denominator[valid]
        summary["conditional_next_source_hit_already_in_previous_top512"] = {
            "rate": float(preserved.sum() / full_hits.sum()) if full_hits.sum() else None,
            "query_bootstrap95": interval(ratios) if len(ratios) else None,
        }
        output.append(summary)
    return output


def gini_from_counts(counts: Counter[int], population: int) -> float:
    if population <= 0:
        return math.nan
    values = np.zeros(population, dtype=np.float64)
    for block_id, count in counts.items():
        if 0 <= block_id < population:
            values[block_id] = count
    total = values.sum()
    if total == 0:
        return 0.0
    values.sort()
    indices = np.arange(1, population + 1, dtype=np.float64)
    return float((2.0 * np.sum(indices * values) / (population * total)) - (population + 1) / population)


def hubness_summary(rows: list[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    query_ids = sorted({int(row["query_id"]) for row in rows})
    prefixes = sorted({int(row["prefix_tokens"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    lookup = row_lookup(rows)
    base_blocks = int(rows[0]["base_blocks"])
    output = []
    count_lookup: dict[tuple[str, int, int], Counter[int]] = {}
    for method in methods:
        for prefix in prefixes:
            for topk in TOPKS:
                counts: Counter[int] = Counter()
                for query_id in query_ids:
                    ranking = lookup[(query_id, prefix, method)]["top_block_ids"][:topk]
                    counts.update(
                        int(item) for item in ranking if int(item) < base_blocks
                    )
                count_lookup[(method, prefix, topk)] = counts
                total = sum(counts.values())
                top_memory_count = max(1, math.ceil(0.01 * base_blocks))
                ordered = sorted(counts.values(), reverse=True)
                effective = (
                    total * total / sum(value * value for value in ordered)
                    if ordered
                    else 0.0
                )
                output.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "prefix_tokens": prefix,
                        "topk": topk,
                        "queries": len(query_ids),
                        "base_blocks": base_blocks,
                        "base_nominations": total,
                        "unique_nominated_base_blocks": len(counts),
                        "effective_nominated_blocks_simpson": effective,
                        "top_1pct_memory_nomination_share": (
                            sum(ordered[:top_memory_count]) / total if total else 0.0
                        ),
                        "top_1pct_nominated_blocks_nomination_share": (
                            sum(ordered[: max(1, math.ceil(0.01 * len(ordered)))])
                            / total
                            if total
                            else 0.0
                        ),
                        "top10_blocks_nomination_share": (
                            sum(ordered[:10]) / total if total else 0.0
                        ),
                        "max_block_query_frequency": (
                            max(ordered) / len(query_ids) if ordered else 0.0
                        ),
                        "gini_over_all_base_blocks": gini_from_counts(
                            counts, base_blocks
                        ),
                    }
                )
    for method in methods:
        for previous, current in zip(prefixes[:-1], prefixes[1:]):
            for topk in TOPKS:
                left = count_lookup[(method, previous, topk)]
                right = count_lookup[(method, current, topk)]
                union = sorted(set(left) | set(right))
                intersection = sorted(set(left) & set(right))
                union_left = [left[item] for item in union]
                union_right = [right[item] for item in union]
                union_statistic = (
                    spearmanr(union_left, union_right).statistic
                    if len(set(union_left)) > 1 and len(set(union_right)) > 1
                    else math.nan
                )
                intersection_left = [left[item] for item in intersection]
                intersection_right = [right[item] for item in intersection]
                intersection_statistic = (
                    spearmanr(intersection_left, intersection_right).statistic
                    if len(intersection) >= 2
                    and len(set(intersection_left)) > 1
                    and len(set(intersection_right)) > 1
                    else math.nan
                )
                recurrent_left = {item for item, count in left.items() if count >= 2}
                recurrent_right = {item for item, count in right.items() if count >= 2}
                output.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "prefix_transition": f"{previous}->{current}",
                        "topk": topk,
                        "frequency_spearman_over_nominated_union": (
                            float(union_statistic)
                            if math.isfinite(float(union_statistic))
                            else None
                        ),
                        "frequency_spearman_over_shared_nominations": (
                            float(intersection_statistic)
                            if math.isfinite(float(intersection_statistic))
                            else None
                        ),
                        "nominated_set_jaccard": jaccard(set(left), set(right)),
                        "shared_nominated_blocks": len(intersection),
                        "recurrent_hub_set_jaccard": jaccard(
                            recurrent_left, recurrent_right
                        ),
                        "recurrent_hubs_previous": len(recurrent_left),
                        "recurrent_hubs_current": len(recurrent_right),
                        "union_nominated_blocks": len(union),
                        "record_type": "temporal_hub_stability",
                    }
                )
    return output


def hub_penalty_rows(dataset: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = row_lookup(rows)
    query_ids = sorted({int(row["query_id"]) for row in rows})
    prefixes = sorted({int(row["prefix_tokens"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    output = []
    for method in methods:
        for prefix in prefixes:
            per_query_counts: dict[int, Counter[int]] = {}
            total_counts: Counter[int] = Counter()
            for query_id in query_ids:
                source_row = lookup[(query_id, prefix, method)]
                base_blocks = int(source_row["base_blocks"])
                counts = Counter(
                    int(item)
                    for item in source_row["top_block_ids"][:64]
                    if int(item) < base_blocks
                )
                per_query_counts[query_id] = counts
                total_counts.update(counts)
            for query_id in query_ids:
                source_row = lookup[(query_id, prefix, method)]
                ranking = [int(item) for item in source_row["top_block_ids"][:64]]
                base_blocks = int(source_row["base_blocks"])
                source = set(range(base_blocks, int(source_row["memory_blocks"])))
                other_queries = max(1, len(query_ids) - 1)
                for alpha in HUB_ALPHAS:
                    scored = []
                    for rank, block_id in enumerate(ranking, start=1):
                        frequency = 0.0
                        if block_id < base_blocks:
                            frequency = (
                                total_counts[block_id]
                                - per_query_counts[query_id][block_id]
                            ) / other_queries
                        score = -math.log(60.0 + rank) - alpha * frequency
                        scored.append((score, -rank, block_id, frequency))
                    scored.sort(reverse=True)
                    selected = [item[2] for item in scored[:8]]
                    output.append(
                        {
                            "dataset": dataset,
                            "query_id": query_id,
                            "method": method,
                            "prefix_tokens": prefix,
                            "alpha": alpha,
                            "source_any_at_8": bool(set(selected) & source),
                            "source_recall_at_8": len(set(selected) & source)
                            / len(source),
                            "source_any_at_64_upper_bound": bool(set(ranking) & source),
                            "selected_mean_loo_hub_frequency": float(
                                np.mean([item[3] for item in scored[:8]])
                            ),
                            "selection_uses_target": False,
                        }
                    )
    return output


def crossfit_hub_mass(
    rows: list[dict[str, Any]], method: str, prefix: int, topk: int
) -> dict[int, float]:
    lookup = row_lookup(rows)
    query_ids = sorted({int(row["query_id"]) for row in rows})
    per_query: dict[int, set[int]] = {}
    for query_id in query_ids:
        row = lookup[(query_id, prefix, method)]
        base_blocks = int(row["base_blocks"])
        per_query[query_id] = {
            int(item)
            for item in row["top_block_ids"][:topk]
            if int(item) < base_blocks
        }
    output: dict[int, float] = {}
    for test_fold in (0, 1):
        train_ids = [query_id for query_id in query_ids if query_id % 2 != test_fold]
        test_ids = [query_id for query_id in query_ids if query_id % 2 == test_fold]
        counts: Counter[int] = Counter()
        for query_id in train_ids:
            counts.update(per_query[query_id])
        hub_count = max(1, math.ceil(0.01 * len(counts)))
        hubs = {
            block_id
            for block_id, _ in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[:hub_count]
        }
        for query_id in test_ids:
            selected = per_query[query_id]
            output[query_id] = len(selected & hubs) / len(selected) if selected else 0.0
    return output


def xsum_text_qk_hub_comparison(
    text_rows: list[dict[str, Any]],
    qk_rows: list[dict[str, Any]],
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    text_method = "bm25_e5_rrf"
    qk_methods = sorted({str(row["method"]) for row in qk_rows})
    output = []
    for prefix in (8, 64):
        text_mass = crossfit_hub_mass(text_rows, text_method, prefix, 64)
        query_ids = sorted(text_mass)
        text_values = np.asarray([text_mass[item] for item in query_ids])
        indices = rng.integers(
            0, len(query_ids), size=(min(samples, 10_000), len(query_ids))
        )
        text_bootstrap = text_values[indices].mean(axis=1)
        output.append(
            {
                "prefix_tokens": prefix,
                "method": text_method,
                "family": "text",
                "topk": 64,
                "crossfit_top_1pct_hub_nomination_mass": float(text_values.mean()),
                "query_bootstrap95": interval(text_bootstrap),
                "delta_vs_text": 0.0,
                "paired_delta_query_bootstrap95": [0.0, 0.0],
            }
        )
        for method in qk_methods:
            qk_mass = crossfit_hub_mass(qk_rows, method, prefix, 64)
            if sorted(qk_mass) != query_ids:
                raise ValueError("XSum text and QK query ids do not align")
            qk_values = np.asarray([qk_mass[item] for item in query_ids])
            qk_bootstrap = qk_values[indices].mean(axis=1)
            paired = qk_values - text_values
            output.append(
                {
                    "prefix_tokens": prefix,
                    "method": method,
                    "family": "qk",
                    "topk": 64,
                    "crossfit_top_1pct_hub_nomination_mass": float(
                        qk_values.mean()
                    ),
                    "query_bootstrap95": interval(qk_bootstrap),
                    "delta_vs_text": float(paired.mean()),
                    "paired_delta_query_bootstrap95": interval(
                        paired[indices].mean(axis=1)
                    ),
                }
            )
    return output


def summarize_hub_penalty(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["dataset"],
                row["method"],
                int(row["prefix_tokens"]),
                float(row["alpha"]),
            )
        ].append(row)
    baselines = {
        (dataset, method, prefix): {
            int(row["query_id"]): row
            for row in group
        }
        for (dataset, method, prefix, alpha), group in groups.items()
        if alpha == 0.0
    }
    output = []
    for key, group in sorted(groups.items()):
        baseline = baselines[key[:3]]
        deltas = [
            float(row["source_any_at_8"])
            - float(baseline[int(row["query_id"])]["source_any_at_8"])
            for row in group
        ]
        summary = {
            "dataset": key[0],
            "method": key[1],
            "prefix_tokens": key[2],
            "alpha": key[3],
            "queries": len(group),
            "source_any_at_8": summarize_values(
                [float(row["source_any_at_8"]) for row in group],
                samples=samples,
                rng=rng,
            ),
            "source_recall_at_8": summarize_values(
                [float(row["source_recall_at_8"]) for row in group],
                samples=samples,
                rng=rng,
            ),
            "source_any_at_64_upper_bound": float(
                np.mean([float(row["source_any_at_64_upper_bound"]) for row in group])
            ),
            "mean_selected_loo_hub_frequency": float(
                np.mean(
                    [float(row["selected_mean_loo_hub_frequency"]) for row in group]
                )
            ),
            "source_any_delta_vs_alpha0": summarize_values(
                deltas, samples=samples, rng=rng
            ),
            "wins_ties_losses_vs_alpha0": [
                sum(value > 0 for value in deltas),
                sum(value == 0 for value in deltas),
                sum(value < 0 for value in deltas),
            ],
        }
        output.append(summary)
    return output


def scope_ranking(
    ranking: list[int],
    *,
    query_scope: int,
    base_blocks: int,
    scope_ids: np.ndarray,
    scope_counts: Counter[int],
    depth: int = 512,
) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for rank, block_id in enumerate(ranking[:depth], start=1):
        scope = int(scope_ids[block_id]) if block_id < base_blocks else query_scope
        if scope >= 0:
            scores[scope] += 1.0 / (60.0 + rank)
    normalized = {
        scope: score / math.sqrt(max(1, scope_counts[scope]))
        for scope, score in scores.items()
    }
    return sorted(normalized, key=lambda scope: (-normalized[scope], scope))


def code_scope_rows(
    rows: list[dict[str, Any]], scope_ids_path: str, metadata_path: str
) -> list[dict[str, Any]]:
    scope_ids = np.asarray(np.load(scope_ids_path, mmap_mode="r"), dtype=np.int64)
    metadata = {
        int(row["query_id"]): int(row["repo_index"])
        for row in read_jsonl(metadata_path)
    }
    lookup = row_lookup(rows)
    query_ids = sorted(metadata)
    prefixes = sorted({int(row["prefix_tokens"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    base_blocks = int(rows[0]["base_blocks"])
    scope_counts = Counter(int(item) for item in scope_ids[:base_blocks] if int(item) >= 0)
    output = []
    for method in methods:
        cached_rankings: dict[tuple[int, int], list[int]] = {}
        cached_scopes: dict[tuple[int, int], list[int]] = {}
        for query_id in query_ids:
            for prefix in prefixes:
                ranking = [
                    int(item)
                    for item in lookup[(query_id, prefix, method)]["top_block_ids"]
                ]
                cached_rankings[(query_id, prefix)] = ranking
                cached_scopes[(query_id, prefix)] = scope_ranking(
                    ranking,
                    query_scope=metadata[query_id],
                    base_blocks=base_blocks,
                    scope_ids=scope_ids,
                    scope_counts=scope_counts,
                )
        for previous, current in zip(prefixes[:-1], prefixes[1:]):
            for query_id in query_ids:
                next_top8 = cached_rankings[(query_id, current)][:8]
                next_top8_scopes = [
                    int(scope_ids[item]) if item < base_blocks else metadata[query_id]
                    for item in next_top8
                ]
                row = {
                    "dataset": "code",
                    "query_id": query_id,
                    "method": method,
                    "prefix_transition": f"{previous}->{current}",
                    "selection_uses_target": False,
                    "top8_block_jaccard": jaccard(
                        set(cached_rankings[(query_id, previous)][:8]),
                        set(cached_rankings[(query_id, current)][:8]),
                    ),
                }
                for depth in (1, 3, 8):
                    left = set(cached_scopes[(query_id, previous)][:depth])
                    right = set(cached_scopes[(query_id, current)][:depth])
                    row[f"top{depth}_scope_jaccard"] = jaccard(left, right)
                    row[f"previous_top{depth}_scope_coverage_of_next_top8"] = (
                        sum(scope in left for scope in next_top8_scopes) / 8.0
                    )
                    row[f"previous_top{depth}_contains_query_scope"] = (
                        metadata[query_id] in left
                    )
                    row[f"next_top{depth}_contains_query_scope"] = (
                        metadata[query_id] in right
                    )
                    row[f"top{depth}_scope_minus_top8_block_jaccard"] = (
                        row[f"top{depth}_scope_jaccard"] - row["top8_block_jaccard"]
                    )
                output.append(row)
    return output


def summarize_scope(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["prefix_transition"])].append(row)
    output = []
    metrics = []
    metrics.append("top8_block_jaccard")
    for depth in (1, 3, 8):
        metrics.extend(
            [
                f"top{depth}_scope_jaccard",
                f"previous_top{depth}_scope_coverage_of_next_top8",
                f"previous_top{depth}_contains_query_scope",
                f"next_top{depth}_contains_query_scope",
                f"top{depth}_scope_minus_top8_block_jaccard",
            ]
        )
    for key, group in sorted(groups.items()):
        summary = {
            "dataset": "code",
            "method": key[0],
            "prefix_transition": key[1],
            "queries": len(group),
        }
        for metric in metrics:
            summary[metric] = summarize_values(
                [float(row[metric]) for row in group], samples=samples, rng=rng
            )
        output.append(summary)
    return output


def main() -> None:
    args = parse_args()
    datasets = {
        "xsum": filter_largest_memory(read_jsonl(args.xsum_rows)),
        "xsum_qk": filter_largest_memory(read_jsonl(args.xsum_qk_rows)),
        "pg19": filter_largest_memory(read_jsonl(args.pg19_rows)),
        "code": filter_largest_memory(read_jsonl(args.code_rows)),
    }
    rng = np.random.default_rng(args.seed)
    incremental = [
        row
        for dataset, rows in datasets.items()
        for row in incremental_rows(dataset, rows)
    ]
    penalty = [
        row
        for dataset, rows in datasets.items()
        for row in hub_penalty_rows(dataset, rows)
    ]
    code_scopes = code_scope_rows(
        datasets["code"], args.code_scope_ids, args.code_metadata
    )
    output = {
        "source": "real 10M XSum, PG19, and LongBench-v2 code retrieval trajectories",
        "protocol": {
            "datasets": {
                name: {
                    "queries": len({int(row["query_id"]) for row in rows}),
                    "memory_tokens": max(int(row["memory_tokens"]) for row in rows),
                    "memory_blocks": int(rows[0]["memory_blocks"]),
                    "prefix_tokens": sorted(
                        {int(row["prefix_tokens"]) for row in rows}
                    ),
                    "methods": sorted({str(row["method"]) for row in rows}),
                }
                for name, rows in datasets.items()
            },
            "selection_uses_target": False,
            "hub_frequency_is_leave_one_query_out": True,
            "source_blocks_excluded_from_hub_frequency": True,
            "hub_penalty_parameters_predeclared": list(HUB_ALPHAS),
            "independent_statistical_unit": "query",
        },
        "incremental_frontier": summarize_incremental(
            incremental, samples=args.bootstrap_samples, rng=rng
        ),
        "code_scope_stability": summarize_scope(
            code_scopes, samples=args.bootstrap_samples, rng=rng
        ),
        "hubness": [
            row
            for dataset, rows in datasets.items()
            for row in hubness_summary(rows, dataset)
        ],
        "xsum_text_qk_hub_concentration": xsum_text_qk_hub_comparison(
            datasets["xsum"],
            datasets["xsum_qk"],
            samples=args.bootstrap_samples,
            rng=rng,
        ),
        "hub_penalty": summarize_hub_penalty(
            penalty, samples=args.bootstrap_samples, rng=rng
        ),
    }
    output_path = Path(args.output_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    detail_path = Path(args.output_rows)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", encoding="utf-8") as handle:
        for row_type, rows in (
            ("incremental_frontier", incremental),
            ("hub_penalty", penalty),
            ("code_scope_stability", code_scopes),
        ):
            for row in rows:
                handle.write(
                    json.dumps(
                        {"row_type": row_type, **row},
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
