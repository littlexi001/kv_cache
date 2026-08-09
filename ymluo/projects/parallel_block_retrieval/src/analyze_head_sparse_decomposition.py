from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure head-specific sparsity, overlap, and cross-query specialization "
            "from frozen per-head block rankings."
        )
    )
    parser.add_argument("--topk_npz", required=True)
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_blocks", type=int, default=39062)
    parser.add_argument("--depths", default="1,2,4,8,16")
    parser.add_argument("--cv_splits", type=int, default=200)
    parser.add_argument("--random_subsets_per_split", type=int, default=20)
    parser.add_argument("--overlap_samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_head_hits(
    block_ids: np.ndarray, queries: list[dict[str, Any]], depth: int
) -> np.ndarray:
    query_count, layer_count, head_count, _ = block_ids.shape
    head_hits = np.zeros((layer_count * head_count, query_count), dtype=bool)
    for query_index, query in enumerate(queries):
        gold = np.asarray(query.get("gold_block_ids", []), dtype=np.int64)
        candidates = block_ids[query_index, :, :, :depth].reshape(
            layer_count * head_count, depth
        )
        head_hits[:, query_index] = np.isin(candidates, gold).any(axis=1)
    return head_hits


def stratified_split(
    queries: list[dict[str, Any]], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    train: list[int] = []
    test: list[int] = []
    datasets = sorted({str(query["dataset"]) for query in queries})
    for dataset in datasets:
        indices = np.asarray(
            [
                index
                for index, query in enumerate(queries)
                if str(query["dataset"]) == dataset
            ],
            dtype=np.int64,
        )
        rng.shuffle(indices)
        cutoff = max(1, len(indices) // 2)
        train.extend(indices[:cutoff].tolist())
        test.extend(indices[cutoff:].tolist())
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def greedy_heads(
    hits: np.ndarray, train_indices: np.ndarray, count: int
) -> list[int]:
    head_count, query_count = hits.shape
    covered = np.zeros(query_count, dtype=bool)
    available = np.ones(head_count, dtype=bool)
    selected: list[int] = []
    for _ in range(count):
        gains = np.sum(hits[:, train_indices] & ~covered[train_indices], axis=1)
        total_hits = np.sum(hits[:, train_indices], axis=1)
        gains[~available] = -1
        total_hits[~available] = -1
        # Primary key: uncovered train gain. Secondary key: total train hits.
        best = int(
            np.lexsort((np.arange(head_count), -total_hits, -gains))[0]
        )
        selected.append(best)
        available[best] = False
        covered |= hits[best]
    return selected


def selection_cross_validation(
    hits: np.ndarray,
    queries: list[dict[str, Any]],
    subset_sizes: list[int],
    splits: int,
    random_subsets_per_split: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_values = {size: [] for size in subset_sizes}
    random_values = {size: [] for size in subset_sizes}
    delta_values = {size: [] for size in subset_sizes}
    selected_macro_values = {size: [] for size in subset_sizes}
    random_macro_values = {size: [] for size in subset_sizes}
    macro_delta_values = {size: [] for size in subset_sizes}
    oracle_values: list[float] = []
    max_size = max(subset_sizes)

    for _ in range(splits):
        train, test = stratified_split(queries, rng)
        selected = greedy_heads(hits, train, max_size)
        oracle_values.append(float(np.any(hits[:, test], axis=0).mean()))
        for size in subset_sizes:
            selected_test_hits = np.any(
                hits[np.asarray(selected[:size]), :][:, test], axis=0
            )
            selected_recall = float(selected_test_hits.mean())
            test_datasets = np.asarray(
                [str(queries[int(index)]["dataset"]) for index in test]
            )
            selected_macro_recall = float(
                np.mean(
                    [
                        selected_test_hits[test_datasets == dataset].mean()
                        for dataset in np.unique(test_datasets)
                    ]
                )
            )
            random_recalls: list[float] = []
            random_macro_recalls: list[float] = []
            for _ in range(random_subsets_per_split):
                random_heads = rng.choice(hits.shape[0], size=size, replace=False)
                random_test_hits = np.any(hits[random_heads, :][:, test], axis=0)
                random_recalls.append(float(random_test_hits.mean()))
                random_macro_recalls.append(
                    float(
                        np.mean(
                            [
                                random_test_hits[test_datasets == dataset].mean()
                                for dataset in np.unique(test_datasets)
                            ]
                        )
                    )
                )
            random_recall = float(np.mean(random_recalls))
            random_macro_recall = float(np.mean(random_macro_recalls))
            selected_values[size].append(selected_recall)
            random_values[size].append(random_recall)
            delta_values[size].append(selected_recall - random_recall)
            selected_macro_values[size].append(selected_macro_recall)
            random_macro_values[size].append(random_macro_recall)
            macro_delta_values[size].append(
                selected_macro_recall - random_macro_recall
            )

    rows: list[dict[str, Any]] = []
    for size in subset_sizes:
        selected_stats = percentile_summary(selected_values[size])
        random_stats = percentile_summary(random_values[size])
        delta_stats = percentile_summary(delta_values[size])
        selected_macro_stats = percentile_summary(selected_macro_values[size])
        random_macro_stats = percentile_summary(random_macro_values[size])
        macro_delta_stats = percentile_summary(macro_delta_values[size])
        rows.append(
            {
                "heads": size,
                "selected_test_recall": selected_stats["mean"],
                "selected_p05": selected_stats["p05"],
                "selected_p95": selected_stats["p95"],
                "random_test_recall": random_stats["mean"],
                "random_p05": random_stats["p05"],
                "random_p95": random_stats["p95"],
                "selected_minus_random": delta_stats["mean"],
                "delta_p05": delta_stats["p05"],
                "delta_p95": delta_stats["p95"],
                "selected_test_macro_recall": selected_macro_stats["mean"],
                "random_test_macro_recall": random_macro_stats["mean"],
                "macro_selected_minus_random": macro_delta_stats["mean"],
                "macro_delta_p05": macro_delta_stats["p05"],
                "macro_delta_p95": macro_delta_stats["p95"],
            }
        )
    return rows, {
        "splits": splits,
        "random_subsets_per_split": random_subsets_per_split,
        "stratified_by": "dataset",
        "mean_all_head_test_oracle": float(np.mean(oracle_values)),
    }


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    intersection = len(set(int(item) for item in left) & set(int(item) for item in right))
    return intersection / (len(left) + len(right) - intersection)


def overlap_diagnostics(
    block_ids: np.ndarray,
    depth: int,
    sample_cap: int,
    rng: np.random.Generator,
    num_blocks: int,
) -> list[dict[str, Any]]:
    query_count, layer_count, head_count, _ = block_ids.shape
    values = block_ids[:, :, :, :depth]
    categories: dict[str, list[tuple[int, int, int, int, int]]] = {
        "gqa_sibling_same_layer": [],
        "same_layer_different_kv": [],
        "same_query_head_cross_layer": [],
        "random_cross_layer": [],
    }

    for query_index in range(query_count):
        for layer in range(layer_count):
            for kv_head in range(head_count // 2):
                categories["gqa_sibling_same_layer"].append(
                    (query_index, layer, 2 * kv_head, layer, 2 * kv_head + 1)
                )
            for query_head in range(head_count):
                categories["same_layer_different_kv"].append(
                    (
                        query_index,
                        layer,
                        query_head,
                        layer,
                        (query_head + 2) % head_count,
                    )
                )
        for query_head in range(head_count):
            for layer in range(layer_count):
                categories["same_query_head_cross_layer"].append(
                    (
                        query_index,
                        layer,
                        query_head,
                        (layer + 7) % layer_count,
                        query_head,
                    )
                )

    while len(categories["random_cross_layer"]) < sample_cap:
        query_index = int(rng.integers(query_count))
        left_layer = int(rng.integers(layer_count))
        right_layer = int(rng.integers(layer_count - 1))
        if right_layer >= left_layer:
            right_layer += 1
        categories["random_cross_layer"].append(
            (
                query_index,
                left_layer,
                int(rng.integers(head_count)),
                right_layer,
                int(rng.integers(head_count)),
            )
        )

    rows: list[dict[str, Any]] = []
    for category, pairs in categories.items():
        if len(pairs) > sample_cap:
            indices = rng.choice(len(pairs), size=sample_cap, replace=False)
            pairs = [pairs[int(index)] for index in indices]
        overlaps = [
            jaccard(values[q, l1, h1], values[q, l2, h2])
            for q, l1, h1, l2, h2 in pairs
        ]
        stats = percentile_summary(overlaps)
        rows.append(
            {
                "depth": depth,
                "category": category,
                "pairs": len(overlaps),
                "mean_jaccard": stats["mean"],
                "median_jaccard": stats["median"],
                "p95_jaccard": stats["p95"],
                "nonzero_overlap": float(np.mean(np.asarray(overlaps) > 0)),
                "independent_uniform_expected_jaccard": depth
                / (2 * num_blocks - depth),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = np.load(args.topk_npz)
    block_ids = np.asarray(payload["block_ids"], dtype=np.int64)
    layers = np.asarray(payload["layers"], dtype=np.int64)
    queries = read_jsonl(Path(args.queries_jsonl))[: block_ids.shape[0]]
    depths = sorted({int(item) for item in args.depths.split(",")})
    if depths[-1] > block_ids.shape[-1]:
        raise ValueError("requested depth exceeds stored Top-K depth")
    if len(queries) != block_ids.shape[0]:
        raise ValueError("query count does not match frozen rankings")

    rng = np.random.default_rng(args.seed)
    query_count, layer_count, head_count, stored_depth = block_ids.shape
    depth_rows: list[dict[str, Any]] = []
    per_layer_rows: list[dict[str, Any]] = []

    for depth in depths:
        union_sizes: list[int] = []
        layer_union_sizes: list[int] = []
        query_hits: list[bool] = []
        gold_head_counts: list[int] = []
        hits = build_head_hits(block_ids, queries, depth).reshape(
            layer_count, head_count, query_count
        )
        for query_index, query in enumerate(queries):
            candidates = block_ids[query_index, :, :, :depth]
            union_sizes.append(int(np.unique(candidates).size))
            layer_union_sizes.extend(
                int(np.unique(candidates[layer_index]).size)
                for layer_index in range(layer_count)
            )
            query_hits.append(bool(hits[:, :, query_index].any()))
            gold_head_counts.append(int(hits[:, :, query_index].sum()))

        union_stats = percentile_summary(union_sizes)
        layer_union_stats = percentile_summary(layer_union_sizes)
        nonzero_gold_heads = [count for count in gold_head_counts if count]
        depth_rows.append(
            {
                "depth_per_head": depth,
                "total_head_slots": layer_count * head_count * depth,
                "mean_unique_blocks": union_stats["mean"],
                "median_unique_blocks": union_stats["median"],
                "mean_slot_redundancy": 1
                - union_stats["mean"] / (layer_count * head_count * depth),
                "mean_corpus_fraction": union_stats["mean"] / args.num_blocks,
                "mean_unique_blocks_per_layer": layer_union_stats["mean"],
                "gold_union_recall": float(np.mean(query_hits)),
                "mean_gold_nominating_heads": float(np.mean(gold_head_counts)),
                "conditional_gold_nominating_heads": float(
                    np.mean(nonzero_gold_heads) if nonzero_gold_heads else 0.0
                ),
            }
        )

        for layer_index, layer in enumerate(layers):
            layer_recall = float(hits[layer_index].any(axis=0).mean())
            layer_unique = [
                int(np.unique(block_ids[q, layer_index, :, :depth]).size)
                for q in range(query_count)
            ]
            per_layer_rows.append(
                {
                    "depth_per_head": depth,
                    "layer": int(layer),
                    "gold_recall_any_head": layer_recall,
                    "mean_unique_blocks": float(np.mean(layer_unique)),
                    "slot_redundancy": 1
                    - float(np.mean(layer_unique)) / (head_count * depth),
                }
            )

    max_depth = depths[-1]
    max_depth_hits = build_head_hits(block_ids, queries, max_depth)
    subset_sizes = [1, 2, 4, 8, 16, 32, 64]
    cv_rows, cv_meta = selection_cross_validation(
        max_depth_hits,
        queries,
        subset_sizes,
        args.cv_splits,
        args.random_subsets_per_split,
        rng,
    )
    overlap_rows = overlap_diagnostics(
        block_ids,
        max_depth,
        args.overlap_samples,
        rng,
        args.num_blocks,
    )

    top_head_rows: list[dict[str, Any]] = []
    flat_hits = max_depth_hits.sum(axis=1)
    for flat_index in np.argsort(flat_hits)[::-1][:32]:
        layer_index, query_head = divmod(int(flat_index), head_count)
        top_head_rows.append(
            {
                "layer": int(layers[layer_index]),
                "query_head": query_head,
                "kv_head": query_head // 2,
                "gold_hits": int(flat_hits[flat_index]),
                "gold_recall": float(flat_hits[flat_index] / query_count),
            }
        )

    top1_unique = [
        int(np.unique(block_ids[:, layer_index, query_head, 0]).size)
        for layer_index in range(layer_count)
        for query_head in range(head_count)
    ]
    block_query_frequency: Counter[int] = Counter()
    for query_index in range(query_count):
        block_query_frequency.update(
            int(item) for item in np.unique(block_ids[query_index, :, :, :max_depth])
        )
    hub_rows = [
        {"block_id": block_id, "queries_nominated": count}
        for block_id, count in block_query_frequency.most_common(100)
    ]
    frequency_values = np.asarray(
        list(block_query_frequency.values()), dtype=np.int64
    )
    universal_hubs = {
        block_id
        for block_id, count in block_query_frequency.items()
        if count == query_count
    }
    gold_hub_frequencies = [
        max(
            (
                block_query_frequency.get(int(block_id), 0)
                for block_id in query.get("gold_block_ids", [])
            ),
            default=0,
        )
        for query in queries
    ]
    universal_hub_gold_queries = sum(
        bool(universal_hubs & set(int(item) for item in query.get("gold_block_ids", [])))
        for query in queries
    )

    summary = {
        "source": "frozen real 10M all-layer/all-query-head QK Top-K rankings",
        "contains_synthetic_vectors": False,
        "selection_uses_gold_for_rankings": False,
        "num_blocks": args.num_blocks,
        "block_tokens": 256,
        "num_tokens": args.num_blocks * 256,
        "queries": query_count,
        "layers": [int(item) for item in layers],
        "query_heads_per_layer": head_count,
        "total_query_heads": layer_count * head_count,
        "stored_depth": stored_depth,
        "depth_summary": depth_rows,
        "overlap_summary": overlap_rows,
        "head_selection_cross_validation": {
            **cv_meta,
            "nomination_depth": max_depth,
            "rows": cv_rows,
        },
        "query_specificity": {
            "unique_top1_blocks_per_head_across_queries": percentile_summary(
                top1_unique
            ),
            "most_nominated_block_query_frequency": hub_rows[:20],
            "block_hubness": {
                "blocks_ever_nominated": int(len(frequency_values)),
                "median_query_frequency": float(np.median(frequency_values)),
                "p95_query_frequency": float(np.percentile(frequency_values, 95)),
                "blocks_nominated_by_all_queries": len(universal_hubs),
                "blocks_nominated_by_at_least_half_queries": int(
                    np.sum(frequency_values >= query_count / 2)
                ),
                "universal_hub_gold_queries": universal_hub_gold_queries,
                "gold_block_query_frequency": percentile_summary(
                    gold_hub_frequencies
                ),
            },
        },
        "gold_usage_contract": {
            "frozen_per_head_rankings": "target-free",
            "descriptive_top_heads": "uses all gold labels for diagnosis only",
            "head_subset_selection": "uses train gold labels and reports held-out test recall",
        },
        "interpretation_contract": {
            "measures": "head-specific candidate-set geometry and gold nomination",
            "does_not_measure": [
                "final generation quality",
                "full-attention token recall",
                "wall-clock speedup",
                "1B-token execution",
            ],
        },
    }

    write_csv(output_dir / "depth_summary.csv", depth_rows)
    write_csv(output_dir / "per_layer_summary.csv", per_layer_rows)
    write_csv(output_dir / "overlap_summary.csv", overlap_rows)
    write_csv(output_dir / "head_selection_cv.csv", cv_rows)
    write_csv(output_dir / "top_heads.csv", top_head_rows)
    write_csv(output_dir / "block_hubs.csv", hub_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
