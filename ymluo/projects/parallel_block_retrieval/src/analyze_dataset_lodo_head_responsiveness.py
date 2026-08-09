from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leave one entire dataset out when estimating label-free head "
            "Top1-block responsiveness, then evaluate raw QK retrieval."
        )
    )
    parser.add_argument("--raw_topk_npz", required=True)
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--head_sizes", default="1,4,16,64")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--num_blocks", type=int, default=39062)
    parser.add_argument("--block_tokens", type=int, default=256)
    parser.add_argument("--rrf_constant", type=float, default=60.0)
    parser.add_argument("--random_subsets_per_fold", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def top1_diversity(ids: np.ndarray, train: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            len(np.unique(ids[train, head, 0])) / len(train)
            for head in range(ids.shape[1])
        ],
        dtype=np.float64,
    )


def single_head_hits(
    ids: np.ndarray, queries: list[dict[str, Any]], test: np.ndarray, depth: int
) -> np.ndarray:
    hits = np.zeros((len(test), ids.shape[1]), dtype=bool)
    for output_index, query_index in enumerate(test):
        gold = np.asarray(
            queries[int(query_index)].get("gold_block_ids", []), dtype=np.int64
        )
        hits[output_index] = np.isin(ids[int(query_index), :, :depth], gold).any(
            axis=-1
        )
    return hits


def selected_metrics(
    ids: np.ndarray,
    queries: list[dict[str, Any]],
    test: np.ndarray,
    selected: np.ndarray,
    depth: int,
    target_blocks: int,
    num_blocks: int,
    rrf_constant: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    union_hits = np.zeros(len(test), dtype=bool)
    rrf_hits = np.zeros(len(test), dtype=bool)
    unique_counts = np.zeros(len(test), dtype=np.int32)
    rank_weights = np.tile(
        1.0 / (rrf_constant + np.arange(1, depth + 1, dtype=np.float64)),
        len(selected),
    )
    for output_index, query_index in enumerate(test):
        gold = np.asarray(
            queries[int(query_index)].get("gold_block_ids", []), dtype=np.int64
        )
        candidates = ids[int(query_index), selected, :depth]
        flat = candidates.reshape(-1)
        unique = np.unique(flat)
        unique_counts[output_index] = len(unique)
        union_hits[output_index] = np.isin(gold, unique).any()
        scores = np.bincount(flat, weights=rank_weights, minlength=num_blocks)
        nominated = np.flatnonzero(scores)
        ranking = nominated[
            np.lexsort((nominated, -scores[nominated]))[:target_blocks]
        ]
        rrf_hits[output_index] = np.isin(gold, ranking).any()
    return union_hits, rrf_hits, unique_counts


def selected_union_hits(
    ids: np.ndarray,
    queries: list[dict[str, Any]],
    test: np.ndarray,
    selected: np.ndarray,
    depth: int,
) -> np.ndarray:
    hits = np.zeros(len(test), dtype=bool)
    for output_index, query_index in enumerate(test):
        gold = np.asarray(
            queries[int(query_index)].get("gold_block_ids", []), dtype=np.int64
        )
        hits[output_index] = np.isin(
            ids[int(query_index), selected, :depth], gold
        ).any()
    return hits


def finite_correlation(
    feature: np.ndarray, target: np.ndarray
) -> dict[str, float | None]:
    if np.all(target == target[0]):
        return {
            "spearman": None,
            "spearman_p": None,
            "pearson": None,
            "pearson_p": None,
        }
    spearman = spearmanr(feature, target)
    pearson = pearsonr(feature, target)
    return {
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = read_jsonl(Path(args.queries_jsonl))
    head_sizes = sorted({int(item) for item in args.head_sizes.split(",")})
    rng = np.random.default_rng(args.seed)

    with np.load(Path(args.raw_topk_npz)) as payload:
        ids_4d = payload["block_ids"]
        layers = payload["layers"].astype(np.int64)
    query_count, layer_count, heads_per_layer, stored_depth = ids_4d.shape
    if len(queries) != query_count:
        raise ValueError("query metadata and ranking counts differ")
    if args.depth > stored_depth or max(head_sizes) > layer_count * heads_per_layer:
        raise ValueError("requested budget exceeds stored rankings")
    ids = ids_4d.reshape(query_count, layer_count * heads_per_layer, stored_depth)
    datasets = np.asarray([str(query["dataset"]) for query in queries])
    unique_datasets = sorted(str(item) for item in np.unique(datasets))

    correlation_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    per_query_union = {
        size: np.zeros(query_count, dtype=bool) for size in head_sizes
    }
    per_query_rrf = {
        size: np.zeros(query_count, dtype=bool) for size in head_sizes
    }
    per_query_unique = {
        size: np.zeros(query_count, dtype=np.int32) for size in head_sizes
    }
    random_replicates = {
        size: np.zeros(
            (args.random_subsets_per_fold, query_count), dtype=bool
        )
        for size in head_sizes
    }
    selected_sets: dict[int, list[set[int]]] = {size: [] for size in head_sizes}
    diversity_ranks: list[np.ndarray] = []

    for heldout_dataset in unique_datasets:
        train = np.flatnonzero(datasets != heldout_dataset)
        test = np.flatnonzero(datasets == heldout_dataset)
        diversity = top1_diversity(ids, train)
        order = np.argsort(-diversity, kind="stable")
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order), dtype=np.int64)
        diversity_ranks.append(rank)
        head_hits = single_head_hits(ids, queries, test, args.depth)
        head_recall = head_hits.mean(axis=0)
        correlation_rows.append(
            {
                "heldout_dataset": heldout_dataset,
                "test_queries": len(test),
                **finite_correlation(diversity, head_recall),
                "mean_all_head_recall": float(head_recall.mean()),
                "top16_diversity_head_mean_recall": float(
                    head_recall[order[:16]].mean()
                ),
            }
        )

        for size in head_sizes:
            selected = order[:size]
            selected_sets[size].append(set(int(item) for item in selected))
            union_hits, rrf_hits, unique_counts = selected_metrics(
                ids,
                queries,
                test,
                selected,
                args.depth,
                args.target_blocks,
                args.num_blocks,
                args.rrf_constant,
            )
            per_query_union[size][test] = union_hits
            per_query_rrf[size][test] = rrf_hits
            per_query_unique[size][test] = unique_counts
            for selected_rank, flat_head in enumerate(selected, start=1):
                layer_index, query_head = divmod(
                    int(flat_head), heads_per_layer
                )
                selected_rows.append(
                    {
                        "heldout_dataset": heldout_dataset,
                        "head_count": size,
                        "selected_rank": selected_rank,
                        "flat_head": int(flat_head),
                        "layer": int(layers[layer_index]),
                        "query_head": query_head,
                        "train_top1_diversity": float(diversity[flat_head]),
                        "heldout_single_head_recall": float(
                            head_recall[flat_head]
                        ),
                    }
                )
            for random_index in range(args.random_subsets_per_fold):
                random_heads = rng.choice(
                    ids.shape[1], size=size, replace=False
                )
                random_union = selected_union_hits(
                    ids,
                    queries,
                    test,
                    random_heads,
                    args.depth,
                )
                random_replicates[size][random_index, test] = random_union

    budget_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for size in head_sizes:
        union_hits = per_query_union[size]
        rrf_hits = per_query_rrf[size]
        unique_counts = per_query_unique[size]
        random_recalls = random_replicates[size].mean(axis=1)
        budget_rows.append(
            {
                "heads": size,
                "depth_per_head": args.depth,
                "head_slots": size * args.depth,
                "mean_unique_blocks": float(unique_counts.mean()),
                "mean_union_tokens": float(
                    unique_counts.mean() * args.block_tokens
                ),
                "union_recall": float(union_hits.mean()),
                f"rrf_recall_at_{args.target_blocks}": float(rrf_hits.mean()),
                "random_union_mean": float(random_recalls.mean()),
                "random_union_p95": float(np.percentile(random_recalls, 95)),
                "empirical_random_p": float(
                    (1 + np.sum(random_recalls >= union_hits.mean()))
                    / (len(random_recalls) + 1)
                ),
            }
        )
        for dataset in unique_datasets:
            mask = datasets == dataset
            dataset_rows.append(
                {
                    "heldout_dataset": dataset,
                    "queries": int(mask.sum()),
                    "heads": size,
                    "union_recall": float(union_hits[mask].mean()),
                    f"rrf_recall_at_{args.target_blocks}": float(
                        rrf_hits[mask].mean()
                    ),
                }
            )
        jaccards = [
            len(left & right) / len(left | right)
            for left, right in combinations(selected_sets[size], 2)
        ]
        stability_rows.append(
            {
                "heads": size,
                "fold_pairs": len(jaccards),
                "mean_selected_set_jaccard": float(np.mean(jaccards)),
                "min_selected_set_jaccard": float(np.min(jaccards)),
                "max_selected_set_jaccard": float(np.max(jaccards)),
            }
        )

    rank_correlations = [
        float(spearmanr(left, right).statistic)
        for left, right in combinations(diversity_ranks, 2)
    ]
    finite_spearman = [
        float(row["spearman"])
        for row in correlation_rows
        if row["spearman"] is not None
    ]
    write_csv(output_dir / "dataset_head_correlations.csv", correlation_rows)
    write_csv(output_dir / "budget_summary.csv", budget_rows)
    write_csv(output_dir / "dataset_summary.csv", dataset_rows)
    write_csv(output_dir / "selected_heads.csv", selected_rows)
    write_csv(output_dir / "selection_stability.csv", stability_rows)
    summary = {
        "experiment": "dataset_leave_one_out_raw_head_responsiveness",
        "exploratory_post_selection_stress_test": True,
        "new_external_queries_required_for_confirmation": True,
        "selection_uses_gold": False,
        "selection_uses_heldout_dataset_queries": False,
        "gold_used_only_for_heldout_evaluation": True,
        "queries": query_count,
        "datasets": {
            dataset: int(np.sum(datasets == dataset))
            for dataset in unique_datasets
        },
        "feature": "train-query raw Top1-block diversity",
        "depth_per_head": args.depth,
        "target_blocks": args.target_blocks,
        "mean_dataset_spearman_feature_vs_head_recall": float(
            np.mean(finite_spearman)
        ),
        "pooled_dataset_head_spearman": float(
            spearmanr(
                np.concatenate(
                    [
                        top1_diversity(
                            ids, np.flatnonzero(datasets != dataset)
                        )
                        for dataset in unique_datasets
                        if np.any(datasets == dataset)
                    ]
                ),
                np.concatenate(
                    [
                        single_head_hits(
                            ids,
                            queries,
                            np.flatnonzero(datasets == dataset),
                            args.depth,
                        ).mean(axis=0)
                        for dataset in unique_datasets
                        if np.any(datasets == dataset)
                    ]
                ),
            ).statistic
        ),
        "diversity_rank_stability": {
            "mean_pairwise_spearman": float(np.mean(rank_correlations)),
            "min_pairwise_spearman": float(np.min(rank_correlations)),
            "max_pairwise_spearman": float(np.max(rank_correlations)),
        },
        "correlations": correlation_rows,
        "budget_summary": budget_rows,
        "selection_stability": stability_rows,
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
