from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate train-only, label-free head gates from raw per-head rankings "
            "on held-out prior-debiased block retrieval."
        )
    )
    parser.add_argument("--raw_topk_npz", required=True)
    parser.add_argument("--candidate_topk_npz", required=True)
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--head_sizes", default="1,2,4,8,16,32,64")
    parser.add_argument("--depths", default="8,16")
    parser.add_argument("--target_blocks", type=int, default=39)
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


def nomination_features(
    raw_ids: np.ndarray, raw_scores: np.ndarray, train: np.ndarray
) -> dict[str, np.ndarray]:
    head_count = raw_ids.shape[1]
    topk = raw_ids.shape[2]
    train_count = len(train)
    top1_diversity = np.empty(head_count, dtype=np.float64)
    nomination_diversity = np.empty(head_count, dtype=np.float64)
    nomination_entropy = np.empty(head_count, dtype=np.float64)
    for head in range(head_count):
        top1 = raw_ids[train, head, 0]
        all_ids = raw_ids[train, head].reshape(-1)
        _unique, counts = np.unique(all_ids, return_counts=True)
        probabilities = counts.astype(np.float64) / counts.sum()
        top1_diversity[head] = len(np.unique(top1)) / train_count
        nomination_diversity[head] = len(counts) / (train_count * topk)
        nomination_entropy[head] = float(
            -(probabilities * np.log(probabilities)).sum()
            / np.log(train_count * topk)
        )
    train_scores = raw_scores[train]
    return {
        "raw_top1_block_diversity": top1_diversity,
        "raw_topk_block_diversity": nomination_diversity,
        "raw_topk_nomination_entropy": nomination_entropy,
        "raw_mean_top1_score": train_scores[:, :, 0].mean(axis=0),
        "raw_top1_score_query_variance": train_scores[:, :, 0].var(axis=0),
        "raw_topk_score_query_variance": train_scores.var(axis=0).mean(axis=1),
        "raw_mean_top1_topk_gap": (
            train_scores[:, :, 0] - train_scores[:, :, -1]
        ).mean(axis=0),
    }


def query_hits(
    candidate_ids: np.ndarray,
    queries: list[dict[str, Any]],
    indices: np.ndarray,
    selected_heads: np.ndarray,
) -> np.ndarray:
    hits = np.zeros(len(indices), dtype=bool)
    for output_index, query_index in enumerate(indices):
        gold = np.asarray(
            queries[int(query_index)].get("gold_block_ids", []), dtype=np.int64
        )
        hits[output_index] = np.isin(
            candidate_ids[int(query_index), selected_heads], gold
        ).any()
    return hits


def budget_metrics(
    candidate_ids: np.ndarray,
    queries: list[dict[str, Any]],
    indices: np.ndarray,
    selected_heads: np.ndarray,
    depth: int,
    target_blocks: int,
    rrf_constant: float,
    num_blocks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    union_hits = np.zeros(len(indices), dtype=bool)
    rrf_hits = np.zeros(len(indices), dtype=bool)
    unique_counts = np.zeros(len(indices), dtype=np.int32)
    rank_weights = np.tile(
        1.0 / (rrf_constant + np.arange(1, depth + 1, dtype=np.float64)),
        len(selected_heads),
    )
    for output_index, query_index in enumerate(indices):
        gold = np.asarray(
            queries[int(query_index)].get("gold_block_ids", []), dtype=np.int64
        )
        ids = candidate_ids[int(query_index), selected_heads, :depth]
        flat_ids = ids.reshape(-1)
        unique_ids = np.unique(flat_ids)
        unique_counts[output_index] = len(unique_ids)
        union_hits[output_index] = np.isin(gold, unique_ids).any()
        rrf_scores = np.bincount(
            flat_ids, weights=rank_weights, minlength=num_blocks
        )
        nominated = np.flatnonzero(rrf_scores)
        ranking = nominated[
            np.lexsort((nominated, -rrf_scores[nominated]))[:target_blocks]
        ]
        rrf_hits[output_index] = np.isin(gold, ranking).any()
    return union_hits, rrf_hits, unique_counts


def macro_recall(hits: np.ndarray, datasets: np.ndarray) -> float:
    return float(
        np.mean([hits[datasets == dataset].mean() for dataset in np.unique(datasets)])
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = read_jsonl(Path(args.queries_jsonl))
    head_sizes = sorted({int(item) for item in args.head_sizes.split(",")})
    depths = sorted({int(item) for item in args.depths.split(",")})
    rng = np.random.default_rng(args.seed)

    with np.load(Path(args.raw_topk_npz)) as payload:
        raw_ids_4d = payload["block_ids"]
        raw_scores_4d = payload["scores"]
        layers = payload["layers"].astype(np.int64)
        fold_ids = payload["fold_ids"].astype(np.int64)
    with np.load(Path(args.candidate_topk_npz)) as payload:
        candidate_ids_4d = payload["block_ids"]
        candidate_fold_ids = payload["fold_ids"].astype(np.int64)
    if not np.array_equal(fold_ids, candidate_fold_ids):
        raise ValueError("raw and candidate fold assignments differ")
    if raw_ids_4d.shape != candidate_ids_4d.shape:
        raise ValueError("raw and candidate ranking shapes differ")
    if len(queries) != raw_ids_4d.shape[0]:
        raise ValueError("query count does not match rankings")

    query_count, layer_count, heads_per_layer, topk = raw_ids_4d.shape
    head_count = layer_count * heads_per_layer
    if max(head_sizes) > head_count:
        raise ValueError("head size exceeds available heads")
    if max(depths) > topk:
        raise ValueError("depth exceeds stored per-head ranking")
    raw_ids = raw_ids_4d.reshape(query_count, head_count, topk)
    raw_scores = raw_scores_4d.reshape(query_count, head_count, topk)
    candidate_ids = candidate_ids_4d.reshape(query_count, head_count, topk)
    num_blocks = int(candidate_ids.max()) + 1
    datasets = np.asarray([str(query["dataset"]) for query in queries])

    feature_hits: dict[tuple[str, int], np.ndarray] = {}
    random_replicate_hits = {
        size: np.zeros(
            (args.random_subsets_per_fold, query_count), dtype=bool
        )
        for size in head_sizes
    }
    selected_by_fold: dict[tuple[str, int], list[set[int]]] = {}
    selected_rows: list[dict[str, Any]] = []
    budget_union_hits: dict[tuple[str, int, int], np.ndarray] = {}
    budget_rrf_hits: dict[tuple[str, int, int], np.ndarray] = {}
    budget_unique_counts: dict[tuple[str, int, int], np.ndarray] = {}
    feature_names: list[str] | None = None

    for fold in sorted(int(item) for item in np.unique(fold_ids)):
        train = np.flatnonzero(fold_ids != fold)
        test = np.flatnonzero(fold_ids == fold)
        features = nomination_features(raw_ids, raw_scores, train)
        if feature_names is None:
            feature_names = list(features)
            feature_hits = {
                (feature, size): np.zeros(query_count, dtype=bool)
                for feature in feature_names
                for size in head_sizes
            }
            budget_union_hits = {
                (feature, size, depth): np.zeros(query_count, dtype=bool)
                for feature in feature_names
                for size in head_sizes
                for depth in depths
            }
            budget_rrf_hits = {
                key: np.zeros(query_count, dtype=bool) for key in budget_union_hits
            }
            budget_unique_counts = {
                key: np.zeros(query_count, dtype=np.int32)
                for key in budget_union_hits
            }
        for feature, values in features.items():
            order = np.argsort(-values, kind="stable")
            for size in head_sizes:
                selected = order[:size]
                feature_hits[(feature, size)][test] = query_hits(
                    candidate_ids, queries, test, selected
                )
                for depth in depths:
                    union_hits, rrf_hits, unique_counts = budget_metrics(
                        candidate_ids,
                        queries,
                        test,
                        selected,
                        depth,
                        args.target_blocks,
                        args.rrf_constant,
                        num_blocks,
                    )
                    key = (feature, size, depth)
                    budget_union_hits[key][test] = union_hits
                    budget_rrf_hits[key][test] = rrf_hits
                    budget_unique_counts[key][test] = unique_counts
                selected_by_fold.setdefault((feature, size), []).append(
                    set(int(item) for item in selected)
                )
                for selected_rank, flat_head in enumerate(selected, start=1):
                    layer_index, query_head = divmod(
                        int(flat_head), heads_per_layer
                    )
                    selected_rows.append(
                        {
                            "fold": fold,
                            "feature": feature,
                            "head_count": size,
                            "selected_rank": selected_rank,
                            "flat_head": int(flat_head),
                            "layer": int(layers[layer_index]),
                            "query_head": query_head,
                            "train_feature_value": float(values[flat_head]),
                        }
                    )

        for size in head_sizes:
            for random_index in range(args.random_subsets_per_fold):
                selected = rng.choice(head_count, size=size, replace=False)
                random_replicate_hits[size][random_index, test] = query_hits(
                    candidate_ids, queries, test, selected
                )

    assert feature_names is not None
    summary_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    for feature in feature_names:
        for size in head_sizes:
            hits = feature_hits[(feature, size)]
            random_hits = random_replicate_hits[size]
            random_probability = random_hits.mean(axis=0)
            random_recalls = random_hits.mean(axis=1)
            empirical_p = (
                1 + int(np.sum(random_recalls >= hits.mean()))
            ) / (len(random_recalls) + 1)
            summary_rows.append(
                {
                    "feature": feature,
                    "heads": size,
                    "test_recall": float(hits.mean()),
                    "test_macro_recall": macro_recall(hits, datasets),
                    "random_expected_recall": float(random_probability.mean()),
                    "random_recall_p05": float(np.percentile(random_recalls, 5)),
                    "random_recall_p95": float(np.percentile(random_recalls, 95)),
                    "empirical_random_p": float(empirical_p),
                    "selected_minus_random": float(
                        hits.mean() - random_probability.mean()
                    ),
                }
            )
            for dataset in np.unique(datasets):
                mask = datasets == dataset
                dataset_rows.append(
                    {
                        "feature": feature,
                        "heads": size,
                        "dataset": str(dataset),
                        "queries": int(mask.sum()),
                        "test_recall": float(hits[mask].mean()),
                        "random_expected_recall": float(
                            random_probability[mask].mean()
                        ),
                    }
                )
            selected_sets = selected_by_fold[(feature, size)]
            jaccards = [
                len(left & right) / len(left | right)
                for left, right in combinations(selected_sets, 2)
            ]
            stability_rows.append(
                {
                    "feature": feature,
                    "heads": size,
                    "fold_pairs": len(jaccards),
                    "mean_fold_jaccard": float(np.mean(jaccards)),
                    "min_fold_jaccard": float(np.min(jaccards)),
                    "max_fold_jaccard": float(np.max(jaccards)),
                }
            )
            for depth in depths:
                key = (feature, size, depth)
                union_hits = budget_union_hits[key]
                rrf_hits = budget_rrf_hits[key]
                unique_counts = budget_unique_counts[key]
                budget_rows.append(
                    {
                        "feature": feature,
                        "heads": size,
                        "depth_per_head": depth,
                        "head_slots": size * depth,
                        "mean_unique_blocks": float(unique_counts.mean()),
                        "mean_union_tokens": float(
                            unique_counts.mean() * args.block_tokens
                        ),
                        "union_recall": float(union_hits.mean()),
                        "union_macro_recall": macro_recall(union_hits, datasets),
                        f"rrf_recall_at_{args.target_blocks}": float(
                            rrf_hits.mean()
                        ),
                        f"rrf_macro_recall_at_{args.target_blocks}": macro_recall(
                            rrf_hits, datasets
                        ),
                    }
                )

    write_csv(output_dir / "feature_summary.csv", summary_rows)
    write_csv(output_dir / "dataset_summary.csv", dataset_rows)
    write_csv(output_dir / "selected_heads.csv", selected_rows)
    write_csv(output_dir / "selection_stability.csv", stability_rows)
    write_csv(output_dir / "budget_summary.csv", budget_rows)
    summary = {
        "experiment": "strict_cross_fitted_label_free_head_gate",
        "exploratory_feature_comparison": True,
        "external_holdout_required": True,
        "selection_uses_gold": False,
        "selection_uses_test_queries": False,
        "gold_used_only_for_held_out_evaluation": True,
        "gate_source": "raw per-head Top-K block IDs and scores on train queries",
        "candidate_source": "cross-fitted prior-debiased per-head rankings",
        "queries": query_count,
        "folds": int(len(np.unique(fold_ids))),
        "layers": layers.tolist(),
        "heads_per_layer": heads_per_layer,
        "total_heads": head_count,
        "topk_per_head": topk,
        "evaluated_depths": depths,
        "target_blocks": args.target_blocks,
        "block_tokens": args.block_tokens,
        "random_subsets_per_fold": args.random_subsets_per_fold,
        "seed": args.seed,
        "feature_summary": summary_rows,
        "selection_stability": stability_rows,
        "budget_summary": budget_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
