from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_unsupervised_head_gate import (
    budget_metrics,
    nomination_features,
    read_jsonl,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired comparison of a strict label-free head gate and matched BM25."
    )
    parser.add_argument("--raw_topk_npz", required=True)
    parser.add_argument("--candidate_topk_npz", required=True)
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument("--bm25_query_results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gate_feature", default="raw_top1_block_diversity")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--rrf_constant", type=float, default=60.0)
    parser.add_argument("--bootstrap_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def exact_mcnemar_p(improved: int, worsened: int) -> float:
    discordant = improved + worsened
    if discordant == 0:
        return 1.0
    tail = min(improved, worsened)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def paired_summary(
    kv_hits: np.ndarray,
    baseline_hits: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    improved = int(np.sum(kv_hits & ~baseline_hits))
    worsened = int(np.sum(~kv_hits & baseline_hits))
    delta = kv_hits.astype(np.float32) - baseline_hits.astype(np.float32)
    sample_indices = rng.integers(
        0, len(kv_hits), size=(bootstrap_samples, len(kv_hits))
    )
    bootstrap = delta[sample_indices].mean(axis=1)
    return {
        "kv_recall": float(kv_hits.mean()),
        "baseline_recall": float(baseline_hits.mean()),
        "kv_minus_baseline": float(delta.mean()),
        "delta_ci_low": float(np.percentile(bootstrap, 2.5)),
        "delta_ci_high": float(np.percentile(bootstrap, 97.5)),
        "kv_only_wins": improved,
        "baseline_only_wins": worsened,
        "ties": len(kv_hits) - improved - worsened,
        "mcnemar_exact_p": exact_mcnemar_p(improved, worsened),
    }


def rrf_ranking(
    ids: np.ndarray, target_blocks: int, rrf_constant: float, num_blocks: int
) -> list[int]:
    depth = ids.shape[1]
    weights = np.tile(
        1.0 / (rrf_constant + np.arange(1, depth + 1, dtype=np.float64)),
        ids.shape[0],
    )
    flat_ids = ids.reshape(-1)
    scores = np.bincount(flat_ids, weights=weights, minlength=num_blocks)
    nominated = np.flatnonzero(scores)
    return nominated[
        np.lexsort((nominated, -scores[nominated]))[:target_blocks]
    ].tolist()


def equal_rank_fusion(
    left: list[int], right: list[int], target_blocks: int, rrf_constant: float
) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in (left, right):
        for rank, block_id in enumerate(ranking, start=1):
            scores[int(block_id)] = scores.get(int(block_id), 0.0) + 1.0 / (
                rrf_constant + rank
            )
    return sorted(scores, key=lambda block_id: (-scores[block_id], block_id))[
        :target_blocks
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = read_jsonl(Path(args.queries_jsonl))
    rng = np.random.default_rng(args.seed)

    with np.load(Path(args.raw_topk_npz)) as payload:
        raw_ids_4d = payload["block_ids"]
        raw_scores_4d = payload["scores"]
        fold_ids = payload["fold_ids"].astype(np.int64)
    with np.load(Path(args.candidate_topk_npz)) as payload:
        candidate_ids_4d = payload["block_ids"]
        candidate_fold_ids = payload["fold_ids"].astype(np.int64)
    if not np.array_equal(fold_ids, candidate_fold_ids):
        raise ValueError("raw and candidate fold assignments differ")
    query_count, layer_count, heads_per_layer, topk = raw_ids_4d.shape
    if args.depth > topk or args.heads > layer_count * heads_per_layer:
        raise ValueError("requested head/depth budget exceeds stored rankings")
    raw_ids = raw_ids_4d.reshape(query_count, -1, topk)
    raw_scores = raw_scores_4d.reshape(query_count, -1, topk)
    candidate_ids = candidate_ids_4d.reshape(query_count, -1, topk)
    num_blocks = int(candidate_ids.max()) + 1

    kv_hits = np.zeros(query_count, dtype=bool)
    kv_rankings: list[list[int] | None] = [None] * query_count
    selected_heads_by_fold: dict[int, list[int]] = {}
    for fold in sorted(int(item) for item in np.unique(fold_ids)):
        train = np.flatnonzero(fold_ids != fold)
        test = np.flatnonzero(fold_ids == fold)
        features = nomination_features(raw_ids, raw_scores, train)
        if args.gate_feature not in features:
            raise ValueError(f"unknown gate feature: {args.gate_feature}")
        selected = np.argsort(-features[args.gate_feature], kind="stable")[
            : args.heads
        ]
        _union_hits, rrf_hits, _unique_counts = budget_metrics(
            candidate_ids,
            queries,
            test,
            selected,
            args.depth,
            args.target_blocks,
            args.rrf_constant,
            num_blocks,
        )
        kv_hits[test] = rrf_hits
        for query_index in test:
            kv_rankings[int(query_index)] = rrf_ranking(
                candidate_ids[int(query_index), selected, : args.depth],
                args.target_blocks,
                args.rrf_constant,
                num_blocks,
            )
        selected_heads_by_fold[fold] = [int(item) for item in selected]
    if any(ranking is None for ranking in kv_rankings):
        raise RuntimeError("failed to rank every query")

    bm25_rows = read_csv(Path(args.bm25_query_results))
    bm25_methods = sorted({row["method"] for row in bm25_rows})
    bm25_by_method: dict[str, dict[int, bool]] = {
        method: {
            int(row["query_id"]): bool(float(row["answer_block_recall"]))
            for row in bm25_rows
            if row["method"] == method
        }
        for method in bm25_methods
    }
    query_ids = [int(query["query_id"]) for query in queries]
    bm25_hits = {
        method: np.asarray(
            [bm25_by_method[method][query_id] for query_id in query_ids], dtype=bool
        )
        for method in bm25_methods
    }
    bm25_rankings = {
        method: {
            int(row["query_id"]): [
                int(item) for item in json.loads(row["ranked_block_ids"])
            ]
            for row in bm25_rows
            if row["method"] == method
        }
        for method in bm25_methods
    }

    paired_rows: list[dict[str, Any]] = []
    fusion_rows: list[dict[str, Any]] = []
    fusion_hits_by_method: dict[str, np.ndarray] = {}
    for method in bm25_methods:
        paired_rows.append(
            {
                "baseline_method": method,
                **paired_summary(
                    kv_hits,
                    bm25_hits[method],
                    args.bootstrap_samples,
                    rng,
                ),
            }
        )
        fused_hits = np.zeros(query_count, dtype=bool)
        for query_index, query in enumerate(queries):
            assert kv_rankings[query_index] is not None
            fused_ranking = equal_rank_fusion(
                bm25_rankings[method][int(query["query_id"])],
                kv_rankings[query_index],
                args.target_blocks,
                args.rrf_constant,
            )
            gold = set(int(item) for item in query.get("gold_block_ids", []))
            fused_hits[query_index] = bool(gold & set(fused_ranking))
        fusion_hits_by_method[method] = fused_hits
        comparison = paired_summary(
            fused_hits,
            bm25_hits[method],
            args.bootstrap_samples,
            rng,
        )
        fusion_rows.append(
            {
                "baseline_method": method,
                "baseline_recall": comparison["baseline_recall"],
                "equal_rrf_fusion_recall": comparison["kv_recall"],
                "fusion_minus_baseline": comparison["kv_minus_baseline"],
                "delta_ci_low": comparison["delta_ci_low"],
                "delta_ci_high": comparison["delta_ci_high"],
                "fusion_only_wins": comparison["kv_only_wins"],
                "baseline_only_wins": comparison["baseline_only_wins"],
                "ties": comparison["ties"],
                "mcnemar_exact_p": comparison["mcnemar_exact_p"],
                "oracle_union_recall": float(
                    np.mean(kv_hits | bm25_hits[method])
                ),
            }
        )

    query_rows = []
    for query_index, query in enumerate(queries):
        row: dict[str, Any] = {
            "query_index": query_index,
            "query_id": int(query["query_id"]),
            "dataset": str(query["dataset"]),
            "fold": int(fold_ids[query_index]),
            "kv_hit": int(kv_hits[query_index]),
        }
        for method in bm25_methods:
            row[f"{method}_hit"] = int(bm25_hits[method][query_index])
            row[f"{method}_equal_rrf_fusion_hit"] = int(
                fusion_hits_by_method[method][query_index]
            )
        query_rows.append(row)

    dataset_rows: list[dict[str, Any]] = []
    datasets = np.asarray([str(query["dataset"]) for query in queries])
    for dataset in np.unique(datasets):
        mask = datasets == dataset
        row = {
            "dataset": str(dataset),
            "queries": int(mask.sum()),
            "kv_recall": float(kv_hits[mask].mean()),
        }
        for method in bm25_methods:
            row[f"{method}_recall"] = float(bm25_hits[method][mask].mean())
            row[f"{method}_equal_rrf_fusion_recall"] = float(
                fusion_hits_by_method[method][mask].mean()
            )
        dataset_rows.append(row)

    write_csv(output_dir / "paired_summary.csv", paired_rows)
    write_csv(output_dir / "query_results.csv", query_rows)
    write_csv(output_dir / "dataset_summary.csv", dataset_rows)
    write_csv(output_dir / "fusion_summary.csv", fusion_rows)
    summary = {
        "experiment": "matched_10m_head_gate_vs_bm25",
        "queries": query_count,
        "selection_uses_gold": False,
        "selection_uses_test_queries": False,
        "gold_used_only_for_evaluation": True,
        "gate_feature": args.gate_feature,
        "heads": args.heads,
        "depth_per_head": args.depth,
        "target_blocks": args.target_blocks,
        "candidate_method": "cross-fitted zscore QK + train-only label-free head gate + RRF",
        "bm25_methods": bm25_methods,
        "paired_summary": paired_rows,
        "equal_rrf_fusion_summary": fusion_rows,
        "dataset_summary": dataset_rows,
        "selected_heads_by_fold": selected_heads_by_fold,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
