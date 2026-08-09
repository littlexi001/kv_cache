from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from run_all_head_consensus_retrieval import block_consensus, rank_consensus


METHODS = ("raw", "centered", "zscore")
CONSENSUS_MODES = ("layer_consensus", "head_vote", "rrf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw and cross-fitted prior-debiased all-head rankings."
    )
    parser.add_argument("--retrieval_dir", required=True)
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_blocks", type=int, default=39062)
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--depths", default="1,2,4,8,16")
    parser.add_argument("--rrf_constant", type=float, default=60.0)
    parser.add_argument("--bootstrap_samples", type=int, default=20000)
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


def macro_mean(hits: np.ndarray, datasets: np.ndarray) -> float:
    return float(
        np.mean([hits[datasets == dataset].mean() for dataset in np.unique(datasets)])
    )


def exact_mcnemar_p(improved: int, worsened: int) -> float:
    discordant = improved + worsened
    if discordant == 0:
        return 1.0
    tail = min(improved, worsened)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def paired_summary(
    raw_hits: np.ndarray,
    candidate_hits: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    improved = int(np.sum(~raw_hits & candidate_hits))
    worsened = int(np.sum(raw_hits & ~candidate_hits))
    count = len(raw_hits)
    deltas = candidate_hits.astype(np.float32) - raw_hits.astype(np.float32)
    sample_indices = rng.integers(0, count, size=(bootstrap_samples, count))
    bootstrap = deltas[sample_indices].mean(axis=1)
    return {
        "raw_recall": float(raw_hits.mean()),
        "candidate_recall": float(candidate_hits.mean()),
        "delta": float(deltas.mean()),
        "delta_ci_low": float(np.percentile(bootstrap, 2.5)),
        "delta_ci_high": float(np.percentile(bootstrap, 97.5)),
        "improved": improved,
        "worsened": worsened,
        "ties": count - improved - worsened,
        "mcnemar_exact_p": exact_mcnemar_p(improved, worsened),
    }


def mean_set_jaccard(left: np.ndarray, right: np.ndarray, depth: int) -> float:
    values: list[float] = []
    for left_row, right_row in zip(
        left[:, :, :, :depth].reshape(-1, depth),
        right[:, :, :, :depth].reshape(-1, depth),
    ):
        left_set = set(int(item) for item in left_row if int(item) >= 0)
        right_set = set(int(item) for item in right_row if int(item) >= 0)
        union = left_set | right_set
        values.append(len(left_set & right_set) / len(union) if union else 1.0)
    return float(np.mean(values))


def evaluate_method(
    block_ids: np.ndarray,
    layers: list[int],
    queries: list[dict[str, Any]],
    datasets: np.ndarray,
    depths: list[int],
    target_blocks: int,
    rrf_constant: float,
    num_blocks: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    query_count = len(queries)
    rows: list[dict[str, Any]] = []
    hit_vectors: dict[str, np.ndarray] = {}
    max_depth = max(depths)
    query_frequency = np.zeros(num_blocks, dtype=np.int32)

    for query_index in range(query_count):
        nominated = np.unique(block_ids[query_index, :, :, :max_depth])
        nominated = nominated[(nominated >= 0) & (nominated < num_blocks)]
        query_frequency[nominated] += 1

    for depth in depths:
        union_hits = np.zeros(query_count, dtype=bool)
        union_sizes = np.zeros(query_count, dtype=np.int32)
        nominating_heads = np.zeros(query_count, dtype=np.int32)
        consensus_hits = {
            mode: np.zeros(query_count, dtype=bool) for mode in CONSENSUS_MODES
        }
        for query_index, query in enumerate(queries):
            gold = set(int(item) for item in query.get("gold_block_ids", []))
            head_ids = block_ids[query_index, :, :, :depth]
            union = set(int(item) for item in head_ids.reshape(-1))
            union.discard(-1)
            union_hits[query_index] = bool(gold & union)
            union_sizes[query_index] = len(union)
            candidates = head_ids.reshape(-1, depth)
            nominating_heads[query_index] = sum(
                bool(gold & set(int(item) for item in row)) for row in candidates
            )
            stats = block_consensus(
                block_ids[query_index],
                rrf_constant=rrf_constant,
                rank_limit=depth,
                layers=layers,
            )
            for mode in CONSENSUS_MODES:
                ranking = rank_consensus(stats, mode)[:target_blocks]
                consensus_hits[mode][query_index] = bool(gold & set(ranking))

        key = f"union_top{depth}"
        hit_vectors[key] = union_hits
        row: dict[str, Any] = {
            "depth_per_head": depth,
            "mean_unique_blocks": float(union_sizes.mean()),
            "median_unique_blocks": float(np.median(union_sizes)),
            "mean_corpus_fraction": float(union_sizes.mean() / num_blocks),
            "gold_union_recall": float(union_hits.mean()),
            "gold_union_macro_recall": macro_mean(union_hits, datasets),
            "mean_gold_nominating_heads": float(nominating_heads.mean()),
            "conditional_gold_nominating_heads": float(
                nominating_heads[union_hits].mean() if union_hits.any() else 0.0
            ),
        }
        for mode, hits in consensus_hits.items():
            hit_vectors[f"{mode}_top{target_blocks}_depth{depth}"] = hits
            row[f"{mode}_recall_at_{target_blocks}"] = float(hits.mean())
            row[f"{mode}_macro_recall_at_{target_blocks}"] = macro_mean(
                hits, datasets
            )
        rows.append(row)

    valid_frequency = query_frequency[query_frequency > 0]
    hubs = {
        "nominated_blocks": int(len(valid_frequency)),
        "median_query_frequency_among_nominated": float(
            np.median(valid_frequency) if len(valid_frequency) else 0.0
        ),
        "p95_query_frequency_among_nominated": float(
            np.percentile(valid_frequency, 95) if len(valid_frequency) else 0.0
        ),
        "blocks_in_at_least_half_queries": int(
            np.sum(query_frequency >= math.ceil(query_count / 2))
        ),
        "blocks_in_every_query": int(np.sum(query_frequency == query_count)),
        "max_query_frequency": int(query_frequency.max(initial=0)),
        "top_hubs": [
            {"block_id": int(block_id), "query_frequency": int(frequency)}
            for block_id, frequency in Counter(
                {
                    int(index): int(value)
                    for index, value in enumerate(query_frequency)
                    if value > 0
                }
            ).most_common(20)
        ],
    }
    return rows, hit_vectors, hubs


def main() -> None:
    args = parse_args()
    retrieval_dir = Path(args.retrieval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = read_jsonl(Path(args.queries_jsonl))
    datasets = np.asarray([str(query["dataset"]) for query in queries])
    depths = sorted({int(value) for value in args.depths.split(",")})
    rng = np.random.default_rng(args.seed)

    arrays: dict[str, np.ndarray] = {}
    layers_by_method: dict[str, list[int]] = {}
    fold_ids_by_method: dict[str, np.ndarray] = {}
    for method in METHODS:
        payload = np.load(retrieval_dir / method / "per_head_topk.npz")
        arrays[method] = payload["block_ids"]
        layers_by_method[method] = [int(item) for item in payload["layers"]]
        fold_ids_by_method[method] = payload["fold_ids"]
        if arrays[method].shape[0] != len(queries):
            raise ValueError(f"{method} query count does not match queries.jsonl")
    if not all(layers_by_method[method] == layers_by_method["raw"] for method in METHODS):
        raise ValueError("method layer layouts do not match")
    if not all(
        np.array_equal(fold_ids_by_method[method], fold_ids_by_method["raw"])
        for method in METHODS
    ):
        raise ValueError("method fold assignments do not match")
    if max(depths) > arrays["raw"].shape[-1]:
        raise ValueError("requested depth exceeds stored rankings")

    all_rows: list[dict[str, Any]] = []
    method_hits: dict[str, dict[str, np.ndarray]] = {}
    hubs: dict[str, Any] = {}
    for method in METHODS:
        rows, hit_vectors, method_hubs = evaluate_method(
            arrays[method],
            layers_by_method[method],
            queries,
            datasets,
            depths,
            args.target_blocks,
            args.rrf_constant,
            args.num_blocks,
        )
        for row in rows:
            all_rows.append({"method": method, **row})
        method_hits[method] = hit_vectors
        hubs[method] = method_hubs

    paired_rows: list[dict[str, Any]] = []
    for method in ("centered", "zscore"):
        for metric, raw_hits in method_hits["raw"].items():
            comparison = paired_summary(
                raw_hits,
                method_hits[method][metric],
                args.bootstrap_samples,
                rng,
            )
            paired_rows.append(
                {"candidate_method": method, "metric": metric, **comparison}
            )

    max_depth = max(depths)
    half_depth = max_depth // 2
    cross_budget_rows: list[dict[str, Any]] = []
    if half_depth in depths:
        cross_metrics = [
            (f"union_top{max_depth}", f"union_top{half_depth}", "union"),
            *[
                (
                    f"{mode}_top{args.target_blocks}_depth{max_depth}",
                    f"{mode}_top{args.target_blocks}_depth{half_depth}",
                    mode,
                )
                for mode in CONSENSUS_MODES
            ],
        ]
        for method in ("centered", "zscore"):
            for raw_metric, candidate_metric, metric_family in cross_metrics:
                comparison = paired_summary(
                    method_hits["raw"][raw_metric],
                    method_hits[method][candidate_metric],
                    args.bootstrap_samples,
                    rng,
                )
                cross_budget_rows.append(
                    {
                        "candidate_method": method,
                        "metric_family": metric_family,
                        "raw_depth_per_head": max_depth,
                        "candidate_depth_per_head": half_depth,
                        **comparison,
                    }
                )

    agreement_rows = [
        {
            "candidate_method": method,
            "depth_per_head": depth,
            "mean_per_query_layer_head_set_jaccard": mean_set_jaccard(
                arrays["raw"], arrays[method], depth
            ),
        }
        for method in ("centered", "zscore")
        for depth in depths
    ]
    write_csv(output_dir / "method_summary.csv", all_rows)
    write_csv(output_dir / "paired_comparisons.csv", paired_rows)
    write_csv(output_dir / "cross_budget_comparisons.csv", cross_budget_rows)
    write_csv(output_dir / "ranking_agreement.csv", agreement_rows)

    summary = {
        "experiment": "query_invariant_block_prior_debiasing_analysis",
        "queries": len(queries),
        "datasets": {
            dataset: int(np.sum(datasets == dataset)) for dataset in np.unique(datasets)
        },
        "num_blocks": args.num_blocks,
        "target_blocks": args.target_blocks,
        "depths": depths,
        "gold_used_only_for_evaluation": True,
        "hubs_at_max_depth": hubs,
        "method_summary": all_rows,
        "paired_comparisons": paired_rows,
        "cross_budget_comparisons": cross_budget_rows,
        "ranking_agreement": agreement_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
