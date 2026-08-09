from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from profile_real_qk import read_jsonl
from run_all_head_consensus_retrieval import block_consensus, rank_consensus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze all-head coverage and consensus failure modes.")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--retrieval_dir", required=True)
    parser.add_argument("--output_json")
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--rrf_constant", type=float, default=60.0)
    return parser.parse_args()


def mean_vote_stats(rows: list[tuple[int, int, int]]) -> dict[str, float]:
    if not rows:
        return {"blocks": 0, "mean_layers": 0.0, "mean_heads": 0.0, "mean_best_rank": 0.0}
    return {
        "blocks": len(rows),
        "mean_layers": statistics.fmean(row[0] for row in rows),
        "mean_heads": statistics.fmean(row[1] for row in rows),
        "mean_best_rank": statistics.fmean(row[2] for row in rows),
    }


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    retrieval_dir = Path(args.retrieval_dir)
    payload = np.load(retrieval_dir / "per_head_topk.npz")
    head_ids = payload["block_ids"]
    layers = [int(item) for item in payload["layers"]]
    queries = read_jsonl(corpus_dir / "queries.jsonl")[: int(head_ids.shape[0])]
    limits = sorted({1, 2, 4, 8, int(head_ids.shape[3])})

    final_recall: dict[str, dict[str, float]] = {}
    for limit in limits:
        method_hits = {mode: 0 for mode in ["layer_consensus", "head_vote", "rrf"]}
        for query_index, query in enumerate(queries):
            gold = set(int(item) for item in query.get("gold_block_ids", []))
            stats = block_consensus(
                head_ids[query_index],
                rrf_constant=args.rrf_constant,
                rank_limit=limit,
                layers=layers,
            )
            for mode in method_hits:
                selected = set(rank_consensus(stats, mode)[: args.target_blocks])
                method_hits[mode] += int(bool(gold & selected))
        final_recall[f"top{limit}_per_head"] = {
            mode: hits / len(queries) for mode, hits in method_hits.items()
        }

    selected_gold: list[tuple[int, int, int]] = []
    dropped_gold: list[tuple[int, int, int]] = []
    cutoffs: list[tuple[int, int, int]] = []
    selected_frequency: Counter[int] = Counter()
    for query_index, query in enumerate(queries):
        stats = block_consensus(
            head_ids[query_index],
            rrf_constant=args.rrf_constant,
            rank_limit=int(head_ids.shape[3]),
            layers=layers,
        )
        ranked = rank_consensus(stats, "layer_consensus")
        selected = set(ranked[: args.target_blocks])
        selected_frequency.update(selected)
        cutoff = stats[ranked[args.target_blocks - 1]]
        cutoffs.append(
            (len(cutoff["layers"]), int(cutoff["head_votes"]), int(cutoff["best_rank"]))
        )
        for block_id in (int(item) for item in query.get("gold_block_ids", [])):
            if block_id not in stats:
                continue
            values = stats[block_id]
            row = (
                len(values["layers"]),
                int(values["head_votes"]),
                int(values["best_rank"]),
            )
            (selected_gold if block_id in selected else dropped_gold).append(row)

    head_hits: dict[tuple[int, int], set[int]] = {}
    for layer_index, layer in enumerate(layers):
        for query_head in range(int(head_ids.shape[2])):
            hits: set[int] = set()
            for query_index, query in enumerate(queries):
                gold = set(int(item) for item in query.get("gold_block_ids", []))
                nominated = set(
                    int(item) for item in head_ids[query_index, layer_index, query_head]
                )
                if gold & nominated:
                    hits.add(query_index)
            head_hits[(layer, query_head)] = hits

    covered: set[int] = set()
    remaining = set(head_hits)
    greedy_heads: list[dict[str, Any]] = []
    while remaining:
        best = max(
            remaining,
            key=lambda pair: (
                len(head_hits[pair] - covered),
                len(head_hits[pair]),
                -pair[0],
                -pair[1],
            ),
        )
        gain = len(head_hits[best] - covered)
        if gain == 0:
            break
        covered.update(head_hits[best])
        remaining.remove(best)
        greedy_heads.append(
            {
                "layer": best[0],
                "query_head": best[1],
                "single_head_hits": len(head_hits[best]),
                "new_hits": gain,
                "cumulative_hits": len(covered),
            }
        )

    dataset_best_heads: list[dict[str, Any]] = []
    for dataset in sorted({str(query["dataset"]) for query in queries}):
        query_ids = [
            index for index, query in enumerate(queries) if str(query["dataset"]) == dataset
        ]
        best = max(
            head_hits,
            key=lambda pair: (
                len(head_hits[pair] & set(query_ids)),
                -pair[0],
                -pair[1],
            ),
        )
        hits = len(head_hits[best] & set(query_ids))
        dataset_best_heads.append(
            {
                "dataset": dataset,
                "queries": len(query_ids),
                "layer": best[0],
                "query_head": best[1],
                "hits": hits,
                "recall": hits / len(query_ids),
            }
        )

    output = {
        "queries": len(queries),
        "layers": len(layers),
        "query_heads_per_layer": int(head_ids.shape[2]),
        "top_per_head": int(head_ids.shape[3]),
        "target_blocks": args.target_blocks,
        "final_recall_by_nomination_depth": final_recall,
        "gold_vote_stats": {
            "selected": mean_vote_stats(selected_gold),
            "nominated_but_dropped": mean_vote_stats(dropped_gold),
            "top39_cutoff": mean_vote_stats(cutoffs),
        },
        "most_repeated_consensus_blocks": [
            {"block_id": block_id, "queries": count}
            for block_id, count in selected_frequency.most_common(20)
        ],
        "greedy_head_coverage_diagnostic": greedy_heads,
        "dataset_best_head_diagnostic": dataset_best_heads,
    }
    output_path = (
        Path(args.output_json)
        if args.output_json
        else retrieval_dir / "head_specialization_analysis.json"
    )
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
