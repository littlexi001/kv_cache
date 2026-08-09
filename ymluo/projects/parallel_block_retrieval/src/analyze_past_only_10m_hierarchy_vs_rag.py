#!/usr/bin/env python3
"""Paired reader and cost comparison of 10M hierarchy versus BM25/E5 RAG."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from analyze_past_only_10m_dynamic_controller import EVIDENCE, read_jsonl


STATES = (128, 256, 512)
HIERARCHY = "multilevel_bm25_book8_segment8"
BASELINES = ("bm25", "e5", "bm25_e5_rrf", "query_only", "random512")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hierarchy_reader_rows",
        type=Path,
        nargs="+",
        default=[
            EVIDENCE / f"pg19_past_only_multilevel_10m_q77_ppl_s{state}_rows_20260715.jsonl"
            for state in STATES
        ],
    )
    parser.add_argument(
        "--rag_reader_rows",
        type=Path,
        nargs="+",
        default=[
            EVIDENCE / f"pg19_past_only_10m_q77_rag_ppl_s{state}_rows_20260715.jsonl"
            for state in STATES
        ],
    )
    parser.add_argument(
        "--hierarchy_retrieval_summary",
        type=Path,
        default=EVIDENCE / "pg19_past_only_multilevel_10m_q77_all_states_20260715.json",
    )
    parser.add_argument(
        "--rag_retrieval_summary",
        type=Path,
        default=EVIDENCE / "pg19_past_only_10m_q77_rag_retrieval_20260715.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVIDENCE / "pg19_past_only_10m_q77_hierarchy_vs_rag_20260715.json",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def state_from_path(path: Path) -> int:
    match = re.search(r"ppl_s(\d+)", path.name)
    if not match:
        raise ValueError(f"cannot infer state from {path}")
    return int(match.group(1))


def load_reader(paths: list[Path]) -> dict[tuple[int, int, str], dict[str, Any]]:
    output = {}
    for path in paths:
        state = state_from_path(path)
        for row in read_jsonl(path):
            output[(int(row["query_id"]), state, str(row["method"]))] = row
    return output


def paired(
    candidate: dict[int, float],
    baseline: dict[int, float],
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    query_ids = sorted(set(candidate) & set(baseline))
    delta = np.asarray([candidate[qid] - baseline[qid] for qid in query_ids])
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    wins = int(np.sum(delta < -1e-12))
    losses = int(np.sum(delta > 1e-12))
    ties = len(delta) - wins - losses
    return {
        "query_groups": len(query_ids),
        "delta_definition": "hierarchy NLL minus baseline NLL; negative favors hierarchy",
        "mean_delta_nll": float(delta.mean()),
        "query_bootstrap95": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
        "geometric_mean_ppl_ratio": float(math.exp(delta.mean())),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "exact_two_sided_sign_p": (
            float(binomtest(wins, wins + losses, p=0.5).pvalue)
            if wins + losses
            else 1.0
        ),
    }


def retrieval_lookup(summary: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (int(row["state_suffix_tokens"]), str(row["method"])): row
        for row in summary["retrieval_quality"]
    }


def main() -> None:
    args = parse_args()
    hierarchy = load_reader(args.hierarchy_reader_rows)
    rag = load_reader(args.rag_reader_rows)
    query_ids = sorted({key[0] for key in hierarchy if key[2] == HIERARCHY})
    for state in STATES:
        for query_id in query_ids:
            if (query_id, state, HIERARCHY) not in hierarchy:
                raise ValueError("hierarchy reader matrix is incomplete")
            for method in BASELINES:
                source = hierarchy if method in {"query_only", "random512"} else rag
                if (query_id, state, method) not in source:
                    raise ValueError(f"missing state={state} query={query_id} method={method}")

    rng = np.random.default_rng(args.seed)
    comparisons: dict[str, Any] = {}
    for baseline in BASELINES:
        comparisons[baseline] = {"per_state": {}}
        combined_candidate: dict[int, list[float]] = defaultdict(list)
        combined_baseline: dict[int, list[float]] = defaultdict(list)
        for state in STATES:
            candidate_values = {
                query_id: float(hierarchy[(query_id, state, HIERARCHY)]["mean_nll"])
                for query_id in query_ids
            }
            source = hierarchy if baseline in {"query_only", "random512"} else rag
            baseline_values = {
                query_id: float(source[(query_id, state, baseline)]["mean_nll"])
                for query_id in query_ids
            }
            comparisons[baseline]["per_state"][str(state)] = paired(
                candidate_values,
                baseline_values,
                samples=args.bootstrap_samples,
                rng=rng,
            )
            for query_id in query_ids:
                combined_candidate[query_id].append(candidate_values[query_id])
                combined_baseline[query_id].append(baseline_values[query_id])
        comparisons[baseline]["combined_states_query_clustered"] = paired(
            {query_id: float(np.mean(values)) for query_id, values in combined_candidate.items()},
            {query_id: float(np.mean(values)) for query_id, values in combined_baseline.items()},
            samples=args.bootstrap_samples,
            rng=rng,
        )

    hierarchy_summary = json.loads(args.hierarchy_retrieval_summary.read_text(encoding="utf-8"))
    rag_summary = json.loads(args.rag_retrieval_summary.read_text(encoding="utf-8"))
    hierarchy_retrieval = retrieval_lookup(hierarchy_summary)
    rag_retrieval = retrieval_lookup(rag_summary)
    online_cost = {}
    for state in STATES:
        hierarchy_row = hierarchy_retrieval[(state, HIERARCHY)]
        online_cost[str(state)] = {
            HIERARCHY: {
                "query_seconds": float(hierarchy_row["mean_query_seconds"]),
                "candidate_blocks": float(hierarchy_row["mean_candidate_blocks"]),
            },
            **{
                method: {
                    "query_seconds": float(rag_retrieval[(state, method)]["mean_query_seconds"]),
                    "candidate_blocks": int(hierarchy_summary["memory_blocks"]),
                }
                for method in ("bm25", "e5", "bm25_e5_rrf")
            },
        }

    e5_dimensions = 768
    e5_float32_bytes = int(hierarchy_summary["memory_blocks"] * e5_dimensions * 4)
    payload = {
        "source": "real strict past-only PG19 9.9M hierarchy versus standard RAG",
        "protocol": {
            "queries": len(query_ids),
            "states": list(STATES),
            "memory_tokens": int(hierarchy_summary["memory_tokens"]),
            "memory_blocks": int(hierarchy_summary["memory_blocks"]),
            "reader_tokens_every_method": 512,
            "future_target_tokens": 128,
            "hierarchy_method": HIERARCHY,
            "rag_methods": ["bm25", "e5", "bm25_e5_rrf"],
            "selection_uses_target": False,
        },
        "paired_hierarchy_vs_baselines": comparisons,
        "online_retrieval_cost": online_cost,
        "index_build": {
            "hierarchy_decode_seconds": float(hierarchy_summary["decode_seconds"]),
            "hierarchy_block_segment_book_index_seconds": float(
                hierarchy_summary["block_index_seconds"]
                + hierarchy_summary["segment_index_seconds"]
                + hierarchy_summary["book_index_seconds"]
            ),
            "hierarchy_index_bytes": int(
                hierarchy_summary["block_index_bytes"]
                + hierarchy_summary["segment_index_bytes"]
                + hierarchy_summary["book_index_bytes"]
            ),
            "rag_bm25_index_seconds": float(rag_summary["bm25_index_seconds"]),
            "rag_e5_passage_index_seconds": float(rag_summary["e5_passage_index_seconds"]),
            "rag_e5_query_embedding_amortized_seconds": float(
                rag_summary["e5_query_embedding_amortized_seconds"]
            ),
            "rag_e5_float32_embedding_bytes_derived": e5_float32_bytes,
            "rag_e5_embedding_derivation": "154688 blocks * 768 dimensions * 4 float32 bytes",
        },
        "boundary": {
            "hierarchy_still_uses_lexical_bm25": True,
            "difference_from_flat_rag": (
                "state-conditioned book/segment routing bounds the fine comparison domain before "
                "the same 512-token reader budget; it is not a claim that lexical retrieval is novel"
            ),
            "model_native_extension_not_tested_here": (
                "retrospective model-loss feedback is analyzed separately and does not improve this paired table"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                baseline: value["combined_states_query_clustered"]
                for baseline, value in comparisons.items()
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
