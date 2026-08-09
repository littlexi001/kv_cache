from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate token-by-head rankings into a bounded multi-vector reasoning-state "
            "retrieval result and measure whether an oracle bridge redirects retrieval."
        )
    )
    parser.add_argument("--retrieval_dynamics", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vote_depth", type=int, default=16)
    parser.add_argument("--final_blocks", type=int, default=39)
    parser.add_argument("--rrf_constant", type=int, default=60)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def rrf(
    top_ids: np.ndarray,
    state_ids: Sequence[int],
    *,
    depth: int,
    target: int,
    constant: int,
) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for state_id in state_ids:
        for head_ids in top_ids[int(state_id)]:
            for rank, block_id in enumerate(head_ids[:depth], start=1):
                scores[int(block_id)] += 1.0 / (constant + rank)
    return [
        block_id
        for block_id, _score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:target]
    ]


def jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def empirical_test(observed: float, null: np.ndarray) -> dict[str, float]:
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "lift_over_null": float(observed / max(float(null.mean()), 1.0e-12)),
        "empirical_p": float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1)),
    }


def main() -> None:
    args = parse_args()
    payload = torch.load(args.retrieval_dynamics, map_location="cpu", weights_only=False)
    top_ids = payload["top_ids"].numpy()
    metadata = payload["state_metadata"]
    by_trajectory_step: dict[tuple[int, int], list[int]] = defaultdict(list)
    for state_id, row in enumerate(metadata):
        if "oracle_step_index" not in row:
            raise KeyError("state metadata has no oracle_step_index; use token_ensemble profile")
        by_trajectory_step[
            (int(row["trajectory_index"]), int(row["oracle_step_index"]))
        ].append(state_id)

    trajectory_ids = sorted({item[0] for item in by_trajectory_step})
    rows = []
    gold1_sets = []
    gold2_sets = []
    step0_sets = []
    step1_sets = []
    for trajectory_id in trajectory_ids:
        state0 = by_trajectory_step[(trajectory_id, 0)]
        state1 = by_trajectory_step[(trajectory_id, 1)]
        first = metadata[state0[0]]
        gold1 = set(int(item) for item in first["hop1_gold_block_ids"])
        gold2 = set(int(item) for item in first["hop2_gold_block_ids"])
        selected0 = rrf(
            top_ids,
            state0,
            depth=args.vote_depth,
            target=args.final_blocks,
            constant=args.rrf_constant,
        )
        selected1 = rrf(
            top_ids,
            state1,
            depth=args.vote_depth,
            target=args.final_blocks,
            constant=args.rrf_constant,
        )
        any0 = set(int(item) for state in state0 for head in top_ids[state] for item in head[:16])
        any1 = set(int(item) for state in state1 for head in top_ids[state] for item in head[:16])
        row = {
            "trajectory_index": trajectory_id,
            "query_id": int(first["query_id"]),
            "split": str(first["split"]),
            "step0_query_vectors": len(state0),
            "step1_query_vectors": len(state1),
            "step_rrf39_jaccard": jaccard(selected0, selected1),
            "hop1_step0_rrf_hit39": bool(gold1 & set(selected0)),
            "hop1_step1_rrf_hit39": bool(gold1 & set(selected1)),
            "hop2_step0_rrf_hit39": bool(gold2 & set(selected0)),
            "hop2_step1_rrf_hit39": bool(gold2 & set(selected1)),
            "hop1_step0_any_token_head_hit16": bool(gold1 & any0),
            "hop2_step1_any_token_head_hit16": bool(gold2 & any1),
            "hop2_newly_retrieved_after_bridge": bool(
                not (gold2 & set(selected0)) and (gold2 & set(selected1))
            ),
        }
        rows.append(row)
        gold1_sets.append(gold1)
        gold2_sets.append(gold2)
        step0_sets.append(set(selected0))
        step1_sets.append(set(selected1))

    summary_metrics = {
        "hop1_step0_rrf_hit39_rate": mean(row["hop1_step0_rrf_hit39"] for row in rows),
        "hop1_step1_rrf_hit39_rate": mean(row["hop1_step1_rrf_hit39"] for row in rows),
        "hop2_step0_rrf_hit39_rate": mean(row["hop2_step0_rrf_hit39"] for row in rows),
        "hop2_step1_rrf_hit39_rate": mean(row["hop2_step1_rrf_hit39"] for row in rows),
        "hop1_step0_any_token_head_hit16_rate": mean(
            row["hop1_step0_any_token_head_hit16"] for row in rows
        ),
        "hop2_step1_any_token_head_hit16_rate": mean(
            row["hop2_step1_any_token_head_hit16"] for row in rows
        ),
        "hop2_newly_retrieved_after_bridge_rate": mean(
            row["hop2_newly_retrieved_after_bridge"] for row in rows
        ),
        "both_correct_step_rrf_hit39_rate": mean(
            row["hop1_step0_rrf_hit39"] and row["hop2_step1_rrf_hit39"] for row in rows
        ),
        "step_rrf39_jaccard_mean": mean(row["step_rrf39_jaccard"] for row in rows),
    }

    rng = np.random.default_rng(args.seed)
    null_hop1 = np.empty(args.permutations, dtype=np.float64)
    null_hop2 = np.empty(args.permutations, dtype=np.float64)
    n = len(rows)
    for permutation_id in range(args.permutations):
        assignment = rng.permutation(n)
        null_hop1[permutation_id] = mean(
            bool(step0_sets[index] & gold1_sets[int(assignment[index])]) for index in range(n)
        )
        null_hop2[permutation_id] = mean(
            bool(step1_sets[index] & gold2_sets[int(assignment[index])]) for index in range(n)
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "bounded multi-vector QK retrieval for natural oracle reasoning states",
        "contains_synthetic_vectors": False,
        "contains_source_context": False,
        "uses_oracle_bridge_state": True,
        "trajectories": n,
        "query_vectors_per_step_mean": mean(
            0.5 * (row["step0_query_vectors"] + row["step1_query_vectors"]) for row in rows
        ),
        "selected_heads": len(payload["selected_heads"]),
        "per_head_vote_depth": args.vote_depth,
        "final_blocks": args.final_blocks,
        "metrics": summary_metrics,
        "permutation_protocol": {
            "description": "shuffle whole step-specific gold block sets across trajectories",
            "permutations": args.permutations,
            "seed": args.seed,
        },
        "permutation_tests": {
            "hop1_step0_rrf39": empirical_test(
                summary_metrics["hop1_step0_rrf_hit39_rate"], null_hop1
            ),
            "hop2_step1_rrf39": empirical_test(
                summary_metrics["hop2_step1_rrf_hit39_rate"], null_hop2
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
