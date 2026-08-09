from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether transient evidence hits in a generated-Q retrieval trajectory "
            "are aligned with the query, persistent over time, and recoverable by bounded "
            "temporal voting."
        )
    )
    parser.add_argument("--retrieval_dynamics", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vote_depth", type=int, default=16)
    parser.add_argument("--final_blocks", type=int, default=39)
    parser.add_argument("--rrf_constant", type=int, default=60)
    parser.add_argument("--windows", default="2,4,8")
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def percentile(values: Iterable[float], q: float) -> float:
    values = list(values)
    return float(np.percentile(values, q)) if values else math.nan


def longest_run(values: Sequence[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def rrf_over_states(
    top_ids: np.ndarray,
    state_ids: Sequence[int],
    *,
    vote_depth: int,
    final_blocks: int,
    constant: int,
) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for state_id in state_ids:
        for head_ids in top_ids[int(state_id)]:
            for rank, block_id in enumerate(head_ids[:vote_depth], start=1):
                scores[int(block_id)] += 1.0 / (constant + rank)
    return [
        block_id
        for block_id, _score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:final_blocks]
    ]


def jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(int(item) for item in left)
    right_set = set(int(item) for item in right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def empirical_test(
    observed: float,
    null_values: np.ndarray,
) -> dict[str, float]:
    return {
        "observed": float(observed),
        "null_mean": float(null_values.mean()),
        "null_p95": float(np.percentile(null_values, 95)),
        "lift_over_null": float(observed / max(float(null_values.mean()), 1.0e-12)),
        "empirical_p": float((1 + np.count_nonzero(null_values >= observed)) / (1 + len(null_values))),
    }


def main() -> None:
    args = parse_args()
    windows = sorted({int(item) for item in args.windows.split(",") if item.strip()})
    if any(window <= 0 for window in windows):
        raise ValueError("windows must contain positive integers")
    payload = torch.load(args.retrieval_dynamics, map_location="cpu", weights_only=False)
    top_ids = payload["top_ids"].numpy()
    metadata = payload["state_metadata"]
    state_rrf = [list(map(int, row)) for row in payload["final_rrf_block_ids"]]

    by_trajectory: dict[int, list[int]] = defaultdict(list)
    for state_id, row in enumerate(metadata):
        by_trajectory[int(row["trajectory_index"])].append(state_id)
    trajectory_ids = sorted(by_trajectory)
    gold_sets: list[set[int]] = []
    rows: list[dict[str, Any]] = []
    predictions: dict[str, list[Any]] = defaultdict(list)

    for trajectory_id in trajectory_ids:
        state_ids = by_trajectory[trajectory_id]
        first = metadata[state_ids[0]]
        gold = set(int(item) for item in first["hop2_gold_block_ids"])
        gold_sets.append(gold)
        per_state_sets = [state_rrf[state_id] for state_id in state_ids]
        per_state_hits = [bool(gold & set(blocks)) for blocks in per_state_sets]
        progress = [float(metadata[state_id]["bridge_progress"]) for state_id in state_ids]

        all_states = rrf_over_states(
            top_ids,
            state_ids,
            vote_depth=args.vote_depth,
            final_blocks=args.final_blocks,
            constant=args.rrf_constant,
        )
        row: dict[str, Any] = {
            "trajectory_index": trajectory_id,
            "query_id": int(first["query_id"]),
            "states": len(state_ids),
            "bridge_generation_hit": bool(first["bridge_generation_hit"]),
            "gold_block_ids": sorted(gold),
            "state_hit_count": int(sum(per_state_hits)),
            "state_hit_fraction": float(mean(per_state_hits)),
            "state_ever_hit": bool(any(per_state_hits)),
            "state_final_hit": bool(per_state_hits[-1]),
            "state_longest_hit_run": longest_run(per_state_hits),
            "state_first_hit": next(
                (index for index, hit in enumerate(per_state_hits) if hit), None
            ),
            "temporal_all_hit": bool(gold & set(all_states)),
            "progress_changed": bool(any(right > left for left, right in zip(progress, progress[1:]))),
        }
        predictions["state_initial"].append(set(per_state_sets[0]))
        predictions["state_final"].append(set(per_state_sets[-1]))
        predictions["state_ever"].append([set(item) for item in per_state_sets])
        predictions["temporal_all"].append(set(all_states))

        for window in windows:
            temporal_sets = []
            for end in range(len(state_ids)):
                start = max(0, end + 1 - window)
                temporal_sets.append(
                    rrf_over_states(
                        top_ids,
                        state_ids[start : end + 1],
                        vote_depth=args.vote_depth,
                        final_blocks=args.final_blocks,
                        constant=args.rrf_constant,
                    )
                )
            temporal_hits = [bool(gold & set(item)) for item in temporal_sets]
            row[f"temporal_w{window}_ever_hit"] = bool(any(temporal_hits))
            row[f"temporal_w{window}_final_hit"] = bool(temporal_hits[-1])
            predictions[f"temporal_w{window}_ever"].append(
                [set(item) for item in temporal_sets]
            )
            predictions[f"temporal_w{window}_final"].append(set(temporal_sets[-1]))
        rows.append(row)

    metrics: dict[str, float] = {
        "state_initial_hit_rate": mean(bool(gold_sets[i] & predictions["state_initial"][i]) for i in range(len(rows))),
        "state_final_hit_rate": mean(bool(gold_sets[i] & predictions["state_final"][i]) for i in range(len(rows))),
        "state_ever_hit_rate": mean(row["state_ever_hit"] for row in rows),
        "state_persistent_ge2_rate": mean(row["state_longest_hit_run"] >= 2 for row in rows),
        "state_persistent_ge3_rate": mean(row["state_longest_hit_run"] >= 3 for row in rows),
        "state_hit_fraction_mean": mean(row["state_hit_fraction"] for row in rows),
        "temporal_all_hit_rate": mean(row["temporal_all_hit"] for row in rows),
    }
    for window in windows:
        metrics[f"temporal_w{window}_ever_hit_rate"] = mean(
            row[f"temporal_w{window}_ever_hit"] for row in rows
        )
        metrics[f"temporal_w{window}_final_hit_rate"] = mean(
            row[f"temporal_w{window}_final_hit"] for row in rows
        )

    transition_rows = []
    for trajectory_id in trajectory_ids:
        state_ids = by_trajectory[trajectory_id]
        for left, right in zip(state_ids, state_ids[1:]):
            progress_delta = float(metadata[right]["bridge_progress"]) - float(
                metadata[left]["bridge_progress"]
            )
            transition_rows.append(
                {
                    "trajectory_index": trajectory_id,
                    "from_state": int(metadata[left]["state_index"]),
                    "to_state": int(metadata[right]["state_index"]),
                    "progress_delta": progress_delta,
                    "progress_changed": progress_delta > 0,
                    "rrf39_jaccard": jaccard(state_rrf[left], state_rrf[right]),
                }
            )
    changed = [row["rrf39_jaccard"] for row in transition_rows if row["progress_changed"]]
    unchanged = [row["rrf39_jaccard"] for row in transition_rows if not row["progress_changed"]]

    rng = np.random.default_rng(args.seed)
    null_by_metric = {
        key: np.empty(args.permutations, dtype=np.float64)
        for key in predictions
    }
    persistent2_null = np.empty(args.permutations, dtype=np.float64)
    persistent3_null = np.empty(args.permutations, dtype=np.float64)
    n = len(rows)
    for permutation_id in range(args.permutations):
        assignment = rng.permutation(n)
        persistent2 = []
        persistent3 = []
        for metric, predicted in predictions.items():
            hits = []
            for index in range(n):
                shuffled_gold = gold_sets[int(assignment[index])]
                value = predicted[index]
                if metric.endswith("_ever") or metric == "state_ever":
                    hits.append(any(shuffled_gold & item for item in value))
                else:
                    hits.append(bool(shuffled_gold & value))
            null_by_metric[metric][permutation_id] = mean(hits)
        for index in range(n):
            shuffled_gold = gold_sets[int(assignment[index])]
            hit_sequence = [
                bool(shuffled_gold & item) for item in predictions["state_ever"][index]
            ]
            run = longest_run(hit_sequence)
            persistent2.append(run >= 2)
            persistent3.append(run >= 3)
        persistent2_null[permutation_id] = mean(persistent2)
        persistent3_null[permutation_id] = mean(persistent3)

    observed_by_prediction = {
        "state_initial": metrics["state_initial_hit_rate"],
        "state_final": metrics["state_final_hit_rate"],
        "state_ever": metrics["state_ever_hit_rate"],
        "temporal_all": metrics["temporal_all_hit_rate"],
    }
    for window in windows:
        observed_by_prediction[f"temporal_w{window}_ever"] = metrics[
            f"temporal_w{window}_ever_hit_rate"
        ]
        observed_by_prediction[f"temporal_w{window}_final"] = metrics[
            f"temporal_w{window}_final_hit_rate"
        ]
    permutation_tests = {
        metric: empirical_test(observed_by_prediction[metric], null_by_metric[metric])
        for metric in observed_by_prediction
    }
    permutation_tests["state_persistent_ge2"] = empirical_test(
        metrics["state_persistent_ge2_rate"], persistent2_null
    )
    permutation_tests["state_persistent_ge3"] = empirical_test(
        metrics["state_persistent_ge3_rate"], persistent3_null
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trajectory_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "transition_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in transition_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "temporal property analysis of exact QK retrieval over real 10M context",
        "contains_synthetic_vectors": False,
        "trajectories": n,
        "states": len(metadata),
        "heads": int(top_ids.shape[1]),
        "per_head_vote_depth": args.vote_depth,
        "working_set_blocks": args.final_blocks,
        "windows": windows,
        "metrics": metrics,
        "semantic_event_alignment": {
            "progress_changed_transitions": len(changed),
            "progress_unchanged_transitions": len(unchanged),
            "changed_rrf39_jaccard_mean": mean(changed),
            "unchanged_rrf39_jaccard_mean": mean(unchanged),
            "difference_changed_minus_unchanged": mean(changed) - mean(unchanged),
        },
        "permutation_protocol": {
            "description": "shuffle whole hop-2 gold block sets across trajectories",
            "permutations": args.permutations,
            "seed": args.seed,
        },
        "permutation_tests": permutation_tests,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
