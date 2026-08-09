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
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test leakage-free token routing for multi-vector state retrieval: use the "
            "current lookup entity or verified compact-state pointer instead of every token."
        )
    )
    parser.add_argument("--retrieval_dynamics", required=True)
    parser.add_argument("--step_profile", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--vote_depth", type=int, default=16)
    parser.add_argument("--final_blocks", type=int, default=39)
    parser.add_argument("--rrf_constant", type=int, default=60)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--max_heads",
        type=int,
        default=0,
        help="Use only the first train-frozen heads; zero keeps every retrieved head.",
    )
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def all_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    if not phrase:
        return []
    output = []
    start = 0
    while True:
        index = text.find(phrase, start)
        if index < 0:
            return output
        output.append((index, index + len(phrase)))
        start = index + 1


def overlaps(span: tuple[int, int], targets: Sequence[tuple[int, int]]) -> bool:
    return any(span[1] > target[0] and span[0] < target[1] for target in targets)


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


def empirical_test(observed: float, null: np.ndarray) -> dict[str, float]:
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "lift_over_null": observed / max(float(null.mean()), 1.0e-12),
        "empirical_p": float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1)),
    }


def main() -> None:
    args = parse_args()
    retrieval = torch.load(args.retrieval_dynamics, map_location="cpu", weights_only=False)
    step_profile = torch.load(args.step_profile, map_location="cpu", weights_only=False)
    top_ids = retrieval["top_ids"].numpy()
    selected_head_count = len(retrieval["selected_heads"])
    if args.max_heads > 0:
        if args.max_heads > selected_head_count:
            raise ValueError("max_heads exceeds retrieved head count")
        selected_head_count = args.max_heads
        top_ids = top_ids[:, :selected_head_count]
    metadata = retrieval["state_metadata"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    step_by_key = {
        (int(step["query_id"]), int(step["step_index"])): (index, step)
        for index, step in enumerate(step_profile["steps"])
    }

    token_routes: dict[tuple[int, int], dict[str, set[int]]] = {}
    for key, (step_index, step) in step_by_key.items():
        compact = [str(item) for item in step["compact_state_before"]]
        parts = [
            str(step.get("lookup_key", "")),
            str(step["step_question"]),
            str(step["question"]),
            *compact,
        ]
        state = " ".join(parts)
        prompt = f"\nCurrent reasoning state: {state}\nRetrieve evidence for the next step:"
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        positions = step_profile["token_positions"][step_index]
        lookup_spans = all_spans(prompt, str(step.get("lookup_key", "")))
        pointer_phrases = []
        for item in compact:
            pointer_phrases.extend([item, item.split(":", maxsplit=1)[-1].strip()])
        pointer_spans = [span for phrase in pointer_phrases for span in all_spans(prompt, phrase)]
        lookup_tokens = set()
        pointer_tokens = set()
        for local_token_index, token_position in enumerate(positions):
            span = tuple(int(item) for item in encoded["offset_mapping"][token_position])
            if overlaps(span, lookup_spans):
                lookup_tokens.add(local_token_index)
            if overlaps(span, pointer_spans):
                pointer_tokens.add(local_token_index)
        primary_pointer = pointer_tokens if int(step["step_index"]) > 0 else lookup_tokens
        token_count = len(positions)
        token_routes[key] = {
            "all": set(range(token_count)),
            "last4": set(range(max(0, token_count - 4), token_count)),
            "last8": set(range(max(0, token_count - 8), token_count)),
            "state_pointer": primary_pointer,
            "pointer_plus_last4": primary_pointer
            | set(range(max(0, token_count - 4), token_count)),
            "lookup_or_pointer": lookup_tokens | pointer_tokens,
        }

    by_trajectory_step: dict[tuple[int, int], list[int]] = defaultdict(list)
    for state_id, row in enumerate(metadata):
        by_trajectory_step[
            (int(row["trajectory_index"]), int(row["oracle_step_index"]))
        ].append(state_id)
    trajectory_ids = sorted({key[0] for key in by_trajectory_step})
    methods = list(next(iter(token_routes.values())))
    predictions = {method: {0: [], 1: []} for method in methods}
    cutoffs = [1, 3, 4, 8, 16, args.final_blocks]
    gold_sets = {0: [], 1: []}
    rows = []
    for trajectory_id in trajectory_ids:
        first_state = by_trajectory_step[(trajectory_id, 0)][0]
        first = metadata[first_state]
        query_id = int(first["query_id"])
        gold_sets[0].append(set(int(item) for item in first["hop1_gold_block_ids"]))
        gold_sets[1].append(set(int(item) for item in first["hop2_gold_block_ids"]))
        row = {
            "trajectory_index": trajectory_id,
            "query_id": query_id,
            "split": str(first["split"]),
            "methods": {},
        }
        for method in methods:
            method_row = {}
            for step_index in (0, 1):
                states = by_trajectory_step[(trajectory_id, step_index)]
                allowed = token_routes[(query_id, step_index)][method]
                selected_states = [
                    state
                    for state in states
                    if int(metadata[state]["state_token_index"]) in allowed
                ]
                if not selected_states:
                    selected_states = states
                blocks = rrf(
                    top_ids,
                    selected_states,
                    depth=args.vote_depth,
                    target=args.final_blocks,
                    constant=args.rrf_constant,
                )
                block_set = set(blocks)
                predictions[method][step_index].append(block_set)
                method_row[f"step{step_index}_query_vectors"] = len(selected_states)
                method_row[f"step{step_index}_hit39"] = bool(
                    block_set & gold_sets[step_index][-1]
                )
                method_row[f"step{step_index}_hit_by_blocks"] = {
                    str(cutoff): bool(
                        set(blocks[:cutoff]) & gold_sets[step_index][-1]
                    )
                    for cutoff in cutoffs
                }
            row["methods"][method] = method_row
        rows.append(row)

    metrics = {}
    for method in methods:
        metrics[method] = {
            "step0_query_vectors_mean": mean(
                row["methods"][method]["step0_query_vectors"] for row in rows
            ),
            "step1_query_vectors_mean": mean(
                row["methods"][method]["step1_query_vectors"] for row in rows
            ),
            "hop1_step0_hit39_rate": mean(
                row["methods"][method]["step0_hit39"] for row in rows
            ),
            "hop2_step1_hit39_rate": mean(
                row["methods"][method]["step1_hit39"] for row in rows
            ),
            "both_steps_hit39_rate": mean(
                row["methods"][method]["step0_hit39"]
                and row["methods"][method]["step1_hit39"]
                for row in rows
            ),
            "hop1_hit_rate_by_blocks": {
                str(cutoff): mean(
                    row["methods"][method]["step0_hit_by_blocks"][str(cutoff)]
                    for row in rows
                )
                for cutoff in cutoffs
            },
            "hop2_hit_rate_by_blocks": {
                str(cutoff): mean(
                    row["methods"][method]["step1_hit_by_blocks"][str(cutoff)]
                    for row in rows
                )
                for cutoff in cutoffs
            },
            "both_steps_hit_rate_by_blocks": {
                str(cutoff): mean(
                    row["methods"][method]["step0_hit_by_blocks"][str(cutoff)]
                    and row["methods"][method]["step1_hit_by_blocks"][str(cutoff)]
                    for row in rows
                )
                for cutoff in cutoffs
            },
        }

    rng = np.random.default_rng(args.seed)
    pointer_null = {0: np.empty(args.permutations), 1: np.empty(args.permutations)}
    n = len(rows)
    for permutation_id in range(args.permutations):
        assignment = rng.permutation(n)
        for step_index in (0, 1):
            pointer_null[step_index][permutation_id] = mean(
                bool(
                    predictions["state_pointer"][step_index][index]
                    & gold_sets[step_index][int(assignment[index])]
                )
                for index in range(n)
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "leakage-free state-pointer token routing over natural oracle states",
        "contains_synthetic_vectors": False,
        "contains_source_context": False,
        "uses_gold_for_token_routing": False,
        "uses_oracle_bridge_state": True,
        "trajectories": n,
        "selected_heads": selected_head_count,
        "per_head_vote_depth": args.vote_depth,
        "final_blocks": args.final_blocks,
        "metrics": metrics,
        "permutation_tests": {
            "state_pointer_hop1": empirical_test(
                metrics["state_pointer"]["hop1_step0_hit39_rate"], pointer_null[0]
            ),
            "state_pointer_hop2": empirical_test(
                metrics["state_pointer"]["hop2_step1_hit39_rate"], pointer_null[1]
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
