from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import binomtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate independent LongMemEval 10M state-trajectory shards."
    )
    parser.add_argument("--data_pattern", required=True)
    parser.add_argument("--plan_pattern", required=True)
    parser.add_argument("--evaluation_pattern", required=True)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def path_for(pattern: str, partition: int) -> Path:
    return Path(pattern.format(partition=partition))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        raise ValueError("quantiles require at least one value")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int
) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_binary(
    baseline: list[dict[str, Any]], treatment: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    base = {str(row["question_id"]): bool(row[metric]) for row in baseline}
    new = {str(row["question_id"]): bool(row[metric]) for row in treatment}
    ids = sorted(set(base) & set(new))
    wins = sum(not base[item] and new[item] for item in ids)
    losses = sum(base[item] and not new[item] for item in ids)
    return {
        "queries": len(ids),
        "baseline_rate": mean(float(base[item]) for item in ids),
        "treatment_rate": mean(float(new[item]) for item in ids),
        "delta": mean(float(new[item]) - float(base[item]) for item in ids),
        "wins": wins,
        "losses": losses,
        "ties": len(ids) - wins - losses,
        "two_sided_binomial_p": (
            float(binomtest(wins, wins + losses, 0.5).pvalue)
            if wins + losses
            else 1.0
        ),
    }


def paired_continuous(
    baseline: list[dict[str, Any]],
    treatment: list[dict[str, Any]],
    metric: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    base = {str(row["question_id"]): float(row[metric]) for row in baseline}
    new = {str(row["question_id"]): float(row[metric]) for row in treatment}
    ids = sorted(set(base) & set(new))
    differences = [new[item] - base[item] for item in ids]
    return {
        "queries": len(ids),
        "baseline_mean": mean(base[item] for item in ids),
        "treatment_mean": mean(new[item] for item in ids),
        "mean_delta": mean(differences),
        "bootstrap_95_ci": bootstrap_mean_ci(
            differences, samples=bootstrap_samples, seed=seed
        ),
        "wins": sum(value > 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
    }


def summarize_quality(
    rows: list[dict[str, Any]], *, topk: int
) -> list[dict[str, Any]]:
    output = []
    for method, state in sorted(
        {(str(row["method"]), str(row["state"])) for row in rows}
    ):
        group = [
            row
            for row in rows
            if row["method"] == method and row["state"] == state
        ]
        positive = [row for row in group if not row["is_abstention"]]
        abstention = [row for row in group if row["is_abstention"]]
        output.append(
            {
                "method": method,
                "state": state,
                "state_order": int(group[0]["state_order"]),
                "queries": len(group),
                "positive_queries": len(positive),
                "abstention_queries": len(abstention),
                "mean_candidate_tokens": mean(
                    float(row["candidate_tokens"]) for row in group
                ),
                "mean_query_milliseconds": 1000
                * float(mean(float(row["query_seconds"]) for row in group)),
                f"exact_block_any_at_{topk}": mean(
                    float(row[f"exact_block_any_at_{topk}"]) for row in positive
                ),
                f"latest_exact_block_any_at_{topk}": mean(
                    float(row[f"latest_exact_block_any_at_{topk}"])
                    for row in positive
                ),
                f"evidence_session_recall_at_{topk}": mean(
                    float(row[f"evidence_session_recall_at_{topk}"])
                    for row in positive
                ),
                f"all_evidence_sessions_at_{topk}": mean(
                    float(row[f"all_evidence_sessions_at_{topk}"])
                    for row in positive
                ),
                f"hard_negative_block_any_at_{topk}": mean(
                    float(row[f"hard_negative_block_any_at_{topk}"])
                    for row in abstention
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.partitions <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("partitions and bootstrap_samples must be positive")
    all_queries: list[dict[str, Any]] = []
    all_plans: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_frontier: list[dict[str, Any]] = []
    shard_summaries = []
    block_sessions_by_partition: dict[int, np.ndarray] = {}
    topk = None
    memory_tokens = set()

    for partition in range(args.partitions):
        data_dir = path_for(args.data_pattern, partition)
        plan_dir = path_for(args.plan_pattern, partition)
        evaluation_dir = path_for(args.evaluation_pattern, partition)
        data_summary = read_json(data_dir / "summary.json")
        plan_summary = read_json(plan_dir / "summary.json")
        evaluation_summary = read_json(evaluation_dir / "summary.json")
        queries = read_jsonl(data_dir / "queries.jsonl")
        owners = read_jsonl(data_dir / "owner_manifest.jsonl")
        block_sessions_by_partition[partition] = np.asarray(
            np.load(data_dir / "base_block_session_rows.npy", mmap_mode="r"),
            dtype=np.int64,
        )
        plans = read_jsonl(plan_dir / "trajectories.jsonl")
        rows = read_jsonl(evaluation_dir / "rows.jsonl")
        frontier = read_jsonl(evaluation_dir / "frontier_rows.jsonl")
        query_by_local = {int(row["query_id"]): row for row in queries}
        owner_to_index = {
            int(row["owner_row"]): index for index, row in enumerate(owners)
        }
        local_to_question = {
            local: str(row["question_id"]) for local, row in query_by_local.items()
        }
        for query in queries:
            item = dict(query)
            item["partition"] = partition
            all_queries.append(item)
        for plan in plans:
            item = dict(plan)
            item["partition"] = partition
            all_plans.append(item)
        for row in rows:
            item = dict(row)
            local = int(row["query_id"])
            query = query_by_local[local]
            item["partition"] = partition
            item["question_id"] = local_to_question[local]
            if str(item["method"]).startswith("owner_router"):
                true_owner = owner_to_index[int(query["owner_row"])]
                item["true_owner_in_routed_set"] = true_owner in set(
                    map(int, item["selected_owner_indices"])
                )
            else:
                item["true_owner_in_routed_set"] = None
            all_rows.append(item)
        for row in frontier:
            item = dict(row)
            item["partition"] = partition
            item["question_id"] = local_to_question[int(row["query_id"])]
            all_frontier.append(item)
        shard_summaries.append(
            {
                "partition": partition,
                "queries": int(data_summary["query_samples"]),
                "positive_queries": int(data_summary["non_abstention_queries"]),
                "abstention_queries": int(data_summary["abstention_queries"]),
                "selected_history_tokens": int(data_summary["selected_history_tokens"]),
                "answer_overlap_queries": int(plan_summary["answer_overlap_queries"]),
                "mean_generation_seconds": float(plan_summary["mean_generation_seconds"]),
                "decode_seconds": float(evaluation_summary["decode_seconds"]),
                "block_index_seconds": float(evaluation_summary["block_index_seconds"]),
                "owner_index_seconds": float(evaluation_summary["owner_index_seconds"]),
                "session_index_seconds": float(evaluation_summary["session_index_seconds"]),
            }
        )
        memory_tokens.add(int(data_summary["memory_tokens"]))
        current_topk = int(evaluation_summary["protocol"]["final_topk"])
        topk = current_topk if topk is None else topk
        if topk != current_topk:
            raise RuntimeError("topk differs across partitions")

    question_ids = [str(row["question_id"]) for row in all_queries]
    if len(question_ids) != len(set(question_ids)):
        raise RuntimeError("question partitions overlap")
    if len(question_ids) != 500:
        raise RuntimeError(f"expected all 500 questions, got {len(question_ids)}")
    if memory_tokens != {10_000_000}:
        raise RuntimeError(f"each shard must contain exactly 10M tokens: {memory_tokens}")
    if any(bool(row["answer_overlap_posthoc"]) for row in all_plans):
        raise RuntimeError("novel answer overlap remains in generated plans")
    assert topk is not None

    positive_rows = [row for row in all_rows if not row["is_abstention"]]
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in positive_rows:
        by_key[(str(row["method"]), str(row["state"]))].append(row)
    methods = sorted({str(row["method"]) for row in all_rows})
    states = sorted(
        {str(row["state"]) for row in all_rows},
        key=lambda state: next(
            int(row["state_order"]) for row in all_rows if row["state"] == state
        ),
    )
    full_state = "question_100pct"
    state_comparisons = []
    for method in methods:
        baseline = by_key[(method, full_state)]
        for state in states:
            if state == full_state:
                continue
            treatment = by_key[(method, state)]
            item = {
                "method": method,
                "baseline_state": full_state,
                "treatment_state": state,
            }
            for metric in (
                f"exact_block_any_at_{topk}",
                f"latest_exact_block_any_at_{topk}",
                f"all_evidence_sessions_at_{topk}",
            ):
                item[metric] = paired_binary(baseline, treatment, metric)
            item[f"evidence_session_recall_at_{topk}"] = paired_continuous(
                baseline,
                treatment,
                f"evidence_session_recall_at_{topk}",
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            base_map = {str(row["question_id"]): row for row in baseline}
            new_map = {str(row["question_id"]): row for row in treatment}
            ids = sorted(set(base_map) & set(new_map))
            item["mean_top_block_jaccard"] = mean(
                len(
                    set(base_map[qid]["top_block_ids"][:topk])
                    & set(new_map[qid]["top_block_ids"][:topk])
                )
                / len(
                    set(base_map[qid]["top_block_ids"][:topk])
                    | set(new_map[qid]["top_block_ids"][:topk])
                )
                for qid in ids
            )
            state_comparisons.append(item)

    consecutive_transitions = []
    for method in methods:
        for previous_state, current_state in zip(states, states[1:]):
            previous = {
                str(row["question_id"]): row
                for row in by_key[(method, previous_state)]
            }
            current = {
                str(row["question_id"]): row
                for row in by_key[(method, current_state)]
            }
            ids = sorted(set(previous) & set(current))
            new_counts = []
            retained_counts = []
            exact_rescues = exact_losses = 0
            all_rescues = all_losses = 0
            for qid in ids:
                old_top = set(map(int, previous[qid]["top_block_ids"][:topk]))
                new_top = set(map(int, current[qid]["top_block_ids"][:topk]))
                new_counts.append(float(len(new_top - old_top)))
                retained_counts.append(float(len(new_top & old_top)))
                old_exact = bool(previous[qid][f"exact_block_any_at_{topk}"])
                new_exact = bool(current[qid][f"exact_block_any_at_{topk}"])
                exact_rescues += int(not old_exact and new_exact)
                exact_losses += int(old_exact and not new_exact)
                old_all = bool(previous[qid][f"all_evidence_sessions_at_{topk}"])
                new_all = bool(current[qid][f"all_evidence_sessions_at_{topk}"])
                all_rescues += int(not old_all and new_all)
                all_losses += int(old_all and not new_all)
            consecutive_transitions.append(
                {
                    "method": method,
                    "previous_state": previous_state,
                    "current_state": current_state,
                    "queries": len(ids),
                    "mean_retained_blocks": mean(retained_counts),
                    "mean_new_blocks": mean(new_counts),
                    "mean_new_tokens_if_kv_pages_cached": 64 * float(mean(new_counts)),
                    "unchanged_topk_fraction": mean(
                        float(value == 0) for value in new_counts
                    ),
                    "exact_block_rescues": exact_rescues,
                    "exact_block_losses": exact_losses,
                    "all_evidence_rescues": all_rescues,
                    "all_evidence_losses": all_losses,
                }
            )

    router_maturation = []
    router_methods = [method for method in methods if method.startswith("owner_router")]
    for method in router_methods:
        for state in states:
            group = by_key[(method, state)]
            router_maturation.append(
                {
                    "method": method,
                    "state": state,
                    "queries": len(group),
                    "true_owner_in_routed_set": mean(
                        float(row["true_owner_in_routed_set"]) for row in group
                    ),
                }
            )

    query_by_question = {
        str(row["question_id"]): row for row in all_queries if not row["is_abstention"]
    }
    rows_by_method_state_question = {
        (str(row["method"]), str(row["state"]), str(row["question_id"])): row
        for row in positive_rows
    }
    trajectory_states = [full_state] + [
        state for state in states if state.startswith("question_full_plan_")
    ]
    bounded_frontier_rows = []
    for method in methods:
        for question_id, query in query_by_question.items():
            ordered = []
            seen = set()
            for state in trajectory_states:
                row = rows_by_method_state_question[(method, state, question_id)]
                for block_id in map(int, row["top_block_ids"][:topk]):
                    if block_id in seen:
                        continue
                    seen.add(block_id)
                    ordered.append(block_id)
            static_ranking = list(
                map(
                    int,
                    rows_by_method_state_question[
                        (method, full_state, question_id)
                    ]["top_block_ids"],
                )
            )
            positives = set(map(int, query["positive_block_ids"]))
            positive_sessions = set(map(int, query["positive_session_rows"]))
            partition = int(query["partition"])
            block_sessions = block_sessions_by_partition[partition]
            for budget in (8, 12, 16, 24, 40):
                dynamic = set(ordered[:budget])
                static = set(static_ranking[:budget])
                dynamic_sessions = {
                    int(block_sessions[block_id])
                    for block_id in dynamic
                    if int(block_sessions[block_id]) >= 0
                }
                static_sessions = {
                    int(block_sessions[block_id])
                    for block_id in static
                    if int(block_sessions[block_id]) >= 0
                }
                bounded_frontier_rows.append(
                    {
                        "method": method,
                        "question_id": question_id,
                        "question_type": str(query["question_type"]),
                        "budget_blocks": budget,
                        "budget_tokens": budget * 64,
                        "dynamic_blocks_used": min(len(ordered), budget),
                        "static_exact_block_any": bool(positives & static),
                        "dynamic_exact_block_any": bool(positives & dynamic),
                        "static_all_evidence_sessions": positive_sessions.issubset(
                            static_sessions
                        ),
                        "dynamic_all_evidence_sessions": positive_sessions.issubset(
                            dynamic_sessions
                        ),
                    }
                )

    bounded_frontier_summary = []
    for method in methods:
        for budget in (8, 12, 16, 24, 40):
            group = [
                row
                for row in bounded_frontier_rows
                if row["method"] == method and row["budget_blocks"] == budget
            ]
            static_exact = [
                {
                    "question_id": row["question_id"],
                    "value": row["static_exact_block_any"],
                }
                for row in group
            ]
            dynamic_exact = [
                {
                    "question_id": row["question_id"],
                    "value": row["dynamic_exact_block_any"],
                }
                for row in group
            ]
            static_all = [
                {
                    "question_id": row["question_id"],
                    "value": row["static_all_evidence_sessions"],
                }
                for row in group
            ]
            dynamic_all = [
                {
                    "question_id": row["question_id"],
                    "value": row["dynamic_all_evidence_sessions"],
                }
                for row in group
            ]
            bounded_frontier_summary.append(
                {
                    "method": method,
                    "budget_blocks": budget,
                    "budget_tokens": budget * 64,
                    "queries": len(group),
                    "mean_dynamic_blocks_used": mean(
                        float(row["dynamic_blocks_used"]) for row in group
                    ),
                    "dynamic_vs_static_exact_block_any": paired_binary(
                        static_exact, dynamic_exact, "value"
                    ),
                    "dynamic_vs_static_all_evidence_sessions": paired_binary(
                        static_all, dynamic_all, "value"
                    ),
                }
            )

    frontier_summary = []
    frontier_by_type = []
    question_type = {str(row["question_id"]): str(row["question_type"]) for row in all_queries}
    for method in methods:
        group = [row for row in all_frontier if row["method"] == method]
        static_exact = [
            {
                "question_id": row["question_id"],
                "value": row["static_exact_block_any"],
            }
            for row in group
        ]
        union_exact = [
            {"question_id": row["question_id"], "value": row["exact_block_any"]}
            for row in group
        ]
        static_all = [
            {
                "question_id": row["question_id"],
                "value": row["static_all_evidence_sessions"],
            }
            for row in group
        ]
        union_all = [
            {
                "question_id": row["question_id"],
                "value": row["all_evidence_sessions"],
            }
            for row in group
        ]
        matched_exact = [
            {
                "question_id": row["question_id"],
                "value": row["matched_static_exact_block_any"],
            }
            for row in group
        ]
        matched_all = [
            {
                "question_id": row["question_id"],
                "value": row["matched_static_all_evidence_sessions"],
            }
            for row in group
        ]
        static16_exact = [
            {
                "question_id": row["question_id"],
                "value": row["static_top16_exact_block_any"],
            }
            for row in group
        ]
        static16_all = [
            {
                "question_id": row["question_id"],
                "value": row["static_top16_all_evidence_sessions"],
            }
            for row in group
        ]
        item = {
            "method": method,
            "queries": len(group),
            "trajectory_states": int(group[0]["states"]),
            "maximum_blocks_without_reuse": int(group[0]["states"]) * topk,
            "mean_unique_blocks": mean(float(row["unique_blocks"]) for row in group),
            "mean_working_set_tokens": mean(
                float(row["working_set_tokens"]) for row in group
            ),
            "working_set_tokens_quantiles": quantiles(
                float(row["working_set_tokens"]) for row in group
            ),
            "mean_matched_static_blocks": mean(
                float(row["matched_static_blocks"]) for row in group
            ),
            "temporal_reuse_fraction": 1
            - float(mean(float(row["unique_blocks"]) for row in group))
            / (int(group[0]["states"]) * topk),
            "exact_block_any": paired_binary(static_exact, union_exact, "value"),
            "all_evidence_sessions": paired_binary(static_all, union_all, "value"),
            "union_vs_matched_static_exact_block_any": paired_binary(
                matched_exact, union_exact, "value"
            ),
            "union_vs_matched_static_all_evidence_sessions": paired_binary(
                matched_all, union_all, "value"
            ),
            "union_vs_static_top16_exact_block_any": paired_binary(
                static16_exact, union_exact, "value"
            ),
            "union_vs_static_top16_all_evidence_sessions": paired_binary(
                static16_all, union_all, "value"
            ),
        }
        frontier_summary.append(item)
        for kind in sorted({question_type[str(row["question_id"])] for row in group}):
            typed = [
                row
                for row in group
                if question_type[str(row["question_id"])] == kind
            ]
            frontier_by_type.append(
                {
                    "method": method,
                    "question_type": kind,
                    "queries": len(typed),
                    "static_exact_block_any": mean(
                        float(row["static_exact_block_any"]) for row in typed
                    ),
                    "trajectory_union_exact_block_any": mean(
                        float(row["exact_block_any"]) for row in typed
                    ),
                    "static_all_evidence_sessions": mean(
                        float(row["static_all_evidence_sessions"]) for row in typed
                    ),
                    "trajectory_union_all_evidence_sessions": mean(
                        float(row["all_evidence_sessions"]) for row in typed
                    ),
                    "matched_static_exact_block_any": mean(
                        float(row["matched_static_exact_block_any"]) for row in typed
                    ),
                    "matched_static_all_evidence_sessions": mean(
                        float(row["matched_static_all_evidence_sessions"])
                        for row in typed
                    ),
                }
            )

    summary = {
        "source": "all-500 LongMemEval generated-state retrieval across independent 10M shards",
        "protocol": {
            "partitions": args.partitions,
            "tokens_per_partition": 10_000_000,
            "partitions_are_independent_not_one_80m_memory": True,
            "questions_are_unique_across_partitions": True,
            "selection_uses_answer": False,
            "novel_answer_overlap_queries": 0,
            "unit_of_inference": "unique question_id",
            "final_topk": topk,
            "per_state_working_set_tokens": topk * 64,
        },
        "queries": len(all_queries),
        "positive_queries": sum(not row["is_abstention"] for row in all_queries),
        "abstention_queries": sum(bool(row["is_abstention"]) for row in all_queries),
        "mean_positive_sessions_per_answerable_query": mean(
            float(len(row["positive_session_rows"]))
            for row in all_queries
            if not row["is_abstention"]
        ),
        "mean_exact_blocks_per_answerable_query": mean(
            float(len(row["positive_block_ids"]))
            for row in all_queries
            if not row["is_abstention"]
        ),
        "question_types": dict(
            sorted(
                (kind, sum(row["question_type"] == kind for row in all_queries))
                for kind in {str(row["question_type"]) for row in all_queries}
            )
        ),
        "generation": {
            "model": str(all_plans[0]["model_name_or_path"]),
            "mean_seconds_per_query": mean(
                float(row["generation_seconds"]) for row in all_plans
            ),
            "mean_generated_tokens": mean(
                float(row["generated_tokens"]) for row in all_plans
            ),
        },
        "shards": shard_summaries,
        "quality": summarize_quality(all_rows, topk=topk),
        "state_vs_full_question": state_comparisons,
        "consecutive_state_transitions": consecutive_transitions,
        "router_maturation": router_maturation,
        "bounded_first_seen_frontier": bounded_frontier_summary,
        "trajectory_frontier": frontier_summary,
        "trajectory_frontier_by_question_type": frontier_by_type,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
