from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate all LongMemEval PCA reader shards.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--coarse_input_dir", type=Path)
    parser.add_argument("--coarse_method", default="owner_metadata_block_bm25")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected_queries", type=int, default=500)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    values = list(values)
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def bootstrap_rate(values: list[float], samples: int, rng: np.random.Generator) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(array), size=(samples, len(array)))
    estimates = array[draws].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def binomial_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, item) for item in range(0, min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def load_coarse_rows(
    input_dir: Path | None, method: str
) -> dict[str, dict[str, Any]]:
    if input_dir is None:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(input_dir.glob("part*/rows.jsonl")):
        for row in read_jsonl(path):
            if str(row["method"]) == method:
                rows[str(row["question_id"])] = row
    return rows


def conditional_rate(
    rows: Iterable[dict[str, Any]], condition: str, condition_value: bool
) -> dict[str, Any]:
    selected = [row for row in rows if bool(row[condition]) is condition_value]
    return {
        "queries": len(selected),
        "answer_at_48": mean(float(row["answer_contains"]) for row in selected),
    }


def joint_retrieval_condition(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if bool(row["exact_block_any"]) and bool(row["all_evidence_sessions"])
    ]
    return {
        "queries": len(selected),
        "answer_at_48": mean(float(row["answer_contains"]) for row in selected),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    for part in sorted(args.input_dir.glob("part*")):
        rows_path = part / "rows.jsonl"
        summary_path = part / "summary.json"
        if rows_path.exists():
            rows.extend(read_jsonl(rows_path))
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["partition"] = part.name
            shards.append(summary)
    deduplicated = {
        (str(row["question_id"]), str(row["method"])): row for row in rows
    }
    rows = list(deduplicated.values())
    methods = sorted({str(row["method"]) for row in rows})
    query_ids = sorted({str(row["question_id"]) for row in rows})
    coarse_by_query = load_coarse_rows(args.coarse_input_dir, args.coarse_method)
    if len(query_ids) != args.expected_queries:
        raise RuntimeError(f"expected {args.expected_queries} queries, found {len(query_ids)}")
    for method in methods:
        count = sum(str(row["method"]) == method for row in rows)
        if count != args.expected_queries:
            raise RuntimeError(f"method {method} has {count}/{args.expected_queries} rows")

    quality: list[dict[str, Any]] = []
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for method in methods:
        group = [row for row in rows if str(row["method"]) == method]
        by_method[method] = {str(row["question_id"]): row for row in group}
        answerable = [row for row in group if not bool(row["is_abstention"])]
        abstention = [row for row in group if bool(row["is_abstention"])]
        contains = [float(row["answer_contains"]) for row in answerable]
        strict = [float(row["strict_correct"]) for row in group]
        quality.append(
            {
                "method": method,
                "queries": len(group),
                "answerable_queries": len(answerable),
                "abstention_queries": len(abstention),
                "answer_at_48": mean(contains),
                "answer_at_48_count": sum(int(value) for value in contains),
                "answer_at_48_all500_reference_contains": mean(
                    float(row["answer_contains"]) for row in group
                ),
                "answer_at_48_bootstrap_95_ci": bootstrap_rate(
                    contains, args.bootstrap_samples, rng
                ),
                "mean_token_f1": mean(float(row["token_f1"]) for row in answerable),
                "abstention_refusal_accuracy": mean(
                    float(row["refusal"]) for row in abstention
                ),
                "strict_all500_accuracy": mean(strict),
                "strict_all500_bootstrap_95_ci": bootstrap_rate(
                    strict, args.bootstrap_samples, rng
                ),
                "exact_block_any": mean(
                    float(row["exact_block_any"]) for row in answerable
                ),
                "all_evidence_sessions": mean(
                    float(row["all_evidence_sessions"]) for row in answerable
                ),
                "reader_conditionals": {
                    "given_exact_block_hit": conditional_rate(
                        answerable, "exact_block_any", True
                    ),
                    "given_exact_block_miss": conditional_rate(
                        answerable, "exact_block_any", False
                    ),
                    "given_all_evidence_sessions_hit": conditional_rate(
                        answerable, "all_evidence_sessions", True
                    ),
                    "given_all_evidence_sessions_miss": conditional_rate(
                        answerable, "all_evidence_sessions", False
                    ),
                    "given_exact_block_and_all_evidence_sessions": (
                        joint_retrieval_condition(answerable)
                    ),
                },
                "by_question_type": [
                    {
                        "question_type": question_type,
                        "queries": len(type_rows),
                        "answer_at_48": mean(
                            float(row["answer_contains"]) for row in type_rows
                        ),
                        "exact_block_any": mean(
                            float(row["exact_block_any"]) for row in type_rows
                        ),
                        "all_evidence_sessions": mean(
                            float(row["all_evidence_sessions"]) for row in type_rows
                        ),
                    }
                    for question_type in sorted(
                        {str(row["question_type"]) for row in answerable}
                    )
                    for type_rows in [[
                        row
                        for row in answerable
                        if str(row["question_type"]) == question_type
                    ]]
                ],
                "retrieved_tokens": quantiles(float(row["retrieved_tokens"]) for row in group),
                "selected_blocks": quantiles(
                    float(row["selected_blocks_after_packing"]) for row in group
                ),
                "generated_tokens": quantiles(float(row["generated_tokens"]) for row in group),
                "coarse_query_seconds": quantiles(
                    float(row["coarse_query_seconds"]) for row in group
                ),
                "query_profile_seconds": quantiles(
                    float(row["query_profile_seconds"]) for row in group
                ),
                "candidate_profile_seconds": quantiles(
                    float(row["candidate_profile_seconds"]) for row in group
                ),
                "pca_score_seconds": quantiles(
                    float(row["pca_score_seconds"]) for row in group
                ),
                "exact_score_seconds": quantiles(
                    float(row.get("exact_score_seconds", 0.0)) for row in group
                ),
                "model_native_retrieval_seconds": quantiles(
                    float(row.get("model_native_retrieval_seconds", 0.0))
                    for row in group
                ),
                "generation_seconds": quantiles(
                    float(row["generation_seconds"]) for row in group
                ),
                "online_total_seconds": quantiles(
                    float(row["online_total_seconds"]) for row in group
                ),
                "max_retrieval_budget_violation": max(
                    0,
                    max(int(row["retrieved_tokens"]) - int(row["retrieval_token_budget"]) for row in group),
                ),
            }
        )

    baseline = by_method["bm25_top31"]
    paired: list[dict[str, Any]] = []
    for method in methods:
        if method == "bm25_top31":
            continue
        treatment = by_method[method]
        ids = sorted(baseline)
        answerable_ids = [item for item in ids if not bool(baseline[item]["is_abstention"])]
        wins = sum(
            bool(treatment[item]["answer_contains"]) and not bool(baseline[item]["answer_contains"])
            for item in answerable_ids
        )

    coarse_ceiling: dict[str, Any] | None = None
    pipeline_decomposition: list[dict[str, Any]] = []
    if coarse_by_query:
        if set(coarse_by_query) != set(query_ids):
            raise RuntimeError(
                f"coarse rows cover {len(coarse_by_query)}/{len(query_ids)} queries"
            )
        answerable_ids = [
            item for item in query_ids if not bool(baseline[item]["is_abstention"])
        ]
        coarse_ceiling = {
            "method": args.coarse_method,
            "candidate_blocks": 128,
            "answerable_queries": len(answerable_ids),
            "scope_candidate_blocks": quantiles(
                float(coarse_by_query[item]["candidate_blocks"]) for item in query_ids
            ),
            "scope_candidate_fraction": quantiles(
                float(coarse_by_query[item]["candidate_fraction"]) for item in query_ids
            ),
            "exact_block_any_at_128": mean(
                float(coarse_by_query[item]["exact_block_any_at_128"])
                for item in answerable_ids
            ),
            "all_evidence_sessions_at_128": mean(
                float(coarse_by_query[item]["all_evidence_sessions_at_128"])
                for item in answerable_ids
            ),
            "query_seconds": quantiles(
                float(coarse_by_query[item]["query_seconds"]) for item in query_ids
            ),
        }
        for method in methods:
            current = by_method[method]
            coarse_exact_hits = [
                item
                for item in answerable_ids
                if bool(coarse_by_query[item]["exact_block_any_at_128"])
            ]
            fine_hits = [
                item for item in coarse_exact_hits if bool(current[item]["exact_block_any"])
            ]
            final_exact_hits = [
                item for item in answerable_ids if bool(current[item]["exact_block_any"])
            ]
            pipeline_decomposition.append(
                {
                    "method": method,
                    "l2_exact_hit_queries": len(coarse_exact_hits),
                    "l1_exact_hit_given_l2_hit": (
                        len(fine_hits) / len(coarse_exact_hits) if coarse_exact_hits else None
                    ),
                    "l1_exact_hits_lost_after_l2": len(coarse_exact_hits) - len(fine_hits),
                    "l0_answer_given_final_exact_hit": mean(
                        float(current[item]["answer_contains"])
                        for item in final_exact_hits
                    ),
                    "final_exact_hit_and_answer": sum(
                        bool(current[item]["answer_contains"])
                        for item in final_exact_hits
                    ),
                    "final_exact_hit_but_reader_miss": sum(
                        not bool(current[item]["answer_contains"])
                        for item in final_exact_hits
                    ),
                    "answer_without_final_exact_hit": sum(
                        bool(current[item]["answer_contains"])
                        and not bool(current[item]["exact_block_any"])
                        for item in answerable_ids
                    ),
                }
            )

    pca_qk_agreement: dict[str, Any] | None = None
    if {"pca64_selected16_top31", "exact_qk_selected16_top31"}.issubset(by_method):
        pca = by_method["pca64_selected16_top31"]
        qk = by_method["exact_qk_selected16_top31"]
        answerable_ids = [
            item for item in query_ids if not bool(baseline[item]["is_abstention"])
        ]
        overlaps = []
        for item in query_ids:
            left = set(map(int, pca[item]["selected_block_ids"]))
            right = set(map(int, qk[item]["selected_block_ids"]))
            overlaps.append(len(left & right) / len(left | right))
        pca_qk_agreement = {
            "mean_selected_block_jaccard": mean(overlaps),
            "exact_block_outcome_agreement": mean(
                float(bool(pca[item]["exact_block_any"]) == bool(qk[item]["exact_block_any"]))
                for item in answerable_ids
            ),
            "answer_at_48_outcome_agreement": mean(
                float(bool(pca[item]["answer_contains"]) == bool(qk[item]["answer_contains"]))
                for item in answerable_ids
            ),
        }
        losses = sum(
            bool(baseline[item]["answer_contains"]) and not bool(treatment[item]["answer_contains"])
            for item in answerable_ids
        )
        paired.append(
            {
                "baseline": "bm25_top31",
                "treatment": method,
                "answer_at_48_wins": wins,
                "answer_at_48_losses": losses,
                "answer_at_48_ties": len(answerable_ids) - wins - losses,
                "two_sided_binomial_p": binomial_two_sided(wins, losses),
                "mean_generation_time_delta_seconds": mean(
                    float(treatment[item]["generation_seconds"])
                    - float(baseline[item]["generation_seconds"])
                    for item in ids
                ),
                "mean_online_total_time_delta_seconds": mean(
                    float(treatment[item]["online_total_seconds"])
                    - float(baseline[item]["online_total_seconds"])
                    for item in ids
                ),
            }
        )

    starts = [float(row["process_started_epoch"]) for row in shards]
    finishes = [float(row["process_finished_epoch"]) for row in shards]
    elapsed = [float(row["process_elapsed_seconds"]) for row in shards]
    parallel_wall = max(finishes) - min(starts) if shards else None
    report = {
        "source": "all-500 LongMemEval 10M per-head PCA64 INT4 reader aggregation",
        "protocol": {
            "partitions": len(shards),
            "tokens_per_partition": 10_000_000,
            "partitions_are_independent": True,
            "queries": len(query_ids),
            "retrieval_token_budget": 2000,
            "max_new_tokens": 48,
            "selection_uses_answer": False,
            "answer_at_48_definition": "normalized reference string occurs in at most 48 generated tokens",
        },
        "quality": quality,
        "coarse_top128_ceiling": coarse_ceiling,
        "pipeline_decomposition": pipeline_decomposition,
        "pca_vs_exact_qk_agreement": pca_qk_agreement,
        "paired_vs_bm25": paired,
        "parallel_runtime": {
            "eight_process_wall_seconds": parallel_wall,
            "sum_gpu_process_seconds": sum(elapsed),
            "observed_parallel_speedup_vs_serial_process_sum": (
                sum(elapsed) / parallel_wall if parallel_wall else None
            ),
            "queries_per_wall_second": (
                len(query_ids) / parallel_wall if parallel_wall else None
            ),
            "steady_state_wall_seconds_if_all_shards_start_together": (
                max(elapsed) if elapsed else None
            ),
            "steady_state_speedup_vs_serial_process_sum": (
                sum(elapsed) / max(elapsed) if elapsed else None
            ),
            "observed_wall_includes_one_shard_restart": True,
        },
        "integrity_audit": {
            "unique_question_method_rows": len(rows),
            "unique_queries": len(query_ids),
            "all_rows_use_10m_memory": all(
                int(row["memory_tokens"]) == 10_000_000 for row in rows
            ),
            "answer_never_used_for_selection": all(
                not bool(row["selection_uses_answer"]) for row in rows
            ),
            "retrieval_budget_violations": sum(
                int(row["retrieved_tokens"]) > int(row["retrieval_token_budget"])
                for row in rows
            ),
            "generation_length_violations": sum(
                int(row["generated_tokens"]) > 48 for row in rows
            ),
        },
        "offline_per_shard": {
            "model_load_seconds": quantiles(
                float(row["model_load_seconds"]) for row in shards
            ),
            "pca_calibration_seconds": quantiles(
                float(row["pca_calibration_seconds"]) for row in shards
            ),
            "needed_block_text_decode_seconds": quantiles(
                float(row["decode_needed_block_text_seconds"]) for row in shards
            ),
            "pca_retained_energy": quantiles(
                float(row["pca_retained_energy_mean"]) for row in shards
            ),
        },
        "shards": shards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
