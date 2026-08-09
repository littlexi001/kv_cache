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
        description="Aggregate all LongMemEval evidence-conditioned 10M shards."
    )
    parser.add_argument("--data_pattern", required=True)
    parser.add_argument("--result_pattern", required=True)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


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
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def paired_binary(
    baseline: list[dict[str, Any]], treatment: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    base = {str(row["question_id"]): bool(row[metric]) for row in baseline}
    new = {str(row["question_id"]): bool(row[metric]) for row in treatment}
    ids = sorted(set(base) & set(new))
    wins = sum(not base[qid] and new[qid] for qid in ids)
    losses = sum(base[qid] and not new[qid] for qid in ids)
    return {
        "queries": len(ids),
        "baseline_rate": mean(float(base[qid]) for qid in ids),
        "treatment_rate": mean(float(new[qid]) for qid in ids),
        "delta": mean(float(new[qid]) - float(base[qid]) for qid in ids),
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
    samples: int,
    seed: int,
) -> dict[str, Any]:
    base = {str(row["question_id"]): float(row[metric]) for row in baseline}
    new = {str(row["question_id"]): float(row[metric]) for row in treatment}
    ids = sorted(set(base) & set(new))
    differences = np.asarray([new[qid] - base[qid] for qid in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(samples, len(differences)))
    sampled = differences[draws].mean(axis=1)
    return {
        "queries": len(ids),
        "baseline_mean": mean(base[qid] for qid in ids),
        "treatment_mean": mean(new[qid] for qid in ids),
        "mean_delta": float(differences.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "wins": int((differences > 0).sum()),
        "losses": int((differences < 0).sum()),
        "ties": int((differences == 0).sum()),
    }


def main() -> None:
    args = parse_args()
    all_queries = []
    all_rows = []
    all_states = []
    shards = []
    for partition in range(args.partitions):
        data_dir = Path(args.data_pattern.format(partition=partition))
        result_dir = Path(args.result_pattern.format(partition=partition))
        data_summary = read_json(data_dir / "summary.json")
        result_summary = read_json(result_dir / "summary.json")
        queries = read_jsonl(data_dir / "queries.jsonl")
        local_to_question = {
            int(row["query_id"]): str(row["question_id"]) for row in queries
        }
        for query in queries:
            item = dict(query)
            item["partition"] = partition
            all_queries.append(item)
        for row in read_jsonl(result_dir / "rows.jsonl"):
            item = dict(row)
            item["partition"] = partition
            item["question_id"] = local_to_question[int(row["query_id"])]
            all_rows.append(item)
        for row in read_jsonl(result_dir / "states.jsonl"):
            item = dict(row)
            item["partition"] = partition
            item["question_id"] = local_to_question[int(row["query_id"])]
            all_states.append(item)
        shards.append(
            {
                "partition": partition,
                "queries": int(data_summary["query_samples"]),
                "positive_queries": int(data_summary["non_abstention_queries"]),
                "mean_state_generation_seconds": float(
                    result_summary["mean_state_generation_seconds"]
                ),
            }
        )

    question_ids = [str(row["question_id"]) for row in all_queries]
    if len(question_ids) != 500 or len(set(question_ids)) != 500:
        raise RuntimeError("expected 500 unique question ids")
    positive_ids = {
        str(row["question_id"]) for row in all_queries if not row["is_abstention"]
    }
    positive_rows = [
        row for row in all_rows if str(row["question_id"]) in positive_ids
    ]
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positive_rows:
        by_method[str(row["method"])].append(row)

    quality = []
    for method in ("static_top8", "static_top12", "evidence_state_dynamic_top12"):
        group = by_method[method]
        metric_k = 8 if method == "static_top8" else 12
        quality.append(
            {
                "method": method,
                "queries": len(group),
                "mean_working_set_tokens": mean(
                    float(row["working_set_tokens"]) for row in group
                ),
                "working_set_tokens_quantiles": quantiles(
                    float(row["working_set_tokens"]) for row in group
                ),
                "mean_query_milliseconds": 1000
                * float(mean(float(row["query_seconds"]) for row in group)),
                "exact_block_any": mean(
                    float(row[f"exact_block_any_at_{metric_k}"]) for row in group
                ),
                "all_evidence_sessions": mean(
                    float(row[f"all_evidence_sessions_at_{metric_k}"]) for row in group
                ),
                "evidence_session_recall": mean(
                    float(row[f"evidence_session_recall_at_{metric_k}"])
                    for row in group
                ),
            }
        )

    static12 = by_method["static_top12"]
    dynamic12 = by_method["evidence_state_dynamic_top12"]
    comparisons = {
        "exact_block_any": paired_binary(
            static12, dynamic12, "exact_block_any_at_12"
        ),
        "all_evidence_sessions": paired_binary(
            static12, dynamic12, "all_evidence_sessions_at_12"
        ),
        "evidence_session_recall": paired_continuous(
            static12,
            dynamic12,
            "evidence_session_recall_at_12",
            samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }

    by_type = []
    query_type = {
        str(row["question_id"]): str(row["question_type"]) for row in all_queries
    }
    for kind in sorted(set(query_type.values())):
        static_group = [
            row for row in static12 if query_type[str(row["question_id"])] == kind
        ]
        dynamic_group = [
            row for row in dynamic12 if query_type[str(row["question_id"])] == kind
        ]
        if not static_group:
            continue
        by_type.append(
            {
                "question_type": kind,
                "queries": len(static_group),
                "exact_block_any": paired_binary(
                    static_group, dynamic_group, "exact_block_any_at_12"
                ),
                "all_evidence_sessions": paired_binary(
                    static_group, dynamic_group, "all_evidence_sessions_at_12"
                ),
            }
        )

    static8_by_question = {
        str(row["question_id"]): row for row in by_method["static_top8"]
    }
    state_audit = []
    for initial_hit in (False, True):
        group = [
            row
            for row in all_states
            if str(row["question_id"]) in positive_ids
            and bool(
                static8_by_question[str(row["question_id"])]["exact_block_any_at_8"]
            )
            == initial_hit
        ]
        state_audit.append(
            {
                "initial_exact_block_hit": initial_hit,
                "queries": len(group),
                "state_mentions_reference": mean(
                    float(row["state_mentions_reference_posthoc"]) for row in group
                ),
                "state_adds_reference_not_in_question": mean(
                    float(row["state_adds_reference_not_in_question_posthoc"])
                    for row in group
                ),
            }
        )

    added_per_refresh = []
    total_added = []
    for state in all_states:
        total_added.append(len(state["dynamic_frontier_block_ids"]) - 8)
        for refresh in state["refreshes"]:
            added_per_refresh.append(len(refresh["added_block_ids"]))

    summary = {
        "source": "all-500 LongMemEval evidence-conditioned state refresh on independent 10M shards",
        "protocol": {
            "partitions": args.partitions,
            "tokens_per_partition": 10_000_000,
            "partitions_are_independent_not_one_80m_memory": True,
            "questions_are_unique_across_partitions": True,
            "selection_uses_answer": False,
            "state_reads_only_initial_512_tokens": True,
            "static_and_dynamic_max_budget_tokens": 768,
            "final_answer_reader_not_run": True,
            "unit_of_inference": "unique question_id",
        },
        "queries": len(all_queries),
        "positive_queries": len(positive_ids),
        "abstention_queries": len(all_queries) - len(positive_ids),
        "mean_state_generation_seconds": mean(
            float(row["generation_seconds"]) for row in all_states
        ),
        "page_innovation": {
            "added_pages_per_refresh": quantiles(added_per_refresh),
            "total_added_pages_across_three_refreshes": quantiles(total_added),
        },
        "quality": quality,
        "dynamic_vs_static_top12": comparisons,
        "dynamic_vs_static_top12_by_question_type": by_type,
        "state_reference_audit": state_audit,
        "shards": shards,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
