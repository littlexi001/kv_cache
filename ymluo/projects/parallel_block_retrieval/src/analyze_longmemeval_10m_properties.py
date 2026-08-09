from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import binomtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze structural, temporal, interference, and abstention properties."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--retrieval_summary", required=True)
    parser.add_argument("--retrieval_rows", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.1)),
        "p90": float(np.quantile(array, 0.9)),
    }


def paired_binary(
    before: list[dict[str, Any]], after: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    before_by_query = {int(row["query_id"]): row for row in before}
    after_by_query = {int(row["query_id"]): row for row in after}
    query_ids = sorted(set(before_by_query) & set(after_by_query))
    wins = sum(
        not bool(before_by_query[qid][metric]) and bool(after_by_query[qid][metric])
        for qid in query_ids
    )
    losses = sum(
        bool(before_by_query[qid][metric]) and not bool(after_by_query[qid][metric])
        for qid in query_ids
    )
    return {
        "queries": len(query_ids),
        "wins": wins,
        "losses": losses,
        "ties": len(query_ids) - wins - losses,
        "two_sided_binomial_p": (
            float(binomtest(wins, wins + losses, 0.5).pvalue)
            if wins + losses
            else 1.0
        ),
    }


def method_depth(method: str, prefix: str) -> int | None:
    match = re.fullmatch(re.escape(prefix) + r"(\d+)_block_bm25", method)
    return int(match.group(1)) if match else None


def summarize_temporal_structure(
    queries: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    sessions_by_owner: dict[int, list[dict[str, Any]]],
    allowed_query_ids: set[int] | None = None,
) -> dict[str, Any]:
    output = {}
    for question_type in sorted({str(row["question_type"]) for row in queries}):
        ages = []
        ranks = []
        normalized_ranks = []
        included_queries = 0
        for query in queries:
            query_id = int(query["query_id"])
            if (
                query["is_abstention"]
                or query["question_type"] != question_type
                or (
                    allowed_query_ids is not None
                    and query_id not in allowed_query_ids
                )
            ):
                continue
            included_queries += 1
            owner_sessions = sorted(
                sessions_by_owner[int(query["owner_row"])],
                key=lambda row: (-int(row["date_minutes"]), int(row["session_row"])),
            )
            recency_rank = {
                int(row["session_row"]): index + 1
                for index, row in enumerate(owner_sessions)
            }
            for session_row in query["positive_session_rows"]:
                session = sessions[int(session_row)]
                ages.append(
                    (int(query["question_date_minutes"]) - int(session["date_minutes"]))
                    / 1440.0
                )
                rank = recency_rank[int(session_row)]
                ranks.append(float(rank))
                normalized_ranks.append(rank / len(owner_sessions))
        output[question_type] = {
            "queries": included_queries,
            "evidence_sessions": len(ages),
            "age_days": quantiles(ages),
            "recency_rank": quantiles(ranks),
            "mean_normalized_recency_rank": mean(normalized_ranks),
            "fraction_in_latest_3": mean(float(rank <= 3) for rank in ranks),
            "fraction_in_latest_8": mean(float(rank <= 8) for rank in ranks),
        }
    return output


def summarize_filtered_quality(
    rows_by_method: dict[str, list[dict[str, Any]]],
    methods: list[str],
    query_ids: set[int],
) -> list[dict[str, Any]]:
    metrics = (
        "exact_block_any_at_8",
        "latest_exact_block_any_at_8",
        "evidence_session_any_at_8",
        "evidence_session_recall_at_8",
        "all_evidence_sessions_at_8",
    )
    output = []
    for method in methods:
        group = [
            row
            for row in rows_by_method.get(method, [])
            if int(row["query_id"]) in query_ids
        ]
        if not group:
            continue
        output.append(
            {
                "method": method,
                "queries": len(group),
                **{
                    metric: mean(float(row[metric]) for row in group)
                    for metric in metrics
                },
            }
        )
    return output


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_summary = read_json(data_dir / "summary.json")
    retrieval_summary = read_json(args.retrieval_summary)
    queries = read_jsonl(data_dir / "queries.jsonl")
    sessions = read_jsonl(data_dir / "session_manifest.jsonl")
    owners = read_jsonl(data_dir / "owner_manifest.jsonl")
    rows = read_jsonl(args.retrieval_rows)
    query_by_id = {int(row["query_id"]): row for row in queries}
    owner_to_index = {
        int(row["owner_row"]): index for index, row in enumerate(owners)
    }
    sessions_by_owner: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for session in sessions:
        sessions_by_owner[int(session["owner_row"])].append(session)

    evidence_cardinality = {
        "positive_sessions": quantiles(
            [
                float(len(row["positive_session_rows"]))
                for row in queries
                if not row["is_abstention"]
            ]
        ),
        "positive_exact_blocks": quantiles(
            [
                float(len(row["positive_block_ids"]))
                for row in queries
                if not row["is_abstention"]
            ]
        ),
        "history_sessions": quantiles(
            [float(row["history_sessions"]) for row in queries]
        ),
    }

    positive_queries = [row for row in queries if not row["is_abstention"]]
    causal_valid_positive_ids = set()
    future_evidence_examples = []
    future_evidence_sessions = 0
    queries_with_future_evidence = 0
    future_evidence_queries_by_type: collections.Counter[str] = collections.Counter()
    queries_with_future_history = 0
    future_history_sessions = 0
    for query in queries:
        question_date = int(query["question_date_minutes"])
        owner_sessions = sessions_by_owner[int(query["owner_row"])]
        future_history = [
            row for row in owner_sessions if int(row["date_minutes"]) > question_date
        ]
        future_history_sessions += len(future_history)
        queries_with_future_history += int(bool(future_history))
        if query["is_abstention"]:
            continue
        future_evidence = [
            sessions[int(session_row)]
            for session_row in query["positive_session_rows"]
            if int(sessions[int(session_row)]["date_minutes"]) > question_date
        ]
        if not future_evidence:
            causal_valid_positive_ids.add(int(query["query_id"]))
            continue
        queries_with_future_evidence += 1
        future_evidence_sessions += len(future_evidence)
        future_evidence_queries_by_type[str(query["question_type"])] += 1
        if len(future_evidence_examples) < 20:
            future_evidence_examples.append(
                {
                    "query_id": int(query["query_id"]),
                    "question_id": str(query["question_id"]),
                    "question_type": str(query["question_type"]),
                    "question_date": str(query["question_date"]),
                    "future_evidence": [
                        {
                            "session_row": int(row["session_row"]),
                            "session_id": str(row["session_id"]),
                            "date": str(row["date"]),
                            "lead_days": (
                                int(row["date_minutes"]) - question_date
                            )
                            / 1440.0,
                        }
                        for row in future_evidence
                    ],
                }
            )

    temporal_causality_audit = {
        "scope": "selected 64-query shared-10M evaluation memory",
        "all_queries": len(queries),
        "positive_queries": len(positive_queries),
        "queries_with_any_future_history_session": queries_with_future_history,
        "future_history_sessions": future_history_sessions,
        "queries_with_future_positive_evidence": queries_with_future_evidence,
        "future_positive_evidence_sessions": future_evidence_sessions,
        "causal_valid_positive_queries": len(causal_valid_positive_ids),
        "future_evidence_queries_by_type": dict(future_evidence_queries_by_type),
        "future_evidence_examples": future_evidence_examples,
        "interpretation": (
            "Official all-history retrieval can retain every haystack session. "
            "Causal temporal claims use only positive queries whose evidence "
            "timestamps do not exceed question_date."
        ),
    }
    temporal = summarize_temporal_structure(queries, sessions, sessions_by_owner)
    causal_temporal = summarize_temporal_structure(
        queries,
        sessions,
        sessions_by_owner,
        allowed_query_ids=causal_valid_positive_ids,
    )

    rows_by_method: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        rows_by_method[str(row["method"])].append(row)

    owner_router = {}
    for method, group in rows_by_method.items():
        depth = method_depth(method, "owner_router")
        if depth is None:
            continue
        owner_router[str(depth)] = {
            "queries": len(group),
            "true_owner_recall": mean(
                float(
                    owner_to_index[
                        int(query_by_id[int(row["query_id"])]["owner_row"])
                    ]
                    in row["selected_owner_indices"]
                )
                for row in group
            ),
        }

    positive_ids = {
        int(row["query_id"]) for row in queries if not row["is_abstention"]
    }
    depth_transitions = []
    family_prefix = "owner_metadata_session"
    available_depths = sorted(
        depth
        for method in rows_by_method
        if (depth := method_depth(method, family_prefix)) is not None
    )
    for before_depth, after_depth in zip(available_depths[:-1], available_depths[1:]):
        before = [
            row
            for row in rows_by_method[
                f"{family_prefix}{before_depth}_block_bm25"
            ]
            if int(row["query_id"]) in positive_ids
        ]
        after = [
            row
            for row in rows_by_method[
                f"{family_prefix}{after_depth}_block_bm25"
            ]
            if int(row["query_id"]) in positive_ids
        ]
        before_by_query = {int(row["query_id"]): row for row in before}
        after_by_query = {int(row["query_id"]): row for row in after}
        depth_transitions.append(
            {
                "before_depth": before_depth,
                "after_depth": after_depth,
                "exact_block_any_at_8": paired_binary(
                    before, after, "exact_block_any_at_8"
                ),
                "latest_exact_block_any_at_8": paired_binary(
                    before, after, "latest_exact_block_any_at_8"
                ),
                "all_evidence_sessions_at_8": paired_binary(
                    before, after, "all_evidence_sessions_at_8"
                ),
                "mean_evidence_session_recall_delta_at_8": mean(
                    float(after_by_query[qid]["evidence_session_recall_at_8"])
                    - float(before_by_query[qid]["evidence_session_recall_at_8"])
                    for qid in sorted(before_by_query)
                ),
            }
        )

    quality_by_method = {
        str(row["method"]): row for row in retrieval_summary["quality"]
    }
    core_methods = [
        "global_block_bm25",
        "session_router8_block_bm25",
        "session_router16_block_bm25",
        "owner_metadata_block_bm25",
        "owner_metadata_session1_block_bm25",
        "owner_metadata_session3_block_bm25",
        "owner_metadata_session8_block_bm25",
        "owner_router8_session8_block_bm25",
        "owner_metadata_recent3_block_bm25",
        "owner_metadata_recent8_block_bm25",
        "owner_metadata_hybrid3_block_bm25",
        "owner_metadata_hybrid8_block_bm25",
    ]
    for pool in (8, 16, 32):
        for depth in (1, 3, 8):
            core_methods.append(
                f"owner_metadata_semantic{pool}_recent{depth}_block_bm25"
            )
    core_quality = [quality_by_method[name] for name in core_methods if name in quality_by_method]
    causal_valid_core_quality = summarize_filtered_quality(
        rows_by_method,
        core_methods,
        causal_valid_positive_ids,
    )

    global_quality = quality_by_method["global_block_bm25"]
    structural_comparisons = []
    for name in (
        "session_router8_block_bm25",
        "session_router16_block_bm25",
        "owner_metadata_block_bm25",
        "owner_metadata_session3_block_bm25",
        "owner_router8_session8_block_bm25",
    ):
        item = quality_by_method[name]
        structural_comparisons.append(
            {
                "method": name,
                "candidate_reduction_vs_global": float(
                    global_quality["mean_candidate_blocks"]
                    / item["mean_candidate_blocks"]
                ),
                "exact_block_any_at_8_delta_vs_global": float(
                    item["exact_block_any_at_8"]
                    - global_quality["exact_block_any_at_8"]
                ),
                "all_evidence_sessions_at_8_delta_vs_global": float(
                    item["all_evidence_sessions_at_8"]
                    - global_quality["all_evidence_sessions_at_8"]
                ),
                "query_speedup_vs_global": float(
                    global_quality["mean_query_seconds"]
                    / item["mean_query_seconds"]
                ),
            }
        )

    quality_by_type = retrieval_summary["quality_by_question_type"]
    temporal_method_types = [
        row
        for row in quality_by_type
        if row["method"]
        in {
            "owner_metadata_session3_block_bm25",
            "owner_metadata_recent3_block_bm25",
            "owner_metadata_hybrid3_block_bm25",
            "owner_metadata_semantic8_recent1_block_bm25",
            "owner_metadata_semantic16_recent1_block_bm25",
            "owner_metadata_semantic16_recent3_block_bm25",
        }
    ]

    output = {
        "source": "LongMemEval shared-10M structural property analysis",
        "protocol": retrieval_summary["protocol"],
        "data_summary": data_summary,
        "evidence_cardinality": evidence_cardinality,
        "temporal_causality_audit": temporal_causality_audit,
        "temporal_evidence_structure_by_type_official_all_history": temporal,
        "temporal_evidence_structure_by_type_causal_valid": causal_temporal,
        "automatic_owner_router": owner_router,
        "structural_comparisons": structural_comparisons,
        "scope_depth_transitions": depth_transitions,
        "core_retrieval_quality": core_quality,
        "causal_valid_core_retrieval_quality": causal_valid_core_quality,
        "temporal_method_quality_by_type": temporal_method_types,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
