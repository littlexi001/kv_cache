from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import binomtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate static versus dynamic LongMemEval reader shards."
    )
    parser.add_argument("--data_pattern", required=True)
    parser.add_argument("--reader_pattern", required=True)
    parser.add_argument("--selection_pattern", required=True)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def paired_binary(
    baseline: dict[str, dict[str, Any]],
    treatment: dict[str, dict[str, Any]],
    metric: str,
    ids: list[str],
) -> dict[str, Any]:
    wins = sum(not bool(baseline[qid][metric]) and bool(treatment[qid][metric]) for qid in ids)
    losses = sum(bool(baseline[qid][metric]) and not bool(treatment[qid][metric]) for qid in ids)
    return {
        "queries": len(ids),
        "baseline_rate": mean(float(baseline[qid][metric]) for qid in ids),
        "treatment_rate": mean(float(treatment[qid][metric]) for qid in ids),
        "delta": mean(
            float(treatment[qid][metric]) - float(baseline[qid][metric]) for qid in ids
        ),
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
    baseline: dict[str, dict[str, Any]],
    treatment: dict[str, dict[str, Any]],
    metric: str,
    ids: list[str],
    *,
    lower_is_better: bool,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    differences = np.asarray(
        [float(treatment[qid][metric]) - float(baseline[qid][metric]) for qid in ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(samples, len(differences)))
    sampled = differences[draws].mean(axis=1)
    wins = differences < 0 if lower_is_better else differences > 0
    losses = differences > 0 if lower_is_better else differences < 0
    output = {
        "queries": len(ids),
        "baseline_mean": mean(float(baseline[qid][metric]) for qid in ids),
        "treatment_mean": mean(float(treatment[qid][metric]) for qid in ids),
        "treatment_minus_baseline": float(differences.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "ties": int((differences == 0).sum()),
    }
    if metric == "reference_nll":
        output["perplexity_ratio_exp_mean_delta"] = math.exp(float(differences.mean()))
    return output


def main() -> None:
    args = parse_args()
    all_queries = []
    all_reader_rows = []
    all_selection_rows = []
    for partition in range(args.partitions):
        data_dir = Path(args.data_pattern.format(partition=partition))
        reader_dir = Path(args.reader_pattern.format(partition=partition))
        selection_dir = Path(args.selection_pattern.format(partition=partition))
        queries = read_jsonl(data_dir / "queries.jsonl")
        local_to_question = {
            int(row["query_id"]): str(row["question_id"]) for row in queries
        }
        for query in queries:
            item = dict(query)
            item["partition"] = partition
            all_queries.append(item)
        for row in read_jsonl(reader_dir / "rows.jsonl"):
            item = dict(row)
            item["partition"] = partition
            item["question_id"] = local_to_question[int(row["query_id"])]
            all_reader_rows.append(item)
        for row in read_jsonl(selection_dir / "rows.jsonl"):
            if str(row["method"]) not in {
                "static_top12",
                "evidence_state_dynamic_top12",
            }:
                continue
            item = dict(row)
            item["partition"] = partition
            item["question_id"] = local_to_question[int(row["query_id"])]
            all_selection_rows.append(item)

    question_ids = [str(row["question_id"]) for row in all_queries]
    if len(question_ids) != 500 or len(set(question_ids)) != 500:
        raise RuntimeError("expected 500 unique questions")
    query_by_id = {str(row["question_id"]): row for row in all_queries}
    reader_by_method: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in all_reader_rows:
        reader_by_method[str(row["method"])][str(row["question_id"])] = row
    selection_by_method: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in all_selection_rows:
        selection_by_method[str(row["method"])][str(row["question_id"])] = row
    static = reader_by_method["static_top12"]
    dynamic = reader_by_method["evidence_state_dynamic_top12"]
    if set(static) != set(question_ids) or set(dynamic) != set(question_ids):
        raise RuntimeError("reader methods do not cover all questions")
    all_ids = sorted(question_ids)
    positive_ids = [qid for qid in all_ids if not query_by_id[qid]["is_abstention"]]
    abstention_ids = [qid for qid in all_ids if query_by_id[qid]["is_abstention"]]

    quality = []
    for method, rows in (
        ("static_top12", static),
        ("evidence_state_dynamic_top12", dynamic),
    ):
        quality.append(
            {
                "method": method,
                "queries": len(rows),
                "mean_working_set_tokens": mean(
                    float(rows[qid]["working_set_tokens"]) for qid in all_ids
                ),
                "mean_reference_nll_all": mean(
                    float(rows[qid]["reference_nll"]) for qid in all_ids
                ),
                "mean_reference_nll_answerable": mean(
                    float(rows[qid]["reference_nll"]) for qid in positive_ids
                ),
                "mean_positive_token_f1": mean(
                    float(rows[qid]["token_f1"]) for qid in positive_ids
                ),
                "positive_exact_match": mean(
                    float(rows[qid]["normalized_exact_match"]) for qid in positive_ids
                ),
                "positive_answer_contains": mean(
                    float(rows[qid]["answer_contains"]) for qid in positive_ids
                ),
                "abstention_refusal_accuracy": mean(
                    float(rows[qid]["predicted_refusal"]) for qid in abstention_ids
                ),
                "mean_nll_seconds": mean(
                    float(rows[qid]["nll_seconds"]) for qid in all_ids
                ),
                "mean_generation_seconds": mean(
                    float(rows[qid]["generation_seconds"]) for qid in all_ids
                ),
            }
        )

    comparisons = {
        "reference_nll_all": paired_continuous(
            static,
            dynamic,
            "reference_nll",
            all_ids,
            lower_is_better=True,
            samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "reference_nll_answerable": paired_continuous(
            static,
            dynamic,
            "reference_nll",
            positive_ids,
            lower_is_better=True,
            samples=args.bootstrap_samples,
            seed=args.seed + 1,
        ),
        "positive_token_f1": paired_continuous(
            static,
            dynamic,
            "token_f1",
            positive_ids,
            lower_is_better=False,
            samples=args.bootstrap_samples,
            seed=args.seed + 2,
        ),
        "positive_exact_match": paired_binary(
            static, dynamic, "normalized_exact_match", positive_ids
        ),
        "positive_answer_contains": paired_binary(
            static, dynamic, "answer_contains", positive_ids
        ),
        "abstention_refusal_accuracy": paired_binary(
            static, dynamic, "predicted_refusal", abstention_ids
        ),
        "generation_seconds": paired_continuous(
            static,
            dynamic,
            "generation_seconds",
            all_ids,
            lower_is_better=True,
            samples=args.bootstrap_samples,
            seed=args.seed + 3,
        ),
    }

    by_type = []
    for kind in sorted({str(row["question_type"]) for row in all_queries}):
        ids = [
            qid
            for qid in positive_ids
            if str(query_by_id[qid]["question_type"]) == kind
        ]
        if not ids:
            continue
        by_type.append(
            {
                "question_type": kind,
                "queries": len(ids),
                "reference_nll": paired_continuous(
                    static,
                    dynamic,
                    "reference_nll",
                    ids,
                    lower_is_better=True,
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                ),
                "token_f1": paired_continuous(
                    static,
                    dynamic,
                    "token_f1",
                    ids,
                    lower_is_better=False,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 1,
                ),
            }
        )

    static_selection = selection_by_method["static_top12"]
    dynamic_selection = selection_by_method["evidence_state_dynamic_top12"]
    retrieval_conditioned = []
    groups: dict[str, list[str]] = defaultdict(list)
    for qid in positive_ids:
        static_all = bool(static_selection[qid]["all_evidence_sessions_at_12"])
        dynamic_all = bool(dynamic_selection[qid]["all_evidence_sessions_at_12"])
        if not static_all and dynamic_all:
            label = "all_evidence_rescued"
        elif static_all and not dynamic_all:
            label = "all_evidence_lost"
        elif static_all:
            label = "both_complete"
        else:
            label = "both_incomplete"
        groups[label].append(qid)
    for label in (
        "all_evidence_rescued",
        "all_evidence_lost",
        "both_complete",
        "both_incomplete",
    ):
        ids = groups[label]
        retrieval_conditioned.append(
            {
                "group": label,
                "queries": len(ids),
                "mean_reference_nll_delta": mean(
                    float(dynamic[qid]["reference_nll"])
                    - float(static[qid]["reference_nll"])
                    for qid in ids
                ),
                "mean_token_f1_delta": mean(
                    float(dynamic[qid]["token_f1"])
                    - float(static[qid]["token_f1"])
                    for qid in ids
                ),
            }
        )

    summary = {
        "source": "all-500 Qwen3-8B reader for evidence-conditioned LongMemEval 10M retrieval",
        "protocol": {
            "partitions": args.partitions,
            "tokens_per_partition": 10_000_000,
            "partitions_are_independent_not_one_80m_memory": True,
            "selection_uses_answer": False,
            "reference_used_only_after_retrieval_for_scoring": True,
            "teacher_forced_reference_nll_is_primary": True,
            "token_f1_is_exploratory_not_official_longmemeval_judge": True,
            "unit_of_inference": "unique question_id",
        },
        "queries": len(all_ids),
        "positive_queries": len(positive_ids),
        "abstention_queries": len(abstention_ids),
        "quality": quality,
        "dynamic_vs_static_top12": comparisons,
        "dynamic_vs_static_by_question_type": by_type,
        "reader_delta_by_retrieval_completeness_change": retrieval_conditioned,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
