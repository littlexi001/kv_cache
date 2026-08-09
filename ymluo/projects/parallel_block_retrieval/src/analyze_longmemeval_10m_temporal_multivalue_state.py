from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare compact and temporal-multivalue LongMemEval state refresh."
    )
    parser.add_argument("--baseline_selection_pattern", required=True)
    parser.add_argument("--treatment_selection_pattern", required=True)
    parser.add_argument("--baseline_reader_pattern", required=True)
    parser.add_argument("--treatment_reader_pattern", required=True)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def paired_binary(
    baseline: dict[str, dict[str, Any]],
    treatment: dict[str, dict[str, Any]],
    ids: list[str],
    metric: str,
) -> dict[str, Any]:
    base = np.asarray([bool(baseline[qid][metric]) for qid in ids])
    new = np.asarray([bool(treatment[qid][metric]) for qid in ids])
    wins = int((~base & new).sum())
    losses = int((base & ~new).sum())
    return {
        "queries": len(ids),
        "baseline_rate": float(base.mean()),
        "treatment_rate": float(new.mean()),
        "delta": float(new.mean() - base.mean()),
        "wins": wins,
        "losses": losses,
        "two_sided_binomial_p": float(
            binomtest(wins, wins + losses, 0.5).pvalue
        )
        if wins + losses
        else 1.0,
    }


def paired_continuous(
    baseline: dict[str, dict[str, Any]],
    treatment: dict[str, dict[str, Any]],
    ids: list[str],
    metric: str,
    *,
    samples: int,
    seed: int,
    lower_is_better: bool = True,
) -> dict[str, Any]:
    base = np.asarray([float(baseline[qid][metric]) for qid in ids])
    new = np.asarray([float(treatment[qid][metric]) for qid in ids])
    difference = new - base
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(samples, len(ids)))
    sampled = difference[draws].mean(axis=1)
    wins = difference < 0 if lower_is_better else difference > 0
    losses = difference > 0 if lower_is_better else difference < 0
    output = {
        "queries": len(ids),
        "baseline_mean": float(base.mean()),
        "treatment_mean": float(new.mean()),
        "treatment_minus_baseline": float(difference.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "ties": int((difference == 0).sum()),
    }
    if metric == "reference_nll":
        output["perplexity_ratio_exp_mean_delta"] = math.exp(
            float(difference.mean())
        )
    return output


def reader_quality(rows: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    return {
        "queries": len(ids),
        "mean_working_set_tokens": float(
            np.mean([rows[qid]["working_set_tokens"] for qid in ids])
        ),
        "mean_reference_nll": float(
            np.mean([rows[qid]["reference_nll"] for qid in ids])
        ),
        "mean_token_f1": float(np.mean([rows[qid]["token_f1"] for qid in ids])),
        "exact_match": float(
            np.mean([rows[qid]["normalized_exact_match"] for qid in ids])
        ),
        "answer_contains": float(
            np.mean([rows[qid]["answer_contains"] for qid in ids])
        ),
    }


def main() -> None:
    args = parse_args()
    baseline_states = {}
    treatment_states = {}
    baseline_selection: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    treatment_selection: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    baseline_reader: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    treatment_reader: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    partitions = {}
    for partition in range(args.partitions):
        baseline_selection_dir = Path(
            args.baseline_selection_pattern.format(partition=partition)
        )
        treatment_selection_dir = Path(
            args.treatment_selection_pattern.format(partition=partition)
        )
        baseline_reader_dir = Path(
            args.baseline_reader_pattern.format(partition=partition)
        )
        treatment_reader_dir = Path(
            args.treatment_reader_pattern.format(partition=partition)
        )
        for row in read_jsonl(baseline_selection_dir / "states.jsonl"):
            if row["question_type"] == "knowledge-update":
                baseline_states[str(row["question_id"])] = row
        for row in read_jsonl(treatment_selection_dir / "states.jsonl"):
            question_id = str(row["question_id"])
            treatment_states[question_id] = row
            partitions[question_id] = partition
        for target, path in (
            (baseline_selection, baseline_selection_dir / "rows.jsonl"),
            (treatment_selection, treatment_selection_dir / "rows.jsonl"),
        ):
            for row in read_jsonl(path):
                if row["question_type"] == "knowledge-update":
                    target[str(row["method"])][str(row["question_id"])] = row
        for target, path in (
            (baseline_reader, baseline_reader_dir / "rows.jsonl"),
            (treatment_reader, treatment_reader_dir / "rows.jsonl"),
        ):
            for row in read_jsonl(path):
                if row["question_type"] == "knowledge-update":
                    target[str(row["method"])][str(row["question_id"])] = row

    ids = sorted(treatment_states)
    if len(ids) != 78 or set(ids) != set(baseline_states):
        raise RuntimeError("expected the same 78 knowledge-update questions")
    positive_ids = [
        qid for qid in ids if not bool(treatment_states[qid]["is_abstention"])
    ]
    for collection in (
        baseline_selection["static_top12"],
        baseline_selection["evidence_state_dynamic_top12"],
        treatment_selection["static_top12"],
        treatment_selection["evidence_state_dynamic_top12"],
        baseline_reader["static_top12"],
        baseline_reader["evidence_state_dynamic_top12"],
        treatment_reader["static_top12"],
        treatment_reader["evidence_state_dynamic_top12"],
    ):
        if set(collection) != set(ids):
            raise RuntimeError("selection or reader rows do not cover all 78 questions")

    old_dynamic_selection = baseline_selection["evidence_state_dynamic_top12"]
    new_dynamic_selection = treatment_selection["evidence_state_dynamic_top12"]
    old_dynamic_reader = baseline_reader["evidence_state_dynamic_top12"]
    new_dynamic_reader = treatment_reader["evidence_state_dynamic_top12"]
    static_reader = treatment_reader["static_top12"]
    old_mentions = {
        qid: {"value": baseline_states[qid]["state_mentions_reference_posthoc"]}
        for qid in ids
    }
    new_mentions = {
        qid: {"value": treatment_states[qid]["state_mentions_reference_posthoc"]}
        for qid in ids
    }
    output = {
        "protocol": {
            "memory_scope": "eight independent real 10M-token shards",
            "question_type": "knowledge-update",
            "queries": len(ids),
            "answerable_queries": len(positive_ids),
            "baseline_state": "32-token compact fact/slot state",
            "treatment_state": "64-token temporal multivalue/date/provenance state",
            "selection_uses_answer": False,
            "reference_used_for_posthoc_metrics_only": True,
        },
        "state_reference_posthoc": paired_binary(
            old_mentions, new_mentions, positive_ids, "value"
        ),
        "mean_generated_state_tokens": {
            "baseline": float(
                np.mean([baseline_states[qid]["generated_tokens"] for qid in ids])
            ),
            "treatment": float(
                np.mean([treatment_states[qid]["generated_tokens"] for qid in ids])
            ),
        },
        "mean_state_generation_seconds": {
            "baseline": float(
                np.mean([baseline_states[qid]["generation_seconds"] for qid in ids])
            ),
            "treatment": float(
                np.mean([treatment_states[qid]["generation_seconds"] for qid in ids])
            ),
        },
        "retrieval_treatment_vs_baseline_dynamic": {
            "exact_block_any_at_12": paired_binary(
                old_dynamic_selection,
                new_dynamic_selection,
                positive_ids,
                "exact_block_any_at_12",
            ),
            "all_evidence_sessions_at_12": paired_binary(
                old_dynamic_selection,
                new_dynamic_selection,
                positive_ids,
                "all_evidence_sessions_at_12",
            ),
            "evidence_session_recall_at_12": paired_continuous(
                old_dynamic_selection,
                new_dynamic_selection,
                positive_ids,
                "evidence_session_recall_at_12",
                samples=args.bootstrap_samples,
                seed=args.seed,
            ),
        },
        "reader_quality": {
            "static_top12": reader_quality(static_reader, positive_ids),
            "compact_dynamic_top12": reader_quality(
                old_dynamic_reader, positive_ids
            ),
            "temporal_multivalue_dynamic_top12": reader_quality(
                new_dynamic_reader, positive_ids
            ),
        },
        "reader_temporal_vs_static": {
            "reference_nll": paired_continuous(
                static_reader,
                new_dynamic_reader,
                positive_ids,
                "reference_nll",
                samples=args.bootstrap_samples,
                seed=args.seed + 1,
            ),
            "token_f1": paired_continuous(
                static_reader,
                new_dynamic_reader,
                positive_ids,
                "token_f1",
                samples=args.bootstrap_samples,
                seed=args.seed + 2,
                lower_is_better=False,
            ),
        },
        "reader_temporal_vs_compact_dynamic": {
            "reference_nll": paired_continuous(
                old_dynamic_reader,
                new_dynamic_reader,
                positive_ids,
                "reference_nll",
                samples=args.bootstrap_samples,
                seed=args.seed + 3,
            ),
            "token_f1": paired_continuous(
                old_dynamic_reader,
                new_dynamic_reader,
                positive_ids,
                "token_f1",
                samples=args.bootstrap_samples,
                seed=args.seed + 4,
                lower_is_better=False,
            ),
        },
        "by_original_state_reference_posthoc": [],
    }
    for mentions_reference in (False, True):
        subset = [
            qid
            for qid in positive_ids
            if bool(baseline_states[qid]["state_mentions_reference_posthoc"])
            == mentions_reference
        ]
        output["by_original_state_reference_posthoc"].append(
            {
                "baseline_state_mentions_reference": mentions_reference,
                "queries": len(subset),
                "new_state_mentions_reference_rate": float(
                    np.mean(
                        [
                            treatment_states[qid]["state_mentions_reference_posthoc"]
                            for qid in subset
                        ]
                    )
                ),
                "new_minus_old_dynamic_nll": float(
                    np.mean(
                        [
                            new_dynamic_reader[qid]["reference_nll"]
                            - old_dynamic_reader[qid]["reference_nll"]
                            for qid in subset
                        ]
                    )
                ),
                "new_minus_static_nll": float(
                    np.mean(
                        [
                            new_dynamic_reader[qid]["reference_nll"]
                            - static_reader[qid]["reference_nll"]
                            for qid in subset
                        ]
                    )
                ),
            }
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
