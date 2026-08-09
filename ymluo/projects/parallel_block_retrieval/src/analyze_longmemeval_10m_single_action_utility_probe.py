from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from analyze_longmemeval_10m_pairwise_set_utility_probe import (
    oof_threshold_gate,
    read_jsonl,
    retrieval_change,
    score_diagnostics,
    selection_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze single-forward conditional set-utility probes."
    )
    parser.add_argument("--probe_pattern", required=True)
    parser.add_argument("--reference_probe_pattern", required=True)
    parser.add_argument("--reader_pattern", required=True)
    parser.add_argument("--selection_pattern", required=True)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []
    for partition in range(args.partitions):
        probe_dir = Path(args.probe_pattern.format(partition=partition))
        reference_dir = Path(args.reference_probe_pattern.format(partition=partition))
        reader_dir = Path(args.reader_pattern.format(partition=partition))
        selection_dir = Path(args.selection_pattern.format(partition=partition))
        probes = {
            str(row["question_id"]): row
            for row in read_jsonl(probe_dir / "rows.jsonl")
        }
        references = {
            str(row["question_id"]): row
            for row in read_jsonl(reference_dir / "rows.jsonl")
        }
        reader: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in read_jsonl(reader_dir / "rows.jsonl"):
            reader[str(row["method"])][str(row["question_id"])] = row
        selection: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in read_jsonl(selection_dir / "rows.jsonl"):
            if row["method"] in {"static_top12", "evidence_state_dynamic_top12"}:
                selection[str(row["method"])][str(row["question_id"])] = row
        for question_id, probe in probes.items():
            static_reader = reader["static_top12"][question_id]
            dynamic_reader = reader["evidence_state_dynamic_top12"][question_id]
            records.append(
                {
                    "question_id": question_id,
                    "partition": partition,
                    "question_type": probe["question_type"],
                    "is_abstention": bool(probe["is_abstention"]),
                    "probe": probe,
                    "reference_probe": references[question_id],
                    "reader_static": static_reader,
                    "reader_dynamic": dynamic_reader,
                    "selection_static": selection["static_top12"][question_id],
                    "selection_dynamic": selection[
                        "evidence_state_dynamic_top12"
                    ][question_id],
                    "nll_delta": float(dynamic_reader["reference_nll"])
                    - float(static_reader["reference_nll"]),
                }
            )
    records.sort(key=lambda row: (row["partition"], row["question_id"]))
    if len(records) != 500 or len({row["question_id"] for row in records}) != 500:
        raise RuntimeError("expected 500 unique questions")
    return records


def main() -> None:
    args = parse_args()
    records = load_records(args)
    groups = np.asarray([record["partition"] for record in records])
    score_keys = (
        "full_old_first_log_odds",
        "full_new_first_log_odds",
        "full_order_average_log_odds",
        "delta_old_first_log_odds",
        "delta_new_first_log_odds",
        "delta_order_average_log_odds",
    )
    scores = {
        key: np.asarray([float(record["probe"][key]) for record in records])
        for key in score_keys
    }
    reference_scores = np.asarray(
        [
            float(record["reference_probe"]["completeness_dynamic_utility_score"])
            for record in records
        ]
    )
    changed = np.asarray(
        [not bool(record["probe"]["sets_identical"]) for record in records]
    )
    output: dict[str, Any] = {
        "protocol": {
            "memory_scope": "eight independent real 10M-token shards",
            "selection_uses_answer_at_test": False,
            "probe_generates_answer": False,
            "single_deployment_forward": "full_old_first_log_odds",
            "outer_validation": "leave-one-10M-shard-out threshold",
            "delta_only_is_workset_context_ablation": True,
        },
        "queries": len(records),
        "changed_candidate_sets": int(changed.sum()),
        "latency_and_tokens_changed": {
            key: {
                "mean_seconds": float(
                    np.mean([record["probe"][f"{key}_seconds"] for record in records if not record["probe"]["sets_identical"]])
                ),
                "mean_prompt_tokens": float(
                    np.mean([record["probe"][f"{key}_prompt_tokens"] for record in records if not record["probe"]["sets_identical"]])
                ),
            }
            for key in ("full_old_first", "delta_old_first")
        },
        "order_sign_agreement_changed": {
            "full": float(
                np.mean(
                    [
                        record["probe"]["full_order_sign_agreement"]
                        for record in records
                        if not record["probe"]["sets_identical"]
                    ]
                )
            ),
            "delta_only": float(
                np.mean(
                    [
                        record["probe"]["delta_order_sign_agreement"]
                        for record in records
                        if not record["probe"]["sets_identical"]
                    ]
                )
            ),
        },
        "score_diagnostics": {
            key: score_diagnostics(records, value) for key, value in scores.items()
        },
        "comparison_to_two_forward_completeness": {
            "reference": score_diagnostics(records, reference_scores),
            **{
                key: {
                    "spearman": float(spearmanr(value, reference_scores).statistic),
                    "sign_agreement_changed": float(
                        np.mean((value[changed] > 0) == (reference_scores[changed] > 0))
                    ),
                }
                for key, value in scores.items()
            },
        },
        "gates": {},
    }
    for offset, (key, value) in enumerate(scores.items()):
        oof, thresholds = oof_threshold_gate(records, value, groups)
        output["gates"][key] = {
            "zero_threshold": selection_summary(
                records,
                value > 0,
                samples=args.bootstrap_samples,
                seed=args.seed + 100 * offset,
            ),
            "oof_threshold": {
                "fold_thresholds": thresholds,
                **selection_summary(
                    records,
                    oof,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 100 * offset + 1,
                ),
            },
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows_path = output_path.with_suffix(".rows.jsonl")
    with rows_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            handle.write(
                json.dumps(
                    {
                        "question_id": record["question_id"],
                        "partition": record["partition"],
                        "question_type": record["question_type"],
                        "nll_delta": record["nll_delta"],
                        "retrieval_change_posthoc": retrieval_change(record),
                        "reference_completeness_score": float(reference_scores[index]),
                        **{key: float(value[index]) for key, value in scores.items()},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
