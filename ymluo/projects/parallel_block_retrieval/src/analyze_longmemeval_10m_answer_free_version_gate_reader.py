from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_longmemeval_10m_temporal_reader_ablation import (
    compare,
    quality,
    read_jsonl,
)


METHODS = ("static_top12", "evidence_state_dynamic_top12")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gate date-labelled old-to-new reader organization using answer-free, "
            "leave-one-10M-shard-out version probabilities."
        )
    )
    parser.add_argument("--router_rows", required=True)
    parser.add_argument("--baseline_pattern", required=True)
    parser.add_argument("--treatment_pattern", required=True)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def selected_rows(
    baseline: dict[str, dict[str, Any]],
    treatment: dict[str, dict[str, Any]],
    ids: list[str],
    use_treatment: dict[str, bool],
) -> dict[str, dict[str, Any]]:
    return {
        question_id: (
            treatment[question_id]
            if use_treatment[question_id]
            else baseline[question_id]
        )
        for question_id in ids
    }


def gate_summary(
    baseline: dict[str, dict[str, Any]],
    treatment: dict[str, dict[str, Any]],
    ids: list[str],
    use_treatment: dict[str, bool],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    selected = selected_rows(baseline, treatment, ids, use_treatment)
    output = {
        "treatment_rate": float(
            np.mean([use_treatment[question_id] for question_id in ids])
        ),
        "quality": quality(selected, ids),
        "versus_baseline": compare(
            baseline,
            selected,
            ids,
            samples=samples,
            seed=seed,
        ),
    }
    answerable = [qid for qid in ids if not baseline[qid]["is_abstention"]]
    abstention = [qid for qid in ids if baseline[qid]["is_abstention"]]
    output["answerable"] = {
        "queries": len(answerable),
        "quality": quality(selected, answerable),
        "versus_baseline": compare(
            baseline,
            selected,
            answerable,
            samples=samples,
            seed=seed + 1,
        ),
    }
    if abstention:
        base_refusal = np.asarray(
            [baseline[qid]["predicted_refusal"] for qid in abstention], dtype=bool
        )
        selected_refusal = np.asarray(
            [selected[qid]["predicted_refusal"] for qid in abstention], dtype=bool
        )
        output["abstention"] = {
            "queries": len(abstention),
            "baseline_refusal_accuracy": float(base_refusal.mean()),
            "selected_refusal_accuracy": float(selected_refusal.mean()),
            "delta": float(selected_refusal.mean() - base_refusal.mean()),
        }
    output["by_question_type"] = {
        question_type: {
            "queries": len(type_ids),
            "treatment_rate": float(
                np.mean([use_treatment[question_id] for question_id in type_ids])
            ),
            "reference_nll_delta": float(
                np.mean(
                    [
                        selected[question_id]["reference_nll"]
                        - baseline[question_id]["reference_nll"]
                        for question_id in type_ids
                    ]
                )
            ),
            "token_f1_delta": float(
                np.mean(
                    [
                        selected[question_id]["token_f1"]
                        - baseline[question_id]["token_f1"]
                        for question_id in type_ids
                    ]
                )
            ),
        }
        for question_type in sorted(
            {baseline[question_id]["question_type"] for question_id in ids}
        )
        for type_ids in [
            [
                question_id
                for question_id in ids
                if baseline[question_id]["question_type"] == question_type
            ]
        ]
    }
    return output


def main() -> None:
    args = parse_args()
    router = {
        str(row["question_id"]): row for row in read_jsonl(Path(args.router_rows))
    }
    configs: dict[str, dict[str, dict[str, Any]]] = {
        "baseline": defaultdict(dict),
        "treatment": defaultdict(dict),
    }
    for partition in range(args.partitions):
        for config_name, pattern in (
            ("baseline", args.baseline_pattern),
            ("treatment", args.treatment_pattern),
        ):
            reader_dir = Path(pattern.format(partition=partition))
            for row in read_jsonl(reader_dir / "rows.jsonl"):
                if row["method"] in METHODS:
                    configs[config_name][str(row["method"])][str(row["question_id"])] = row
    ids = sorted(configs["baseline"][METHODS[0]])
    if len(ids) != 500 or set(router) != set(ids):
        raise RuntimeError("router and reader rows must cover the same 500 questions")
    for config in configs.values():
        for method in METHODS:
            if set(config[method]) != set(ids):
                raise RuntimeError("all reader configurations must cover the same questions")

    gate_definitions = {
        "always_old_to_new": {question_id: True for question_id in ids},
        "version_term_heuristic": {
            question_id: bool(router[question_id]["heuristic_version_term"])
            for question_id in ids
        },
        "question_text_oof_p05": {
            question_id: float(router[question_id]["question_text_probability"]) >= 0.5
            for question_id in ids
        },
        "question_plus_page_dates_oof_p05": {
            question_id: float(
                router[question_id]["question_text_plus_page_dates_probability"]
            )
            >= 0.5
            for question_id in ids
        },
        "question_state_page_dates_oof_p05": {
            question_id: float(
                router[question_id]["question_state_plus_page_dates_probability"]
            )
            >= 0.5
            for question_id in ids
        },
        "posthoc_question_type_upper_bound": {
            question_id: router[question_id]["question_type"] == "knowledge-update"
            for question_id in ids
        },
    }
    output = {
        "protocol": {
            "memory_scope": "eight independent real 10M-token shards",
            "same_selected_page_ids": True,
            "treatment": "display page dates and order old-to-new",
            "router_uses_answer_at_test": False,
            "learned_router_outer_validation": "leave-one-10M-shard-out",
            "posthoc_type_gate_is_not_deployable": True,
        },
        "queries": len(ids),
        "methods": {},
    }
    for method_index, method in enumerate(METHODS):
        baseline = configs["baseline"][method]
        treatment = configs["treatment"][method]
        output["methods"][method] = {
            "baseline_quality": quality(baseline, ids),
            "treatment_quality": quality(treatment, ids),
            "gates": {
                gate_name: gate_summary(
                    baseline,
                    treatment,
                    ids,
                    use_treatment,
                    samples=args.bootstrap_samples,
                    seed=args.seed + method_index * 1_000 + gate_index * 20,
                )
                for gate_index, (gate_name, use_treatment) in enumerate(
                    gate_definitions.items()
                )
            },
        }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
