from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from analyze_longmemeval_10m_answer_free_version_router import (
    cue_features,
    fit_text_fold,
    page_features,
    read_jsonl,
)
from analyze_longmemeval_10m_utility_gate import paired_bootstrap


METHODS = ("static_top12", "evidence_state_dynamic_top12")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict reader utility of chronological page ordering from answer-free "
            "question/state text and retrieved-page date metadata."
        )
    )
    parser.add_argument("--data_pattern", required=True)
    parser.add_argument("--selection_pattern", required=True)
    parser.add_argument("--baseline_pattern", required=True)
    parser.add_argument("--order_pattern", required=True)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def selection_summary(
    records: list[dict[str, Any]],
    method: str,
    use_order: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_nll = np.asarray(
        [record["baseline"][method]["reference_nll"] for record in records]
    )
    order_nll = np.asarray(
        [record["order"][method]["reference_nll"] for record in records]
    )
    selected_nll = np.where(use_order, order_nll, baseline_nll)
    answerable = np.asarray([not record["is_abstention"] for record in records])
    baseline_f1 = np.asarray(
        [record["baseline"][method]["token_f1"] for record in records]
    )
    order_f1 = np.asarray([record["order"][method]["token_f1"] for record in records])
    selected_f1 = np.where(use_order, order_f1, baseline_f1)
    return {
        "order_selection_rate": float(use_order.mean()),
        "reference_nll_vs_baseline": paired_bootstrap(
            selected_nll,
            baseline_nll,
            samples=samples,
            seed=seed,
        ),
        "answerable_token_f1": {
            "baseline": float(baseline_f1[answerable].mean()),
            "selected": float(selected_f1[answerable].mean()),
            "delta": float(
                (selected_f1[answerable] - baseline_f1[answerable]).mean()
            ),
        },
        "by_question_type": {
            question_type: {
                "queries": len(indices),
                "selection_rate": float(use_order[indices].mean()),
                "selected_minus_baseline_nll": float(
                    (selected_nll - baseline_nll)[indices].mean()
                ),
            }
            for question_type in sorted({record["question_type"] for record in records})
            for indices in [
                np.asarray(
                    [
                        index
                        for index, record in enumerate(records)
                        if record["question_type"] == question_type
                    ]
                )
            ]
        },
    }


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for partition in range(args.partitions):
        data_dir = Path(args.data_pattern.format(partition=partition))
        selection_dir = Path(args.selection_pattern.format(partition=partition))
        queries = {int(row["query_id"]): row for row in read_jsonl(data_dir / "queries.jsonl")}
        states = {
            int(row["query_id"]): row for row in read_jsonl(selection_dir / "states.jsonl")
        }
        static_selection = {
            int(row["query_id"]): row
            for row in read_jsonl(selection_dir / "rows.jsonl")
            if row["method"] == "static_top12"
        }
        reader_configs: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
            "baseline": defaultdict(dict),
            "order": defaultdict(dict),
        }
        for config, pattern in (
            ("baseline", args.baseline_pattern),
            ("order", args.order_pattern),
        ):
            for row in read_jsonl(
                Path(pattern.format(partition=partition)) / "rows.jsonl"
            ):
                if row["method"] in METHODS:
                    reader_configs[config][str(row["method"])][
                        str(row["question_id"])
                    ] = row
        block_dates = np.load(data_dir / "base_block_date_minutes.npy", mmap_mode="r")
        block_sessions = np.load(data_dir / "base_block_session_rows.npy", mmap_mode="r")
        for query_id, query in queries.items():
            question_id = str(query["question_id"])
            question = str(query["question"])
            state = str(states[query_id]["state_text"])
            block_ids = list(map(int, static_selection[query_id]["top_block_ids"]))
            dense = (
                cue_features(question)
                + cue_features(state)
                + page_features(
                    block_ids,
                    block_dates,
                    block_sessions,
                    int(query["question_date_minutes"]),
                )
            )
            records.append(
                {
                    "partition": partition,
                    "question_id": question_id,
                    "question_type": str(query["question_type"]),
                    "is_abstention": bool(query["is_abstention"]),
                    "text": question + " [STATE] " + state,
                    "dense": dense,
                    "baseline": {
                        method: reader_configs["baseline"][method][question_id]
                        for method in METHODS
                    },
                    "order": {
                        method: reader_configs["order"][method][question_id]
                        for method in METHODS
                    },
                }
            )
    records.sort(key=lambda row: (row["partition"], row["question_id"]))
    if len(records) != 500:
        raise RuntimeError(f"expected 500 records, got {len(records)}")
    groups = np.asarray([record["partition"] for record in records])
    output: dict[str, Any] = {
        "protocol": {
            "memory_scope": "eight independent real 10M-token shards",
            "test_time_features": "question/state text and static selected-page dates",
            "answer_or_gold_used_at_test": False,
            "outer_validation": "leave-one-10M-shard-out",
            "treatment": "same selected pages ordered old-to-new without date labels",
        },
        "queries": len(records),
        "methods": {},
    }
    for method_index, method in enumerate(METHODS):
        delta = np.asarray(
            [
                record["order"][method]["reference_nll"]
                - record["baseline"][method]["reference_nll"]
                for record in records
            ],
            dtype=np.float64,
        )
        beneficial = delta < 0.0
        probabilities = np.zeros(len(records), dtype=np.float64)
        ridge_predictions = np.zeros(len(records), dtype=np.float64)
        for held_out in sorted(set(groups)):
            train = groups != held_out
            test = groups == held_out
            train_indices = np.flatnonzero(train)
            test_indices = np.flatnonzero(test)
            train_x, test_x, _ = fit_text_fold(
                [records[index]["text"] for index in train_indices],
                [records[index]["text"] for index in test_indices],
                beneficial[train],
                seed=args.seed + method_index,
            )
            scaler = StandardScaler()
            train_dense = scaler.fit_transform(
                np.asarray([records[index]["dense"] for index in train_indices])
            )
            test_dense = scaler.transform(
                np.asarray([records[index]["dense"] for index in test_indices])
            )
            train_x = hstack([train_x, csr_matrix(train_dense)], format="csr")
            test_x = hstack([test_x, csr_matrix(test_dense)], format="csr")
            classifier = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2_000,
                solver="liblinear",
                random_state=args.seed + method_index,
            )
            classifier.fit(train_x, beneficial[train])
            probabilities[test] = classifier.predict_proba(test_x)[:, 1]
            lower, upper = np.quantile(delta[train], [0.025, 0.975])
            regressor = Ridge(alpha=100.0, solver="lsqr")
            regressor.fit(train_x, np.clip(delta[train], lower, upper))
            ridge_predictions[test] = regressor.predict(test_x)

        auc = float(roc_auc_score(beneficial, probabilities))
        probability_gate = probabilities >= 0.5
        ridge_gate = ridge_predictions < 0.0
        output["methods"][method] = {
            "beneficial_queries": int(beneficial.sum()),
            "beneficial_rate": float(beneficial.mean()),
            "classifier": {
                "benefit_sign_auc": auc,
                "probability_vs_negative_nll_delta_spearman": float(
                    spearmanr(probabilities, -delta).statistic
                ),
                "gate": selection_summary(
                    records,
                    method,
                    probability_gate,
                    samples=args.bootstrap_samples,
                    seed=args.seed + method_index * 100,
                ),
            },
            "ridge": {
                "predicted_vs_nll_delta_spearman": float(
                    spearmanr(ridge_predictions, delta).statistic
                ),
                "gate": selection_summary(
                    records,
                    method,
                    ridge_gate,
                    samples=args.bootstrap_samples,
                    seed=args.seed + method_index * 100 + 10,
                ),
            },
            "always_order": selection_summary(
                records,
                method,
                np.ones(len(records), dtype=bool),
                samples=args.bootstrap_samples,
                seed=args.seed + method_index * 100 + 20,
            ),
        }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
