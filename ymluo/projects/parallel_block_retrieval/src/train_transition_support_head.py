from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_stepwise_set_utility import mcnemar_exact_p


FEATURE_NAMES = [
    "heuristic_score",
    "yes_no_margin",
    "answer_mean_logprob",
    "anchor_present",
    "repeats_anchor",
    "relation_supported",
    "output_grounded",
    "query_memory_overlap",
    "grounding_ratio",
    "novel_grounded_terms",
    "negative_retrieval_rank",
    "negative_generated_tokens",
]

FEATURE_SETS = {
    "support_only": [1, 2],
    "heuristic_structured": [0, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "full_transition": list(range(len(FEATURE_NAMES))),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a train-only pairwise state-transition support head."
    )
    parser.add_argument("--train_rows_path", required=True)
    parser.add_argument("--dev_rows_path", required=True)
    parser.add_argument("--test_rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def transition_features(row: dict[str, Any]) -> np.ndarray:
    traces = sorted(row["heuristic_trace"], key=lambda item: int(item["branch_index"]))
    values = []
    for branch_index, trace in enumerate(traces):
        values.append(
            [
                float(trace["score"]),
                float(row["yes_no_scores"][branch_index]),
                float(row["answer_logprob_scores"][branch_index]),
                float(bool(trace["anchor_present"])),
                float(bool(trace["repeats_anchor"])),
                float(bool(trace["relation_supported"])),
                float(bool(trace["output_grounded"])),
                float(trace["query_memory_overlap"]),
                float(trace["grounding_ratio"]),
                float(len(trace["novel_grounded_terms"])),
                -float(row["branch_retrieval_ranks"][branch_index]),
                -float(row["branch_generated_tokens"][branch_index]),
            ]
        )
    return np.asarray(values, dtype=np.float64)


def pairwise_examples(
    rows: Sequence[dict[str, Any]], feature_indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, int]:
    examples = []
    labels = []
    usable_queries = 0
    for row in rows:
        hits = [bool(item) for item in row["branch_target_hits"]]
        positives = [index for index, hit in enumerate(hits) if hit]
        negatives = [index for index, hit in enumerate(hits) if not hit]
        if not positives or not negatives:
            continue
        usable_queries += 1
        features = transition_features(row)[:, list(feature_indices)]
        for positive in positives:
            for negative in negatives:
                difference = features[positive] - features[negative]
                examples.extend([difference, -difference])
                labels.extend([1, 0])
    return np.asarray(examples), np.asarray(labels), usable_queries


def fit_model(
    rows: Sequence[dict[str, Any]], feature_indices: Sequence[int]
) -> tuple[Any, int, int]:
    train_x, train_y, usable = pairwise_examples(rows, feature_indices)
    if not len(train_y):
        raise ValueError("no positive-negative transition pairs are available")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=17),
    )
    model.fit(train_x, train_y)
    return model, usable, len(train_y)


def model_indices(
    rows: Sequence[dict[str, Any]], model: Any, feature_indices: Sequence[int]
) -> list[int]:
    return [
        int(
            np.argmax(
                model.decision_function(
                    transition_features(row)[:, list(feature_indices)]
                )
            )
        )
        for row in rows
    ]


def runtime_transition_scores(
    features: np.ndarray, parameters: dict[str, Any]
) -> np.ndarray:
    mean = np.asarray(parameters["feature_mean"], dtype=np.float64)
    scale = np.asarray(parameters["feature_scale"], dtype=np.float64)
    weight = np.asarray(parameters["linear_weight"], dtype=np.float64)
    intercept = float(parameters["linear_intercept"])
    return ((features - mean) / scale) @ weight + intercept


def evaluate_indices(
    rows: Sequence[dict[str, Any]], indices: Sequence[int]
) -> list[bool]:
    return [
        bool(row["branch_target_hits"][index])
        for row, index in zip(rows, indices)
    ]


def summarize_method(
    rows: Sequence[dict[str, Any]], indices: Sequence[int]
) -> dict[str, Any]:
    hits = evaluate_indices(rows, indices)
    heuristic = evaluate_indices(
        rows, [int(row["heuristic_index"]) for row in rows]
    )
    wins = sum(new and not old for new, old in zip(hits, heuristic))
    losses = sum(old and not new for new, old in zip(hits, heuristic))
    return {
        "accuracy": sum(hits) / len(hits),
        "wins_losses_vs_heuristic": [wins, losses],
        "mcnemar_p_vs_heuristic": mcnemar_exact_p(wins, losses),
    }


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(Path(args.train_rows_path))
    dev_rows = read_jsonl(Path(args.dev_rows_path))
    test_rows = read_jsonl(Path(args.test_rows_path))
    payload: dict[str, Any] = {
        "source": "train-only pairwise transition support head",
        "selection_uses_dev_or_test_labels": False,
        "train_queries": len(train_rows),
        "dev_queries": len(dev_rows),
        "test_queries": len(test_rows),
        "feature_names": FEATURE_NAMES,
        "heuristic": {
            "train_accuracy": sum(
                evaluate_indices(
                    train_rows,
                    [int(row["heuristic_index"]) for row in train_rows],
                )
            )
            / len(train_rows),
            "dev_accuracy": sum(
                evaluate_indices(
                    dev_rows, [int(row["heuristic_index"]) for row in dev_rows]
                )
            )
            / len(dev_rows),
            "test_accuracy": sum(
                evaluate_indices(
                    test_rows, [int(row["heuristic_index"]) for row in test_rows]
                )
            )
            / len(test_rows),
        },
        "oracle_any_branch": {
            "train": sum(row["any_branch_hit"] for row in train_rows) / len(train_rows),
            "dev": sum(row["any_branch_hit"] for row in dev_rows) / len(dev_rows),
            "test": sum(row["any_branch_hit"] for row in test_rows) / len(test_rows),
        },
        "methods": {},
    }
    output_rows = [
        {
            "query_id": int(row["query_id"]),
            "heuristic_index": int(row["heuristic_index"]),
        }
        for row in test_rows
    ]
    for method_name, feature_indices in FEATURE_SETS.items():
        model, usable, examples = fit_model(train_rows, feature_indices)
        train_indices = model_indices(train_rows, model, feature_indices)
        dev_indices = model_indices(dev_rows, model, feature_indices)
        test_indices = model_indices(test_rows, model, feature_indices)
        logistic = model.named_steps["logisticregression"]
        scaler = model.named_steps["standardscaler"]
        payload["methods"][method_name] = {
            "feature_indices": feature_indices,
            "features": [FEATURE_NAMES[index] for index in feature_indices],
            "train_usable_queries": usable,
            "pairwise_examples": examples,
            "standardized_coefficients": [
                float(item) for item in logistic.coef_[0]
            ],
            "runtime_parameters": {
                "feature_mean": [float(item) for item in scaler.mean_],
                "feature_scale": [float(item) for item in scaler.scale_],
                "linear_weight": [float(item) for item in logistic.coef_[0]],
                "linear_intercept": float(logistic.intercept_[0]),
            },
            "train": summarize_method(train_rows, train_indices),
            "dev": summarize_method(dev_rows, dev_indices),
            "test": summarize_method(test_rows, test_indices),
        }
        for row, selected_index in zip(output_rows, test_indices):
            row[f"{method_name}_index"] = selected_index

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows_path = output_path.with_name(f"{output_path.stem}_test_rows.jsonl")
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
