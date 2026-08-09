from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a leakage-free 3/16/512 block risk gate on retrieval confidence features."
    )
    parser.add_argument("--rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--target_recalls", default="0.75,0.80,0.85,0.90")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def feature_vector(row: dict[str, Any]) -> list[float]:
    scores = [float(item) for item in row["lexical_top_scores"]]
    scores.extend([0.0] * (16 - len(scores)))
    top = max(abs(scores[0]), 1.0e-6)
    normalized_scores = [item / top for item in scores[:8]]
    return [
        float(row["step_index"]),
        float(row["query_term_count"]),
        math.log1p(float(row["anchor_candidate_count"])),
        float(row["anchor_lexical_top3_overlap"]),
        float(row["anchor_lexical_top16_overlap"]),
        float(row["lexical_top1_gap"]),
        scores[0],
        scores[0] - scores[2],
        scores[2] - scores[3],
        scores[0] - scores[15],
        *normalized_scores,
    ]


def fit_gate(rows: Sequence[dict[str, Any]], budget: int, model_kind: str) -> Any:
    train = [row for row in rows if str(row["split"]) == "train"]
    features = np.asarray([feature_vector(row) for row in train], dtype=np.float64)
    labels = np.asarray(
        [int(int(row["lexical_rank"]) == 0 or int(row["lexical_rank"]) > budget) for row in train],
        dtype=np.int64,
    )
    if model_kind == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=0,
            ),
        )
    elif model_kind == "histgb":
        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=0,
        )
    elif model_kind == "forest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=0,
        )
    else:
        raise ValueError(f"unknown model kind: {model_kind}")
    model.fit(features, labels)
    return model


def probabilities(model: Any, rows: Sequence[dict[str, Any]]) -> np.ndarray:
    features = np.asarray([feature_vector(row) for row in rows], dtype=np.float64)
    return model.predict_proba(features)[:, 1]


def policy_metrics(
    rows: Sequence[dict[str, Any]],
    miss3: np.ndarray,
    miss16: np.ndarray,
    threshold3: float,
    threshold16: float,
) -> dict[str, float]:
    budgets = np.where(miss16 >= threshold16, 512, np.where(miss3 >= threshold3, 16, 3))
    ranks = np.asarray([int(row["lexical_rank"]) for row in rows], dtype=np.int64)
    hits = (ranks > 0) & (ranks <= budgets)
    return {
        "recall": float(hits.mean()),
        "mean_blocks": float(budgets.mean()),
        "p50_blocks": float(np.quantile(budgets, 0.5)),
        "p95_blocks": float(np.quantile(budgets, 0.95)),
        "fraction_budget3": float((budgets == 3).mean()),
        "fraction_budget16": float((budgets == 16).mean()),
        "fraction_budget512": float((budgets == 512).mean()),
    }


def fixed_metrics(rows: Sequence[dict[str, Any]], budget: int) -> dict[str, float]:
    ranks = np.asarray([int(row["lexical_rank"]) for row in rows], dtype=np.int64)
    return {
        "budget": budget,
        "recall": float(((ranks > 0) & (ranks <= budget)).mean()),
        "mean_blocks": float(budget),
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.rows_path))
    dev = [row for row in rows if str(row["split"]) == "dev"]
    test = [row for row in rows if str(row["split"]) == "test"]
    model_kinds = ("logistic", "histgb", "forest")
    predictions = {}
    for model_kind in model_kinds:
        model3 = fit_gate(rows, 3, model_kind)
        model16 = fit_gate(rows, 16, model_kind)
        predictions[model_kind] = {
            "dev_p3": probabilities(model3, dev),
            "dev_p16": probabilities(model16, dev),
            "test_p3": probabilities(model3, test),
            "test_p16": probabilities(model16, test),
        }
    thresholds = np.linspace(0.0, 1.0, 51)
    candidates = []
    for model_kind in model_kinds:
        dev_p3 = predictions[model_kind]["dev_p3"]
        dev_p16 = predictions[model_kind]["dev_p16"]
        for threshold3 in thresholds:
            for threshold16 in thresholds:
                metrics = policy_metrics(
                    dev, dev_p3, dev_p16, float(threshold3), float(threshold16)
                )
                candidates.append(
                    {
                        "model": model_kind,
                        "threshold3": float(threshold3),
                        "threshold16": float(threshold16),
                        **metrics,
                    }
                )
    selected = []
    for target in [float(item) for item in args.target_recalls.split(",") if item.strip()]:
        feasible = [item for item in candidates if item["recall"] >= target]
        if not feasible:
            continue
        winner = min(
            feasible,
            key=lambda item: (
                item["mean_blocks"],
                -item["recall"],
                item["model"],
                item["threshold3"],
                item["threshold16"],
            ),
        )
        test_p3 = predictions[winner["model"]]["test_p3"]
        test_p16 = predictions[winner["model"]]["test_p16"]
        test_metrics = policy_metrics(
            test,
            test_p3,
            test_p16,
            winner["threshold3"],
            winner["threshold16"],
        )
        by_step_type = {}
        for step_type in sorted({str(row["step_type"]) for row in test}):
            indices = [
                index for index, row in enumerate(test) if str(row["step_type"]) == step_type
            ]
            group = [test[index] for index in indices]
            by_step_type[step_type] = policy_metrics(
                group,
                test_p3[indices],
                test_p16[indices],
                winner["threshold3"],
                winner["threshold16"],
            )
        selected.append(
            {
                "dev_target_recall": target,
                "model": winner["model"],
                "threshold3": winner["threshold3"],
                "threshold16": winner["threshold16"],
                "dev": {
                    key: value
                    for key, value in winner.items()
                    if not key.startswith("threshold") and key != "model"
                },
                "test": test_metrics,
                "test_by_step_type": by_step_type,
            }
        )
    payload = {
        "source": args.rows_path,
        "selection_uses_gold": False,
        "training_labels_use_gold": True,
        "threshold_selection_split": "dev",
        "evaluation_split": "test",
        "feature_names": [
            "step_index",
            "query_term_count",
            "log_anchor_candidate_count",
            "anchor_lexical_top3_overlap",
            "anchor_lexical_top16_overlap",
            "lexical_top1_gap",
            "lexical_top1",
            "lexical_top1_minus_top3",
            "lexical_top3_minus_top4",
            "lexical_top1_minus_top16",
            *[f"normalized_lexical_top{index}" for index in range(1, 9)],
        ],
        "candidate_models": list(model_kinds),
        "fixed_dev": [fixed_metrics(dev, budget) for budget in (3, 16, 512)],
        "fixed_test": [fixed_metrics(test, budget) for budget in (3, 16, 512)],
        "selected_policies": selected,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
