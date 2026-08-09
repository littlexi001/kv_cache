#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


NUMERIC_KEYS = (
    "raw_prefix_tokens",
    "raw_prompt_tokens",
    "page_count",
    "ours_score_max",
    "ours_score_mean",
    "ours_score_gap2",
    "ours_score_gap3",
    "ours_score_entropy",
    "ours_score_positive_fraction",
)


def read_method(paths: list[Path], method: str) -> dict[str, dict[str, str]]:
    table: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("method") == method:
                    table[row["sample_id"]] = row
    return table


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except ValueError:
        return 0.0


def prediction(row: dict[str, str]) -> str:
    return row.get("longbench_v2_pred", "").strip().upper()


def query_text(raw: dict[str, Any]) -> str:
    question = str(raw.get("question", "")).strip()
    query = (
        f"What is the correct answer to this question: {question}\n"
        "Choices:\n"
        f"(A) {str(raw.get('choice_A', '')).strip()}\n"
        f"(B) {str(raw.get('choice_B', '')).strip()}\n"
        f"(C) {str(raw.get('choice_C', '')).strip()}\n"
        f"(D) {str(raw.get('choice_D', '')).strip()}"
    )
    return "\n".join(
        [
            str(raw.get("domain", "")),
            str(raw.get("sub_domain", "")),
            query,
        ]
    )


def numeric_features(row: dict[str, str], raw: dict[str, Any]) -> list[float]:
    question = str(raw.get("question", ""))
    choices = [str(raw.get(f"choice_{label}", "")) for label in "ABCD"]
    return [
        *[number(row, key) for key in NUMERIC_KEYS],
        float(len(question)),
        float(sum(len(choice) for choice in choices)),
        float(max((len(choice) for choice in choices), default=0)),
        float(len(str(raw.get("context", "")))),
    ]


def fit_feature_parts(records: list[dict[str, Any]], indices: np.ndarray) -> tuple[Any, Any, Any]:
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=6000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w\-]+\b",
    )
    categorical = DictVectorizer(sparse=True)
    scaler = StandardScaler()
    word.fit([records[idx]["text"] for idx in indices])
    categorical.fit([records[idx]["categorical"] for idx in indices])
    scaler.fit(np.asarray([records[idx]["numeric"] for idx in indices], dtype=np.float64))
    return word, categorical, scaler


def transform(records: list[dict[str, Any]], indices: np.ndarray, parts: tuple[Any, Any, Any]) -> sparse.csr_matrix:
    word, categorical, scaler = parts
    text_matrix = word.transform([records[idx]["text"] for idx in indices])
    category_matrix = categorical.transform([records[idx]["categorical"] for idx in indices])
    numeric = scaler.transform(np.asarray([records[idx]["numeric"] for idx in indices], dtype=np.float64))
    return sparse.hstack([text_matrix, category_matrix, sparse.csr_matrix(numeric)], format="csr")


def evaluate_threshold(records: list[dict[str, Any]], probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    choose_b2048 = probabilities >= threshold
    selected = [record["b2048"] if use_b2048 else record["base"] for record, use_b2048 in zip(records, choose_b2048)]
    count = max(1, len(records))
    labels = np.asarray([record["budget_changes_prediction"] for record in records], dtype=np.int64)
    missed_changes = int(((labels == 1) & ~choose_b2048).sum())
    selected_score = sum(number(row, "score") for row in selected) / count
    selected_kv = sum(number(row, "keep_fraction") for row in selected) / count
    selected_online = sum(number(row, "online_seconds") for row in selected)
    full_online = sum(number(record["full"], "online_seconds") for record in records)
    return {
        "threshold": threshold,
        "b2048_rate": float(choose_b2048.mean()),
        "change_recall": float(((labels == 1) & choose_b2048).sum() / max(1, int((labels == 1).sum()))),
        "missed_changes": missed_changes,
        "score": selected_score,
        "score_over_full": selected_score
        / max(1e-12, sum(number(record["full"], "score") for record in records) / count),
        "score_over_b2048": selected_score
        / max(1e-12, sum(number(record["b2048"], "score") for record in records) / count),
        "prediction_agreement_with_b2048": sum(
            prediction(row) == prediction(record["b2048"]) for row, record in zip(selected, records)
        )
        / count,
        "mean_kv_ratio": selected_kv,
        "online_speed_vs_full": full_online / max(1e-12, selected_online),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--base", nargs="+", required=True, type=Path)
    parser.add_argument("--b2048", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    raw_rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    raw_index = {str(row.get("_id", "")): row for row in raw_rows if isinstance(row, dict)}
    full = read_method(args.base, "full_kv")
    base = read_method(args.base, "ours_page_gather")
    b2048 = read_method(args.b2048, "ours_page_gather")
    sample_ids = sorted(full.keys() & base.keys() & b2048.keys() & raw_index.keys())
    records: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        raw = raw_index[sample_id]
        base_row = base[sample_id]
        records.append(
            {
                "sample_id": sample_id,
                "text": query_text(raw),
                "categorical": {
                    f"domain={raw.get('domain', '')}": 1.0,
                    f"sub_domain={raw.get('sub_domain', '')}": 1.0,
                    f"operator={base_row.get('ours_operator_mode', '')}": 1.0,
                },
                "numeric": numeric_features(base_row, raw),
                "budget_changes_prediction": int(prediction(base_row) != prediction(b2048[sample_id])),
                "full": full[sample_id],
                "base": base_row,
                "b2048": b2048[sample_id],
            }
        )
    labels = np.asarray([record["budget_changes_prediction"] for record in records], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        raise RuntimeError("Budget-change labels contain only one class")

    probabilities = np.zeros(len(records), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    dummy = np.zeros(len(records), dtype=np.float64)
    for train_indices, test_indices in splitter.split(dummy, labels):
        parts = fit_feature_parts(records, train_indices)
        x_train = transform(records, train_indices, parts)
        x_test = transform(records, test_indices, parts)
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=args.seed,
        )
        model.fit(x_train, labels[train_indices])
        class_index = list(model.classes_).index(1)
        probabilities[test_indices] = model.predict_proba(x_test)[:, class_index]

    thresholds = [round(value, 2) for value in np.linspace(0.05, 0.95, 19)]
    frontier = [evaluate_threshold(records, probabilities, threshold) for threshold in thresholds]
    full_score = sum(number(record["full"], "score") for record in records) / max(1, len(records))
    b2048_score = sum(number(record["b2048"], "score") for record in records) / max(1, len(records))
    feasible = [row for row in frontier if row["score"] + 1e-12 >= 0.95 * b2048_score]
    selected = min(feasible, key=lambda row: (row["mean_kv_ratio"], -row["score"])) if feasible else max(
        frontier, key=lambda row: (row["score"], -row["mean_kv_ratio"])
    )
    payload = {
        "protocol": "stratified out-of-fold diagnostic on design_dev only",
        "samples": len(records),
        "changed_predictions": int(labels.sum()),
        "changed_prediction_rate": float(labels.mean()),
        "full_score": full_score,
        "base_score": sum(number(record["base"], "score") for record in records) / max(1, len(records)),
        "b2048_score": b2048_score,
        "base_mean_kv_ratio": sum(number(record["base"], "keep_fraction") for record in records)
        / max(1, len(records)),
        "b2048_mean_kv_ratio": sum(number(record["b2048"], "keep_fraction") for record in records)
        / max(1, len(records)),
        "selected_operating_point": selected,
        "frontier": frontier,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.output_dir / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_id", "budget_changes_prediction", "change_probability", "domain", "sub_domain"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, probability in zip(records, probabilities):
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "budget_changes_prediction": record["budget_changes_prediction"],
                    "change_probability": probability,
                    "domain": record["full"].get("domain", ""),
                    "sub_domain": record["full"].get("sub_domain", ""),
                }
            )
    with (args.output_dir / "frontier.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frontier[0]))
        writer.writeheader()
        writer.writerows(frontier)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
