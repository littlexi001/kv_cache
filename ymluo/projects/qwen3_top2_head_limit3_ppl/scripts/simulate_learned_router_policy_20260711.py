#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from train_oracle_action_router_20260711 import (
    build_dataset,
    make_preprocessor,
    normalize_action,
    read_csv,
    row_matrix,
    sample_key,
    to_float,
    write_csv,
)


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def row_map(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    return {sample_key(row): row for row in read_csv(path)}


def load_candidate_action_rows(candidate_dirs: list[Path]) -> dict[tuple[str, str, str], dict[str, dict[str, str]]]:
    by_key: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for directory in candidate_dirs:
        action = normalize_action(directory.name)
        for row in read_csv(directory / "task_results.csv"):
            key = sample_key(row)
            if not key[1] or not key[2]:
                continue
            current = by_key[key].get(action)
            if current is None:
                by_key[key][action] = row
                continue
            # Prefer lower-KV implementations for the same normalized action unless quality differs.
            current_tuple = (to_float(current.get("score", "")), -to_float(current.get("keep_fraction", "")))
            candidate_tuple = (to_float(row.get("score", "")), -to_float(row.get("keep_fraction", "")))
            if candidate_tuple > current_tuple:
                by_key[key][action] = row
    return by_key


def full_like_row(full_row: dict[str, str] | None) -> dict[str, float]:
    return {
        "score": to_float(full_row.get("score", "")) if full_row else 0.0,
        "kv": 1.0,
        "online": to_float(full_row.get("online_seconds", "")) if full_row else 0.0,
        "total": to_float(full_row.get("total_seconds", "")) if full_row else 0.0,
    }


def metrics_from_rows(items: list[dict[str, float]]) -> dict[str, float]:
    return {
        "samples": len(items),
        "score": mean([item["score"] for item in items]),
        "kv_keep": mean([item["kv"] for item in items]),
        "online_seconds": mean([item["online"] for item in items]),
        "total_seconds": mean([item["total"] for item in items]),
    }


def as_metric(row: dict[str, str] | None, full: bool = False) -> dict[str, float]:
    if row is None:
        return {"score": 0.0, "kv": 1.0 if full else 0.0, "online": 0.0, "total": 0.0}
    return {
        "score": to_float(row.get("score", "")),
        "kv": 1.0 if full else to_float(row.get("keep_fraction", "")),
        "online": to_float(row.get("online_seconds", "")),
        "total": to_float(row.get("total_seconds", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--reference_results", required=True)
    parser.add_argument("--full_results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_dirs", nargs="*", default=[])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, y_danger, y_action = build_dataset(Path(args.labels_csv), Path(args.reference_results))
    X = row_matrix(rows)
    full_by_key = row_map(Path(args.full_results))
    v300_by_key = row_map(Path(args.reference_results))
    action_by_key = load_candidate_action_rows([Path(item) for item in args.candidate_dirs])
    keys = [(row["benchmark"], row["task"], row["sample_id"]) for row in rows]

    y_binary = np.array([1 if item == "danger" else 0 for item in y_danger])
    y_action_array = np.array(y_action)
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    threshold_items: dict[float, list[dict[str, float]]] = {threshold: [] for threshold in thresholds}
    action_items: list[dict[str, float]] = []
    baseline_full: list[dict[str, float]] = []
    baseline_v300: list[dict[str, float]] = []
    oracle_min_safe: list[dict[str, float]] = []
    fold_rows: list[dict[str, Any]] = []

    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=20260711)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y_binary), start=1):
        danger_model = Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=20260711 + fold,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        action_model = Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=20260811 + fold,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_binary[train_idx]
        y_test = y_binary[test_idx]
        action_train = y_action_array[train_idx]
        action_test = y_action_array[test_idx]
        danger_model.fit(X_train, y_train)
        action_model.fit(X_train, action_train)
        danger_prob = danger_model.predict_proba(X_test)[:, list(danger_model.classes_).index(1)]
        action_pred = list(action_model.predict(X_test))
        pred_05 = (danger_prob >= 0.5).astype(int)
        fold_rows.append(
            {
                "fold": fold,
                "danger_f1": f1_score(y_test, pred_05),
                "danger_positive_rate": float(np.mean(pred_05)),
                "action_accuracy": float(np.mean(action_pred == action_test)),
                "samples": len(test_idx),
            }
        )

        for local_i, idx in enumerate(test_idx):
            key = keys[idx]
            full_metric = as_metric(full_by_key.get(key), full=True)
            v300_metric = as_metric(v300_by_key.get(key), full=False)
            baseline_full.append(full_metric)
            baseline_v300.append(v300_metric)
            # Oracle label was generated with min-safe action under the configured KV constraint.
            label_action = y_action[idx]
            if label_action == "full_kv":
                oracle_min_safe.append(full_metric)
            else:
                oracle_row = action_by_key.get(key, {}).get(label_action) or v300_by_key.get(key)
                oracle_min_safe.append(as_metric(oracle_row, full=False))
            for threshold in thresholds:
                threshold_items[threshold].append(full_metric if danger_prob[local_i] >= threshold else v300_metric)
            predicted_action = action_pred[local_i]
            if predicted_action == "full_kv":
                action_items.append(full_metric)
            else:
                action_row = action_by_key.get(key, {}).get(predicted_action)
                if action_row is None:
                    action_row = v300_by_key.get(key)
                action_items.append(as_metric(action_row, full=False))

    summary_rows: list[dict[str, Any]] = []
    for name, items in [
        ("full_kv", baseline_full),
        ("v300_reference", baseline_v300),
        ("oracle_min_safe_from_labels", oracle_min_safe),
        ("action_classifier_policy", action_items),
    ]:
        row = {"policy": name}
        row.update(metrics_from_rows(items))
        summary_rows.append(row)
    for threshold, items in threshold_items.items():
        row = {"policy": f"danger_fallback_threshold_{threshold:.2f}"}
        row.update(metrics_from_rows(items))
        row["fallback_rate"] = mean([1.0 if item["kv"] >= 0.999 else 0.0 for item in items])
        summary_rows.append(row)

    write_csv(output_dir / "policy_summary.csv", summary_rows)
    write_csv(output_dir / "fold_metrics.csv", fold_rows)
    payload = {
        "fold_metrics": fold_rows,
        "policy_summary": summary_rows,
        "thresholds": thresholds,
        "note": "Cross-validated predictions. Metrics are averaged over test folds; repeated samples across folds are expected.",
    }
    (output_dir / "policy_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
