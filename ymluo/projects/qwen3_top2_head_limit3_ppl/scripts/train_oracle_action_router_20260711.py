#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "raw_prefix_tokens",
    "raw_prompt_tokens",
    "page_count",
    "context_length_field",
    "budget_tokens",
    "sink_tokens",
    "recent_tokens",
    "page_tokens",
    "keep_fraction",
    "kept_context_tokens",
    "ours_score_max",
    "ours_score_mean",
    "ours_score_gap2",
    "ours_score_gap3",
    "ours_score_entropy",
    "ours_score_positive_fraction",
    "ours_query_coverage_terms",
    "ours_query_coverage_covered",
    "ours_query_coverage_recall",
    "ours_score_risk_linear_value",
    "ours_score_risk_min_gap2",
    "ours_score_risk_min_gap3",
    "ours_score_risk_max_entropy",
    "ours_score_risk_raw_prefix_at_most",
    "ours_coverage_certificate_terms",
    "ours_coverage_certificate_covered",
    "ours_coverage_certificate_recall",
    "ours_coverage_certificate_tokens",
    "ours_coverage_risk_initial_terms",
    "ours_coverage_risk_initial_recall",
    "ours_coarse_to_fine_candidate_pages",
    "ours_coarse_to_fine_candidate_tokens",
    "ours_graph_bridge_pairs",
    "ours_graph_bridge_tokens",
    "ours_bridge_active",
    "ours_graph_bridge_active",
    "ours_coarse_to_fine_active",
    "ours_anchor_window_active",
    "ours_label_support_active",
    "ours_passage_closure_active",
    "ours_structured_fingerprint_active",
    "ours_layer_router_active",
    "ours_output_verifier_active",
    "ours_grounding_verifier_active",
    "ours_consistency_verifier_active",
    "ours_score_risk_active",
    "ours_score_risk_triggered",
    "ours_coverage_risk_active",
    "ours_coverage_risk_triggered",
]


CATEGORICAL_FEATURES = [
    "task",
    "metric",
    "ours_scorer",
    "ours_layer_router_mode",
    "ours_action_router_selected_action",
]


DROP_FEATURE_PATTERN = re.compile(r"^(?:score|prediction|answers|generated|prefill|decode|query|online|total)")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("benchmark", ""), row.get("task", ""), row.get("sample_id", ""))


def to_float(value: Any) -> float:
    try:
        text = str(value)
        if text == "" or text.lower() in {"nan", "none"}:
            return np.nan
        return float(text)
    except Exception:
        return np.nan


def normalize_action(label: str) -> str:
    if not label or label == "full_kv_required":
        return "full_kv"
    text = label.lower()
    if "v315_b128_bm25bridge" in text:
        return "qasper_b128_bm25_1536"
    if "v318_qasper_b128_bm25bridge_1280" in text:
        return "qasper_b128_bm25_1280"
    if "v319_qasper_b128_bm25bridge_1024" in text:
        return "qasper_b128_bm25_1024"
    if "v306_repobench_bounded_retry" in text:
        return "repobench_bounded_retry"
    if "v311_safe_speedpatch" in text:
        return "safe_speedpatch"
    if "v310_b16_microspan_speed" in text:
        return "b16_microspan_speed"
    if "v309_b16_microspan_quality" in text:
        return "b16_microspan_quality"
    if "v313_b16_windowvote_speed" in text:
        return "b16_windowvote_speed"
    if "v312_b16_windowvote_quality" in text:
        return "b16_windowvote_quality"
    if "v305_bounded3k" in text:
        return "bounded3k"
    if "v300_" in text:
        return "v300_main"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")[:64] or "unknown"


def build_dataset(labels_path: Path, reference_path: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    ref_rows = {sample_key(row): row for row in read_csv(reference_path)}
    label_rows = read_csv(labels_path)
    rows: list[dict[str, Any]] = []
    y_danger: list[str] = []
    y_action: list[str] = []
    for label in label_rows:
        key = sample_key(label)
        ref = ref_rows.get(key)
        if ref is None:
            continue
        row: dict[str, Any] = {}
        for feature in NUMERIC_FEATURES:
            row[feature] = to_float(ref.get(feature, ""))
        for feature in CATEGORICAL_FEATURES:
            row[feature] = str(ref.get(feature, "") or "")
        min_safe = normalize_action(label.get("min_safe_label", ""))
        row["task"] = key[1]
        row["benchmark"] = key[0]
        row["sample_id"] = key[2]
        row["full_score"] = to_float(label.get("full_score", ""))
        row["safe_threshold"] = to_float(label.get("safe_threshold", ""))
        rows.append(row)
        y_action.append(min_safe)
        y_danger.append("danger" if min_safe == "full_kv" else "safe_sparse")
    return rows, y_danger, y_action


def row_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # sklearn can consume a list of dicts through a ColumnTransformer with column names after DataFrame conversion.
    import pandas as pd

    return pd.DataFrame(rows)


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=3)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def train_eval_classifier(
    X: Any,
    y: list[str],
    output_dir: Path,
    name: str,
    classifier: Any,
) -> dict[str, Any]:
    if len(set(y)) < 2:
        raise ValueError(f"{name} needs at least two labels")
    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=20260711)
    reports: list[dict[str, Any]] = []
    best_model: Pipeline | None = None
    best_macro = -1.0
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        model = Pipeline(steps=[("preprocess", make_preprocessor()), ("clf", classifier)])
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]
        model.fit(X_train, y_train)
        pred = list(model.predict(X_test))
        acc = accuracy_score(y_test, pred)
        macro = f1_score(y_test, pred, average="macro")
        weighted = f1_score(y_test, pred, average="weighted")
        reports.append(
            {
                "fold": fold,
                "accuracy": acc,
                "macro_f1": macro,
                "weighted_f1": weighted,
                "support": len(y_test),
            }
        )
        if macro > best_macro:
            best_macro = macro
            best_model = model
    final_model = Pipeline(steps=[("preprocess", make_preprocessor()), ("clf", classifier)])
    final_model.fit(X, y)
    with (output_dir / f"{name}_model.pkl").open("wb") as handle:
        pickle.dump(final_model, handle)
    (output_dir / f"{name}_cv.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return {
        "name": name,
        "labels": dict(Counter(y)),
        "mean_accuracy": float(np.mean([item["accuracy"] for item in reports])),
        "mean_macro_f1": float(np.mean([item["macro_f1"] for item in reports])),
        "mean_weighted_f1": float(np.mean([item["weighted_f1"] for item in reports])),
        "folds": reports,
    }


def heuristic_report(rows: list[dict[str, Any]], y_danger: list[str]) -> dict[str, Any]:
    y_true = [item == "danger" for item in y_danger]
    pred = [bool(to_float(row.get("ours_score_risk_triggered", 0)) > 0.5) for row in rows]
    tp = sum(p and t for p, t in zip(pred, y_true))
    fp = sum(p and not t for p, t in zip(pred, y_true))
    fn = sum((not p) and t for p, t in zip(pred, y_true))
    tn = sum((not p) and (not t) for p, t in zip(pred, y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "heuristic": "ours_score_risk_triggered",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(1, len(y_true)),
    }


def task_summary(rows: list[dict[str, Any]], y_danger: list[str], y_action: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[str(row.get("task", ""))].append(idx)
    out: list[dict[str, Any]] = []
    for task, indices in sorted(grouped.items()):
        actions = [y_action[i] for i in indices]
        danger = [y_danger[i] for i in indices]
        out.append(
            {
                "task": task,
                "samples": len(indices),
                "danger_rate": sum(item == "danger" for item in danger) / max(1, len(indices)),
                "top_actions": json.dumps(Counter(actions).most_common(6), ensure_ascii=False),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--reference_results", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, y_danger, y_action = build_dataset(Path(args.labels_csv), Path(args.reference_results))
    X = row_matrix(rows)
    if not rows:
        raise SystemExit("No rows after joining labels and reference results.")

    write_csv(output_dir / "router_dataset_preview.csv", rows[:200])
    write_csv(output_dir / "task_label_summary.csv", task_summary(rows, y_danger, y_action))

    danger_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=20260711,
        n_jobs=-1,
    )
    action_model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=20260711,
        n_jobs=-1,
    )
    summary = {
        "rows": len(rows),
        "danger_label_counts": dict(Counter(y_danger)),
        "action_label_counts": dict(Counter(y_action)),
        "heuristic_score_risk": heuristic_report(rows, y_danger),
        "danger_classifier": train_eval_classifier(X, y_danger, output_dir, "danger", danger_model),
        "action_classifier": train_eval_classifier(X, y_action, output_dir, "action", action_model),
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
    }
    (output_dir / "router_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
