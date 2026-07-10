#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


LEAKAGE_COLUMNS = {
    "no_bridge_score",
    "bridge_score",
    "delta",
    "label_bridge",
    "label_or_tie_bridge",
    "selected_page_jaccard",
    "no_selected_pages",
    "bridge_selected_pages",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def build_task_policy(rows: list[dict[str, str]], min_delta: float) -> tuple[dict[str, str], list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    policy: dict[str, str] = {}
    table: list[dict[str, Any]] = []
    for task, items in sorted(by_task.items()):
        deltas = [as_float(row["delta"]) for row in items]
        no_scores = [as_float(row["no_bridge_score"]) for row in items]
        bridge_scores = [as_float(row["bridge_score"]) for row in items]
        action = "bridge" if mean(deltas) > min_delta else "no_bridge"
        policy[task] = action
        table.append(
            {
                "task": task,
                "n": len(items),
                "no_bridge_mean": f"{mean(no_scores):.6f}",
                "bridge_mean": f"{mean(bridge_scores):.6f}",
                "mean_delta": f"{mean(deltas):.6f}",
                "bridge_win_rate": f"{sum(delta > min_delta for delta in deltas) / len(deltas):.6f}",
                "policy_action": action,
            }
        )
    return policy, table


def score_policy(rows: list[dict[str, str]], actions: list[str]) -> float:
    total = 0.0
    for row, action in zip(rows, actions):
        total += as_float(row["bridge_score"] if action == "bridge" else row["no_bridge_score"])
    return total / max(1, len(rows))


def task_policy_actions(rows: list[dict[str, str]], policy: dict[str, str]) -> list[str]:
    return [policy.get(row["task"], "no_bridge") for row in rows]


def feature_rows(rows: list[dict[str, str]], extra_rows: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (row["task"], row["sample_id"])
        extra = extra_rows.get(key, {})
        features: dict[str, Any] = {
            "task": row.get("task", ""),
            "metric": row.get("metric", ""),
            "raw_prefix_tokens": as_float(row.get("raw_prefix_tokens")),
            "page_count": as_float(row.get("page_count")),
        }
        for name in [
            "ours_score_max",
            "ours_score_mean",
            "ours_score_gap2",
            "ours_score_gap3",
            "ours_score_entropy",
            "ours_score_positive_fraction",
        ]:
            if name in extra:
                features[name] = as_float(extra.get(name))
        out.append(features)
    return out


def load_extra_features(path: str | None) -> dict[tuple[str, str], dict[str, str]]:
    if not path:
        return {}
    rows = read_csv(Path(path))
    return {(row.get("task", ""), row.get("sample_id", "")): row for row in rows}


def train_sklearn_gate(
    labels: list[dict[str, str]],
    features: list[dict[str, Any]],
    output_dir: Path,
    min_delta: float,
) -> dict[str, Any]:
    try:
        import joblib  # type: ignore
        import numpy as np  # type: ignore
        from sklearn.feature_extraction import DictVectorizer  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.metrics import accuracy_score  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except Exception as exc:
        return {"available": False, "reason": repr(exc)}

    feature_names = sorted({key for row in features for key in row})
    y = np.array([int(as_float(row["delta"]) > min_delta) for row in labels], dtype=int)
    X = features
    if len(set(y.tolist())) < 2:
        return {"available": False, "reason": "only one class in labels"}

    def make_model() -> Pipeline:
        return Pipeline(
            steps=[
                ("vec", DictVectorizer(sparse=False)),
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )

    clf = make_model()
    clf.fit(X, y)
    train_pred = clf.predict(X)
    train_actions = ["bridge" if pred else "no_bridge" for pred in train_pred]
    train_score = score_policy(labels, train_actions)

    loo_pred: list[int] = []
    if len(labels) >= 3:
        for idx in range(len(labels)):
            train_x = [item for j, item in enumerate(X) if j != idx]
            train_y = np.array([item for j, item in enumerate(y) if j != idx], dtype=int)
            if len(set(train_y.tolist())) < 2:
                loo_pred.append(0)
                continue
            model = make_model()
            model.fit(train_x, train_y)
            loo_pred.append(int(model.predict([X[idx]])[0]))
    loo_actions = ["bridge" if pred else "no_bridge" for pred in loo_pred]
    loo_score = score_policy(labels, loo_actions) if loo_actions else 0.0

    joblib.dump(clf, output_dir / "bridge_gate_logreg.joblib")
    prediction_rows = []
    for row, feats, pred in zip(labels, features, train_pred):
        prediction_rows.append(
            {
                "task": row["task"],
                "sample_id": row["sample_id"],
                "label_bridge": int(as_float(row["delta"]) > min_delta),
                "pred_bridge": int(pred),
                "delta": row["delta"],
                **{f"feature_{key}": value for key, value in feats.items()},
            }
        )
    write_csv(output_dir / "bridge_gate_logreg_predictions.csv", prediction_rows)
    return {
        "available": True,
        "train_accuracy": float(accuracy_score(y, train_pred)),
        "train_policy_score": float(train_score),
        "loo_accuracy": float(accuracy_score(y, np.array(loo_pred, dtype=int))) if loo_pred else None,
        "loo_policy_score": float(loo_score) if loo_actions else None,
        "features": feature_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_results", default="")
    parser.add_argument("--min_delta", type=float, default=1e-9)
    args = parser.parse_args()

    rows = read_csv(Path(args.labels_csv))
    if not rows:
        raise SystemExit("No labels found")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_policy, task_table = build_task_policy(rows, args.min_delta)
    task_actions = task_policy_actions(rows, task_policy)
    no_score = mean(as_float(row["no_bridge_score"]) for row in rows)
    all_bridge_score = mean(as_float(row["bridge_score"]) for row in rows)
    task_score = score_policy(rows, task_actions)
    oracle_score = mean(max(as_float(row["no_bridge_score"]), as_float(row["bridge_score"])) for row in rows)

    extra = load_extra_features(args.feature_results or None)
    feats = feature_rows(rows, extra)
    learned = train_sklearn_gate(rows, feats, output_dir, args.min_delta)

    write_csv(output_dir / "bridge_gate_task_policy.csv", task_table)
    (output_dir / "bridge_gate_task_policy.json").write_text(json.dumps(task_policy, indent=2), encoding="utf-8")

    summary = {
        "examples": len(rows),
        "no_bridge_score": no_score,
        "all_bridge_score": all_bridge_score,
        "task_policy_score": task_score,
        "sample_oracle_score": oracle_score,
        "task_policy": task_policy,
        "task_policy_bridge_tasks": sorted(task for task, action in task_policy.items() if action == "bridge"),
        "task_policy_is_oracle": abs(task_score - oracle_score) <= 1e-9,
        "learned_gate": learned,
        "min_delta": args.min_delta,
        "feature_results": args.feature_results,
        "leakage_columns_excluded": sorted(LEAKAGE_COLUMNS),
    }
    (output_dir / "bridge_gate_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Bridge Gate Training Report",
        "",
        f"- examples: {len(rows)}",
        f"- no_bridge_score: {no_score:.6f}",
        f"- all_bridge_score: {all_bridge_score:.6f}",
        f"- task_policy_score: {task_score:.6f}",
        f"- sample_oracle_score: {oracle_score:.6f}",
        f"- task_policy_bridge_tasks: {', '.join(sorted(task for task, action in task_policy.items() if action == 'bridge')) or '(none)'}",
        f"- task_policy_is_oracle: {abs(task_score - oracle_score) <= 1e-9}",
        f"- learned_gate_available: {learned.get('available')}",
    ]
    if learned.get("available"):
        lines.extend(
            [
                f"- learned_train_accuracy: {learned.get('train_accuracy'):.6f}",
                f"- learned_train_policy_score: {learned.get('train_policy_score'):.6f}",
                f"- learned_loo_accuracy: {learned.get('loo_accuracy')}",
                f"- learned_loo_policy_score: {learned.get('loo_policy_score')}",
            ]
        )
    else:
        lines.append(f"- learned_gate_reason: {learned.get('reason')}")
    lines.extend(
        [
            "",
            "| Task | n | No bridge | Bridge | Delta | Win rate | Action |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in task_table:
        lines.append(
            "| {task} | {n} | {no_bridge_mean} | {bridge_mean} | {mean_delta} | {bridge_win_rate} | {policy_action} |".format(
                **row
            )
        )
    (output_dir / "bridge_gate_training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
