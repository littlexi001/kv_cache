#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SAFE_FEATURE_COLUMNS = [
    "task",
    "metric",
    "raw_prefix_tokens",
    "page_count",
    "ours_score_max",
    "ours_score_mean",
    "ours_score_gap2",
    "ours_score_gap3",
    "ours_score_entropy",
    "ours_score_positive_fraction",
]


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


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def action_names(rows: list[dict[str, str]]) -> list[str]:
    actions = []
    for key in rows[0]:
        if key.startswith("score_"):
            actions.append(key[len("score_") :])
    return sorted(actions)


def load_feature_map(path: str | None) -> dict[tuple[str, str], dict[str, str]]:
    if not path:
        return {}
    rows = read_csv(Path(path))
    return {(row.get("task", ""), row.get("sample_id", "")): row for row in rows}


def build_features(labels: list[dict[str, str]], feature_rows: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in labels:
        key = (row.get("task", ""), row.get("sample_id", ""))
        extra = feature_rows.get(key, {})
        feats: dict[str, Any] = {
            "task": row.get("task", ""),
            "metric": row.get("metric", ""),
        }
        for name in SAFE_FEATURE_COLUMNS:
            if name in {"task", "metric"}:
                continue
            source = extra if name in extra else row
            feats[name] = fnum(source.get(name))
        out.append(feats)
    return out


def score_policy(rows: list[dict[str, str]], actions: list[str]) -> float:
    return mean(fnum(row.get(f"score_{action}")) for row, action in zip(rows, actions))


def build_task_policy(rows: list[dict[str, str]], actions: list[str]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    policy = {}
    table = []
    for task, task_rows in sorted(by_task.items()):
        means = {action: mean(fnum(row.get(f"score_{action}")) for row in task_rows) for action in actions}
        keeps = {action: mean(fnum(row.get(f"keep_{action}"), 1.0) for row in task_rows) for action in actions}
        best = sorted(actions, key=lambda action: (-means[action], keeps[action], action))[0]
        policy[task] = best
        table.append(
            {
                "task": task,
                "n": len(task_rows),
                "policy_action": best,
                **{f"mean_score_{action}": f"{means[action]:.6f}" for action in actions},
                **{f"mean_keep_{action}": f"{keeps[action]:.6f}" for action in actions},
            }
        )
    return policy, table


def train_sklearn_router(
    labels: list[dict[str, str]],
    features: list[dict[str, Any]],
    actions: list[str],
    output_dir: Path,
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

    y = np.array([row["best_action"] for row in labels], dtype=object)
    if len(set(y.tolist())) < 2:
        return {"available": False, "reason": "only one class in labels"}

    def make_model() -> Pipeline:
        return Pipeline(
            steps=[
                ("vec", DictVectorizer(sparse=False)),
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )

    clf = make_model()
    clf.fit(features, y)
    pred = clf.predict(features)
    train_actions = [str(item) for item in pred.tolist()]
    train_score = score_policy(labels, train_actions)

    loo_pred: list[str] = []
    if len(labels) >= 3:
        for idx in range(len(labels)):
            train_x = [item for j, item in enumerate(features) if j != idx]
            train_y = np.array([item for j, item in enumerate(y) if j != idx], dtype=object)
            if len(set(train_y.tolist())) < 2:
                majority = Counter(train_y.tolist()).most_common(1)[0][0]
                loo_pred.append(str(majority))
                continue
            model = make_model()
            model.fit(train_x, train_y)
            loo_pred.append(str(model.predict([features[idx]])[0]))
    loo_score = score_policy(labels, loo_pred) if loo_pred else 0.0

    joblib.dump(clf, output_dir / "action_router_logreg.joblib")
    prediction_rows = []
    for row, feats, pred_action in zip(labels, features, train_actions):
        prediction_rows.append(
            {
                "task": row["task"],
                "sample_id": row["sample_id"],
                "label_action": row["best_action"],
                "pred_action": pred_action,
                **{f"feature_{key}": value for key, value in feats.items()},
            }
        )
    write_csv(output_dir / "action_router_predictions.csv", prediction_rows)

    return {
        "available": True,
        "train_accuracy": float(accuracy_score(y, pred)),
        "train_policy_score": float(train_score),
        "loo_accuracy": float(accuracy_score(y, np.array(loo_pred, dtype=object))) if loo_pred else None,
        "loo_policy_score": float(loo_score) if loo_pred else None,
        "features": SAFE_FEATURE_COLUMNS,
        "classes": actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_results", default="")
    args = parser.parse_args()

    labels = read_csv(Path(args.labels_csv))
    if not labels:
        raise SystemExit("No labels found")
    actions = action_names(labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_map = load_feature_map(args.feature_results or None)
    features = build_features(labels, feature_map)
    task_policy, task_table = build_task_policy(labels, actions)
    task_actions = [task_policy[row["task"]] for row in labels]
    sample_oracle_actions = [row["best_action"] for row in labels]
    action_scores = {action: mean(fnum(row.get(f"score_{action}")) for row in labels) for action in actions}
    learned = train_sklearn_router(labels, features, actions, output_dir)

    summary = {
        "examples": len(labels),
        "actions": actions,
        "action_scores": action_scores,
        "sample_oracle_score": score_policy(labels, sample_oracle_actions),
        "task_policy_score": score_policy(labels, task_actions),
        "task_policy": task_policy,
        "learned_router": learned,
        "feature_results": args.feature_results,
        "safe_feature_columns": SAFE_FEATURE_COLUMNS,
    }
    write_csv(output_dir / "task_action_policy.csv", task_table)
    (output_dir / "task_action_policy.json").write_text(json.dumps(task_policy, indent=2), encoding="utf-8")
    (output_dir / "action_router_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Action Router Training Report",
        "",
        f"- examples: {len(labels)}",
        f"- actions: {', '.join(actions)}",
        f"- sample_oracle_score: {summary['sample_oracle_score']:.6f}",
        f"- task_policy_score: {summary['task_policy_score']:.6f}",
        f"- learned_router_available: {learned.get('available')}",
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
        lines.append(f"- learned_router_reason: {learned.get('reason')}")
    lines.extend(
        [
            "",
            "| Action | Mean score |",
            "|---|---:|",
        ]
    )
    for action in actions:
        lines.append(f"| {action} | {action_scores[action]:.6f} |")
    lines.extend(["", "| Task | n | Policy |", "|---|---:|---|"])
    for row in task_table:
        lines.append(f"| {row['task']} | {row['n']} | {row['policy_action']} |")
    (output_dir / "action_router_training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
