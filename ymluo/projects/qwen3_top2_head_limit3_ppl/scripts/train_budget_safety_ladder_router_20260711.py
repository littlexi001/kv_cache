#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_NUMERIC_FEATURES = [
    "raw_prefix_tokens",
    "raw_prompt_tokens",
    "context_length_field",
    "page_count",
    "ours_score_max",
    "ours_score_mean",
    "ours_score_gap2",
    "ours_score_gap3",
    "ours_score_entropy",
    "ours_score_positive_fraction",
]

BUDGET_FEATURES = [
    "candidate_budget_tokens",
    "candidate_budget_log2",
    "candidate_budget_prompt_fraction",
    "candidate_budget_context_fraction",
    "candidate_budget_page_fraction",
]

FAMILY_BY_TASK = {
    "narrativeqa": "single_doc_qa",
    "qasper": "single_doc_qa",
    "multifieldqa_en": "single_doc_qa",
    "hotpotqa": "multi_doc_qa",
    "2wikimqa": "multi_doc_qa",
    "musique": "multi_doc_qa",
    "gov_report": "summarization",
    "qmsum": "summarization",
    "multi_news": "summarization",
    "trec": "fewshot",
    "triviaqa": "fewshot",
    "samsum": "fewshot",
    "passage_count": "synthetic",
    "passage_retrieval_en": "synthetic",
    "lcc": "code",
    "repobench-p": "code",
}


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


def fnum(row: dict[str, str] | dict[str, Any] | None, key: str) -> float:
    if row is None:
        return 0.0
    try:
        return float(row.get(key, "") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        task = row.get("task", "")
        sample_id = row.get("sample_id", "")
        if task and sample_id:
            out[(task, sample_id)] = row
    return out


def parse_candidate_arg(text: str) -> tuple[str, Path]:
    if "=" in text:
        name, path = text.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(text)
    return path.name, path


def parse_budget_from_action(action: str) -> int:
    marker = "budget_b"
    if marker not in action:
        raise ValueError(f"Cannot parse budget action: {action}")
    tail = action.split(marker, 1)[1]
    digits = []
    for char in tail:
        if char.isdigit():
            digits.append(char)
        else:
            break
    if not digits:
        raise ValueError(f"Cannot parse budget action: {action}")
    return int("".join(digits))


def fold_for_key(task: str, sample_id: str, folds: int) -> int:
    digest = hashlib.md5(f"{task}\t{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, folds)


def task_family(task: str) -> str:
    return FAMILY_BY_TASK.get(task, "other")


def quality_target(reference_score: float, ratio: float, margin: float) -> float:
    return max(0.0, ratio * reference_score, reference_score - margin)


def candidate_cost(row: dict[str, str]) -> tuple[float, float, float]:
    return (fnum(row, "keep_fraction"), fnum(row, "online_seconds"), -fnum(row, "score"))


def category_features(rows: list[dict[str, Any]], task_encoding: str) -> list[str]:
    categories: set[str] = set()
    if task_encoding in {"family", "both"}:
        for family in sorted(set(FAMILY_BY_TASK.values()) | {"other"}):
            categories.add(f"family={family}")
    if task_encoding in {"task", "both"}:
        for row in rows:
            categories.add(f"task={row['task']}")
    return sorted(categories)


def base_feature_row(reference_row: dict[str, str]) -> dict[str, Any]:
    task = reference_row.get("task", "")
    row: dict[str, Any] = {
        "task": task,
        "task_family": task_family(task),
    }
    for name in BASE_NUMERIC_FEATURES:
        row[name] = fnum(reference_row, name)
    return row


def add_budget_features(row: dict[str, Any], action: str, budget_tokens: int) -> dict[str, Any]:
    out = dict(row)
    raw_prompt_tokens = max(1.0, float(out.get("raw_prompt_tokens", 0.0) or 0.0))
    raw_prefix_tokens = max(1.0, float(out.get("raw_prefix_tokens", 0.0) or 0.0))
    context_length = max(1.0, float(out.get("context_length_field", 0.0) or 0.0))
    page_count = max(1.0, float(out.get("page_count", 0.0) or 0.0))
    out["candidate_action"] = action
    out["candidate_budget_tokens"] = float(budget_tokens)
    out["candidate_budget_log2"] = math.log2(max(1.0, float(budget_tokens)))
    out["candidate_budget_prompt_fraction"] = float(budget_tokens) / raw_prompt_tokens
    out["candidate_budget_context_fraction"] = float(budget_tokens) / max(raw_prefix_tokens, context_length)
    out["candidate_budget_page_fraction"] = float(budget_tokens) / page_count
    return out


def make_matrix(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    category_names: list[str],
    task_encoding: str,
) -> list[list[float]]:
    category_index = {name: idx for idx, name in enumerate(category_names)}
    matrix: list[list[float]] = []
    for row in rows:
        vector = [float(row.get(name, 0.0) or 0.0) for name in feature_names]
        one_hot = [0.0] * len(category_names)
        task = str(row["task"])
        if task_encoding in {"family", "both"}:
            name = f"family={task_family(task)}"
            if name in category_index:
                one_hot[category_index[name]] = 1.0
        if task_encoding in {"task", "both"}:
            name = f"task={task}"
            if name in category_index:
                one_hot[category_index[name]] = 1.0
        vector.extend(one_hot)
        matrix.append(vector)
    return matrix


def positive_probability(model: Any, matrix: list[list[float]]) -> list[float]:
    if not matrix:
        return []
    if isinstance(model, dict):
        return [float(model.get("constant_probability", 0.0)) for _ in matrix]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(matrix)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            index = classes.index(1)
            return [float(row[index]) for row in probabilities]
        return [0.0 for _ in probabilities]
    predictions = model.predict(matrix)
    return [float(value) for value in predictions]


def summarize_predictions(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped["ALL"].append(row)
        grouped[str(row["task"])].append(row)
    out: list[dict[str, Any]] = []
    for task, subset in sorted(grouped.items(), key=lambda item: (item[0] != "ALL", item[0])):
        ref_score = mean([float(row["reference_score"]) for row in subset])
        learned_score = mean([float(row["learned_score"]) for row in subset])
        oracle_score = mean([float(row["oracle_score"]) for row in subset])
        ref_kv = mean([float(row["reference_kv_keep"]) for row in subset])
        learned_kv = mean([float(row["learned_kv_keep"]) for row in subset])
        oracle_kv = mean([float(row["oracle_kv_keep"]) for row in subset])
        ref_online = mean([float(row["reference_online_seconds"]) for row in subset])
        learned_online = mean([float(row["learned_online_seconds"]) for row in subset])
        oracle_online = mean([float(row["oracle_online_seconds"]) for row in subset])
        safe = [
            row
            for row in subset
            if float(row["learned_score"]) + 1e-12 >= float(row["quality_target"])
        ]
        exact = [row for row in subset if row["learned_action"] == row["oracle_action"]]
        out.append(
            {
                "split": split_name,
                "task": task,
                "samples": len(subset),
                "reference_score": ref_score,
                "learned_score": learned_score,
                "oracle_score": oracle_score,
                "learned_vs_reference": learned_score / ref_score if ref_score > 0 else "",
                "oracle_vs_reference": oracle_score / ref_score if ref_score > 0 else "",
                "reference_kv_keep": ref_kv,
                "learned_kv_keep": learned_kv,
                "oracle_kv_keep": oracle_kv,
                "reference_online_seconds": ref_online,
                "learned_online_seconds": learned_online,
                "oracle_online_seconds": oracle_online,
                "learned_speed_vs_reference": ref_online / learned_online if learned_online > 0 else "",
                "oracle_speed_vs_reference": ref_online / oracle_online if oracle_online > 0 else "",
                "safe_rate": len(safe) / max(1, len(subset)),
                "oracle_action_accuracy": len(exact) / max(1, len(subset)),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--min_nonreference_candidates", type=int, default=0)
    parser.add_argument("--quality_ratio", type=float, default=1.0)
    parser.add_argument("--quality_margin", type=float, default=0.0)
    parser.add_argument("--task_encoding", choices=["none", "family", "task", "both"], default="both")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test_fold", type=int, default=0)
    parser.add_argument("--safe_probability_threshold", type=float, default=0.5)
    parser.add_argument("--model", choices=["random_forest", "extra_trees"], default="random_forest")
    parser.add_argument("--class_weight_mode", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--max_depth", type=int, default=7)
    parser.add_argument("--min_samples_leaf", type=int, default=4)
    parser.add_argument("--random_seed", type=int, default=17)
    args = parser.parse_args()

    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_rows = by_key(read_csv(Path(args.reference_dir) / "task_results.csv"))
    if not reference_rows:
        raise FileNotFoundError(f"No task_results.csv rows in {args.reference_dir}")

    candidates: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    action_budgets: dict[str, int] = {}
    for item in args.candidate:
        name, path = parse_candidate_arg(item)
        rows = by_key(read_csv(path / "task_results.csv"))
        if not rows:
            continue
        candidates[name] = rows
        action_budgets[name] = parse_budget_from_action(name)
    if not candidates:
        raise ValueError("Need at least one budget candidate")
    ordered_actions = sorted(action_budgets, key=lambda name: action_budgets[name])

    sample_rows: list[dict[str, Any]] = []
    binary_rows: list[dict[str, Any]] = []
    candidate_metric_rows: list[dict[str, Any]] = []
    for key, ref_row in sorted(reference_rows.items()):
        task, sample_id = key
        ref_score = fnum(ref_row, "score")
        target = quality_target(ref_score, args.quality_ratio, args.quality_margin)
        available = [(name, table[key]) for name, table in candidates.items() if key in table]
        if len(available) < args.min_nonreference_candidates:
            continue
        base = base_feature_row(ref_row)
        safe = [(name, row) for name, row in available if fnum(row, "score") + 1e-12 >= target]
        if safe:
            oracle_name, oracle_row = min(safe, key=lambda item: candidate_cost(item[1]))
        else:
            oracle_name, oracle_row = "reference", ref_row
        sample_record = {
            **base,
            "sample_id": sample_id,
            "fold": fold_for_key(task, sample_id, args.folds),
            "oracle_action": oracle_name,
            "reference_score": ref_score,
            "reference_kv_keep": fnum(ref_row, "keep_fraction"),
            "reference_online_seconds": fnum(ref_row, "online_seconds"),
            "oracle_score": fnum(oracle_row, "score"),
            "oracle_kv_keep": fnum(oracle_row, "keep_fraction"),
            "oracle_online_seconds": fnum(oracle_row, "online_seconds"),
            "quality_target": target,
            "available_actions": ",".join(name for name, _row in sorted(available, key=lambda item: action_budgets[item[0]])),
        }
        sample_rows.append(sample_record)
        for name, cand_row in available:
            budget = action_budgets[name]
            is_safe = int(fnum(cand_row, "score") + 1e-12 >= target)
            binary = add_budget_features(sample_record, name, budget)
            binary["is_safe"] = is_safe
            binary["candidate_score"] = fnum(cand_row, "score")
            binary["candidate_kv_keep"] = fnum(cand_row, "keep_fraction")
            binary["candidate_online_seconds"] = fnum(cand_row, "online_seconds")
            binary_rows.append(binary)
            candidate_metric_rows.append(
                {
                    "task": task,
                    "sample_id": sample_id,
                    "candidate": name,
                    "budget_tokens": budget,
                    "score": fnum(cand_row, "score"),
                    "kv_keep": fnum(cand_row, "keep_fraction"),
                    "online_seconds": fnum(cand_row, "online_seconds"),
                    "is_safe": is_safe,
                    "reference_score": ref_score,
                    "quality_target": target,
                }
            )

    if not sample_rows or not binary_rows:
        raise ValueError("No joined samples after applying candidate coverage filter")

    train_samples = [row for row in sample_rows if int(row["fold"]) != args.test_fold]
    test_samples = [row for row in sample_rows if int(row["fold"]) == args.test_fold]
    if not test_samples:
        test_samples = train_samples
    train_keys = {(row["task"], row["sample_id"]) for row in train_samples}
    train_binary = [row for row in binary_rows if (row["task"], row["sample_id"]) in train_keys]

    categories = category_features(sample_rows, args.task_encoding)
    feature_names = BASE_NUMERIC_FEATURES + BUDGET_FEATURES + categories
    numeric_feature_names = BASE_NUMERIC_FEATURES + BUDGET_FEATURES
    x_train = make_matrix(train_binary, numeric_feature_names, categories, args.task_encoding)
    y_train = [int(row["is_safe"]) for row in train_binary]
    if len(set(y_train)) == 1:
        model: Any = {"constant_probability": float(y_train[0])}
    elif args.model == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_seed,
            class_weight="balanced" if args.class_weight_mode == "balanced" else None,
        )
        model.fit(x_train, y_train)
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_seed,
            class_weight="balanced_subsample" if args.class_weight_mode == "balanced" else None,
        )
        model.fit(x_train, y_train)

    candidate_by_key: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for name, table in candidates.items():
        for key, row in table.items():
            candidate_by_key[key][name] = row

    prediction_rows: list[dict[str, Any]] = []
    for split_name, rows in [("train", train_samples), ("test", test_samples), ("all", sample_rows)]:
        expanded_rows: list[dict[str, Any]] = []
        expanded_owner: list[dict[str, Any]] = []
        for sample in rows:
            key = (str(sample["task"]), str(sample["sample_id"]))
            available = candidate_by_key.get(key, {})
            for action in ordered_actions:
                if action not in available:
                    continue
                expanded_rows.append(add_budget_features(sample, action, action_budgets[action]))
                expanded_owner.append(sample)
        probabilities = positive_probability(
            model,
            make_matrix(expanded_rows, numeric_feature_names, categories, args.task_encoding),
        )
        by_sample: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
        for row, sample, probability in zip(expanded_rows, expanded_owner, probabilities):
            by_sample[(str(sample["task"]), str(sample["sample_id"]))].append(
                (str(row["candidate_action"]), float(probability))
            )
        for sample in rows:
            key = (str(sample["task"]), str(sample["sample_id"]))
            scored_actions = sorted(by_sample.get(key, []), key=lambda item: action_budgets[item[0]])
            selected_action = "reference"
            selected_probability = 0.0
            for action, probability in scored_actions:
                if probability >= args.safe_probability_threshold:
                    selected_action = action
                    selected_probability = probability
                    break
            available = candidate_by_key.get(key, {})
            learned_row = available.get(selected_action, reference_rows[key])
            prediction_rows.append(
                {
                    "split": split_name,
                    "task": sample["task"],
                    "task_family": sample["task_family"],
                    "sample_id": sample["sample_id"],
                    "fold": sample["fold"],
                    "oracle_action": sample["oracle_action"],
                    "learned_action": selected_action,
                    "learned_confidence": selected_probability,
                    "available_actions": sample["available_actions"],
                    "reference_score": sample["reference_score"],
                    "learned_score": fnum(learned_row, "score"),
                    "oracle_score": sample["oracle_score"],
                    "quality_target": sample["quality_target"],
                    "reference_kv_keep": sample["reference_kv_keep"],
                    "learned_kv_keep": fnum(learned_row, "keep_fraction"),
                    "oracle_kv_keep": sample["oracle_kv_keep"],
                    "reference_online_seconds": sample["reference_online_seconds"],
                    "learned_online_seconds": fnum(learned_row, "online_seconds"),
                    "oracle_online_seconds": sample["oracle_online_seconds"],
                    "safe_probabilities": ";".join(f"{action}:{probability:.6f}" for action, probability in scored_actions),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for split_name in ["train", "test", "all"]:
        split_rows = [row for row in prediction_rows if row["split"] == split_name]
        summary_rows.extend(summarize_predictions(split_rows, split_name))

    label_rows = [
        {"is_safe": label, "count": count}
        for label, count in Counter(int(row["is_safe"]) for row in binary_rows).most_common()
    ]
    action_rows = [
        {"oracle_action": action, "count": count}
        for action, count in Counter(str(row["oracle_action"]) for row in sample_rows).most_common()
    ]
    feature_rows: list[dict[str, Any]] = []
    if not isinstance(model, dict) and hasattr(model, "feature_importances_"):
        for name, importance in sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda item: float(item[1]),
            reverse=True,
        ):
            feature_rows.append({"feature": name, "importance": float(importance)})

    metadata = {
        "router_type": "budget_safety_ladder_v1",
        "reference_dir": args.reference_dir,
        "candidate_names": ordered_actions,
        "action_budgets": action_budgets,
        "quality_ratio": args.quality_ratio,
        "quality_margin": args.quality_margin,
        "task_encoding": args.task_encoding,
        "model": args.model,
        "class_weight_mode": args.class_weight_mode,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "safe_probability_threshold": args.safe_probability_threshold,
        "folds": args.folds,
        "test_fold": args.test_fold,
        "samples": len(sample_rows),
        "binary_rows": len(binary_rows),
        "min_nonreference_candidates": args.min_nonreference_candidates,
        "feature_names": feature_names,
        "numeric_feature_names": numeric_feature_names,
        "category_features": categories,
    }
    write_csv(output_dir / "training_samples.csv", sample_rows)
    write_csv(output_dir / "binary_safety_labels.csv", binary_rows)
    write_csv(output_dir / "candidate_metrics.csv", candidate_metric_rows)
    write_csv(output_dir / "router_predictions.csv", prediction_rows)
    write_csv(output_dir / "router_summary.csv", summary_rows)
    write_csv(output_dir / "label_distribution.csv", label_rows)
    write_csv(output_dir / "oracle_action_distribution.csv", action_rows)
    write_csv(output_dir / "feature_importance.csv", feature_rows)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"model": model, "metadata": metadata}, handle)

    test_all = next(row for row in summary_rows if row["split"] == "test" and row["task"] == "ALL")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "samples": len(sample_rows),
                "binary_rows": len(binary_rows),
                "safe_labels": label_rows,
                "oracle_actions": action_rows,
                "test_all": test_all,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
