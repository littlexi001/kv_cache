#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NUMERIC_FEATURES = [
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
    "ours_query_coverage_terms",
    "ours_query_coverage_covered",
    "ours_query_coverage_recall",
]

PRESELECTION_NUMERIC_FEATURES = [
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


def fnum(row: dict[str, str] | None, key: str) -> float:
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


def label_for_dir(path: Path) -> str:
    name = path.name
    if name.startswith("riskkv_v19_"):
        name = name[len("riskkv_v19_") :]
    suffixes = [
        "_m100_bDyn_pDyn",
        "_m20_bDyn_pDyn",
        "_20260711_b16_compressed_mid_smoke_m20_bDyn_pDyn",
        "_20260711_b16_manyblocks_xl_smoke_m20_bDyn_pDyn",
        "_20260711_bm25_bridge_smoke_m20_bDyn_pDyn",
        "_20260711_qa_shortdecode_smoke_m20_bDyn_pDyn",
        "_20260711_qasper_bm25_budget_smoke_m20_bDyn_pDyn",
        "_20260711_b16_windowvote_sweep_m100_bDyn_pDyn",
        "_20260711_b16_microspan_sweep_m100_bDyn_pDyn",
        "_20260711_b16_purefine_sweep_m100_bDyn_pDyn",
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
    return name


def parse_candidate_arg(text: str) -> tuple[str, Path]:
    if "=" in text:
        name, path = text.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(text)
    return label_for_dir(path), path


def fold_for_key(task: str, sample_id: str, folds: int) -> int:
    digest = hashlib.md5(f"{task}\t{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, folds)


def quality_target(reference_score: float, ratio: float, margin: float) -> float:
    return max(0.0, ratio * reference_score, reference_score - margin)


def candidate_cost(row: dict[str, str], cost_mode: str) -> tuple[float, float, float]:
    kv = fnum(row, "keep_fraction")
    online = fnum(row, "online_seconds")
    if cost_mode == "online":
        return (online, kv, -fnum(row, "score"))
    if cost_mode == "hybrid":
        return (kv + 0.05 * online, online, -fnum(row, "score"))
    return (kv, online, -fnum(row, "score"))


def task_family(task: str) -> str:
    return FAMILY_BY_TASK.get(task, "other")


def numeric_feature_names(feature_set: str) -> list[str]:
    if feature_set == "preselection":
        return list(PRESELECTION_NUMERIC_FEATURES)
    return list(NUMERIC_FEATURES)


def feature_values(
    reference_row: dict[str, str],
    task_encoding: str,
    numeric_features: list[str],
) -> tuple[list[float], list[str], list[Any]]:
    task = reference_row.get("task", "")
    numeric = [fnum(reference_row, key) for key in numeric_features]
    categories: list[str] = []
    if task_encoding in {"family", "both"}:
        categories.append(f"family={task_family(task)}")
    if task_encoding in {"task", "both"}:
        categories.append(f"task={task}")
    values: list[Any] = []
    values.extend(numeric)
    values.extend(categories)
    return numeric, categories, values


def build_feature_space(
    rows: list[dict[str, Any]],
    task_encoding: str,
    feature_set: str,
) -> tuple[list[str], list[str]]:
    categories: set[str] = set()
    if task_encoding in {"family", "both"}:
        for family in sorted(set(FAMILY_BY_TASK.values()) | {"other"}):
            categories.add(f"family={family}")
    if task_encoding in {"task", "both"}:
        for row in rows:
            categories.add(f"task={row['task']}")
    return numeric_feature_names(feature_set), sorted(categories)


def make_matrix(
    rows: list[dict[str, Any]],
    numeric_features: list[str],
    category_features: list[str],
    task_encoding: str,
) -> list[list[float]]:
    category_index = {name: idx for idx, name in enumerate(category_features)}
    matrix: list[list[float]] = []
    for row in rows:
        vector = [float(row.get(feature, 0.0) or 0.0) for feature in numeric_features]
        one_hot = [0.0] * len(category_features)
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


def row_metrics(row: dict[str, str]) -> dict[str, float]:
    return {
        "score": fnum(row, "score"),
        "kv_keep": fnum(row, "keep_fraction"),
        "online_seconds": fnum(row, "online_seconds"),
    }


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
    parser.add_argument("--reference_dir", required=True, help="Current practical baseline, e.g. v300.")
    parser.add_argument("--full_dir", default="", help="Optional full-KV directory for reporting.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate", action="append", default=[], help="name=dir or dir. Can be repeated.")
    parser.add_argument("--include_reference_candidate", action="store_true", default=True)
    parser.add_argument(
        "--min_nonreference_candidates",
        type=int,
        default=0,
        help="Skip samples with fewer than this many non-reference candidate rows. Useful for partial sweeps.",
    )
    parser.add_argument("--quality_ratio", type=float, default=0.95)
    parser.add_argument("--quality_margin", type=float, default=0.0)
    parser.add_argument("--cost_mode", choices=["kv", "online", "hybrid"], default="kv")
    parser.add_argument(
        "--feature_set",
        choices=["all", "preselection"],
        default="all",
        help="preselection drops post-selection query coverage features so the model is directly deployable online.",
    )
    parser.add_argument("--task_encoding", choices=["none", "family", "task", "both"], default="family")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test_fold", type=int, default=0)
    parser.add_argument("--model", choices=["decision_tree", "random_forest"], default="random_forest")
    parser.add_argument("--class_weight_mode", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--min_samples_leaf", type=int, default=3)
    parser.add_argument(
        "--confidence_fallback_threshold",
        type=float,
        default=0.0,
        help="If max predicted action probability is below this threshold, fall back to reference.",
    )
    parser.add_argument("--random_seed", type=int, default=13)
    args = parser.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier, export_text

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_rows = by_key(read_csv(Path(args.reference_dir) / "task_results.csv"))
    if not reference_rows:
        raise FileNotFoundError(f"No task_results.csv rows in {args.reference_dir}")
    full_rows = by_key(read_csv(Path(args.full_dir) / "task_results.csv")) if args.full_dir else {}

    candidates: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    if args.include_reference_candidate:
        candidates["reference"] = reference_rows
    for item in args.candidate:
        name, path = parse_candidate_arg(item)
        rows = by_key(read_csv(path / "task_results.csv"))
        if rows:
            candidates[name] = rows
    if len(candidates) < 2:
        raise ValueError("Need at least one non-reference candidate with task_results.csv")

    numeric_features_for_records = numeric_feature_names(args.feature_set)
    training_rows: list[dict[str, Any]] = []
    candidate_metric_rows: list[dict[str, Any]] = []
    for key, ref_row in sorted(reference_rows.items()):
        task, sample_id = key
        ref_score = fnum(ref_row, "score")
        target = quality_target(ref_score, args.quality_ratio, args.quality_margin)
        available: list[tuple[str, dict[str, str]]] = []
        for name, table in candidates.items():
            row = table.get(key)
            if row is not None:
                available.append((name, row))
                metric = row_metrics(row)
                candidate_metric_rows.append(
                    {
                        "task": task,
                        "sample_id": sample_id,
                        "candidate": name,
                        "score": metric["score"],
                        "kv_keep": metric["kv_keep"],
                        "online_seconds": metric["online_seconds"],
                        "is_safe": int(metric["score"] + 1e-12 >= target),
                        "reference_score": ref_score,
                        "quality_target": target,
                    }
                )
        nonreference_count = sum(1 for name, _row in available if name != "reference")
        if nonreference_count < args.min_nonreference_candidates:
            continue
        if not available:
            continue
        safe = [(name, row) for name, row in available if fnum(row, "score") + 1e-12 >= target]
        if not safe:
            safe = sorted(available, key=lambda item: fnum(item[1], "score"), reverse=True)[:1]
        oracle_name, oracle_row = min(safe, key=lambda item: candidate_cost(item[1], args.cost_mode))
        numeric, categories, _ = feature_values(ref_row, args.task_encoding, numeric_features_for_records)
        record: dict[str, Any] = {
            "task": task,
            "task_family": task_family(task),
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
            "available_actions": ",".join(sorted(name for name, _row in available)),
        }
        for feature, value in zip(numeric_features_for_records, numeric):
            record[feature] = value
        for category in categories:
            record[category] = 1
        training_rows.append(record)

    if not training_rows:
        raise ValueError("No training rows after joining candidates with reference")

    numeric_features, category_features = build_feature_space(training_rows, args.task_encoding, args.feature_set)
    feature_names = numeric_features + category_features
    train_rows = [row for row in training_rows if int(row["fold"]) != args.test_fold]
    test_rows = [row for row in training_rows if int(row["fold"]) == args.test_fold]
    if not test_rows:
        test_rows = train_rows

    x_train = make_matrix(train_rows, numeric_features, category_features, args.task_encoding)
    y_train = [str(row["oracle_action"]) for row in train_rows]
    if len(set(y_train)) == 1:
        model: Any = {"constant": y_train[0]}
    elif args.model == "decision_tree":
        clf = DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_seed,
            class_weight="balanced" if args.class_weight_mode == "balanced" else None,
        )
        clf.fit(x_train, y_train)
        model = clf
    else:
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_seed,
            class_weight="balanced_subsample" if args.class_weight_mode == "balanced" else None,
        )
        clf.fit(x_train, y_train)
        model = clf

    def predict(rows: list[dict[str, Any]]) -> list[tuple[str, float]]:
        if isinstance(model, dict):
            return [(str(model["constant"]), 1.0) for _ in rows]
        matrix = make_matrix(rows, numeric_features, category_features, args.task_encoding)
        labels = [str(item) for item in model.predict(matrix)]
        confidences = [1.0] * len(labels)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(matrix)
            confidences = [float(max(row)) if len(row) else 0.0 for row in probabilities]
        return list(zip(labels, confidences))

    candidate_by_key: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for name, table in candidates.items():
        for key, row in table.items():
            candidate_by_key[key][name] = row

    prediction_rows: list[dict[str, Any]] = []
    for split_name, rows in [("train", train_rows), ("test", test_rows), ("all", training_rows)]:
        pred_actions = predict(rows)
        for row, (action, confidence) in zip(rows, pred_actions):
            key = (str(row["task"]), str(row["sample_id"]))
            available = candidate_by_key.get(key, {})
            fallback_used = 0
            confidence_fallback_used = 0
            if action != "reference" and confidence < args.confidence_fallback_threshold:
                action = "reference"
                confidence_fallback_used = 1
            if action not in available:
                action = "reference"
                fallback_used = 1
            learned = available[action]
            prediction_rows.append(
                {
                    "split": split_name,
                    "task": row["task"],
                    "task_family": row["task_family"],
                    "sample_id": row["sample_id"],
                    "fold": row["fold"],
                    "oracle_action": row["oracle_action"],
                    "learned_action": action,
                    "learned_confidence": confidence,
                    "fallback_used": fallback_used,
                    "confidence_fallback_used": confidence_fallback_used,
                    "available_actions": row["available_actions"],
                    "reference_score": row["reference_score"],
                    "learned_score": fnum(learned, "score"),
                    "oracle_score": row["oracle_score"],
                    "quality_target": row["quality_target"],
                    "reference_kv_keep": row["reference_kv_keep"],
                    "learned_kv_keep": fnum(learned, "keep_fraction"),
                    "oracle_kv_keep": row["oracle_kv_keep"],
                    "reference_online_seconds": row["reference_online_seconds"],
                    "learned_online_seconds": fnum(learned, "online_seconds"),
                    "oracle_online_seconds": row["oracle_online_seconds"],
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for split_name in ["train", "test", "all"]:
        split_rows = [row for row in prediction_rows if row["split"] == split_name]
        summary_rows.extend(summarize_predictions(split_rows, split_name))

    label_rows = [
        {"oracle_action": action, "count": count}
        for action, count in Counter(row["oracle_action"] for row in training_rows).most_common()
    ]

    feature_rows: list[dict[str, Any]] = []
    if not isinstance(model, dict) and hasattr(model, "feature_importances_"):
        for name, importance in sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda item: float(item[1]),
            reverse=True,
        ):
            feature_rows.append({"feature": name, "importance": float(importance)})

    write_csv(output_dir / "training_labels.csv", training_rows)
    write_csv(output_dir / "candidate_metrics.csv", candidate_metric_rows)
    write_csv(output_dir / "router_predictions.csv", prediction_rows)
    write_csv(output_dir / "router_summary.csv", summary_rows)
    write_csv(output_dir / "label_distribution.csv", label_rows)
    write_csv(output_dir / "feature_importance.csv", feature_rows)

    metadata = {
        "reference_dir": args.reference_dir,
        "full_dir": args.full_dir,
        "candidate_names": sorted(candidates),
        "quality_ratio": args.quality_ratio,
        "quality_margin": args.quality_margin,
        "cost_mode": args.cost_mode,
        "task_encoding": args.task_encoding,
        "feature_set": args.feature_set,
        "model": args.model,
        "class_weight_mode": args.class_weight_mode,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "confidence_fallback_threshold": args.confidence_fallback_threshold,
        "folds": args.folds,
        "test_fold": args.test_fold,
        "samples": len(training_rows),
        "min_nonreference_candidates": args.min_nonreference_candidates,
        "feature_names": feature_names,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"model": model, "metadata": metadata}, handle)
    if not isinstance(model, dict) and args.model == "decision_tree":
        (output_dir / "decision_tree.txt").write_text(export_text(model, feature_names=feature_names), encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "samples": len(training_rows), "labels": label_rows[:10]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
