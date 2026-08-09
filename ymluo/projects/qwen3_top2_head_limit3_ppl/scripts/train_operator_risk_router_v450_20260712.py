#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import train_policy_action_planner_v378_20260712 as v378
import train_source_router_v437_20260712 as v437
from run_controlled_public_kv_benchmark_v1 import Example


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict[str, Any] | None, key: str) -> float:
    if row is None:
        return 0.0
    try:
        return float(row.get(key, "") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["task"], row["sample_id"]): row
        for row in rows
        if row.get("task") and row.get("sample_id")
    }


def content_only_text(example: Example) -> str:
    context = example.context
    if len(context) > 8000:
        context = context[:4000] + "\n...\n" + context[-4000:]
    return "\n".join(["QUERY:", example.query, "CONTEXT:", context])


def make_records(
    examples: list[Example],
    full_table: dict[tuple[str, str], dict[str, str]],
    system_table: dict[tuple[str, str], dict[str, str]],
    sparse_table: dict[tuple[str, str], dict[str, str]],
    quality_ratio: float,
) -> list[dict[str, Any]]:
    example_by_key = {(example.task, example.sample_id): example for example in examples}
    keys = sorted(set(example_by_key) & set(full_table) & set(system_table) & set(sparse_table))
    records: list[dict[str, Any]] = []
    for key in keys:
        task, sample_id = key
        example = example_by_key[key]
        full = full_table[key]
        system = system_table[key]
        pure = sparse_table[key]
        full_score = fnum(full, "score")
        target = quality_ratio * full_score
        system_safe = int(fnum(system, "score") + 1e-12 >= target)
        sparse_safe = int(fnum(pure, "score") + 1e-12 >= target)
        if system_safe:
            oracle_action = "system"
        elif sparse_safe:
            oracle_action = "sparse"
        else:
            oracle_action = "full"
        record: dict[str, Any] = {
            "task": task,
            "family": v378.task_family(task),
            "sample_id": sample_id,
            "fold": v378.fold_for_key(task, sample_id),
            "text": content_only_text(example),
            "quality_target": target,
            "full_score": full_score,
            "full_kv": 1.0,
            "full_online": fnum(full, "online_seconds"),
            "full_total": fnum(full, "total_seconds"),
            "system_score": fnum(system, "score"),
            "system_kv": fnum(system, "keep_fraction"),
            "system_online": fnum(system, "online_seconds"),
            "system_total": fnum(system, "total_seconds"),
            "system_safe": system_safe,
            "system_direct_used": int(fnum(system, "ours_direct_structured_answer_used") > 0),
            "sparse_score": fnum(pure, "score"),
            "sparse_kv": fnum(pure, "keep_fraction"),
            "sparse_online": fnum(pure, "online_seconds"),
            "sparse_total": fnum(pure, "total_seconds"),
            "sparse_safe": sparse_safe,
            "oracle_action": oracle_action,
        }
        for feature in NUMERIC_FEATURES:
            record[feature] = fnum(pure, feature)
        records.append(record)
    return records


class FeatureBuilder:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            max_features=12000,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w\-]+\b",
        )
        self.mean = np.zeros(len(NUMERIC_FEATURES), dtype=np.float32)
        self.scale = np.ones(len(NUMERIC_FEATURES), dtype=np.float32)

    @staticmethod
    def numeric(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [[float(row.get(feature, 0.0) or 0.0) for feature in NUMERIC_FEATURES] for row in rows],
            dtype=np.float32,
        )

    def fit_transform(self, rows: list[dict[str, Any]]) -> sparse.csr_matrix:
        text_matrix = self.vectorizer.fit_transform([str(row["text"]) for row in rows])
        numeric = self.numeric(rows)
        self.mean = numeric.mean(axis=0)
        self.scale = numeric.std(axis=0)
        self.scale[self.scale < 1e-6] = 1.0
        numeric = (numeric - self.mean) / self.scale
        return sparse.hstack([text_matrix, sparse.csr_matrix(numeric)], format="csr")

    def transform(self, rows: list[dict[str, Any]]) -> sparse.csr_matrix:
        text_matrix = self.vectorizer.transform([str(row["text"]) for row in rows])
        numeric = (self.numeric(rows) - self.mean) / self.scale
        return sparse.hstack([text_matrix, sparse.csr_matrix(numeric)], format="csr")


def fit_binary_model(x: sparse.csr_matrix, labels: np.ndarray, seed: int) -> Any:
    unique = np.unique(labels)
    if len(unique) == 1:
        return {"constant": float(unique[0])}
    model = LogisticRegression(
        C=1.5,
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=seed,
    )
    model.fit(x, labels)
    return model


def safe_probabilities(model: Any, x: sparse.csr_matrix) -> np.ndarray:
    if isinstance(model, dict):
        return np.full(x.shape[0], float(model["constant"]), dtype=np.float64)
    probs = model.predict_proba(x)
    classes = list(model.classes_)
    return probs[:, classes.index(1)] if 1 in classes else np.zeros(x.shape[0], dtype=np.float64)


def calibrate_threshold(probabilities: np.ndarray, labels: np.ndarray, max_false_safe: float) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for threshold in [round(item / 100, 2) for item in range(101)] + [1.01]:
        selected = probabilities >= threshold
        count = int(selected.sum())
        false_safe = int(((labels == 0) & selected).sum())
        risk = false_safe / max(1, count)
        candidates.append(
            {
                "threshold": threshold,
                "selected": count,
                "coverage": count / max(1, len(labels)),
                "false_safe": false_safe,
                "false_safe_rate": risk,
            }
        )
    feasible = [row for row in candidates if float(row["false_safe_rate"]) <= max_false_safe]
    return max(
        feasible or candidates,
        key=lambda row: (int(row["selected"]), -float(row["false_safe_rate"]), float(row["threshold"])),
    )


def evaluate_policy(
    rows: list[dict[str, Any]],
    system_prob: np.ndarray,
    sparse_prob: np.ndarray,
    system_threshold: float,
    sparse_threshold: float,
    scenario: str,
    holdout_type: str,
    holdout: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row, p_system, p_sparse in zip(rows, system_prob, sparse_prob):
        if p_system >= system_threshold:
            action = "system"
        elif p_sparse >= sparse_threshold:
            action = "sparse"
        else:
            action = "full"
        score = float(row[f"{action}_score"])
        kv = float(row[f"{action}_kv"])
        online = float(row[f"{action}_online"])
        total = float(row[f"{action}_total"])
        safe = int(score + 1e-12 >= float(row["quality_target"]))
        predictions.append(
            {
                "scenario": scenario,
                "holdout_type": holdout_type,
                "holdout": holdout,
                "task": row["task"],
                "family": row["family"],
                "sample_id": row["sample_id"],
                "oracle_action": row["oracle_action"],
                "selected_action": action,
                "system_safe_probability": float(p_system),
                "sparse_safe_probability": float(p_sparse),
                "system_threshold": system_threshold,
                "sparse_threshold": sparse_threshold,
                "full_score": row["full_score"],
                "score": score,
                "kv": kv,
                "online": online,
                "total": total,
                "safe": safe,
            }
        )
    full_score = np.mean([float(row["full_score"]) for row in predictions])
    score = np.mean([float(row["score"]) for row in predictions])
    full_online = np.mean([float(row["full_online"]) for row in rows])
    full_total = np.mean([float(row["full_total"]) for row in rows])
    online = np.mean([float(row["online"]) for row in predictions])
    total = np.mean([float(row["total"]) for row in predictions])
    actions = Counter(str(row["selected_action"]) for row in predictions)
    summary = {
        "scenario": scenario,
        "holdout_type": holdout_type,
        "holdout": holdout,
        "samples": len(rows),
        "full_score": full_score,
        "score": score,
        "score_vs_full": score / full_score if full_score > 0 else "",
        "kv": np.mean([float(row["kv"]) for row in predictions]),
        "online_speed_vs_full": full_online / online if online > 0 else "",
        "total_speed_vs_full": full_total / total if total > 0 else "",
        "unsafe_rate": 1.0 - np.mean([float(row["safe"]) for row in predictions]),
        "oracle_action_accuracy": np.mean(
            [row["selected_action"] == row["oracle_action"] for row in predictions]
        ),
        "system_rate": actions["system"] / max(1, len(rows)),
        "sparse_rate": actions["sparse"] / max(1, len(rows)),
        "full_fallback_rate": actions["full"] / max(1, len(rows)),
        "system_threshold": system_threshold,
        "sparse_threshold": sparse_threshold,
    }
    return predictions, summary


def run_scenario(
    records: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    scenario: str,
    holdout_type: str,
    holdout: str,
    max_false_safe: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    del records
    builder = FeatureBuilder()
    x_train = builder.fit_transform(train_rows)
    x_cal = builder.transform(calibration_rows)
    x_test = builder.transform(test_rows)
    models = {
        action: fit_binary_model(
            x_train,
            np.asarray([int(row[f"{action}_safe"]) for row in train_rows]),
            450 + index,
        )
        for index, action in enumerate(["system", "sparse"])
    }
    thresholds: dict[str, float] = {}
    calibration: list[dict[str, Any]] = []
    test_probs: dict[str, np.ndarray] = {}
    for action in ["system", "sparse"]:
        cal_prob = safe_probabilities(models[action], x_cal)
        cal_labels = np.asarray([int(row[f"{action}_safe"]) for row in calibration_rows])
        selected = calibrate_threshold(cal_prob, cal_labels, max_false_safe)
        thresholds[action] = float(selected["threshold"])
        calibration.append(
            {
                "scenario": scenario,
                "holdout_type": holdout_type,
                "holdout": holdout,
                "action": action,
                **selected,
            }
        )
        test_probs[action] = safe_probabilities(models[action], x_test)
    predictions, summary = evaluate_policy(
        test_rows,
        test_probs["system"],
        test_probs["sparse"],
        thresholds["system"],
        thresholds["sparse"],
        scenario,
        holdout_type,
        holdout,
    )
    return predictions, summary, calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(v437.ROOT))
    parser.add_argument(
        "--longbench-zip",
        default="outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/longbench_data.zip",
    )
    parser.add_argument("--full-results", default="outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv")
    parser.add_argument(
        "--system-results",
        default="outputs/riskkv_v19_v437_source_router_v428_v437_m100_20260712_m100_bDyn_pDyn/task_results.csv",
    )
    parser.add_argument(
        "--sparse-results",
        default="outputs/riskkv_v19_v440_true_pure_source_router_v440_m20_20260712_m20_bDyn_pDyn/task_results.csv",
    )
    parser.add_argument("--max-samples-per-task", type=int, default=20)
    parser.add_argument("--quality-ratio", type=float, default=0.95)
    parser.add_argument("--max-false-safe", type=float, default=0.10)
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_v450_operator_risk_router_20260712")
    args = parser.parse_args()

    root = Path(args.root)
    examples = v437.load_longbench_examples(
        root / args.longbench_zip,
        v437.TASKS,
        args.max_samples_per_task,
        1234,
    )
    records = make_records(
        examples,
        by_key(read_csv(root / args.full_results)),
        by_key(read_csv(root / args.system_results)),
        by_key(read_csv(root / args.sparse_results)),
        args.quality_ratio,
    )
    if not records:
        raise RuntimeError("No matched Full/system/sparse samples")

    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    scenarios: list[tuple[str, str, str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = []
    scenarios.append(
        (
            "sample_holdout",
            "sample",
            "fold0",
            [row for row in records if int(row["fold"]) not in {0, 1}],
            [row for row in records if int(row["fold"]) == 1],
            [row for row in records if int(row["fold"]) == 0],
        )
    )
    for task in v437.TASKS:
        scenarios.append(
            (
                f"loto_{task}",
                "task",
                task,
                [row for row in records if row["task"] != task and int(row["fold"]) != 1],
                [row for row in records if row["task"] != task and int(row["fold"]) == 1],
                [row for row in records if row["task"] == task],
            )
        )
    families = sorted({str(row["family"]) for row in records})
    for family in families:
        scenarios.append(
            (
                f"lofo_{family}",
                "family",
                family,
                [row for row in records if row["family"] != family and int(row["fold"]) != 1],
                [row for row in records if row["family"] != family and int(row["fold"]) == 1],
                [row for row in records if row["family"] == family],
            )
        )

    for scenario, holdout_type, holdout, train_rows, cal_rows, test_rows in scenarios:
        if not train_rows or not cal_rows or not test_rows:
            continue
        predictions, summary, calibration = run_scenario(
            records,
            train_rows,
            cal_rows,
            test_rows,
            scenario,
            holdout_type,
            holdout,
            args.max_false_safe,
        )
        prediction_rows.extend(predictions)
        summary_rows.append(summary)
        calibration_rows.extend(calibration)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "matched_frontier.csv", records)
    write_csv(output_dir / "router_predictions.csv", prediction_rows)
    write_csv(output_dir / "router_summary.csv", summary_rows)
    write_csv(output_dir / "calibration.csv", calibration_rows)
    metadata = {
        "router": "v450_operator_risk_feasibility",
        "diagnostic_only": True,
        "actions": ["system", "sparse", "full"],
        "uses_task_name_as_feature": False,
        "uses_prompt_template": False,
        "numeric_features": NUMERIC_FEATURES,
        "quality_ratio": args.quality_ratio,
        "max_false_safe": args.max_false_safe,
        "samples": len(records),
        "action_counts": Counter(str(row["oracle_action"]) for row in records),
        "caveat": "The system action still contains benchmark-specific v437 policy internals; this run only tests whether request-level risk routing is learnable.",
    }
    metadata["action_counts"] = dict(metadata["action_counts"])
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output_dir)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
