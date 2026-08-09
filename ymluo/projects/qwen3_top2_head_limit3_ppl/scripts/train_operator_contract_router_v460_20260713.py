#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import train_policy_action_planner_v378_20260712 as v378
import train_source_router_v437_20260712 as v437
from run_controlled_public_kv_benchmark_v1 import Example, operator_router_example_text


CONTRACT_BY_TASK = {
    "narrativeqa": "retrieve",
    "qasper": "retrieve",
    "multifieldqa_en": "retrieve",
    "hotpotqa": "retrieve",
    "2wikimqa": "retrieve",
    "musique": "retrieve",
    "triviaqa": "retrieve",
    "gov_report": "aggregate",
    "qmsum": "aggregate",
    "multi_news": "aggregate",
    "samsum": "aggregate",
    "trec": "structured",
    "passage_count": "structured",
    "passage_retrieval_en": "structured",
    "lcc": "code",
    "repobench-p": "code",
}


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


def make_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.98,
                    max_features=18000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b[\w\-]+\b",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=2.0,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_model(examples: list[Example], seed: int) -> Pipeline:
    model = make_model(seed)
    model.fit(
        [operator_router_example_text(example) for example in examples],
        [CONTRACT_BY_TASK[example.task] for example in examples],
    )
    return model


def predict(model: Pipeline, examples: list[Example], threshold: float, default: str) -> list[dict[str, Any]]:
    if not examples:
        return []
    texts = [operator_router_example_text(example) for example in examples]
    raw = model.predict(texts)
    probabilities = model.predict_proba(texts)
    rows: list[dict[str, Any]] = []
    for example, raw_action, probs in zip(examples, raw, probabilities):
        confidence = float(np.max(probs))
        action = str(raw_action)
        reason = ""
        if action != default and confidence < threshold:
            action = default
            reason = "low_confidence"
        target = CONTRACT_BY_TASK[example.task]
        rows.append(
            {
                "task": example.task,
                "family": v378.task_family(example.task),
                "sample_id": example.sample_id,
                "fold": v378.fold_for_key(example.task, example.sample_id),
                "target": target,
                "raw_action": str(raw_action),
                "action": action,
                "confidence": confidence,
                "fallback_reason": reason,
                "correct": int(action == target),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], split: str, holdout: str) -> dict[str, Any]:
    confusion = Counter(f"{row['target']}->{row['action']}" for row in rows)
    return {
        "split": split,
        "holdout": holdout,
        "samples": len(rows),
        "accuracy": sum(int(row["correct"]) for row in rows) / max(1, len(rows)),
        "fallback_rate": sum(bool(row["fallback_reason"]) for row in rows) / max(1, len(rows)),
        "mean_confidence": sum(float(row["confidence"]) for row in rows) / max(1, len(rows)),
        "confusion": json.dumps(dict(confusion), sort_keys=True),
    }


def select_threshold(model: Pipeline, examples: list[Example], default: str) -> tuple[float, list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    for threshold in [round(item / 100, 2) for item in range(101)] + [1.01]:
        rows = predict(model, examples, threshold, default)
        summary = summarize(rows, "calibration", "fold1")
        trials.append({**summary, "threshold": threshold})
    selected = max(
        trials,
        key=lambda row: (float(row["accuracy"]), -float(row["fallback_rate"]), -float(row["threshold"])),
    )
    return float(selected["threshold"]), trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(v437.ROOT))
    parser.add_argument(
        "--longbench-zip",
        default="outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/longbench_data.zip",
    )
    parser.add_argument("--max-samples-per-task", type=int, default=100)
    parser.add_argument("--default-action", default="retrieve")
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_v460_operator_contract_router_20260713")
    parser.add_argument("--config-out", default="configs/riskkv_operator_contract_router_v460_20260713.json")
    args = parser.parse_args()

    root = Path(args.root)
    examples = v437.load_longbench_examples(
        root / args.longbench_zip,
        list(CONTRACT_BY_TASK),
        args.max_samples_per_task,
        1234,
    )
    train = [example for example in examples if v378.fold_for_key(example.task, example.sample_id) not in {0, 1}]
    calibration = [example for example in examples if v378.fold_for_key(example.task, example.sample_id) == 1]
    test = [example for example in examples if v378.fold_for_key(example.task, example.sample_id) == 0]
    diagnostic_model = fit_model(train, 460)
    threshold, threshold_rows = select_threshold(diagnostic_model, calibration, args.default_action)

    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for split, subset in [("train", train), ("calibration", calibration), ("test", test)]:
        rows = predict(diagnostic_model, subset, threshold, args.default_action)
        for row in rows:
            row["split"] = split
            row["holdout"] = "sample"
        prediction_rows.extend(rows)
        summary_rows.append(summarize(rows, split, "sample"))

    for index, task in enumerate(CONTRACT_BY_TASK):
        loto_train = [example for example in examples if example.task != task]
        loto_test = [example for example in examples if example.task == task]
        loto_model = fit_model(loto_train, 461 + index)
        rows = predict(loto_model, loto_test, 0.0, args.default_action)
        for row in rows:
            row["split"] = "loto"
            row["holdout"] = task
        prediction_rows.extend(rows)
        summary_rows.append(summarize(rows, "loto", task))

    final_train = [example for example in examples if v378.fold_for_key(example.task, example.sample_id) != 0]
    final_model = fit_model(final_train, 476)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "router_type": "operator_contract_router_v460",
        "input_type": "content_only",
        "contracts": sorted(set(CONTRACT_BY_TASK.values())),
        "confidence_fallback_threshold": threshold,
        "default_action": args.default_action,
        "uses_task_name_as_feature": False,
        "uses_prompt_template": False,
        "training_excludes_fold0": True,
        "teacher_labels_derived_from_task_for_training_only": True,
    }
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"model": final_model, "metadata": metadata}, handle)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "router_predictions.csv", prediction_rows)
    write_csv(output_dir / "router_summary.csv", summary_rows)
    write_csv(output_dir / "threshold_sweep.csv", threshold_rows)

    base_config = {
        "__comment": "v460 generic operator base. No LongBench task names or legacy task-policy sources.",
        "*": {
            "operator_mode": "retrieve",
            "budget_tokens": 1024,
            "page_tokens": 16,
            "sink_tokens": 32,
            "recent_tokens": 64,
            "scorer": "hybrid_late_mmr_multiscale_idf_flow",
            "direct_structured_answer": False,
            "short_decode": False,
        },
    }
    base_path = root / "configs/riskkv_operator_contract_base_v460_20260713.json"
    base_path.write_text(json.dumps(base_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    config = {
        "__extends": base_path.name,
        "__comment": "v460 learned request-contract router with generic operator fragments and no task-name input.",
        "__operator_router": {
            "model_path": str((output_dir / "model.pkl").relative_to(root)).replace("\\", "/"),
            "confidence_threshold": threshold,
            "default_action": args.default_action,
            "actions": {
                "retrieve": {
                    "budget_tokens": 1024,
                    "page_tokens": 16,
                    "sink_tokens": 32,
                    "recent_tokens": 64,
                    "scorer": "hybrid_late_mmr_multiscale_idf_flow",
                    "direct_structured_answer": False,
                },
                "aggregate": {
                    "budget_tokens": 128,
                    "page_tokens": 16,
                    "sink_tokens": 32,
                    "recent_tokens": 32,
                    "scorer": "hybrid_late_mmr_multiscale_idf_spread_flow",
                    "direct_structured_answer": True,
                    "direct_summary_max_words": 256,
                },
                "structured": {
                    "budget_tokens": 128,
                    "page_tokens": 16,
                    "sink_tokens": 32,
                    "recent_tokens": 32,
                    "scorer": "hybrid_late_mmr_multiscale_idf_flow",
                    "structured_fingerprint": True,
                    "structured_fingerprint_budget_fraction": 0.30,
                    "direct_structured_answer": True,
                },
                "code": {
                    "budget_tokens": 2048,
                    "page_tokens": 128,
                    "sink_tokens": 64,
                    "recent_tokens": 128,
                    "scorer": "hybrid_late_mmr_multiscale_idf_flow",
                    "direct_structured_answer": False,
                },
                "full": {"full_fallback": True, "direct_structured_answer": False},
            },
        },
    }
    config_path = root / args.config_out
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_dir)
    print(config_path)
    print(json.dumps({"threshold": threshold, "summaries": summary_rows, "metadata": metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
