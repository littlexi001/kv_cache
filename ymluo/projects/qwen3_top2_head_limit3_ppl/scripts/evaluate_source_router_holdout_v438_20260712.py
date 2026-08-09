#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import train_policy_action_planner_v378_20260712 as v378
import train_source_router_v437_20260712 as v437
from run_controlled_public_kv_benchmark_v1 import Example, source_router_example_text


def content_only_text(example: Example) -> str:
    context = example.context
    if len(context) > 8000:
        context = context[:4000] + "\n...\n" + context[-4000:]
    return "\n".join(["QUERY:", example.query, "CONTEXT:", context])


def fit_router(
    examples: list[Example],
    labels_by_key: dict[tuple[str, str], str],
    text_fn: Callable[[Example], str],
) -> Any:
    labels = [labels_by_key[(example.task, example.sample_id)] for example in examples]
    unique = sorted(set(labels))
    if len(unique) == 1:
        return {"constant": unique[0]}

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.98,
                    max_features=15000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b[\w\-]+\b",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=3.0,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="liblinear",
                    random_state=438,
                ),
            ),
        ]
    )
    model.fit([text_fn(example) for example in examples], labels)
    return model


def predict(model: Any, text: str) -> tuple[str, float]:
    if isinstance(model, dict) and "constant" in model:
        return str(model["constant"]), 1.0
    return v437.predict_action(model, text)


def materialize(
    examples: list[Example],
    model: Any,
    threshold: float,
    text_fn: Callable[[Example], str],
    labels_by_key: dict[tuple[str, str], str],
    base_table: dict[tuple[str, str], dict[str, str]],
    source_tables: dict[str, dict[tuple[str, str], dict[str, str]]],
    full_table: dict[tuple[str, str], dict[str, str]],
    prediction_cache: dict[tuple[str, str], tuple[str, float]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        key = (example.task, example.sample_id)
        if prediction_cache is not None and key in prediction_cache:
            pred_source, confidence = prediction_cache[key]
        else:
            pred_source, confidence = predict(model, text_fn(example))
        selected_source = pred_source
        if selected_source != "base" and confidence < threshold:
            selected_source = "base"
        policy_row = source_tables.get(selected_source, base_table).get(key) or base_table.get(key)
        full_row = full_table.get(key)
        if policy_row is None or full_row is None:
            continue
        rows.append(
            {
                "task": example.task,
                "family": v378.task_family(example.task),
                "sample_id": example.sample_id,
                "teacher_source": labels_by_key[key],
                "pred_source": pred_source,
                "selected_source": selected_source,
                "confidence": confidence,
                "full_score": v437.fnum(full_row, "score"),
                "score": v437.fnum(policy_row, "score"),
                "kv": v437.fnum(policy_row, "keep_fraction"),
                "online": v437.fnum(policy_row, "online_seconds"),
            }
        )
    return rows


def choose_threshold(
    examples: list[Example],
    model: Any,
    text_fn: Callable[[Example], str],
    labels_by_key: dict[tuple[str, str], str],
    base_table: dict[tuple[str, str], dict[str, str]],
    source_tables: dict[str, dict[tuple[str, str], dict[str, str]]],
    full_table: dict[tuple[str, str], dict[str, str]],
    full_online: float,
    quality_ratio: float,
    kv_limit: float,
    speed_min: float,
) -> tuple[float, dict[str, Any]]:
    prediction_cache = {
        (example.task, example.sample_id): predict(model, text_fn(example)) for example in examples
    }
    trials: list[dict[str, Any]] = []
    for threshold in [round(item / 100, 2) for item in range(101)] + [1.01]:
        rows = materialize(
            examples,
            model,
            threshold,
            text_fn,
            labels_by_key,
            base_table,
            source_tables,
            full_table,
            prediction_cache,
        )
        summary = next(
            row for row in v437.summarize_predictions(rows, full_online, "calibration") if row["task"] == "ALL"
        )
        summary = {
            **summary,
            "threshold": threshold,
            "feasible": int(
                float(summary["vs_full"]) >= quality_ratio
                and float(summary["kv"]) <= kv_limit
                and float(summary["speed_vs_full"]) >= speed_min
            ),
        }
        trials.append(summary)
    feasible = [row for row in trials if int(row["feasible"])]
    if feasible:
        selected = max(feasible, key=lambda row: (float(row["score"]), -float(row["kv"])))
    else:
        selected = max(trials, key=lambda row: (float(row["vs_full"]), -float(row["kv"])))
    return float(selected["threshold"]), selected


def aggregate(rows: list[dict[str, Any]], full_online: float, **tags: Any) -> dict[str, Any]:
    summary = next(row for row in v437.summarize_predictions(rows, full_online, "holdout") if row["task"] == "ALL")
    return {**tags, **summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(v437.ROOT))
    parser.add_argument(
        "--teacher-policy",
        default="configs/riskkv_task_policy_v428_v427_plus_repobench_20260712.json",
    )
    parser.add_argument(
        "--base-policy",
        default="configs/riskkv_task_policy_v417_expanded_knapsack030_20260712.json",
    )
    parser.add_argument(
        "--longbench-zip",
        default="outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/longbench_data.zip",
    )
    parser.add_argument("--full-results", default="outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv")
    parser.add_argument("--max-samples-per-task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--quality-ratio", type=float, default=0.95)
    parser.add_argument("--kv-limit", type=float, default=0.10)
    parser.add_argument("--speed-min", type=float, default=2.5)
    parser.add_argument("--full-online", type=float, default=3.0988)
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_v438_source_router_holdout_20260712")
    args = parser.parse_args()

    root = Path(args.root)
    teacher_path = v437.resolve_policy_path(root, root, args.teacher_policy)
    base_policy_path = v437.resolve_policy_path(root, root, args.base_policy)
    base_policy = v437.rel_to_root(root, base_policy_path)
    examples = v437.load_longbench_examples(
        root / args.longbench_zip,
        v437.TASKS,
        args.max_samples_per_task,
        args.seed,
    )
    labels_by_key = {
        (example.task, example.sample_id): v437.source_label_for_task(
            root, teacher_path, example.task, base_policy_path
        )
        for example in examples
    }
    source_specs = v437.collect_source_specs(root, teacher_path, v437.TASKS, base_policy_path)
    cache: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    base_table = v437.table_for_policy(root, base_policy, cache)
    source_tables = {
        "base": base_table,
        **{label: v437.table_for_policy(root, spec, cache) for label, spec in source_specs.items()},
    }
    full_table = v437.frontier.read_table(root / args.full_results)

    text_modes: dict[str, Callable[[Example], str]] = {
        "full_prompt": source_router_example_text,
        "content_only": content_only_text,
    }
    holdout_specs = {
        "task": {task: {task} for task in v437.TASKS},
        "family": {
            family: {task for task in v437.TASKS if v378.task_family(task) == family}
            for family in sorted({v378.task_family(task) for task in v437.TASKS})
        },
    }

    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for text_mode, text_fn in text_modes.items():
        for holdout_type, groups in holdout_specs.items():
            mode_rows: list[dict[str, Any]] = []
            for holdout_name, holdout_tasks in groups.items():
                train_examples = [
                    example
                    for example in examples
                    if example.task not in holdout_tasks and v378.fold_for_key(example.task, example.sample_id) != 1
                ]
                calibration_examples = [
                    example
                    for example in examples
                    if example.task not in holdout_tasks and v378.fold_for_key(example.task, example.sample_id) == 1
                ]
                holdout_examples = [example for example in examples if example.task in holdout_tasks]
                model = fit_router(train_examples, labels_by_key, text_fn)
                threshold, calibration_summary = choose_threshold(
                    calibration_examples,
                    model,
                    text_fn,
                    labels_by_key,
                    base_table,
                    source_tables,
                    full_table,
                    args.full_online,
                    args.quality_ratio,
                    args.kv_limit,
                    args.speed_min,
                )
                rows = materialize(
                    holdout_examples,
                    model,
                    threshold,
                    text_fn,
                    labels_by_key,
                    base_table,
                    source_tables,
                    full_table,
                )
                for row in rows:
                    row.update(
                        {
                            "text_mode": text_mode,
                            "holdout_type": holdout_type,
                            "holdout": holdout_name,
                            "threshold": threshold,
                        }
                    )
                mode_rows.extend(rows)
                prediction_rows.extend(rows)
                summary_rows.append(
                    aggregate(
                        rows,
                        args.full_online,
                        text_mode=text_mode,
                        holdout_type=holdout_type,
                        holdout=holdout_name,
                        holdout_tasks=",".join(sorted(holdout_tasks)),
                        threshold=threshold,
                        calibration_vs_full=calibration_summary["vs_full"],
                        calibration_kv=calibration_summary["kv"],
                    )
                )
            summary_rows.append(
                aggregate(
                    mode_rows,
                    args.full_online,
                    text_mode=text_mode,
                    holdout_type=holdout_type,
                    holdout="ALL",
                    holdout_tasks="",
                    threshold="",
                    calibration_vs_full="",
                    calibration_kv="",
                )
            )

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    v437.write_csv(output_dir / "holdout_predictions.csv", prediction_rows)
    v437.write_csv(output_dir / "holdout_summary.csv", summary_rows)
    metadata = {
        "evaluation": "source_router_v438_strict_holdout",
        "teacher_policy": v437.rel_to_root(root, teacher_path),
        "base_policy": base_policy,
        "source_specs": source_specs,
        "max_samples_per_task": args.max_samples_per_task,
        "text_modes": list(text_modes),
        "holdout_types": list(holdout_specs),
        "task_families": v378.FAMILY_BY_TASK,
        "training_excludes_entire_holdout_group": True,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    key_rows = [row for row in summary_rows if row["holdout"] == "ALL"]
    print(output_dir)
    print(json.dumps(key_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
