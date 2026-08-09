#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SRC_DIR = ROOT / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train_frontier_mode_router_v421_20260712 as frontier  # noqa: E402
import train_policy_action_planner_v378_20260712 as v378  # noqa: E402
from run_controlled_public_kv_benchmark_v1 import (  # noqa: E402
    Example,
    LONG_BENCH_PROMPTS,
    source_router_example_text,
)


TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

RESULT_TABLE_OVERRIDES = {
    "riskkv_task_policy_v417_expanded_knapsack030_20260712.json": (
        "outputs/riskkv_v19_v417_expanded_knapsack030_20260712_expanded030_v417_m100_bDyn_pDyn/task_results.csv"
    ),
    "riskkv_task_policy_v421_frontier_router035_20260712.json": (
        "outputs/riskkv_v19_v421_frontier_router035_20260712_frontier_v421_m100_bDyn_pDyn/task_results.csv"
    ),
    "riskkv_task_policy_v427_v417_source_v421_winners_20260712.json": (
        "outputs/riskkv_v19_v427_v417_source_v421_winners_20260712_v427_m100_bDyn_pDyn/task_results.csv"
    ),
    "riskkv_task_policy_v428_v427_plus_repobench_20260712.json": (
        "outputs/riskkv_v19_v428_v427_plus_repobench_20260712_v428_m100_bDyn_pDyn/task_results.csv"
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_table(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["task"], row["sample_id"]): row for row in csv.DictReader(handle)}


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


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def fnum(row: dict[str, str] | None, key: str) -> float:
    if row is None:
        return 0.0
    try:
        return float(row.get(key, "") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_policy_path(root: Path, base_dir: Path, spec: str) -> Path:
    path = Path(spec)
    if path.is_absolute():
        return path
    if (root / path).exists():
        return root / path
    return base_dir / path


def rel_to_root(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def source_alias(path: Path) -> str:
    match = re.search(r"_v(\d+)_", path.name)
    if match:
        return f"v{match.group(1)}"
    return path.stem.replace("riskkv_task_policy_", "")


def table_for_policy(
    root: Path,
    policy_rel: str,
    cache: dict[str, dict[tuple[str, str], dict[str, str]]],
) -> dict[tuple[str, str], dict[str, str]]:
    policy_name = Path(policy_rel).name
    override = RESULT_TABLE_OVERRIDES.get(policy_name)
    if override and (root / override).exists():
        key = f"override:{policy_name}"
        if key not in cache:
            cache[key] = read_table(root / override)
        return cache[key]
    return frontier.synthesize_action_rows(root, policy_rel, cache)



def same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def source_label_for_task(root: Path, config_path: Path, task: str, stop_policy: Path | None = None) -> str:
    if stop_policy is not None and same_path(config_path, stop_policy):
        return "base"
    payload = load_json(config_path)
    label = "base"
    if "__extends" in payload:
        parent = resolve_policy_path(root, config_path.parent, str(payload["__extends"]))
        label = source_label_for_task(root, parent, task, stop_policy)
    task_sources = payload.get("__task_sources", {})
    if isinstance(task_sources, dict) and task in task_sources:
        entry = task_sources[task]
        policy_spec = entry if isinstance(entry, str) else str(entry.get("policy", ""))
        if policy_spec:
            source_path = resolve_policy_path(root, config_path.parent, policy_spec)
            label = source_alias(source_path)
    return label


def collect_source_specs(root: Path, config_path: Path, tasks: list[str], stop_policy: Path | None = None) -> dict[str, str]:
    if stop_policy is not None and same_path(config_path, stop_policy):
        return {}
    specs: dict[str, str] = {}
    payload = load_json(config_path)
    if "__extends" in payload:
        parent = resolve_policy_path(root, config_path.parent, str(payload["__extends"]))
        specs.update(collect_source_specs(root, parent, tasks, stop_policy))
    task_sources = payload.get("__task_sources", {})
    if isinstance(task_sources, dict):
        for task in tasks:
            if task not in task_sources:
                continue
            entry = task_sources[task]
            policy_spec = entry if isinstance(entry, str) else str(entry.get("policy", ""))
            if not policy_spec:
                continue
            source_path = resolve_policy_path(root, config_path.parent, policy_spec)
            specs[source_alias(source_path)] = rel_to_root(root, source_path)
    return specs


def load_longbench_examples(zip_path: Path, tasks: list[str], max_samples: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for task in tasks:
            info = LONG_BENCH_PROMPTS[task]
            name = f"data/{task}.jsonl"
            if name not in names:
                raise FileNotFoundError(f"{name} not found in {zip_path}")
            rows = [json.loads(line) for line in archive.open(name).read().decode("utf-8").splitlines() if line.strip()]
            if len(rows) > max_samples:
                rows = rows[:max_samples]
            rng.shuffle(rows)
            rows = sorted(rows, key=lambda row: str(row.get("_id", "")))[:max_samples]
            for row in rows:
                examples.append(
                    Example(
                        benchmark="longbench",
                        task=task,
                        sample_id=str(row.get("_id", len(examples))),
                        context=str(row["context"]),
                        query=str(row["input"]),
                        answers=[str(answer) for answer in row["answers"]],
                        prefix_template=str(info["prefix"]),
                        suffix_template=str(info["suffix"]),
                        metric=str(info["metric"]),
                        max_new_tokens=int(info["max_new_tokens"]),
                        length=int(row.get("length", 0) or 0),
                        all_classes=[str(item) for item in (row.get("all_classes") or [])],
                        no_chat=bool(info.get("no_chat", False)),
                    )
                )
    return examples


def predict_action(model: Any, text: str) -> tuple[str, float]:
    action = str(model.predict([text])[0])
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])
        if len(probabilities) and len(probabilities[0]):
            confidence = float(max(probabilities[0]))
    return action, confidence


def summarize_predictions(
    rows: list[dict[str, Any]],
    full_online: float,
    split: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped["ALL"].append(row)
        grouped[str(row["task"])].append(row)
    out: list[dict[str, Any]] = []
    for task, subset in sorted(grouped.items(), key=lambda item: (item[0] != "ALL", item[0])):
        full_score = mean([float(row["full_score"]) for row in subset])
        score = mean([float(row["score"]) for row in subset])
        kv = mean([float(row["kv"]) for row in subset])
        online = mean([float(row["online"]) for row in subset])
        out.append(
            {
                "split": split,
                "task": task,
                "samples": len(subset),
                "full_score": full_score,
                "score": score,
                "vs_full": score / max(1e-9, full_score),
                "kv": kv,
                "online": online,
                "speed_vs_full": full_online / max(1e-9, online),
                "label_accuracy": mean([1.0 if row["pred_source"] == row["teacher_source"] else 0.0 for row in subset]),
                "fallback_rate": mean([1.0 if row["selected_source"] == "base" else 0.0 for row in subset]),
                "mean_confidence": mean([float(row["confidence"]) for row in subset]),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--teacher-policy", default="configs/riskkv_task_policy_v428_v427_plus_repobench_20260712.json")
    parser.add_argument("--base-policy", default="configs/riskkv_task_policy_v417_expanded_knapsack030_20260712.json")
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
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_v437_source_router_v428_20260712")
    parser.add_argument("--config-out", default="configs/riskkv_task_policy_v437_source_router_v428_20260712.json")
    args = parser.parse_args()

    root = Path(args.root)
    teacher_path = resolve_policy_path(root, root, args.teacher_policy)
    base_policy_path = resolve_policy_path(root, root, args.base_policy)
    base_policy = rel_to_root(root, base_policy_path)
    tasks = TASKS
    examples = load_longbench_examples(root / args.longbench_zip, tasks, args.max_samples_per_task, args.seed)
    labels_by_key = {
        (example.task, example.sample_id): source_label_for_task(root, teacher_path, example.task, base_policy_path)
        for example in examples
    }
    source_specs = collect_source_specs(root, teacher_path, tasks, base_policy_path)

    train_examples = [ex for ex in examples if v378.fold_for_key(ex.task, ex.sample_id) not in {0, 1}]
    cal_examples = [ex for ex in examples if v378.fold_for_key(ex.task, ex.sample_id) == 1]
    test_examples = [ex for ex in examples if v378.fold_for_key(ex.task, ex.sample_id) == 0]

    labels = sorted(set(labels_by_key.values()))
    if len(labels) == 1:
        model: Any = {"constant": labels[0]}
    else:
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
                        random_state=437,
                    ),
                ),
            ]
        )
        model.fit(
            [source_router_example_text(example) for example in train_examples],
            [labels_by_key[(example.task, example.sample_id)] for example in train_examples],
        )

    cache: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    base_table = table_for_policy(root, base_policy, cache)
    source_tables = {
        "base": base_table,
        **{label: table_for_policy(root, spec, cache) for label, spec in source_specs.items()},
    }
    full_table = frontier.read_table(root / args.full_results)

    def materialize(split: str, split_examples: list[Example], threshold: float) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for example in split_examples:
            key = (example.task, example.sample_id)
            teacher_source = labels_by_key[key]
            if isinstance(model, dict) and "constant" in model:
                pred_source, confidence = str(model["constant"]), 1.0
            else:
                pred_source, confidence = predict_action(model, source_router_example_text(example))
            selected_source = pred_source
            if selected_source != "base" and confidence < threshold:
                selected_source = "base"
            table = source_tables.get(selected_source, base_table)
            row = table.get(key) or base_table.get(key)
            full = full_table.get(key)
            if row is None or full is None:
                continue
            out.append(
                {
                    "split": split,
                    "task": example.task,
                    "sample_id": example.sample_id,
                    "fold": v378.fold_for_key(example.task, example.sample_id),
                    "teacher_source": teacher_source,
                    "pred_source": pred_source,
                    "selected_source": selected_source,
                    "confidence": confidence,
                    "full_score": fnum(full, "score"),
                    "score": fnum(row, "score"),
                    "kv": fnum(row, "keep_fraction"),
                    "online": fnum(row, "online_seconds"),
                }
            )
        return out

    threshold_trials: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    for threshold in [round(item / 100, 2) for item in range(0, 101)] + [1.01]:
        rows = materialize("calibration", cal_examples, threshold)
        summary = next(row for row in summarize_predictions(rows, args.full_online, "calibration") if row["task"] == "ALL")
        trial = {
            **summary,
            "threshold": threshold,
            "feasible": int(
                float(summary["vs_full"]) >= args.quality_ratio
                and float(summary["kv"]) <= args.kv_limit
                and float(summary["speed_vs_full"]) >= args.speed_min
            ),
        }
        threshold_trials.append(trial)
        if int(trial["feasible"]):
            feasible.append(trial)
    if feasible:
        selected = max(feasible, key=lambda row: (float(row["score"]), -float(row["kv"]), float(row["speed_vs_full"])))
    else:
        selected = max(threshold_trials, key=lambda row: (float(row["vs_full"]), -float(row["kv"]), float(row["speed_vs_full"])))
    threshold = float(selected["threshold"])

    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for split, split_examples in [
        ("train", train_examples),
        ("calibration", cal_examples),
        ("test", test_examples),
        ("all", examples),
    ]:
        rows = materialize(split, split_examples, threshold)
        prediction_rows.extend(rows)
        summary_rows.extend(summarize_predictions(rows, args.full_online, split))

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "router_type": "source_router_v437",
        "input_type": "example_text",
        "teacher_policy": rel_to_root(root, teacher_path),
        "base_policy": base_policy,
        "source_specs": source_specs,
        "source_labels": labels,
        "confidence_fallback_threshold": threshold,
        "default_source": "base",
        "folds": 5,
        "test_fold": 0,
        "calibration_fold": 1,
        "uses_task_name_as_feature": False,
    }
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"model": model, "metadata": metadata}, handle)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "threshold_sweep.csv", threshold_trials)
    write_csv(output_dir / "source_router_predictions.csv", prediction_rows)
    write_csv(output_dir / "source_router_summary.csv", summary_rows)

    config = {
        "__extends": Path(base_policy).name,
        "__comment": (
            "v437: source-router distilled from v428. The router predicts the source policy from raw "
            "prompt/context/query text; it does not use the benchmark task name as a model feature."
        ),
        "__source_router": {
            "model_path": rel_to_root(root, output_dir / "model.pkl"),
            "confidence_threshold": threshold,
            "default_source": "base",
            "sources": {label: {"policy": spec} for label, spec in source_specs.items()},
        },
    }
    config_path = root / args.config_out
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    all_summary = next(row for row in summary_rows if row["split"] == "all" and row["task"] == "ALL")
    cal_summary = next(row for row in summary_rows if row["split"] == "calibration" and row["task"] == "ALL")
    test_summary = next(row for row in summary_rows if row["split"] == "test" and row["task"] == "ALL")
    print(output_dir)
    print(config_path)
    print(json.dumps({"all": all_summary, "calibration": cal_summary, "test": test_summary, "metadata": metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
