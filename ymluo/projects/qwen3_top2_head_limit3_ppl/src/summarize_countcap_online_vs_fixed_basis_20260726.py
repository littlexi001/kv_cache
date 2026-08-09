from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_rows(pattern: str) -> list[dict[str, str]]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def empirical_quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stratified_paired_macro_bootstrap(
    scores_by_task: dict[str, list[tuple[float, float]]],
    *,
    seed: int,
    replicates: int = 10_000,
) -> tuple[float, float]:
    """Bootstrap fixed-minus-online Macro while preserving task weights."""
    rng = random.Random(seed)
    deltas = []
    for _ in range(replicates):
        task_deltas = []
        for task in sorted(scores_by_task):
            pairs = scores_by_task[task]
            sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            task_deltas.append(
                mean(fixed - online for online, fixed in sampled)
            )
        deltas.append(mean(task_deltas))
    return (
        empirical_quantile(deltas, 0.025),
        empirical_quantile(deltas, 0.975),
    )


def summarize_model(
    rows: list[dict[str, str]],
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keyed: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        keyed[(row["task"], row["sample_id"])][row["method"]] = row
    expected = {"online", "fixed"}
    if not keyed or any(set(pair) != expected for pair in keyed.values()):
        raise ValueError(f"{model} rows are not strict online/fixed pairs")

    scores_by_task: dict[str, list[tuple[float, float]]] = {}
    task_rows = []
    prediction_matches = []
    for task in sorted({task for task, _ in keyed}):
        pairs = [
            pair
            for (pair_task, _), pair in keyed.items()
            if pair_task == task
        ]
        paired_scores = [
            (
                float(pair["online"]["score"]),
                float(pair["fixed"]["score"]),
            )
            for pair in pairs
        ]
        scores_by_task[task] = paired_scores
        online_score = mean(online for online, _ in paired_scores)
        fixed_score = mean(fixed for _, fixed in paired_scores)
        matches = [
            pair["online"]["prediction"] == pair["fixed"]["prediction"]
            for pair in pairs
        ]
        prediction_matches.extend(matches)
        task_rows.append(
            {
                "model": model,
                "task": task,
                "samples": len(pairs),
                "online_score": online_score,
                "fixed_score": fixed_score,
                "fixed_relative": (
                    fixed_score / online_score if online_score else 0.0
                ),
                "prediction_agreement": sum(matches) / len(matches),
            }
        )

    online_macro = mean(row["online_score"] for row in task_rows)
    fixed_macro = mean(row["fixed_score"] for row in task_rows)
    bootstrap_low, bootstrap_high = stratified_paired_macro_bootstrap(
        scores_by_task,
        seed=20260726 + sum(ord(character) for character in model),
    )
    pairs = list(keyed.values())
    overall = [
        {
            "model": model,
            "tasks": len(task_rows),
            "paired_samples": len(pairs),
            "online_macro": online_macro,
            "fixed_macro": fixed_macro,
            "fixed_minus_online_macro": fixed_macro - online_macro,
            "fixed_minus_online_macro_ci95_low": bootstrap_low,
            "fixed_minus_online_macro_ci95_high": bootstrap_high,
            "fixed_relative": (
                fixed_macro / online_macro if online_macro else 0.0
            ),
            "prediction_agreement": (
                sum(prediction_matches) / len(prediction_matches)
            ),
            "online_index_build_seconds": mean(
                float(pair["online"]["index_build_seconds"])
                for pair in pairs
            ),
            "fixed_index_build_seconds": mean(
                float(pair["fixed"]["index_build_seconds"])
                for pair in pairs
            ),
            "fixed_index_build_speedup": mean(
                float(pair["online"]["index_build_seconds"])
                / float(pair["fixed"]["index_build_seconds"])
                for pair in pairs
                if float(pair["fixed"]["index_build_seconds"]) > 0.0
            ),
            "fixed_total_speedup": mean(
                float(pair["online"]["total_seconds"])
                / float(pair["fixed"]["total_seconds"])
                for pair in pairs
                if float(pair["fixed"]["total_seconds"]) > 0.0
            ),
        }
    ]
    return overall, task_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_glob", required=True)
    parser.add_argument("--qwen_glob", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    overall = []
    tasks = []
    for model, pattern in (
        ("Llama-3.1-8B-Instruct", args.llama_glob),
        ("Qwen3-4B-Instruct", args.qwen_glob),
    ):
        model_overall, model_tasks = summarize_model(
            read_rows(pattern),
            model,
        )
        overall.extend(model_overall)
        tasks.extend(model_tasks)
    payload = {"overall": overall, "by_task": tasks}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "overall.csv", overall)
    write_csv(args.output_dir / "by_task.csv", tasks)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
