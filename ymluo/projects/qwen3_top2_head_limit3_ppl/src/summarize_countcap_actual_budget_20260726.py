from __future__ import annotations

import argparse
import csv
import glob
import json
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


def summarize_model(
    rows: list[dict[str, str]],
    model: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError(f"{model} has no rows")
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("diagnostics_enabled", "").lower() not in {
            "true",
            "1",
        }:
            raise ValueError(f"{model} contains a row without diagnostics")
        by_task[row["task"]].append(row)

    def aggregate(subset: list[dict[str, str]]) -> dict[str, float]:
        return {
            "samples": len(subset),
            "prompt_tokens_mean": mean(
                float(row["prompt_tokens"]) for row in subset
            ),
            "target_fraction_mean": mean(
                float(row["configured_attention_fraction"])
                for row in subset
            ),
            "actual_fraction_mean": mean(
                float(row["attention_link_ratio"]) for row in subset
            ),
            "actual_fraction_p95_mean": mean(
                float(row["selected_history_fraction_p95"])
                for row in subset
            ),
            "actual_fraction_max": max(
                float(row["selected_history_fraction_max"])
                for row in subset
            ),
            "actual_count_mean": mean(
                float(row["selected_history_count_mean"])
                for row in subset
            ),
            "actual_count_p95_mean": mean(
                float(row["selected_history_count_p95"])
                for row in subset
            ),
            "actual_count_max": max(
                float(row["selected_history_count_max"])
                for row in subset
            ),
            "candidate_overflow_head_fraction_mean": mean(
                float(row["sampled_candidate_overflow_fraction"])
                for row in subset
            ),
            "sampled_quantile_fallback_rate_mean": mean(
                float(row["sampled_quantile_fallback"])
                for row in subset
            ),
        }

    tasks = [
        {"model": model, "task": task, **aggregate(subset)}
        for task, subset in sorted(by_task.items())
    ]
    overall = {
        "model": model,
        "tasks": len(tasks),
        **aggregate(rows),
    }
    return overall, tasks


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
    by_task = []
    for model, pattern in (
        ("Llama-3.1-8B-Instruct", args.llama_glob),
        ("Qwen3-4B-Instruct", args.qwen_glob),
    ):
        model_overall, model_tasks = summarize_model(
            read_rows(pattern),
            model,
        )
        overall.append(model_overall)
        by_task.extend(model_tasks)

    payload = {"overall": overall, "by_task": by_task}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "overall.csv", overall)
    write_csv(args.output_dir / "by_task.csv", by_task)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
