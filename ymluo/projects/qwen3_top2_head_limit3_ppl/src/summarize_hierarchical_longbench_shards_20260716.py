from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sharded hierarchical LongBench result files."
    )
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--expected_tasks", type=int, default=0)
    parser.add_argument("--expected_samples_per_method", type=int, default=0)
    parser.add_argument(
        "--auto_gate_prompt_tokens",
        type=int,
        default=0,
        help="If positive, add an auto_length_gate method: Full below the threshold.",
    )
    return parser.parse_args()


def read_rows(pattern: str) -> list[dict[str, str]]:
    merged: dict[tuple[str, str, str], dict[str, str]] = {}
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matched {pattern!r}")
    for name in paths:
        with Path(name).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["task"], row["sample_id"], row["method"])
                if key in merged:
                    raise ValueError(f"duplicate method row {key} in {name}")
                merged[key] = row
    return sorted(
        merged.values(), key=lambda row: (row["task"], row["sample_id"], row["method"])
    )


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def add_auto_length_gate(
    rows: list[dict[str, str]], threshold: int
) -> list[dict[str, str]]:
    if threshold <= 0:
        return rows
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["task"], row["sample_id"]), {})[
            row["method"]
        ] = row
    gated = list(rows)
    for key, methods in grouped.items():
        full = methods.get("full_kv")
        sparse = methods.get("hierarchical_pca_perhead")
        if full is None:
            raise ValueError(f"{key}: auto length gate requires FullKV")
        prompt_tokens = int(full["prompt_tokens"])
        selected = full if prompt_tokens < threshold or sparse is None else sparse
        synthetic = dict(selected)
        synthetic["method"] = "auto_length_gate"
        gated.append(synthetic)
    return gated


def summarize(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_task: list[dict[str, Any]] = []
    overall: list[dict[str, Any]] = []
    methods = sorted({row["method"] for row in rows})
    tasks = sorted({row["task"] for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        task_rows: list[dict[str, Any]] = []
        for task in tasks:
            subset = [row for row in method_rows if row["task"] == task]
            if not subset:
                continue
            item = {
                "task": task,
                "method": method,
                "samples": len(subset),
                "score": mean(subset, "score"),
                "mean_kv_ratio": mean(subset, "kv_ratio"),
                "mean_cache_hit_rate": mean(
                    [row for row in subset if row["cache_hit_rate"]],
                    "cache_hit_rate",
                )
                if any(row["cache_hit_rate"] for row in subset)
                else None,
                "mean_prefill_seconds": mean(subset, "prefill_seconds"),
                "mean_conversion_seconds": mean(subset, "conversion_seconds"),
                "mean_query_seconds": mean(subset, "query_seconds"),
                "mean_decode_seconds": mean(subset, "decode_seconds"),
                "mean_online_seconds": mean(subset, "online_seconds"),
                "mean_total_seconds": mean(subset, "total_seconds"),
            }
            by_task.append(item)
            task_rows.append(item)
        overall.append(
            {
                "task": "ALL",
                "method": method,
                "samples": len(method_rows),
                "tasks": len(task_rows),
                "macro_score": sum(row["score"] for row in task_rows) / len(task_rows),
                "mean_kv_ratio": mean(method_rows, "kv_ratio"),
                "mean_prefill_seconds": mean(method_rows, "prefill_seconds"),
                "mean_conversion_seconds": mean(method_rows, "conversion_seconds"),
                "mean_query_seconds": mean(method_rows, "query_seconds"),
                "mean_decode_seconds": mean(method_rows, "decode_seconds"),
                "mean_online_seconds": mean(method_rows, "online_seconds"),
                "mean_total_seconds": mean(method_rows, "total_seconds"),
            }
        )
    return by_task, overall


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input_glob)
    rows = add_auto_length_gate(rows, args.auto_gate_prompt_tokens)
    by_task, overall = summarize(rows)
    if args.expected_tasks > 0:
        actual_tasks = len({row["task"] for row in rows})
        if actual_tasks != args.expected_tasks:
            raise ValueError(
                f"expected {args.expected_tasks} tasks, found {actual_tasks}"
            )
    if args.expected_samples_per_method > 0:
        for item in overall:
            if item["samples"] != args.expected_samples_per_method:
                raise ValueError(
                    f"{item['method']} has {item['samples']} samples; expected "
                    f"{args.expected_samples_per_method}"
                )
    write_csv(args.output_dir / "sample_results.csv", rows)
    write_csv(args.output_dir / "summary_by_task.csv", by_task)
    write_csv(args.output_dir / "summary_overall.csv", overall)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {"overall": overall, "by_task": by_task},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
