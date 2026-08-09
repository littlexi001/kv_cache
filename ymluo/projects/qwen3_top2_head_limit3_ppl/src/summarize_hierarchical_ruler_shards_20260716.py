from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import summarize_hierarchical_longbench_shards_20260716 as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sharded hierarchical RULER result files."
    )
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--expected_task_lengths", type=int, default=0)
    parser.add_argument("--expected_samples_per_method", type=int, default=0)
    parser.add_argument("--auto_gate_requested_length", type=int, default=0)
    return parser.parse_args()


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def timing_summary(
    rows: list[dict[str, str]], group_name: str, group_value: str
) -> dict[str, Any]:
    task_scores = []
    for task in sorted({row["base_task"] for row in rows}):
        subset = [row for row in rows if row["base_task"] == task]
        task_scores.append(mean(subset, "score"))
    return {
        "group": group_name,
        "value": group_value,
        "method": rows[0]["method"],
        "samples": len(rows),
        "tasks": len(task_scores),
        "macro_score": sum(task_scores) / len(task_scores),
        "mean_prompt_tokens": mean(rows, "prompt_tokens"),
        "mean_kv_ratio": mean(rows, "kv_ratio"),
        "mean_prefill_seconds": mean(rows, "prefill_seconds"),
        "mean_conversion_seconds": mean(rows, "conversion_seconds"),
        "mean_query_seconds": mean(rows, "query_seconds"),
        "mean_decode_seconds": mean(rows, "decode_seconds"),
        "mean_online_seconds": mean(rows, "online_seconds"),
        "mean_total_seconds": mean(rows, "total_seconds"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_length_gated_rows(
    rows: list[dict[str, str]], threshold: int
) -> list[dict[str, str]]:
    if threshold <= 0:
        return []
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row["task"], row["sample_id"])
        grouped.setdefault(key, {})[row["method"]] = row
    gated: list[dict[str, str]] = []
    policy_name = f"hierarchical_length_gate_{threshold}"
    for key, methods in sorted(grouped.items()):
        missing = {"full_kv", "hierarchical_pca_perhead"} - set(methods)
        if missing:
            raise ValueError(f"{key} is missing paired methods: {sorted(missing)}")
        requested_length = int(methods["full_kv"]["requested_length"])
        source = (
            methods["hierarchical_pca_perhead"]
            if requested_length >= threshold
            else methods["full_kv"]
        )
        selected = dict(source)
        selected["method"] = policy_name
        gated.append(selected)
    return gated


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = base.read_rows(args.input_glob)
    task_lengths = sorted({row["task"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    if args.expected_task_lengths > 0 and len(task_lengths) != args.expected_task_lengths:
        raise ValueError(
            f"expected {args.expected_task_lengths} task-length groups, "
            f"found {len(task_lengths)}"
        )

    by_task_length, overall = base.summarize(rows)
    by_length: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        if (
            args.expected_samples_per_method > 0
            and len(method_rows) != args.expected_samples_per_method
        ):
            raise ValueError(
                f"{method} has {len(method_rows)} samples; expected "
                f"{args.expected_samples_per_method}"
            )
        for length in sorted({row["requested_length"] for row in method_rows}, key=int):
            subset = [
                row for row in method_rows if row["requested_length"] == length
            ]
            by_length.append(timing_summary(subset, "length", length))

    gated_rows = build_length_gated_rows(rows, args.auto_gate_requested_length)
    gated_by_task_length: list[dict[str, Any]] = []
    gated_overall: list[dict[str, Any]] = []
    gated_by_length: list[dict[str, Any]] = []
    if gated_rows:
        gated_by_task_length, gated_overall = base.summarize(gated_rows)
        for length in sorted({row["requested_length"] for row in gated_rows}, key=int):
            subset = [row for row in gated_rows if row["requested_length"] == length]
            gated_by_length.append(timing_summary(subset, "length", length))

    payload = {
        "overall": overall,
        "by_length": by_length,
        "by_task_length": by_task_length,
        "gated_overall": gated_overall,
        "gated_by_length": gated_by_length,
        "gated_by_task_length": gated_by_task_length,
    }
    write_csv(args.output_dir / "sample_results.csv", rows)
    write_csv(args.output_dir / "summary_overall.csv", overall)
    write_csv(args.output_dir / "summary_by_length.csv", by_length)
    write_csv(args.output_dir / "summary_by_task_length.csv", by_task_length)
    write_csv(args.output_dir / "gated_sample_results.csv", gated_rows)
    write_csv(args.output_dir / "gated_summary_overall.csv", gated_overall)
    write_csv(args.output_dir / "gated_summary_by_length.csv", gated_by_length)
    write_csv(
        args.output_dir / "gated_summary_by_task_length.csv", gated_by_task_length
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "overall": overall,
                "by_length": by_length,
                "gated_overall": gated_overall,
                "gated_by_length": gated_by_length,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
