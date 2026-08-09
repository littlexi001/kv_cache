#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import train_source_router_v437_20260712 as v437
from run_controlled_public_kv_benchmark_v1 import infer_operator_contract
from train_operator_contract_router_v460_20260713 import CONTRACT_BY_TASK


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(v437.ROOT))
    parser.add_argument(
        "--longbench-zip",
        default="outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/longbench_data.zip",
    )
    parser.add_argument("--max-samples-per-task", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_v461_operator_contract_schema_20260713")
    args = parser.parse_args()

    root = Path(args.root)
    examples = v437.load_longbench_examples(
        root / args.longbench_zip,
        list(CONTRACT_BY_TASK),
        args.max_samples_per_task,
        1234,
    )

    prediction_rows: list[dict[str, Any]] = []
    for example in examples:
        action, confidence, reason = infer_operator_contract(example)
        target = CONTRACT_BY_TASK[example.task]
        prediction_rows.append(
            {
                "task": example.task,
                "sample_id": example.sample_id,
                "target": target,
                "action": action,
                "correct": int(action == target),
                "confidence": confidence,
                "reason": reason,
                "query": example.query.replace("\n", " ")[:240],
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for task in ["overall", *CONTRACT_BY_TASK]:
        rows = prediction_rows if task == "overall" else [row for row in prediction_rows if row["task"] == task]
        confusion = Counter(f"{row['target']}->{row['action']}" for row in rows)
        summary_rows.append(
            {
                "task": task,
                "samples": len(rows),
                "accuracy": sum(int(row["correct"]) for row in rows) / max(1, len(rows)),
                "errors": sum(not int(row["correct"]) for row in rows),
                "confusion": json.dumps(dict(confusion), sort_keys=True),
            }
        )

    output_dir = root / args.output_dir
    write_csv(output_dir / "schema_predictions.csv", prediction_rows)
    write_csv(output_dir / "schema_errors.csv", [row for row in prediction_rows if not int(row["correct"])])
    write_csv(output_dir / "schema_summary.csv", summary_rows)
    (output_dir / "schema_summary.json").write_text(
        json.dumps(
            {
                "router": "request_schema_v1",
                "uses_task_name": False,
                "uses_prompt_template": False,
                "rows": summary_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for row in summary_rows:
        print(f"{row['task']:24s} n={row['samples']:4d} accuracy={row['accuracy']:.4f} errors={row['errors']}")


if __name__ == "__main__":
    main()
