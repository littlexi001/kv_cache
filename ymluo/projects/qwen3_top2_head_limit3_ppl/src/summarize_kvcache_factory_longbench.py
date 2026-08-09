from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_controlled_public_kv_benchmark_v1 import LONG_BENCH_PROMPTS, score_prediction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize KVCache-Factory LongBench prediction JSON files.")
    parser.add_argument("--input_dir", required=True, help="Directory containing model_budget/task/method.json files.")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def normalize_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def load_prediction_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        return normalize_records(json.loads(text))
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(f"JSONL line {line_no}: {exc.msg}", exc.doc, exc.pos) from exc
            if isinstance(item, dict):
                records.append(item)
        return records


def pick_prediction(row: dict[str, Any]) -> str:
    for key in ("prediction", "pred", "output", "outputs", "response", "text"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def pick_answers(row: dict[str, Any]) -> list[str]:
    for key in ("answers", "answer", "ground_truth", "ground_truths"):
        value = row.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
    return []


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*/*/*.json")):
        model_budget = path.parent.parent.name
        task = path.parent.name
        method = path.stem
        if task not in LONG_BENCH_PROMPTS:
            continue
        match = re.search(r"_(\d+)$", model_budget)
        budget = int(match.group(1)) if match else ""
        if path.stat().st_size == 0:
            rows.append(
                {
                    "model_budget": model_budget,
                    "budget": budget,
                    "task": task,
                    "method": method,
                    "samples": 0,
                    "score": "",
                    "status": "EMPTY",
                    "path": str(path),
                }
            )
            continue
        try:
            records = load_prediction_records(path)
        except Exception as exc:
            rows.append(
                {
                    "model_budget": model_budget,
                    "budget": budget,
                    "task": task,
                    "method": method,
                    "samples": 0,
                    "score": "",
                    "status": f"READ_FAILED:{type(exc).__name__}",
                    "path": str(path),
                }
            )
            continue
        scores: list[float] = []
        for record in records:
            prediction = pick_prediction(record)
            answers = pick_answers(record)
            if not answers:
                continue
            scores.append(
                score_prediction(
                    str(LONG_BENCH_PROMPTS[task]["metric"]),
                    prediction,
                    answers,
                    task=task,
                )
            )
        rows.append(
            {
                "model_budget": model_budget,
                "budget": budget,
                "task": task,
                "method": method,
                "samples": len(scores),
                "score": sum(scores) / len(scores) if scores else "",
                "status": "OK" if scores else "NO_SCORE",
                "path": str(path),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "OK":
            grouped[str(row["method"])].append(row)
    for method, subset in sorted(grouped.items()):
        values = [float(row["score"]) for row in subset]
        rows.append(
            {
                "model_budget": "ALL",
                "budget": "",
                "task": "ALL",
                "method": method,
                "samples": sum(int(row["samples"]) for row in subset),
                "score": sum(values) / len(values) if values else "",
                "status": "OK",
                "path": "",
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model_budget", "budget", "task", "method", "samples", "score", "status", "path"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
