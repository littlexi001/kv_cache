from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


RULER_TASKS = {
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "cwe",
    "fwe",
    "vt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize KVCache-Factory RULER outputs. The score is the same "
            "string_match_all formula used by KVCache-Factory eval_ruler.py, "
            "implemented locally to avoid optional metrics.py dependencies."
        )
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def normalize_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        return normalize_records(json.loads(text))
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.extend(normalize_records(json.loads(line)))
        return rows


def string_match_all(predictions: list[str], references: list[list[str]]) -> float:
    if not predictions:
        return 0.0
    score = 0.0
    for pred, ref in zip(predictions, references):
        if not ref:
            continue
        pred_lower = pred.lower()
        score += sum(1.0 if str(answer).lower() in pred_lower else 0.0 for answer in ref) / len(ref)
    return round(score / len(predictions) * 100.0, 2)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*/*/*/*.json")):
        model_budget = path.parents[2].name
        context_length = path.parents[1].name
        task = path.parent.name
        method = path.stem
        if task not in RULER_TASKS:
            continue
        match = re.search(r"_(\d+)$", model_budget)
        budget = int(match.group(1)) if match else ""
        try:
            records = read_records(path)
            predictions = [str(row.get("pred", "")) for row in records]
            references = []
            for row in records:
                value = row.get("answers", [])
                if isinstance(value, list):
                    references.append([str(item) for item in value])
                else:
                    references.append([str(value)])
            score = string_match_all(predictions, references)
            status = "OK" if records else "NO_RECORDS"
        except Exception as exc:
            records = []
            score = ""
            status = f"READ_FAILED:{type(exc).__name__}"
        rows.append(
            {
                "model_budget": model_budget,
                "budget": budget,
                "context_length": context_length,
                "task": task,
                "method": method,
                "samples": len(records),
                "score": score,
                "status": status,
                "path": str(path),
            }
        )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "OK":
            grouped[(str(row["model_budget"]), str(row["context_length"]), str(row["method"]))].append(row)
    for (model_budget, context_length, method), subset in sorted(grouped.items()):
        scores = [float(row["score"]) for row in subset]
        rows.append(
            {
                "model_budget": model_budget,
                "budget": subset[0]["budget"],
                "context_length": context_length,
                "task": "ALL",
                "method": method,
                "samples": sum(int(row["samples"]) for row in subset),
                "score": sum(scores) / len(scores) if scores else "",
                "status": "OK",
                "path": "",
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model_budget", "budget", "context_length", "task", "method", "samples", "score", "status", "path"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
