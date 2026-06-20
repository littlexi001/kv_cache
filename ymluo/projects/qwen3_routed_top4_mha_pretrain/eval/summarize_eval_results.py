from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def format_value(row: dict[str, Any]) -> str:
    if "accuracy" in row:
        return f"{100.0 * float(row['accuracy']):.2f}%"
    if "ppl" in row:
        return f"ce={float(row['ce']):.4f}, ppl={float(row['ppl']):.2f}"
    return "n/a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task"], {})[row["model"]] = row

    print("| task | routed_checkpoint | official_qwen3_0p6b | gap | examples/tokens |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for task in sorted(by_task):
        routed = by_task[task].get("routed_checkpoint")
        official = by_task[task].get("official_qwen3_0p6b")
        routed_text = format_value(routed) if routed else ""
        official_text = format_value(official) if official else ""
        gap = ""
        if routed and official and "accuracy" in routed and "accuracy" in official:
            gap = f"{100.0 * (float(routed['accuracy']) - float(official['accuracy'])):.2f} pp"
        count = ""
        row_for_count = routed or official
        if row_for_count:
            count = str(row_for_count.get("examples", row_for_count.get("tokens", "")))
        print(f"| {task} | {routed_text} | {official_text} | {gap} | {count} |")


if __name__ == "__main__":
    main()
