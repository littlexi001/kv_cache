from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = str(path)
                rows.append(row)
    return rows


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in {"", "nan", "None"}:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def mean(values: list[float]) -> str:
    return "" if not values else f"{sum(values) / len(values):.4f}"


def summarize(rows: list[dict[str, str]], group_by: list[str]) -> list[dict[str, str]]:
    buckets: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(col, "") for col in group_by)].append(row)

    out: list[dict[str, str]] = []
    for key, bucket in buckets.items():
        n = len(bucket)
        cand = sum(int(float(row.get("candidate_correct", "0"))) for row in bucket) / max(1, n)
        gen = sum(int(float(row.get("generation_correct", "0"))) for row in bucket) / max(1, n)
        margins = [value for row in bucket if (value := as_float(row, "candidate_margin")) is not None]
        selectivities = [
            value for row in bucket if (value := as_float(row, "rule_attention_selectivity")) is not None
        ]
        item = {col: value for col, value in zip(group_by, key)}
        item.update(
            {
                "cases": str(n),
                "candidate_acc": f"{cand:.4f}",
                "generation_acc": f"{gen:.4f}",
                "mean_margin": mean(margins),
                "attention_samples": str(len(selectivities)),
                "mean_selectivity": mean(selectivities),
            }
        )
        out.append(item)

    def sort_key(item: dict[str, str]) -> tuple:
        values: list[object] = []
        for col in group_by:
            raw = item.get(col, "")
            try:
                values.append(float(raw))
            except ValueError:
                values.append(raw)
        return tuple(values)

    return sorted(out, key=sort_key)


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(col, "") for col in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize clean local-rule ablation results.")
    parser.add_argument("results_csv", nargs="+", type=Path)
    parser.add_argument("--group_by", required=True, help="Comma-separated result columns to group by.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    group_by = [item.strip() for item in args.group_by.split(",") if item.strip()]
    rows = summarize(read_rows(args.results_csv), group_by)
    columns = group_by + [
        "cases",
        "candidate_acc",
        "generation_acc",
        "mean_margin",
        "attention_samples",
        "mean_selectivity",
    ]
    text = markdown_table(rows, columns)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

