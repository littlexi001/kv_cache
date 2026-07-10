#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
OUTPUT = ROOT / "outputs" / "riskkv_flow_v12_summary_20260709.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows: list[dict[str, str]] = []
    for summary in sorted((ROOT / "outputs").glob("riskkv_*20260709*/summary.csv")):
        exp = summary.parent.name
        for row in read_rows(summary):
            if row.get("benchmark") != "longbench":
                continue
            rows.append(
                {
                    "experiment": exp,
                    "task": row.get("task", ""),
                    "method": row.get("method", ""),
                    "samples": row.get("samples", ""),
                    "score": row.get("score", ""),
                    "mean_keep_fraction": row.get("mean_keep_fraction", ""),
                    "mean_kept_context_tokens": row.get("mean_kept_context_tokens", ""),
                    "mean_total_seconds": row.get("mean_total_seconds", ""),
                    "mean_online_seconds": row.get("mean_online_seconds", ""),
                    "mean_kv_gather_seconds": row.get("mean_kv_gather_seconds", ""),
                }
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "task",
                "method",
                "samples",
                "score",
                "mean_keep_fraction",
                "mean_kept_context_tokens",
                "mean_total_seconds",
                "mean_online_seconds",
                "mean_kv_gather_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
