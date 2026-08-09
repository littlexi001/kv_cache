#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")


def main() -> None:
    rows_out: list[dict[str, str]] = []
    for path in sorted((ROOT / "outputs").glob("riskkv_v19_learned_budget_router_v5_clean_budget_covered320_*_20260711")):
        summary_path = path / "router_summary.csv"
        if not summary_path.exists():
            continue
        rows = list(csv.DictReader(summary_path.open(newline="", encoding="utf-8")))
        for split in ["test", "all"]:
            row = next(item for item in rows if item["split"] == split and item["task"] == "ALL")
            rows_out.append(
                {
                    "router": path.name,
                    "split": split,
                    "samples": row["samples"],
                    "score_ratio": row["learned_vs_reference"],
                    "reference_score": row["reference_score"],
                    "learned_score": row["learned_score"],
                    "reference_kv": row["reference_kv_keep"],
                    "learned_kv": row["learned_kv_keep"],
                    "speed_vs_reference": row["learned_speed_vs_reference"],
                    "safe_rate": row["safe_rate"],
                    "oracle_action_accuracy": row["oracle_action_accuracy"],
                }
            )
    fields = [
        "router",
        "split",
        "samples",
        "score_ratio",
        "reference_score",
        "learned_score",
        "reference_kv",
        "learned_kv",
        "speed_vs_reference",
        "safe_rate",
        "oracle_action_accuracy",
    ]
    out_path = ROOT / "outputs/riskkv_v19_learned_budget_router_v5_clean_budget_summary_20260711.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
    print(out_path)
    for row in rows_out:
        if row["split"] == "test":
            print(row)


if __name__ == "__main__":
    main()
