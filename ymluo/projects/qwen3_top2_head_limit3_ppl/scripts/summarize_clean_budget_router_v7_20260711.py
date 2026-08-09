#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")


def collect_rows() -> list[dict[str, str]]:
    rows_out: list[dict[str, str]] = []
    patterns = [
        "riskkv_v19_learned_budget_router_v5_clean_budget_covered320_*_20260711",
        "riskkv_v19_learned_budget_router_v6_noweight_covered320_*_20260711",
        "riskkv_v19_learned_budget_router_v7_safety_ladder_covered320_*_20260711",
    ]
    for pattern in patterns:
        for path in sorted((ROOT / "outputs").glob(pattern)):
            summary_path = path / "router_summary.csv"
            if not summary_path.exists():
                continue
            with summary_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
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
                        "kv_relative": str(
                            float(row["learned_kv_keep"]) / float(row["reference_kv_keep"])
                            if float(row["reference_kv_keep"]) > 0
                            else ""
                        ),
                        "speed_vs_reference": row["learned_speed_vs_reference"],
                        "safe_rate": row["safe_rate"],
                        "oracle_action_accuracy": row["oracle_action_accuracy"],
                    }
                )
    return rows_out


def main() -> None:
    rows = collect_rows()
    out_path = ROOT / "outputs/riskkv_v19_learned_budget_router_v7_compare_summary_20260711.csv"
    fields = list(rows[0]) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)
    test_rows = [row for row in rows if row["split"] == "test"]
    test_rows.sort(
        key=lambda row: (
            float(row["score_ratio"]),
            -float(row["kv_relative"]),
            float(row["safe_rate"]),
        ),
        reverse=True,
    )
    for row in test_rows[:30]:
        print(row)


if __name__ == "__main__":
    main()
