#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_eval_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("kind") != "pcic_r_eval":
                continue
            rule = json.loads(row.get("rescue_rule") or "{}")
            rows.append({"row": row, "rule": rule})
    return rows


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def analyze_case(case_id: str, task: str, path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in load_eval_rows(path):
        row = item["row"]
        rule = item["rule"]
        block = str(row.get("block", ""))
        final_combo = str(row.get("combo", ""))
        initial_combo = str(rule.get("sentinel_cascade_initial_selected_combo", ""))
        route = str(rule.get("sentinel_cascade_initial_route", ""))
        extended = inum(rule.get("sentinel_cascade_extended", 0))
        early = inum(rule.get("sentinel_cascade_accepted_early", 0))
        extension_seconds = sum(
            fnum(value)
            for value in (rule.get("sentinel_cascade_extension_seconds") or {}).values()
        )
        initial_seconds = sum(
            fnum(value)
            for value in (rule.get("sentinel_cascade_initial_seconds") or {}).values()
        )
        gate_seconds = fnum(row.get("gate_seconds"))
        anchors = rule.get("sentinel_cascade_anchor_combos") or []
        anchor_hit = int(initial_combo in anchors)
        no_change = int(bool(initial_combo) and initial_combo == final_combo)
        out.append(
            {
                "case_id": case_id,
                "task": task,
                "path": str(path),
                "block": block,
                "final_combo": final_combo,
                "initial_combo": initial_combo,
                "initial_route": route,
                "extended": extended,
                "early": early,
                "anchor_hit": anchor_hit,
                "no_change_after_extension": no_change,
                "extension_seconds": extension_seconds,
                "initial_seconds": initial_seconds,
                "gate_seconds": gate_seconds,
                "initial_margin": fnum(rule.get("sentinel_cascade_initial_best_margin")),
                "initial_pairwise_delta": fnum(rule.get("sentinel_cascade_initial_pairwise_delta")),
                "horizon_gain": fnum(rule.get("sentinel_horizon_gain")),
                "horizon_gain_ratio": fnum(rule.get("sentinel_horizon_gain_ratio")),
                "delta_ppl": fnum(row.get("delta_ppl")),
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for case_id, case_rows in sorted(by_case.items()):
        extended_rows = [row for row in case_rows if int(row["extended"]) == 1]
        no_change_rows = [row for row in extended_rows if int(row["no_change_after_extension"]) == 1]
        anchor_no_change_rows = [
            row for row in extended_rows if int(row["anchor_hit"]) == 1 and int(row["no_change_after_extension"]) == 1
        ]
        extension_seconds = sum(float(row["extension_seconds"]) for row in extended_rows)
        avoidable_seconds = sum(float(row["extension_seconds"]) for row in no_change_rows)
        anchor_avoidable_seconds = sum(float(row["extension_seconds"]) for row in anchor_no_change_rows)
        summaries.append(
            {
                "case_id": case_id,
                "task": case_rows[0]["task"],
                "blocks": len(case_rows),
                "extended_blocks": len(extended_rows),
                "no_change_extended_blocks": len(no_change_rows),
                "anchor_no_change_extended_blocks": len(anchor_no_change_rows),
                "extension_seconds": extension_seconds,
                "avoidable_seconds_if_initial_kept": avoidable_seconds,
                "avoidable_fraction_if_initial_kept": avoidable_seconds / max(extension_seconds, 1e-9),
                "avoidable_seconds_if_anchor_hit_kept": anchor_avoidable_seconds,
                "avoidable_fraction_if_anchor_hit_kept": anchor_avoidable_seconds / max(extension_seconds, 1e-9),
                "avg_delta_ppl": sum(float(row["delta_ppl"]) for row in case_rows) / max(1, len(case_rows)),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", nargs=3, metavar=("CASE_ID", "TASK", "CSV"), required=True)
    parser.add_argument("--block_csv_out", type=Path, required=True)
    parser.add_argument("--summary_csv_out", type=Path, required=True)
    parser.add_argument("--md_out", type=Path, required=True)
    args = parser.parse_args()

    block_rows: list[dict[str, Any]] = []
    for case_id, task, csv_path in args.case:
        block_rows.extend(analyze_case(case_id, task, Path(csv_path)))
    summary_rows = summarize(block_rows)
    write_csv(args.block_csv_out, block_rows)
    write_csv(args.summary_csv_out, summary_rows)

    lines = [
        "# PCIC Extension Waste 后验分析（2026-06-29）",
        "",
        "## 目的",
        "",
        "该分析只解析已有 `pcic_r_blockwise_results.csv`，不跑模型。目标是量化 cascade extension 中有多少 block 最终选择没有变化，从而判断下一步是否值得设计更强 skip gate。",
        "",
        f"block 级 CSV：`{args.block_csv_out}`",
        f"summary CSV：`{args.summary_csv_out}`",
        "",
        "## 汇总",
        "",
        "| case | task | blocks | extended | no-change ext | avoidable s | avoidable frac | anchor-hit avoidable s | avg ΔPPL |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['case_id']} | {row['task']} | {row['blocks']} | {row['extended_blocks']} | "
            f"{row['no_change_extended_blocks']} | {row['avoidable_seconds_if_initial_kept']:.3f} | "
            f"{row['avoidable_fraction_if_initial_kept']:.3f} | "
            f"{row['avoidable_seconds_if_anchor_hit_kept']:.3f} | {row['avg_delta_ppl']:.6f} |"
        )
    lines += [
        "",
        "## 解释",
        "",
        "- `no-change ext`：已经运行 extension，但最终 combo 与 early initial combo 相同。",
        "- `avoidable s`：如果能提前识别这些 no-change block，理论上可省的 extension 秒数。",
        "- 这是后验上界，不是可直接写进主方法的规则；它用于判断下一步 skip-gate 设计是否有空间。",
    ]
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
