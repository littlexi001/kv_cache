#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def corrected_gate_for_rule(rule: dict[str, Any], selected_combo: str) -> tuple[float, float, float]:
    initial_seconds = rule.get("sentinel_cascade_initial_seconds") or {}
    extension_seconds = rule.get("sentinel_cascade_extension_seconds") or {}
    initial_gate = sum(
        fnum(seconds)
        for name, seconds in initial_seconds.items()
        if str(name) != selected_combo
    )
    extension_gate = sum(
        fnum(seconds)
        for name, seconds in extension_seconds.items()
        if str(name) != selected_combo
    )
    return initial_gate + extension_gate, initial_gate, extension_gate


def summarize_case(case_id: str, task: str, mode: str, path: Path) -> dict[str, Any]:
    rows = [row for row in csv.DictReader(path.open(newline="", encoding="utf-8")) if row.get("kind") == "pcic_r_eval"]
    old_gate = 0.0
    corrected_gate = 0.0
    corrected_initial_gate = 0.0
    corrected_extension_gate = 0.0
    baseline = 0.0
    method = 0.0
    selected_seconds = 0.0
    skipped = 0
    extended = 0
    combos: list[str] = []
    delta_ppl = 0.0
    for row in rows:
        rule = json.loads(row.get("rescue_rule") or "{}")
        combo = str(row.get("combo", ""))
        combos.append(combo)
        old_gate += fnum(row.get("gate_seconds"))
        baseline += fnum(row.get("baseline_seconds"))
        method += fnum(row.get("method_seconds"), fnum(row.get("seconds")))
        selected_seconds += fnum(row.get("seconds"))
        delta_ppl += fnum(row.get("delta_ppl"))
        skipped += int(rule.get("sentinel_cascade_skipped_by_anchor_nonpositive_gain", 0) or 0)
        extended += int(rule.get("sentinel_cascade_extended", 0) or 0)
        if "sentinel_cascade_initial_seconds" in rule:
            gate, initial_gate, extension_gate = corrected_gate_for_rule(rule, combo)
            corrected_gate += gate
            corrected_initial_gate += initial_gate
            corrected_extension_gate += extension_gate
        else:
            corrected_gate += fnum(row.get("gate_seconds"))
    corrected_method = selected_seconds + corrected_gate
    return {
        "case_id": case_id,
        "task": task,
        "mode": mode,
        "blocks": len(rows),
        "avg_delta_ppl": delta_ppl / max(1, len(rows)),
        "old_gate_s": old_gate,
        "corrected_gate_s": corrected_gate,
        "corrected_initial_gate_s": corrected_initial_gate,
        "corrected_extension_gate_s": corrected_extension_gate,
        "old_method_ratio": method / max(baseline, 1e-9),
        "corrected_method_ratio": corrected_method / max(baseline, 1e-9),
        "extended": extended,
        "skipped": skipped,
        "combos": "/".join(combos),
    }


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
    parser.add_argument("--case", action="append", nargs=4, metavar=("CASE_ID", "TASK", "MODE", "CSV"), required=True)
    parser.add_argument("--csv_out", type=Path, required=True)
    parser.add_argument("--md_out", type=Path, required=True)
    args = parser.parse_args()

    rows = [summarize_case(case_id, task, mode, Path(csv_path)) for case_id, task, mode, csv_path in args.case]
    write_csv(args.csv_out, rows)

    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task"]), {})[str(row["mode"])] = row

    lines = [
        "# Corrected Cascade Gate Seconds（2026-06-29）",
        "",
        "## 背景",
        "",
        "旧 `gate_seconds` 在 extended cascade 中只统计 extended candidate set，漏掉了未进入 extension 的初始候选 probe。因此 extended runs 的 gate 成本被低估。本分析只读已有 CSV，从 `rescue_rule.sentinel_cascade_initial_seconds` 和 `sentinel_cascade_extension_seconds` 重新计算 corrected gate。",
        "",
        f"原始 CSV：`{args.csv_out}`",
        "",
        "## 结果表",
        "",
        "| case | task | mode | avg_delta_ppl | old gate_s | corrected gate_s | corrected method/base | extended | skipped | combos |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['task']} | {row['mode']} | {row['avg_delta_ppl']:.6f} | "
            f"{row['old_gate_s']:.3f} | {row['corrected_gate_s']:.3f} | "
            f"{row['corrected_method_ratio']:.3f} | {row['extended']} | {row['skipped']} | `{row['combos']}` |"
        )
    lines += [
        "",
        "## Base vs Skip corrected",
        "",
        "| task | ΔPPL change | corrected gate_s change | corrected method/base change | skipped | same combos |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for task, task_rows in by_task.items():
        if "base" not in task_rows or "skip" not in task_rows:
            continue
        base = task_rows["base"]
        skip = task_rows["skip"]
        lines.append(
            f"| {task} | {skip['avg_delta_ppl'] - base['avg_delta_ppl']:.6f} | "
            f"{skip['corrected_gate_s'] - base['corrected_gate_s']:.3f} | "
            f"{skip['corrected_method_ratio'] - base['corrected_method_ratio']:.3f} | "
            f"{skip['skipped']} | {str(skip['combos'] == base['combos'])} |"
        )
    lines += [
        "",
        "## 解释",
        "",
        "- corrected gate 更接近真实候选 probe 成本。",
        "- 旧文档里的 extended-run gate_s 需要谨慎解释；质量结论不受影响。",
        "- 后续所有 gate/speed claim 应优先使用 corrected gate 或修正后的 runner 输出。",
    ]
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
