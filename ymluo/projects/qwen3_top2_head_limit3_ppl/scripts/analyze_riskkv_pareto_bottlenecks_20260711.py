#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


INTERESTING_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "qmsum",
    "repobench-p",
]

COUNT_FIELDS = [
    "ours_score_risk_triggered",
    "ours_coverage_risk_triggered",
    "ours_action_router_selected_action",
    "ours_retry_full_fallback_active",
    "ours_consistency_full_fallback_active",
    "ours_direct_structured_answer_used",
    "ours_short_decode_active",
    "ours_score_risk_budget_tokens",
    "ours_coverage_risk_escalation_budget",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = row.get(key, "")
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda item: (-item[1], item[0])))


def summarize_task(rows: list[dict[str, str]]) -> dict[str, Any]:
    keep = [fnum(row, "keep_fraction") for row in rows]
    online = [fnum(row, "online_seconds") for row in rows]
    score = [fnum(row, "score") for row in rows]
    total = [fnum(row, "total_seconds") for row in rows]
    return {
        "samples": len(rows),
        "score": mean(score),
        "kv_keep": mean(keep),
        "online_seconds": mean(online),
        "total_seconds": mean(total),
        "high_kv_over_50": sum(value > 0.50 for value in keep),
        "very_high_kv_over_75": sum(value > 0.75 for value in keep),
        "fullish_over_90": sum(value > 0.90 for value in keep),
    }


def top_high_kv_rows(rows: list[dict[str, str]], limit: int = 12) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: (fnum(row, "keep_fraction"), fnum(row, "online_seconds")), reverse=True)
    out = []
    for row in selected[:limit]:
        out.append(
            {
                "task": row.get("task", ""),
                "sample_id": row.get("sample_id", ""),
                "score": fnum(row, "score"),
                "kv_keep": fnum(row, "keep_fraction"),
                "online_seconds": fnum(row, "online_seconds"),
                "raw_prefix_tokens": fnum(row, "raw_prefix_tokens"),
                "kept_prefix_tokens": fnum(row, "kept_prefix_tokens"),
                "score_risk": row.get("ours_score_risk_triggered", ""),
                "coverage_risk": row.get("ours_coverage_risk_triggered", ""),
                "action": row.get("ours_action_router_selected_action", ""),
                "retry_full": row.get("ours_retry_full_fallback_active", ""),
                "consistency_full": row.get("ours_consistency_full_fallback_active", ""),
            }
        )
    return out


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# RiskKV Pareto Bottleneck Analysis")
    lines.append("")
    lines.append(f"Input: `{summary['input']}`")
    lines.append("")
    lines.append("## Task Summary")
    lines.append("")
    lines.append("| Task | Samples | Score | KV keep | Online s | >50% KV | >75% KV | >90% KV |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for task, row in summary["task_summary"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    task,
                    str(row["samples"]),
                    f"{row['score']:.6f}",
                    f"{100 * row['kv_keep']:.2f}%",
                    f"{row['online_seconds']:.4f}",
                    str(row["high_kv_over_50"]),
                    str(row["very_high_kv_over_75"]),
                    str(row["fullish_over_90"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Trigger Counts")
    lines.append("")
    for task, field_counts in summary["trigger_counts"].items():
        lines.append(f"### {task}")
        lines.append("")
        for field, values in field_counts.items():
            rendered = ", ".join(f"`{key or '<empty>'}`={value}" for key, value in list(values.items())[:6])
            lines.append(f"- `{field}`: {rendered}")
        lines.append("")
    lines.append("## Highest KV Samples")
    lines.append("")
    lines.append("| Task | Sample | Score | KV keep | Online s | Raw prefix | Kept prefix | Risk | Coverage | Action |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for row in summary["top_high_kv_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["sample_id"]),
                    f"{row['score']:.4f}",
                    f"{100 * row['kv_keep']:.2f}%",
                    f"{row['online_seconds']:.4f}",
                    f"{row['raw_prefix_tokens']:.0f}",
                    f"{row['kept_prefix_tokens']:.0f}",
                    str(row["score_risk"]),
                    str(row["coverage_risk"]),
                    str(row["action"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- Tasks with many >75% KV samples should not be optimized by global budget shrinkage; they need a new middle action or a selective fallback router."
    )
    lines.append(
        "- If bounded fallback preserves task score while reducing >75% KV samples, promote it to M150 / extra-50 validation."
    )
    lines.append(
        "- If bounded fallback drops score, use these trigger fields to train a selective router instead of applying bounded fallback globally."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(input_dir / "task_results.csv")
    task_summary: dict[str, Any] = {}
    trigger_counts: dict[str, Any] = {}
    for task in sorted({row.get("task", "") for row in rows}):
        task_rows = [row for row in rows if row.get("task") == task]
        task_summary[task] = summarize_task(task_rows)
        if task in INTERESTING_TASKS:
            trigger_counts[task] = {
                field: counts(task_rows, field)
                for field in COUNT_FIELDS
                if field in task_rows[0]
            }

    interesting_rows = [row for row in rows if row.get("task") in INTERESTING_TASKS]
    summary = {
        "input": str(input_dir),
        "task_summary": task_summary,
        "trigger_counts": trigger_counts,
        "top_high_kv_rows": top_high_kv_rows(interesting_rows),
    }
    (output_dir / "bottleneck_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "bottleneck_summary.md", summary)
    print(json.dumps({"output_dir": str(output_dir), "tasks": len(task_summary)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
