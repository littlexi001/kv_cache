#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
LOG_DIR = ROOT / "outputs" / "logs"
DOC_DIR = ROOT / "doc"
OUT_JSON = ROOT / "outputs" / "riskkv_lowkv_running_progress_20260712.json"
OUT_MD = DOC_DIR / "section172_lowkv_running_progress_20260712.md"

MARKER_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+longbench/([^/]+)/")
RESULT_RE = re.compile(
    r"ours_page_gather:\s+score=([0-9.]+)\s+kept=([0-9.]+)/([0-9.]+)\s+online=([0-9.]+)s"
)
ALL_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]
HARD_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "repobench-p",
]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def infer_task_order(path: Path, expected: int) -> list[str]:
    name = path.name
    if "_hard_" in name or "hardtask" in name:
        return HARD_TASKS
    if expected and expected % len(ALL_TASKS) == 0:
        return ALL_TASKS
    if expected and expected % len(HARD_TASKS) == 0:
        return HARD_TASKS
    return []


def parse_log(path: Path) -> dict[str, object] | None:
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return None
    task = ""
    expected = 0
    last_marker = ""
    rows: list[dict[str, object]] = []
    task_order: list[str] = []
    samples_per_task = 0
    for line in lines:
        marker = MARKER_RE.search(line)
        if marker:
            expected = int(marker.group(2))
            task = marker.group(3)
            last_marker = line[:240]
            if not task_order:
                task_order = infer_task_order(path, expected)
                samples_per_task = expected // len(task_order) if task_order else 0
            continue
        result = RESULT_RE.search(line)
        if result:
            if task_order and samples_per_task > 0:
                task_idx = min(len(task_order) - 1, len(rows) // samples_per_task)
                assigned_task = task_order[task_idx]
            else:
                assigned_task = task
            if not assigned_task:
                continue
            score = float(result.group(1))
            kept = float(result.group(2))
            raw = max(1.0, float(result.group(3)))
            online = float(result.group(4))
            rows.append(
                {
                    "task": assigned_task,
                    "score": score,
                    "kv": kept / raw,
                    "online": online,
                }
            )
    if not rows:
        return None
    tasks = sorted({str(row["task"]) for row in rows})
    per_task = []
    for item in tasks:
        subset = [row for row in rows if row["task"] == item]
        per_task.append(
            {
                "task": item,
                "n": len(subset),
                "score": mean([float(row["score"]) for row in subset]),
                "kv": mean([float(row["kv"]) for row in subset]),
                "online": mean([float(row["online"]) for row in subset]),
            }
        )
    return {
        "log": str(path.relative_to(ROOT)),
        "samples": len(rows),
        "expected": expected,
        "progress": len(rows) / max(1, expected),
        "tasks": len(tasks),
        "score": mean([float(row["score"]) for row in rows]),
        "kv": mean([float(row["kv"]) for row in rows]),
        "online": mean([float(row["online"]) for row in rows]),
        "last_marker": last_marker,
        "per_task": per_task,
    }


def main() -> None:
    candidates = []
    for pattern in [
        "riskkv_v19_v36*20260712*.log",
        "riskkv_v19_v37*20260712*.log",
        "riskkv_v19_v38*20260712*.log",
        "riskkv_v19_v39*20260712*.log",
    ]:
        candidates.extend(LOG_DIR.glob(pattern))
    summaries = []
    for path in sorted(set(candidates)):
        parsed = parse_log(path)
        if parsed:
            summaries.append(parsed)
    summaries.sort(key=lambda item: (-float(item["progress"]), -float(item["score"]), float(item["kv"])))
    OUT_JSON.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# section172：Low-KV running progress 自动解析",
        "",
        "说明：这里直接解析正在写入的 log，因此是 partial progress，不替代最终 `task_results.csv`。",
        "",
        "| run log | samples | expected | progress | tasks | score | KV keep | online |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {log} | {samples} | {expected} | {progress:.1%} | {tasks} | {score:.4f} | {kv:.2%} | {online:.4f} |".format(
                **item
            )
        )
    lines.append("")
    for item in summaries:
        lines += [
            f"## {item['log']}",
            "",
            f"last marker: `{item['last_marker']}`",
            "",
            "| task | n | score | KV keep | online |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in item["per_task"]:
            lines.append("| {task} | {n} | {score:.4f} | {kv:.2%} | {online:.4f} |".format(**row))
        lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_JSON)
    print(OUT_MD)
    for item in summaries[:12]:
        print(
            json.dumps(
                {
                    key: item[key]
                    for key in ["log", "samples", "expected", "progress", "tasks", "score", "kv", "online"]
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
