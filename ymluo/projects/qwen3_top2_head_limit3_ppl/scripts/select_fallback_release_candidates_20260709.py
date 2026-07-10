#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


RESULT_RE = re.compile(
    r"ours_page_gather:\s+score=([0-9.]+)\s+kept=(\d+)/(\d+)\s+online=([0-9.]+)s"
)

LONG_BENCH_TASKS = [
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

TARGET_TASKS = ["hotpotqa", "musique", "trec", "passage_count", "repobench-p", "qasper"]


def parse_task_list(raw: str, default: list[str]) -> list[str]:
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def summarize_log(path: Path, tasks: list[str], samples_per_task: int) -> dict[str, tuple[int, float, float, float]]:
    groups: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    if not path.exists():
        return {}
    result_idx = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RESULT_RE.search(line)
        if not match:
            continue
        task_idx = min(result_idx // max(1, samples_per_task), len(tasks) - 1)
        task = tasks[task_idx]
        result_idx += 1
        score = float(match.group(1))
        kept = int(match.group(2))
        raw = int(match.group(3))
        online = float(match.group(4))
        groups[task].append((score, kept / max(1, raw), online))
    summary = {}
    for task, rows in groups.items():
        n = len(rows)
        summary[task] = (
            n,
            sum(row[0] for row in rows) / n,
            sum(row[1] for row in rows) / n,
            sum(row[2] for row in rows) / n,
        )
    return summary


def parse_candidate_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, raw_path = spec.split("=", 1)
        return name, Path(raw_path)
    path = Path(spec)
    return path.stem, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", required=True)
    parser.add_argument("--baseline-tasks", default=",".join(LONG_BENCH_TASKS))
    parser.add_argument("--candidate-tasks", default=",".join(TARGET_TASKS))
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--quality-floor", type=float, default=0.02)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("candidates", nargs="+", help="NAME=LOG_PATH entries.")
    args = parser.parse_args()

    baseline_tasks = parse_task_list(args.baseline_tasks, LONG_BENCH_TASKS)
    candidate_tasks = parse_task_list(args.candidate_tasks, TARGET_TASKS)
    baseline = summarize_log(Path(args.baseline_log), baseline_tasks, args.samples_per_task)
    candidates = [
        (name, summarize_log(path, candidate_tasks, args.samples_per_task))
        for name, path in map(parse_candidate_spec, args.candidates)
    ]

    print(
        "task,baseline_samples,baseline_score,baseline_keep,selected_action,"
        "selected_samples,selected_score,selected_keep,delta_score,delta_keep,status"
    )
    for task in candidate_tasks:
        base = baseline.get(task)
        if not base:
            print(f"{task},0,,,,,,,,missing_baseline")
            continue
        base_n, base_score, base_keep, _ = base
        selected_name = "baseline"
        selected = base
        status = "keep_baseline"
        for name, summary in candidates:
            row = summary.get(task)
            if not row:
                continue
            n, score, keep, online = row
            if n < args.min_samples:
                continue
            if score >= base_score - args.quality_floor and keep < selected[2]:
                selected_name = name
                selected = row
                status = "release"
        sel_n, sel_score, sel_keep, _ = selected
        print(
            f"{task},{base_n},{base_score:.6f},{base_keep:.6f},{selected_name},"
            f"{sel_n},{sel_score:.6f},{sel_keep:.6f},"
            f"{sel_score - base_score:.6f},{sel_keep - base_keep:.6f},{status}"
        )


if __name__ == "__main__":
    main()
