#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path


HEADER_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+([^/\s]+)/([^/\s]+)/")
RESULT_RE = re.compile(r"^\s+ours_page_gather:\s+score=([0-9.]+)\s+kept=(\d+)/(\d+)\s+online=([0-9.]+)s")
DEFAULT_TASKS = [
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


def summarize(
    path: Path,
    name: str,
    tasks: list[str],
    samples_per_task: int,
) -> list[tuple[str, int, float, float, float]]:
    groups: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    current_task = ""
    result_idx = 0
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        header = HEADER_RE.match(line)
        if header:
            current_task = header.group(4)
            continue
        result = RESULT_RE.match(line)
        if not result or not current_task:
            continue
        if samples_per_task > 0 and tasks:
            task_idx = min(result_idx // samples_per_task, len(tasks) - 1)
            current_task = tasks[task_idx]
        result_idx += 1
        score = float(result.group(1))
        kept = int(result.group(2))
        raw = int(result.group(3))
        online = float(result.group(4))
        groups[current_task].append((score, kept / max(1, raw), online))
        groups["ALL"].append((score, kept / max(1, raw), online))
    rows = []
    for task, items in sorted(groups.items()):
        n = len(items)
        rows.append(
            (
                task,
                n,
                sum(item[0] for item in items) / n,
                sum(item[1] for item in items) / n,
                sum(item[2] for item in items) / n,
            )
        )
    return rows


def main() -> None:
    samples_per_task = 20
    tasks = DEFAULT_TASKS
    specs: list[str] = []
    args = iter(sys.argv[1:])
    for arg in args:
        if arg == "--samples-per-task":
            samples_per_task = int(next(args))
        elif arg == "--tasks":
            tasks = [item.strip() for item in next(args).split(",") if item.strip()]
        else:
            specs.append(arg)
    print("name,task,samples,score,keep,online")
    for spec in specs:
        if "=" in spec:
            name, raw_path = spec.split("=", 1)
        else:
            raw_path = spec
            name = Path(raw_path).stem
        for task, samples, score, keep, online in summarize(Path(raw_path), name, tasks, samples_per_task):
            print(f"{name},{task},{samples},{score:.6f},{keep:.6f},{online:.6f}")


if __name__ == "__main__":
    main()
