from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


EXAMPLE_RE = re.compile(
    r"\[(?P<idx>\d+)/(?P<total>\d+)\]\s+(?P<benchmark>[^/]+)/(?P<task>[^/]+)/(?P<sample>\S+)\s+"
    r"prefix_tokens=(?P<prefix>\d+)\s+pages=(?P<pages>\d+)\s+budget=(?P<budget>\d+)"
)
RESULT_RE = re.compile(
    r"(?P<method>full_kv|ours_page_gather):\s+score=(?P<score>-?\d+(?:\.\d+)?)\s+"
    r"kept=(?P<kept>\d+(?:\.\d+)?)/(?P<total>\d+(?:\.\d+)?)\s+online=(?P<online>\d+(?:\.\d+)?)s"
)


RUN_LOGS = {
    "v427_m200": "outputs/logs/riskkv_v19_v427_v417_source_v421_winners_20260712_v427_m200_validate_m200_bDyn_pDyn.log",
    "v428_m200": "outputs/logs/riskkv_v19_v428_v427_plus_repobench_20260712_v428_m200_validate_m200_bDyn_pDyn.log",
    "full_m200": "outputs/logs/riskkv_full_kv_longbench_m200_20260712.log",
    "v427_ruler": "outputs/logs/riskkv_v427_ruler_m50_b384_20260712.log",
    "full_ruler": "outputs/logs/riskkv_full_kv_ruler_m50_20260712.log",
    "v429_m100": "outputs/logs/riskkv_v19_v429_source_best_frontiers_20260712_v429_m100_m100_bDyn_pDyn.log",
    "v430_m100": "outputs/logs/riskkv_v19_v430_composer_kv06_speed6_task20_20260712_v430_m100_m100_bDyn_pDyn.log",
    "v431_m100": "outputs/logs/riskkv_v19_v431_composer_kv08_speed5_task25_20260712_v431_m100_m100_bDyn_pDyn.log",
    "v433_m100": "outputs/logs/riskkv_v19_v433_dpcomposer_kv06_speed6_task20_20260712_v433_m100_m100_bDyn_pDyn.log",
    "v434_m100": "outputs/logs/riskkv_v19_v434_dpcomposer_kv08_speed5_task25_20260712_v434_m100_m100_bDyn_pDyn.log",
    "v435_m100": "outputs/logs/riskkv_v19_v435_dpcomposer_kv10_speed35_task35_20260712_v435_m100_m100_bDyn_pDyn.log",
    "v436_ruler": "outputs/logs/riskkv_v436_ruler_lowkv_b224_m50_20260712.log",
}


def parse_log(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    current: dict[str, float | str] | None = None
    ordinal = 0
    if not path.exists():
        return rows
    for line in path.read_text(errors="ignore").splitlines():
        example_match = EXAMPLE_RE.search(line)
        if example_match:
            current = {
                "benchmark": example_match.group("benchmark"),
                "task": example_match.group("task"),
                "sample": example_match.group("sample"),
                "idx": float(example_match.group("idx")),
                "total_examples": float(example_match.group("total")),
                "prefix_tokens": float(example_match.group("prefix")),
                "pages": float(example_match.group("pages")),
                "budget": float(example_match.group("budget")),
            }
            continue
        result_match = RESULT_RE.search(line)
        if result_match and current is not None:
            ordinal += 1
            total = float(result_match.group("total"))
            item = dict(current)
            item.update(
                {
                    "ordinal": float(ordinal),
                    "method": result_match.group("method"),
                    "score": float(result_match.group("score")),
                    "kept": float(result_match.group("kept")),
                    "total": total,
                    "kv": float(result_match.group("kept")) / max(total, 1.0),
                    "online": float(result_match.group("online")),
                }
            )
            rows.append(item)
    return rows


def aggregate(rows: list[dict[str, float | str]]) -> dict[str, float]:
    return {
        "n": float(len(rows)),
        "score": sum(float(row["score"]) for row in rows) / len(rows),
        "kv": sum(float(row["kv"]) for row in rows) / len(rows),
        "online": sum(float(row["online"]) for row in rows) / len(rows),
    }


def print_run(name: str, path: Path, by_task: bool) -> None:
    rows = parse_log(path)
    if not rows:
        print(f"{name:14s} NO_ROWS path={path}")
        return
    agg = aggregate(rows)
    print(
        f"{name:14s} n={int(agg['n']):4d} score={agg['score']:.4f} "
        f"kv={agg['kv']:.2%} online={agg['online']:.4f}s"
    )
    if not by_task:
        return
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)
    for task, subset in sorted(grouped.items()):
        sub = aggregate(subset)
        print(
            f"  {task:24s} n={int(sub['n']):4d} score={sub['score']:.4f} "
            f"kv={sub['kv']:.2%} online={sub['online']:.4f}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=",".join(RUN_LOGS))
    parser.add_argument("--by-task", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in [item.strip() for item in args.runs.split(",") if item.strip()]:
        if name not in RUN_LOGS:
            print(f"{name:14s} UNKNOWN_RUN")
            continue
        print_run(name, Path(RUN_LOGS[name]), args.by_task)


if __name__ == "__main__":
    main()
