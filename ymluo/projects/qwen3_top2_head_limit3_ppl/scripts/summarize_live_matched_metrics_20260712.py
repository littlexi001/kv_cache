from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from summarize_live_log_metrics_20260712 import RUN_LOGS, aggregate, parse_log


def key_of(row: dict[str, float | str], mode: str) -> tuple[object, ...]:
    if mode == "sample":
        return (str(row["benchmark"]), str(row["task"]), str(row["sample"]))
    if mode == "ordinal":
        return (int(float(row["ordinal"])),)
    raise ValueError(f"unknown match mode {mode!r}")


def summarize_pair(candidate_name: str, baseline_name: str, by_task: bool, match_mode: str) -> None:
    if candidate_name not in RUN_LOGS:
        print(f"{candidate_name:14s} UNKNOWN_RUN")
        return
    if baseline_name not in RUN_LOGS:
        print(f"{baseline_name:14s} UNKNOWN_BASELINE")
        return
    cand_rows = parse_log(Path(RUN_LOGS[candidate_name]))
    base_rows = parse_log(Path(RUN_LOGS[baseline_name]))
    cand_by_key = {key_of(row, match_mode): row for row in cand_rows}
    base_by_key = {key_of(row, match_mode): row for row in base_rows}
    keys = sorted(set(cand_by_key) & set(base_by_key))
    if not keys:
        print(f"{candidate_name:14s} vs {baseline_name:14s} NO_MATCHED_ROWS match={match_mode}")
        return

    cand = [cand_by_key[key] for key in keys]
    base = [base_by_key[key] for key in keys]
    cand_agg = aggregate(cand)
    base_agg = aggregate(base)
    print(
        f"{candidate_name:14s} vs {baseline_name:14s} n={len(keys):4d} match={match_mode} "
        f"score={cand_agg['score']:.4f} base={base_agg['score']:.4f} "
        f"ret={cand_agg['score']/max(base_agg['score'], 1e-9):.2%} "
        f"kv={cand_agg['kv']:.2%} speed={base_agg['online']/max(cand_agg['online'], 1e-9):.2f}x "
        f"cand_online={cand_agg['online']:.4f}s base_online={base_agg['online']:.4f}s"
    )
    if not by_task:
        return
    grouped: dict[str, list[tuple[dict[str, float | str], dict[str, float | str]]]] = defaultdict(list)
    for key in keys:
        grouped[str(cand_by_key[key].get("task", "unknown"))].append((cand_by_key[key], base_by_key[key]))
    for task, pairs in sorted(grouped.items()):
        cand_task = [pair[0] for pair in pairs]
        base_task = [pair[1] for pair in pairs]
        cand_task_agg = aggregate(cand_task)
        base_task_agg = aggregate(base_task)
        print(
            f"  {task:24s} n={len(pairs):4d} "
            f"score={cand_task_agg['score']:.4f} base={base_task_agg['score']:.4f} "
            f"ret={cand_task_agg['score']/max(base_task_agg['score'], 1e-9):.2%} "
            f"kv={cand_task_agg['kv']:.2%} speed={base_task_agg['online']/max(cand_task_agg['online'], 1e-9):.2f}x"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="full_m200")
    parser.add_argument("--runs", default="v427_m200,v428_m200,v429_m100,v430_m100,v431_m100")
    parser.add_argument(
        "--match",
        choices=["ordinal", "sample"],
        default="ordinal",
        help="Use ordinal for live logs with log_every>1. Sample mode only works when every sample id is logged.",
    )
    parser.add_argument("--by-task", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for run in [item.strip() for item in args.runs.split(",") if item.strip()]:
        summarize_pair(run, args.baseline, args.by_task, args.match)


if __name__ == "__main__":
    main()
