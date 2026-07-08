from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_blocksize_router_distill_from_sweeps import finite_float, load_sweep_rows, threshold_for, write_csv  # noqa: E402
from run_qwen8b_paper_benchmarks import parse_csv_tuple  # noqa: E402


@dataclass(frozen=True)
class Config:
    benchmark_output_dirs: tuple[str, ...]
    output_dir: str
    feature_block_tokens: int
    summary_rouge_slack: float
    quality_mode: str
    allowed_label_regex: str
    alpha: float
    min_cases: int


@dataclass
class ActionStats:
    group: str
    action: str
    cases: int
    success_rate: float
    failure_rate: float
    avg_token_ratio: float
    avg_seconds: float
    avg_score: float
    selected_floor: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Calibrate block-size risk floors from sweep trials under an empirical failure-rate constraint."
    )
    parser.add_argument("--benchmark_output_dirs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_block_tokens", type=int, default=512)
    parser.add_argument("--summary_rouge_slack", type=float, default=0.03)
    parser.add_argument("--quality_mode", choices=["full", "best", "best_or_full"], default="best_or_full")
    parser.add_argument("--allowed_label_regex", default="")
    parser.add_argument("--alpha", type=float, default=0.0, help="Maximum empirical failure rate allowed for a floor.")
    parser.add_argument("--min_cases", type=int, default=3)
    args = parser.parse_args()
    return Config(
        benchmark_output_dirs=parse_csv_tuple(args.benchmark_output_dirs),
        output_dir=args.output_dir,
        feature_block_tokens=args.feature_block_tokens,
        summary_rouge_slack=args.summary_rouge_slack,
        quality_mode=args.quality_mode,
        allowed_label_regex=args.allowed_label_regex,
        alpha=args.alpha,
        min_cases=args.min_cases,
    )


def group_key(row: dict[str, Any]) -> str:
    benchmark = str(row["benchmark"])
    if benchmark.startswith("ruler_"):
        return benchmark
    return benchmark


def case_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["benchmark"]), str(row["task"]), str(row["case_id"])


def dedupe_rows_by_label(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row["label"])
        old = best.get(label)
        if old is None:
            best[label] = row
            continue
        old_key = (
            finite_float(old.get("score")),
            -finite_float(old.get("token_ratio_vs_full_raw"), 1.0),
            -finite_float(old.get("seconds"), 0.0),
        )
        new_key = (
            finite_float(row.get("score")),
            -finite_float(row.get("token_ratio_vs_full_raw"), 1.0),
            -finite_float(row.get("seconds"), 0.0),
        )
        if new_key > old_key:
            best[label] = row
    return list(best.values())


def allowed_rows(rows: list[dict[str, Any]], config: Config) -> list[dict[str, Any]]:
    out = [row for row in rows if row["label"] != "full_raw"]
    if config.allowed_label_regex:
        matched = [row for row in out if re.fullmatch(config.allowed_label_regex, str(row["label"]))]
        if matched:
            out = matched
    return out


def calibrate(config: Config) -> tuple[list[ActionStats], list[dict[str, Any]]]:
    rows, _, _ = load_sweep_rows(config)
    grouped_cases: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        score = finite_float(row.get("score"))
        ratio = finite_float(row.get("token_ratio_vs_full_raw"))
        if math.isfinite(score) and math.isfinite(ratio):
            grouped_cases.setdefault(case_key(row), []).append(row)

    per_group_action: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for _, raw_case_rows in grouped_cases.items():
        case_rows = dedupe_rows_by_label(raw_case_rows)
        threshold, _, _ = threshold_for(case_rows, config)
        candidates = allowed_rows(case_rows, config)
        if not candidates:
            continue
        group = group_key(candidates[0])
        for row in candidates:
            per_group_action.setdefault((group, str(row["label"])), []).append(
                {
                    "success": int(finite_float(row["score"]) + 1e-12 >= threshold),
                    "token": finite_float(row["token_ratio_vs_full_raw"], 1.0),
                    "seconds": finite_float(row["seconds"], 0.0),
                    "score": finite_float(row["score"], 0.0),
                }
            )

    stats: list[ActionStats] = []
    by_group: dict[str, list[ActionStats]] = {}
    for (group, action), items in sorted(per_group_action.items()):
        if len(items) < config.min_cases:
            continue
        success_rate = sum(item["success"] for item in items) / len(items)
        row = ActionStats(
            group=group,
            action=action,
            cases=len(items),
            success_rate=success_rate,
            failure_rate=1.0 - success_rate,
            avg_token_ratio=sum(item["token"] for item in items) / len(items),
            avg_seconds=sum(item["seconds"] for item in items) / len(items),
            avg_score=sum(item["score"] for item in items) / len(items),
            selected_floor=0,
        )
        stats.append(row)
        by_group.setdefault(group, []).append(row)

    selected: list[dict[str, Any]] = []
    for group, items in sorted(by_group.items()):
        feasible = [row for row in items if row.failure_rate <= config.alpha + 1e-12]
        if not feasible:
            feasible = sorted(items, key=lambda row: (row.failure_rate, row.avg_token_ratio, row.action))[:1]
        chosen = min(feasible, key=lambda row: (row.avg_token_ratio, row.avg_seconds, row.action))
        chosen.selected_floor = 1
        selected.append(
            {
                "group": group,
                "floor_action": chosen.action,
                "cases": chosen.cases,
                "success_rate": chosen.success_rate,
                "failure_rate": chosen.failure_rate,
                "avg_token_ratio": chosen.avg_token_ratio,
                "avg_seconds": chosen.avg_seconds,
                "alpha": config.alpha,
            }
        )
    return stats, selected


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats, selected = calibrate(config)
    write_csv(output_dir / "floor_action_stats.csv", [asdict(row) for row in stats])
    write_csv(output_dir / "calibrated_floors.csv", selected)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "selected": selected,
                "num_action_stats": len(stats),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("group,floor_action,cases,success_rate,failure_rate,avg_token_ratio")
    for row in selected:
        print(
            f"{row['group']},{row['floor_action']},{row['cases']},"
            f"{row['success_rate']:.4f},{row['failure_rate']:.4f},{row['avg_token_ratio']:.4f}"
        )
    print(f"saved calibration to {output_dir}")


if __name__ == "__main__":
    main()
