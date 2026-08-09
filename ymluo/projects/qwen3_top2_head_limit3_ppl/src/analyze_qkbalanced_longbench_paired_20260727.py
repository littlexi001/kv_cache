from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FULL = "full_kv"
OURS = "countcap_fullprompt_qkbalanced_packed_direct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired LongBench analysis for QK-balanced packed retrieval."
    )
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def interval(values: np.ndarray) -> dict[str, float]:
    low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "lower_2p5": float(low),
        "median": float(median),
        "upper_97p5": float(high),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def optional_mean(values: list[float | None]) -> float | None:
    measured = [value for value in values if value is not None]
    return float(np.mean(measured)) if measured else None


def main() -> None:
    args = parse_args()
    with args.input_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paired: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        method = str(row["method"])
        if method in {FULL, OURS}:
            paired[(str(row["task"]), str(row["sample_id"]))][method] = row
    incomplete = [key for key, methods in paired.items() if set(methods) != {FULL, OURS}]
    if incomplete:
        raise ValueError(f"incomplete pairs: {incomplete[:5]}")
    if not paired:
        raise ValueError("no Full/Ours pairs")

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_rows = []
    for (task, sample_id), methods in sorted(paired.items()):
        full = methods[FULL]
        ours = methods[OURS]
        diagnostics_enabled = (
            str(ours.get("diagnostics_enabled", "")).lower() == "true"
        )
        pair = {
            "task": task,
            "sample_id": sample_id,
            "full_score": float(full["score"]),
            "ours_score": float(ours["score"]),
            "score_delta": float(ours["score"]) - float(full["score"]),
            "prediction_exact": int(
                str(ours["prediction"]) == str(full["prediction"])
            ),
            "full_generated_tokens": int(full["generated_tokens"]),
            "ours_generated_tokens": int(ours["generated_tokens"]),
            "full_online_seconds": float(full["online_seconds"]),
            "ours_online_seconds": float(ours["online_seconds"]),
            "attention_link_ratio": (
                float(ours["attention_link_ratio"])
                if diagnostics_enabled
                else None
            ),
            "index_ratio_of_full_kv": (
                float(
                    ours.get(
                        "packed_index_ratio_of_full_kv",
                        0.0,
                    )
                    or 0.0
                )
                if diagnostics_enabled
                else None
            ),
        }
        pair_rows.append(pair)
        by_task[task].append(pair)

    task_rows = []
    for task, task_pairs in sorted(by_task.items()):
        full_score = float(
            np.mean([row["full_score"] for row in task_pairs])
        )
        ours_score = float(
            np.mean([row["ours_score"] for row in task_pairs])
        )
        task_rows.append(
            {
                "task": task,
                "samples": len(task_pairs),
                "full_score": full_score,
                "ours_score": ours_score,
                "score_delta": ours_score - full_score,
                "quality_retention": (
                    ours_score / full_score if full_score > 0.0 else ""
                ),
                "prediction_exact_rate": float(
                    np.mean(
                        [row["prediction_exact"] for row in task_pairs]
                    )
                ),
                "attention_link_ratio": optional_mean(
                    [row["attention_link_ratio"] for row in task_pairs]
                ),
                "index_ratio_of_full_kv": optional_mean(
                    [
                        row["index_ratio_of_full_kv"]
                        for row in task_pairs
                    ]
                ),
                "paired_online_speedup": float(
                    np.sum(
                        [row["full_online_seconds"] for row in task_pairs]
                    )
                    / np.sum(
                        [row["ours_online_seconds"] for row in task_pairs]
                    )
                ),
            }
        )

    macro_full = float(np.mean([row["full_score"] for row in task_rows]))
    macro_ours = float(np.mean([row["ours_score"] for row in task_rows]))
    task_names = sorted(by_task)
    rng = np.random.default_rng(args.seed)
    bootstrap_full = np.empty(args.bootstrap_replicates, dtype=np.float64)
    bootstrap_ours = np.empty(args.bootstrap_replicates, dtype=np.float64)
    for replicate in range(args.bootstrap_replicates):
        sampled_tasks = rng.choice(task_names, size=len(task_names), replace=True)
        full_task_scores = []
        ours_task_scores = []
        for task in sampled_tasks:
            task_pairs = by_task[str(task)]
            indices = rng.integers(
                0, len(task_pairs), size=len(task_pairs)
            )
            full_task_scores.append(
                np.mean([task_pairs[index]["full_score"] for index in indices])
            )
            ours_task_scores.append(
                np.mean([task_pairs[index]["ours_score"] for index in indices])
            )
        bootstrap_full[replicate] = np.mean(full_task_scores)
        bootstrap_ours[replicate] = np.mean(ours_task_scores)
    bootstrap_delta = bootstrap_ours - bootstrap_full
    bootstrap_retention = np.divide(
        bootstrap_ours,
        bootstrap_full,
        out=np.full_like(bootstrap_ours, np.nan),
        where=bootstrap_full > 0.0,
    )
    valid_retention = bootstrap_retention[np.isfinite(bootstrap_retention)]

    total_full_online = sum(
        row["full_online_seconds"] for row in pair_rows
    )
    total_ours_online = sum(
        row["ours_online_seconds"] for row in pair_rows
    )
    summary = {
        "protocol": {
            "pairs": len(pair_rows),
            "tasks": len(task_rows),
            "bootstrap": (
                "resample tasks, then paired samples within each task"
            ),
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
        },
        "point_estimate": {
            "macro_full_score": macro_full,
            "macro_ours_score": macro_ours,
            "macro_score_delta": macro_ours - macro_full,
            "macro_quality_retention": macro_ours / macro_full,
            "prediction_exact_rate": float(
                np.mean([row["prediction_exact"] for row in pair_rows])
            ),
            "generated_length_exact_rate": float(
                np.mean(
                    [
                        row["full_generated_tokens"]
                        == row["ours_generated_tokens"]
                        for row in pair_rows
                    ]
                )
            ),
            "attention_link_ratio": optional_mean(
                [row["attention_link_ratio"] for row in pair_rows]
            ),
            "index_ratio_of_full_kv": optional_mean(
                [row["index_ratio_of_full_kv"] for row in pair_rows]
            ),
            "paired_online_speedup": (
                total_full_online / total_ours_online
            ),
        },
        "measurement_note": (
            "null attention/index fields mean attention diagnostics were "
            "disabled to avoid perturbing the quality harness; memory and "
            "kernel speed are reported by the physical-cache benchmark"
        ),
        "bootstrap_95_percent": {
            "macro_score_delta": interval(bootstrap_delta),
            "macro_quality_retention": interval(valid_retention),
        },
        "bootstrap_probabilities": {
            "quality_retention_ge_0p95": float(
                np.mean(valid_retention >= 0.95)
            ),
            "quality_retention_ge_0p98": float(
                np.mean(valid_retention >= 0.98)
            ),
            "ours_macro_ge_full": float(
                np.mean(bootstrap_delta >= 0.0)
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_samples.csv", pair_rows)
    write_csv(args.output_dir / "per_task.csv", task_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
