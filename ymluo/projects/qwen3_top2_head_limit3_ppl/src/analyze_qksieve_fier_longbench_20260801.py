#!/usr/bin/env python
"""Join a FIER-only run with frozen matched-sample LongBench controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FULL = "full_kv"
QKSIEVE = "qksieve_fullprompt_auto_plain_fulltopk"
FIER = "fier_rtn1_g32_fulltopk"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_root", type=Path, required=True)
    parser.add_argument("--fier_root", type=Path, required=True)
    parser.add_argument("--expected_pairs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(root: Path) -> tuple[list[dict[str, str]], list[Path]]:
    paths = sorted(root.glob("shard[0-9]*/sample_results.csv"))
    if not paths:
        raise ValueError(f"no shard sample_results.csv files under {root}")
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows, paths


def sample_key(row: dict[str, str]) -> tuple[str, str]:
    return row["task"], row["sample_id"]


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def macro_score(rows: dict[tuple[str, str], dict[str, str]]) -> float:
    by_task: dict[str, list[float]] = defaultdict(list)
    for (task, _), row in rows.items():
        by_task[task].append(float(row["score"]))
    return mean([mean(scores) for scores in by_task.values()])


def unique_method_rows(
    rows: list[dict[str, str]], method: str
) -> dict[tuple[str, str], dict[str, str]]:
    selected = [row for row in rows if row["method"] == method]
    indexed = {sample_key(row): row for row in selected}
    if len(indexed) != len(selected):
        raise ValueError(f"duplicate {method} task/sample rows")
    return indexed


def analyze(
    reference_rows: list[dict[str, str]],
    fier_rows: list[dict[str, str]],
    *,
    expected_pairs: int,
) -> dict[str, Any]:
    full = unique_method_rows(reference_rows, FULL)
    qksieve = unique_method_rows(reference_rows, QKSIEVE)
    fier = unique_method_rows(fier_rows, FIER)
    expected_keys = set(full)
    if not (
        len(expected_keys) == expected_pairs
        and set(qksieve) == expected_keys
        and set(fier) == expected_keys
    ):
        raise ValueError(
            "FIER comparison is not strict matched-sample data: "
            f"full={len(full)} qksieve={len(qksieve)} "
            f"fier={len(fier)} expected={expected_pairs}"
        )
    tasks = sorted({task for task, _ in expected_keys})
    if len(tasks) != 16:
        raise ValueError(f"expected 16 LongBench tasks, got {len(tasks)}")

    for key in sorted(expected_keys):
        qksieve_row = qksieve[key]
        fier_row = fier[key]
        if fier_row["executed_path"] != FIER:
            raise ValueError(f"{key}: FIER executed_path mismatch")
        if fier_row["configured_score_mode"] != "fier_rtn1_g32_fulltopk":
            raise ValueError(f"{key}: FIER score-mode mismatch")
        if abs(float(fier_row["configured_index_bits_per_token"]) - 256.0) > 1e-6:
            raise ValueError(f"{key}: FIER index-rate mismatch")
        if (
            fier_row["configured_attention_tokens"]
            != qksieve_row["configured_attention_tokens"]
        ):
            raise ValueError(f"{key}: active-token budget mismatch")
        for field in ("prompt_tokens", "prefix_tokens", "suffix_tokens"):
            if fier_row[field] != qksieve_row[field]:
                raise ValueError(f"{key}: {field} protocol mismatch")

    full_macro = macro_score(full)
    method_rows = {QKSIEVE: qksieve, FIER: fier}
    methods: dict[str, Any] = {}
    for method, indexed in method_rows.items():
        score = macro_score(indexed)
        methods[method] = {
            "macro_score": score,
            "quality_retention": score / full_macro if full_macro else None,
            "configured_index_bits_per_token_per_kv_head": (
                240.0 if method == QKSIEVE else 256.0
            ),
            "mean_active_token_ratio": mean(
                [
                    min(
                        int(row["prefix_tokens"]),
                        int(float(row["configured_attention_tokens"])),
                    )
                    / max(1, int(row["prefix_tokens"]))
                    for row in indexed.values()
                ]
            ),
        }

    per_task: dict[str, Any] = {}
    for task in tasks:
        task_keys = sorted(key for key in expected_keys if key[0] == task)
        full_score = mean([float(full[key]["score"]) for key in task_keys])
        per_task[task] = {"samples": len(task_keys), "full": full_score}
        for method, indexed in method_rows.items():
            score = mean([float(indexed[key]["score"]) for key in task_keys])
            per_task[task][method] = {
                "score": score,
                "relative_full": score / full_score if full_score else None,
            }

    return {
        "schema": "qksieve_fier_matched_longbench_v1",
        "strict_pairs": expected_pairs,
        "tasks": len(tasks),
        "full_macro": full_macro,
        "methods": methods,
        "per_task": per_task,
        "fairness_contract": {
            "same_samples": True,
            "same_prompt_protocol": True,
            "same_length_only_active_token_schedule": True,
            "same_exact_selected_kv_attention_consumer": True,
            "full_fallback": False,
            "exact_candidate_rerank": False,
            "recent_or_sink_reservation": False,
            "fier_variant": "RTN-1 with group size 32",
        },
        "latency_claim": {
            "valid": False,
            "reason": (
                "This run is a deterministic reference-quality control. "
                "Direct packed CUDA selector timing is reported separately."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    reference_rows, reference_paths = read_rows(args.reference_root)
    fier_rows, fier_paths = read_rows(args.fier_root)
    report = analyze(
        reference_rows, fier_rows, expected_pairs=args.expected_pairs
    )
    project_root = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__),
        project_root / "src/run_sample_calibrated_longbench_20260717.py",
        project_root / "src/run_head_top2_targeted_ppl_20260714.py",
        project_root / "src/fier_rtn1_cuda_20260728.py",
    ]
    report["source_sha256"] = {
        str(path.relative_to(project_root)): sha256(path)
        for path in source_paths
    }
    report["input_sha256"] = {
        **{
            f"reference/{path.relative_to(args.reference_root)}": sha256(path)
            for path in reference_paths
        },
        **{
            f"fier/{path.relative_to(args.fier_root)}": sha256(path)
            for path in fier_paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
