#!/usr/bin/env python
"""Join a UNIQUE-only run with frozen matched-sample LongBench controls."""

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
UNIQUE = "unique_p8_fullprompt_matchedbudget"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_root", type=Path, required=True)
    parser.add_argument("--unique_root", type=Path, required=True)
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


def active_ratio(row: dict[str, str], method: str) -> float:
    measured = row.get("selected_history_fraction_mean", "")
    if measured not in {"", "nan"} and float(measured) > 0.0:
        return float(measured)
    history_count = int(row["prefix_tokens"])
    configured = int(float(row["configured_attention_tokens"]))
    loaded = configured
    if method == UNIQUE:
        loaded = 8 * ((configured + 7) // 8)
    return min(history_count, loaded) / max(1, history_count)


def analyze(
    reference_rows: list[dict[str, str]],
    unique_rows: list[dict[str, str]],
    *,
    expected_pairs: int,
) -> dict[str, Any]:
    full = unique_method_rows(reference_rows, FULL)
    qksieve = unique_method_rows(reference_rows, QKSIEVE)
    unique = unique_method_rows(unique_rows, UNIQUE)
    expected_keys = set(full)
    if not (
        len(expected_keys) == expected_pairs
        and set(qksieve) == expected_keys
        and set(unique) == expected_keys
    ):
        raise ValueError(
            "UNIQUE comparison is not strict matched-sample data: "
            f"full={len(full)} qksieve={len(qksieve)} "
            f"unique={len(unique)} expected={expected_pairs}"
        )
    tasks = sorted({task for task, _ in expected_keys})
    if len(tasks) != 16:
        raise ValueError(f"expected 16 LongBench tasks, got {len(tasks)}")

    for key in sorted(expected_keys):
        qksieve_row = qksieve[key]
        unique_row = unique[key]
        if unique_row["executed_path"] != UNIQUE:
            raise ValueError(f"{key}: UNIQUE executed_path mismatch")
        if unique_row["configured_score_mode"] != "unique_p8_meanstd_fulltopk":
            raise ValueError(f"{key}: UNIQUE score-mode mismatch")
        if abs(float(unique_row["configured_index_bits_per_token"]) - 258.0) > 1e-6:
            raise ValueError(f"{key}: UNIQUE index-rate mismatch")
        if (
            unique_row["configured_attention_tokens"]
            != qksieve_row["configured_attention_tokens"]
        ):
            raise ValueError(f"{key}: configured token budget mismatch")
        for field in ("prompt_tokens", "prefix_tokens", "suffix_tokens"):
            if unique_row[field] != qksieve_row[field]:
                raise ValueError(f"{key}: {field} protocol mismatch")

    full_macro = macro_score(full)
    method_rows = {QKSIEVE: qksieve, UNIQUE: unique}
    methods: dict[str, Any] = {}
    for method, indexed in method_rows.items():
        score = macro_score(indexed)
        methods[method] = {
            "macro_score": score,
            "quality_retention": score / full_macro if full_macro else None,
            "configured_index_bits_per_token_per_kv_head": (
                240.0 if method == QKSIEVE else 258.0
            ),
            "mean_loaded_token_ratio": mean(
                [active_ratio(row, method) for row in indexed.values()]
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
        "schema": "qksieve_unique_matched_longbench_v1",
        "strict_pairs": expected_pairs,
        "tasks": len(tasks),
        "full_macro": full_macro,
        "methods": methods,
        "per_task": per_task,
        "fairness_contract": {
            "same_samples": True,
            "same_prompt_protocol": True,
            "same_length_only_configured_token_schedule": True,
            "page_rounding_reported_as_loaded_token_ratio": True,
            "same_exact_selected_kv_attention_consumer": True,
            "full_fallback": False,
            "exact_candidate_rerank": False,
            "recent_or_sink_reservation": False,
            "unique_variant": (
                "paper Eq. (1)-(6): page size 8, lambda 0.5, "
                "GQA-group maximum"
            ),
        },
        "latency_claim": {
            "valid": False,
            "reason": (
                "Formula-faithful quality reference, not the authors' "
                "unreleased fused CUDA implementation."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    reference_rows, reference_paths = read_rows(args.reference_root)
    unique_rows, unique_paths = read_rows(args.unique_root)
    report = analyze(
        reference_rows, unique_rows, expected_pairs=args.expected_pairs
    )
    project_root = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__),
        project_root / "src/run_sample_calibrated_longbench_20260717.py",
        project_root / "src/run_head_top2_targeted_ppl_20260714.py",
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
            f"unique/{path.relative_to(args.unique_root)}": sha256(path)
            for path in unique_paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
