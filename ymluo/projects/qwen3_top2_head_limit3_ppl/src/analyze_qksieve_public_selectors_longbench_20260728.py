#!/usr/bin/env python
"""Strict paired LongBench quality report for public selector controls.

Quest and both SparQ references share QKSieve's active-token schedule.  The
formula-complete SparQ path adds the paper's local mask, temperature, selected
mass, and mean-Value correction.  These are reference PyTorch paths, so this
report deliberately does not compare their latency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FULL = "full_kv"
QKSIEVE = "qksieve_fullprompt_auto_plain_fulltopk"
QUEST = "quest_p16_fullprompt_matchedbudget"
RABITQ = "rabitqcache_rtn1_fullprompt_matchedbudget"
SPARQ_SELECTOR = "sparq_r32_selector_fullprompt_matchedbudget"
SPARQ_FORMULA = "sparq_r32_formula_fullprompt_matchedbudget"
EXPECTED_METHODS = (
    FULL,
    QKSIEVE,
    QUEST,
    RABITQ,
    SPARQ_SELECTOR,
    SPARQ_FORMULA,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--expected_pairs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def key(row: dict[str, str]) -> tuple[str, str]:
    return row["task"], row["sample_id"]


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
    return sum(values) / len(values)


def macro_score(rows: dict[tuple[str, str], dict[str, str]]) -> float:
    by_task: dict[str, list[float]] = defaultdict(list)
    for (task, _), row in rows.items():
        by_task[task].append(float(row["score"]))
    return mean([mean(scores) for scores in by_task.values()])


def configured_loaded_tokens(row: dict[str, str]) -> int:
    history = int(row["prefix_tokens"])
    target = min(history, int(float(row["configured_attention_tokens"])))
    if row["method"] == QUEST:
        return min(history, math.ceil(target / 16) * 16)
    return target


def analyze(
    rows: list[dict[str, str]],
    *,
    expected_pairs: int,
) -> dict[str, Any]:
    by_method = {
        method: {
            key(row): row
            for row in rows
            if row["method"] == method
        }
        for method in EXPECTED_METHODS
    }
    method_counts = Counter(row["method"] for row in rows)
    unexpected = sorted(set(method_counts) - set(EXPECTED_METHODS))
    expected_keys = set(by_method[FULL])
    complete = bool(
        not unexpected
        and len(expected_keys) == expected_pairs
        and all(set(by_method[method]) == expected_keys for method in EXPECTED_METHODS)
        and len(rows) == expected_pairs * len(EXPECTED_METHODS)
    )
    if not complete:
        raise ValueError(
            "public-selector LongBench rows are not strict complete pairs: "
            f"counts={method_counts}, expected_pairs={expected_pairs}"
        )
    if len({task for task, _ in expected_keys}) != 16:
        raise ValueError("expected all 16 English LongBench tasks")

    score_modes = {
        QKSIEVE: (
            "pca_hierarchical_autoqmsetotal15z_"
            "qkmetric_packed_fulltopk"
        ),
        QUEST: "quest_p16_fulltopk",
        RABITQ: "rabitqcache_rtn1_fulltopk",
        SPARQ_SELECTOR: "sparq_r32_selector_fulltopk",
        SPARQ_FORMULA: "sparq_r32_meanvalue_fulltopk",
    }
    index_bits = {
        QKSIEVE: 240.0,
        QUEST: 256.0,
        RABITQ: 224.0,
        SPARQ_SELECTOR: 0.0,
        SPARQ_FORMULA: 0.0,
    }
    contract_errors: list[str] = []
    for method in (QKSIEVE, QUEST, RABITQ, SPARQ_SELECTOR, SPARQ_FORMULA):
        for sample_key, row in by_method[method].items():
            if row["executed_path"] != method:
                contract_errors.append(f"{sample_key}/{method}: executed_path")
            if row["configured_score_mode"] != score_modes[method]:
                contract_errors.append(f"{sample_key}/{method}: score_mode")
            if (
                abs(
                    float(row["configured_index_bits_per_token"])
                    - index_bits[method]
                )
                > 1.0e-6
            ):
                contract_errors.append(f"{sample_key}/{method}: index rate")
            qksieve_row = by_method[QKSIEVE][sample_key]
            if (
                row["configured_attention_tokens"]
                != qksieve_row["configured_attention_tokens"]
            ):
                contract_errors.append(f"{sample_key}/{method}: token budget")
    if contract_errors:
        raise ValueError(
            "public selector contract errors: "
            + "; ".join(contract_errors[:20])
        )

    full_macro = macro_score(by_method[FULL])
    methods: dict[str, Any] = {}
    for method in (QKSIEVE, QUEST, RABITQ, SPARQ_SELECTOR, SPARQ_FORMULA):
        method_macro = macro_score(by_method[method])
        loaded_ratios = [
            configured_loaded_tokens(row) / max(1, int(row["prefix_tokens"]))
            for row in by_method[method].values()
        ]
        methods[method] = {
            "macro_score": method_macro,
            "quality_retention": (
                method_macro / full_macro if full_macro else None
            ),
            "configured_mean_loaded_token_ratio": mean(loaded_ratios),
            "configured_index_bits_per_token_per_kv_head": index_bits[method],
            "score_mode": score_modes[method],
        }

    per_task: dict[str, Any] = {}
    for task in sorted({task for task, _ in expected_keys}):
        task_keys = sorted(key for key in expected_keys if key[0] == task)
        full_score = mean(
            [float(by_method[FULL][sample_key]["score"]) for sample_key in task_keys]
        )
        per_task[task] = {
            "samples": len(task_keys),
            "full": full_score,
            **{
                method: {
                    "score": (
                        method_score := mean(
                            [
                                float(by_method[method][sample_key]["score"])
                                for sample_key in task_keys
                            ]
                        )
                    ),
                    "relative_full": (
                        method_score / full_score if full_score else None
                    ),
                }
                for method in (
                    QKSIEVE,
                    QUEST,
                    RABITQ,
                    SPARQ_SELECTOR,
                    SPARQ_FORMULA,
                )
            },
        }

    return {
        "schema": "qksieve_public_selector_longbench_v1",
        "strict_pairs": expected_pairs,
        "rows": len(rows),
        "tasks": 16,
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
            "quest_page_size": 16,
            "quest_loaded_tokens_are_page_granular": True,
            "rabitq_variant": (
                "official centered random-rotation 1-bit estimator; "
                "full-prefill Query/Key centroids; matched top-k budget"
            ),
            "sparq_selector_variant": (
                "selector-only control; excludes SparQ mean-Value correction"
            ),
            "sparq_formula_variant": (
                "r=32; paper temperature; local window=floor(k/4); "
                "approximate selected mass; running mean-Value correction"
            ),
        },
        "latency_claim": {
            "valid": False,
            "reason": (
                "Quest and SparQ paths are reference PyTorch paths; "
                "kernel-fair latency requires optimized implementations."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.run_root.glob("shard[0-9]*/sample_results.csv"))
    if not paths:
        raise SystemExit("no shard sample_results.csv files found")
    report = analyze(
        read_rows(paths),
        expected_pairs=args.expected_pairs,
    )
    project_root = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__),
        project_root / "src/run_sample_calibrated_longbench_20260717.py",
        project_root / "src/run_head_top2_targeted_ppl_20260714.py",
        project_root / "src/qabs_cuda_kernels.py",
    ]
    report["source_sha256"] = {
        str(path.relative_to(project_root)): sha256(path)
        for path in source_paths
    }
    report["input_sha256"] = {
        str(path.relative_to(args.run_root)): sha256(path)
        for path in paths
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
