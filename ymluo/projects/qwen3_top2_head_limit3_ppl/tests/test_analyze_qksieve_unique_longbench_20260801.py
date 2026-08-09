from __future__ import annotations

import sys
from pathlib import Path

import pytest


SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))
from analyze_qksieve_unique_longbench_20260801 import (  # noqa: E402
    FULL,
    QKSIEVE,
    UNIQUE,
    active_ratio,
    analyze,
)


def row(task: str, sample: int, method: str) -> dict[str, str]:
    score_mode = {
        FULL: "full_kv",
        QKSIEVE: "pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk",
        UNIQUE: "unique_p8_meanstd_fulltopk",
    }[method]
    return {
        "task": task,
        "sample_id": str(sample),
        "method": method,
        "executed_path": method,
        "score": "0.5",
        "configured_score_mode": score_mode,
        "configured_index_bits_per_token": "258" if method == UNIQUE else "240",
        "configured_attention_tokens": "256",
        "prompt_tokens": "4096",
        "prefix_tokens": "4096",
        "suffix_tokens": "0",
        "selected_history_fraction_mean": "0.0625",
    }


def test_unique_report_requires_strict_pairs_and_formula_contract() -> None:
    reference = []
    unique = []
    for task_index in range(16):
        task = f"task-{task_index}"
        reference.extend((row(task, 0, FULL), row(task, 0, QKSIEVE)))
        unique.append(row(task, 0, UNIQUE))
    report = analyze(reference, unique, expected_pairs=16)
    assert report["strict_pairs"] == 16
    assert report["methods"][UNIQUE]["quality_retention"] == 1.0
    assert report["methods"][UNIQUE]["mean_loaded_token_ratio"] == 0.0625
    assert report["latency_claim"]["valid"] is False


def test_unique_report_rejects_budget_mismatch() -> None:
    reference = []
    unique = []
    for task_index in range(16):
        task = f"task-{task_index}"
        reference.extend((row(task, 0, FULL), row(task, 0, QKSIEVE)))
        unique.append(row(task, 0, UNIQUE))
    unique[0]["configured_attention_tokens"] = "512"
    with pytest.raises(ValueError, match="budget mismatch"):
        analyze(reference, unique, expected_pairs=16)


def test_unique_ratio_page_rounds_when_runtime_diagnostics_are_absent() -> None:
    value = row("task", 0, UNIQUE)
    value["prefix_tokens"] = "7501"
    value["configured_attention_tokens"] = "451"
    value["selected_history_fraction_mean"] = "0.0"
    assert active_ratio(value, UNIQUE) == 456 / 7501
