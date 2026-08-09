from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import qksieve_robust_contract_20260810 as contract  # noqa: E402
from verify_qksieve_robust_paper_evidence_20260810 import verify  # noqa: E402


def test_audit_accepts_complete_synthetic_evidence() -> None:
    persistent_rows = [
        {
            "history_tokens": length,
            "warm_speedup": 2.0,
            "cold_speedup": 0.5,
            "amortized_speedup": 1.5,
            "append_only_speedup": 2.0,
            "reuse_tokens_equal": True,
            "index_buffers_reused_without_rebuild": True,
            "rewind_value_layers_correct": True,
            "persistent_contract_passed": True,
            "independent_lifecycle_audit": {
                "layers": 32,
                "snapshots": 6,
                "rewinds": 5,
                "post_decode_index_lag_tokens": 1,
                "all_index_buffers_stable": True,
                "deterministic_replay": True,
            },
        }
        for length in (32768, 65536)
    ]
    ruler_length_samples = {
        4096: 10,
        8192: 10,
        16384: 10,
        32768: 10,
        65536: 5,
        131072: 5,
    }
    ruler_tasks = (
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multivalue",
        "niah_multiquery",
        "vt",
        "cwe",
        "fwe",
        "qa_squad",
        "qa_hotpot",
    )
    ruler_lengths = {
        str(length): {"quality_retention": 1.0, "cells": 13}
        for length in ruler_length_samples
    }
    ruler_cells = {
        f"{task}@{length}": {
            "task": task,
            "length": length,
            "samples": samples,
        }
        for task in ruler_tasks
        for length, samples in ruler_length_samples.items()
    }
    model_per_task = {
        f"task{i}": {"samples": 10} for i in range(16)
    }
    model_row = {
        "strict_pairs": 160,
        "tasks": 16,
        "full_fallback_count": 0,
        "quality_retention": 1.0,
        "quality_retention_95ci": [0.99, 1.01],
        "mean_attention_fraction": 0.06,
        "per_task": model_per_task,
    }
    attention_rows = [
        {
            "history_tokens": length,
            "full_mha_ms": 8.0,
            "qksieve_robust_ms": 2.0,
            "qksieve_fast_ms": 1.0,
            "fier_ms": 4.0,
            "robust_speedup": 4.0,
            "fast_speedup": 8.0,
            "fier_speedup": 2.0,
            "robust_vs_fier": 2.0,
        }
        for length in (65536, 131072)
    ]
    decode_rows = [
        {
            "history_tokens": length,
            "full_steady_ms_per_token": 12.0,
            "qksieve_steady_ms_per_token": 4.0,
            "steady_decode_speedup": 3.0,
            "qksieve_prebuild_seconds_median": 1.0,
        }
        for length in (65536, 131072)
    ]
    persistent_system_rows = [
        {
            "history_tokens": length,
            "cold_speedup": 0.5,
            "warm_speedup": 2.0,
            "shared_prefix_amortized_speedup": 1.5,
            "append_only_speedup": 2.0,
            "qksieve_index_build_seconds_median": 1.0,
        }
        for length in (65536, 131072)
    ]
    longbench_per_task = {
        f"task{i}": {
            "samples": 235 if i < 6 else 234,
            "full_kv": {"score": 1.0},
            contract.METHOD: {"score": 1.0},
        }
        for i in range(16)
    }

    report = verify(
        PROJECT_ROOT,
        persistent={
            "schema": "qksieve_persistent_kv_summary_v2",
            "all_correct": True,
            "rows": persistent_rows,
        },
        longbench={
            "schema": "qksieve_robust_longbench_summary_v1",
            "frozen_contract": contract.contract_payload(),
            "strict_pairs": 3750,
            "rows": 7500,
            "tasks": 16,
            "full_fallback_count": 0,
            "value_sketch_tail_alpha": contract.VALUE_SKETCH_TAIL_ALPHA,
            "methods": {
                "full_kv": {"macro_score": 1.0},
                contract.METHOD: {
                    "macro_score": 1.0,
                    "quality_retention": 1.0,
                },
            },
            "per_task": longbench_per_task,
            "bootstrap": {
                "resamples": 10000,
                "macro_score_delta_95ci": [-0.01, 0.01],
                "quality_retention_95ci": [0.99, 1.01],
            },
        },
        ruler={
            "schema": "qksieve_robust_ruler_summary_v1",
            "frozen_contract": contract.contract_payload(),
            "strict_pairs": 650,
            "rows": 1300,
            "tasks": list(ruler_tasks),
            "length_samples": ruler_length_samples,
            "per_length": ruler_lengths,
            "per_task_length": ruler_cells,
            "overall": {"quality_retention": 1.0, "cells": 78},
            "bootstrap": {
                "resamples": 10000,
                "macro_score_delta_95ci": [-0.01, 0.01],
                "quality_retention_95ci": [0.99, 1.01],
            },
            "fallback_count": 0,
        },
        multimodel={
            "schema": "qksieve_robust_multimodel_summary_v1",
            "frozen_contract": contract.contract_payload(),
            "models": {
                tag: dict(model_row)
                for tag in ("llama31_8b", "qwen3_4b", "mistral_7b")
            },
        },
        h100={
            "schema": "qksieve_h100_matched_system_summary_v1",
            "expected_seeds": 3,
            "hardware": {"device_names": ["NVIDIA H100 80GB HBM3"]},
            "frozen_contract": contract.contract_payload(),
            "methods": {"main": contract.METHOD},
            "attention": attention_rows,
            "steady_decode": decode_rows,
            "persistent_requests": persistent_system_rows,
            "claim_boundary": "matched H100 evidence",
        },
    )

    assert report["complete"]
    assert not report["missing"]


def test_audit_reports_missing_sections() -> None:
    report = verify(PROJECT_ROOT)
    assert not report["complete"]
    assert set(report["missing"]) == {
        "persistent",
        "longbench",
        "ruler",
        "multimodel",
        "h100",
    }


def test_audit_rejects_persistent_summary_without_independent_audit() -> None:
    rows = [
        {
            "history_tokens": length,
            "warm_speedup": 2.0,
            "cold_speedup": 0.5,
            "amortized_speedup": 1.5,
            "append_only_speedup": 2.0,
            "reuse_tokens_equal": True,
            "index_buffers_reused_without_rebuild": True,
            "rewind_value_layers_correct": True,
            "persistent_contract_passed": True,
        }
        for length in (32768, 65536)
    ]
    try:
        verify(
            PROJECT_ROOT,
            persistent={
                "schema": "qksieve_persistent_kv_summary_v2",
                "all_correct": True,
                "rows": rows,
            },
        )
    except AssertionError as error:
        assert "independent audit" in str(error)
    else:
        raise AssertionError("missing independent audit was accepted")


def test_audit_rejects_non_h100_system_evidence() -> None:
    rows = [
        {
            "history_tokens": length,
            "full_mha_ms": 8.0,
            "qksieve_robust_ms": 2.0,
            "qksieve_fast_ms": 1.0,
            "fier_ms": 4.0,
            "robust_speedup": 4.0,
            "fast_speedup": 8.0,
            "fier_speedup": 2.0,
            "robust_vs_fier": 2.0,
        }
        for length in (65536, 131072)
    ]
    try:
        verify(
            PROJECT_ROOT,
            h100={
                "schema": "qksieve_h100_matched_system_summary_v1",
                "expected_seeds": 3,
                "hardware": {"device_names": ["NVIDIA RTX 3090"]},
                "frozen_contract": contract.contract_payload(),
                "methods": {"main": contract.METHOD},
                "attention": rows,
                "steady_decode": [],
                "persistent_requests": [],
                "claim_boundary": "synthetic",
            },
        )
    except AssertionError as error:
        assert "non-H100" in str(error)
    else:
        raise AssertionError("non-H100 evidence was accepted")
