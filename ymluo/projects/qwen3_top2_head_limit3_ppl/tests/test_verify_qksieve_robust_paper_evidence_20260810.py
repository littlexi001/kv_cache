from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import qksieve_robust_contract_20260810 as contract  # noqa: E402
from verify_qksieve_robust_paper_evidence_20260810 import (  # noqa: E402
    AUDITED_IMPLEMENTATION_SHA,
    PERSISTENT_MODEL_HASHES,
    PERSISTENT_RUN_MANIFEST_SHA,
    PERSISTENT_SOFTWARE,
    validate_frozen_sources,
    verify,
)


def persistent_protocol_audit() -> dict:
    return {
        "schema": "qksieve_persistent_run_protocol_audit_v1",
        "passed": True,
        "manifest_sha256": PERSISTENT_RUN_MANIFEST_SHA,
        "audited_implementation_commit_sha": AUDITED_IMPLEMENTATION_SHA,
        "lengths": [32768, 65536],
        "seeds": [20260810, 20260811, 20260812],
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "driver": "555.42.02",
        "software": PERSISTENT_SOFTWARE,
        "model_hashes": PERSISTENT_MODEL_HASHES,
    }


def test_frozen_source_manifest_accepts_exact_bytes_and_rejects_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "kernel.py"
    source.parent.mkdir(parents=True)
    source.write_text("frozen = True\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    config = tmp_path / "configs"
    config.mkdir()
    (config / "qksieve_robust_source_manifest_20260810.json").write_text(
        json.dumps(
            {
                "schema": "qksieve_frozen_source_manifest_v1",
                "audited_implementation_commit_sha": (
                    "f300fb280a597ceb124d454cdfc9a0a1665d6a04"
                ),
                "recorded_from_commit": "test",
                "files": {"src/kernel.py": digest},
            }
        ),
        encoding="utf-8",
    )

    assert validate_frozen_sources(tmp_path)["files"]["src/kernel.py"] == digest
    source.write_text("frozen = False\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="frozen source drifted"):
        validate_frozen_sources(tmp_path)


def test_audit_accepts_complete_synthetic_evidence() -> None:
    persistent_rows = [
        {
            "history_tokens": length,
            "seed": seed,
            "warm_speedup": 2.0,
            "cold_speedup": 0.5,
            "cold_end_to_end_speedup": 0.4,
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
        for seed in (20260810, 20260811, 20260812)
    ]
    aggregate_rows = [
        {
            "history_tokens": length,
            "seed_count": 3,
            "seeds": [20260810, 20260811, 20260812],
            **{
                key: value
                for field, value in {
                    "full_warm_ms_per_token": 100.0,
                    "qksieve_warm_ms_per_token": 50.0,
                    "full_cold_end_to_end_ms_per_token": 200.0,
                    "qksieve_cold_end_to_end_ms_per_token": 500.0,
                    "warm_speedup": 2.0,
                    "cold_speedup": 0.5,
                    "cold_end_to_end_speedup": 0.4,
                    "amortized_speedup": 1.5,
                    "append_only_speedup": 2.0,
                    "qksieve_prebuild_seconds": 1.0,
                }.items()
                for key in (
                    field,
                    f"{field}_bootstrap_ci95_low",
                    f"{field}_bootstrap_ci95_high",
                )
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
    ruler_bootstrap = {
        "resamples": 10000,
        "macro_score_delta_95ci": [-0.01, 0.01],
        "quality_retention_95ci": [0.99, 1.01],
    }
    ruler_lengths = {
        str(length): {
            "quality_retention": 1.0,
            "cells": 13,
            "bootstrap": dict(ruler_bootstrap),
        }
        for length in ruler_length_samples
    }
    ruler_cells = {
        f"{task}@{length}": {
            "task": task,
            "length": length,
            "samples": samples,
            "full_kv": {"score": 1.0},
            "bootstrap": dict(ruler_bootstrap),
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
    shrinkage_rows = []
    for shrinkage in (0.0, 0.25, 0.5, 0.75, 0.9):
        for fraction in (0.01, 0.02, 0.04):
            row = {
                "shrinkage": shrinkage,
                "selected_fraction": fraction,
                "conditions": 100,
            }
            for metric in (
                "top2_recall",
                "selected_attention_mass",
                "top2_attention_mass_recall",
                "score_pearson",
                "score_rmse",
            ):
                row[metric] = 0.9
                row[f"{metric}_paired"] = {
                    "clusters": 20,
                    "delta_vs_production": 0.0,
                    "ci95": [-0.01, 0.01],
                }
            shrinkage_rows.append(row)
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
            "full_kv_bytes": 1000.0,
            "qksieve_key_index_bytes": 50.0,
            "qksieve_valuesketch_bytes": 20.0,
            "qksieve_total_auxiliary_bytes": 70.0,
            "qksieve_key_index_ratio_of_full_kv": 0.05,
            "qksieve_total_auxiliary_ratio_of_full_kv": 0.07,
            "fier_index_bytes": 10.0,
            "fier_index_ratio_of_full_kv": 0.01,
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
            "full_peak_allocated_bytes_total": 100.0,
            "qksieve_peak_allocated_bytes_total": 120.0,
            "qksieve_to_full_peak_allocated_ratio": 1.2,
            "full_peak_reserved_bytes_total": 200.0,
            "qksieve_peak_reserved_bytes_total": 240.0,
            "qksieve_to_full_peak_reserved_ratio": 1.2,
        }
        for length in (65536, 131072)
    ]
    persistent_system_rows = [
        {
            "history_tokens": length,
            "cold_speedup": 0.5,
            "cold_end_to_end_speedup": 0.5,
            "warm_speedup": 2.0,
            "shared_prefix_amortized_speedup": 1.5,
            "append_only_speedup": 2.0,
            "qksieve_index_build_seconds_median": 1.0,
            "full_cold_peak_allocated_bytes_total": 100.0,
            "qksieve_cold_peak_allocated_bytes_total": 120.0,
            "qksieve_to_full_cold_peak_allocated_ratio": 1.2,
            "full_cold_peak_reserved_bytes_total": 200.0,
            "qksieve_cold_peak_reserved_bytes_total": 240.0,
            "qksieve_to_full_cold_peak_reserved_ratio": 1.2,
            "full_lifecycle_peak_allocated_bytes_total": 110.0,
            "qksieve_lifecycle_peak_allocated_bytes_total": 132.0,
            "qksieve_to_full_lifecycle_peak_allocated_ratio": 1.2,
            "full_lifecycle_peak_reserved_bytes_total": 220.0,
            "qksieve_lifecycle_peak_reserved_bytes_total": 264.0,
            "qksieve_to_full_lifecycle_peak_reserved_ratio": 1.2,
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
            "aggregate_rows": aggregate_rows,
            "missing_pairs": [],
            "statistics": {
                "point_estimate": "median_across_independent_process_repetitions"
            },
            "protocol_audit": persistent_protocol_audit(),
        },
        longbench={
            "schema": "qksieve_robust_longbench_summary_v1",
            "frozen_contract": contract.contract_payload(),
            "strict_pairs": 3750,
            "rows": 7500,
            "tasks": 16,
            "full_fallback_count": 0,
            "value_sketch_tail_alpha": contract.VALUE_SKETCH_TAIL_ALPHA,
            "sample_count_audit": {
                "schema": "qksieve_decode_mean_sample_count_v1",
                "rows": 3750,
                "max_abs_error": 0.0,
            },
            "summarizer_sha256": "0" * 64,
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
        shrinkage={
            "schema": "qksieve_shrinkage_sensitivity_v1",
            "complete": True,
            "method": "qk_balanced",
            "calibration_source": "prefill_tail",
            "production_shrinkage": 0.75,
            "labels": [
                "qwen3_4b_sports32k",
                "qwen3_4b_medicine32k",
                "llama31_8b_sports32k",
                "llama31_8b_medicine32k",
            ],
            "shrinkages": [0.0, 0.25, 0.5, 0.75, 0.9],
            "selected_fractions": [0.01, 0.02, 0.04],
            "strict_paired_conditions": 100,
            "bootstrap_samples": 10000,
            "aggregate": shrinkage_rows,
            "per_label": {
                label: [{} for _ in range(15)]
                for label in (
                    "qwen3_4b_sports32k",
                    "qwen3_4b_medicine32k",
                    "llama31_8b_sports32k",
                    "llama31_8b_medicine32k",
                )
            },
            "acceptance": {
                "passed": True,
                "checks": [{}, {}, {}],
                "failures": [],
            },
            "source_sha256": {"summary.json": "abc"},
            "claim_boundary": "paired mechanism-only sensitivity evidence",
        },
        shrinkage_equivalence={
            "schema": "qksieve_shrinkage_fast_grid_equivalence_v1",
            "passed": True,
            "conditions": 30720,
            "allocation_conditions": 40,
            "condition_keys_identical": True,
            "allocations_identical": True,
            "metrics": {
                name: {
                    "max_abs_difference": 1e-7,
                    "mean_abs_difference": 1e-8,
                    "max_tolerance": 1e-6,
                    "mean_tolerance": 1e-6,
                }
                for name in (
                    "top2_recall",
                    "selected_attention_mass",
                    "oracle_top2_attention_mass",
                    "top2_attention_mass_recall",
                    "score_pearson",
                    "score_rmse",
                )
            },
            "source_sha256": {
                "reference_rows": "a" * 64,
                "candidate_rows": "b" * 64,
                "reference_allocations": "c" * 64,
                "candidate_allocations": "d" * 64,
            },
            "claim_boundary": "one-token numerical ties only",
        },
        h100={
            "schema": "qksieve_h100_matched_system_summary_v1",
            "expected_seeds": 3,
            "hardware": {
                "device_names": ["NVIDIA H100 80GB HBM3"],
                "software": {
                    "python": "3.11.0",
                    "pytorch": "2.7.0",
                    "transformers": "4.55.0",
                    "cuda_runtime": "12.8",
                    "cudnn": 90501,
                },
            },
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
        "shrinkage",
        "shrinkage_equivalence",
        "h100",
    }


def test_audit_rejects_persistent_summary_without_independent_audit() -> None:
    rows = [
        {
            "history_tokens": length,
            "seed": seed,
            "warm_speedup": 2.0,
            "cold_speedup": 0.5,
            "cold_end_to_end_speedup": 0.4,
            "amortized_speedup": 1.5,
            "append_only_speedup": 2.0,
            "reuse_tokens_equal": True,
            "index_buffers_reused_without_rebuild": True,
            "rewind_value_layers_correct": True,
            "persistent_contract_passed": True,
        }
        for length in (32768, 65536)
        for seed in (20260810, 20260811, 20260812)
    ]
    try:
        verify(
            PROJECT_ROOT,
            persistent={
                "schema": "qksieve_persistent_kv_summary_v2",
                "all_correct": True,
                "rows": rows,
                "protocol_audit": persistent_protocol_audit(),
            },
        )
    except AssertionError as error:
        assert "independent audit" in str(error)
    else:
        raise AssertionError("missing independent audit was accepted")


def test_audit_rejects_persistent_summary_without_protocol_audit() -> None:
    persistent_rows = [
        {
            "history_tokens": length,
            "seed": seed,
            "warm_speedup": 2.0,
            "cold_speedup": 0.5,
            "cold_end_to_end_speedup": 0.4,
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
        for seed in (20260810, 20260811, 20260812)
    ]
    aggregate_rows = [
        {
            "history_tokens": length,
            "seed_count": 3,
            "seeds": [20260810, 20260811, 20260812],
            **{
                key: value
                for field, value in {
                    "full_warm_ms_per_token": 100.0,
                    "qksieve_warm_ms_per_token": 50.0,
                    "full_cold_end_to_end_ms_per_token": 200.0,
                    "qksieve_cold_end_to_end_ms_per_token": 500.0,
                    "warm_speedup": 2.0,
                    "cold_speedup": 0.5,
                    "cold_end_to_end_speedup": 0.4,
                    "amortized_speedup": 1.5,
                    "append_only_speedup": 2.0,
                    "qksieve_prebuild_seconds": 1.0,
                }.items()
                for key in (
                    field,
                    f"{field}_bootstrap_ci95_low",
                    f"{field}_bootstrap_ci95_high",
                )
            },
        }
        for length in (32768, 65536)
    ]
    with pytest.raises(AssertionError, match="protocol audit"):
        verify(
            PROJECT_ROOT,
            persistent={
                "schema": "qksieve_persistent_kv_summary_v2",
                "all_correct": True,
                "rows": persistent_rows,
                "aggregate_rows": aggregate_rows,
                "missing_pairs": [],
                "statistics": {
                    "point_estimate": (
                        "median_across_independent_process_repetitions"
                    )
                },
            },
        )


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
