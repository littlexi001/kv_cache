from __future__ import annotations

import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import qksieve_robust_contract_20260810 as contract  # noqa: E402
from merge_qksieve_ruler_distributed_20260810 import merge_rows  # noqa: E402


def row(task: str, sample_id: str, method: str, prediction: str = "ok") -> dict[str, str]:
    base_task, _, length = task.rpartition("_")
    result = {
        "task": task,
        "base_task": base_task,
        "requested_length": length,
        "sample_id": sample_id,
        "method": method,
        "executed_path": method,
        "configured_index_bits_per_token": "306.0",
        "packed_qmse_sample_count": "512.0",
        "packed_qmse_value_sketch_rank": "16.0",
        "packed_qmse_value_sketch_bits": "4.0",
        "packed_qmse_value_sketch_executed": "1.0",
        "packed_qmse_value_sketch_tail_alpha": "0.5",
        "packed_qmse_debug_value_sketch_disabled": "0.0",
        "sampled_quantile_fallback": "0.0",
        "configured_score_mode": contract.SCORE_MODE,
        "configured_attention_fraction": "0.02",
        "configured_attention_tokens": "1280",
        "configured_candidate_fraction": "0.02",
        "configured_projection_dim": "128",
        "diagnostics_enabled": "True",
        "prediction": prediction,
        "score": "1.0",
        "prefill_seconds": "1.0",
        "query_seconds": "1.0",
        "decode_seconds": "1.0",
        "online_seconds": "2.0",
        "total_seconds": "3.0",
    }
    if method == "full_kv":
        result["configured_score_mode"] = "full_kv"
        for name in (
            "configured_index_bits_per_token",
            "packed_qmse_sample_count",
            "packed_qmse_value_sketch_rank",
            "packed_qmse_value_sketch_bits",
            "packed_qmse_value_sketch_executed",
            "packed_qmse_value_sketch_tail_alpha",
            "packed_qmse_debug_value_sketch_disabled",
            "sampled_quantile_fallback",
        ):
            result[name] = ""
    return result


def pair(task: str, sample_id: str) -> list[dict[str, str]]:
    return [
        row(task, sample_id, "full_kv"),
        row(task, sample_id, contract.METHOD),
    ]


def test_merge_prefers_primary_and_fills_missing_pair() -> None:
    primary = pair("task_a_4096", "a")
    supplement = pair("task_a_4096", "a") + pair("task_a_4096", "b")
    merged, audit = merge_rows(
        primary,
        supplement,
        expected_tasks=("task_a",),
        expected_length_samples={4096: 2},
    )
    assert len(merged) == 4
    assert audit["strict_pairs"] == 2
    assert audit["duplicate_rows_primary_preferred"] == 2
    assert audit["duplicate_output_mismatches"] == 0


def test_merge_rejects_incomplete_grid() -> None:
    with pytest.raises(AssertionError, match="incomplete pairs"):
        merge_rows(
            [row("task_a_4096", "a", "full_kv")],
            [],
            expected_tasks=("task_a",),
            expected_length_samples={4096: 1},
        )


def test_merge_audits_duplicate_output_mismatch() -> None:
    primary = pair("task_a_4096", "a")
    supplement = pair("task_a_4096", "a")
    supplement[1]["prediction"] = "different"
    merged, audit = merge_rows(
        primary,
        supplement,
        expected_tasks=("task_a",),
        expected_length_samples={4096: 1},
    )
    assert len(merged) == 2
    assert audit["duplicate_output_mismatches"] == 1


def test_merge_rejects_frozen_config_drift() -> None:
    primary = pair("task_a_4096", "a")
    supplement = pair("task_a_4096", "a")
    supplement[1]["packed_qmse_sample_count"] = "1024.0"
    with pytest.raises(AssertionError, match="packed_qmse_sample_count drifted"):
        merge_rows(
            primary,
            supplement,
            expected_tasks=("task_a",),
            expected_length_samples={4096: 1},
        )
