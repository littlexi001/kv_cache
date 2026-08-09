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
    ruler_lengths = {
        str(length): {"quality_retention": 1.0}
        for length in (4096, 8192, 16384, 32768, 65536, 131072)
    }
    model_row = {
        "strict_pairs": 160,
        "tasks": 16,
        "quality_retention_95ci": [0.99, 1.01],
    }
    system_rows = [
        {"history_tokens": length} for length in (65536, 131072)
    ]

    report = verify(
        PROJECT_ROOT,
        persistent={
            "schema": "qksieve_persistent_kv_summary_v2",
            "all_correct": True,
            "rows": persistent_rows,
        },
        ruler={
            "schema": "qksieve_robust_ruler_summary_v1",
            "frozen_contract": contract.contract_payload(),
            "strict_pairs": 650,
            "tasks": [f"task{i}" for i in range(13)],
            "per_length": ruler_lengths,
            "overall": {"quality_retention": 1.0},
            "bootstrap": {
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
            "attention": system_rows,
            "steady_decode": system_rows,
            "persistent_requests": system_rows,
        },
    )

    assert report["complete"]
    assert not report["missing"]


def test_audit_reports_missing_sections() -> None:
    report = verify(PROJECT_ROOT)
    assert not report["complete"]
    assert set(report["missing"]) == {
        "persistent",
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
