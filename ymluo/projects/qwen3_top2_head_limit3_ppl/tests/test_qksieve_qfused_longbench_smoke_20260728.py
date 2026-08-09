from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import analyze_qksieve_qfused_longbench_smoke_20260728 as smoke  # noqa: E402


def test_cuda_validator_does_not_hardcode_gqa4_query_heads() -> None:
    validator = (
        SRC / "validate_qksieve_qfused_cuda_20260728.py"
    ).read_text(encoding="utf-8")

    assert "grouped_query.shape[1] * args.group_count" in validator
    assert ".reshape(1, 32, history_tokens)" not in validator
    assert "paired_measure_ms" in validator
    assert "statistics.median" in validator


def _row(method: str, sample_id: str) -> dict[str, str]:
    score_mode = {
        smoke.FULL: "",
        smoke.FROZEN: smoke.FROZEN_SCORE_MODE,
        smoke.QFUSED: smoke.QFUSED_SCORE_MODE,
    }[method]
    is_qfused = method == smoke.QFUSED
    is_sparse = method != smoke.FULL
    return {
        "task": "narrativeqa",
        "sample_id": sample_id,
        "method": method,
        "executed_path": method,
        "configured_score_mode": score_mode,
        "diagnostics_enabled": "True",
        "configured_attention_fraction": "0.06",
        "configured_attention_tokens": "450",
        "configured_candidate_fraction": "0.06",
        "configured_projection_dim": "48",
        "packed_qmse_index_bits_per_token": "240",
        "packed_qmse_fused_query_prepare_requested": (
            "1" if is_qfused else "0"
        ),
        "packed_qmse_fused_query_prepare_executed": (
            "1" if is_qfused else "0"
        ),
        "packed_qmse_allocation_frozen_before_query": (
            "1" if is_sparse else "0"
        ),
        "prediction": f"answer {sample_id}",
        "score": "0.8",
        "query_seconds": "0.1",
        "decode_seconds": "0.4",
        "online_seconds": "0.5",
    }


def test_paired_smoke_accepts_proven_qfused_execution() -> None:
    rows = [
        _row(method, sample_id)
        for sample_id in ("0", "1")
        for method in smoke.EXPECTED_METHODS
    ]

    report = smoke.analyze(
        rows,
        {"all_passed": True},
        min_prediction_match=0.875,
        max_mean_score_delta=0.01,
    )

    assert report["complete_triplets"]
    assert report["execution"]["proven"]
    assert report["quality"]["passed"]
    assert report["promotion_smoke_passed"]
    assert not report["timing_is_promotion_evidence"]


def test_paired_smoke_rejects_requested_but_unexecuted_fusion() -> None:
    rows = [
        _row(method, "0")
        for method in smoke.EXPECTED_METHODS
    ]
    qfused = next(row for row in rows if row["method"] == smoke.QFUSED)
    qfused["packed_qmse_fused_query_prepare_executed"] = "0"

    report = smoke.analyze(
        rows,
        {"all_passed": True},
        min_prediction_match=0.875,
        max_mean_score_delta=0.01,
    )

    assert not report["execution"]["proven"]
    assert not report["promotion_smoke_passed"]
