from __future__ import annotations

import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_qkbalanced_factorial_20260727 import (  # noqa: E402
    AUTO_PLAIN,
    AUTO_QSCALE,
    AUTO_QSCALE_OAS,
    FIXED_PLAIN,
    FIXED_QSCALE,
    FIXED_QSCALE_OAS,
    FULL,
    analyze_rows,
    factorial_contrasts,
)


def row(
    task: str,
    sample_id: str,
    method: str,
    score: float,
) -> dict[str, str]:
    return {
        "task": task,
        "sample_id": sample_id,
        "method": method,
        "score": str(score),
        "prediction": f"{method}-{sample_id}",
        "online_seconds": "2.0" if method != FULL else "4.0",
        "packed_index_ratio_of_full_kv": (
            "0.06" if method != FULL else "0"
        ),
        "packed_qmse_index_bits_per_token": (
            "240" if method != FULL else "0"
        ),
    }


def test_factorial_contrasts_have_the_expected_signs() -> None:
    contrasts = factorial_contrasts(
        {
            FULL: 1.0,
            AUTO_PLAIN: 0.90,
            FIXED_PLAIN: 0.88,
            AUTO_QSCALE: 0.94,
            FIXED_QSCALE: 0.93,
        }
    )
    assert abs(contrasts["fixed_minus_auto_plain"] + 0.02) < 1e-12
    assert abs(contrasts["fixed_minus_auto_qscale"] + 0.01) < 1e-12
    assert abs(contrasts["qscale_minus_plain_auto"] - 0.04) < 1e-12
    assert abs(contrasts["allocation_x_qscale_interaction"] - 0.01) < 1e-12


def test_factorial_analysis_requires_strict_five_way_pairs() -> None:
    reference = []
    factorial = []
    for task in ("task_a", "task_b"):
        for sample in ("0", "1"):
            reference.extend(
                (
                    row(task, sample, FULL, 1.0),
                    row(task, sample, AUTO_PLAIN, 0.90),
                )
            )
            factorial.extend(
                (
                    row(task, sample, FIXED_PLAIN, 0.88),
                    row(task, sample, AUTO_QSCALE, 0.94),
                    row(task, sample, FIXED_QSCALE, 0.93),
                )
            )
    summary, task_rows = analyze_rows(
        reference,
        factorial,
        bootstrap_replicates=100,
        seed=7,
    )
    assert summary["protocol"]["samples"] == 4
    assert summary["protocol"]["tasks"] == 2
    assert summary["macro_scores"][FULL] == 1.0
    assert abs(summary["macro_scores"][AUTO_QSCALE] - 0.94) < 1e-12
    assert summary["paired_online_speedup_vs_full"][FIXED_QSCALE] == 2.0
    assert len(task_rows) == 2


def test_factorial_analysis_can_include_oas_scale_cells() -> None:
    reference = []
    factorial = []
    for task in ("task_a", "task_b"):
        for sample in ("0", "1"):
            reference.extend(
                (
                    row(task, sample, FULL, 1.0),
                    row(task, sample, AUTO_PLAIN, 0.90),
                )
            )
            factorial.extend(
                (
                    row(task, sample, FIXED_PLAIN, 0.88),
                    row(task, sample, AUTO_QSCALE, 0.89),
                    row(task, sample, FIXED_QSCALE, 0.87),
                    row(task, sample, AUTO_QSCALE_OAS, 0.94),
                    row(task, sample, FIXED_QSCALE_OAS, 0.93),
                )
            )
    summary, _ = analyze_rows(
        reference,
        factorial,
        bootstrap_replicates=100,
        seed=7,
        include_qscale_oas=True,
    )
    assert len(summary["protocol"]["methods"]) == 7
    assert (
        summary["factorial_contrasts"]["qscale_oas_minus_raw_auto"]
        == pytest.approx(0.05)
    )
    assert (
        summary["quality_retention_vs_full"][AUTO_QSCALE_OAS]
        == pytest.approx(0.94)
    )
