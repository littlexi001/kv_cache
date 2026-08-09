from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_qksieve_deployment_longbench_20260801 import (  # noqa: E402
    DEPLOYMENT,
    DEPLOYMENT_SCORE_MODE,
    FULL,
    REFERENCE,
    analyze,
)


def row(task: str, sample: str, method: str, score: float) -> dict[str, str]:
    deployment = method == DEPLOYMENT
    return {
        "task": task,
        "sample_id": sample,
        "method": method,
        "score": str(score),
        "executed_path": method,
        "configured_score_mode": (
            DEPLOYMENT_SCORE_MODE if deployment else method
        ),
        "configured_index_bits_per_token": "240" if deployment else "0",
        "configured_attention_tokens": "256",
        "prompt_tokens": "4104",
        "prefix_tokens": "4096",
        "suffix_tokens": "8",
        "selected_history_fraction_mean": "0.0625" if deployment else "",
        "sampled_quantile_fallback": "0",
    }


def test_deployment_analysis_requires_strict_same_samples() -> None:
    reference_rows: list[dict[str, str]] = []
    deployment_rows: list[dict[str, str]] = []
    for index in range(16):
        task = f"task{index}"
        sample = f"sample{index}"
        reference_rows.append(row(task, sample, FULL, 1.0))
        reference_rows.append(row(task, sample, REFERENCE, 0.99))
        deployment_rows.append(row(task, sample, DEPLOYMENT, 0.98))

    report = analyze(reference_rows, deployment_rows, expected_pairs=16)

    assert report["strict_pairs"] == 16
    assert report["full_macro"] == 1.0
    assert report["methods"][REFERENCE]["quality_retention"] == 0.99
    assert report["methods"][DEPLOYMENT]["quality_retention"] == 0.98
    assert report["fairness_contract"]["full_fallback"] is False


def test_deployment_analysis_rejects_fallback() -> None:
    reference_rows: list[dict[str, str]] = []
    deployment_rows: list[dict[str, str]] = []
    for index in range(16):
        task = f"task{index}"
        sample = f"sample{index}"
        reference_rows.append(row(task, sample, FULL, 1.0))
        reference_rows.append(row(task, sample, REFERENCE, 1.0))
        deployment_rows.append(row(task, sample, DEPLOYMENT, 1.0))
    deployment_rows[0]["sampled_quantile_fallback"] = "1"

    try:
        analyze(reference_rows, deployment_rows, expected_pairs=16)
    except ValueError as error:
        assert "fallback" in str(error)
    else:
        raise AssertionError("a fallback row was accepted")
