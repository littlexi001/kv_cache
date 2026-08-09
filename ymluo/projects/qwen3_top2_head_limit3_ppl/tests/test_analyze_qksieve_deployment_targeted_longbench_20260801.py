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
)
from analyze_qksieve_deployment_targeted_longbench_20260801 import (  # noqa: E402
    analyze_targeted,
)


def make_row(
    task: str, sample: str, method: str, score: float
) -> dict[str, str]:
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
        "prompt_tokens": "4104",
        "prefix_tokens": "4096",
        "suffix_tokens": "8",
        "selected_history_fraction_mean": "0.0625" if deployment else "",
        "sampled_quantile_fallback": "0",
    }


def test_targeted_analysis_is_strict_and_paired() -> None:
    full_rows = []
    sparse_rows = []
    for task in ("qasper", "multi_news"):
        for index in range(2):
            full_rows.append(make_row(task, str(index), FULL, 1.0))
            sparse_rows.append(
                make_row(task, str(index), DEPLOYMENT, 0.98)
            )
    report = analyze_targeted(
        full_rows,
        sparse_rows,
        tasks=["qasper", "multi_news"],
        expected_per_task=2,
        bootstrap_replicates=100,
    )
    assert report["strict_pairs"] == 4
    assert report["quality_retention"] == 0.98
    assert report["per_task"]["qasper"]["samples"] == 2
    assert report["fairness_contract"]["full_fallback"] is False
