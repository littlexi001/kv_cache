from __future__ import annotations

import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import analyze_qksieve_binarypc_longbench_20260801 as analysis  # noqa: E402


def _rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    reference: list[dict[str, str]] = []
    binarypc: list[dict[str, str]] = []
    for task_index in range(16):
        common = {
            "task": f"task{task_index}",
            "sample_id": "0",
            "prompt_tokens": "1024",
            "prefix_tokens": "1000",
            "suffix_tokens": "24",
            "configured_attention_tokens": "256",
        }
        reference.extend(
            [
                {
                    **common,
                    "method": analysis.FULL,
                    "score": "1.0",
                    "executed_path": analysis.FULL,
                    "configured_score_mode": "full_kv",
                    "configured_index_bits_per_token": "0",
                },
                {
                    **common,
                    "method": analysis.QKSIEVE,
                    "score": "0.99",
                    "executed_path": analysis.QKSIEVE,
                    "configured_score_mode": (
                        "pca_hierarchical_autoqmsetotal15z_"
                        "qkmetric_packed_fulltopk"
                    ),
                    "configured_index_bits_per_token": "240",
                },
            ]
        )
        binarypc.append(
            {
                **common,
                "method": analysis.BINARYPC,
                "score": "0.98",
                "executed_path": analysis.BINARYPC,
                "configured_score_mode": "binarypc_offline64_fulltopk",
                "configured_index_bits_per_token": "64",
            }
        )
    return reference, binarypc


def test_binarypc_analyzer_requires_strict_same_sample_budget() -> None:
    reference, binarypc = _rows()
    report = analysis.analyze(reference, binarypc, expected_pairs=16)

    assert report["strict_pairs"] == 16
    assert report["tasks"] == 16
    assert report["methods"][analysis.QKSIEVE]["quality_retention"] == 0.99
    assert report["methods"][analysis.BINARYPC]["quality_retention"] == 0.98
    assert report["fairness_contract"]["binarypc_first_two_full_layers"] is False
    assert report["latency_claim"]["valid"] is False

    broken = [dict(row) for row in binarypc]
    broken[0]["configured_attention_tokens"] = "255"
    with pytest.raises(ValueError, match="active-token budget"):
        analysis.analyze(reference, broken, expected_pairs=16)
