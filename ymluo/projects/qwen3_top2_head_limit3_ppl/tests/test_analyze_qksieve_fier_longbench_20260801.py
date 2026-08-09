from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_qksieve_fier_longbench_20260801 import (
    FIER,
    FULL,
    QKSIEVE,
    analyze,
)


def row(task: str, method: str) -> dict[str, str]:
    score_mode = {
        FULL: "full_kv",
        QKSIEVE: "pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk",
        FIER: "fier_rtn1_g32_fulltopk",
    }[method]
    index_bits = {FULL: 0, QKSIEVE: 240, FIER: 256}[method]
    return {
        "task": task,
        "sample_id": "0",
        "method": method,
        "executed_path": method,
        "configured_score_mode": score_mode,
        "configured_index_bits_per_token": str(index_bits),
        "configured_attention_tokens": "16000" if method == FULL else "1000",
        "prompt_tokens": "16032",
        "prefix_tokens": "16000",
        "suffix_tokens": "32",
        "score": "1.0",
    }


def test_fier_report_requires_strict_protocol_pairs() -> None:
    reference = []
    fier = []
    for index in range(16):
        task = f"task{index}"
        reference.extend([row(task, FULL), row(task, QKSIEVE)])
        fier.append(row(task, FIER))

    report = analyze(reference, fier, expected_pairs=16)
    assert report["strict_pairs"] == 16
    assert report["methods"][FIER]["quality_retention"] == 1.0
    assert report["methods"][FIER]["mean_active_token_ratio"] == 1 / 16

    broken = [dict(item) for item in fier]
    broken[0]["configured_attention_tokens"] = "999"
    try:
        analyze(reference, broken, expected_pairs=16)
    except ValueError as error:
        assert "active-token budget" in str(error)
    else:
        raise AssertionError("mismatched FIER budget was accepted")
