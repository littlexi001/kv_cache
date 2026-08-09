from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_hierarchical_ruler_probe_20260716 as data_runner  # noqa: E402
import summarize_qksieve_ruler_20260728 as summarizer  # noqa: E402
import verify_qksieve_frozen_evidence_20260728 as verifier  # noqa: E402


def _rows() -> list[dict[str, str]]:
    rows = []
    for task in summarizer.EXPECTED_TASKS:
        for length, samples in summarizer.EXPECTED_LENGTH_SAMPLES.items():
            history = length - 64
            budget = min(
                history,
                1280,
                max(256, math.ceil(0.06 * history)),
            )
            for index in range(samples):
                common = {
                    "task": f"{task}_{length}",
                    "base_task": task,
                    "requested_length": str(length),
                    "sample_id": f"{task}_{length}_{index}",
                    "prompt_tokens": str(length),
                    "prefix_tokens": str(history),
                    "suffix_tokens": "64",
                    "configured_attention_tokens": str(budget),
                }
                rows.append(
                    {
                        **common,
                        "method": summarizer.REFERENCE_METHOD,
                        "score": "1.0",
                        "configured_score_mode": "full_kv",
                        "generated_tokens": "8",
                        "online_seconds": "0.8",
                        "decode_seconds": "0.72",
                        "total_seconds": "1.8",
                    }
                )
                rows.append(
                    {
                        **common,
                        "method": summarizer.METHOD,
                        "score": "0.99",
                        "configured_score_mode": summarizer.SCORE_MODE,
                        "generated_tokens": "4",
                        "online_seconds": "0.2",
                        "decode_seconds": "0.18",
                        "total_seconds": "1.2",
                    }
                )
    return rows


def test_ruler_default_task_set_matches_official_13_tasks() -> None:
    assert tuple(data_runner.DEFAULT_TASKS.split(",")) == (
        summarizer.EXPECTED_TASKS
    )


def test_strict_ruler_summary_is_paired_and_token_normalized() -> None:
    summary = summarizer.summarize(
        _rows(),
        PROJECT_ROOT,
        bootstrap_resamples=20,
        seed=17,
    )

    assert summary["strict_pairs"] == 650
    assert summary["tasks"] == 13
    assert summary["overall"]["quality_retention"] == pytest.approx(0.99)
    assert summary["overall"]["geomean_online_tpot_speedup"] == pytest.approx(
        2.0
    )
    assert summary["overall"]["geomean_decode_tpot_speedup"] == pytest.approx(
        2.0
    )
    assert len(summary["overall"]["quality_retention_95ci"]) == 2


def test_strict_ruler_summary_rejects_a_missing_method_row() -> None:
    rows = _rows()
    rows.pop()
    with pytest.raises(AssertionError, match="method counts differ"):
        summarizer.summarize(
            rows,
            PROJECT_ROOT,
            bootstrap_resamples=0,
            seed=17,
        )


def test_targeted_ruler_protocol_override_remains_strict() -> None:
    rows = [
        row
        for row in _rows()
        if int(row["requested_length"]) in {8192, 16384}
        and row["sample_id"].endswith(("_0", "_1"))
    ]
    expected = {8192: 2, 16384: 2}

    summary = summarizer.summarize(
        rows,
        PROJECT_ROOT,
        bootstrap_resamples=20,
        seed=17,
        expected_length_samples=expected,
    )

    assert summary["strict_pairs"] == 52
    assert summary["lengths"] == [8192, 16384]
    assert summary["protocol"]["samples_per_task_length"] == expected
    assert summary["protocol"]["formal_protocol"] is False


def test_targeted_protocol_parser_rejects_invalid_or_duplicate_entries() -> None:
    assert summarizer.parse_expected_length_samples("8192:2,16384:1") == {
        8192: 2,
        16384: 1,
    }
    with pytest.raises(ValueError, match="duplicate length"):
        summarizer.parse_expected_length_samples("8192:2,8192:1")


def test_samepath_gate_requires_frozen_rate_and_component_breakdown() -> None:
    cells = {}
    for length in verifier.EXPECTED_LENGTHS:
        for horizon in verifier.EXPECTED_HORIZONS:
            counts = [
                min(
                    length + offset,
                    1280,
                    max(256, math.ceil(0.06 * (length + offset))),
                )
                for offset in range(horizon - 1)
            ]
            cells[f"{length}@{horizon}"] = {
                "history_tokens": length,
                "decode_steps": horizon,
                "qksieve": {
                    "peak_allocated_bytes_total": 1,
                    "steady_seconds_per_step": 0.1,
                    "configured_attention_tokens": sum(counts) / len(counts),
                    "index_ratio_of_full_kv": 240.0 / 4096.0,
                },
            }
    fields = (
        "fused_query_prepare_ms",
        "packed_scan_ms",
        "torch_topk_ms",
        "explicit_kv_gather_ms",
        "gathered_sdpa_ms",
        "exact_sparse_attention_ms",
        "historical_index_project_encode_ms",
        "per_token_index_project_encode_ms",
        "complete_sparse_path_with_index_append_ms",
        "attention_speedup_including_index_append",
        "full_sdpa_ms",
    )
    breakdown_rows = [
        {
            "history_tokens": length,
            **{field: 1.0 for field in fields},
        }
        for length in verifier.EXPECTED_LENGTHS
    ]
    summary = {
        "frozen_method": dict(verifier.FROZEN_METHOD),
        "cells": cells,
        "attention_breakdown": {"rows": breakdown_rows},
    }

    validated = verifier.validate_samepath(summary)

    assert len(validated["cells"]) == 12
    assert len(validated["attention_breakdown"]["rows"]) == 4
