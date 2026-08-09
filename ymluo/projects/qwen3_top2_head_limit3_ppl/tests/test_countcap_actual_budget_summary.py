from __future__ import annotations

import pytest
import torch

from summarize_countcap_actual_budget_20260726 import summarize_model
from run_sample_calibrated_longbench_20260717 import aggregate_sparse_stats
from run_critical_position_budget_probe_20260715 import (
    summarize_attention_records,
)


def test_actual_budget_summary_uses_measured_counts() -> None:
    rows = [
        {
            "task": "a",
            "diagnostics_enabled": "True",
            "prompt_tokens": "1000",
            "configured_attention_fraction": "0.06",
            "attention_link_ratio": "0.07",
            "selected_history_fraction_p95": "0.08",
            "selected_history_fraction_max": "0.09",
            "selected_history_count_mean": "70",
            "selected_history_count_p95": "80",
            "selected_history_count_max": "90",
            "sampled_candidate_overflow_fraction": "0.01",
            "sampled_quantile_fallback": "0.0",
        },
        {
            "task": "b",
            "diagnostics_enabled": "True",
            "prompt_tokens": "2000",
            "configured_attention_fraction": "0.06",
            "attention_link_ratio": "0.05",
            "selected_history_fraction_p95": "0.07",
            "selected_history_fraction_max": "0.10",
            "selected_history_count_mean": "100",
            "selected_history_count_p95": "140",
            "selected_history_count_max": "200",
            "sampled_candidate_overflow_fraction": "0.03",
            "sampled_quantile_fallback": "0.0",
        },
    ]

    overall, tasks = summarize_model(rows, "model")

    assert len(tasks) == 2
    assert overall["actual_fraction_mean"] == pytest.approx(0.06)
    assert overall["actual_count_mean"] == pytest.approx(85.0)
    assert overall["actual_count_max"] == pytest.approx(200.0)
    assert overall["candidate_overflow_head_fraction_mean"] == pytest.approx(
        0.02
    )
    assert overall["sampled_quantile_fallback_rate_mean"] == 0.0


def test_longbench_aggregation_propagates_overflow_diagnostics() -> None:
    summary = aggregate_sparse_stats(
        [
            {
                "sampled_candidate_overflow_fraction_mean": 0.02,
                "sampled_quantile_fallback_mean": 0.0,
            },
            {
                "sampled_candidate_overflow_fraction_mean": 0.04,
                "sampled_quantile_fallback_mean": 0.0,
            },
        ]
    )

    assert summary["sampled_candidate_overflow_fraction"] == pytest.approx(
        0.03
    )
    assert summary["sampled_quantile_fallback"] == 0.0


def test_longbench_aggregation_proves_qfused_execution() -> None:
    summary = aggregate_sparse_stats(
        [
            {
                "packed_qmse_fused_query_prepare_requested_mean": 1.0,
                "packed_qmse_fused_query_prepare_executed_mean": 1.0,
                "packed_qmse_allocation_frozen_before_query_mean": 1.0,
            },
            {
                "packed_qmse_fused_query_prepare_requested_mean": 1.0,
                "packed_qmse_fused_query_prepare_executed_mean": 1.0,
                "packed_qmse_allocation_frozen_before_query_mean": 1.0,
            },
        ]
    )

    assert summary["packed_qmse_fused_query_prepare_requested"] == 1.0
    assert summary["packed_qmse_fused_query_prepare_executed"] == 1.0
    assert summary["packed_qmse_allocation_frozen_before_query"] == 1.0


def test_attention_summary_keeps_qfused_layer_diagnostics() -> None:
    summary = summarize_attention_records(
        [
            {
                "layer": 0,
                "packed_qmse_fused_query_prepare_requested": torch.tensor(1.0),
                "packed_qmse_fused_query_prepare_executed": torch.tensor(1.0),
                "packed_qmse_allocation_frozen_before_query": torch.tensor(1.0),
            },
            {
                "layer": 1,
                "packed_qmse_fused_query_prepare_requested": torch.tensor(1.0),
                "packed_qmse_fused_query_prepare_executed": torch.tensor(1.0),
                "packed_qmse_allocation_frozen_before_query": torch.tensor(1.0),
            },
        ]
    )

    assert summary["packed_qmse_fused_query_prepare_requested_mean"] == 1.0
    assert summary["packed_qmse_fused_query_prepare_executed_min"] == 1.0
    assert summary["packed_qmse_allocation_frozen_before_query_mean"] == 1.0
