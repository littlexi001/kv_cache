from __future__ import annotations

import pytest

from summarize_128k_low_peak_ablation_20260716 import summarize


def make_payloads():
    shared = {
        "topic": "religion",
        "window_index": 0,
        "history_tokens": 128000,
        "query_tokens": 256,
        "eval_tokens": 2,
        "target_token_ids": [1, 2],
    }
    full = {
        **shared,
        "ppl": 10.0,
        "prefill_seconds": 10.0,
        "synchronized_model_forward_seconds": 5.0,
        "process_peak_gpu_allocated_during_prefill_decode": 1000,
    }
    sparse = {
        **shared,
        "projection_dim": 64,
        "index_bits": 4,
        "candidate_fraction": 0.015,
        "attention_fraction": 0.015,
        "candidate_selection_mode": "per_head_stream",
        "exact_cache_fraction": 0.032,
        "stream_group_size": 2,
        "directory_backend": "fused",
        "ppl": 10.5,
        "prefill_seconds": 10.0,
        "cache_conversion_seconds": 2.0,
        "online_seconds": 2.5,
        "hierarchical_over_final_length_full_kv": 0.1,
        "process_peak_gpu_allocated_during_prefill_conversion": 900,
        "pinned_host_bytes": 1000,
        "original_remote_full_gpu_kv_bytes": 1000,
        "token_nll": [1.0, 2.0],
    }
    offloaded = {
        **sparse,
        "prefill_cache_mode": "offloaded_exact",
        "process_peak_gpu_allocated_during_prefill_conversion": 500,
        "token_nll": [1.1, 1.9],
    }
    return full, sparse, offloaded


def test_low_peak_summary_reports_peak_and_quality_tradeoff() -> None:
    payload = summarize(*make_payloads())
    assert payload["rows"][2]["peak_gpu_bytes_ratio"] == pytest.approx(0.5)
    assert payload["rows"][2]["quality_retention"] == pytest.approx(10.0 / 10.5)
    assert payload["offloaded_peak_reduction_vs_dynamic"] == pytest.approx(4 / 9)
    assert payload["offloaded_vs_dynamic_max_abs_token_nll_delta"] == pytest.approx(0.1)
