from __future__ import annotations

import pytest

from compose_fused_cache_e2e_20260715 import compose_fused_cache_e2e


def fixtures() -> tuple[dict, dict]:
    physical = {
        "topic": "religion",
        "history_tokens": 32000,
        "remote_tokens": 31744,
        "query_tokens": 256,
        "eval_tokens": 256,
        "final_cache_length": 32255,
        "directory_backend": "fused",
        "timing_is_synchronized_per_token": True,
        "ppl": 12.3426,
        "prefill_seconds": 18.18,
        "cache_conversion_seconds": 6.68,
        "prefill_plus_conversion_seconds": 24.86,
        "synchronized_model_forward_seconds": 33.11,
        "hierarchical_persistent_gpu_bytes": 421716736,
        "hierarchical_over_final_length_full_kv": 0.09975,
        "pinned_host_bytes": 4228907008,
        "process_gpu_allocated_after_conversion": 16493408768,
        "process_peak_gpu_allocated_during_prefill_conversion": 21462139392,
    }
    full = {
        "topic": "religion",
        "history_tokens": 32000,
        "remote_tokens": 31744,
        "query_tokens": 256,
        "eval_tokens": 256,
        "final_cache_length": 32255,
        "attention_implementation": "sdpa",
        "ppl": 12.2758,
        "prefill_seconds": 16.70,
        "synchronized_model_forward_seconds": 50.71,
        "final_gpu_kv_bytes": 4227727360,
        "process_gpu_allocated_after_decode": 20297539584,
        "process_peak_gpu_allocated_during_prefill_decode": 21462139392,
    }
    return physical, full


def test_composes_same_protocol_e2e_result() -> None:
    physical, full = fixtures()
    summary = compose_fused_cache_e2e(physical, full, gpu="RTX 3090")

    assert summary["speedup"]["decode"] == pytest.approx(50.71 / 33.11)
    assert summary["speedup"]["total_for_protocol"] > 1.0
    assert summary["speedup"]["amortized_break_even_decode_steps"] < 511
    assert all(summary["checks"].values())
    assert summary["status"] == "fused_hf_cache_quality_storage_and_e2e_validated"


def test_marks_measured_result_when_quality_target_fails() -> None:
    physical, full = fixtures()
    physical["ppl"] = 20.0

    summary = compose_fused_cache_e2e(physical, full, gpu="RTX 3090")

    assert not summary["checks"]["quality_retention_at_least_95_percent"]
    assert summary["status"] == "fused_hf_cache_measured_with_failed_target"


def test_rejects_unsynchronized_or_mismatched_results() -> None:
    physical, full = fixtures()
    physical["timing_is_synchronized_per_token"] = False
    with pytest.raises(ValueError, match="synchronized"):
        compose_fused_cache_e2e(physical, full, gpu="RTX 3090")

    physical, full = fixtures()
    full["query_tokens"] = 128
    with pytest.raises(ValueError, match="protocol mismatch"):
        compose_fused_cache_e2e(physical, full, gpu="RTX 3090")
