from __future__ import annotations

import pytest

from summarize_128k_attention_bottleneck_20260716 import FIELDS, summarize_payload


def make_payload() -> dict[str, float | str]:
    payload = {field: 0.01 for field in FIELDS}
    payload["index_order"] = "random"
    payload.update(
        {
            "mapped_host_fill_cache_attention_ms_per_layer": 1.0,
            "gqa_hybrid_mapped_attention_cache_update_ms_per_layer": 0.8,
            "hybrid_mapped_attention_cache_update_ms_per_layer": 1.2,
            "hybrid_pack_cache_update_attention_ms_per_layer": 0.9,
            "online_address_sort_fill_cache_attention_ms_per_layer": 1.1,
            "online_miss_compact_sort_fill_cache_attention_ms_per_layer": 1.05,
        }
    )
    return payload


def test_summary_reports_matched_host_path_ratios() -> None:
    payload = make_payload()
    payload["selected_sdpa_attention_max_abs_error"] = 0.1
    row = summarize_payload("matched", payload)
    assert row["direct_gqa_over_miss_fill_latency"] == pytest.approx(0.8)
    assert row["direct_query_head_over_miss_fill_latency"] == pytest.approx(1.2)
    assert row["pack_then_attention_over_miss_fill_latency"] == pytest.approx(0.9)
    assert row["online_address_sort_over_current_order_latency"] == pytest.approx(
        1.1
    )
    assert row["online_miss_sort_over_current_order_latency"] == pytest.approx(
        1.05
    )


def test_summary_rejects_numerically_incorrect_kernel() -> None:
    payload = make_payload()
    payload["gqa_hybrid_max_abs_error"] = 0.03
    with pytest.raises(ValueError, match="gqa_hybrid_max_abs_error"):
        summarize_payload("bad", payload)
