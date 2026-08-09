from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "summarize_qksieve_deployment_profile_multiseed_20260801.py"
)
SPEC = importlib.util.spec_from_file_location("profile_summary", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def document(profile: str, speed: float, latency: float) -> dict:
    return {
        "schema": MODULE.SCHEMA,
        "hardware": {"device_name": "test-gpu"},
        "rows": [
            {
                "profile": profile,
                "history_tokens": 32768,
                "logical_index_bits_per_token_per_kv_head": 112,
                "attention_complete_direct_ms": latency,
                "attention_speedup_vs_full_preexpanded_sdpa": speed,
                "fused_sampled_retrieval_direct_ms": 0.05,
                "exact_sparse_attention_direct_ms": 0.1,
                "query_plus_retrieval_direct_ms": 0.12,
                "historical_index_build_direct_ms": 1.4,
                "per_token_index_append_direct_ms": 0.1,
                "mean_selected_tokens_per_query_head": 1280,
            }
        ],
    }


def test_aggregate_uses_median_and_observed_range() -> None:
    result = MODULE.aggregate_documents(
        [
            ("seed-a", document("fixed410_b112", 3.0, 0.2)),
            ("seed-b", document("fixed410_b112", 2.0, 0.3)),
            ("seed-c", document("fixed410_b112", 4.0, 0.1)),
        ],
        {32768},
    )
    row = result["rows"][0]
    assert row["repeats"] == 3
    assert row["attention_speedup_vs_full_preexpanded_sdpa"] == {
        "median": 3.0,
        "minimum": 2.0,
        "maximum": 4.0,
    }
    assert row["attention_complete_direct_ms"]["median"] == 0.2
    assert result["missing_profile_length_pairs"] == []


def test_unrequested_lengths_are_ignored() -> None:
    result = MODULE.aggregate_documents(
        [("seed-a", document("fixed410_b112", 3.0, 0.2))],
        {131072},
    )
    assert result["rows"] == []


def test_legacy_string_hardware_is_supported() -> None:
    legacy = document("fixed410_b112", 3.0, 0.2)
    legacy["hardware"] = "legacy-gpu"
    result = MODULE.aggregate_documents([("legacy", legacy)], {32768})
    assert result["contract"]["hardware"] == ["legacy-gpu"]
