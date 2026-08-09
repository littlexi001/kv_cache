from __future__ import annotations

import pytest

from compose_physical_attention_path_20260715 import compose_physical_attention_path


def fixtures() -> tuple[dict, dict, dict]:
    profile = {
        "history_count": 131072,
        "candidate_mode": "shared_mean",
        "budget_fraction": 0.02,
        "online_composed_ms": 0.74946044921875,
        "final_sparse_attention_ms": 0.537594871520996,
    }
    directory = {
        "history_count": 131072,
        "exact_cache_fraction": 0.0319976806640625,
        "total_resident_fraction": 0.09957090020179749,
        "fused_directory_total_ms_per_layer": 0.05021183967590333,
    }
    data_path = {
        "history_count": 131072,
        "attention_fraction": 0.02,
        "cache_hit_rate": 0.7868024924379972,
        "selected_fraction": 0.004263950151240056,
        "mapped_host_fill_cache_attention_ms_per_layer": 0.7410073852539063,
        "max_abs_error": 0.0,
        "resident_cache_attention_max_abs_error": 0.0,
    }
    return profile, directory, data_path


def test_composes_online_prefix_and_all_physical_components() -> None:
    profile, directory, data_path = fixtures()
    summary = compose_physical_attention_path(
        profile, directory, data_path, full_attention_ms=2.628
    )

    latency = summary["latency_ms_per_layer"]
    assert latency["online_retrieval_prefix"] == pytest.approx(0.2118655777)
    assert latency["ours_total"] == pytest.approx(1.0030848026)
    assert summary["attention_path_speedup"] == pytest.approx(2.6199180699)
    assert summary["is_numerically_exact"]
    assert summary["is_below_ten_percent_resident"]
    assert summary["is_above_2p5x_attention_path"]


def test_rejects_history_mismatch() -> None:
    profile, directory, data_path = fixtures()
    data_path["history_count"] = 65536

    with pytest.raises(ValueError, match="history count mismatch"):
        compose_physical_attention_path(
            profile, directory, data_path, full_attention_ms=2.628
        )


def test_rejects_budget_mismatch() -> None:
    profile, directory, data_path = fixtures()
    data_path["attention_fraction"] = 0.01

    with pytest.raises(ValueError, match="budget mismatch"):
        compose_physical_attention_path(
            profile, directory, data_path, full_attention_ms=2.628
        )
