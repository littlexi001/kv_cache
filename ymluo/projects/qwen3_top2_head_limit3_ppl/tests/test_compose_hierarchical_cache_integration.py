from __future__ import annotations

import pytest

from compose_hierarchical_cache_integration_20260715 import (
    compose_hierarchical_cache_integration,
)


def make_row(topic: str, ppl: float, algorithm_ppl: float) -> dict:
    return {
        "topic": topic,
        "history_tokens": 32000,
        "remote_tokens": 31744,
        "query_tokens": 256,
        "eval_tokens": 256,
        "projection_dim": 32,
        "candidate_fraction": 0.02,
        "exact_cache_fraction": 0.032,
        "ppl": ppl,
        "known_reference_ppl": algorithm_ppl,
        "mean_cache_hit_rate": 0.8,
        "hierarchical_persistent_gpu_bytes": 418561792,
        "pinned_host_bytes": 4228907008,
        "hierarchical_over_final_length_full_kv": 0.099,
        "hierarchical_over_capacity_equivalent_full_kv": 0.09898,
    }


def test_composes_integrated_quality_and_physical_storage() -> None:
    rows = [
        make_row("sports", 9.07, 9.058),
        make_row("medicine", 10.737, 10.734),
        make_row("religion", 12.343, 12.341),
    ]
    full = {"sports": 8.79, "medicine": 10.224, "religion": 12.276}

    summary = compose_hierarchical_cache_integration(rows, full)

    assert summary["checks"]["implementation_drift_below_0p5_percent"]
    assert summary["checks"]["quality_retention_at_least_95_percent"]
    assert summary["checks"]["persistent_gpu_kv_below_10_percent"]
    assert summary["physical_storage"]["final_length_full_kv_fraction_mean"] == pytest.approx(0.099)


def test_rejects_protocol_mismatch() -> None:
    rows = [make_row("sports", 9.07, 9.058), make_row("medicine", 10.737, 10.734)]
    rows[1]["query_tokens"] = 128

    with pytest.raises(ValueError, match="protocol mismatch"):
        compose_hierarchical_cache_integration(
            rows, {"sports": 8.79, "medicine": 10.224}
        )


def test_rejects_missing_full_reference() -> None:
    with pytest.raises(ValueError, match="missing full PPL"):
        compose_hierarchical_cache_integration(
            [make_row("sports", 9.07, 9.058)], {}
        )
