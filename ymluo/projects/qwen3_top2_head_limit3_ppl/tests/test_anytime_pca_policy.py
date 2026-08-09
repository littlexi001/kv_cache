from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_anytime_pca_policy_20260717 import (  # noqa: E402
    evaluate_policy,
    routed_case_indices,
)


def make_cases() -> list[dict[str, float | int | str]]:
    cases = []
    for head, gain in enumerate((0.01, 0.04, 0.02, 0.03)):
        base_mass = 0.90
        cases.append(
            {
                "topic": "test",
                "record_index": 0,
                "layer": 0,
                "query_head": head,
                "query_energy_coverage": 0.9 - gain,
                "base_proxy_top2_mass": 0.8,
                "base_tail_score_std": 1.0,
                "base_margin_to_2k_sigma": 1.0,
                "base_margin_to_4k_sigma": 2.0,
                "base_boundary_band_1sigma_ratio": gain,
                "base_top2_attention_mass_recall": base_mass,
                "enhanced_top2_attention_mass_recall": base_mass + gain,
                "base_top2_recall": 0.6,
                "enhanced_top2_recall": 0.6 + gain,
            }
        )
    return cases


def test_routing_selects_exact_fraction_within_each_layer() -> None:
    cases = make_cases()
    selected = routed_case_indices(
        cases, 0.5, lambda case: case["base_boundary_band_1sigma_ratio"]
    )

    assert selected == {1, 3}


def test_oracle_gain_policy_selects_largest_improvements() -> None:
    result = evaluate_policy(make_cases(), "oracle_gain", 0.5, 64, 96)

    assert result["route_fraction_actual"] == 0.5
    assert result["average_rank"] == 80.0
    assert abs(result["top2_attention_mass_recall_mean"] - 0.9175) < 1.0e-9
