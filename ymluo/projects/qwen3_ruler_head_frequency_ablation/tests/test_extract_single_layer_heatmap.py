from __future__ import annotations

import math

from src.extract_single_layer_heatmap import extract


def _row(layer: int, group: int, frequency_start: int) -> dict:
    return {
        "variant": f"dense_l{layer}_g{group}_f{frequency_start:02d}_{frequency_start + 7:02d}",
        "official_score_mean": 0.5,
        "paired_official_delta": layer / 1000,
        "mean_nll_improvement": 0.01,
        "gold_answer_ppl_from_mean_nll": 200.0,
        "improved_score_samples": 1,
        "degraded_score_samples": 0,
        "spec": {
            "stage": "dense_layer_band",
            "region": {
                "layer": layer,
                "head_group": group,
                "frequency_pairs": list(range(frequency_start, frequency_start + 8)),
            },
        },
    }


def test_extract_returns_all_single_layer_cells() -> None:
    rows = [
        _row(layer, group, start)
        for layer in range(18, 36)
        for group in range(8)
        for start in range(0, 64, 8)
    ]
    result = extract(rows)
    assert result["setup"]["single_layer_configurations"] == 1152
    assert len(result["cells"]) == 1152
    assert result["cells"][0]["layer"] == 18
    assert result["cells"][-1]["layer"] == 35
    assert abs(result["cells"][0]["official_delta_pp"] - 1.8) < 1e-12
    assert result["cells"][0]["gold_ppl"] == 200.0
    expected_relative_change = 100.0 * (math.exp(-0.01) - 1.0)
    assert abs(result["cells"][0]["gold_ppl_relative_change_percent"] - expected_relative_change) < 1e-12
