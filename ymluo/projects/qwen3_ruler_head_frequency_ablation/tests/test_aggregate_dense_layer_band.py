from __future__ import annotations

from src.aggregate_dense_layer_band import aggregate


def _row(layer: int, group: int, frequency_start: int, delta: float) -> dict:
    return {
        "variant": f"dense_l{layer}_g{group}_f{frequency_start:02d}_{frequency_start + 7:02d}",
        "paired_official_delta": delta,
        "mean_nll_improvement": delta / 2,
        "spec": {
            "stage": "dense_layer_band",
            "region": {
                "layer": layer,
                "head_group": group,
                "frequency_pairs": list(range(frequency_start, frequency_start + 8)),
            },
        },
    }


def test_aggregate_builds_complete_grid() -> None:
    rows = [{"variant": "native_rope"}]
    for layer in range(18, 36):
        for group in range(8):
            for frequency_start in range(0, 64, 8):
                rows.append(_row(layer, group, frequency_start, layer / 1000))

    result = aggregate(rows)

    assert result["setup"]["single_layer_configurations"] == 1152
    assert result["setup"]["aggregated_cells"] == 192
    assert result["setup"]["native_baseline_rows"] == 1
    first = result["cells"][0]
    assert first["layer_block"] == "L18-23"
    assert first["configuration_count"] == 6
    assert first["best_layer"] == 23
    assert first["worst_layer"] == 18
    assert abs(first["mean_official_delta_pp"] - 2.05) < 1e-9
