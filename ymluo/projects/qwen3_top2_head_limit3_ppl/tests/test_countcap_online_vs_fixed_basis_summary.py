from __future__ import annotations

import pytest

from summarize_countcap_online_vs_fixed_basis_20260726 import summarize_model


def test_summary_uses_strict_pairs_and_macro_average() -> None:
    rows = []
    for task, online, fixed in (
        ("a", 1.0, 0.9),
        ("b", 0.5, 0.5),
    ):
        for method, score, prediction in (
            ("online", online, "same"),
            ("fixed", fixed, "same"),
        ):
            rows.append(
                {
                    "task": task,
                    "sample_id": "0",
                    "method": method,
                    "score": str(score),
                    "prediction": prediction,
                    "index_build_seconds": "2" if method == "online" else "1",
                    "total_seconds": "4" if method == "online" else "3",
                }
            )

    overall, tasks = summarize_model(rows, "model")

    assert len(tasks) == 2
    assert overall[0]["online_macro"] == pytest.approx(0.75)
    assert overall[0]["fixed_macro"] == pytest.approx(0.70)
    assert overall[0]["fixed_minus_online_macro"] == pytest.approx(-0.05)
    assert (
        overall[0]["fixed_minus_online_macro_ci95_low"] - 1.0e-12
        <= overall[0]["fixed_minus_online_macro"]
        <= overall[0]["fixed_minus_online_macro_ci95_high"] + 1.0e-12
    )
    assert overall[0]["prediction_agreement"] == 1.0
    assert overall[0]["fixed_index_build_speedup"] == pytest.approx(2.0)
