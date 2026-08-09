from __future__ import annotations

import json
import sys

from plot_pca_int4_delta_invariance_20260726 import METHODS, main


def test_plot_accepts_production_aligned_error_chain(
    tmp_path,
    monkeypatch,
) -> None:
    rows = []
    for method, _, _ in METHODS:
        for fraction in (0.02, 0.04, 0.06, 0.08):
            rows.append(
                {
                    "method": method,
                    "fraction": fraction,
                    "retained_attention_mass_mean": 0.9 + fraction,
                    "retained_attention_mass_p10": 0.88 + fraction,
                    "attention_output_cosine_mean": 0.98,
                }
            )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"candidate_overall": rows}),
        encoding="utf-8",
    )
    output = tmp_path / "plot.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot",
            "--summary_path",
            str(summary),
            "--output_path",
            str(output),
        ],
    )

    main()

    assert output.stat().st_size > 0
