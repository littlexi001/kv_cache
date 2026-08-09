from pathlib import Path
import sys


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from analyze_countcap_quality_diagnostic_20260723 import METHODS, analyze


def test_quality_diagnostic_decomposes_budget_and_retrieval_gaps():
    scores = {
        "full_kv": 1.0,
        "exact_top2_fullprompt": 0.94,
        "exact_massadaptive_fullprompt": 0.98,
        "countcap_fullprompt": 0.90,
        "countcap_massadaptive_fullprompt": 0.97,
    }
    rows = []
    for task in ("qasper", "musique"):
        for sample_id in ("0", "1"):
            for method in METHODS:
                rows.append(
                    {
                        "task": task,
                        "sample_id": sample_id,
                        "method": method,
                        "score": str(scores[method]),
                        "online_seconds": "1.0" if method == "full_kv" else "2.0",
                        "configured_attention_fraction": "1.0" if method == "full_kv" else "0.06",
                        "attention_link_ratio": "1.0" if method == "full_kv" else "0.03",
                    }
                )

    result = analyze(rows)

    assert result["samples"] == 4
    assert result["tasks"] == 2
    assert abs(result["gap_decomposition"]["fixed_top2_budget_gap"] - 0.06) < 1e-9
    assert abs(result["gap_decomposition"]["top2_retrieval_gap"] - 0.04) < 1e-9
    assert abs(result["gap_decomposition"]["exact_adaptive_recovery"] - 0.04) < 1e-9
    assert result["diagnosis"]["fixed_top2_budget_is_safe"] is False
    assert result["diagnosis"]["exact_adaptive_meets_quality_floor"] is True
