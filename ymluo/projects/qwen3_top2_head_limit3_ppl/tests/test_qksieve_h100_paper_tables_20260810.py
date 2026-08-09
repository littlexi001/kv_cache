from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "papers"
    / "countcap_iclr2027"
    / "scripts"
    / "make_qksieve_h100_tables.py"
)
SPEC = importlib.util.spec_from_file_location("qksieve_h100_tables", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def summary() -> dict:
    return {
        "attention": [
            {
                "history_tokens": 65536,
                "full_mha_ms": 8.0,
                "qksieve_fast_ms": 1.0,
                "qksieve_robust_ms": 2.0,
                "fier_ms": 4.0,
                "fast_speedup": 8.0,
                "robust_speedup": 4.0,
                "robust_vs_fier": 2.0,
                "qksieve_total_auxiliary_ratio_of_full_kv": 0.07471,
            }
        ],
        "steady_decode": [
            {
                "history_tokens": 65536,
                "full_steady_ms_per_token": 12.0,
                "qksieve_steady_ms_per_token": 4.0,
                "steady_decode_speedup": 3.0,
                "qksieve_prebuild_seconds_median": 1.25,
                "qksieve_to_full_peak_allocated_ratio": 1.08,
            }
        ],
        "persistent_requests": [
            {
                "history_tokens": 65536,
                "cold_speedup": 0.8,
                "cold_end_to_end_speedup": 0.5,
                "warm_speedup": 2.0,
                "shared_prefix_amortized_speedup": 1.5,
                "append_only_speedup": 1.8,
                "qksieve_to_full_cold_peak_allocated_ratio": 1.1,
            }
        ],
    }


def test_h100_tables_render_all_three_timing_contracts() -> None:
    text = MODULE.render(summary(), chinese=False, provenance="abc123")
    assert "Generated from frozen H100 evidence: abc123" in text
    assert "64K & 8.000 & 1.000 & 2.000 & 4.000" in text
    assert "8.00$\\times$ & 4.00$\\times$ & 2.00$\\times$ & 7.47\\%" in text
    assert "64K & 12.000 & 4.000 & 3.00$\\times$ & 1.250 & 1.080$\\times$" in text
    assert "0.80$\\times$ & 0.50$\\times$ & 2.00$\\times$" in text


def test_h100_tables_have_synchronized_chinese_labels() -> None:
    text = MODULE.render(summary(), chinese=True, provenance="abc123")
    assert "匹配 H100" in text
    assert "Cold E2E" in text
    assert "峰值显存比" in text

