from __future__ import annotations

import csv
import math

import pytest

from summarize_countcap_multimodel_long_speed_20260726 import summarize


def write_case(path, history_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, nll, prefill, decode, attention in (
        ("full_attention", 2.0, 4.0, 8.0, history_tokens),
        ("direct_countcap", 2.02, 4.5, 4.0, 1280),
    ):
        rows.append(
            {
                "method": method,
                "tokens": 101,
                "nll": nll,
                "dense_prompt_seconds": prefill,
                "sparse_decode_seconds": decode,
                "configured_attention_tokens_mean": attention,
                "actual_attention_tokens_mean": attention,
                "actual_attention_tokens_min": attention,
                "actual_attention_tokens_max": attention,
                "cache_mode": "auto",
                "used_preallocated_cache": True,
                "history_tokens": history_tokens,
                "projection_dim": 48 if method == "direct_countcap" else 0,
                "score_mode": (
                    "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_"
                    "qprojscan_qkvsplitauto"
                    if method == "direct_countcap"
                    else ""
                ),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_reports_paired_quality_budget_and_speed(tmp_path) -> None:
    write_case(
        tmp_path / "qwen3_4b" / "length64000_mixed_a" / "case_summary.csv",
        64000,
    )
    rows = summarize(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "qwen3_4b"
    assert row["paired_cases"] == 1
    assert row["decode_speedup"] == 2.0
    assert row["configured_attention_ratio"] == pytest.approx(0.02)
    assert row["ppl_retention"] == pytest.approx(math.exp(-0.02))
    assert row["additional_fixed_seconds_per_case"] == pytest.approx(0.5)
    assert row["saved_seconds_per_decode_step"] == pytest.approx(0.04)
    assert row["break_even_decode_steps"] == pytest.approx(12.5)


def test_summarize_rejects_non_frozen_cache_path(tmp_path) -> None:
    path = (
        tmp_path
        / "qwen3_4b"
        / "length64000_mixed_a"
        / "case_summary.csv"
    )
    write_case(path, 64000)
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    rows[0]["used_preallocated_cache"] = "False"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="did not use preallocated cache"):
        summarize(tmp_path)
