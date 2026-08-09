import csv
import json

import summarize_128k_multitopic_windows_20260716 as summary


def test_summary_reports_position_gaps_and_physical_ratio(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    full = {
        "topic": "computer",
        "window_index": 0,
        "target_token_ids": [11, 12],
        "target_token_texts": ["a", "b"],
        "token_nll": [1.0, 2.0],
        "nll": 1.5,
        "ppl": 4.4817,
        "prefill_seconds": 10.0,
        "synchronized_model_forward_seconds": 4.0,
        "process_peak_gpu_allocated_during_prefill_decode": 1000,
    }
    sparse = {
        **full,
        "token_nll": [1.1, 2.2],
        "nll": 1.65,
        "ppl": 5.2069,
        "attention_fraction": 0.015,
        "candidate_fraction": 0.02,
        "hierarchical_over_final_length_full_kv": 0.10,
        "pinned_host_bytes": 1000,
        "original_remote_full_gpu_kv_bytes": 1000,
        "mean_cache_hit_rate": 0.8,
        "cache_conversion_seconds": 1.0,
        "online_seconds": 2.0,
        "process_peak_gpu_allocated_during_prefill_conversion": 900,
    }
    (input_dir / "computer_w0_full.json").write_text(
        json.dumps(full), encoding="utf-8"
    )
    (input_dir / "computer_w0_sparse.json").write_text(
        json.dumps(sparse), encoding="utf-8"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "summary",
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--expected_cases",
            "1",
            "--bootstrap_samples",
            "20",
        ],
    )

    summary.main()

    paired = list(csv.DictReader((output_dir / "paired_cases.csv").open()))
    assert float(paired[0]["logical_attention_token_ratio"]) == 0.015
    assert float(paired[0]["persistent_gpu_tensor_bytes_ratio"]) == 0.10
    positions = list(
        csv.DictReader((output_dir / "position_nll_gaps.csv").open())
    )
    assert [round(float(row["delta_nll"]), 6) for row in positions] == [0.1, 0.2]
    aggregate = list(csv.DictReader((output_dir / "position_summary.csv").open()))
    assert len(aggregate) == 2
