import csv
import json

import summarize_batch_throughput_20260716 as summary


def make_case(batch: int) -> dict[str, float | int]:
    return {
        "history_tokens": 32000,
        "batch_size": batch,
        "decode_steps_per_sequence": 10,
        "full_mean_ppl": 10.0,
        "sparse_mean_ppl": 10.5,
        "full_prefill_seconds": 2.0 * batch,
        "full_forward_seconds": 1.0 * batch,
        "sparse_prefill_seconds": 2.0 * batch,
        "sparse_conversion_seconds": 0.5 * batch,
        "sparse_forward_seconds": 0.5 * batch,
        "full_tokens_per_second": 10.0,
        "sparse_tokens_per_second": 20.0,
        "throughput_speedup": 2.0,
        "sparse_over_final_full_kv": 0.1,
        "full_peak_gpu_allocated_bytes": 1000 * batch,
        "sparse_peak_gpu_allocated_bytes": 900 * batch,
        "mean_cache_hit_rate": 0.8,
    }


def test_batch_summary_reports_decode_and_protocol_metrics(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for batch in (1, 2):
        (input_dir / f"l32000_b{batch}.json").write_text(
            json.dumps(make_case(batch)), encoding="utf-8"
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
            "2",
        ],
    )

    summary.main()

    rows = list(csv.DictReader((output_dir / "summary.csv").open()))
    assert len(rows) == 2
    assert float(rows[0]["decode_throughput_speedup"]) == 2.0
    assert float(rows[0]["protocol_speedup"]) == 1.0
    assert float(rows[1]["sparse_batch_efficiency"]) == 0.5
