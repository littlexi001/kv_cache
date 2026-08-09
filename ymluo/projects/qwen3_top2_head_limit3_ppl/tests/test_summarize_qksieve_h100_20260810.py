from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from summarize_qksieve_h100_20260810 import summarize  # noqa: E402
import qksieve_robust_contract_20260810 as contract  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_h100_summary_pairs_all_three_timing_contracts(tmp_path: Path) -> None:
    software = {
        "python": "3.11.0",
        "pytorch": "2.7.0",
        "transformers": "4.55.0",
        "cuda_runtime": "12.8",
        "cudnn": 90501,
    }
    for seed_index in range(2):
        write_json(
            tmp_path / "attention" / f"seed{seed_index}.json",
            {
                "gpu": "NVIDIA H100 80GB HBM3",
                "rows": [
                    {
                        "history_count": length,
                        "full_mha_sdpa_ms": 8.0,
                        "qksieve_valuesketch_complete_ms": 2.0,
                        "qksieve_complete_ms": 1.0,
                        "fier_complete_ms": 4.0,
                        "qksieve_valuesketch_candidate_counts_equal": True,
                        "qksieve_valuesketch_candidate_sets_equal": True,
                        "qksieve_valuesketch_tail_alpha": 0.5,
                        "full_kv_bytes": 1000,
                        "qksieve_index_bytes": 50,
                        "qksieve_valuesketch_bytes": 20,
                        "fier_index_bytes": 10,
                    }
                    for length in (65536, 131072)
                ],
            },
        )
        for length in (65536, 131072):
            seed_dir = f"seed{seed_index}"
            write_json(
                tmp_path / "decode" / f"n{length}" / seed_dir / "full.json",
                {
                    "method": "full",
                    "history_tokens": length,
                    "gpu_name": "NVIDIA H100 80GB HBM3",
                    "steady_mean_ms_per_token": 12.0,
                    "prebuild_wall_seconds": 0.0,
                    "software": software,
                    "peak_memory": {
                        "allocated_bytes_total": 100,
                        "reserved_bytes_total": 200,
                    },
                },
            )
            write_json(
                tmp_path
                / "decode"
                / f"n{length}"
                / seed_dir
                / "qksieve_valuesketch_top1280.json",
                {
                    "method": "qksieve_valuesketch_top1280",
                    "history_tokens": length,
                    "gpu_name": "NVIDIA H100 80GB HBM3",
                    "steady_mean_ms_per_token": 4.0,
                    "prebuild_wall_seconds": 1.0,
                    "score_mode": contract.SCORE_MODE,
                    "value_sketch_disabled": False,
                    "value_sketch_tail_alpha": 0.5,
                    "software": software,
                    "peak_memory": {
                        "allocated_bytes_total": 120,
                        "reserved_bytes_total": 240,
                    },
                },
            )
            common = {
                "history_tokens": length,
                "cold_persistent_request_ms_per_token": 20.0,
                "cold_end_to_end_request_ms_per_token": 100.0,
                "shared_prefix_warm_mean_ms_per_token": 10.0,
                "shared_prefix_amortized_ms_per_token": 12.0,
                "append_only_ms_per_token": 8.0,
                "prebuild_wall_seconds": 0.0,
                "gpu_name": "NVIDIA H100 80GB HBM3",
                "software": software,
                "cold_peak_memory": {
                    "allocated_bytes_total": 100,
                    "reserved_bytes_total": 200,
                },
                "lifecycle_peak_memory": {
                    "allocated_bytes_total": 110,
                    "reserved_bytes_total": 220,
                },
            }
            write_json(
                tmp_path / "persistent" / f"n{length}" / seed_dir / "full.json",
                {"method": "full", **common},
            )
            write_json(
                tmp_path
                / "persistent"
                / f"n{length}"
                / seed_dir
                / "qksieve_robust.json",
                {
                    "method": "qksieve_robust",
                    **common,
                    "cold_persistent_request_ms_per_token": 40.0,
                    "cold_end_to_end_request_ms_per_token": 200.0,
                    "shared_prefix_warm_mean_ms_per_token": 5.0,
                    "shared_prefix_amortized_ms_per_token": 6.0,
                    "append_only_ms_per_token": 4.0,
                    "prebuild_wall_seconds": 2.0,
                    "score_mode": contract.SCORE_MODE,
                    "value_sketch_tail_alpha": 0.5,
                    "persistent_contract_passed": True,
                    "cold_peak_memory": {
                        "allocated_bytes_total": 120,
                        "reserved_bytes_total": 240,
                    },
                    "lifecycle_peak_memory": {
                        "allocated_bytes_total": 132,
                        "reserved_bytes_total": 264,
                    },
                },
            )

    payload = summarize(tmp_path, expected_seeds=2)

    assert payload["attention"][0]["robust_speedup"] == 4.0
    assert payload["attention"][0]["qksieve_total_auxiliary_ratio_of_full_kv"] == 0.07
    assert payload["steady_decode"][0]["steady_decode_speedup"] == 3.0
    assert payload["steady_decode"][0]["qksieve_to_full_peak_allocated_ratio"] == 1.2
    persistent = payload["persistent_requests"][0]
    assert persistent["cold_speedup"] == 0.5
    assert persistent["cold_end_to_end_speedup"] == 0.5
    assert persistent["warm_speedup"] == 2.0
    assert persistent["qksieve_to_full_cold_peak_reserved_ratio"] == 1.2
    assert persistent["qksieve_to_full_lifecycle_peak_allocated_ratio"] == 1.2
    assert payload["hardware"]["device_names"] == ["NVIDIA H100 80GB HBM3"]
    assert payload["hardware"]["software"] == software
    assert payload["frozen_contract"] == contract.contract_payload()
