from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from summarize_qksieve_h100_20260810 import summarize  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_h100_summary_pairs_all_three_timing_contracts(tmp_path: Path) -> None:
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
                    "value_sketch_disabled": False,
                    "value_sketch_tail_alpha": 0.5,
                },
            )
            common = {
                "history_tokens": length,
                "cold_persistent_request_ms_per_token": 20.0,
                "shared_prefix_warm_mean_ms_per_token": 10.0,
                "shared_prefix_amortized_ms_per_token": 12.0,
                "append_only_ms_per_token": 8.0,
                "prebuild_wall_seconds": 0.0,
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
                    "shared_prefix_warm_mean_ms_per_token": 5.0,
                    "shared_prefix_amortized_ms_per_token": 6.0,
                    "append_only_ms_per_token": 4.0,
                    "prebuild_wall_seconds": 2.0,
                    "persistent_contract_passed": True,
                },
            )

    payload = summarize(tmp_path, expected_seeds=2)

    assert payload["attention"][0]["robust_speedup"] == 4.0
    assert payload["steady_decode"][0]["steady_decode_speedup"] == 3.0
    persistent = payload["persistent_requests"][0]
    assert persistent["cold_speedup"] == 0.5
    assert persistent["warm_speedup"] == 2.0
