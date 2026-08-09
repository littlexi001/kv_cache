from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from summarize_qksieve_persistent_kv_20260810 import summarize  # noqa: E402


def write_result(root: Path, method: str, warm: float, cold: float) -> None:
    path = root / "n32768" / "seed7" / f"{method}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "qksieve_persistent_kv_lifecycle_v2",
        "method": method,
        "history_tokens": 32768,
        "shared_prefix_warm_mean_ms_per_token": warm,
        "cold_persistent_request_ms_per_token": cold,
        "shared_prefix_amortized_ms_per_token": warm + 1.0,
        "append_only_ms_per_token": warm - 1.0,
        "prebuild_wall_seconds": 1.25 if method == "qksieve_robust" else 0.0,
        "reuse_tokens_equal": True,
        "index_buffers_reused_without_rebuild": True,
        "rewind_value_layers_correct": True,
        "persistent_contract_passed": True,
        "branches": [{"first_step_ms": warm}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summary_pairs_methods_and_computes_speedups(tmp_path: Path) -> None:
    write_result(tmp_path, "full", warm=80.0, cold=80.0)
    write_result(tmp_path, "qksieve_robust", warm=40.0, cold=100.0)

    result = summarize(tmp_path)

    assert result["all_correct"]
    assert not result["missing_pairs"]
    row = result["rows"][0]
    assert row["warm_speedup"] == 2.0
    assert row["cold_speedup"] == 0.8
    assert row["qksieve_prebuild_seconds"] == 1.25
