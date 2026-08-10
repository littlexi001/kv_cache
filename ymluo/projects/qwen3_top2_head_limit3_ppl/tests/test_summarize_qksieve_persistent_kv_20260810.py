from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from summarize_qksieve_persistent_kv_20260810 import summarize  # noqa: E402


def write_result(
    root: Path,
    method: str,
    warm: float,
    cold: float,
    *,
    seed: int = 7,
) -> None:
    path = root / "n32768" / f"seed{seed}" / f"{method}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "qksieve_persistent_kv_lifecycle_v2",
        "method": method,
        "history_tokens": 32768,
        "shared_prefix_warm_mean_ms_per_token": warm,
        "cold_persistent_request_ms_per_token": cold,
        "cold_end_to_end_request_ms_per_token": cold * 2.0,
        "shared_prefix_amortized_ms_per_token": warm + 1.0,
        "append_only_ms_per_token": warm - 1.0,
        "prebuild_wall_seconds": 1.25 if method == "qksieve_robust" else 0.0,
        "reuse_tokens_equal": True,
        "index_buffers_reused_without_rebuild": True,
        "rewind_value_layers_correct": True,
        "persistent_contract_passed": True,
        "branches": [{"first_step_ms": warm}],
    }
    if method == "qksieve_robust":
        history = 32768
        branch_steps = 32
        append_steps = 128

        def snapshot(indexed_count: int) -> dict:
            return {
                "layer_count": 1,
                "layers": [
                    {
                        "layer": 0,
                        "key_indexed_count": indexed_count,
                        "value_indexed_count": indexed_count,
                        "key_rebuild_count": 1,
                        "value_rebuild_count": 1,
                        "key_code_ptr": 11,
                        "key_scale_ptr": 12,
                        "value_code_ptr": 13,
                        "value_minimum_ptr": 14,
                        "value_scale_ptr": 15,
                    }
                ],
            }

        replay = [1, 2, 3]
        payload.update(
            branch_count=4,
            branch_steps=branch_steps,
            append_steps=append_steps,
            post_decode_index_lag_tokens=1,
            qk_prebuild={"layers": 1},
            key_index_prebuild={"layers": 1, "existing_layers": 0},
            value_prebuild={"layers": 1},
            value_install={"layers": 1},
            runtime_extension_preload={
                "variablebit": 0.1,
                "query": 0.1,
                "mixedblock": 0.1,
                "value_attention": 0.1,
            },
            initial_persistent_state_snapshot=snapshot(history),
            persistent_state_snapshots=[
                *[
                    snapshot(history + branch_steps - 1)
                    for _ in range(5)
                ],
                snapshot(history + append_steps - 1),
            ],
            rewinds=[
                {
                    "active_length": history,
                    "key_layers": 1,
                    "value_layers": 1,
                }
                for _ in range(5)
            ],
            branches=[
                {
                    "first_step_ms": warm,
                    "generated_token_ids": replay,
                    "generated_token_sha256": "same",
                }
                for _ in range(5)
            ],
            reuse_hash_equal=True,
        )
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
    assert row["cold_end_to_end_speedup"] == 0.8
    assert row["qksieve_prebuild_seconds"] == 1.25
    aggregate = result["aggregate_rows"][0]
    assert aggregate["seed_count"] == 1
    assert aggregate["seeds"] == [7]
    assert aggregate["warm_speedup"] == 2.0
    assert aggregate["warm_speedup_bootstrap_ci95_low"] == 2.0
    assert aggregate["warm_speedup_bootstrap_ci95_high"] == 2.0


def test_summary_aggregates_independent_seeds(tmp_path: Path) -> None:
    for seed, sparse_warm in ((7, 40.0), (8, 50.0), (9, 80.0)):
        write_result(tmp_path, "full", warm=80.0, cold=80.0, seed=seed)
        write_result(
            tmp_path,
            "qksieve_robust",
            warm=sparse_warm,
            cold=100.0,
            seed=seed,
        )

    aggregate = summarize(tmp_path)["aggregate_rows"][0]

    assert aggregate["seed_count"] == 3
    assert aggregate["seeds"] == [7, 8, 9]
    assert aggregate["warm_speedup"] == 1.6
    assert aggregate["warm_speedup_min"] == 1.0
    assert aggregate["warm_speedup_max"] == 2.0
    assert aggregate["warm_speedup_bootstrap_ci95_low"] == 1.0
    assert aggregate["warm_speedup_bootstrap_ci95_high"] == 2.0


def test_protocol_audit_is_required_for_publication_summary(tmp_path: Path) -> None:
    write_result(tmp_path, "full", warm=80.0, cold=80.0)
    write_result(tmp_path, "qksieve_robust", warm=40.0, cold=100.0)

    try:
        summarize(tmp_path, require_protocol=True)
    except AssertionError as error:
        assert "manifest is missing" in str(error)
    else:
        raise AssertionError("publication summary accepted a missing manifest")
