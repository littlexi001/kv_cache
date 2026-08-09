from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import qksieve_robust_contract_20260810 as contract  # noqa: E402
import run_sample_calibrated_longbench_20260717 as long_runner  # noqa: E402
import summarize_qksieve_robust_longbench_20260810 as long_summary  # noqa: E402
import summarize_qksieve_robust_multimodel_20260810 as multi_summary  # noqa: E402
import summarize_qksieve_robust_ruler_20260810 as ruler_summary  # noqa: E402


def _sparse_fields(history: int) -> dict[str, str]:
    return {
        "executed_path": contract.METHOD,
        "configured_score_mode": contract.SCORE_MODE,
        "configured_index_bits_per_token": "306",
        "prefix_tokens": str(history),
        "configured_attention_tokens": str(
            contract.expected_attention_tokens(history)
        ),
        "configured_sampled_quantile_sample_count": str(
            contract.expected_configured_sample_count(history)
        ),
        "packed_qmse_sample_count": str(
            contract.expected_effective_sample_count(history)
        ),
        "packed_qmse_value_sketch_rank": "16",
        "packed_qmse_value_sketch_bits": "4",
        "packed_qmse_value_sketch_executed": "1",
        "packed_qmse_value_sketch_tail_alpha": "0.5",
        "packed_qmse_debug_value_sketch_disabled": "0",
        "qk_prebuild_layers": "32",
        "qk_batched_allocation_layers": "32",
    }


def _longbench_rows() -> list[dict[str, str]]:
    common = {
        "task": "narrativeqa",
        "sample_id": "heldout-0",
        "score": "0.5",
        "prompt_tokens": "7500",
        "generated_tokens": "16",
        "configured_attention_fraction": "0.06",
        "prefill_seconds": "1",
        "query_seconds": "0.1",
        "decode_seconds": "0.4",
        "online_seconds": "0.5",
        "total_seconds": "1.5",
    }
    return [
        {**common, "method": "full_kv"},
        {
            **common,
            **_sparse_fields(7400),
            "method": contract.METHOD,
            "score": "0.49",
        },
    ]


def _ruler_rows() -> list[dict[str, str]]:
    history = 4000
    common = {
        "task": "niah_single_1_4096",
        "base_task": "niah_single_1",
        "requested_length": "4096",
        "sample_id": "niah_single_1_4096_0",
        "score": "1",
        "prompt_tokens": "4096",
        "prefix_tokens": str(history),
        "suffix_tokens": "96",
        "generated_tokens": "8",
        "configured_attention_fraction": str(
            contract.expected_attention_tokens(history) / history
        ),
        "prefill_seconds": "1",
        "query_seconds": "0.1",
        "decode_seconds": "0.4",
        "online_seconds": "0.5",
        "total_seconds": "1.5",
    }
    return [
        {**common, "method": "full_kv"},
        {
            **common,
            **_sparse_fields(history),
            "method": contract.METHOD,
            "score": "0.99",
        },
    ]


def test_contract_distinguishes_configured_and_effective_samples() -> None:
    assert contract.expected_configured_sample_count(131072) == 6656
    assert contract.expected_effective_sample_count(131072) == 512
    contract.audit_sparse_row(_sparse_fields(131072))


def test_frozen_method_is_registered_by_the_shared_runner() -> None:
    assert long_runner.parse_methods(f"full_kv,{contract.METHOD}") == [
        "full_kv",
        contract.METHOD,
    ]
    assert long_runner.QKSIEVE_FROZEN_C64_METHOD == contract.METHOD


def test_contract_rejects_wrong_tail_alpha() -> None:
    row = _sparse_fields(32768)
    row["packed_qmse_value_sketch_tail_alpha"] = "1.0"
    with pytest.raises(AssertionError, match="alpha mismatch"):
        contract.audit_sparse_row(row)


def test_contract_rejects_uncapped_effective_sample_count() -> None:
    row = _sparse_fields(32768)
    row["packed_qmse_sample_count"] = row[
        "configured_sampled_quantile_sample_count"
    ]
    with pytest.raises(AssertionError, match="effective sample-count"):
        contract.audit_sparse_row(row)


def test_robust_longbench_summary_adds_task_bootstrap() -> None:
    payload = long_summary.summarize(
        _longbench_rows(),
        expected_pairs=1,
        expected_tasks=1,
        bootstrap_resamples=20,
        seed=7,
    )
    assert payload["schema"] == "qksieve_robust_longbench_summary_v1"
    assert payload["effective_sample_count_mean"] == 512
    assert payload["methods"][contract.METHOD][
        "quality_retention"
    ] == pytest.approx(0.98)
    assert payload["bootstrap"]["quality_retention_95ci"] == pytest.approx(
        [0.98, 0.98]
    )


def test_robust_ruler_summary_requires_strict_contract() -> None:
    payload = ruler_summary.summarize(
        _ruler_rows(),
        ("niah_single_1",),
        {4096: 1},
        bootstrap_resamples=20,
        seed=7,
    )
    assert payload["strict_pairs"] == 1
    assert payload["fallback_count"] == 0
    assert payload["overall"]["quality_retention"] == pytest.approx(0.99)
    assert math.isclose(payload["effective_sample_count_mean"], 512)


def test_multimodel_summary_requires_identical_frozen_contract(
    tmp_path: Path,
) -> None:
    for tag, retention in (("llama", 0.99), ("qwen", 1.01)):
        root = tmp_path / tag
        root.mkdir()
        (root / "manifest.txt").write_text(tag, encoding="utf-8")
        (root / "paired_summary.json").write_text(
            json.dumps(
                {
                    "schema": "qksieve_robust_longbench_summary_v1",
                    "strict_pairs": 1,
                    "tasks": 1,
                    "full_fallback_count": 0,
                    "frozen_contract": contract.contract_payload(),
                    "methods": {
                        "full_kv": {"macro_score": 0.5},
                        contract.METHOD: {
                            "macro_score": 0.5 * retention,
                            "quality_retention": retention,
                            "prompt_tokens": 7000,
                        },
                    },
                    "bootstrap": {"quality_retention_95ci": [0.98, 1.02]},
                    "attention_fraction_mean": 0.06,
                    "effective_sample_count_mean": 512,
                }
            ),
            encoding="utf-8",
        )
    payload = multi_summary.summarize(tmp_path, ("llama", "qwen"), 1, 1)
    assert payload["minimum_quality_retention"] == pytest.approx(0.99)


def test_quality_launchers_pin_postfreeze_runtime_contract() -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in (
        "launch_qksieve_robust_ruler_20260810.sh",
        "launch_qksieve_robust_multimodel_longbench_20260810.sh",
    ):
        text = (scripts / name).read_text(encoding="utf-8")
        assert "QKSIEVE_MAX_QUANTILE_SAMPLE_COUNT=512" in text
        assert "QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=0.5" in text
        assert "QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH=0" in text
        assert "--minimum_sparse_prefix_tokens 0" in text
        assert "numerical_freeze_commit_sha=" in text
        assert "audited_implementation_commit_sha=" in text


def test_h100_launcher_records_frozen_provenance() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "launch_qksieve_h100_matched_20260810.sh"
    ).read_text(encoding="utf-8")
    assert "schema=qksieve_h100_matched_protocol_v1" in script
    assert "numerical_freeze_commit_sha=" in script
    assert "audited_implementation_commit_sha=" in script
    assert "nvidia-smi --query-gpu=" in script
