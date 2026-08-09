from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_sample_calibrated_longbench_20260717 as longbench  # noqa: E402
import run_sample_calibrated_ruler_20260717 as ruler  # noqa: E402
import run_head_top2_targeted_ppl_20260714 as head  # noqa: E402
import summarize_qksieve_frozen_longbench_20260807 as summary  # noqa: E402
import summarize_qksieve_frozen_c64_ruler_20260807 as ruler_summary  # noqa: E402
import summarize_qksieve_resident_key_ab_20260807 as resident_summary  # noqa: E402
import summarize_qksieve_valuesketch_weak_task_ab_20260807 as value_summary  # noqa: E402
import summarize_qksieve_tailalpha_length_ab_20260807 as alpha_summary  # noqa: E402
import collect_real_qk_trace_20260715 as trace_collector  # noqa: E402
import analyze_qksieve_tail_partition_calibration_20260803 as tail_analysis  # noqa: E402


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        countcap_direct_fraction_override=0.0,
        sampled_quantile_sample_count=256,
        sampled_quantile_target_tail_count=0,
    )


def test_frozen_method_is_shared_by_longbench_and_ruler() -> None:
    method = longbench.QKSIEVE_FROZEN_C64_METHOD
    assert longbench.parse_methods(f"full_kv,{method}") == [
        "full_kv",
        method,
    ]
    assert ruler.probe.QKSIEVE_FROZEN_C64_METHOD == method
    assert longbench.uses_dense_prompt_suffix(method)


def test_frozen_method_has_dynamic_budget_and_c64_resolution() -> None:
    expected = {
        8192: (492, 1280),
        16384: (984, 1280),
        32768: (1280, 1792),
        65536: (1280, 3328),
        131072: (1280, 6656),
    }
    for history_tokens, (attention_tokens, sample_count) in expected.items():
        config = longbench.sparse_method_config(
            longbench.QKSIEVE_FROZEN_C64_METHOD,
            history_tokens,
            longbench.FROZEN_BUDGET_FRACTIONS,
            _args(),
        )
        assert config["attention_tokens"] == attention_tokens
        assert config["budget_fractions"] == (
            attention_tokens / history_tokens,
        )
        assert config["sampled_quantile_sample_count"] == sample_count
        assert config["score_mode"] == longbench.QKSIEVE_FROZEN_C64_SCORE_MODE


def test_frozen_method_reports_complete_auxiliary_index_rate() -> None:
    assert (
        longbench.configured_index_bits_per_token(
            longbench.QKSIEVE_FROZEN_C64_SCORE_MODE
        )
        == 306.0
    )
    assert (
        longbench.QKSIEVE_FROZEN_C64_SCORE_MODE
        in longbench.QKSIEVE_GLOBAL_WMMA_SAMPLED_SCORE_MODES
    )
    assert (
        longbench.QKSIEVE_FROZEN_C64_METHOD
        not in longbench.QKSIEVE_GLOBAL_WMMA_SAMPLED_METHODS
    )


def test_every_sampled_qk_mode_captures_dense_suffix_queries() -> None:
    assert (
        longbench.QKSIEVE_GLOBAL_WMMA_SAMPLED_SCORE_MODES
        <= longbench.PACKED_QUERY_CALIBRATED_SCORE_MODES
    )


def test_debug_value_sketch_switch_is_isolated_and_reversible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, object] = {}
    monkeypatch.delenv("QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH", raising=False)
    head._configure_packed_qmse_state(
        state, longbench.QKSIEVE_FROZEN_C64_SCORE_MODE
    )
    assert state["packed_qmse_value_sketch_rank"] == 16
    assert state["packed_qmse_value_sketch_bits"] == 4
    assert state["packed_qmse_deterministic_compaction"]
    assert not state["packed_qmse_debug_value_sketch_disabled"]

    monkeypatch.setenv("QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH", "1")
    head._configure_packed_qmse_state(
        state, longbench.QKSIEVE_FROZEN_C64_SCORE_MODE
    )
    assert state["packed_qmse_value_sketch_rank"] == 0
    assert state["packed_qmse_value_sketch_bits"] == 0
    assert not state["packed_qmse_deterministic_compaction"]
    assert state["packed_qmse_debug_value_sketch_disabled"]

    monkeypatch.delenv("QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH")
    head._configure_packed_qmse_state(
        state, longbench.QKSIEVE_FROZEN_C64_SCORE_MODE
    )
    assert state["packed_qmse_value_sketch_rank"] == 16
    assert state["packed_qmse_value_sketch_bits"] == 4
    assert state["packed_qmse_deterministic_compaction"]
    assert not state["packed_qmse_debug_value_sketch_disabled"]
    assert (
        longbench.QKSIEVE_FROZEN_C64_SCORE_MODE
        in longbench.PACKED_QUERY_CALIBRATED_SCORE_MODES
    )


def test_trace_collector_reads_local_jsonl_without_dataset_download(
    tmp_path: Path,
) -> None:
    source = tmp_path / "contexts.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"context": value})
            for value in ("alpha beta", "gamma delta")
        )
        + "\n",
        encoding="utf-8",
    )

    class Tokenizer:
        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            return {"input_ids": [ord(char) for char in text]}

    first = trace_collector.encode_jsonl_stream(
        Tokenizer(),
        source,
        "context",
        96,
        7,
        repeat_documents=True,
    )
    second = trace_collector.encode_jsonl_stream(
        Tokenizer(),
        source,
        "context",
        96,
        7,
        repeat_documents=True,
    )
    assert len(first) == 96
    assert first == second


def test_block_moment_quantization_respects_requested_precision() -> None:
    import torch

    tensor = torch.tensor([[0.1, -0.7, 1.3], [2.1, -3.2, 0.4]])
    fp16 = tail_analysis.quantize_block_moment(tensor, 16, (1,))
    int4 = tail_analysis.quantize_block_moment(tensor, 4, (1,))
    assert torch.equal(fp16, tensor.half().float())
    assert torch.isfinite(int4).all()
    assert not torch.equal(int4, tensor)


def test_frozen_csv_contract_audits_budget_samples_and_no_fallback() -> None:
    summary.audit_sparse_row(
        {
            "executed_path": longbench.QKSIEVE_FROZEN_C64_METHOD,
            "configured_score_mode": longbench.QKSIEVE_FROZEN_C64_SCORE_MODE,
            "configured_index_bits_per_token": "306",
            "prefix_tokens": "7500",
            "configured_attention_tokens": "450",
            "configured_sampled_quantile_sample_count": "1280",
            "qk_prebuild_layers": "36",
            "qk_batched_allocation_layers": "36",
        }
    )


def test_ruler_records_the_same_frozen_execution_contract() -> None:
    config = longbench.sparse_method_config(
        longbench.QKSIEVE_FROZEN_C64_METHOD,
        16384,
        longbench.FROZEN_BUDGET_FRACTIONS,
        _args(),
    )
    fields = ruler.method_audit_fields(
        longbench.QKSIEVE_FROZEN_C64_METHOD,
        config,
        {
            "qk_prebuild_seconds": 0.25,
            "qk_prebuild_layers": 36,
            "qk_batched_allocation_layers": 36,
        },
    )
    expected = {
        "executed_path": longbench.QKSIEVE_FROZEN_C64_METHOD,
        "configured_sampled_quantile_sample_count": 1280,
        "configured_index_bits_per_token": 306.0,
        "index_build_seconds": 0.0,
        "qk_prebuild_seconds": 0.25,
        "qk_prebuild_layers": 36,
        "qk_batched_allocation_layers": 36,
    }
    assert {key: fields[key] for key in expected} == expected
    for key in (
        "packed_qmse_sample_count",
        "packed_qmse_value_sketch_rank",
        "packed_qmse_value_sketch_bits",
        "packed_qmse_value_sketch_executed",
        "packed_qmse_value_sketch_tail_alpha",
        "packed_qmse_debug_value_sketch_disabled",
        "sampled_candidate_overflow_fraction",
        "sampled_quantile_fallback",
    ):
        assert fields[key] == 0.0


def test_ruler_summary_requires_strict_frozen_pairs() -> None:
    history_tokens = 4000
    config = longbench.sparse_method_config(
        longbench.QKSIEVE_FROZEN_C64_METHOD,
        history_tokens,
        longbench.FROZEN_BUDGET_FRACTIONS,
        _args(),
    )
    common = {
        "base_task": "niah_single_1",
        "requested_length": "4096",
        "sample_id": "sample-0",
        "score": "1.0",
        "prompt_tokens": "4032",
        "prefix_tokens": str(history_tokens),
        "suffix_tokens": "32",
        "generated_tokens": "10",
        "prefill_seconds": "1.0",
        "query_seconds": "0.1",
        "decode_seconds": "1.0",
        "online_seconds": "1.1",
        "total_seconds": "2.1",
    }
    full = {
        **common,
        "method": "full_kv",
        "configured_attention_fraction": "1.0",
    }
    sparse = {
        **common,
        "method": longbench.QKSIEVE_FROZEN_C64_METHOD,
        "executed_path": longbench.QKSIEVE_FROZEN_C64_METHOD,
        "configured_score_mode": longbench.QKSIEVE_FROZEN_C64_SCORE_MODE,
        "configured_index_bits_per_token": "306",
        "configured_attention_tokens": str(config["attention_tokens"]),
        "configured_attention_fraction": str(
            config["budget_fractions"][-1]
        ),
        "configured_sampled_quantile_sample_count": str(
            config["sampled_quantile_sample_count"]
        ),
        "qk_prebuild_layers": "36",
        "qk_batched_allocation_layers": "36",
    }
    payload = ruler_summary.summarize(
        [full, sparse], ("niah_single_1",), {4096: 1}
    )
    assert payload["strict_pairs"] == 1
    assert payload["fallback_count"] == 0
    assert payload["overall"]["quality_retention"] == 1.0


def test_resident_key_summary_counts_build_only_for_first_request(
    tmp_path: Path,
) -> None:
    base_row = {
        "method": "direct_countcap",
        "score_mode": longbench.QKSIEVE_FROZEN_C64_SCORE_MODE,
        "max_exact_tokens_per_head": 1280,
        "requested_quantile_sample_count_per_head": 1792,
        "packed_mean_bits_by_band": [4, 4, 2, 1, 0, 0, 0, 0],
        "packed_active_fraction_by_band": [1, 1, 1, 1, 0, 0, 0, 0],
        "nll": 1.0,
        "ppl": 2.0,
        "target_nll_delta_mean": 0.0,
        "quality_retention": 1.0,
        "fixed_sparse_overhead_seconds": 0.4,
        "steady_sparse_seconds_per_step": 0.05,
    }
    for mode, hits, qk_seconds, resident_seconds, fixed in (
        ("off", 0, 0.2, 0.0, 0.4),
        ("on", 36, 0.1, 0.3, 0.3),
    ):
        payload = {
            "history_tokens": 32768,
            "eval_tokens": 32,
            "target_token_ids_sha256": "same",
            "resident_key_factor_precompute": {
                "total_seconds": resident_seconds
            },
            "rows": [
                {
                    **base_row,
                    "fixed_sparse_overhead_seconds": fixed,
                    "packed_parallel_qk_prebuild": {
                        "layers": 36,
                        "resident_key_hits": hits,
                        "total_seconds": qk_seconds,
                    },
                }
            ],
        }
        path = tmp_path / mode / "quality"
        path.mkdir(parents=True)
        (path / "summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    result = resident_summary.summarize(tmp_path)
    assert result["numerically_identical"]
    assert result["request_local_qk_prebuild_seconds"]["speedup"] == 2.0
    assert result["subsequent_request_fixed_seconds"]["speedup"] == pytest.approx(
        4 / 3
    )
    assert result["first_request_fixed_seconds"]["speedup"] == pytest.approx(
        2 / 3
    )


def test_value_sketch_summary_requires_disabled_runtime_contract() -> None:
    def row(task: str, sample_id: str, method: str, score: float) -> dict[str, str]:
        return {
            "task": task,
            "sample_id": sample_id,
            "method": method,
            "executed_path": method,
            "score": str(score),
            "prediction": f"{method}-{task}",
        }

    reference: list[dict[str, str]] = []
    ab: list[dict[str, str]] = []
    for task in value_summary.TASKS:
        reference.append(row(task, "0", "full_kv", 1.0))
        reference.append(row(task, "0", value_summary.METHOD, 0.9))
        no_value = row(task, "0", value_summary.METHOD, 0.95)
        no_value.update(
            {
                "packed_qmse_debug_value_sketch_disabled": "1",
                "packed_qmse_value_sketch_rank": "0",
                "packed_qmse_value_sketch_bits": "0",
            }
        )
        ab.append(no_value)
    payload = value_summary.summarize(reference, ab, expected_pairs=3)
    assert payload["strict_triples"] == 3
    assert payload["macro_quality_retention"]["frozen_current"] == pytest.approx(0.9)
    assert payload["macro_quality_retention"]["no_value_sketch"] == pytest.approx(0.95)


def test_value_sketch_summary_accepts_strict_alpha_zero_ablation() -> None:
    reference: list[dict[str, str]] = []
    ab: list[dict[str, str]] = []
    for task in value_summary.TASKS:
        for method, score in (("full_kv", 1.0), (value_summary.METHOD, 0.9)):
            reference.append(
                {
                    "task": task,
                    "sample_id": "0",
                    "method": method,
                    "executed_path": method,
                    "score": str(score),
                    "prediction": method,
                }
            )
        ab.append(
            {
                "task": task,
                "sample_id": "0",
                "method": value_summary.METHOD,
                "executed_path": value_summary.METHOD,
                "score": "0.95",
                "prediction": "alpha0",
                "packed_qmse_debug_value_sketch_disabled": "0",
                "packed_qmse_value_sketch_rank": "16",
                "packed_qmse_value_sketch_bits": "4",
                "packed_qmse_value_sketch_tail_alpha": "0",
                "sampled_candidate_overflow_fraction": "0",
            }
        )
    payload = value_summary.summarize(
        reference, ab, expected_pairs=3, ablation="tail_alpha0"
    )
    assert payload["ablation"] == "tail_alpha0"
    assert payload["debug_contract"]["value_sketch_rank"] == 16


def test_tail_alpha_length_summary_compares_identical_targets(tmp_path: Path) -> None:
    target_hash = "same-targets"
    for name, alpha, sparse_nll in (("a1", 1.0, 1.1), ("a0", 0.0, 1.05)):
        root = tmp_path / name
        for history in (32768, 65536, 131072):
            output = root / f"n{history}"
            output.mkdir(parents=True)
            payload = {
                "target_token_ids_sha256": target_hash,
                "rows": [
                    {"method": "full_attention", "nll": 1.0},
                    {
                        "method": "direct_countcap",
                        "nll": sparse_nll,
                        "packed_value_sketch_tail_alpha": alpha,
                        "actual_attention_tokens_mean": 1280,
                        "actual_attention_fraction_mean": 0.01,
                        "steady_sparse_seconds_per_step": 0.1,
                        "fixed_sparse_overhead_seconds": 0.5,
                    },
                ],
            }
            (output / "summary.json").write_text(json.dumps(payload))
    result = alpha_summary.summarize(tmp_path / "a1", tmp_path / "a0")
    row = result["lengths"]["32768"]
    assert row["alpha0_minus_alpha1_nll"] == pytest.approx(-0.05)
    assert row["alpha0_quality_retention"] > row["alpha1_quality_retention"]
