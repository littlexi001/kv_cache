"""Auditable numerical contract for the frozen QKSieve-Robust method."""

from __future__ import annotations

import math
from typing import Any, Mapping

import run_sample_calibrated_longbench_20260717 as benchmark


METHOD = benchmark.QKSIEVE_FROZEN_C64_METHOD
SCORE_MODE = benchmark.QKSIEVE_FROZEN_C64_SCORE_MODE
INDEX_BITS_PER_TOKEN_PER_HEAD = 306.0
MAX_QUANTILE_SAMPLE_COUNT = 512
VALUE_SKETCH_RANK = 16
VALUE_SKETCH_BITS = 4
VALUE_SKETCH_TAIL_ALPHA = 0.5


def expected_attention_tokens(history_tokens: int) -> int:
    if history_tokens <= 0:
        raise ValueError("history_tokens must be positive")
    return min(
        history_tokens,
        1280,
        max(256, math.ceil(0.06 * history_tokens)),
    )


def expected_configured_sample_count(history_tokens: int) -> int:
    budget = expected_attention_tokens(history_tokens)
    return benchmark.tail_resolution_sample_count(64, budget / history_tokens)


def expected_effective_sample_count(history_tokens: int) -> int:
    return min(
        history_tokens,
        MAX_QUANTILE_SAMPLE_COUNT,
        expected_configured_sample_count(history_tokens),
    )


def _number(row: Mapping[str, Any], field: str) -> float:
    if field not in row or row[field] in (None, ""):
        raise AssertionError(f"missing frozen-contract field: {field}")
    return float(row[field])


def audit_sparse_row(row: Mapping[str, Any]) -> None:
    """Reject any row that did not execute the frozen Robust path exactly."""

    if row.get("executed_path") != METHOD:
        raise AssertionError(
            f"quality/cost fallback detected: {row.get('executed_path')}"
        )
    if row.get("configured_score_mode") != SCORE_MODE:
        raise AssertionError("frozen score mode mismatch")
    if not math.isclose(
        _number(row, "configured_index_bits_per_token"),
        INDEX_BITS_PER_TOKEN_PER_HEAD,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise AssertionError("auxiliary index must be 306 bits/token/head")

    history_tokens = int(_number(row, "prefix_tokens"))
    expected_budget = expected_attention_tokens(history_tokens)
    actual_budget = int(_number(row, "configured_attention_tokens"))
    if actual_budget != expected_budget:
        raise AssertionError(
            f"budget mismatch for {history_tokens}: "
            f"{actual_budget} != {expected_budget}"
        )

    expected_configured = expected_configured_sample_count(history_tokens)
    actual_configured = int(
        _number(row, "configured_sampled_quantile_sample_count")
    )
    if actual_configured != expected_configured:
        raise AssertionError(
            "configured sample-count mismatch: "
            f"{actual_configured} != {expected_configured}"
        )
    expected_effective = expected_effective_sample_count(history_tokens)
    actual_effective = int(round(_number(row, "packed_qmse_sample_count")))
    if actual_effective != expected_effective:
        raise AssertionError(
            "effective sample-count mismatch: "
            f"{actual_effective} != {expected_effective}"
        )

    rank = int(round(_number(row, "packed_qmse_value_sketch_rank")))
    bits = int(round(_number(row, "packed_qmse_value_sketch_bits")))
    executed = _number(row, "packed_qmse_value_sketch_executed")
    alpha = _number(row, "packed_qmse_value_sketch_tail_alpha")
    disabled = _number(row, "packed_qmse_debug_value_sketch_disabled")
    if rank != VALUE_SKETCH_RANK or bits != VALUE_SKETCH_BITS:
        raise AssertionError(f"ValueSketch mismatch: rank={rank}, bits={bits}")
    if executed <= 0.0:
        raise AssertionError("ValueSketch was configured but not executed")
    if not math.isclose(
        alpha, VALUE_SKETCH_TAIL_ALPHA, rel_tol=0.0, abs_tol=1e-8
    ):
        raise AssertionError(
            f"ValueSketch alpha mismatch: {alpha} != {VALUE_SKETCH_TAIL_ALPHA}"
        )
    if disabled != 0.0:
        raise AssertionError("ValueSketch debug-disable path was active")

    prebuilt = int(_number(row, "qk_prebuild_layers"))
    batched = int(_number(row, "qk_batched_allocation_layers"))
    if prebuilt <= 0 or batched != prebuilt:
        raise AssertionError(
            f"QK prebuild/batched allocation mismatch: {prebuilt}/{batched}"
        )


def contract_payload() -> dict[str, Any]:
    return {
        "method": METHOD,
        "score_mode": SCORE_MODE,
        "budget": "min(N,1280,max(256,ceil(0.06*N)))",
        "configured_quantile_tail_anchors": 64,
        "effective_quantile_samples_max": MAX_QUANTILE_SAMPLE_COUNT,
        "auxiliary_index_bits_per_token_per_head": (
            INDEX_BITS_PER_TOKEN_PER_HEAD
        ),
        "value_sketch": {
            "rank": VALUE_SKETCH_RANK,
            "bits": VALUE_SKETCH_BITS,
            "block_tokens": 256,
            "tail_alpha": VALUE_SKETCH_TAIL_ALPHA,
        },
        "full_attention_fallback": False,
        "length_switch": False,
    }
