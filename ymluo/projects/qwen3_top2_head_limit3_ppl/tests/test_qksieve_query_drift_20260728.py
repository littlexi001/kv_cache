from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import analyze_qksieve_query_drift_20260728 as drift  # noqa: E402
import collect_qksieve_teacher_forced_drift_20260728 as collector  # noqa: E402
import verify_qksieve_frozen_evidence_20260728 as verifier  # noqa: E402


def synthetic_trace() -> dict[str, object]:
    generator = torch.Generator().manual_seed(20260728)
    prompt_tokens = 128
    query_heads = 2
    prompt_tail = torch.randn(
        1,
        query_heads,
        32,
        drift.HEAD_DIM,
        generator=generator,
    )
    prompt_key = torch.randn(
        1,
        1,
        prompt_tokens,
        drift.HEAD_DIM,
        generator=generator,
    )
    current_key = torch.randn(
        1,
        1,
        1,
        drift.HEAD_DIM,
        generator=generator,
    )
    steps = (0, 7, 63, 255)
    records = []
    for index, step in enumerate(steps):
        query = prompt_tail[:, :, -(index + 1), :].unsqueeze(-2).clone()
        if step == 255:
            query[..., :8] += 5.0
        records.append(
            {
                "layer": 0,
                "step": step,
                "query": query,
                "key": (
                    torch.cat((prompt_key, current_key), dim=-2)
                    if step == 0
                    else None
                ),
                "value": None,
                "scaling": drift.HEAD_DIM**-0.5,
            }
        )
    return {
        "schema": drift.TRACE_SCHEMA,
        "model_name_or_path": "synthetic-model",
        "task": "synthetic",
        "sample_id": "sample-0",
        "method": drift.FROZEN_METHOD,
        "score_mode": drift.FROZEN_SCORE_MODE,
        "prompt_wrapper": "llama3",
        "prompt_truncation_mode": "official",
        "prompt_tokens": prompt_tokens,
        "prefix_tokens": prompt_tokens - 32,
        "suffix_tokens": 32,
        "query_calibration_tokens": 8,
        "recorded_prefill_query_tail_tokens": 32,
        "qk_metric_query_shrinkage": 0.75,
        "trace_layers": (0,),
        "trace_steps": steps,
        "generated_ids": list(range(300)),
        "prefill_query_tail": {0: prompt_tail},
        "records": records,
    }


def test_query_drift_analysis_is_strict_and_rate_matched() -> None:
    result = drift.analyze_payloads(
        [(Path("synthetic.pt"), synthetic_trace())],
        sample_stride=1,
        device=torch.device("cpu"),
    )

    assert len(result["allocations"]) == 5
    assert len(result["per_query"]) == 5 * 4 * 2
    assert {
        row["position_bucket"] for row in result["per_head_bucket"]
    } == {"0000-0063", "0064-0255"}
    assert all(
        row["allocated_index_bits"] <= drift.PHYSICAL_INDEX_BITS
        and row["reserved_index_bits"] == drift.PHYSICAL_INDEX_BITS
        for row in result["allocations"]
    )
    assert all(
        row["sampled_allocation_regret_diag"] >= 0.0
        for row in result["per_head_bucket"]
    )
    assert all(
        0.0 <= row["selected_attention_mass"] <= 1.0
        and 0.0 <= row["oracle_active_attention_mass"] <= 1.0
        for row in result["per_query"]
    )
    assert result["summary"]["coverage"]["max_observed_step"] == 255
    assert not result["summary"]["coverage"]["covers_1k_decode_query"]

    production_rows = [
        row
        for row in result["per_head_bucket"]
        if row["query_sample_count"] == 8
    ]
    early = next(
        row
        for row in production_rows
        if row["position_bucket"] == "0000-0063"
    )
    late = next(
        row
        for row in production_rows
        if row["position_bucket"] == "0064-0255"
    )
    assert (
        late["raw_covariance_drift_op_relative"]
        > early["raw_covariance_drift_op_relative"]
    )


def test_query_drift_validation_rejects_protocol_mismatch() -> None:
    payload = synthetic_trace()
    payload["method"] = "experimental"
    with pytest.raises(ValueError, match="expected frozen method"):
        drift.validate_trace(
            payload,
            trace_path=Path("bad-method.pt"),
            sample_counts=drift.DEFAULT_SAMPLE_COUNTS,
        )

    payload = synthetic_trace()
    payload["records"] = [
        record for record in payload["records"] if record["step"] != 0
    ]
    with pytest.raises(ValueError, match="include step zero"):
        drift.validate_trace(
            payload,
            trace_path=Path("missing-zero.pt"),
            sample_counts=drift.DEFAULT_SAMPLE_COUNTS,
        )

    payload = synthetic_trace()
    payload["prefill_query_tail"] = {
        0: payload["prefill_query_tail"][0][..., -16:, :]
    }
    with pytest.raises(ValueError, match="cannot support 32"):
        drift.validate_trace(
            payload,
            trace_path=Path("short-tail.pt"),
            sample_counts=drift.DEFAULT_SAMPLE_COUNTS,
        )


def test_query_drift_accepts_untraced_prompt_tail_layers() -> None:
    payload = synthetic_trace()
    payload["prefill_query_tail"][1] = payload["prefill_query_tail"][0].clone()
    validated = drift.validate_trace(
        payload,
        trace_path=Path("extra-prompt-layer.pt"),
        sample_counts=drift.DEFAULT_SAMPLE_COUNTS,
    )

    assert validated["trace_layers"] == (0,)
    assert set(validated["prompt_tail"]) == {0, 1}

    payload = synthetic_trace()
    payload["prefill_query_tail"] = {1: payload["prefill_query_tail"][0]}
    with pytest.raises(ValueError, match="missing trace layers"):
        drift.validate_trace(
            payload,
            trace_path=Path("missing-prompt-layer.pt"),
            sample_counts=drift.DEFAULT_SAMPLE_COUNTS,
        )


def test_teacher_forced_schema_and_bundle_are_explicit() -> None:
    stream = list(range(100))
    bundle, continuation = collector.build_teacher_forced_bundle(
        stream,
        history_tokens=64,
        recorded_query_tokens=32,
        continuation_steps=16,
    )
    assert bundle.input_ids.tolist() == [list(range(64))]
    assert bundle.query_start == 32
    assert bundle.suffix_token_count == 32
    assert continuation == list(range(64, 80))

    payload = synthetic_trace()
    payload["schema"] = drift.TEACHER_FORCED_TRACE_SCHEMA
    payload["trace_kind"] = "teacher_forced_corpus_continuation"
    payload["sequence_ids"] = payload.pop("generated_ids")
    validated = drift.validate_trace(
        payload,
        trace_path=Path("teacher.pt"),
        sample_counts=drift.DEFAULT_SAMPLE_COUNTS,
    )
    assert validated["trace_kind"] == "teacher_forced_corpus_continuation"


def test_query_drift_helpers_match_frozen_budget() -> None:
    assert drift.direct_target_count(2048) == 256
    assert drift.direct_target_count(8192) == math.ceil(0.06 * 8192)
    assert drift.direct_target_count(128000) == 1280
    assert drift.physical_index_bits((4, 4, 2, 1, 0, 0, 0, 0)) == 240
    assert drift.position_bucket(0) == "0000-0063"
    assert drift.position_bucket(4096) == "4096+"


def test_frozen_evidence_verifier_separates_trace_kinds() -> None:
    protocol = {
        "method": drift.FROZEN_METHOD,
        "score_mode": drift.FROZEN_SCORE_MODE,
        "query_sample_counts": [1, 4, 8],
        "production_query_samples": 8,
        "reserved_physical_index_bits": 240,
        "query_shrinkage": 0.75,
        "no_rerank_router_recent_sink_or_full_fallback": True,
    }
    counts = {
        "per_query_rows": 10,
        "per_head_bucket_rows": 5,
        "allocation_rows": 5,
    }
    free = {
        "schema": "qksieve_query_drift_analysis_v1",
        "protocol": protocol,
        "coverage": {
            "trace_kinds": ["free_generation"],
            "trace_count": 2,
            "max_observed_step": 63,
        },
        "counts": counts,
        "trace_sha256": {"trace.pt": "abc"},
    }
    validated = verifier.validate_query_drift(
        free,
        expected_kind="free_generation",
    )
    assert validated["kind"] == "free_generation"

    teacher = copy.deepcopy(free)
    teacher["protocol"]["query_sample_counts"] = [1, 4, 8, 16, 32]
    teacher["coverage"] = {
        "trace_kinds": ["teacher_forced_corpus_continuation"],
        "trace_count": 6,
        "max_observed_step": 4095,
        "covers_1k_decode_query": True,
        "covers_2k_decode_query": True,
        "covers_4k_decode_query": False,
    }
    with pytest.raises(AssertionError, match="covers_4k"):
        verifier.validate_query_drift(
            teacher,
            expected_kind="teacher_forced_corpus_continuation",
        )
