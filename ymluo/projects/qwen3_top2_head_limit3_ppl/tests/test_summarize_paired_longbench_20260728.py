import hashlib

import pytest

from summarize_paired_longbench_20260728 import summarize
from verify_qksieve_frozen_evidence_20260728 import (
    FIER_METHOD,
    FIER_SCORE_MODE,
    FROZEN_METHOD,
    METHOD,
    SCORE_MODE,
    validate_fier_comparison,
)


def _row(task, sample_id, method, score, seconds):
    return {
        "task": task,
        "sample_id": sample_id,
        "method": method,
        "score": str(score),
        "prefill_seconds": str(seconds * 2),
        "query_seconds": str(seconds),
        "decode_seconds": str(seconds * 3),
        "online_seconds": str(seconds * 4),
        "total_seconds": str(seconds * 6),
    }


def test_strict_multi_method_summary_reports_macro_and_speed():
    methods = ("full_kv", "qksieve", "fier")
    rows = []
    scores = {
        "full_kv": (1.0, 0.5, 0.8, 0.4),
        "qksieve": (1.0, 0.5, 0.8, 0.4),
        "fier": (0.8, 0.4, 0.6, 0.3),
    }
    seconds = {"full_kv": 2.0, "qksieve": 1.0, "fier": 1.25}
    keys = (("a", "0"), ("a", "1"), ("b", "0"), ("b", "1"))
    for method in methods:
        for (task, sample_id), score in zip(keys, scores[method]):
            rows.append(
                _row(task, sample_id, method, score, seconds[method])
            )

    result = summarize(
        rows=rows,
        methods=methods,
        reference_method="full_kv",
        expected_pairs=4,
        expected_tasks=2,
        bootstrap_resamples=100,
        seed=7,
    )

    assert result["strict_pairs"] == 4
    assert result["tasks"] == 2
    assert result["methods"]["full_kv"]["macro_score"] == 0.675
    assert result["methods"]["qksieve"]["quality_retention"] == 1.0
    assert result["methods"]["qksieve"]["online_speedup"] == 2.0
    assert result["methods"]["fier"]["online_speedup"] == 1.6
    assert (
        len(result["methods"]["fier"]["quality_retention_95ci"]) == 2
    )


def test_frozen_evidence_requires_the_true_packed_fier_contract(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("packed = True\n", encoding="utf-8")
    summary = {
        "strict_pairs": 3750,
        "tasks": 16,
        "counts": {
            "full_kv": 3750,
            METHOD: 3750,
            FIER_METHOD: 3750,
        },
        "methods": {
            "full_kv": {"online_seconds": 2.0},
            METHOD: {
                "online_seconds": 1.0,
                "quality_retention_95ci": [0.99, 1.01],
            },
            FIER_METHOD: {
                "online_seconds": 1.2,
                "quality_retention_95ci": [0.95, 0.98],
            },
        },
    }
    contract = {
        "protocol": {
            "fallback": False,
            "rerank": False,
            "recent_or_sink_reservation": False,
        },
        "shared_budget": FROZEN_METHOD["budget"],
        "qksieve": {
            "score_mode": SCORE_MODE,
            "index_bits_per_token_per_kv_head": 240,
        },
        "fier": {
            "score_mode": FIER_SCORE_MODE,
            "sequence_group_size": 32,
            "index_bits_per_token_per_kv_head": 256,
        },
        "shared_final_attention": (
            "qabs_cuda_kernels exact sparse attention"
        ),
        "source_sha256": {
            "kernel.py": hashlib.sha256(source.read_bytes()).hexdigest()
        },
    }

    result = validate_fier_comparison(
        summary, contract, project_root=tmp_path
    )

    assert result["strict_pairs"] == 3750
    contract["fier"]["score_mode"] = "fier_rtn1_g32_fulltopk"
    with pytest.raises(AssertionError, match="packed FIER mode mismatch"):
        validate_fier_comparison(
            summary, contract, project_root=tmp_path
        )
