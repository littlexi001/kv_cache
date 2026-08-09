"""Verify that QKSieve quality and speed artifacts use one frozen method.

This verifier intentionally rejects partially compatible artifacts.  A paper
table is valid only when its embedded method contract and source hashes agree
with the current implementation and with every other supplied artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


METHOD = "qksieve_fullprompt_auto_plain_fulltopk"
SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk"
)
FIER_METHOD = "fier_rtn1_g32_packed_fulltopk"
FIER_SCORE_MODE = "fier_rtn1_g32_packed_fulltopk"
KEYPCA_UNIFORM1_SCORE_MODE = (
    "pca_hierarchical_fixed11111111_packed_fulltopk"
)
QKBALANCED_UNIFORM1_SCORE_MODE = (
    "pca_hierarchical_fixed11111111_qkmetric_packed_fulltopk"
)
RANDOM_UNIFORM1_SCORE_MODE = (
    "pca_hierarchical_fixed11111111_random_packed_fulltopk"
)
KEYPCA_AUTOKEY_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_packed_fulltopk"
)
QKBALANCED_AUTOKEY_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_packed_fulltopk"
)
FROZEN_METHOD = {
    "method": METHOD,
    "score_mode": SCORE_MODE,
    "query_shrinkage": 0.75,
    "query_tail_tokens": 8,
    "index_bits_per_token_per_kv_head": 240,
    "budget": "min(N, 1280, max(256, ceil(0.06*N)))",
    "proxy_topk_dtype": "float32",
    "exact_kv_dtype": "float16",
    "rerank": False,
    "fallback": False,
    "recent_or_sink_reservation": False,
}
EXPECTED_LENGTHS = {16_000, 32_000, 64_000, 128_000}
EXPECTED_HORIZONS = {64, 256, 1024}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the frozen QKSieve paper evidence chain."
    )
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument(
        "--longbench_summary",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--ruler_summary", type=Path)
    parser.add_argument("--samepath_summary", type=Path)
    parser.add_argument("--multimodel_summary", type=Path)
    parser.add_argument("--fier_summary", type=Path)
    parser.add_argument("--fier_contract", type=Path)
    parser.add_argument("--query_drift_free_summary", type=Path)
    parser.add_argument("--query_drift_teacher_summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_contract(summary: dict[str, Any], label: str) -> None:
    actual = summary.get("frozen_method")
    if actual != FROZEN_METHOD:
        raise AssertionError(
            f"{label}: frozen method mismatch\n"
            f"expected={FROZEN_METHOD}\nactual={actual}"
        )


def validate_source_hashes(
    summary: dict[str, Any],
    project_root: Path,
    label: str,
) -> None:
    hashes = summary.get("source_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise AssertionError(f"{label}: source_sha256 is missing")
    for relative, expected in hashes.items():
        path = project_root / relative
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(
                f"{label}: source drift for {relative}: "
                f"artifact={expected}, current={actual}"
            )


def validate_runtime_source(project_root: Path) -> dict[str, Any]:
    src_root = project_root / "src"
    sys.path.insert(0, str(src_root))
    import run_head_top2_targeted_ppl_20260714 as attention
    import run_sample_calibrated_longbench_20260717 as longbench

    if longbench.QKSIEVE_FULLTOPK_METHOD != METHOD:
        raise AssertionError("LongBench method identifier drifted")
    if longbench.QKSIEVE_FULLTOPK_SCORE_MODE != SCORE_MODE:
        raise AssertionError("LongBench score-mode mapping drifted")
    if SCORE_MODE not in attention._PACKED_QMSE_SCORE_MODES:
        raise AssertionError("frozen score mode is not a packed-qMSE mode")
    if longbench.FIER_RTN1_G32_PACKED_FULLTOPK_METHOD != FIER_METHOD:
        raise AssertionError("packed FIER method identifier drifted")
    if (
        longbench.FIER_RTN1_G32_PACKED_FULLTOPK_SCORE_MODE
        != FIER_SCORE_MODE
    ):
        raise AssertionError("packed FIER score-mode mapping drifted")
    if FIER_SCORE_MODE not in attention._FIER_PACKED_SCORE_MODES:
        raise AssertionError("packed FIER mode is not registered")
    for score_mode in (
        KEYPCA_UNIFORM1_SCORE_MODE,
        QKBALANCED_UNIFORM1_SCORE_MODE,
        RANDOM_UNIFORM1_SCORE_MODE,
    ):
        schedule = attention._PACKED_QMSE_FIXED_ALLOCATIONS.get(
            score_mode
        )
        if schedule != (1,) * 8:
            raise AssertionError(
                f"uniform-1bit ablation drifted: {score_mode}"
            )
        physical_bits = 16 * sum(schedule) + 16 * sum(
            bit > 0 for bit in schedule
        )
        if physical_bits != 256:
            raise AssertionError(
                f"uniform-1bit ablation is not 256 bit: {score_mode}"
            )
    if KEYPCA_AUTOKEY_SCORE_MODE not in attention._PACKED_QMSE_SCORE_MODES:
        raise AssertionError("without-Query-covariance mode is not registered")
    if attention._packed_qmse_mode_contract(
        KEYPCA_AUTOKEY_SCORE_MODE
    ) != ("key_pca", "key_mse"):
        raise AssertionError("without-Query-covariance contract drifted")
    if attention._packed_qmse_mode_contract(
        QKBALANCED_AUTOKEY_SCORE_MODE
    ) != ("qk_metric", "key_mse"):
        raise AssertionError("Key-MSE-only allocation contract drifted")
    if (
        longbench.QKSIEVE_RANDOM_UNIFORM1_FULLTOPK_SCORE_MODE
        != RANDOM_UNIFORM1_SCORE_MODE
        or longbench.QKSIEVE_KEYPCA_AUTOKEY_FULLTOPK_SCORE_MODE
        != KEYPCA_AUTOKEY_SCORE_MODE
        or longbench.QKSIEVE_QKBALANCED_AUTOKEY_FULLTOPK_SCORE_MODE
        != QKBALANCED_AUTOKEY_SCORE_MODE
    ):
        raise AssertionError("causal-ablation LongBench mapping drifted")

    source = inspect.getsource(attention._packed_qmse_spectral_attention)
    required_fragments = (
        'state.get("packed_qmse_full_topk", False)',
        "variablebit_cuda.scores(",
        "torch.topk(",
        "final_attention_ragged_self",
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise AssertionError(f"frozen attention path lost stages: {missing}")

    config_args = SimpleNamespace(
        countcap_direct_fraction_override=0.0,
    )
    observed_budgets = {}
    for length, expected in (
        (2_000, 256),
        (4_000, 256),
        (8_000, 480),
        (16_000, 960),
        (24_000, 1_280),
        (32_000, 1_280),
        (64_000, 1_280),
        (128_000, 1_280),
    ):
        tokens, fraction = longbench.countcap_direct_budget(length)
        if tokens != expected:
            raise AssertionError(
                f"budget drift at N={length}: expected={expected}, actual={tokens}"
            )
        config = longbench.sparse_method_config(
            METHOD,
            length,
            (0.01,),
            config_args,
        )
        if config["score_mode"] != SCORE_MODE:
            raise AssertionError(f"runtime score mode drift at N={length}")
        if int(config["attention_tokens"]) != expected:
            raise AssertionError(f"runtime attention budget drift at N={length}")
        if float(config["candidate_fraction"]) != fraction:
            raise AssertionError(f"runtime candidate budget drift at N={length}")
        observed_budgets[str(length)] = expected

    return {
        "method": longbench.QKSIEVE_FULLTOPK_METHOD,
        "score_mode": longbench.QKSIEVE_FULLTOPK_SCORE_MODE,
        "required_stages_present": True,
        "observed_budgets": observed_budgets,
        "fier_method": longbench.FIER_RTN1_G32_PACKED_FULLTOPK_METHOD,
        "fier_score_mode": (
            longbench.FIER_RTN1_G32_PACKED_FULLTOPK_SCORE_MODE
        ),
        "controlled_uniform1_score_modes": {
            "key_pca": KEYPCA_UNIFORM1_SCORE_MODE,
            "qk_balanced": QKBALANCED_UNIFORM1_SCORE_MODE,
            "random_rotation": RANDOM_UNIFORM1_SCORE_MODE,
            "physical_bits": 256,
        },
        "without_query_covariance_score_modes": {
            "entire_method": KEYPCA_AUTOKEY_SCORE_MODE,
            "allocation_only": QKBALANCED_AUTOKEY_SCORE_MODE,
        },
    }


def validate_longbench(summary: dict[str, Any], label: str) -> dict[str, Any]:
    validate_contract(summary, label)
    if int(summary.get("strict_pairs", -1)) != 3750:
        raise AssertionError(f"{label}: expected 3,750 strict pairs")
    if int(summary.get("tasks", -1)) != 16:
        raise AssertionError(f"{label}: expected 16 tasks")
    if summary.get("quality_retention_95ci") is None:
        raise AssertionError(f"{label}: paired quality CI is missing")
    return {
        "model_tag": summary.get("model_tag"),
        "quality_retention": summary.get("quality_retention"),
        "quality_retention_95ci": summary.get("quality_retention_95ci"),
        "paired_online_speedup": summary.get("paired_online_speedup"),
    }


def validate_ruler(summary: dict[str, Any]) -> dict[str, Any]:
    validate_contract(summary, "RULER")
    if int(summary.get("strict_pairs", -1)) != 650:
        raise AssertionError("RULER: expected 650 strict pairs")
    if int(summary.get("tasks", -1)) != 13:
        raise AssertionError("RULER: expected all 13 official tasks")
    expected = {4096, 8192, 16384, 32768, 65536, 131072}
    if set(map(int, summary.get("lengths", []))) != expected:
        raise AssertionError("RULER: length grid is incomplete")
    overall = summary.get("overall")
    if not isinstance(overall, dict):
        raise AssertionError("RULER: overall paired summary is missing")
    if overall.get("quality_retention_95ci") is None:
        raise AssertionError("RULER: paired quality CI is missing")
    return {
        "overall": overall,
        "per_length": summary.get("per_length"),
    }


def validate_samepath(summary: dict[str, Any]) -> dict[str, Any]:
    validate_contract(summary, "same-path speed")
    cells = summary.get("cells")
    if not isinstance(cells, dict) or len(cells) != 12:
        raise AssertionError("same-path speed: expected 12 length/horizon cells")
    observed = {
        (int(row["history_tokens"]), int(row["decode_steps"]))
        for row in cells.values()
    }
    expected = {
        (length, horizon)
        for length in EXPECTED_LENGTHS
        for horizon in EXPECTED_HORIZONS
    }
    if observed != expected:
        raise AssertionError("same-path speed: length/horizon grid is incomplete")
    for label, row in cells.items():
        if int(row["qksieve"]["peak_allocated_bytes_total"]) <= 0:
            raise AssertionError(f"same-path speed: peak memory missing for {label}")
        if float(row["qksieve"]["steady_seconds_per_step"]) <= 0:
            raise AssertionError(f"same-path speed: invalid timing for {label}")
        history_tokens = int(row["history_tokens"])
        decode_steps = int(row["decode_steps"])
        expected_counts = [
            min(
                history_tokens + offset,
                1280,
                max(256, math.ceil(0.06 * (history_tokens + offset))),
            )
            for offset in range(max(1, decode_steps - 1))
        ]
        expected_active = sum(expected_counts) / len(expected_counts)
        observed_active = float(
            row["qksieve"]["configured_attention_tokens"]
        )
        if abs(observed_active - expected_active) > 1.0:
            raise AssertionError(
                f"same-path speed: active-token drift for {label}"
            )
        index_ratio = float(row["qksieve"]["index_ratio_of_full_kv"])
        if abs(index_ratio - 240.0 / 4096.0) > 1.0e-6:
            raise AssertionError(
                f"same-path speed: physical index rate drift for {label}"
            )
    breakdown = summary.get("attention_breakdown")
    if not isinstance(breakdown, dict):
        raise AssertionError("same-path speed: attention breakdown is missing")
    breakdown_rows = breakdown.get("rows")
    if not isinstance(breakdown_rows, list) or len(breakdown_rows) != 4:
        raise AssertionError(
            "same-path speed: expected four attention breakdown lengths"
        )
    if {
        int(row["history_tokens"]) for row in breakdown_rows
    } != EXPECTED_LENGTHS:
        raise AssertionError(
            "same-path speed: attention breakdown length grid differs"
        )
    required_breakdown_fields = (
        "fused_query_prepare_ms",
        "packed_scan_ms",
        "torch_topk_ms",
        "explicit_kv_gather_ms",
        "gathered_sdpa_ms",
        "exact_sparse_attention_ms",
        "historical_index_project_encode_ms",
        "per_token_index_project_encode_ms",
        "complete_sparse_path_with_index_append_ms",
        "attention_speedup_including_index_append",
        "full_sdpa_ms",
    )
    for row in breakdown_rows:
        for field in required_breakdown_fields:
            if float(row.get(field, 0.0)) <= 0:
                raise AssertionError(
                    f"same-path speed: {field} missing at "
                    f"N={row.get('history_tokens')}"
                )
    return {"cells": cells, "attention_breakdown": breakdown}


def validate_fier_comparison(
    summary: dict[str, Any],
    contract: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    expected_methods = {
        "full_kv",
        METHOD,
        FIER_METHOD,
    }
    if int(summary.get("strict_pairs", -1)) != 3750:
        raise AssertionError("FIER comparison: expected 3,750 strict pairs")
    if int(summary.get("tasks", -1)) != 16:
        raise AssertionError("FIER comparison: expected 16 tasks")
    methods = summary.get("methods")
    if not isinstance(methods, dict) or set(methods) != expected_methods:
        raise AssertionError("FIER comparison: method set mismatch")
    counts = summary.get("counts")
    if counts != {method: 3750 for method in expected_methods}:
        raise AssertionError("FIER comparison: row counts mismatch")
    for method in (METHOD, FIER_METHOD):
        if methods[method].get("quality_retention_95ci") is None:
            raise AssertionError(
                f"FIER comparison: paired CI missing for {method}"
            )
        if float(methods[method].get("online_seconds", 0.0)) <= 0.0:
            raise AssertionError(
                f"FIER comparison: online timing missing for {method}"
            )

    protocol = contract.get("protocol")
    if not isinstance(protocol, dict):
        raise AssertionError("FIER comparison: protocol contract missing")
    if any(
        bool(protocol.get(field))
        for field in (
            "fallback",
            "rerank",
            "recent_or_sink_reservation",
        )
    ):
        raise AssertionError(
            "FIER comparison: hidden fallback/rerank/reservation enabled"
        )
    if contract.get("shared_budget") != FROZEN_METHOD["budget"]:
        raise AssertionError("FIER comparison: token budget mismatch")
    qksieve = contract.get("qksieve")
    fier = contract.get("fier")
    if not isinstance(qksieve, dict) or not isinstance(fier, dict):
        raise AssertionError("FIER comparison: method contracts missing")
    if qksieve.get("score_mode") != SCORE_MODE:
        raise AssertionError("FIER comparison: QKSieve mode mismatch")
    if int(qksieve.get("index_bits_per_token_per_kv_head", -1)) != 240:
        raise AssertionError("FIER comparison: QKSieve index size mismatch")
    if fier.get("score_mode") != FIER_SCORE_MODE:
        raise AssertionError("FIER comparison: packed FIER mode mismatch")
    if int(fier.get("sequence_group_size", -1)) != 32:
        raise AssertionError("FIER comparison: FIER group size mismatch")
    if int(fier.get("index_bits_per_token_per_kv_head", -1)) != 256:
        raise AssertionError("FIER comparison: FIER index size mismatch")
    if (
        contract.get("shared_final_attention")
        != "qabs_cuda_kernels exact sparse attention"
    ):
        raise AssertionError(
            "FIER comparison: final attention kernel is not shared"
        )
    validate_source_hashes(contract, project_root, "FIER comparison")
    return {
        "strict_pairs": summary["strict_pairs"],
        "qksieve": methods[METHOD],
        "fier": methods[FIER_METHOD],
        "contract": contract,
    }


def validate_query_drift(
    summary: dict[str, Any],
    *,
    expected_kind: str,
) -> dict[str, Any]:
    if summary.get("schema") != "qksieve_query_drift_analysis_v1":
        raise AssertionError("Query drift: schema mismatch")
    protocol = summary.get("protocol")
    coverage = summary.get("coverage")
    counts = summary.get("counts")
    if not all(isinstance(value, dict) for value in (protocol, coverage, counts)):
        raise AssertionError("Query drift: protocol/coverage/counts missing")
    if protocol.get("method") != METHOD:
        raise AssertionError("Query drift: method mismatch")
    if protocol.get("score_mode") != SCORE_MODE:
        raise AssertionError("Query drift: score mode mismatch")
    if int(protocol.get("production_query_samples", -1)) != 8:
        raise AssertionError("Query drift: production Query count differs")
    if int(protocol.get("reserved_physical_index_bits", -1)) != 240:
        raise AssertionError("Query drift: physical index rate differs")
    if not math.isclose(
        float(protocol.get("query_shrinkage", -1.0)),
        0.75,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise AssertionError("Query drift: shrinkage differs")
    if protocol.get("no_rerank_router_recent_sink_or_full_fallback") is not True:
        raise AssertionError("Query drift: hidden special path is possible")
    if any(int(counts.get(field, 0)) <= 0 for field in (
        "per_query_rows",
        "per_head_bucket_rows",
        "allocation_rows",
    )):
        raise AssertionError("Query drift: one or more result tables are empty")

    kinds = set(map(str, coverage.get("trace_kinds", [])))
    if expected_kind == "free_generation":
        if kinds != {"free_generation"}:
            raise AssertionError("Query drift: free-generation trace kind differs")
        if protocol.get("query_sample_counts") != [1, 4, 8]:
            raise AssertionError("Query drift: free-generation sample grid differs")
    elif expected_kind == "teacher_forced_corpus_continuation":
        if kinds != {"teacher_forced_corpus_continuation"}:
            raise AssertionError("Query drift: teacher-forced trace kind differs")
        if protocol.get("query_sample_counts") != [1, 4, 8, 16, 32]:
            raise AssertionError("Query drift: teacher-forced sample grid differs")
        if int(coverage.get("trace_count", -1)) != 6:
            raise AssertionError("Query drift: expected six topic traces")
        if int(coverage.get("max_observed_step", -1)) != 4095:
            raise AssertionError("Query drift: 4K registered step is missing")
        for field in (
            "covers_1k_decode_query",
            "covers_2k_decode_query",
            "covers_4k_decode_query",
        ):
            if coverage.get(field) is not True:
                raise AssertionError(f"Query drift: {field} is false")
    else:
        raise ValueError(f"unsupported Query-drift kind: {expected_kind}")
    if not isinstance(summary.get("trace_sha256"), dict) or not summary[
        "trace_sha256"
    ]:
        raise AssertionError("Query drift: trace hashes are missing")
    return {
        "kind": expected_kind,
        "coverage": coverage,
        "counts": counts,
        "by_query_sample_count": summary.get("by_query_sample_count"),
        "drift_by_query_sample_count_and_position": summary.get(
            "drift_by_query_sample_count_and_position"
        ),
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    report: dict[str, Any] = {
        "schema": "qksieve_frozen_evidence_v1",
        "frozen_method": FROZEN_METHOD,
        "runtime_source": validate_runtime_source(project_root),
        "longbench": [],
    }

    for index, path in enumerate(args.longbench_summary):
        summary = read_json(path)
        label = f"LongBench[{index}]"
        validate_source_hashes(summary, project_root, label)
        report["longbench"].append(validate_longbench(summary, label))

    if args.ruler_summary is not None:
        summary = read_json(args.ruler_summary)
        validate_source_hashes(summary, project_root, "RULER")
        report["ruler"] = validate_ruler(summary)

    if args.samepath_summary is not None:
        summary = read_json(args.samepath_summary)
        validate_source_hashes(summary, project_root, "same-path speed")
        report["samepath"] = validate_samepath(summary)

    if args.multimodel_summary is not None:
        summary = read_json(args.multimodel_summary)
        validate_source_hashes(summary, project_root, "multi-model")
        if summary.get("identical_frozen_method") is not True:
            raise AssertionError("multi-model summary is not configuration-identical")
        if summary.get("frozen_method") != FROZEN_METHOD:
            raise AssertionError("multi-model frozen method mismatch")
        models = summary.get("models")
        expected_models = {"llama31_8b", "qwen3_4b", "mistral_7b"}
        if not isinstance(models, dict) or set(models) != expected_models:
            raise AssertionError("multi-model model set differs")
        expected_wrappers = {
            "llama31_8b": "llama3",
            "qwen3_4b": "qwen3",
            "mistral_7b": "tokenizer_chat",
        }
        for tag, model in models.items():
            if int(model.get("strict_pairs", -1)) != 3750:
                raise AssertionError(f"multi-model: {tag} lacks 3,750 pairs")
            if int(model.get("tasks", -1)) != 16:
                raise AssertionError(f"multi-model: {tag} lacks 16 tasks")
            if model.get("quality_retention_95ci") is None:
                raise AssertionError(f"multi-model: {tag} lacks quality CI")
            if model.get("prompt_wrapper") != expected_wrappers[tag]:
                raise AssertionError(
                    f"multi-model: {tag} prompt wrapper differs"
                )
            identity = model.get("model_identity_sha256")
            if not isinstance(identity, dict) or "config.json" not in identity:
                raise AssertionError(
                    f"multi-model: {tag} identity hashes are missing"
                )
        report["multimodel"] = models

    if (args.fier_summary is None) != (args.fier_contract is None):
        raise AssertionError(
            "FIER summary and method contract must be supplied together"
        )
    if args.fier_summary is not None:
        report["fier"] = validate_fier_comparison(
            read_json(args.fier_summary),
            read_json(args.fier_contract),
            project_root,
        )

    if args.query_drift_free_summary is not None:
        summary = read_json(args.query_drift_free_summary)
        validate_source_hashes(summary, project_root, "Query drift free")
        report["query_drift_free"] = validate_query_drift(
            summary,
            expected_kind="free_generation",
        )

    if args.query_drift_teacher_summary is not None:
        summary = read_json(args.query_drift_teacher_summary)
        validate_source_hashes(summary, project_root, "Query drift teacher")
        report["query_drift_teacher"] = validate_query_drift(
            summary,
            expected_kind="teacher_forced_corpus_continuation",
        )

    if not report["longbench"]:
        report["incomplete"] = ["LongBench summary not supplied"]
    required_sections = {
        "ruler",
        "samepath",
        "multimodel",
        "fier",
        "query_drift_free",
        "query_drift_teacher",
    }
    missing_sections = sorted(required_sections - report.keys())
    if missing_sections:
        report.setdefault("incomplete", []).extend(
            f"{section} summary not supplied" for section in missing_sections
        )
    report["complete"] = "incomplete" not in report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
