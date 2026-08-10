#!/usr/bin/env python
"""Reject incomplete or contract-drifting QKSieve-Robust paper evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import qksieve_robust_contract_20260810 as contract


RULER_LENGTH_SAMPLES = {
    "4096": 10,
    "8192": 10,
    "16384": 10,
    "32768": 10,
    "65536": 5,
    "131072": 5,
}
RULER_LENGTHS = set(RULER_LENGTH_SAMPLES)
RULER_TASKS = {
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_squad",
    "qa_hotpot",
}
SYSTEM_LENGTHS = {65536, 131072}
MODELS = {"llama31_8b", "qwen3_4b", "mistral_7b"}
SHRINKAGE_LABELS = {
    "qwen3_4b_sports32k",
    "qwen3_4b_medicine32k",
    "llama31_8b_sports32k",
    "llama31_8b_medicine32k",
}
SHRINKAGES = {0.0, 0.25, 0.5, 0.75, 0.9}
SHRINKAGE_FRACTIONS = {0.01, 0.02, 0.04}
NUMERICAL_FREEZE_SHA = "328e01718deebfdfc80dbd8e588a1a95a1832b59"
AUDITED_IMPLEMENTATION_SHA = "f300fb280a597ceb124d454cdfc9a0a1665d6a04"
SOURCE_MANIFEST = "qksieve_robust_source_manifest_20260810.json"


def _finite_number(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AssertionError(f"{label} is not finite")
    return result


def _validate_interval(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise AssertionError(f"{label} is missing or malformed")
    lower = _finite_number(value[0], label)
    upper = _finite_number(value[1], label)
    if lower > upper:
        raise AssertionError(f"{label} has reversed bounds")
    return [lower, upper]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True, type=Path)
    parser.add_argument("--persistent_summary", type=Path)
    parser.add_argument("--longbench_summary", type=Path)
    parser.add_argument("--ruler_summary", type=Path)
    parser.add_argument("--multimodel_summary", type=Path)
    parser.add_argument("--shrinkage_summary", type=Path)
    parser.add_argument("--shrinkage_equivalence", type=Path)
    parser.add_argument("--h100_summary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def validate_frozen_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "configs" / "qksieve_robust_iclr2027_frozen_20260810.json"
    payload = read_json(path)
    numerical = payload["numerical_contract"]
    value_tail = numerical["value_tail"]
    if payload.get("numerical_freeze_commit_sha") != NUMERICAL_FREEZE_SHA:
        raise AssertionError("numerical freeze commit is missing or drifted")
    if (
        payload.get("audited_implementation_commit_sha")
        != AUDITED_IMPLEMENTATION_SHA
    ):
        raise AssertionError("audited implementation commit is missing or drifted")
    if payload["score_mode"] != contract.SCORE_MODE:
        raise AssertionError("frozen score mode drifted")
    if numerical["quantile_samples_max"] != contract.MAX_QUANTILE_SAMPLE_COUNT:
        raise AssertionError("frozen quantile sample cap drifted")
    if (value_tail["rank"], value_tail["bits"], value_tail["tail_alpha"]) != (
        contract.VALUE_SKETCH_RANK,
        contract.VALUE_SKETCH_BITS,
        contract.VALUE_SKETCH_TAIL_ALPHA,
    ):
        raise AssertionError("frozen ValueSketch contract drifted")
    forbidden = (
        "router",
        "task_rule",
        "full_attention_fallback",
        "sink_or_recent_reservation",
    )
    if any(bool(numerical[name]) for name in forbidden):
        raise AssertionError("frozen method enabled a forbidden special path")
    return payload


def validate_frozen_sources(project_root: Path) -> dict[str, Any]:
    manifest_path = project_root / "configs" / SOURCE_MANIFEST
    payload = read_json(manifest_path)
    if payload.get("schema") != "qksieve_frozen_source_manifest_v1":
        raise AssertionError("frozen source manifest schema mismatch")
    if payload.get("audited_implementation_commit_sha") != AUDITED_IMPLEMENTATION_SHA:
        raise AssertionError("frozen source manifest implementation SHA drifted")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise AssertionError("frozen source manifest is empty")

    root = project_root.resolve()
    observed: dict[str, str] = {}
    for relative, expected in sorted(files.items()):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise AssertionError(f"unsafe frozen source path: {relative}")
        path = (root / relative_path).resolve()
        if root not in path.parents or not path.is_file():
            raise AssertionError(f"frozen source is missing: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise AssertionError(
                f"frozen source drifted: {relative}; expected {expected}, observed {digest}"
            )
        observed[relative] = digest
    return {
        "manifest": str(manifest_path),
        "recorded_from_commit": payload.get("recorded_from_commit"),
        "files": observed,
    }


def validate_persistent(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "qksieve_persistent_kv_summary_v2":
        raise AssertionError("persistent summary schema mismatch")
    if payload.get("all_correct") is not True:
        raise AssertionError("persistent lifecycle audit failed")
    if payload.get("missing_pairs") not in (None, []):
        raise AssertionError("persistent summary has missing Full/Robust pairs")
    rows = payload.get("rows")
    expected_lengths = {32768, 65536}
    expected_seeds = {20260810, 20260811, 20260812}
    if not isinstance(rows, list) or len(rows) != 6:
        raise AssertionError("persistent summary requires six paired process runs")
    for history_tokens in expected_lengths:
        source = [
            row for row in rows if int(row["history_tokens"]) == history_tokens
        ]
        if len(source) != 3 or {int(row["seed"]) for row in source} != expected_seeds:
            raise AssertionError(
                "persistent summary requires three fixed process repetitions "
                f"at {history_tokens} tokens"
            )
    for row in rows:
        for field in (
            "warm_speedup",
            "cold_speedup",
            "cold_end_to_end_speedup",
            "amortized_speedup",
            "append_only_speedup",
        ):
            if not math.isfinite(float(row[field])) or float(row[field]) <= 0.0:
                raise AssertionError(f"invalid persistent metric: {field}")
        if not all(
            bool(row[field])
            for field in (
                "reuse_tokens_equal",
                "index_buffers_reused_without_rebuild",
                "rewind_value_layers_correct",
                "persistent_contract_passed",
            )
        ):
            raise AssertionError("persistent row failed a lifecycle check")
        audit = row.get("independent_lifecycle_audit")
        if not isinstance(audit, dict):
            raise AssertionError("persistent row lacks an independent audit")
        expected_audit = {
            "layers": 32,
            "snapshots": 6,
            "rewinds": 5,
            "post_decode_index_lag_tokens": 1,
            "all_index_buffers_stable": True,
            "deterministic_replay": True,
        }
        if any(audit.get(name) != value for name, value in expected_audit.items()):
            raise AssertionError("persistent independent lifecycle audit drifted")

    aggregate_rows = payload.get("aggregate_rows")
    if (
        not isinstance(aggregate_rows, list)
        or len(aggregate_rows) != 2
        or {int(row["history_tokens"]) for row in aggregate_rows}
        != expected_lengths
    ):
        raise AssertionError("persistent summary lacks two aggregate rows")
    aggregate_metrics = (
        "full_warm_ms_per_token",
        "qksieve_warm_ms_per_token",
        "full_cold_end_to_end_ms_per_token",
        "qksieve_cold_end_to_end_ms_per_token",
        "warm_speedup",
        "cold_speedup",
        "cold_end_to_end_speedup",
        "amortized_speedup",
        "append_only_speedup",
        "qksieve_prebuild_seconds",
    )
    for row in aggregate_rows:
        if int(row.get("seed_count", -1)) != 3 or {
            int(seed) for seed in row.get("seeds", [])
        } != expected_seeds:
            raise AssertionError("persistent aggregate seed coverage drifted")
        for field in aggregate_metrics:
            value = float(row[field])
            low = float(row[f"{field}_bootstrap_ci95_low"])
            high = float(row[f"{field}_bootstrap_ci95_high"])
            if not all(math.isfinite(item) and item > 0.0 for item in (value, low, high)):
                raise AssertionError(f"invalid persistent aggregate: {field}")
            if not low <= value <= high:
                raise AssertionError(f"persistent aggregate interval misses: {field}")
    statistics_payload = payload.get("statistics")
    if not isinstance(statistics_payload, dict) or statistics_payload.get(
        "point_estimate"
    ) != "median_across_independent_process_repetitions":
        raise AssertionError("persistent process-repetition statistics drifted")
    return {
        "rows": rows,
        "aggregate_rows": aggregate_rows,
        "claim_boundary": payload.get("claim_boundary"),
    }


def validate_longbench(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "qksieve_robust_longbench_summary_v1":
        raise AssertionError("LongBench summary schema mismatch")
    if payload.get("frozen_contract") != contract.contract_payload():
        raise AssertionError("LongBench frozen contract drifted")
    if int(payload.get("strict_pairs", -1)) != 3750:
        raise AssertionError("LongBench does not contain 3,750 strict pairs")
    if int(payload.get("rows", -1)) != 7500:
        raise AssertionError("LongBench does not contain 7,500 rows")
    if int(payload.get("tasks", -1)) != 16:
        raise AssertionError("LongBench does not contain 16 tasks")
    if int(payload.get("full_fallback_count", -1)) != 0:
        raise AssertionError("LongBench observed a Full fallback")
    if float(payload.get("value_sketch_tail_alpha", -1.0)) != float(
        contract.VALUE_SKETCH_TAIL_ALPHA
    ):
        raise AssertionError("LongBench ValueSketch shrinkage drifted")
    sample_count_audit = payload.get("sample_count_audit")
    if not isinstance(sample_count_audit, dict):
        raise AssertionError("LongBench lacks the decode-mean sample-count audit")
    if sample_count_audit.get("schema") != (
        "qksieve_decode_mean_sample_count_v1"
    ):
        raise AssertionError("LongBench sample-count audit schema drifted")
    if int(sample_count_audit.get("rows", -1)) != 3750:
        raise AssertionError("LongBench sample-count audit is incomplete")
    if _finite_number(
        sample_count_audit.get("max_abs_error"),
        "LongBench sample-count audit error",
    ) > 1e-6:
        raise AssertionError("LongBench sample-count audit failed")
    summarizer_sha = payload.get("summarizer_sha256")
    if not isinstance(summarizer_sha, str) or len(summarizer_sha) != 64:
        raise AssertionError("LongBench summarizer SHA256 is missing")
    methods = payload.get("methods")
    if not isinstance(methods, dict) or set(methods) != {
        "full_kv",
        contract.METHOD,
    }:
        raise AssertionError("LongBench method set drifted")
    per_task = payload.get("per_task")
    if not isinstance(per_task, dict) or len(per_task) != 16:
        raise AssertionError("LongBench per-task table is incomplete")
    if sum(int(row.get("samples", 0)) for row in per_task.values()) != 3750:
        raise AssertionError("LongBench per-task sample counts do not sum to 3,750")
    bootstrap = payload.get("bootstrap", {})
    if int(bootstrap.get("resamples", 0)) < 10000:
        raise AssertionError("LongBench bootstrap has fewer than 10,000 resamples")
    _validate_interval(
        bootstrap.get("macro_score_delta_95ci"),
        "LongBench macro-score interval",
    )
    _validate_interval(
        bootstrap.get("quality_retention_95ci"),
        "LongBench retention interval",
    )
    if _finite_number(methods["full_kv"].get("macro_score"), "Full macro") <= 0:
        raise AssertionError("LongBench Full macro score is not positive")
    if _finite_number(
        methods[contract.METHOD].get("quality_retention"),
        "LongBench retention",
    ) <= 0:
        raise AssertionError("LongBench quality retention is not positive")
    return {
        "methods": methods,
        "per_task": per_task,
        "bootstrap": bootstrap,
        "sample_count_audit": sample_count_audit,
    }


def validate_ruler(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "qksieve_robust_ruler_summary_v1":
        raise AssertionError("RULER summary schema mismatch")
    if payload.get("frozen_contract") != contract.contract_payload():
        raise AssertionError("RULER frozen contract drifted")
    if int(payload.get("strict_pairs", -1)) != 650:
        raise AssertionError("RULER does not contain 650 strict pairs")
    if int(payload.get("rows", -1)) != 1300:
        raise AssertionError("RULER does not contain 1,300 rows")
    if set(payload.get("tasks", [])) != RULER_TASKS:
        raise AssertionError("RULER task set differs from the formal 13 tasks")
    length_samples = {
        str(length): int(samples)
        for length, samples in payload.get("length_samples", {}).items()
    }
    if length_samples != RULER_LENGTH_SAMPLES:
        raise AssertionError("RULER per-length sample counts drifted")
    if set(payload.get("per_length", {})) != RULER_LENGTHS:
        raise AssertionError("RULER length grid is incomplete")
    per_task_length = payload.get("per_task_length")
    if not isinstance(per_task_length, dict) or len(per_task_length) != 78:
        raise AssertionError("RULER task-length table is incomplete")
    observed_cells: set[tuple[str, str]] = set()
    observed_samples = 0
    for row in per_task_length.values():
        task = str(row.get("task"))
        length = str(int(row.get("length", 0)))
        if task not in RULER_TASKS or length not in RULER_LENGTH_SAMPLES:
            raise AssertionError("RULER task-length cell is outside the protocol")
        cell = (task, length)
        if cell in observed_cells:
            raise AssertionError("RULER task-length cell is duplicated")
        observed_cells.add(cell)
        samples = int(row.get("samples", -1))
        if samples != RULER_LENGTH_SAMPLES[length]:
            raise AssertionError("RULER task-length cell sample count drifted")
        cell_bootstrap = row.get("bootstrap", {})
        if int(cell_bootstrap.get("resamples", 0)) < 10000:
            raise AssertionError("RULER cell bootstrap is incomplete")
        _validate_interval(
            cell_bootstrap.get("macro_score_delta_95ci"),
            "RULER cell macro-score interval",
        )
        full_cell = row.get("full_kv")
        if not isinstance(full_cell, dict):
            raise AssertionError("RULER cell lacks Full metrics")
        full_score = _finite_number(
            full_cell.get("score"),
            "RULER cell Full score",
        )
        if full_score > 0.0:
            _validate_interval(
                cell_bootstrap.get("quality_retention_95ci"),
                "RULER cell retention interval",
            )
        observed_samples += samples
    if observed_samples != 650:
        raise AssertionError("RULER task-length samples do not sum to 650")
    for length, row in payload["per_length"].items():
        if int(row.get("cells", -1)) != len(RULER_TASKS):
            raise AssertionError(f"RULER {length} does not aggregate 13 tasks")
        length_bootstrap = row.get("bootstrap", {})
        if int(length_bootstrap.get("resamples", 0)) < 10000:
            raise AssertionError(f"RULER {length} bootstrap is incomplete")
        _validate_interval(
            length_bootstrap.get("macro_score_delta_95ci"),
            f"RULER {length} macro-score interval",
        )
        _validate_interval(
            length_bootstrap.get("quality_retention_95ci"),
            f"RULER {length} retention interval",
        )
    if int(payload.get("overall", {}).get("cells", -1)) != 78:
        raise AssertionError("RULER overall aggregate does not contain 78 cells")
    bootstrap = payload.get("bootstrap", {})
    if int(bootstrap.get("resamples", 0)) < 10000:
        raise AssertionError("RULER bootstrap has fewer than 10,000 resamples")
    _validate_interval(
        bootstrap.get("macro_score_delta_95ci"),
        "RULER macro-score interval",
    )
    _validate_interval(
        bootstrap.get("quality_retention_95ci"),
        "RULER retention interval",
    )
    if int(payload.get("fallback_count", -1)) != 0:
        raise AssertionError("RULER observed a Full fallback")
    return {
        "overall": payload["overall"],
        "per_length": payload["per_length"],
        "bootstrap": bootstrap,
    }


def validate_multimodel(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "qksieve_robust_multimodel_summary_v1":
        raise AssertionError("multi-model summary schema mismatch")
    if payload.get("frozen_contract") != contract.contract_payload():
        raise AssertionError("multi-model frozen contract drifted")
    models = payload.get("models")
    if not isinstance(models, dict) or set(models) != MODELS:
        raise AssertionError("multi-model evidence lacks Llama/Qwen/Mistral")
    for tag, row in models.items():
        if int(row.get("strict_pairs", -1)) != 160:
            raise AssertionError(f"{tag} does not contain 160 strict pairs")
        if int(row.get("tasks", -1)) != 16:
            raise AssertionError(f"{tag} does not contain 16 tasks")
        if int(row.get("full_fallback_count", -1)) != 0:
            raise AssertionError(f"{tag} observed a Full fallback")
        _validate_interval(
            row.get("quality_retention_95ci"),
            f"{tag} retention interval",
        )
        if _finite_number(row.get("quality_retention"), f"{tag} retention") <= 0:
            raise AssertionError(f"{tag} quality retention is not positive")
        fraction = _finite_number(
            row.get("mean_attention_fraction"),
            f"{tag} attention fraction",
        )
        if not 0.0 < fraction <= 1.0:
            raise AssertionError(f"{tag} attention fraction is invalid")
        per_task = row.get("per_task")
        if not isinstance(per_task, dict) or len(per_task) != 16:
            raise AssertionError(f"{tag} per-task table is incomplete")
        if sum(int(item.get("samples", 0)) for item in per_task.values()) != 160:
            raise AssertionError(f"{tag} per-task samples do not sum to 160")
    return models


def validate_shrinkage(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "qksieve_shrinkage_sensitivity_v1":
        raise AssertionError("shrinkage sensitivity schema mismatch")
    if payload.get("complete") is not True:
        raise AssertionError("shrinkage sensitivity result is incomplete")
    if payload.get("method") != "qk_balanced":
        raise AssertionError("shrinkage sensitivity method drifted")
    if payload.get("calibration_source") != "prefill_tail":
        raise AssertionError("shrinkage sensitivity did not use prompt calibration")
    if not math.isclose(float(payload.get("production_shrinkage", -1.0)), 0.75):
        raise AssertionError("production shrinkage drifted")
    if set(payload.get("labels", [])) != SHRINKAGE_LABELS:
        raise AssertionError("shrinkage sensitivity trace grid drifted")
    if {float(value) for value in payload.get("shrinkages", [])} != SHRINKAGES:
        raise AssertionError("shrinkage sensitivity lambda grid drifted")
    if {
        float(value) for value in payload.get("selected_fractions", [])
    } != SHRINKAGE_FRACTIONS:
        raise AssertionError("shrinkage sensitivity sparsity grid drifted")
    paired_conditions = int(payload.get("strict_paired_conditions", 0))
    if paired_conditions <= 0:
        raise AssertionError("shrinkage sensitivity has no strict pairs")
    if int(payload.get("bootstrap_samples", 0)) < 10000:
        raise AssertionError("shrinkage sensitivity bootstrap is incomplete")

    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, list) or len(aggregate) != 15:
        raise AssertionError("shrinkage sensitivity aggregate grid is incomplete")
    observed: set[tuple[float, float]] = set()
    required_metrics = (
        "top2_recall",
        "selected_attention_mass",
        "top2_attention_mass_recall",
        "score_pearson",
        "score_rmse",
    )
    for row in aggregate:
        key = (float(row["shrinkage"]), float(row["selected_fraction"]))
        if key in observed:
            raise AssertionError("shrinkage sensitivity aggregate cell is duplicated")
        observed.add(key)
        if int(row.get("conditions", -1)) <= 0:
            raise AssertionError("shrinkage sensitivity cell has no conditions")
        for metric in required_metrics:
            _finite_number(row.get(metric), f"shrinkage {metric}")
            paired = row.get(f"{metric}_paired")
            if not isinstance(paired, dict) or int(paired.get("clusters", 0)) <= 0:
                raise AssertionError(f"shrinkage {metric} lacks paired clusters")
            _finite_number(
                paired.get("delta_vs_production"),
                f"shrinkage {metric} paired delta",
            )
            _validate_interval(paired.get("ci95"), f"shrinkage {metric} interval")
    expected = {
        (shrinkage, fraction)
        for shrinkage in SHRINKAGES
        for fraction in SHRINKAGE_FRACTIONS
    }
    if observed != expected:
        raise AssertionError("shrinkage sensitivity aggregate cells drifted")

    per_label = payload.get("per_label")
    if not isinstance(per_label, dict) or set(per_label) != SHRINKAGE_LABELS:
        raise AssertionError("shrinkage sensitivity per-trace grid is incomplete")
    if any(
        not isinstance(rows, list) or len(rows) != 15
        for rows in per_label.values()
    ):
        raise AssertionError("shrinkage sensitivity per-trace cells are incomplete")
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict) or not isinstance(
        acceptance.get("passed"), bool
    ):
        raise AssertionError("shrinkage sensitivity acceptance result is missing")
    if len(acceptance.get("checks", [])) != 3:
        raise AssertionError("shrinkage sensitivity acceptance checks are incomplete")
    if not isinstance(payload.get("source_sha256"), dict) or not payload[
        "source_sha256"
    ]:
        raise AssertionError("shrinkage sensitivity source hashes are missing")
    claim_boundary = payload.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise AssertionError("shrinkage sensitivity claim boundary is missing")
    return {
        "production_shrinkage": payload["production_shrinkage"],
        "strict_paired_conditions": paired_conditions,
        "aggregate": aggregate,
        "per_label": per_label,
        "acceptance": acceptance,
        "claim_boundary": claim_boundary,
    }


def validate_shrinkage_equivalence(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "qksieve_shrinkage_fast_grid_equivalence_v1":
        raise AssertionError("shrinkage equivalence schema mismatch")
    if payload.get("passed") is not True:
        raise AssertionError("shrinkage fast-grid equivalence failed")
    if int(payload.get("conditions", -1)) != 30720:
        raise AssertionError("shrinkage equivalence condition count drifted")
    if int(payload.get("allocation_conditions", -1)) != 40:
        raise AssertionError("shrinkage equivalence allocation count drifted")
    if payload.get("condition_keys_identical") is not True:
        raise AssertionError("shrinkage equivalence keys differ")
    if payload.get("allocations_identical") is not True:
        raise AssertionError("shrinkage equivalence allocations differ")
    metrics = payload.get("metrics")
    expected_metrics = {
        "top2_recall",
        "selected_attention_mass",
        "oracle_top2_attention_mass",
        "top2_attention_mass_recall",
        "score_pearson",
        "score_rmse",
    }
    if not isinstance(metrics, dict) or set(metrics) != expected_metrics:
        raise AssertionError("shrinkage equivalence metric grid drifted")
    for name, row in metrics.items():
        maximum = _finite_number(
            row.get("max_abs_difference"), f"{name} maximum difference"
        )
        average = _finite_number(
            row.get("mean_abs_difference"), f"{name} mean difference"
        )
        maximum_limit = _finite_number(
            row.get("max_tolerance"), f"{name} maximum tolerance"
        )
        mean_limit = _finite_number(
            row.get("mean_tolerance"), f"{name} mean tolerance"
        )
        if maximum > maximum_limit or average > mean_limit:
            raise AssertionError(f"shrinkage equivalence {name} exceeds tolerance")
    source_hashes = payload.get("source_sha256")
    if not isinstance(source_hashes, dict) or len(source_hashes) != 4:
        raise AssertionError("shrinkage equivalence source hashes are incomplete")
    for label, digest in source_hashes.items():
        if not isinstance(label, str) or not label.strip():
            raise AssertionError("shrinkage equivalence source label is invalid")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise AssertionError("shrinkage equivalence source hash is invalid")
    claim_boundary = payload.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise AssertionError("shrinkage equivalence claim boundary is missing")
    return {
        "conditions": payload["conditions"],
        "allocation_conditions": payload["allocation_conditions"],
        "metrics": metrics,
        "claim_boundary": claim_boundary,
    }


def _positive_finite(row: dict[str, Any], field: str, label: str) -> None:
    if field not in row:
        raise AssertionError(f"H100 {label} is missing {field}")
    value = float(row[field])
    if not math.isfinite(value) or value <= 0.0:
        raise AssertionError(f"H100 {label} has invalid {field}: {value}")


def _validate_h100_rows(
    rows: Any,
    label: str,
    required_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(SYSTEM_LENGTHS):
        raise AssertionError(f"H100 {label} lacks 64K/128K")
    if {int(row["history_tokens"]) for row in rows} != SYSTEM_LENGTHS:
        raise AssertionError(f"H100 {label} lacks 64K/128K")
    for row in rows:
        for field in required_fields:
            _positive_finite(row, field, label)
    return rows


def validate_h100(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "qksieve_h100_matched_system_summary_v1":
        raise AssertionError("H100 summary schema mismatch")
    if payload.get("frozen_contract") != contract.contract_payload():
        raise AssertionError("H100 frozen contract drifted")
    if int(payload.get("expected_seeds", 0)) < 3:
        raise AssertionError("H100 result requires at least three seeds")
    hardware = payload.get("hardware")
    names = hardware.get("device_names") if isinstance(hardware, dict) else None
    if not isinstance(names, list) or not names:
        raise AssertionError("H100 hardware identity is missing")
    if any("H100" not in str(name) for name in names):
        raise AssertionError("H100 evidence contains a non-H100 device")
    software = hardware.get("software")
    required_software = ("python", "pytorch", "transformers", "cuda_runtime", "cudnn")
    if not isinstance(software, dict) or any(
        software.get(field) in (None, "") for field in required_software
    ):
        raise AssertionError("H100 software environment is incomplete")
    methods = payload.get("methods")
    if not isinstance(methods, dict) or methods.get("main") != contract.METHOD:
        raise AssertionError("H100 main method identity drifted")
    claim_boundary = payload.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise AssertionError("H100 claim boundary is missing")
    return {
        "hardware": hardware,
        "attention": _validate_h100_rows(
            payload.get("attention"),
            "attention",
            (
                "full_mha_ms",
                "qksieve_robust_ms",
                "qksieve_fast_ms",
                "fier_ms",
                "robust_speedup",
                "fast_speedup",
                "fier_speedup",
                "robust_vs_fier",
                "full_kv_bytes",
                "qksieve_key_index_bytes",
                "qksieve_valuesketch_bytes",
                "qksieve_total_auxiliary_bytes",
                "qksieve_key_index_ratio_of_full_kv",
                "qksieve_total_auxiliary_ratio_of_full_kv",
                "fier_index_bytes",
                "fier_index_ratio_of_full_kv",
            ),
        ),
        "steady_decode": _validate_h100_rows(
            payload.get("steady_decode"),
            "decode",
            (
                "full_steady_ms_per_token",
                "qksieve_steady_ms_per_token",
                "steady_decode_speedup",
                "qksieve_prebuild_seconds_median",
                "full_peak_allocated_bytes_total",
                "qksieve_peak_allocated_bytes_total",
                "qksieve_to_full_peak_allocated_ratio",
                "full_peak_reserved_bytes_total",
                "qksieve_peak_reserved_bytes_total",
                "qksieve_to_full_peak_reserved_ratio",
            ),
        ),
        "persistent_requests": _validate_h100_rows(
            payload.get("persistent_requests"),
            "requests",
            (
                "cold_speedup",
                "cold_end_to_end_speedup",
                "warm_speedup",
                "shared_prefix_amortized_speedup",
                "append_only_speedup",
                "qksieve_index_build_seconds_median",
                "full_cold_peak_allocated_bytes_total",
                "qksieve_cold_peak_allocated_bytes_total",
                "qksieve_to_full_cold_peak_allocated_ratio",
                "full_cold_peak_reserved_bytes_total",
                "qksieve_cold_peak_reserved_bytes_total",
                "qksieve_to_full_cold_peak_reserved_ratio",
                "full_lifecycle_peak_allocated_bytes_total",
                "qksieve_lifecycle_peak_allocated_bytes_total",
                "qksieve_to_full_lifecycle_peak_allocated_ratio",
                "full_lifecycle_peak_reserved_bytes_total",
                "qksieve_lifecycle_peak_reserved_bytes_total",
                "qksieve_to_full_lifecycle_peak_reserved_ratio",
            ),
        ),
        "claim_boundary": claim_boundary,
    }


def verify(
    project_root: Path,
    *,
    persistent: dict[str, Any] | None = None,
    longbench: dict[str, Any] | None = None,
    ruler: dict[str, Any] | None = None,
    multimodel: dict[str, Any] | None = None,
    shrinkage: dict[str, Any] | None = None,
    shrinkage_equivalence: dict[str, Any] | None = None,
    h100: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "qksieve_robust_paper_evidence_audit_v1",
        "frozen_config": validate_frozen_config(project_root),
        "frozen_sources": validate_frozen_sources(project_root),
    }
    validators = {
        "persistent": (persistent, validate_persistent),
        "longbench": (longbench, validate_longbench),
        "ruler": (ruler, validate_ruler),
        "multimodel": (multimodel, validate_multimodel),
        "shrinkage": (shrinkage, validate_shrinkage),
        "shrinkage_equivalence": (
            shrinkage_equivalence,
            validate_shrinkage_equivalence,
        ),
        "h100": (h100, validate_h100),
    }
    missing: list[str] = []
    for name, (payload, validator) in validators.items():
        if payload is None:
            missing.append(name)
        else:
            report[name] = validator(payload)
    report["missing"] = missing
    report["complete"] = not missing
    return report


def main() -> None:
    args = parse_args()
    optional_paths = {
        "persistent": args.persistent_summary,
        "longbench": args.longbench_summary,
        "ruler": args.ruler_summary,
        "multimodel": args.multimodel_summary,
        "shrinkage": args.shrinkage_summary,
        "shrinkage_equivalence": args.shrinkage_equivalence,
        "h100": args.h100_summary,
    }
    report = verify(
        args.project_root,
        **{
            name: read_json(path) if path is not None else None
            for name, path in optional_paths.items()
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
