#!/usr/bin/env python
"""Reject incomplete or contract-drifting QKSieve-Robust paper evidence."""

from __future__ import annotations

import argparse
import json
import math
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
NUMERICAL_FREEZE_SHA = "328e01718deebfdfc80dbd8e588a1a95a1832b59"
AUDITED_IMPLEMENTATION_SHA = "f300fb280a597ceb124d454cdfc9a0a1665d6a04"


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


def validate_persistent(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "qksieve_persistent_kv_summary_v2":
        raise AssertionError("persistent summary schema mismatch")
    if payload.get("all_correct") is not True:
        raise AssertionError("persistent lifecycle audit failed")
    rows = payload.get("rows")
    if not isinstance(rows, list) or {int(row["history_tokens"]) for row in rows} != {
        32768,
        65536,
    }:
        raise AssertionError("persistent summary lacks 32K/64K")
    for row in rows:
        for field in (
            "warm_speedup",
            "cold_speedup",
            "amortized_speedup",
            "append_only_speedup",
        ):
            if float(row[field]) <= 0.0:
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
    return {"rows": rows, "claim_boundary": payload.get("claim_boundary")}


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
    h100: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "qksieve_robust_paper_evidence_audit_v1",
        "frozen_config": validate_frozen_config(project_root),
    }
    validators = {
        "persistent": (persistent, validate_persistent),
        "longbench": (longbench, validate_longbench),
        "ruler": (ruler, validate_ruler),
        "multimodel": (multimodel, validate_multimodel),
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
