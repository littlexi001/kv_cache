#!/usr/bin/env python
"""Reject incomplete or contract-drifting QKSieve-Robust paper evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import qksieve_robust_contract_20260810 as contract


RULER_LENGTHS = {"4096", "8192", "16384", "32768", "65536", "131072"}
SYSTEM_LENGTHS = {65536, 131072}
MODELS = {"llama31_8b", "qwen3_4b", "mistral_7b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True, type=Path)
    parser.add_argument("--persistent_summary", type=Path)
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
    return {"rows": rows, "claim_boundary": payload.get("claim_boundary")}


def validate_ruler(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "qksieve_robust_ruler_summary_v1":
        raise AssertionError("RULER summary schema mismatch")
    if payload.get("frozen_contract") != contract.contract_payload():
        raise AssertionError("RULER frozen contract drifted")
    if int(payload.get("strict_pairs", -1)) != 650:
        raise AssertionError("RULER does not contain 650 strict pairs")
    if len(payload.get("tasks", [])) != 13:
        raise AssertionError("RULER does not contain 13 tasks")
    if set(payload.get("per_length", {})) != RULER_LENGTHS:
        raise AssertionError("RULER length grid is incomplete")
    bootstrap = payload.get("bootstrap", {})
    if not all(
        field in bootstrap
        for field in ("macro_score_delta_95ci", "quality_retention_95ci")
    ):
        raise AssertionError("RULER bootstrap intervals are missing")
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
        if row.get("quality_retention_95ci") is None:
            raise AssertionError(f"{tag} confidence interval is missing")
    return models


def _validate_h100_rows(rows: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or {
        int(row["history_tokens"]) for row in rows
    } != SYSTEM_LENGTHS:
        raise AssertionError(f"H100 {label} lacks 64K/128K")
    return rows


def validate_h100(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "qksieve_h100_matched_system_summary_v1":
        raise AssertionError("H100 summary schema mismatch")
    if int(payload.get("expected_seeds", 0)) < 3:
        raise AssertionError("H100 result requires at least three seeds")
    return {
        "attention": _validate_h100_rows(payload.get("attention"), "attention"),
        "steady_decode": _validate_h100_rows(
            payload.get("steady_decode"), "decode"
        ),
        "persistent_requests": _validate_h100_rows(
            payload.get("persistent_requests"), "requests"
        ),
        "claim_boundary": payload.get("claim_boundary"),
    }


def verify(
    project_root: Path,
    *,
    persistent: dict[str, Any] | None = None,
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
