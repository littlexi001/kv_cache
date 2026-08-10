#!/usr/bin/env python
"""Merge primary and tail-accelerator RULER rows under a strict contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import qksieve_robust_contract_20260810 as contract


EXPECTED_TASKS = (
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
)
EXPECTED_LENGTH_SAMPLES = {
    4096: 10,
    8192: 10,
    16384: 10,
    32768: 10,
    65536: 5,
    131072: 5,
}
METHODS = ("full_kv", contract.METHOD)
NUMERICAL_FREEZE_SHA = "328e01718deebfdfc80dbd8e588a1a95a1832b59"
AUDITED_IMPLEMENTATION_SHA = "f300fb280a597ceb124d454cdfc9a0a1665d6a04"
RUNNER_SHA = "5904eef089a3fc7e56e878c1718fa446bf10d38e080da73a1790c781200b01ad"
CONFIG_SHA = "4712565e231a681bf77da16e2a8e60074c41da3fae6d06f8e3e00d319776fa33"
LONG_DATA_SHA = "b1aaf339afd8d4d8f7246563112356301c416ba7083081ab21e1e7002ab7e7a3"
ROBUST_FIELDS = {
    "executed_path": contract.METHOD,
    "configured_index_bits_per_token": 306.0,
    "packed_qmse_sample_count": 512.0,
    "packed_qmse_value_sketch_rank": 16.0,
    "packed_qmse_value_sketch_bits": 4.0,
    "packed_qmse_value_sketch_executed": 1.0,
    "packed_qmse_value_sketch_tail_alpha": 0.5,
    "packed_qmse_debug_value_sketch_disabled": 0.0,
    "sampled_quantile_fallback": 0.0,
    "configured_score_mode": contract.SCORE_MODE,
}
CONFIG_FIELDS = tuple(ROBUST_FIELDS) + (
    "configured_attention_fraction",
    "configured_attention_tokens",
    "configured_candidate_fraction",
    "configured_projection_dim",
    "diagnostics_enabled",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(root: Path) -> list[Path]:
    paths = sorted(root.glob("shard[0-9]*/sample_results.csv"))
    if not paths:
        raise AssertionError(f"no shard CSVs found under {root}")
    return paths


def _manifest(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    values: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^([0-9a-f]{64})  (.+)$", line)
        if match:
            hashes[Path(match.group(2).replace("\\", "/")).name] = match.group(1)
        elif "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values, hashes


def _runtime(root: Path, role: str) -> dict[str, Any]:
    path = root / "runtime_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "qksieve_runtime_provenance_v1",
        "role": role,
        "audited_implementation_commit_sha": AUDITED_IMPLEMENTATION_SHA,
        "numerical_freeze_commit_sha": NUMERICAL_FREEZE_SHA,
        "visible_cuda_devices": "0,1,2,3,4,5,6,7",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise AssertionError(f"{role} runtime provenance drifted")
    software = payload.get("software", {})
    if not str(software.get("python", "")).startswith("3.10.") or {
        key: software.get(key)
        for key in ("pytorch", "transformers", "numpy", "cuda_runtime", "cudnn")
    } != {
        "pytorch": "2.7.1+cu126",
        "transformers": "4.53.1",
        "numpy": "2.2.6",
        "cuda_runtime": "12.6",
        "cudnn": 90501,
    }:
        raise AssertionError(f"{role} software stack drifted")
    gpus = payload.get("gpus")
    if (
        not isinstance(gpus, list)
        or len(gpus) != 8
        or {int(row.get("index", -1)) for row in gpus} != set(range(8))
        or any(
            row.get("name") != "NVIDIA GeForce RTX 3090"
            or int(row.get("memory_mib", -1)) != 24576
            for row in gpus
        )
    ):
        raise AssertionError(f"{role} GPU inventory drifted")
    if payload.get("run_manifest", {}).get("sha256") != sha256(
        root / "manifest.txt"
    ):
        raise AssertionError(f"{role} run-manifest hash drifted")
    model = payload.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("files"), dict):
        raise AssertionError(f"{role} model hashes are missing")
    return payload


def validate_distributed_protocol(
    primary_root: Path,
    supplement_root: Path,
) -> dict[str, Any]:
    primary_manifest = primary_root / "manifest.txt"
    supplement_manifest = supplement_root / "manifest.txt"
    prompt_path = primary_root / "prompt_length_audit.json"
    for path in (primary_manifest, supplement_manifest, prompt_path):
        if not path.is_file():
            raise AssertionError(f"distributed RULER protocol file is missing: {path}")
    primary_values, primary_hashes = _manifest(primary_manifest)
    supplement_values, supplement_hashes = _manifest(supplement_manifest)
    primary_expected = {
        "schema": "qksieve_robust_ruler_protocol_v1",
        "numerical_freeze_commit_sha": NUMERICAL_FREEZE_SHA,
        "audited_implementation_commit_sha": AUDITED_IMPLEMENTATION_SHA,
        "tasks": ",".join(EXPECTED_TASKS),
        "short_lengths": "4096,8192,16384,32768; samples=10",
        "long_lengths": "65536,131072; samples=5",
        "method": contract.METHOD,
        "max_quantile_samples": "512",
        "value_tail_alpha": "0.5",
    }
    supplement_expected = {
        "schema": "qksieve_ruler_tail_accelerator_protocol_v1",
        "numerical_freeze_commit_sha": NUMERICAL_FREEZE_SHA,
        "audited_implementation_commit_sha": AUDITED_IMPLEMENTATION_SHA,
        "method": contract.METHOD,
        "gpus": "0,1,2,3,4,5,6,7",
        "order": "reverse_of_frozen_jsonl",
    }
    if any(primary_values.get(key) != value for key, value in primary_expected.items()):
        raise AssertionError("primary RULER manifest drifted")
    if any(
        supplement_values.get(key) != value
        for key, value in supplement_expected.items()
    ):
        raise AssertionError("supplement RULER manifest drifted")
    for hashes, role in (
        (primary_hashes, "primary"),
        (supplement_hashes, "supplement"),
    ):
        expected_hashes = {
            "qksieve_robust_iclr2027_frozen_20260810.json": CONFIG_SHA,
            "run_sample_calibrated_ruler_20260717.py": RUNNER_SHA,
            "llama31_8b_ruler13_64k128k_m5_seed42.jsonl": LONG_DATA_SHA,
        }
        if any(hashes.get(name) != digest for name, digest in expected_hashes.items()):
            raise AssertionError(f"{role} frozen input hash drifted")

    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    if (
        prompt.get("schema") != "qksieve_ruler_prompt_length_audit_v1"
        or int(prompt.get("model_max_position_embeddings", -1)) != 131072
        or int(prompt.get("expected_rows", -1)) != 650
        or int(prompt.get("observed_rows", -1)) != 650
        or prompt.get("expected_rows_ok") is not True
        or int(prompt.get("overflow_rows", -1)) != 0
        or prompt.get("all_within_model_limit") is not True
    ):
        raise AssertionError("RULER prompt-length audit failed")

    primary_runtime = _runtime(primary_root, "primary")
    supplement_runtime = _runtime(supplement_root, "supplement")
    primary_model = primary_runtime["model"]
    supplement_model = supplement_runtime["model"]
    if primary_model["files"] != supplement_model["files"]:
        raise AssertionError("distributed RULER model file hashes differ")
    for model, role in (
        (primary_model, "primary"),
        (supplement_model, "supplement"),
    ):
        download = model.get("download_manifest")
        payload = download.get("payload") if isinstance(download, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("model_id") != "LLM-Research/Meta-Llama-3.1-8B-Instruct"
            or int(payload.get("weight_bytes", -1)) != 16060556376
            or payload.get("config_contract")
            != {
                "model_type": "llama",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
            }
        ):
            raise AssertionError(f"{role} model identity drifted")
    model_files_sha = hashlib.sha256(
        json.dumps(primary_model["files"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "qksieve_ruler_distributed_protocol_audit_v1",
        "passed": True,
        "primary_manifest_sha256": sha256(primary_manifest),
        "supplement_manifest_sha256": sha256(supplement_manifest),
        "prompt_length_audit_sha256": sha256(prompt_path),
        "primary_runtime_sha256": sha256(primary_root / "runtime_provenance.json"),
        "supplement_runtime_sha256": sha256(
            supplement_root / "runtime_provenance.json"
        ),
        "model_files_sha256": model_files_sha,
        "primary_driver": primary_runtime["gpus"][0]["driver"],
        "supplement_driver": supplement_runtime["gpus"][0]["driver"],
    }


def read_rows(paths: Iterable[Path]) -> tuple[list[str], list[dict[str, str]]]:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise AssertionError(f"CSV has no header: {path}")
            if header is None:
                header = list(reader.fieldnames)
            elif list(reader.fieldnames) != header:
                raise AssertionError(f"CSV header drifted: {path}")
            for row in reader:
                row["_source_path"] = str(path)
                rows.append(row)
    if header is None or not rows:
        raise AssertionError("distributed RULER input is empty")
    return header, rows


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["task"], row["sample_id"], row["method"]


def _number(value: str, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AssertionError(f"non-finite {label}")
    return result


def validate_robust_row(row: dict[str, str]) -> None:
    if row.get("diagnostics_enabled") != "True":
        raise AssertionError("RULER row lacks attention diagnostics")
    for field, expected in ROBUST_FIELDS.items():
        observed = row.get(field)
        if isinstance(expected, str):
            if observed != expected:
                raise AssertionError(f"Robust {field} drifted")
        elif not math.isclose(
            _number(str(observed), field), expected, rel_tol=0.0, abs_tol=1e-9
        ):
            raise AssertionError(f"Robust {field} drifted")


def validate_full_row(row: dict[str, str]) -> None:
    if row.get("executed_path") != "full_kv":
        raise AssertionError("Full row did not execute full_kv")
    if row.get("configured_score_mode") != "full_kv":
        raise AssertionError("Full score mode drifted")
    if row.get("diagnostics_enabled") != "True":
        raise AssertionError("Full RULER row lacks diagnostics")


def merge_rows(
    primary_rows: list[dict[str, str]],
    supplement_rows: list[dict[str, str]],
    *,
    expected_tasks: tuple[str, ...] = EXPECTED_TASKS,
    expected_length_samples: dict[int, int] = EXPECTED_LENGTH_SAMPLES,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    expected_method_set = set(METHODS)
    duplicate_rows = 0
    duplicate_output_mismatches = 0
    duplicate_output_mismatch_records: list[dict[str, Any]] = []
    duplicate_timing_mismatches = 0
    sources: dict[
        str,
        dict[tuple[str, str], dict[str, dict[str, str]]],
    ] = {}
    for source_name, rows in (
        ("primary", primary_rows),
        ("supplement", supplement_rows),
    ):
        local_seen: set[tuple[str, str, str]] = set()
        grouped: dict[
            tuple[str, str], dict[str, dict[str, str]]
        ] = defaultdict(dict)
        for row in rows:
            key = row_key(row)
            if key in local_seen:
                raise AssertionError(f"duplicate key within {source_name}: {key}")
            local_seen.add(key)
            if row["method"] == contract.METHOD:
                validate_robust_row(row)
            elif row["method"] == "full_kv":
                validate_full_row(row)
            else:
                raise AssertionError(f"unexpected RULER method: {row['method']}")
            grouped[(row["task"], row["sample_id"])][row["method"]] = row
        sources[source_name] = grouped

    primary = sources["primary"]
    supplement = sources["supplement"]
    chosen_pairs: dict[
        tuple[str, str], dict[str, dict[str, str]]
    ] = {}
    primary_pairs_selected = 0
    supplement_pairs_selected = 0
    primary_partial_pairs_discarded = 0
    supplement_partial_pairs_ignored = 0
    incomplete: dict[str, dict[str, list[str]]] = {}
    for sample_key in sorted(set(primary) | set(supplement)):
        primary_pair = primary.get(sample_key, {})
        supplement_pair = supplement.get(sample_key, {})
        primary_complete = set(primary_pair) == expected_method_set
        supplement_complete = set(supplement_pair) == expected_method_set

        for method in sorted(set(primary_pair) & set(supplement_pair)):
            duplicate_rows += 1
            previous = primary_pair[method]
            candidate = supplement_pair[method]
            key = (*sample_key, method)
            if any(
                previous.get(field) != candidate.get(field)
                for field in CONFIG_FIELDS
            ):
                raise AssertionError(f"duplicate configuration drifted: {key}")
            if (
                previous.get("prediction") != candidate.get("prediction")
                or previous.get("score") != candidate.get("score")
            ):
                duplicate_output_mismatches += 1
                duplicate_output_mismatch_records.append(
                    {
                        "task": sample_key[0],
                        "sample_id": sample_key[1],
                        "method": method,
                        "primary_score": previous.get("score"),
                        "supplement_score": candidate.get("score"),
                        "score_mismatch": previous.get("score")
                        != candidate.get("score"),
                        "primary_prediction_sha256": hashlib.sha256(
                            previous.get("prediction", "").encode("utf-8")
                        ).hexdigest(),
                        "supplement_prediction_sha256": hashlib.sha256(
                            candidate.get("prediction", "").encode("utf-8")
                        ).hexdigest(),
                    }
                )
            if any(
                previous.get(field) != candidate.get(field)
                for field in (
                    "prefill_seconds",
                    "query_seconds",
                    "decode_seconds",
                    "online_seconds",
                    "total_seconds",
                )
            ):
                duplicate_timing_mismatches += 1

        if primary_complete:
            chosen_pairs[sample_key] = primary_pair
            primary_pairs_selected += 1
            if supplement_pair and not supplement_complete:
                supplement_partial_pairs_ignored += 1
        elif supplement_complete:
            chosen_pairs[sample_key] = supplement_pair
            supplement_pairs_selected += 1
            if primary_pair:
                primary_partial_pairs_discarded += 1
        else:
            incomplete[f"{sample_key[0]}::{sample_key[1]}"] = {
                "primary": sorted(primary_pair),
                "supplement": sorted(supplement_pair),
            }
    if incomplete:
        raise AssertionError(
            "pair-atomic distributed merge has incomplete pairs: "
            f"{incomplete}"
        )

    cells: Counter[tuple[str, int]] = Counter()
    for task, _sample_id in chosen_pairs:
        base_task, _, length_text = task.rpartition("_")
        if not length_text.isdigit():
            raise AssertionError(f"invalid RULER task: {task}")
        cells[(base_task, int(length_text))] += 1
    expected_cells = {
        (task, length): count
        for task in expected_tasks
        for length, count in expected_length_samples.items()
    }
    if dict(cells) != expected_cells:
        missing = {
            f"{task}@{length}": expected - cells.get((task, length), 0)
            for (task, length), expected in expected_cells.items()
            if cells.get((task, length), 0) != expected
        }
        extra = sorted(set(cells) - set(expected_cells))
        raise AssertionError(f"RULER cell grid drifted: differences={missing}, extra={extra}")

    method_order = {method: index for index, method in enumerate(METHODS)}
    chosen = [
        row
        for pair in chosen_pairs.values()
        for row in pair.values()
    ]
    merged = sorted(
        chosen,
        key=lambda row: (
            int(row["requested_length"]),
            row["base_task"],
            row["sample_id"],
            method_order[row["method"]],
        ),
    )
    for row in merged:
        row.pop("_source_path", None)
    audit = {
        "schema": "qksieve_ruler_distributed_merge_v1",
        "rows": len(merged),
        "strict_pairs": len(chosen_pairs),
        "tasks": len(expected_tasks),
        "lengths": sorted(expected_length_samples),
        "per_length_pairs": {
            str(length): sum(
                cells[(task, length)] for task in expected_tasks
            )
            for length in sorted(expected_length_samples)
        },
        "duplicate_rows_primary_preferred": duplicate_rows,
        "duplicate_output_mismatches": duplicate_output_mismatches,
        "duplicate_output_mismatch_records": duplicate_output_mismatch_records,
        "duplicate_timing_mismatches": duplicate_timing_mismatches,
        "primary_pairs_selected": primary_pairs_selected,
        "supplement_pairs_selected": supplement_pairs_selected,
        "primary_partial_pairs_discarded": primary_partial_pairs_discarded,
        "supplement_partial_pairs_ignored": supplement_partial_pairs_ignored,
        "cross_host_pair_composition_count": 0,
        "claim_boundary": (
            "A strict pair is selected atomically from one host. Complete "
            "primary pairs are authoritative; otherwise a complete supplement "
            "pair is used. Complementary half-pairs are rejected. Duplicate "
            "outputs are audited."
        ),
    }
    return merged, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary_root", required=True, type=Path)
    parser.add_argument("--supplement_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_audit = validate_distributed_protocol(
        args.primary_root,
        args.supplement_root,
    )
    primary_paths = discover(args.primary_root)
    supplement_paths = discover(args.supplement_root)
    primary_header, primary_rows = read_rows(primary_paths)
    supplement_header, supplement_rows = read_rows(supplement_paths)
    if primary_header != supplement_header:
        raise AssertionError("primary and supplement headers differ")
    merged, audit = merge_rows(primary_rows, supplement_rows)
    output = args.output_root / "shard0" / "sample_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=primary_header)
        writer.writeheader()
        writer.writerows(merged)
    audit["sources"] = {
        "primary": {str(path): sha256(path) for path in primary_paths},
        "supplement": {str(path): sha256(path) for path in supplement_paths},
    }
    audit["merged_sha256"] = sha256(output)
    audit["protocol_audit"] = protocol_audit
    (args.output_root / "merge_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
