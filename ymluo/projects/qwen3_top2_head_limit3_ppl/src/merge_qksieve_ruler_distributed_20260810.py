#!/usr/bin/env python
"""Merge primary and tail-accelerator RULER rows under a strict contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
    (args.output_root / "merge_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
