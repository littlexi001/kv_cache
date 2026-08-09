#!/usr/bin/env python
"""Validate that cache-resident Key factors preserve QKSieve outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def sparse_row(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in payload["rows"] if row["method"] != "full_attention"]
    if len(rows) != 1:
        raise ValueError(f"expected one sparse row, found {len(rows)}")
    return rows[0]


def merged_hashes(payload: dict[str, Any]) -> dict[int, dict[str, str]]:
    merged: dict[int, dict[str, str]] = {}
    for row in payload.get("index_hashes", []):
        layer = int(row["layer"])
        target = merged.setdefault(layer, {})
        for key, value in row.items():
            if key == "layer":
                continue
            previous = target.get(key)
            if previous is not None and previous != value:
                raise ValueError(f"conflicting {key} hashes at layer {layer}")
            target[key] = str(value)
    return merged


def token_signature(payload: dict[str, Any]) -> list[tuple[int, float]]:
    variants = payload["requested_variants"]
    if len(variants) != 1:
        raise ValueError("validation requires one sparse variant")
    rows = payload["token_rows"][variants[0]]
    return [(int(row["token_id"]), float(row["nll"])) for row in rows]


KEY_HASH_FIELDS = {
    "basis",
    "query_basis",
    "packed_qmse_allocation",
    "packed_qmse_key_second_moment",
    "packed_qmse_query_second_moment",
    "packed_index.packed_codes_active",
    "packed_index.key_scales_active",
    "packed_index.score_bias_active",
}


def select_key_hashes(
    payload: dict[int, dict[str, str]],
) -> dict[int, dict[str, str]]:
    return {
        layer: {
            key: value for key, value in row.items() if key in KEY_HASH_FIELDS
        }
        for layer, row in payload.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("resident_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_summary = json.loads(
        (args.baseline_root / "legacy/quality/summary.json").read_text()
    )
    resident_summary = json.loads(
        (args.resident_root / "legacy/quality/summary.json").read_text()
    )
    baseline_profile = json.loads(
        (args.baseline_root / "legacy/index_profile.json").read_text()
    )
    resident_profile = json.loads(
        (args.resident_root / "legacy/index_profile.json").read_text()
    )
    baseline_row = sparse_row(baseline_summary)
    resident_row = sparse_row(resident_summary)
    baseline_hashes = merged_hashes(baseline_profile)
    resident_hashes = merged_hashes(resident_profile)
    key_hash_equal = select_key_hashes(baseline_hashes) == select_key_hashes(
        resident_hashes
    )
    baseline_tokens = token_signature(baseline_summary)
    resident_tokens = token_signature(resident_summary)
    token_ids_equal = [item[0] for item in baseline_tokens] == [
        item[0] for item in resident_tokens
    ]
    max_nll_difference = max(
        (
            abs(baseline[1] - resident[1])
            for baseline, resident in zip(baseline_tokens, resident_tokens)
        ),
        default=0.0,
    )
    result = {
        "schema": "qksieve_resident_key_ab_validation_v1",
        "history_tokens": baseline_summary["history_tokens"],
        "eval_tokens": baseline_summary["eval_tokens"],
        "all_profile_hashes_equal": baseline_hashes == resident_hashes,
        "active_key_index_hashes_equal": key_hash_equal,
        "hashed_layers_baseline": len(baseline_hashes),
        "hashed_layers_resident": len(resident_hashes),
        "generated_token_ids_equal": token_ids_equal,
        "maximum_token_nll_difference": max_nll_difference,
        "baseline_nll": baseline_row["nll"],
        "resident_nll": resident_row["nll"],
        "baseline_fixed_s": baseline_row["fixed_sparse_overhead_seconds"],
        "resident_fixed_s": resident_row["fixed_sparse_overhead_seconds"],
        "baseline_qk_prebuild": baseline_row["packed_parallel_qk_prebuild"],
        "resident_qk_prebuild": resident_row["packed_parallel_qk_prebuild"],
        "resident_key_build": resident_summary.get(
            "resident_key_factor_precompute", {}
        ),
    }
    result["warm_fixed_speedup"] = (
        float(result["baseline_fixed_s"])
        / float(result["resident_fixed_s"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not key_hash_equal or not token_ids_equal:
        raise SystemExit("resident Key factors changed the numerical path")


if __name__ == "__main__":
    main()
