#!/usr/bin/env python
"""Summarize matched H100 attention, decode, and request measurements."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


LENGTHS = (65536, 131072)
METHODS = ("full", "qksieve_valuesketch_top1280")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected_seeds", type=int, default=3)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def median(values: list[float]) -> float:
    if not values:
        raise AssertionError("cannot aggregate an empty measurement")
    return float(statistics.median(values))


def summarize_attention(run_root: Path, expected_seeds: int) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((run_root / "attention").glob("seed*.json")):
        payload = load_json(path)
        if "H100" not in str(payload.get("gpu", "")):
            raise AssertionError(f"non-H100 attention result: {path}")
        for row in payload["rows"]:
            grouped[int(row["history_count"])].append(row)

    rows: list[dict[str, Any]] = []
    for length in LENGTHS:
        measurements = grouped[length]
        if len(measurements) != expected_seeds:
            raise AssertionError(
                f"attention {length} has {len(measurements)} seeds, "
                f"expected {expected_seeds}"
            )
        full = median([float(row["full_mha_sdpa_ms"]) for row in measurements])
        robust = median(
            [float(row["qksieve_valuesketch_complete_ms"]) for row in measurements]
        )
        fast = median(
            [float(row["qksieve_complete_ms"]) for row in measurements]
        )
        fier = median([float(row["fier_complete_ms"]) for row in measurements])
        rows.append(
            {
                "history_tokens": length,
                "full_mha_ms": full,
                "qksieve_robust_ms": robust,
                "qksieve_fast_ms": fast,
                "fier_ms": fier,
                "robust_speedup": full / robust,
                "fast_speedup": full / fast,
                "fier_speedup": full / fier,
                "robust_vs_fier": fier / robust,
            }
        )
    return rows


def paired_payloads(
    root: Path,
    expected_seeds: int,
    method_names: tuple[str, str],
) -> dict[int, list[tuple[dict[str, Any], dict[str, Any]]]]:
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in sorted(root.glob("n*/seed*/*.json")):
        payload = load_json(path)
        length = int(payload["history_tokens"])
        seed = path.parent.name
        grouped[(length, seed)][str(payload["method"])] = payload

    paired: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for (length, _seed), methods in grouped.items():
        if all(name in methods for name in method_names):
            paired[length].append((methods[method_names[0]], methods[method_names[1]]))
    for length in LENGTHS:
        if len(paired[length]) != expected_seeds:
            raise AssertionError(
                f"{root.name} {length} has {len(paired[length])} pairs, "
                f"expected {expected_seeds}"
            )
    return paired


def summarize_decode(run_root: Path, expected_seeds: int) -> list[dict[str, Any]]:
    paired = paired_payloads(run_root / "decode", expected_seeds, METHODS)
    rows: list[dict[str, Any]] = []
    for length in LENGTHS:
        pairs = paired[length]
        for full, sparse in pairs:
            if "H100" not in str(full.get("gpu_name", "")):
                raise AssertionError("non-H100 decode result")
            if sparse.get("value_sketch_disabled"):
                raise AssertionError("Robust decode disabled ValueSketch")
            if float(sparse.get("value_sketch_tail_alpha", -1.0)) != 0.5:
                raise AssertionError("Robust decode used the wrong tail alpha")
        full_ms = median(
            [float(full["steady_mean_ms_per_token"]) for full, _ in pairs]
        )
        robust_ms = median(
            [float(sparse["steady_mean_ms_per_token"]) for _, sparse in pairs]
        )
        rows.append(
            {
                "history_tokens": length,
                "full_steady_ms_per_token": full_ms,
                "qksieve_steady_ms_per_token": robust_ms,
                "steady_decode_speedup": full_ms / robust_ms,
                "qksieve_prebuild_seconds_median": median(
                    [float(sparse["prebuild_wall_seconds"]) for _, sparse in pairs]
                ),
            }
        )
    return rows


def summarize_persistent(run_root: Path, expected_seeds: int) -> list[dict[str, Any]]:
    paired = paired_payloads(
        run_root / "persistent",
        expected_seeds,
        ("full", "qksieve_robust"),
    )
    metric_fields = {
        "cold_speedup": "cold_persistent_request_ms_per_token",
        "warm_speedup": "shared_prefix_warm_mean_ms_per_token",
        "shared_prefix_amortized_speedup": "shared_prefix_amortized_ms_per_token",
        "append_only_speedup": "append_only_ms_per_token",
    }
    rows: list[dict[str, Any]] = []
    for length in LENGTHS:
        pairs = paired[length]
        for _full, sparse in pairs:
            if not sparse.get("persistent_contract_passed"):
                raise AssertionError("persistent lifecycle contract failed")
        row: dict[str, Any] = {"history_tokens": length}
        for output_name, field in metric_fields.items():
            row[output_name] = median(
                [float(full[field]) / float(sparse[field]) for full, sparse in pairs]
            )
        row["qksieve_index_build_seconds_median"] = median(
            [float(sparse["prebuild_wall_seconds"]) for _, sparse in pairs]
        )
        rows.append(row)
    return rows


def summarize(run_root: Path, expected_seeds: int) -> dict[str, Any]:
    return {
        "schema": "qksieve_h100_matched_system_summary_v1",
        "expected_seeds": expected_seeds,
        "attention": summarize_attention(run_root, expected_seeds),
        "steady_decode": summarize_decode(run_root, expected_seeds),
        "persistent_requests": summarize_persistent(run_root, expected_seeds),
        "claim_boundary": (
            "Matched H100 measurements with resident GPU K/V. Attention is a "
            "single MHA-layer path; decode is whole-model steady generation; "
            "persistent requests separately include per-request index build."
        ),
    }


def main() -> None:
    args = parse_args()
    payload = summarize(args.run_root, args.expected_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
