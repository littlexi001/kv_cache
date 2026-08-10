#!/usr/bin/env python
"""Pair Full and QKSieve persistent-KV lifecycle measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any

import qksieve_robust_contract_20260810 as contract


POINTER_FIELDS = (
    "key_code_ptr",
    "key_scale_ptr",
    "value_code_ptr",
    "value_minimum_ptr",
    "value_scale_ptr",
)
PRELOADED_EXTENSIONS = {
    "variablebit",
    "query",
    "mixedblock",
    "value_attention",
}
AGGREGATE_FIELDS = (
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
    "qksieve_first_step_ms",
)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260810
AUDITED_IMPLEMENTATION_SHA = "f300fb280a597ceb124d454cdfc9a0a1665d6a04"
EXPECTED_SEEDS = {20260810, 20260811, 20260812}
EXPECTED_LENGTHS = {32768: 2, 65536: 3}
EXPECTED_SOFTWARE = {
    "python": "3.10.20",
    "pytorch": "2.7.1+cu126",
    "transformers": "4.53.1",
    "cuda_runtime": "12.6",
    "cudnn": 90501,
}
EXPECTED_MODEL_HASHES = {
    "pytorch_model-00001-of-00002.bin": (
        "e15eff64c7ef2159ecd7228424d4d3ba813e9bcda2f6cb543accbe5028bd0ae0"
    ),
    "pytorch_model-00002-of-00002.bin": (
        "0f85245cab4358e94a5cadce299ddb16964a22d86eece081caa5e05616f3828a"
    ),
    "pytorch_model.bin.index.json": (
        "e572e08c4d4e81c7916197f6fcd2956a2f05e5919f28d72c9ba4f351efae1e29"
    ),
    "config.json": (
        "bf8239b8842439a1149effb9af58e5eba5db867d414abaf4c071b3ba48a6a215"
    ),
    "tokenizer.model": (
        "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"
    ),
}
EXPECTED_SOURCE_MANIFEST_SHA = (
    "f5922f9802af5d6925692503e2d3cf1f1566563da1e14dcb76188ead796be79a"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_rows(run_root: Path) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
    pairs: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for path in run_root.glob("n*/seed*/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "qksieve_persistent_kv_lifecycle_v2":
            continue
        key = (int(payload["history_tokens"]), int(path.parent.name[4:]))
        pairs.setdefault(key, {})[str(payload["method"])] = payload
    return pairs


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_run_protocol(
    run_root: Path,
    pairs: dict[tuple[int, int], dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    manifest = run_root / "manifest.txt"
    if not manifest.is_file():
        raise AssertionError("persistent multirun manifest is missing")
    text = manifest.read_text(encoding="utf-8")
    values = {
        key: value
        for key, value in (
            line.split("=", 1)
            for line in text.splitlines()
            if "=" in line and not re.match(r"^[0-9a-f]{64}  ", line)
        )
    }
    expected_values = {
        "schema": "qksieve_persistent_multiseed_protocol_v1",
        "seeds": "20260810,20260811,20260812",
        "audited_implementation_commit_sha": AUDITED_IMPLEMENTATION_SHA,
        "python": EXPECTED_SOFTWARE["python"],
        "torch": EXPECTED_SOFTWARE["pytorch"],
        "torch_cuda": EXPECTED_SOFTWARE["cuda_runtime"],
        "transformers": EXPECTED_SOFTWARE["transformers"],
    }
    for field, expected in expected_values.items():
        if values.get(field) != expected:
            raise AssertionError(f"persistent manifest {field} drifted")

    file_hashes: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^([0-9a-f]{64})  (.+)$", line)
        if match:
            file_hashes[Path(match.group(2).replace("\\", "/")).name] = match.group(1)
    if file_hashes.get("qksieve_robust_source_manifest_20260810.json") != (
        EXPECTED_SOURCE_MANIFEST_SHA
    ):
        raise AssertionError("persistent frozen-source manifest hash drifted")
    for name, expected in EXPECTED_MODEL_HASHES.items():
        if file_hashes.get(name) != expected:
            raise AssertionError(f"persistent model file hash drifted: {name}")

    gpu_rows = [
        line for line in text.splitlines() if "NVIDIA GeForce RTX 3090" in line
    ]
    if len(gpu_rows) != 8 or any("555.42.02" not in row for row in gpu_rows):
        raise AssertionError("persistent RTX 3090 hardware manifest drifted")
    expected_pair_keys = {
        (history_tokens, seed)
        for history_tokens in EXPECTED_LENGTHS
        for seed in EXPECTED_SEEDS
    }
    if set(pairs) != expected_pair_keys:
        raise AssertionError("persistent multirun length/seed grid drifted")

    for (history_tokens, _seed), methods in pairs.items():
        if set(methods) != {"full", "qksieve_robust"}:
            raise AssertionError("persistent Full/Robust method pair drifted")
        full = methods["full"]
        sparse = methods["qksieve_robust"]
        expected_device_count = EXPECTED_LENGTHS[history_tokens]
        for payload in (full, sparse):
            if payload.get("gpu_name") != "NVIDIA GeForce RTX 3090":
                raise AssertionError("persistent result GPU drifted")
            if payload.get("software") != EXPECTED_SOFTWARE:
                raise AssertionError("persistent result software stack drifted")
            if not str(payload.get("model_name_or_path", "")).replace(
                "\\", "/"
            ).endswith("/Yarn-Llama-2-7b-128k"):
                raise AssertionError("persistent model identity drifted")
            devices = [
                item
                for item in str(payload.get("visible_cuda_devices", "")).split(",")
                if item
            ]
            if len(devices) != expected_device_count:
                raise AssertionError("persistent model-sharding width drifted")
            if (
                int(payload.get("history_tokens", -1)) != history_tokens
                or int(payload.get("branch_count", -1)) != 4
                or int(payload.get("branch_steps", -1)) != 32
                or int(payload.get("append_steps", -1)) != 128
            ):
                raise AssertionError("persistent workload protocol drifted")
            if float(payload.get("cold_end_to_end_request_ms_per_token", 0.0)) <= 0:
                raise AssertionError("persistent cold-E2E timing is missing")
        if full.get("budget_policy") != "full_history":
            raise AssertionError("persistent Full budget policy drifted")
        if sparse.get("score_mode") != contract.SCORE_MODE:
            raise AssertionError("persistent Robust score mode drifted")
        if sparse.get("budget_policy") != (
            "min(1280, max(256, ceil(0.06 * history_tokens)))"
        ):
            raise AssertionError("persistent Robust budget policy drifted")
        if float(sparse.get("value_sketch_tail_alpha", -1.0)) != (
            contract.VALUE_SKETCH_TAIL_ALPHA
        ):
            raise AssertionError("persistent ValueSketch alpha drifted")
    return {
        "schema": "qksieve_persistent_run_protocol_audit_v1",
        "passed": True,
        "manifest_sha256": sha256(manifest),
        "audited_implementation_commit_sha": AUDITED_IMPLEMENTATION_SHA,
        "lengths": sorted(EXPECTED_LENGTHS),
        "seeds": sorted(EXPECTED_SEEDS),
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "driver": "555.42.02",
        "software": EXPECTED_SOFTWARE,
        "model_hashes": EXPECTED_MODEL_HASHES,
    }


def ratio(full: dict[str, Any], sparse: dict[str, Any], field: str) -> float:
    return float(full[field]) / float(sparse[field])


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty list")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_median_interval(
    values: list[float],
    *,
    seed: int,
) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    samples = [
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return percentile(samples, 0.025), percentile(samples, 0.975)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_history: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_history.setdefault(int(row["history_tokens"]), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for history_tokens, source in sorted(by_history.items()):
        source = sorted(source, key=lambda row: int(row["seed"]))
        aggregate: dict[str, Any] = {
            "history_tokens": history_tokens,
            "seed_count": len(source),
            "seeds": [int(row["seed"]) for row in source],
        }
        for field_index, field in enumerate(AGGREGATE_FIELDS):
            values = [float(row[field]) for row in source]
            low, high = bootstrap_median_interval(
                values,
                seed=BOOTSTRAP_SEED + history_tokens + field_index,
            )
            aggregate[field] = statistics.median(values)
            aggregate[f"{field}_min"] = min(values)
            aggregate[f"{field}_max"] = max(values)
            aggregate[f"{field}_bootstrap_ci95_low"] = low
            aggregate[f"{field}_bootstrap_ci95_high"] = high
        aggregates.append(aggregate)
    return aggregates


def audit_snapshot(
    snapshot: dict[str, Any],
    *,
    expected_layers: int,
    expected_indexed_count: int,
) -> tuple[tuple[int | None, ...], ...]:
    layers = snapshot.get("layers")
    if not isinstance(layers, list) or len(layers) != expected_layers:
        raise AssertionError("persistent snapshot layer count differs")
    if int(snapshot.get("layer_count", -1)) != expected_layers:
        raise AssertionError("persistent snapshot layer_count differs")
    if {int(layer["layer"]) for layer in layers} != set(range(expected_layers)):
        raise AssertionError("persistent snapshot layer IDs differ")
    signature: list[tuple[int | None, ...]] = []
    for layer in sorted(layers, key=lambda item: int(item["layer"])):
        if int(layer["key_indexed_count"]) != expected_indexed_count:
            raise AssertionError("persistent Key index length differs")
        if int(layer["value_indexed_count"]) != expected_indexed_count:
            raise AssertionError("persistent Value index length differs")
        if any(layer.get(field) is None for field in POINTER_FIELDS):
            raise AssertionError("persistent snapshot lacks an index pointer")
        signature.append(
            (
                int(layer["key_rebuild_count"]),
                int(layer["value_rebuild_count"]),
                *(int(layer[field]) for field in POINTER_FIELDS),
            )
        )
    return tuple(signature)


def audit_sparse_lifecycle(sparse: dict[str, Any]) -> dict[str, Any]:
    """Recompute lifecycle invariants without trusting result booleans."""
    if sparse.get("schema") != "qksieve_persistent_kv_lifecycle_v2":
        raise AssertionError("persistent sparse schema differs")
    history = int(sparse["history_tokens"])
    branch_count = int(sparse["branch_count"])
    branch_steps = int(sparse["branch_steps"])
    append_steps = int(sparse["append_steps"])
    qk_prebuild = sparse.get("qk_prebuild", {})
    key_prebuild = sparse.get("key_index_prebuild", {})
    value_prebuild = sparse.get("value_prebuild", {})
    value_install = sparse.get("value_install", {})
    expected_layers = int(qk_prebuild.get("layers", 0))
    if expected_layers <= 0:
        raise AssertionError("QK prebuild did not cover any layer")
    if (
        int(key_prebuild.get("layers", 0))
        + int(key_prebuild.get("existing_layers", 0))
        != expected_layers
        or int(value_prebuild.get("layers", 0)) != expected_layers
        or int(value_install.get("layers", 0)) != expected_layers
    ):
        raise AssertionError("Key/Value prebuild layer coverage differs")
    preload = sparse.get("runtime_extension_preload")
    if not isinstance(preload, dict) or not PRELOADED_EXTENSIONS.issubset(preload):
        raise AssertionError("runtime CUDA extensions were not preloaded")
    if any(float(preload[name]) < 0.0 for name in PRELOADED_EXTENSIONS):
        raise AssertionError("runtime extension preload timing is invalid")
    if int(sparse.get("post_decode_index_lag_tokens", -1)) != 1:
        raise AssertionError("post-decode index lag is not exactly one token")

    signatures = [
        audit_snapshot(
            sparse["initial_persistent_state_snapshot"],
            expected_layers=expected_layers,
            expected_indexed_count=history,
        )
    ]
    snapshots = sparse.get("persistent_state_snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != branch_count + 2:
        raise AssertionError("persistent snapshot count differs")
    for snapshot in snapshots[: branch_count + 1]:
        signatures.append(
            audit_snapshot(
                snapshot,
                expected_layers=expected_layers,
                expected_indexed_count=history + branch_steps - 1,
            )
        )
    signatures.append(
        audit_snapshot(
            snapshots[-1],
            expected_layers=expected_layers,
            expected_indexed_count=history + append_steps - 1,
        )
    )
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise AssertionError("persistent index pointers or rebuild counts changed")

    rewinds = sparse.get("rewinds")
    if not isinstance(rewinds, list) or len(rewinds) != branch_count + 1:
        raise AssertionError("persistent rewind count differs")
    for rewind in rewinds:
        if (
            int(rewind["active_length"]) != history
            or int(rewind["key_layers"]) != expected_layers
            or int(rewind["value_layers"]) != expected_layers
        ):
            raise AssertionError("persistent rewind did not cover every layer")

    branches = sparse.get("branches")
    if not isinstance(branches, list) or len(branches) != branch_count + 1:
        raise AssertionError("persistent branch count differs")
    if (
        branches[0]["generated_token_ids"]
        != branches[-1]["generated_token_ids"]
        or branches[0]["generated_token_sha256"]
        != branches[-1]["generated_token_sha256"]
    ):
        raise AssertionError("rewound branch replay is not deterministic")
    if not all(
        bool(sparse.get(field))
        for field in (
            "reuse_tokens_equal",
            "reuse_hash_equal",
            "index_buffers_reused_without_rebuild",
            "rewind_value_layers_correct",
            "persistent_contract_passed",
        )
    ):
        raise AssertionError("runtime lifecycle flag reports failure")
    return {
        "layers": expected_layers,
        "snapshots": len(snapshots),
        "rewinds": len(rewinds),
        "post_decode_index_lag_tokens": 1,
        "all_index_buffers_stable": True,
        "deterministic_replay": True,
    }


def summarize(
    run_root: Path,
    *,
    require_protocol: bool = False,
) -> dict[str, Any]:
    pairs = load_rows(run_root)
    protocol_audit = audit_run_protocol(run_root, pairs) if require_protocol else None
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for (history_tokens, seed), methods in sorted(pairs.items()):
        full = methods.get("full")
        sparse = methods.get("qksieve_robust")
        if full is None or sparse is None:
            missing.append(
                {
                    "history_tokens": history_tokens,
                    "seed": seed,
                    "methods": sorted(methods),
                }
            )
            continue
        lifecycle_audit = audit_sparse_lifecycle(sparse)
        rows.append(
            {
                "history_tokens": history_tokens,
                "seed": seed,
                "full_warm_ms_per_token": full[
                    "shared_prefix_warm_mean_ms_per_token"
                ],
                "qksieve_warm_ms_per_token": sparse[
                    "shared_prefix_warm_mean_ms_per_token"
                ],
                "warm_speedup": ratio(
                    full,
                    sparse,
                    "shared_prefix_warm_mean_ms_per_token",
                ),
                "cold_speedup": ratio(
                    full,
                    sparse,
                    "cold_persistent_request_ms_per_token",
                ),
                "full_cold_end_to_end_ms_per_token": full[
                    "cold_end_to_end_request_ms_per_token"
                ],
                "qksieve_cold_end_to_end_ms_per_token": sparse[
                    "cold_end_to_end_request_ms_per_token"
                ],
                "cold_end_to_end_speedup": ratio(
                    full,
                    sparse,
                    "cold_end_to_end_request_ms_per_token",
                ),
                "amortized_speedup": ratio(
                    full,
                    sparse,
                    "shared_prefix_amortized_ms_per_token",
                ),
                "append_only_speedup": ratio(
                    full,
                    sparse,
                    "append_only_ms_per_token",
                ),
                "qksieve_prebuild_seconds": sparse["prebuild_wall_seconds"],
                "qksieve_first_step_ms": sparse["branches"][0][
                    "first_step_ms"
                ],
                "reuse_tokens_equal": bool(sparse["reuse_tokens_equal"]),
                "index_buffers_reused_without_rebuild": bool(
                    sparse["index_buffers_reused_without_rebuild"]
                ),
                "rewind_value_layers_correct": bool(
                    sparse["rewind_value_layers_correct"]
                ),
                "persistent_contract_passed": bool(
                    sparse["persistent_contract_passed"]
                ),
                "independent_lifecycle_audit": lifecycle_audit,
            }
        )
    aggregates = aggregate_rows(rows)
    return {
        "schema": "qksieve_persistent_kv_summary_v2",
        "run_root": str(run_root),
        "rows": rows,
        "aggregate_rows": aggregates,
        "statistics": {
            "point_estimate": "median_across_independent_process_repetitions",
            "interval": "process_repetition_bootstrap_median_percentile_95",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "missing_pairs": missing,
        "all_correct": bool(rows)
        and not missing
        and all(
            row["reuse_tokens_equal"]
            and row["index_buffers_reused_without_rebuild"]
            and row["rewind_value_layers_correct"]
            and row["persistent_contract_passed"]
            and row["independent_lifecycle_audit"][
                "all_index_buffers_stable"
            ]
            for row in rows
        ),
        "claim_boundary": (
            "Token equality checks deterministic rewind/replay within each "
            "method. Quality relative to Full is measured by the separate "
            "LongBench and RULER suites. Timing intervals resample independent "
            "process repetitions of one fixed deterministic workload; they do "
            "not represent workload-distribution or cross-hardware variance."
        ),
        "protocol_audit": protocol_audit,
    }


def main() -> None:
    args = parse_args()
    summary = summarize(args.run_root, require_protocol=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
