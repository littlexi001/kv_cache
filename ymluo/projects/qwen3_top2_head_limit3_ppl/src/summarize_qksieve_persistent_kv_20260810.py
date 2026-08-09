#!/usr/bin/env python
"""Pair Full and QKSieve persistent-KV lifecycle measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def ratio(full: dict[str, Any], sparse: dict[str, Any], field: str) -> float:
    return float(full[field]) / float(sparse[field])


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


def summarize(run_root: Path) -> dict[str, Any]:
    pairs = load_rows(run_root)
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
    return {
        "schema": "qksieve_persistent_kv_summary_v2",
        "run_root": str(run_root),
        "rows": rows,
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
            "LongBench and RULER suites."
        ),
    }


def main() -> None:
    args = parse_args()
    summary = summarize(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
