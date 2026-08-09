#!/usr/bin/env python
"""Pair Full and QKSieve persistent-KV lifecycle measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_rows(run_root: Path) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
    pairs: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for path in run_root.glob("n*/seed*/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "qksieve_persistent_kv_lifecycle_v1":
            continue
        key = (int(payload["history_tokens"]), int(path.parent.name[4:]))
        pairs.setdefault(key, {})[str(payload["method"])] = payload
    return pairs


def ratio(full: dict[str, Any], sparse: dict[str, Any], field: str) -> float:
    return float(full[field]) / float(sparse[field])


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
                "reuse_tokens_equal": bool(sparse["reuse_tokens_equal"]),
                "index_buffers_reused_without_rebuild": bool(
                    sparse["index_buffers_reused_without_rebuild"]
                ),
            }
        )
    return {
        "schema": "qksieve_persistent_kv_summary_v1",
        "run_root": str(run_root),
        "rows": rows,
        "missing_pairs": missing,
        "all_correct": bool(rows)
        and not missing
        and all(
            row["reuse_tokens_equal"]
            and row["index_buffers_reused_without_rebuild"]
            for row in rows
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
