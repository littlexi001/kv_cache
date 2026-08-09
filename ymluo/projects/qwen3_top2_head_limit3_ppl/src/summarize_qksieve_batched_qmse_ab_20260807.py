#!/usr/bin/env python
"""Summarize real-model A/B runs of cross-layer qMSE allocation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def sparse_row(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in payload["rows"] if row["method"] != "full_attention"]
    if len(rows) != 1:
        raise ValueError(f"expected one sparse row, found {len(rows)}")
    return rows[0]


def load(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads((root / "legacy/quality/summary.json").read_text())
    return payload, sparse_row(payload)


def token_ids(payload: dict[str, Any]) -> list[int]:
    variant = payload["requested_variants"][0]
    return [int(row["token_id"]) for row in payload["token_rows"][variant]]


def median(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    pairs = []
    for repeat_dir in sorted(args.run_root.glob("r*")):
        baseline_path = repeat_dir / "baseline/legacy/quality/summary.json"
        batched_path = repeat_dir / "batched/legacy/quality/summary.json"
        if not baseline_path.exists() or not batched_path.exists():
            continue
        baseline_payload, baseline = load(repeat_dir / "baseline")
        batched_payload, batched = load(repeat_dir / "batched")
        pairs.append(
            {
                "repeat": repeat_dir.name,
                "baseline": baseline,
                "batched": batched,
                "token_ids_equal": token_ids(baseline_payload)
                == token_ids(batched_payload),
                "target_hash_equal": baseline_payload["target_token_ids_sha256"]
                == batched_payload["target_token_ids_sha256"],
            }
        )
    if not pairs:
        raise FileNotFoundError("no completed A/B pairs")
    baseline_rows = [pair["baseline"] for pair in pairs]
    batched_rows = [pair["batched"] for pair in pairs]
    baseline_fixed = median(baseline_rows, "fixed_sparse_overhead_seconds")
    batched_fixed = median(batched_rows, "fixed_sparse_overhead_seconds")
    baseline_step = median(baseline_rows, "steady_sparse_seconds_per_step")
    batched_step = median(batched_rows, "steady_sparse_seconds_per_step")
    result = {
        "schema": "qksieve_batched_qmse_realmodel_ab_v1",
        "pairs": len(pairs),
        "all_target_hashes_equal": all(pair["target_hash_equal"] for pair in pairs),
        "all_generated_token_ids_equal": all(pair["token_ids_equal"] for pair in pairs),
        "baseline_fixed_s_median": baseline_fixed,
        "batched_fixed_s_median": batched_fixed,
        "fixed_speedup": baseline_fixed / batched_fixed,
        "fixed_seconds_saved": baseline_fixed - batched_fixed,
        "baseline_sparse_ms_median": 1000.0 * baseline_step,
        "batched_sparse_ms_median": 1000.0 * batched_step,
        "steady_speed_ratio": baseline_step / batched_step,
        "baseline_nll_median": median(baseline_rows, "nll"),
        "batched_nll_median": median(batched_rows, "nll"),
        "per_pair": [
            {
                "repeat": pair["repeat"],
                "token_ids_equal": pair["token_ids_equal"],
                "baseline_fixed_s": pair["baseline"]["fixed_sparse_overhead_seconds"],
                "batched_fixed_s": pair["batched"]["fixed_sparse_overhead_seconds"],
                "baseline_sparse_ms": 1000.0 * float(
                    pair["baseline"]["steady_sparse_seconds_per_step"]
                ),
                "batched_sparse_ms": 1000.0 * float(
                    pair["batched"]["steady_sparse_seconds_per_step"]
                ),
                "batched_qk_prebuild": pair["batched"]["packed_parallel_qk_prebuild"],
            }
            for pair in pairs
        ],
    }
    (args.run_root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
