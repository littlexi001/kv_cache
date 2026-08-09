#!/usr/bin/env python
"""Summarize repeated real-model A/B runs of fused Value-sketch append."""

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


def token_ids(payload: dict[str, Any]) -> list[int]:
    variant = payload["requested_variants"][0]
    return [int(row["token_id"]) for row in payload["token_rows"][variant]]


def load(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, sparse_row(payload)


def median(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    pairs = []
    for repeat_dir in sorted(args.run_root.glob("r*")):
        baseline_path = repeat_dir / "baseline/legacy/quality/summary.json"
        fused_path = repeat_dir / "fused/legacy/quality/summary.json"
        if not baseline_path.exists() or not fused_path.exists():
            continue
        baseline_payload, baseline = load(baseline_path)
        fused_payload, fused = load(fused_path)
        pairs.append(
            {
                "repeat": repeat_dir.name,
                "baseline": baseline,
                "fused": fused,
                "token_ids_equal": token_ids(baseline_payload)
                == token_ids(fused_payload),
                "target_hash_equal": baseline_payload[
                    "target_token_ids_sha256"
                ]
                == fused_payload["target_token_ids_sha256"],
            }
        )
    if not pairs:
        raise FileNotFoundError("no completed A/B pairs")
    baseline_rows = [pair["baseline"] for pair in pairs]
    fused_rows = [pair["fused"] for pair in pairs]
    baseline_step = median(baseline_rows, "steady_sparse_seconds_per_step")
    fused_step = median(fused_rows, "steady_sparse_seconds_per_step")
    baseline_fixed = median(baseline_rows, "fixed_sparse_overhead_seconds")
    fused_fixed = median(fused_rows, "fixed_sparse_overhead_seconds")
    result = {
        "schema": "qksieve_wometric_append_realmodel_ab_v1",
        "pairs": len(pairs),
        "all_target_hashes_equal": all(
            pair["target_hash_equal"] for pair in pairs
        ),
        "all_generated_token_ids_equal": all(
            pair["token_ids_equal"] for pair in pairs
        ),
        "baseline_sparse_ms_median": 1000.0 * baseline_step,
        "fused_sparse_ms_median": 1000.0 * fused_step,
        "sparse_decode_speedup": baseline_step / fused_step,
        "baseline_fixed_s_median": baseline_fixed,
        "fused_fixed_s_median": fused_fixed,
        "fixed_speedup": baseline_fixed / fused_fixed,
        "baseline_nll_median": median(baseline_rows, "nll"),
        "fused_nll_median": median(fused_rows, "nll"),
        "per_pair": [
            {
                "repeat": pair["repeat"],
                "token_ids_equal": pair["token_ids_equal"],
                "baseline_sparse_ms": 1000.0
                * float(pair["baseline"]["steady_sparse_seconds_per_step"]),
                "fused_sparse_ms": 1000.0
                * float(pair["fused"]["steady_sparse_seconds_per_step"]),
                "baseline_nll": pair["baseline"]["nll"],
                "fused_nll": pair["fused"]["nll"],
            }
            for pair in pairs
        ],
    }
    output = args.run_root / "summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
