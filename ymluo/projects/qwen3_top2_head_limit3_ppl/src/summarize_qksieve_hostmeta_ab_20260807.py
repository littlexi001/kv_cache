#!/usr/bin/env python
"""Summarize one-shot host-metadata synchronization A/B runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads((root / "legacy/quality/summary.json").read_text())
    rows = [row for row in payload["rows"] if row["method"] != "full_attention"]
    if len(rows) != 1:
        raise ValueError(f"expected one sparse row, found {len(rows)}")
    return payload, rows[0]


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
        if not (repeat_dir / "baseline/legacy/quality/summary.json").exists():
            continue
        if not (repeat_dir / "hostmeta/legacy/quality/summary.json").exists():
            continue
        baseline_payload, baseline = load(repeat_dir / "baseline")
        host_payload, hostmeta = load(repeat_dir / "hostmeta")
        pairs.append(
            {
                "repeat": repeat_dir.name,
                "baseline": baseline,
                "hostmeta": hostmeta,
                "token_ids_equal": token_ids(baseline_payload)
                == token_ids(host_payload),
            }
        )
    if not pairs:
        raise FileNotFoundError("no completed host-metadata A/B pairs")
    baseline_rows = [pair["baseline"] for pair in pairs]
    host_rows = [pair["hostmeta"] for pair in pairs]
    baseline_fixed = median(baseline_rows, "fixed_sparse_overhead_seconds")
    host_fixed = median(host_rows, "fixed_sparse_overhead_seconds")
    baseline_step = median(baseline_rows, "steady_sparse_seconds_per_step")
    host_step = median(host_rows, "steady_sparse_seconds_per_step")
    result = {
        "schema": "qksieve_host_metadata_realmodel_ab_v1",
        "pairs": len(pairs),
        "all_generated_token_ids_equal": all(
            pair["token_ids_equal"] for pair in pairs
        ),
        "baseline_fixed_s_median": baseline_fixed,
        "hostmeta_fixed_s_median": host_fixed,
        "fixed_speedup": baseline_fixed / host_fixed,
        "fixed_seconds_saved": baseline_fixed - host_fixed,
        "baseline_sparse_ms_median": 1000.0 * baseline_step,
        "hostmeta_sparse_ms_median": 1000.0 * host_step,
        "steady_speed_ratio": baseline_step / host_step,
        "per_pair": [
            {
                "repeat": pair["repeat"],
                "token_ids_equal": pair["token_ids_equal"],
                "baseline_fixed_s": pair["baseline"][
                    "fixed_sparse_overhead_seconds"
                ],
                "hostmeta_fixed_s": pair["hostmeta"][
                    "fixed_sparse_overhead_seconds"
                ],
                "baseline_sparse_ms": 1000.0
                * float(pair["baseline"]["steady_sparse_seconds_per_step"]),
                "hostmeta_sparse_ms": 1000.0
                * float(pair["hostmeta"]["steady_sparse_seconds_per_step"]),
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
