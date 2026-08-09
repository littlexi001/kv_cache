#!/usr/bin/env python
"""Compare FIER top-512 unsplit and split-16 exact attention."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median


VARIANT = "qksieve_qmse_requestlocal_fier_rtn1_g32_fulltopk_k512"


def row(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text())
    full = next(item for item in payload["rows"] if item["variant"] == "full_attention")
    sparse = next(item for item in payload["rows"] if item["variant"] == VARIANT)
    return full, sparse


def main() -> None:
    baseline_root = Path(sys.argv[1])
    optimized_root = Path(sys.argv[2])
    split = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    baseline_ms = []
    optimized_ms = []
    baseline_speedups = []
    optimized_speedups = []
    quality = []
    for round_index in (1, 2, 3):
        baseline_path = baseline_root / f"round{round_index}" / VARIANT / "summary.json"
        optimized_path = optimized_root / f"round{round_index}" / "summary.json"
        baseline_full, baseline = row(baseline_path)
        optimized_full, optimized = row(optimized_path)
        baseline_step = float(baseline["steady_sparse_seconds_per_step"])
        optimized_step = float(optimized["steady_sparse_seconds_per_step"])
        baseline_ms.append(1000.0 * baseline_step)
        optimized_ms.append(1000.0 * optimized_step)
        baseline_speedups.append(
            float(baseline_full["steady_sparse_seconds_per_step"]) / baseline_step
        )
        optimized_speedups.append(
            float(optimized_full["steady_sparse_seconds_per_step"]) / optimized_step
        )
        quality.append(float(optimized["quality_retention"]))
    result = {
        "schema": "fier_split_override_realmodel_ab_v1",
        "history_tokens": 65536,
        "budget": 512,
        "split": split,
        "baseline_unsplit_ms_median": median(baseline_ms),
        "optimized_ms_median": median(optimized_ms),
        "sparse_path_speedup": median(baseline_ms) / median(optimized_ms),
        "baseline_decode_speedup_median": median(baseline_speedups),
        "optimized_decode_speedup_median": median(optimized_speedups),
        "quality_retention_median": median(quality),
        "baseline_ms": baseline_ms,
        "optimized_ms": optimized_ms,
    }
    (optimized_root / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
