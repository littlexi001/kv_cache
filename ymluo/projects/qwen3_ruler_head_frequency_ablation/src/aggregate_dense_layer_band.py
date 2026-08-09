from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


LAYER_BLOCKS = ((18, 23), (24, 29), (30, 35))
FREQUENCY_STARTS = tuple(range(0, 64, 8))
HEAD_GROUPS = tuple(range(8))


def _block_for_layer(layer: int) -> tuple[int, int]:
    for start, end in LAYER_BLOCKS:
        if start <= layer <= end:
            return start, end
    raise ValueError(f"layer {layer} is outside the expected deep-layer range")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    native_rows = [row for row in rows if row.get("variant") == "native_rope"]

    for row in rows:
        spec = row.get("spec") or {}
        if spec.get("stage") != "dense_layer_band":
            continue
        region = spec.get("region") or {}
        layer = int(region["layer"])
        head_group = int(region["head_group"])
        frequencies = [int(value) for value in region["frequency_pairs"]]
        if len(frequencies) != 8 or frequencies != list(
            range(frequencies[0], frequencies[0] + 8)
        ):
            raise ValueError(f"unexpected frequency band in {row['variant']}: {frequencies}")
        frequency_start = frequencies[0]
        block_start, block_end = _block_for_layer(layer)
        grouped[(block_start, block_end, head_group, frequency_start)].append(row)

    expected_keys = {
        (block_start, block_end, head_group, frequency_start)
        for block_start, block_end in LAYER_BLOCKS
        for head_group in HEAD_GROUPS
        for frequency_start in FREQUENCY_STARTS
    }
    missing = sorted(expected_keys - grouped.keys())
    extra = sorted(grouped.keys() - expected_keys)
    if missing or extra:
        raise ValueError(f"grid mismatch: missing={missing}, extra={extra}")

    cells: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        block_start, block_end, head_group, frequency_start = key
        values = grouped[key]
        layers = sorted(int(row["spec"]["region"]["layer"]) for row in values)
        expected_layers = list(range(block_start, block_end + 1))
        if layers != expected_layers:
            raise ValueError(
                f"cell {key} has layers {layers}; expected {expected_layers}"
            )

        deltas = [100.0 * float(row["paired_official_delta"]) for row in values]
        nll_improvements = [float(row["mean_nll_improvement"]) for row in values]
        best = max(values, key=lambda row: float(row["paired_official_delta"]))
        worst = min(values, key=lambda row: float(row["paired_official_delta"]))
        mean_delta = sum(deltas) / len(deltas)
        variance = sum((value - mean_delta) ** 2 for value in deltas) / len(deltas)

        cells.append(
            {
                "layer_block": f"L{block_start}-{block_end}",
                "layer_start": block_start,
                "layer_end": block_end,
                "head_group": head_group,
                "query_heads": f"Q{4 * head_group}-{4 * head_group + 3}",
                "frequency_band": f"F{frequency_start}-{frequency_start + 7}",
                "frequency_start": frequency_start,
                "configuration_count": len(values),
                "mean_official_delta_pp": mean_delta,
                "std_official_delta_pp": math.sqrt(variance),
                "best_layer": int(best["spec"]["region"]["layer"]),
                "best_official_delta_pp": 100.0
                * float(best["paired_official_delta"]),
                "worst_layer": int(worst["spec"]["region"]["layer"]),
                "worst_official_delta_pp": 100.0
                * float(worst["paired_official_delta"]),
                "mean_gold_nll_improvement": sum(nll_improvements)
                / len(nll_improvements),
            }
        )

    return {
        "metric": {
            "name": "mean official score change",
            "unit": "percentage points",
            "formula": "mean over the six single-layer ablations in each layer block of 100 * (intervention score - native RoPE score)",
            "higher_is_better": True,
        },
        "setup": {
            "model": "Qwen3-8B",
            "benchmark": "RULER-32K discovery subset",
            "samples_per_configuration": 6,
            "layer_blocks": [f"L{start}-{end}" for start, end in LAYER_BLOCKS],
            "head_groups": list(HEAD_GROUPS),
            "frequency_bands": [
                f"F{start}-{start + 7}" for start in FREQUENCY_STARTS
            ],
            "intervention": "set the selected RoPE frequency pairs to identity rotation in one layer and one KV head group",
            "native_baseline_rows": len(native_rows),
            "single_layer_configurations": sum(
                cell["configuration_count"] for cell in cells
            ),
            "aggregated_cells": len(cells),
        },
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    result = aggregate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["setup"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
