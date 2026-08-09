from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


LAYERS = tuple(range(18, 36))
HEAD_GROUPS = tuple(range(8))
FREQUENCY_STARTS = tuple(range(0, 64, 8))


def extract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for row in rows:
        spec = row.get("spec") or {}
        if spec.get("stage") != "dense_layer_band":
            continue
        region = spec.get("region") or {}
        layer = int(region["layer"])
        group = int(region["head_group"])
        frequencies = [int(value) for value in region["frequency_pairs"]]
        start = frequencies[0]
        key = (layer, group, start)
        if key in seen:
            raise ValueError(f"duplicate single-layer cell: {key}")
        if layer not in LAYERS or group not in HEAD_GROUPS or start not in FREQUENCY_STARTS:
            raise ValueError(f"cell outside expected grid: {key}")
        if frequencies != list(range(start, start + 8)):
            raise ValueError(f"non-contiguous frequency band in {row['variant']}")
        seen.add(key)
        cells.append(
            {
                "layer": layer,
                "head_group": group,
                "query_heads": f"Q{4 * group}-{4 * group + 3}",
                "frequency_band": f"F{start}-{start + 7}",
                "frequency_start": start,
                "official_score": float(row["official_score_mean"]),
                "official_delta_pp": 100.0 * float(row["paired_official_delta"]),
                "gold_nll_improvement": float(row["mean_nll_improvement"]),
                "gold_ppl": float(row["gold_answer_ppl_from_mean_nll"]),
                "gold_ppl_relative_change_percent": 100.0
                * (math.exp(-float(row["mean_nll_improvement"])) - 1.0),
                "score_improved_samples": int(row["improved_score_samples"]),
                "score_degraded_samples": int(row["degraded_score_samples"]),
            }
        )

    expected = {
        (layer, group, frequency_start)
        for layer in LAYERS
        for group in HEAD_GROUPS
        for frequency_start in FREQUENCY_STARTS
    }
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra:
        raise ValueError(f"single-layer grid mismatch: missing={missing}, extra={extra}")

    cells.sort(key=lambda cell: (cell["layer"], cell["frequency_start"], cell["head_group"]))
    return {
        "metrics": [
            {
                "name": "Gold answer PPL",
                "unit": "PPL",
                "formula": "exp(mean Gold token NLL)",
                "higher_is_better": False,
            },
            {
                "name": "Gold answer PPL relative change",
                "unit": "percent",
                "formula": "100 * (PPL intervention / PPL native - 1)",
                "higher_is_better": False,
            },
            {
                "name": "single-layer official score change",
                "unit": "percentage points",
                "formula": "100 * (official score after one-layer intervention - native RoPE official score)",
                "higher_is_better": True,
            },
        ],
        "setup": {
            "model": "Qwen3-8B",
            "benchmark": "RULER-32K discovery subset",
            "samples_per_configuration": 6,
            "intervention": "remove one 8-pair RoPE frequency band in exactly one layer and one KV head group",
            "layers": list(LAYERS),
            "single_layer_configurations": len(cells),
        },
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    result = extract(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["setup"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
