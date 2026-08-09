from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LAYERS = [25]
HEAD_GROUP = 3
FREQUENCY_PAIR = 46


def relative_atom(alpha: float, start: int, score_blend: float = 1.0) -> dict[str, Any]:
    return {
        "layers": LAYERS,
        "head_groups": [HEAD_GROUP],
        "frequency_pairs": [FREQUENCY_PAIR],
        "frequency_scale": alpha,
        "position_warp_start": start,
        "warp_mode": "relative_distance",
        "score_blend": score_blend,
    }


def make_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "name": "native_rope",
            "stage": "f46_relative_distance",
            "reference_original": True,
            "atoms": [],
        }
    ]
    for start in (8192, 16384):
        for score_blend in (0.25, 0.5, 0.75, 1.0):
            specs.append(
                {
                    "name": (
                        f"l25_g3_f46_relative_s{start}_a0_b{score_blend:g}"
                    ),
                    "stage": "f46_relative_distance",
                    "atoms": [relative_atom(0.0, start, score_blend)],
                }
            )
        specs.append(
            {
                "name": f"l25_g3_f46_relative_s{start}_a0.25_b1",
                "stage": "f46_relative_distance",
                "atoms": [relative_atom(0.25, start, 1.0)],
            }
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"mode": "f46_relative_distance", "specs": make_specs()},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(make_specs())} specs to {args.output}")


if __name__ == "__main__":
    main()
