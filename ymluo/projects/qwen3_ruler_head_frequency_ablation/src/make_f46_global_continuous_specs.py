from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LAYERS = [25]
HEAD_GROUP = 3
FREQUENCY_PAIR = 46


def base_atom() -> dict[str, Any]:
    return {
        "layers": LAYERS,
        "head_groups": [HEAD_GROUP],
        "frequency_pairs": [FREQUENCY_PAIR],
    }


def make_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "name": "native_rope",
            "stage": "f46_global_continuous",
            "reference_original": True,
            "atoms": [],
        }
    ]
    for blend in (0.1, 0.2, 0.3, 0.4, 0.5, 0.75):
        specs.append(
            {
                "name": f"l25_g3_f46_nope_score_blend_b{blend:g}",
                "stage": "f46_global_continuous",
                "atoms": [
                    base_atom()
                    | {
                        "frequency_scale": 0.0,
                        "position_warp_start": 0,
                        "warp_mode": "relative_distance",
                        "score_blend": blend,
                    }
                ],
            }
        )
    for alpha in (0.125, 0.25):
        specs.append(
            {
                "name": f"l25_g3_f46_absolute_phase_a{alpha:g}",
                "stage": "f46_global_continuous",
                "atoms": [base_atom() | {"frequency_scale": alpha}],
            }
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    specs = make_specs()
    args.output.write_text(
        json.dumps(
            {"mode": "f46_global_continuous", "specs": specs},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(specs)} specs to {args.output}")


if __name__ == "__main__":
    main()
