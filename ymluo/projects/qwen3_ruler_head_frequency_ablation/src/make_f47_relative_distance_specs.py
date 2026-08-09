from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LAYERS = list(range(18, 24))
HEAD_GROUP = 4
FREQUENCY_PAIR = 47


def relative_atom(alpha: float, start: int) -> dict[str, Any]:
    return {
        "layers": LAYERS,
        "head_groups": [HEAD_GROUP],
        "frequency_pairs": [FREQUENCY_PAIR],
        "frequency_scale": alpha,
        "position_warp_start": start,
        "warp_mode": "relative_distance",
    }


def make_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "name": "native_rope",
            "stage": "f47_relative_distance",
            "reference_original": True,
            "atoms": [],
        }
    ]
    for start in (8192, 16384):
        for alpha in (0.0, 0.25, 0.5):
            specs.append(
                {
                    "name": f"l18_23_g4_f47_relative_s{start}_a{alpha:g}",
                    "stage": "f47_relative_distance",
                    "atoms": [relative_atom(alpha, start)],
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
            {"mode": "f47_relative_distance", "specs": make_specs()},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(make_specs())} specs to {args.output}")


if __name__ == "__main__":
    main()
