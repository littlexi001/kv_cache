from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GROUPS = list(range(8))


def progressive_condition(name: str, stages: list[tuple[int, int, int]]) -> dict[str, Any]:
    atoms = []
    for layer_start, frequency_start, frequency_end in stages:
        atoms.append(
            {
                "layers": list(range(layer_start, 36)),
                "head_groups": GROUPS,
                "frequency_pairs": list(range(frequency_start, frequency_end)),
                "frequency_scale": 0.0,
            }
        )
    return {"name": name, "stage": "progressive_spectral", "atoms": atoms}


def progressive_specs() -> list[dict[str, Any]]:
    return [
        progressive_condition(
            "progressive_ridge",
            [(18, 0, 4), (21, 4, 8), (24, 8, 12), (30, 12, 16)],
        ),
        progressive_condition(
            "progressive_late",
            [(24, 0, 4), (27, 4, 8), (30, 8, 12), (33, 12, 16)],
        ),
        progressive_condition(
            "progressive_conservative",
            [(21, 0, 4), (27, 4, 8), (33, 8, 12)],
        ),
        progressive_condition(
            "progressive_validated",
            [(21, 0, 8), (24, 8, 12), (30, 12, 16)],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"stage": "progressive_spectral", "specs": progressive_specs()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
