from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GROUPS = list(range(8))


def condition(name: str, layer_start: int, width: int) -> dict[str, Any]:
    return {
        "name": name,
        "stage": "final_candidate",
        "atoms": [
            {
                "layers": list(range(layer_start, 36)),
                "head_groups": GROUPS,
                "frequency_pairs": list(range(width)),
                "frequency_scale": 0.0,
            }
        ],
    }


def final_specs() -> list[dict[str, Any]]:
    return [
        {"name": "native_rope", "stage": "final_candidate", "atoms": []},
        condition("late_l24_f00_11_delete", 24, 12),
        condition("late_l30_f00_15_delete", 30, 16),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"stage": "final_candidate", "specs": final_specs()}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
