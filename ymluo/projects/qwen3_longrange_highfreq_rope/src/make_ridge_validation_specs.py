from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GROUPS = list(range(8))


def condition(layer_start: int, width: int, scale: float = 0.0) -> dict[str, Any]:
    suffix = "delete" if scale == 0.0 else f"scale{int(scale * 100):03d}"
    return {
        "name": f"late_l{layer_start:02d}_f00_{width - 1:02d}_{suffix}",
        "stage": "ridge_validation",
        "atoms": [
            {
                "layers": list(range(layer_start, 36)),
                "head_groups": GROUPS,
                "frequency_pairs": list(range(width)),
                "frequency_scale": scale,
            }
        ],
    }


def ridge_specs() -> list[dict[str, Any]]:
    return [
        condition(18, 4),
        condition(21, 8),
        condition(24, 12),
        condition(27, 12),
        condition(30, 12),
        condition(30, 16),
        condition(24, 8, 0.75),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"stage": "ridge_validation", "specs": ridge_specs()}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
