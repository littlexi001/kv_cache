from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GROUPS = list(range(8))


def condition(name: str, layers: list[int], frequencies: list[int], scale: float = 0.0) -> dict[str, Any]:
    return {
        "name": name,
        "stage": "late_spectral_grid",
        "atoms": [
            {
                "layers": layers,
                "head_groups": GROUPS,
                "frequency_pairs": frequencies,
                "frequency_scale": scale,
            }
        ],
    }


def grid_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"name": "native_rope", "stage": "late_spectral_grid", "atoms": []}
    ]
    for layer_start in (18, 21, 24, 27, 30, 33):
        for width in (4, 8, 12, 16):
            specs.append(
                condition(
                    f"late_l{layer_start:02d}_f00_{width - 1:02d}_delete",
                    list(range(layer_start, 36)),
                    list(range(width)),
                )
            )
    specs.extend(
        [
            condition(
                "late_l24_f00_07_scale025",
                list(range(24, 36)),
                list(range(8)),
                0.25,
            ),
            condition(
                "late_l24_f00_07_scale050",
                list(range(24, 36)),
                list(range(8)),
                0.50,
            ),
            condition(
                "late_l24_f00_07_scale075",
                list(range(24, 36)),
                list(range(8)),
                0.75,
            ),
            condition(
                "periodic_every4_full_nope",
                list(range(3, 36, 4)),
                list(range(64)),
            ),
            condition(
                "periodic_every4_f00_07_nope",
                list(range(3, 36, 4)),
                list(range(8)),
            ),
        ]
    )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specs = grid_specs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"stage": "late_spectral_grid", "specs": specs}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(specs)} specs to {args.output}")


if __name__ == "__main__":
    main()
