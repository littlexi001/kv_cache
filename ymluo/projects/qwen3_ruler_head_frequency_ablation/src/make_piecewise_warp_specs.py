from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def atom(
    layers: list[int],
    group: int,
    frequencies: list[int],
    alpha: float,
    start: int,
) -> dict[str, Any]:
    return {
        "layers": layers,
        "head_groups": [group],
        "frequency_pairs": frequencies,
        "frequency_scale": alpha,
        "position_warp_start": start,
    }


def spec(name: str, atoms: list[dict[str, Any]]) -> dict[str, Any]:
    return {"name": name, "stage": "piecewise_warp_validation", "atoms": atoms}


def piecewise_specs() -> list[dict[str, Any]]:
    output = [spec("native_rope", [])]
    for start in (8192, 16384, 24576):
        for alpha in (0.0, 0.25, 0.5):
            output.append(
                spec(
                    f"dual_piecewise_s{start}_a{alpha:g}",
                    [
                        atom([25], 3, [46], alpha, start),
                        atom(list(range(18, 24)), 4, [47], alpha, start),
                    ],
                )
            )
    for start in (16384, 24576):
        for alpha in (0.0, 0.25, 0.5):
            output.append(
                spec(
                    f"l25_g3_f46_piecewise_s{start}_a{alpha:g}",
                    [atom([25], 3, [46], alpha, start)],
                )
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    specs = piecewise_specs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"mode": "piecewise_warp", "specs": specs}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(specs)} specs to {args.output}")


if __name__ == "__main__":
    main()
