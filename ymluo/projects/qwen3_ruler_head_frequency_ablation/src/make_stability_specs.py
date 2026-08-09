from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def atom(layers: list[int], group: int, frequencies: list[int], alpha: float) -> dict[str, Any]:
    return {
        "layers": layers,
        "head_groups": [group],
        "frequency_pairs": frequencies,
        "frequency_scale": alpha,
    }


def spec(name: str, atoms: list[dict[str, Any]]) -> dict[str, Any]:
    return {"name": name, "stage": "stability_validation", "atoms": atoms}


def validation_specs() -> list[dict[str, Any]]:
    output = [spec("native_rope", [])]
    for alpha in (0.0, 0.125, 0.25, 0.5, 0.75):
        output.append(spec(f"l25_g3_f46_a{alpha:g}", [atom([25], 3, [46], alpha)]))
    for alpha in (0.0, 0.25, 0.5):
        output.append(
            spec(f"l18_23_g4_f47_a{alpha:g}", [atom(list(range(18, 24)), 4, [47], alpha)])
        )
    for alpha in (0.0, 0.25):
        output.append(
            spec(f"l25_g3_f40_47_a{alpha:g}", [atom([25], 3, list(range(40, 48)), alpha)])
        )
    for alpha in (0.0, 0.25):
        output.append(
            spec(
                f"dual_f46_f47_a{alpha:g}",
                [
                    atom([25], 3, [46], alpha),
                    atom(list(range(18, 24)), 4, [47], alpha),
                ],
            )
        )
    return output


def smoke_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "original_native",
            "stage": "stability_smoke",
            "bypass_original": True,
            "reference_original": True,
            "atoms": [],
        },
        {
            "name": "patched_native_replay",
            "stage": "stability_smoke",
            "compare_to_original": True,
            "atoms": [],
        },
        spec("smoke_l25_g3_f46_a0", [atom([25], 3, [46], 0.0)]),
        spec("smoke_l25_g3_f46_a0.5", [atom([25], 3, [46], 0.5)]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "validation"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    specs = smoke_specs() if args.mode == "smoke" else validation_specs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"mode": args.mode, "specs": specs}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(specs)} specs to {args.output}")


if __name__ == "__main__":
    main()
