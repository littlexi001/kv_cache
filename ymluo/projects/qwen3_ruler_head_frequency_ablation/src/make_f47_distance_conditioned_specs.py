from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LAYERS = list(range(18, 24))
HEAD_GROUP = 4
FREQUENCY_PAIR = 47


def intervention_atom(alpha: float, start: int | None = None) -> dict[str, Any]:
    atom: dict[str, Any] = {
        "layers": LAYERS,
        "head_groups": [HEAD_GROUP],
        "frequency_pairs": [FREQUENCY_PAIR],
        "frequency_scale": alpha,
    }
    if start is not None:
        atom["position_warp_start"] = start
    return atom


def make_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"name": "native_rope", "stage": "f47_distance_conditioned", "atoms": []},
        {
            "name": "l18_23_g4_f47_fixed_a0",
            "stage": "f47_distance_conditioned",
            "atoms": [intervention_atom(0.0)],
        },
    ]
    for start in (8192, 16384, 24576):
        for alpha in (0.0, 0.25, 0.5):
            specs.append(
                {
                    "name": f"l18_23_g4_f47_piecewise_s{start}_a{alpha:g}",
                    "stage": "f47_distance_conditioned",
                    "atoms": [intervention_atom(alpha, start)],
                }
            )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    specs = make_specs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"mode": "f47_distance_conditioned", "specs": specs},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(specs)} specs to {args.output}")


if __name__ == "__main__":
    main()
