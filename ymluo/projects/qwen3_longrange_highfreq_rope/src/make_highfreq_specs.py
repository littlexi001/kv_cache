from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LAYERS = list(range(36))
DEEP_LAYERS = list(range(18, 36))
LATE_LAYERS = list(range(24, 36))
GROUPS = list(range(8))


def absolute_atom(layers: list[int], frequencies: list[int], scale: float) -> dict[str, Any]:
    return {
        "layers": layers,
        "head_groups": GROUPS,
        "frequency_pairs": frequencies,
        "frequency_scale": scale,
    }


def relative_atom(
    layers: list[int], frequencies: list[int], scale: float, start: int,
    query_tail_tokens: int = 1,
) -> dict[str, Any]:
    return {
        "layers": layers,
        "head_groups": GROUPS,
        "frequency_pairs": frequencies,
        "frequency_scale": scale,
        "position_warp_start": start,
        "query_tail_tokens": query_tail_tokens,
        "score_blend": 1.0,
        "warp_mode": "relative_distance",
    }


def condition(name: str, atom: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "name": name,
        "stage": "highfreq_falsification",
        "atoms": [] if atom is None else [atom],
    }


def discovery_specs() -> list[dict[str, Any]]:
    high8 = list(range(0, 8))
    high16 = list(range(0, 16))
    return [
        condition("native_rope", None),
        condition("global_f00_07_delete", absolute_atom(LAYERS, high8, 0.0)),
        condition("global_f00_15_delete", absolute_atom(LAYERS, high16, 0.0)),
        condition("deep_f00_07_delete", absolute_atom(DEEP_LAYERS, high8, 0.0)),
        condition("deep_f00_15_delete", absolute_atom(DEEP_LAYERS, high16, 0.0)),
        condition("late_f00_07_delete", absolute_atom(LATE_LAYERS, high8, 0.0)),
        condition("deep_f00_07_scale025", absolute_atom(DEEP_LAYERS, high8, 0.25)),
        condition("deep_f00_07_scale050", absolute_atom(DEEP_LAYERS, high8, 0.50)),
        condition("deep_f08_15_delete", absolute_atom(DEEP_LAYERS, list(range(8, 16)), 0.0)),
        condition("deep_f16_23_delete", absolute_atom(DEEP_LAYERS, list(range(16, 24)), 0.0)),
        condition("deep_f56_63_delete", absolute_atom(DEEP_LAYERS, list(range(56, 64)), 0.0)),
        condition("remote4096_f00_07_stop", relative_atom(DEEP_LAYERS, high8, 0.0, 4096)),
        condition("remote8192_f00_07_stop", relative_atom(DEEP_LAYERS, high8, 0.0, 8192)),
        condition("remote4096_f00_07_scale025", relative_atom(DEEP_LAYERS, high8, 0.25, 4096)),
        condition("remote8192_f00_07_scale025", relative_atom(DEEP_LAYERS, high8, 0.25, 8192)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specs = discovery_specs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"stage": "discovery", "specs": specs}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(specs)} specs to {args.output}")


if __name__ == "__main__":
    main()
