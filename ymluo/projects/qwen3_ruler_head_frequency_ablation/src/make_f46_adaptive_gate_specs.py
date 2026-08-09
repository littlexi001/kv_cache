from __future__ import annotations

import argparse
import json
from pathlib import Path


LAYER = 25
HEAD_GROUP = 3
FREQUENCY = 46


def repair_atom() -> dict[str, object]:
    return {
        "layers": [LAYER],
        "head_groups": [HEAD_GROUP],
        "frequency_pairs": [FREQUENCY],
        "frequency_scale": 0.25,
        "position_warp_start": 8192,
        "warp_mode": "relative_distance",
        "score_blend": 1.0,
    }


def make_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {
            "name": "native_rope",
            "stage": "f46_adaptive_gate",
            "reference_original": True,
            "atoms": [],
        },
        {
            "name": "l25_g3_f46_relative_s8192_a0.25_b1_fixed",
            "stage": "f46_adaptive_gate",
            "atoms": [repair_atom()],
        },
    ]
    configs = (
        (0.05, 8, 0.5),
        (0.10, 8, 0.5),
        (0.20, 8, 0.5),
        (0.10, 16, 0.5),
        (0.10, 16, 0.75),
    )
    for mass_scale, topk, concentration_scale in configs:
        atom = repair_atom() | {
            "adaptive_gate": "remote_concentration",
            "adaptive_remote_mass_scale": mass_scale,
            "adaptive_topk": topk,
            "adaptive_topk_mass_scale": concentration_scale,
        }
        specs.append(
            {
                "name": (
                    f"l25_g3_f46_adaptive_m{mass_scale:g}_k{topk}_c{concentration_scale:g}"
                ),
                "stage": "f46_adaptive_gate",
                "atoms": [atom],
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
            {
                "mode": "f46_adaptive_gate",
                "selection_note": "Discovery only on RULER-32K seeds 43-44.",
                "specs": make_specs(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
