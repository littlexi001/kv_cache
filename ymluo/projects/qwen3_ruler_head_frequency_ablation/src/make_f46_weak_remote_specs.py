from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_specs() -> list[dict[str, object]]:
    return [
        {
            "name": "native_rope",
            "stage": "f46_weak_remote_exploratory",
            "reference_original": True,
            "atoms": [],
        },
        {
            "name": "l25_g3_f46_relative_s8192_a0_b0.25",
            "stage": "f46_weak_remote_exploratory",
            "atoms": [
                {
                    "layers": [25],
                    "head_groups": [3],
                    "frequency_pairs": [46],
                    "frequency_scale": 0.0,
                    "position_warp_start": 8192,
                    "warp_mode": "relative_distance",
                    "score_blend": 0.25,
                }
            ],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "mode": "f46_weak_remote_exploratory",
                "selection_note": (
                    "Pre-existing validation-grid candidate; exploratory frozen-test "
                    "comparison after the confirmatory candidate was selected."
                ),
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
