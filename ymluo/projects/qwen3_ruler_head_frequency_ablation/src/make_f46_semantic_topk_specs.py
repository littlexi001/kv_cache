from __future__ import annotations

import argparse
import json
from pathlib import Path


def relative_atom(
    *,
    scale: float = 0.25,
    blend: float = 1.0,
    topk_fraction: float | None = None,
    replace_full_score: bool = False,
    semantic_score_blend: float = 1.0,
) -> dict[str, object]:
    atom: dict[str, object] = {
        "layers": [25],
        "head_groups": [3],
        "frequency_pairs": [46],
        "frequency_scale": scale,
        "position_warp_start": 8192,
        "warp_mode": "relative_distance",
        "score_blend": blend,
    }
    if topk_fraction is not None:
        atom.update(
            {
                "adaptive_gate": "semantic_topk",
                "adaptive_topk_fraction": topk_fraction,
                "adaptive_minimum_topk": 1,
                "adaptive_replace_full_score": replace_full_score,
                "adaptive_semantic_score_blend": semantic_score_blend,
            }
        )
    return atom


def make_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {
            "name": "native_rope",
            "stage": "f46_semantic_topk",
            "reference_original": True,
            "atoms": [],
        },
        {
            "name": "l25_g3_f46_relative_s8192_a0.25_b1_fixed",
            "stage": "f46_semantic_topk",
            "atoms": [relative_atom()],
        },
    ]
    for fraction in (0.005, 0.01, 0.02, 0.04):
        specs.append(
            {
                "name": f"l25_g3_f46_semantic_top{100*fraction:g}pct_a0.25",
                "stage": "f46_semantic_topk",
                "atoms": [relative_atom(topk_fraction=fraction)],
            }
        )
    specs.append(
        {
            "name": "l25_g3_f46_semantic_top2pct_a0",
            "stage": "f46_semantic_topk",
            "atoms": [relative_atom(scale=0.0, topk_fraction=0.02)],
        }
    )
    for fraction, blend in ((0.01, 0.25), (0.02, 0.25), (0.02, 0.5)):
        specs.append(
            {
                "name": (
                    f"l25_g3_semantic_reloc_top{100*fraction:g}pct_b{blend:g}"
                ),
                "stage": "f46_semantic_topk",
                "atoms": [
                    relative_atom(
                        topk_fraction=fraction,
                        replace_full_score=True,
                        semantic_score_blend=blend,
                    )
                ],
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
            {
                "purpose": (
                    "Retrieve remote candidates with analytically de-RoPE'd content "
                    "scores, then repair F46 only for the selected keys."
                ),
                "specs": specs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
