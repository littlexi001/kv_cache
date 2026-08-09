#!/usr/bin/env python
"""Plot the frozen Robust RULER and cross-model quality evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ruler",
        type=Path,
        default=ROOT / "data" / "qksieve_robust_ruler_summary.json",
    )
    parser.add_argument(
        "--multimodel",
        type=Path,
        default=ROOT / "data" / "qksieve_robust_multimodel_summary.json",
    )
    parser.add_argument(
        "--output_pdf",
        type=Path,
        default=ROOT / "figures" / "qksieve_quality_generalization.pdf",
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=ROOT / "figures" / "qksieve_quality_generalization.png",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def validate(ruler: dict[str, Any], multimodel: dict[str, Any]) -> str:
    if ruler.get("schema") != "qksieve_robust_ruler_summary_v1":
        raise ValueError("RULER summary schema mismatch")
    if multimodel.get("schema") != "qksieve_robust_multimodel_summary_v1":
        raise ValueError("multi-model summary schema mismatch")
    method = ruler.get("frozen_contract", {}).get("method")
    if not method:
        raise ValueError("RULER summary lacks the frozen method identifier")
    if multimodel.get("frozen_contract", {}).get("method") != method:
        raise ValueError("RULER and multi-model method identifiers differ")
    return str(method)


def main() -> None:
    args = parse_args()
    ruler = read_json(args.ruler)
    multimodel = read_json(args.multimodel)
    method = validate(ruler, multimodel)

    lengths = sorted(int(value) for value in ruler["per_length"])
    full_scores = [
        float(ruler["per_length"][str(length)]["full_kv"]["score"])
        for length in lengths
    ]
    ours_scores = [
        float(ruler["per_length"][str(length)][method]["score"])
        for length in lengths
    ]

    model_order = ("llama31_8b", "qwen3_4b", "mistral_7b")
    labels = ("Llama-3.1-8B", "Qwen3-4B", "Mistral-7B")
    rows = [multimodel["models"][name] for name in model_order]
    retentions = [100.0 * float(row["quality_retention"]) for row in rows]
    intervals = [row["quality_retention_95ci"] for row in rows]
    lower = [ret - 100.0 * float(ci[0]) for ret, ci in zip(retentions, intervals)]
    upper = [100.0 * float(ci[1]) - ret for ret, ci in zip(retentions, intervals)]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65))
    left, right = axes

    left.plot(
        [length / 1024 for length in lengths],
        full_scores,
        color="#3f4648",
        marker="o",
        linewidth=1.7,
        markersize=4.2,
        label="Full",
    )
    left.plot(
        [length / 1024 for length in lengths],
        ours_scores,
        color="#1f6f78",
        marker="s",
        linewidth=1.7,
        markersize=4.0,
        label="QKSieve-Robust",
    )
    left.set_xscale("log", base=2)
    left.set_xticks(
        [length / 1024 for length in lengths],
        [f"{length // 1024}K" for length in lengths],
    )
    left.set_xlabel("RULER context length")
    left.set_ylabel("Macro score")
    left.grid(axis="y", color="#d7dbdc", linewidth=0.6)
    left.legend(frameon=False, fontsize=7.4, loc="lower left")
    left.set_title("(a) Length generalization", fontsize=9)

    positions = list(range(len(labels)))
    right.errorbar(
        positions,
        retentions,
        yerr=[lower, upper],
        fmt="o",
        color="#c15f2a",
        ecolor="#6a7072",
        elinewidth=1.1,
        capsize=3.0,
        markersize=5.0,
    )
    right.axhline(100.0, color="#60686b", linewidth=0.9, linestyle="--")
    right.set_xticks(positions, labels, rotation=16, ha="right")
    right.set_ylabel("Quality retention (%)")
    right.grid(axis="y", color="#d7dbdc", linewidth=0.6)
    right.set_title("(b) Model transfer", fontsize=9)
    spread = max(
        [abs(value - 100.0) for value in retentions]
        + lower
        + upper
        + [1.0]
    )
    right.set_ylim(100.0 - 1.35 * spread, 100.0 + 1.35 * spread)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=7.6)

    fig.tight_layout(w_pad=2.2)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, bbox_inches="tight")
    fig.savefig(args.output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
