#!/usr/bin/env python
"""Plot the frozen Robust RULER and cross-model quality evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RULER_LENGTH_SAMPLES = {
    "4096": 10,
    "8192": 10,
    "16384": 10,
    "32768": 10,
    "65536": 5,
    "131072": 5,
}
RULER_LENGTHS = set(RULER_LENGTH_SAMPLES)
RULER_TASKS = {
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_squad",
    "qa_hotpot",
}
MODEL_ORDER = ("llama31_8b", "qwen3_4b", "mistral_7b")


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
    contract = ruler.get("frozen_contract", {})
    method = contract.get("method")
    if not method:
        raise ValueError("RULER summary lacks the frozen method identifier")
    if multimodel.get("frozen_contract") != contract:
        raise ValueError("RULER and multi-model frozen contracts differ")
    if contract.get("full_attention_fallback") is not False:
        raise ValueError("the frozen method must not use Full fallback")
    if contract.get("length_switch") is not False:
        raise ValueError("the frozen method must not use a length switch")
    if contract.get("budget") != "min(N,1280,max(256,ceil(0.06*N)))":
        raise ValueError("the frozen token budget drifted")
    value_sketch = contract.get("value_sketch", {})
    if (
        value_sketch.get("rank"),
        value_sketch.get("bits"),
        value_sketch.get("block_tokens"),
        value_sketch.get("tail_alpha"),
    ) != (16, 4, 256, 0.5):
        raise ValueError("the frozen ValueSketch contract drifted")

    if ruler.get("strict_pairs") != 650 or ruler.get("rows") != 1300:
        raise ValueError("RULER evidence is not the complete 650-pair run")
    if set(ruler.get("tasks", [])) != RULER_TASKS:
        raise ValueError("RULER evidence does not cover the formal 13 tasks")
    length_samples = {
        str(length): int(samples)
        for length, samples in ruler.get("length_samples", {}).items()
    }
    if length_samples != RULER_LENGTH_SAMPLES:
        raise ValueError("RULER sample grid drifted")
    if set(ruler.get("per_length", {})) != RULER_LENGTHS:
        raise ValueError("RULER evidence does not cover the frozen length grid")
    if ruler.get("fallback_count") != 0:
        raise ValueError("RULER evidence observed a Full fallback")
    if ruler.get("bootstrap", {}).get("quality_retention_95ci") is None:
        raise ValueError("RULER quality confidence interval is missing")
    for length, row in ruler["per_length"].items():
        if row.get("cells") != 13:
            raise ValueError(f"RULER {length}: incomplete task aggregate")
        if row.get("bootstrap", {}).get("quality_retention_95ci") is None:
            raise ValueError(f"RULER {length}: confidence interval is missing")
        for field in ("full_macro", "qksieve_macro"):
            if field not in row:
                raise ValueError(f"RULER {length}: missing {field}")

    models = multimodel.get("models", {})
    if set(models) != set(MODEL_ORDER):
        raise ValueError("multi-model evidence lacks Llama/Qwen/Mistral")
    for model, row in models.items():
        if row.get("strict_pairs") != 160 or row.get("tasks") != 16:
            raise ValueError(f"{model}: incomplete LongBench screen")
        if row.get("full_fallback_count") != 0:
            raise ValueError(f"{model}: Full fallback was observed")
        if len(row.get("per_task", {})) != 16:
            raise ValueError(f"{model}: per-task evidence is incomplete")
        if row.get("quality_retention_95ci") is None:
            raise ValueError(f"{model}: quality confidence interval is missing")
    return str(method)


def main() -> None:
    args = parse_args()
    ruler = read_json(args.ruler)
    multimodel = read_json(args.multimodel)
    validate(ruler, multimodel)

    lengths = sorted(int(value) for value in ruler["per_length"])
    full_scores = [
        float(ruler["per_length"][str(length)]["full_macro"])
        for length in lengths
    ]
    ours_scores = [
        float(ruler["per_length"][str(length)]["qksieve_macro"])
        for length in lengths
    ]

    labels = ("Llama-3.1-8B", "Qwen3-4B", "Mistral-7B")
    rows = [multimodel["models"][name] for name in MODEL_ORDER]
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
