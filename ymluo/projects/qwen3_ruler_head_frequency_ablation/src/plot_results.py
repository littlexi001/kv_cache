from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BEST_SCORE = "finalist_0_ref_r0_freq_b0_g4_f47"
BEST_HELDOUT_NLL = "finalist_6_cross_r2_l25_g3_f46"
SCREEN_IDS = {
    "niah_multikey_3_32768_0",
    "fwe_32768_0",
    "cwe_32768_0",
    "niah_multivalue_32768_0",
    "qa_squad_32768_1",
    "qa_hotpot_32768_0",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def coarse_heatmaps(run_dir: Path, output: Path) -> None:
    rows = [row for row in read_json(run_dir / "coarse" / "summary.json") if row["variant"] != "native_rope"]
    values = np.zeros((3, 8, 8), dtype=float)
    for row in rows:
        region = row["spec"]["region"]
        values[int(region["block_index"]), int(region["head_group"]), int(region["band_index"])] = float(row["utility"])
    bound = float(np.max(np.abs(values)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    titles = ("Layers 18–23", "Layers 24–29", "Layers 30–35")
    image = None
    for block, ax in enumerate(axes):
        image = ax.imshow(values[block], cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        ax.set_title(titles[block])
        ax.set_xlabel("RoPE frequency band (pair indices)")
        ax.set_xticks(range(8), [f"{8*i}–{8*i+7}" for i in range(8)], rotation=45, ha="right")
        ax.set_ylabel("GQA head-group")
        ax.set_yticks(range(8), [f"G{i}" for i in range(8)])
    assert image is not None
    fig.colorbar(image, ax=axes, label="screen utility; positive means better than native RoPE")
    fig.suptitle("Coarse 6-example search: useful regions concentrate around frequencies 40–47")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def final_comparison(run_dir: Path, output: Path) -> None:
    summary = {row["variant"]: row for row in read_json(run_dir / "finalists" / "summary.json")}
    names = ["Native RoPE", "Highest score\nL18–23 G4 F47", "Stable Gold PPL\nL25 G3 F46"]
    variants = ["native_rope", BEST_SCORE, BEST_HELDOUT_NLL]
    scores = [100.0 * float(summary[name]["official_score_mean"]) for name in variants]
    ppls = [float(summary[name]["gold_answer_ppl_from_mean_nll"]) for name in variants]
    colors = ["#718096", "#2b6cb0", "#2f855a"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, values, title, ylabel, ylim in (
        (axes[0], scores, "Official RULER-32K score (higher is better)", "mean official score (%)", (75, 93)),
        (axes[1], ppls, "Gold answer PPL (lower is better)", "PPL from mean Gold NLL", (0, 42)),
    ):
        bars = ax.bar(names, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + (0.5 if ax is axes[0] else 0.7), f"{value:.2f}", ha="center")
    fig.suptitle("Complete 26-example RULER-32K validation")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def heldout_nll(run_dir: Path, output: Path) -> None:
    rows = read_jsonl(run_dir / "finalists" / "merged_rows.jsonl")
    by_sample = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_id"]][row["variant"]] = row
    deltas = []
    for sample_id, pair in by_sample.items():
        if sample_id in SCREEN_IDS:
            continue
        delta = float(pair["native_rope"]["gold_answer_mean_nll"]) - float(pair[BEST_HELDOUT_NLL]["gold_answer_mean_nll"])
        deltas.append((sample_id, delta))
    deltas.sort(key=lambda item: item[1])
    labels = [item[0] for item in deltas]
    values = [item[1] for item in deltas]
    colors = ["#2f855a" if value > 0 else "#c53030" for value in values]
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax.bar(range(len(values)), values, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(values)), labels, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("native Gold NLL − intervention Gold NLL")
    ax.set_title("Held-out 20 examples: L25 G3 F46 improves Gold NLL on 18/20 examples")
    ax.text(0.01, 0.97, "Positive (green) means the correct answer became more probable", transform=ax.transAxes, va="top")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coarse_heatmaps(args.run_dir, args.output_dir / "coarse_utility_heatmaps.png")
    final_comparison(args.run_dir, args.output_dir / "final_comparison.png")
    heldout_nll(args.run_dir, args.output_dir / "heldout_nll_deltas.png")


if __name__ == "__main__":
    main()
