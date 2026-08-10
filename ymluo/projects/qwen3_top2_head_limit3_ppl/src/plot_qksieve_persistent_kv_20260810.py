#!/usr/bin/env python
"""Plot independently audited persistent-KV lifecycle results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    summary = load(args.run_root / "independent_summary.json")
    if summary.get("all_correct") is not True:
        raise AssertionError("persistent lifecycle audit did not pass")
    rows = sorted(
        summary["aggregate_rows"], key=lambda row: row["history_tokens"]
    )
    if [row["history_tokens"] for row in rows] != [32768, 65536]:
        raise AssertionError("expected matched 32K and 64K results")
    if any(int(row.get("seed_count", -1)) != 3 for row in rows):
        raise AssertionError("expected three process repetitions per length")

    labels = (
        "Cold\nindex",
        "Cold\nE2E",
        "Warm",
        "4-branch\navg.",
        "Append\nonly",
    )
    fields = (
        "cold_speedup",
        "cold_end_to_end_speedup",
        "warm_speedup",
        "amortized_speedup",
        "append_only_speedup",
    )
    colors = ("#3B82F6", "#F59E0B")
    x = np.arange(len(labels))
    width = 0.34

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.05))
    speed_axis, build_axis = axes

    for index, row in enumerate(rows):
        values = [float(row[field]) for field in fields]
        lower = [
            value - float(row[f"{field}_bootstrap_ci95_low"])
            for field, value in zip(fields, values)
        ]
        upper = [
            float(row[f"{field}_bootstrap_ci95_high"]) - value
            for field, value in zip(fields, values)
        ]
        positions = x + (index - 0.5) * width
        bars = speed_axis.bar(
            positions,
            values,
            width,
            yerr=np.asarray([lower, upper]),
            capsize=2.5,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
            label=f"{row['history_tokens'] // 1024}K",
            color=colors[index],
            edgecolor="white",
            linewidth=0.5,
        )
        speed_axis.bar_label(bars, fmt="%.2fx", padding=2, fontsize=7)
    speed_axis.axhline(1.0, color="#374151", linestyle="--", linewidth=1)
    speed_axis.set_xticks(x, labels)
    speed_axis.set_ylabel("Whole-model speedup vs. Full")
    speed_axis.set_ylim(0.0, 2.85)
    speed_axis.set_title("(a) Measured request lifecycle", pad=38)
    speed_axis.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    speed_axis.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.7)

    stage_names = ("QK factors", "Key encode", "ValueSketch")
    stage_colors = ("#2563EB", "#10B981", "#EF4444")
    lengths = [row["history_tokens"] for row in rows]
    bottoms = np.zeros(len(lengths))
    for stage_name, key, color in zip(
        stage_names,
        ("qk_prebuild", "key_index_prebuild", "value_prebuild"),
        stage_colors,
    ):
        values = []
        for length in lengths:
            stage_values = [
                float(
                    load(
                        args.run_root
                        / f"n{length}"
                        / f"seed{seed}"
                        / "qksieve_robust.json"
                    )[key]["total_seconds"]
                )
                for seed in (20260810, 20260811, 20260812)
            ]
            values.append(statistics.median(stage_values))
        build_axis.bar(
            np.arange(len(lengths)),
            values,
            bottom=bottoms,
            width=0.56,
            label=stage_name,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms += np.asarray(values)
    for position, row in enumerate(rows):
        total = float(row["qksieve_prebuild_seconds"])
        low = float(row["qksieve_prebuild_seconds_bootstrap_ci95_low"])
        high = float(row["qksieve_prebuild_seconds_bootstrap_ci95_high"])
        build_axis.errorbar(
            position,
            total,
            yerr=np.asarray([[total - low], [high - total]]),
            fmt="none",
            ecolor="#111827",
            capsize=3,
            linewidth=1,
        )
        build_axis.text(position, high + 0.05, f"{total:.2f}s", ha="center", fontsize=8)
    build_axis.set_xticks(np.arange(len(lengths)), ["32K", "64K"])
    build_axis.set_ylabel("One-time build latency (s)")
    build_axis.set_ylim(0.0, 1.95)
    build_axis.set_title("(b) Directly timed index construction", pad=38)
    build_axis.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=0.8,
        handlelength=1.4,
    )
    build_axis.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.7)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout(w_pad=1.7)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / "persistent_kv_lifecycle.pdf", bbox_inches="tight")
    figure.savefig(args.output_dir / "persistent_kv_lifecycle.png", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
