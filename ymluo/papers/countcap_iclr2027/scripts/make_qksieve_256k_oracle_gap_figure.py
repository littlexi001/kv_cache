#!/usr/bin/env python
"""Plot the strict-256K Exact-QK oracle gap of the low-bit selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VARIANTS = {
    "exact_qk_oracle_k1280": ("Exact FP16 QK oracle", 1280),
    "exact_qk_oracle_k2560": ("Exact FP16 QK oracle", 2560),
    "qksieve_keymse_fulltopk_k1280": ("QK-balanced low-bit proxy", 1280),
    "qksieve_keymse_fulltopk_k2560": ("QK-balanced low-bit proxy", 2560),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output_pdf", required=True, type=Path)
    parser.add_argument("--output_png", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = {
        str(row["variant"]): row
        for row in payload["summaries"]
        if str(row["variant"]) in VARIANTS
    }
    missing = sorted(set(VARIANTS) - set(rows))
    if missing:
        raise ValueError(f"missing variants: {missing}")

    series = {}
    for variant, (label, budget) in VARIANTS.items():
        series.setdefault(label, []).append((budget, rows[variant]))
    for values in series.values():
        values.sort(key=lambda item: item[0])

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "font.family": "DejaVu Sans",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.35))
    colors = {
        "Exact FP16 QK oracle": "#0b7285",
        "QK-balanced low-bit proxy": "#c2410c",
    }
    markers = {
        "Exact FP16 QK oracle": "o",
        "QK-balanced low-bit proxy": "s",
    }

    for label, values in series.items():
        budgets = np.asarray([budget for budget, _ in values])
        retention = 100 * np.asarray(
            [float(row["quality_retention"]) for _, row in values]
        )
        interval = 100 * np.asarray(
            [row["quality_retention_95ci"] for _, row in values],
            dtype=float,
        )
        errors = np.maximum(
            0.0,
            np.vstack(
                [
                    retention - interval[:, 0],
                    interval[:, 1] - retention,
                ]
            ),
        )
        kl = np.asarray(
            [float(row["kl_full_to_sparse"]) for _, row in values]
        )
        axes[0].errorbar(
            budgets,
            retention,
            yerr=errors,
            color=colors[label],
            marker=markers[label],
            linewidth=1.8,
            markersize=4.8,
            capsize=3,
            label=label,
        )
        axes[1].plot(
            budgets,
            kl,
            color=colors[label],
            marker=markers[label],
            linewidth=1.8,
            markersize=4.8,
            label=label,
        )

    axes[0].axhline(
        100.0,
        color="#374151",
        linestyle="--",
        linewidth=0.9,
        label="Full",
    )
    axes[0].axhline(
        95.0,
        color="#9ca3af",
        linestyle=":",
        linewidth=0.9,
    )
    axes[0].set_title("(a) Causal-PPL retention")
    axes[0].set_ylabel("Relative to Full (%)")
    axes[0].set_ylim(75.0, 103.0)

    axes[1].set_title("(b) Distribution error")
    axes[1].set_ylabel(r"$D_{\mathrm{KL}}(\mathrm{Full}\Vert\mathrm{Sparse})$")
    axes[1].set_yscale("log")

    for axis in axes:
        axis.set_xlabel("Exact tokens per Query head")
        axis.set_xticks([1280, 2560])
        axis.set_xlim(1120, 2720)
        axis.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
    )
    figure.tight_layout(rect=(0.0, 0.15, 1.0, 1.0), pad=0.7, w_pad=1.6)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_pdf, bbox_inches="tight")
    figure.savefig(args.output_png, dpi=240, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
