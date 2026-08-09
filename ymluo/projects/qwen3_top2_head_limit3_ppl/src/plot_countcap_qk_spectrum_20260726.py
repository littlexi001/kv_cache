#!/usr/bin/env python3
"""Plot the softmax-effective QK spectrum and Key-PCA diagnostics."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = {
    "llama31_8b": "Llama-3.1-8B",
    "qwen3_4b": "Qwen3-4B",
}
MODEL_COLORS = {
    "llama31_8b": "#0072B2",
    "qwen3_4b": "#D55E00",
}
RANKS = (8, 16, 24, 32, 48, 64)


def load_grouped_means(path: Path) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            for key, value in row.items():
                if key in {"model", "topic", "trace_path"} or value == "":
                    continue
                try:
                    grouped[model][key].append(float(value))
                except ValueError:
                    continue
    return {
        model: {
            key: float(np.mean(values))
            for key, values in metrics.items()
            if values
        }
        for model, metrics in grouped.items()
    }


def percent_axis(axis: plt.Axes) -> None:
    axis.set_ylim(0.0, 1.04)
    axis.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
    )


def plot_rank1_modes(
    axis: plt.Axes, summaries: dict[str, dict[str, float]]
) -> None:
    categories = ("Raw QK\nrank-1", "Centered QK\nrank-1", "Removed\nrow mean")
    keys = (
        "qk_rank1_energy_fraction",
        "centered_qk_rank1_energy_fraction",
        "softmax_invariant_row_mean_energy_fraction",
    )
    x = np.arange(len(categories))
    width = 0.34
    for index, model in enumerate(summaries):
        values = [summaries[model][key] for key in keys]
        axis.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model),
        )
    axis.set_xticks(x, categories)
    axis.set_title("(a) Mean-mode artifact")
    axis.set_ylabel("Energy fraction")
    percent_axis(axis)


def plot_cumulative_energy(
    axis: plt.Axes, summaries: dict[str, dict[str, float]]
) -> None:
    for model, summary in summaries.items():
        color = MODEL_COLORS.get(model)
        label = MODEL_LABELS.get(model, model)
        centered = [
            summary[f"centered_qk_energy_retained_optimal_rank{rank}"]
            for rank in RANKS
        ]
        key = [
            summary[f"key_energy_retained_rank{rank}"] for rank in RANKS
        ]
        axis.plot(
            RANKS,
            centered,
            marker="o",
            color=color,
            linewidth=2.0,
            label=f"{label}, centered QK",
        )
        axis.plot(
            RANKS,
            key,
            marker="s",
            color=color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
            label=f"{label}, Key",
        )
    axis.set_title("(b) Cumulative spectral energy")
    axis.set_xlabel("Rank")
    axis.set_ylabel("Retained energy")
    axis.set_xticks(RANKS)
    percent_axis(axis)
    axis.legend(loc="lower right", frameon=False, ncol=1)


def plot_rank48_fidelity(
    axis: plt.Axes, summaries: dict[str, dict[str, float]]
) -> None:
    categories = (
        "Optimal\nQK-SVD",
        "Full Key\nPCA",
        "First-2K\nKey PCA",
    )
    keys = (
        "centered_qk_energy_retained_optimal_rank48",
        "centered_qk_energy_retained_uncentered_key_pca48",
        "centered_production_prefix_pca_qk_fidelity",
    )
    x = np.arange(len(categories))
    width = 0.34
    for index, model in enumerate(summaries):
        values = [summaries[model][key] for key in keys]
        axis.bar(
            x + (index - 0.5) * width,
            values,
            width,
            color=MODEL_COLORS.get(model),
            label=MODEL_LABELS.get(model, model),
        )
    axis.set_xticks(x, categories)
    axis.set_title("(c) Rank-48 centered-QK fidelity")
    axis.set_ylabel("Retained energy")
    percent_axis(axis)


def plot_covariance_alignment(
    axis: plt.Axes, summaries: dict[str, dict[str, float]]
) -> None:
    categories = ("Raw Key", "Centered Key")
    keys = (
        "key_query_covariance_commutator_ratio",
        "centered_key_query_covariance_commutator_ratio",
    )
    x = np.arange(len(categories))
    width = 0.34
    for index, model in enumerate(summaries):
        values = [summaries[model][key] for key in keys]
        axis.bar(
            x + (index - 0.5) * width,
            values,
            width,
            color=MODEL_COLORS.get(model),
            label=MODEL_LABELS.get(model, model),
        )
    axis.set_xticks(x, categories)
    axis.set_title("(d) Key/Query covariance mismatch")
    axis.set_ylabel("Normalized commutator")
    axis.set_ylim(bottom=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_png", type=Path, required=True)
    parser.add_argument("--output_pdf", type=Path, required=True)
    args = parser.parse_args()

    summaries = load_grouped_means(args.input_csv)
    if not summaries:
        raise ValueError(f"No numeric rows found in {args.input_csv}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.1))
    plot_rank1_modes(axes[0, 0], summaries)
    plot_cumulative_energy(axes[0, 1], summaries)
    plot_rank48_fidelity(axes[1, 0], summaries)
    plot_covariance_alignment(axes[1, 1], summaries)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_png, dpi=240, bbox_inches="tight")
    figure.savefig(args.output_pdf, bbox_inches="tight")
    plt.close(figure)
    print(args.output_png)
    print(args.output_pdf)


if __name__ == "__main__":
    main()
