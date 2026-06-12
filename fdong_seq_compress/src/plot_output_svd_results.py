from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot output-SVD sensitivity and attention-mask summaries.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--sensitivity-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    sensitivity_dir = Path(args.sensitivity_dir)
    mask_dir = Path(args.mask_dir)

    energy = pd.read_csv(artifact_dir / "singular_value_energy.csv")
    sensitivity = pd.read_csv(sensitivity_dir / "svd_direction_sensitivity.csv")
    singles = sensitivity[sensitivity["mode"] == "remove_single_direction"].sort_values("direction_index")
    bands = sensitivity[sensitivity["mode"] != "remove_single_direction"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(energy.direction_index, energy.cumulative_energy, color="#245b78")
    for level in (0.5, 0.8, 0.9, 0.95, 0.99):
        axes[0, 0].axhline(level, color="#bbbbbb", linewidth=0.6)
    axes[0, 0].set(
        xlabel="Singular direction index",
        ylabel="Cumulative uncentered energy",
        title="Final-X singular energy spectrum",
    )

    axes[0, 1].plot(singles.direction_index, singles.delta_loss, color="#9f2d20", marker="o", markersize=2.5)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_yscale("symlog", linthresh=1e-3)
    axes[0, 1].set(
        xlabel="Removed direction index (stride=8)",
        ylabel="Delta cross-entropy loss (symlog)",
        title="Single-direction sensitivity",
    )

    without_first = singles[singles.direction_index > 0]
    axes[1, 0].plot(
        without_first.direction_index,
        without_first.delta_loss,
        color="#9f2d20",
        marker="o",
        markersize=2.5,
    )
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_yscale("symlog", linthresh=1e-4)
    axes[1, 0].set(
        xlabel="Removed direction index (direction 0 excluded)",
        ylabel="Delta loss",
        title="Sensitivity beyond the common/top direction",
    )

    pivot = bands.pivot(index="band_fraction", columns="mode", values="ppl_ratio")
    x = np.arange(len(pivot))
    width = 0.36
    axes[1, 1].bar(x - width / 2, pivot["remove_top_band"], width, label="remove top band", color="#a33c2f")
    axes[1, 1].bar(x + width / 2, pivot["remove_tail_band"], width, label="remove tail band", color="#39765a")
    axes[1, 1].set_xticks(x, [f"{100 * value:g}%" for value in pivot.index])
    axes[1, 1].set_yscale("log")
    axes[1, 1].axhline(1, color="black", linewidth=0.8)
    axes[1, 1].set(
        xlabel="Removed hidden dimensions",
        ylabel="PPL ratio vs baseline (log)",
        title="Top vs tail subspace ablation",
    )
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(sensitivity_dir / "svd_sensitivity_summary.png", dpi=190)
    plt.close(fig)

    metrics = pd.read_csv(mask_dir / "condition_metrics.csv")
    order = [
        "full",
        "score_top_all",
        "score_top_without_front",
        "score_top_without_end",
        "score_top_without_answer",
        "score_top_without_other",
    ]
    metrics = metrics.set_index("condition").loc[order].reset_index()
    labels = ["Full", "Top 2%", "Top 2%\n-front", "Top 2%\n-end", "Top 2%\n-answer", "Top 2%\n-other"]
    colors = ["#555555", "#2d6f8e", "#557a46", "#b17a25", "#a33c2f", "#76558f"]
    x = np.arange(len(metrics))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes[0, 0].bar(x, metrics.ppl, color=colors)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Answer-token PPL (log)")
    axes[0, 0].set_title("Task loss under attention masks")

    axes[0, 1].bar(x, metrics.margin_mean, color=colors)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_ylabel("Correct vs strongest-competitor margin")
    axes[0, 1].set_title("Top-1 decision margin")

    width = 0.36
    axes[1, 0].bar(x - width / 2, metrics.x_cosine_to_full_mean, width, label="cosine to full", color="#2d6f8e")
    axes[1, 0].bar(x + width / 2, metrics.x_relative_l2_to_full_mean, width, label="relative L2", color="#b17a25")
    axes[1, 0].set_ylim(0, 1.08)
    axes[1, 0].set_title("Final-X geometry relative to full")
    axes[1, 0].legend()

    aggregate_wrong_mass = np.exp(metrics.loss) - 1
    axes[1, 1].bar(x, aggregate_wrong_mass, color=colors)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("exp(mean loss) - 1 (log)")
    axes[1, 1].set_title("Effective aggregate competition proxy")
    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(mask_dir / "mask_output_summary.png", dpi=190)
    plt.close(fig)

    projections = pd.read_csv(mask_dir / "svd_projection_shift_by_condition.csv")
    conditions = order[1:]
    matrix = np.array(
        [
            projections[projections.condition == condition]
            .sort_values("direction_index")
            .mean_abs_delta_coefficient.to_numpy()[:128]
            for condition in conditions
        ]
    )
    fig, axis = plt.subplots(figsize=(15, 4.5))
    image = axis.imshow(np.log10(matrix + 1e-4), aspect="auto", cmap="magma")
    axis.set_yticks(range(len(conditions)), labels[1:])
    axis.set_xlabel("Full-attention SVD direction index (first 128)")
    axis.set_title("Mask-induced final-X coefficient shifts")
    fig.colorbar(image, ax=axis, label="log10 mean absolute coefficient shift")
    fig.tight_layout()
    fig.savefig(mask_dir / "mask_svd_shift_heatmap_top128.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
