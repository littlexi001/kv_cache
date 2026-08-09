from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = (
    ("exact_qk", "Exact QK", "#111111"),
    ("full_svd48_fp32", "Full SVD48", "#0072B2"),
    (
        "sampled_full_pca48_fp32",
        "Sampled-full PCA48",
        "#009E73",
    ),
    (
        "production_prefix_pca48_fp32",
        "First-2K PCA48",
        "#CC79A7",
    ),
    (
        "production_pca48_int4k_int8q",
        "First-2K + INT4/8",
        "#D55E00",
    ),
    (
        "production_pca48_int4k_int8q_sampled_quantile_uncapped",
        "+ 256-point threshold",
        "#E69F00",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.summary_path.read_text(encoding="utf-8"))
    rows = report["candidate_overall"]
    indexed = {
        (str(row["method"]), float(row["fraction"])): row for row in rows
    }
    fractions = sorted({float(row["fraction"]) for row in rows})

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 180,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.25))
    for method, label, color in METHODS:
        mass = [
            indexed[(method, fraction)]["retained_attention_mass_mean"]
            for fraction in fractions
        ]
        mass_p10 = [
            indexed[(method, fraction)]["retained_attention_mass_p10"]
            for fraction in fractions
        ]
        cosine = [
            indexed[(method, fraction)]["attention_output_cosine_mean"]
            for fraction in fractions
        ]
        axes[0].plot(
            [100.0 * fraction for fraction in fractions],
            mass,
            marker="o",
            linewidth=1.8,
            color=color,
            label=label,
        )
        axes[0].plot(
            [100.0 * fraction for fraction in fractions],
            mass_p10,
            linestyle=":",
            linewidth=1.0,
            color=color,
            alpha=0.65,
        )
        axes[1].plot(
            [100.0 * fraction for fraction in fractions],
            cosine,
            marker="o",
            linewidth=1.8,
            color=color,
            label=label,
        )

    axes[0].set_title("(a) Retained full-attention mass")
    axes[0].set_xlabel("Candidate budget (%)")
    axes[0].set_ylabel("Attention mass")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="lower right")
    axes[0].text(
        0.02,
        0.02,
        "solid: mean, dotted: p10",
        transform=axes[0].transAxes,
        fontsize=8,
    )

    axes[1].set_title("(b) Exact-QKV output cosine")
    axes[1].set_xlabel("Candidate budget (%)")
    axes[1].set_ylabel("Cosine to Full output")
    axes[1].grid(alpha=0.2)

    fraction = 0.04
    stage_order = [method for method, _, _ in METHODS]
    stage_labels = [
        "Exact\nQK",
        "Full\nSVD48",
        "Sampled-full\nPCA48",
        "First-2K\nPCA48",
        "First-2K\n+ INT4/8",
        "+ sampled\nthreshold",
    ]
    stage_colors = [color for _, _, color in METHODS]
    stage_mass = [
        indexed[(method, fraction)]["retained_attention_mass_mean"]
        for method in stage_order
    ]
    axes[2].bar(stage_labels, stage_mass, color=stage_colors, width=0.68)
    axes[2].set_ylim(min(stage_mass) - 0.004, max(stage_mass) + 0.004)
    axes[2].set_title("(c) Error ledger at 4%")
    axes[2].set_ylabel("Attention mass")
    axes[2].grid(axis="y", alpha=0.2)
    for index, value in enumerate(stage_mass):
        axes[2].text(index, value + 0.0005, f"{100.0 * value:.2f}%", ha="center", fontsize=8)

    figure.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_path, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
