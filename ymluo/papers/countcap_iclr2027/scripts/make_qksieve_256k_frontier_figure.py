#!/usr/bin/env python
"""Plot the native-boundary 256K QKSieve quality-speed frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        required=True,
        action="append",
        type=Path,
        help="Multi-window summary JSON; repeat to merge frontier sweeps.",
    )
    parser.add_argument("--output_pdf", required=True, type=Path)
    parser.add_argument("--output_png", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    seen_variants = set()
    for path in args.summary:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["summaries"]:
            variant = str(row["variant"])
            if variant in seen_variants:
                raise ValueError(f"duplicate frontier variant: {variant}")
            seen_variants.add(variant)
            summaries.append(row)
    rows = sorted(
        summaries,
        key=lambda row: float(row["active_ratio_mean"]),
    )
    if len(rows) < 2:
        raise ValueError("at least two frontier points are required")

    active = 100 * np.asarray(
        [float(row["active_ratio_mean"]) for row in rows]
    )
    retention = 100 * np.asarray(
        [float(row["quality_retention"]) for row in rows]
    )
    interval = 100 * np.asarray(
        [row["quality_retention_95ci"] for row in rows],
        dtype=float,
    )
    retention_error = np.maximum(
        0.0,
        np.vstack(
            [retention - interval[:, 0], interval[:, 1] - retention]
        ),
    )
    kl = np.asarray(
        [float(row["kl_full_to_sparse"]) for row in rows]
    )
    top1 = 100 * np.asarray(
        [float(row["top1_agreement"]) for row in rows]
    )
    steady = np.asarray([float(row["steady_speedup"]) for row in rows])
    online = np.asarray(
        [float(row["online_decode_speedup"]) for row in rows]
    )

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "font.family": "DejaVu Sans",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.25))
    colors = {
        "retention": "#0b7285",
        "top1": "#2f9e44",
        "kl": "#c2410c",
        "steady": "#1d4ed8",
        "online": "#7c3aed",
    }

    axes[0].errorbar(
        active,
        retention,
        yerr=retention_error,
        color=colors["retention"],
        marker="o",
        linewidth=1.7,
        capsize=2.5,
        label="PPL retention",
    )
    axes[0].plot(
        active,
        top1,
        color=colors["top1"],
        marker="s",
        linewidth=1.4,
        label="Top-1 agreement",
    )
    axes[0].axhline(
        95.0,
        color="#6b7280",
        linestyle="--",
        linewidth=0.9,
    )
    axes[0].set_title("(a) Distribution quality")
    axes[0].set_ylabel("Relative to Full (%)")
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].plot(
        active,
        kl,
        color=colors["kl"],
        marker="D",
        linewidth=1.7,
    )
    axes[1].set_title("(b) Full-to-sparse KL")
    axes[1].set_ylabel("KL divergence")

    axes[2].plot(
        active,
        steady,
        color=colors["steady"],
        marker="o",
        linewidth=1.7,
        label="Steady decode",
    )
    axes[2].plot(
        active,
        online,
        color=colors["online"],
        marker="s",
        linewidth=1.4,
        label="Online decode",
    )
    axes[2].axhline(
        1.0,
        color="#6b7280",
        linestyle="--",
        linewidth=0.9,
    )
    axes[2].set_title("(c) Measured speedup")
    axes[2].set_ylabel("Speedup over Full")
    axes[2].legend(frameon=False, loc="upper right")

    for axis in axes:
        axis.set_xlabel("Active exact KV (%)")
        axis.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlim(active.min() - 0.25, active.max() + 0.25)

    figure.suptitle(
        "Qwen3-4B at the native 256K boundary, sampled quantile with c=64",
        y=1.03,
        fontsize=9.5,
    )
    figure.tight_layout(pad=0.65, w_pad=1.15)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_pdf, bbox_inches="tight")
    figure.savefig(args.output_png, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
