from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot learned-router risk threshold sweep.")
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="THRESHOLD=summary.json; may be repeated.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = []
    for item in args.summary:
        threshold_raw, path_raw = item.split("=", 1)
        threshold = float(threshold_raw)
        summary = json.loads(Path(path_raw).read_text(encoding="utf-8"))
        learned = next(
            row
            for row in summary["action_summary"]
            if row["action"] == "learned_conformal"
        )
        points.append((threshold, learned))
    points.sort(key=lambda item: item[0])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    fig.patch.set_facecolor("white")
    x = [float(row["mean_selected_blocks"]) for _, row in points]
    mean_delta = [float(row["mean_delta_nll_vs_full"]) for _, row in points]
    low = [float(row["delta_nll_ci95_low"]) for _, row in points]
    high = [float(row["delta_nll_ci95_high"]) for _, row in points]
    tail = [float(row["p95_abs_delta_nll"]) for _, row in points]
    axes[0].errorbar(
        x,
        mean_delta,
        yerr=[
            [mean - lower for mean, lower in zip(mean_delta, low)],
            [upper - mean for mean, upper in zip(mean_delta, high)],
        ],
        marker="o",
        color="#e45756",
        capsize=4,
        linewidth=1.8,
    )
    axes[1].plot(x, tail, marker="o", color="#4c78a8", linewidth=1.8)
    for threshold, row in points:
        label = f"ε={threshold:.2f}"
        point_x = float(row["mean_selected_blocks"])
        axes[0].annotate(
            label,
            (point_x, float(row["mean_delta_nll_vs_full"])),
            xytext=(6, 6),
            textcoords="offset points",
        )
        axes[1].annotate(
            label,
            (point_x, float(row["p95_abs_delta_nll"])),
            xytext=(6, 6),
            textcoords="offset points",
        )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    axes[0].set_xlabel("Mean logical KV blocks")
    axes[0].set_ylabel("Mean first-token ΔNLL vs full")
    axes[1].set_xlabel("Mean logical KV blocks")
    axes[1].set_ylabel("p95 absolute first-token ΔNLL")
    for axis in axes:
        axis.grid(alpha=0.28)
    fig.suptitle("Learned conformal router risk/quality sweep")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
