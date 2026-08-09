from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot frozen uniform versus cross-layer allocated policies."
    )
    parser.add_argument("--policy", action="append", required=True, help="LABEL=summary.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for item in args.policy:
        label, path_raw = item.rsplit("=", 1)
        summary = json.loads(Path(path_raw).read_text(encoding="utf-8"))
        learned = next(
            row
            for row in summary["action_summary"]
            if row["action"] == "learned_conformal"
        )
        rows.append((label, learned))

    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    fig.patch.set_facecolor("white")
    for index, (label, row) in enumerate(rows):
        color = colors[index % len(colors)]
        x = 100.0 * float(row["mean_physical_gqa_saving_rate"])
        mean = float(row["mean_delta_nll_vs_full"])
        low = float(row["delta_nll_ci95_low"])
        high = float(row["delta_nll_ci95_high"])
        tail = float(row["p95_abs_delta_nll"])
        axes[0].errorbar(
            x,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            ms=9,
            capsize=4,
            color=color,
        )
        axes[1].scatter(x, tail, s=90, color=color)
        axes[0].annotate(label, (x, mean), xytext=(6, 6), textcoords="offset points")
        axes[1].annotate(label, (x, tail), xytext=(6, 6), textcoords="offset points")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    axes[0].set_xlabel("Physical GQA KV saving (%)")
    axes[0].set_ylabel("Mean first-token ΔNLL vs full")
    axes[1].set_xlabel("Physical GQA KV saving (%)")
    axes[1].set_ylabel("p95 absolute first-token ΔNLL")
    for axis in axes:
        axis.grid(alpha=0.28)
    fig.suptitle("Frozen cross-layer allocation on external zero-overlap holdout64")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
