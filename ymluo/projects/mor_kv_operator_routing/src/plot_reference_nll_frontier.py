from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "full": "Full",
    "learned_conformal": "Learned conformal",
    "risk_oracle": "Exact risk oracle",
    "qk_top_blocks": "Fixed QK-8",
    "lexical_blocks": "Fixed lexical-8",
    "uniform": "Fixed uniform-8",
    "streaming": "Fixed streaming-2",
}

COLORS = {
    "full": "#7f7f7f",
    "learned_conformal": "#e45756",
    "risk_oracle": "#54a24b",
    "qk_top_blocks": "#f58518",
    "lexical_blocks": "#b279a2",
    "uniform": "#72b7b2",
    "streaming": "#4c78a8",
}

MEAN_OFFSETS = {
    "full": (7, 5),
    "learned_conformal": (7, 7),
    "risk_oracle": (9, 12),
    "qk_top_blocks": (9, -6),
    "lexical_blocks": (9, 10),
    "uniform": (9, -11),
}

TAIL_OFFSETS = {
    "full": (7, 5),
    "learned_conformal": (7, 7),
    "risk_oracle": (9, -16),
    "qk_top_blocks": (9, 5),
    "lexical_blocks": (9, 5),
    "uniform": (9, 5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot causal NLL cost/tail frontiers.")
    parser.add_argument("--action_summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Query-disjoint causal sparse-attention reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.action_summary).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    fig.patch.set_facecolor("white")
    for row in rows:
        action = row["action"]
        x = float(row["mean_selected_blocks"])
        mean = float(row["mean_delta_nll_vs_full"])
        low = float(row["delta_nll_ci95_low"])
        high = float(row["delta_nll_ci95_high"])
        color = COLORS.get(action, "#333333")
        axes[0].errorbar(
            x,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            ms=9,
            capsize=4,
            color=color,
        )
        axes[1].scatter(
            x,
            float(row["p95_abs_delta_nll"]),
            s=85,
            color=color,
        )
        label = LABELS.get(action, action)
        for axis_index, y in [
            (0, mean),
            (1, float(row["p95_abs_delta_nll"])),
        ]:
            axes[axis_index].annotate(
                label,
                (x, y),
                xytext=(
                    MEAN_OFFSETS.get(action, (7, 7))
                    if axis_index == 0
                    else TAIL_OFFSETS.get(action, (7, 7))
                ),
                textcoords="offset points",
                fontsize=9.5,
            )

    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Mean logical KV blocks (lower is better)")
    axes[0].set_ylabel("Mean first-answer-token ΔNLL vs full")
    axes[0].grid(alpha=0.28)
    axes[1].set_xlabel("Mean logical KV blocks (lower is better)")
    axes[1].set_ylabel("p95 absolute first-token ΔNLL")
    axes[1].grid(alpha=0.28)
    fig.suptitle(args.title, fontsize=15)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
