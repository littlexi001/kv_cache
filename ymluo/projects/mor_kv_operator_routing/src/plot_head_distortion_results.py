from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot causal head-router risk/compute results.")
    parser.add_argument("--router_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(args.router_summary).read_text(encoding="utf-8"))
    policies = {row["policy"]: row for row in summary["policies"]}
    selected = [
        "learned_conformal",
        "static_head_prior",
        "test_oracle",
        "fixed_qk_top_blocks",
        "fixed_lexical_blocks",
        "fixed_streaming",
        "fixed_full",
    ]
    labels = {
        "learned_conformal": "Learned conformal",
        "static_head_prior": "Static head prior",
        "test_oracle": "Test oracle",
        "fixed_qk_top_blocks": "Fixed QK-8",
        "fixed_lexical_blocks": "Fixed lexical-8",
        "fixed_streaming": "Fixed streaming-2",
        "fixed_full": "Full",
    }
    colors = {
        "learned_conformal": "#e45756",
        "static_head_prior": "#4c78a8",
        "test_oracle": "#54a24b",
        "fixed_qk_top_blocks": "#f58518",
        "fixed_lexical_blocks": "#b279a2",
        "fixed_streaming": "#72b7b2",
        "fixed_full": "#777777",
    }
    annotation_offsets = {
        "learned_conformal": (-112, -14),
        "static_head_prior": (10, 8),
        "test_oracle": (8, 6),
        "fixed_qk_top_blocks": (8, 6),
        "fixed_lexical_blocks": (8, 6),
        "fixed_streaming": (8, 6),
        "fixed_full": (8, 6),
    }

    figure, axis = plt.subplots(figsize=(8.8, 5.8))
    for name in selected:
        row = policies[name]
        violation_percent = 100.0 * row["violation_rate"]
        plotted_violation = max(violation_percent, 0.05)
        axis.scatter(
            row["mean_physical_gqa_blocks"],
            plotted_violation,
            s=85,
            color=colors[name],
            label=labels[name],
            zorder=3,
        )
        axis.annotate(
            labels[name],
            (row["mean_physical_gqa_blocks"], plotted_violation),
            xytext=annotation_offsets[name],
            textcoords="offset points",
            fontsize=8.5,
        )
    axis.axhline(5.0, color="black", linestyle="--", linewidth=1.0, label="5% risk target")
    axis.set_xlabel("Mean physical GQA KV blocks (lower is better)")
    axis.set_ylabel("Relative-output-error violations (%)")
    axis.set_title("Query-disjoint causal KV operator routing")
    axis.set_yscale("log")
    axis.set_ylim(0.04, 100.0)
    axis.grid(alpha=0.28)
    figure.tight_layout()
    path = output_dir / "risk_vs_physical_gqa_blocks.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)

    comparison = [
        "learned_conformal",
        "static_head_prior",
        "test_oracle",
        "fixed_qk_top_blocks",
        "fixed_full",
    ]
    x = np.arange(len(comparison))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    logical = [policies[name]["mean_selected_blocks"] for name in comparison]
    physical = [policies[name]["mean_physical_gqa_blocks"] for name in comparison]
    axis.bar(x - width / 2, logical, width, label="Logical per query head", color="#9ecae9")
    axis.bar(x + width / 2, physical, width, label="Physical GQA union", color="#3182bd")
    axis.set_xticks(x, [labels[name] for name in comparison], rotation=18, ha="right")
    axis.set_ylabel("Mean KV blocks")
    axis.set_title("Logical decisions versus physical GQA cache")
    axis.grid(axis="y", alpha=0.28)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "logical_vs_physical_gqa_blocks.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
