from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


MODE_LABELS = {
    "residual_linear": "Residual fusion",
    "q_pre_current_phase": "Pre-RoPE fusion + current phase",
    "q_native_phase": "Native-phase post-RoPE fusion",
}
COLORS = {
    "offset64": "#4C78A8",
    "diverse1": "#F58518",
    "diverse2": "#54A24B",
    "diverse4": "#E45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in (
            "layer",
            "alpha",
            "gold_vs_fixed_competitor_margin",
            "critical_qk",
            "critical_evidence_attention_weighted",
        ):
            row[field] = float(row[field])
    return rows


def plot_metric(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    output_path: Path,
    metric: str,
    ylabel: str,
) -> None:
    layers = sorted({int(row["layer"]) for row in rows})
    modes = [mode for mode in MODE_LABELS if any(row["mode"] == mode for row in rows)]
    figure, axes = plt.subplots(
        len(layers),
        len(modes),
        figsize=(5.1 * len(modes), 3.8 * len(layers)),
        squeeze=False,
        sharex=True,
    )
    for row_index, layer in enumerate(layers):
        for column_index, mode in enumerate(modes):
            axis = axes[row_index][column_index]
            subset = [
                row for row in rows if int(row["layer"]) == layer and row["mode"] == mode
            ]
            for strategy in sorted({row["strategy"] for row in subset}):
                points = sorted(
                    [row for row in subset if row["strategy"] == strategy],
                    key=lambda row: float(row["alpha"]),
                )
                axis.plot(
                    [0.0] + [float(row["alpha"]) for row in points],
                    [float(baseline[metric])] + [float(row[metric]) for row in points],
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    color=COLORS.get(strategy),
                    label=strategy,
                )
            if metric == "gold_vs_fixed_competitor_margin":
                axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
            axis.grid(alpha=0.22)
            axis.set_title(f"L{layer} · {MODE_LABELS[mode]}")
            axis.set_xlabel("history mixture α")
            axis.set_ylabel(ylabel)
            if row_index == 0 and column_index == len(modes) - 1:
                axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(input_dir / "interventions.csv")
    with (input_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    baseline = summary["target_baseline"]
    plot_metric(
        rows,
        baseline,
        output_dir / "margin_curves.png",
        "gold_vs_fixed_competitor_margin",
        "nine − fixed competitor logit",
    )
    plot_metric(
        rows,
        baseline,
        output_dir / "critical_qk_curves.png",
        "critical_qk",
        "weighted evidence QK",
    )
    plot_metric(
        rows,
        baseline,
        output_dir / "attention_mass_curves.png",
        "critical_evidence_attention_weighted",
        "weighted evidence attention mass",
    )


if __name__ == "__main__":
    main()
