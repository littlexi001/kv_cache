from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BANDS = (
    (0, 15, "pairs 0–15 (highest frequency)"),
    (16, 31, "pairs 16–31"),
    (32, 47, "pairs 32–47"),
    (48, 63, "pairs 48–63 (lowest frequency)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--pair-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary_csv).sort_values(
        "relative_distance"
    )
    pairs = pd.read_csv(args.pair_csv)
    band_frames: list[pd.DataFrame] = []
    for lower, upper, label in BANDS:
        frame = pairs[
            (pairs["pair"] >= lower) & (pairs["pair"] <= upper)
        ]
        by_head = (
            frame.groupby(
                ["target_context_tokens", "relative_distance", "head"],
                as_index=False,
            )["pair_qk_contribution"]
            .sum()
        )
        aggregate = (
            by_head.groupby(
                ["target_context_tokens", "relative_distance"],
                as_index=False,
            )
            .agg(
                mean_band_qk=("pair_qk_contribution", "mean"),
                min_band_qk=("pair_qk_contribution", "min"),
                max_band_qk=("pair_qk_contribution", "max"),
            )
        )
        aggregate["band"] = label
        aggregate["pair_start"] = lower
        aggregate["pair_end"] = upper
        band_frames.append(aggregate)
    bands = pd.concat(band_frames, ignore_index=True)
    bands.to_csv(output_dir / "frequency_band_summary.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    x = summary["relative_distance"] / 1024
    axes[0].plot(
        x,
        summary["mean_gold_qk"],
        marker="o",
        linewidth=2.2,
        color="#0f766e",
    )
    axes[0].axhline(0, color="#64748b", linewidth=1, linestyle="--")
    axes[0].set_ylabel("Mean layer-0 gold QK")
    axes[0].set_title(
        "Qwen3-8B layer 0: exact RoPE phase contribution"
    )

    colors = ("#dc2626", "#f59e0b", "#2563eb", "#7c3aed")
    for (_, _, label), color in zip(BANDS, colors, strict=True):
        frame = bands[bands["band"] == label].sort_values(
            "relative_distance"
        )
        axes[1].plot(
            frame["relative_distance"] / 1024,
            frame["mean_band_qk"],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
    axes[1].axhline(0, color="#64748b", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Evidence–query distance (K tokens)")
    axes[1].set_ylabel("Mean QK contribution")
    axes[1].legend(frameon=False, ncol=2, fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    figure = output_dir / "first_layer_rope_phase_bands.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    manifest = {
        "summary_rows": int(len(summary)),
        "pair_rows": int(len(pairs)),
        "frequency_bands": [label for _, _, label in BANDS],
        "band_summary_csv": "frequency_band_summary.csv",
        "figure": figure.name,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
