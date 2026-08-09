from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "original_47": "Original",
    "gap_plus_48": "Distance +48",
    "co_shift_plus_48": "Co-shift +48\n(relative fixed)",
}
COLORS = ["#3A86FF", "#EF476F", "#06A77D"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload["conditions"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "prompt_tokens",
                "hop2_relative_distance",
                "p_basket",
                "strongest_wrong",
                "p_strongest_wrong",
                "output_margin",
                "gold_ppl",
                "evidence_mass_all",
                "evidence_mass_l30_l33",
                "evidence_qk_all",
                "evidence_qk_l30_l33",
            ]
        )
        for row in rows:
            summary = row["summary"]
            writer.writerow(
                [
                    row["name"],
                    row["geometry"]["prompt_tokens"],
                    row["geometry"]["query_minus_evidence"]["hop2_result"],
                    summary["gold_probability"],
                    summary["strongest_wrong_token"],
                    summary["strongest_wrong_probability"],
                    summary["gold_vs_strongest_wrong_margin"],
                    summary["gold_ppl"],
                    summary["atomic_evidence_mass_all"],
                    summary["atomic_evidence_mass_l30_l33"],
                    summary["atomic_evidence_qk_mean_all"],
                    summary["atomic_evidence_qk_mean_l30_l33"],
                ]
            )

    x = np.arange(len(rows))
    labels = [LABELS[row["name"]] for row in rows]
    gold = np.array([row["summary"]["gold_probability"] for row in rows])
    wrong = np.array([row["summary"]["strongest_wrong_probability"] for row in rows])
    margins = np.array([row["summary"]["gold_vs_strongest_wrong_margin"] for row in rows])
    masses = np.array([row["summary"]["atomic_evidence_mass_all"] * 100 for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    width = 0.34
    axes[0].bar(x - width / 2, gold, width, color=COLORS, alpha=0.95, label="P(basket)")
    axes[0].bar(x + width / 2, wrong, width, color=COLORS, alpha=0.38, hatch="//", label="P(strongest wrong)")
    for index, (left, right) in enumerate(zip(gold, wrong)):
        axes[0].text(index - width / 2, left + 0.025, f"{left:.3f}", ha="center", fontsize=9)
        axes[0].text(index + width / 2, right + 0.025, f"{right:.3f}", ha="center", fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("A. Next-token probability")
    axes[0].set_ylabel("Probability")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].axhline(0, color="#6B7280", linewidth=1.2, linestyle="--")
    axes[1].bar(x, margins, color=COLORS, width=0.58)
    for index, value in enumerate(margins):
        offset = 0.1 if value >= 0 else -0.18
        axes[1].text(index, value + offset, f"{value:+.3f}", ha="center", va="center", fontsize=9)
    axes[1].set_ylim(-2.4, 2.4)
    axes[1].set_title("B. Gold vs. strongest-wrong margin")
    axes[1].set_ylabel("Log-probability margin")

    axes[2].bar(x, masses, color=COLORS, width=0.58)
    for index, value in enumerate(masses):
        axes[2].text(index, value + 0.035, f"{value:.3f}%", ha="center", fontsize=9)
    axes[2].set_ylim(0, max(1.7, float(masses.max()) * 1.22))
    axes[2].set_title("C. Atomic evidence attention mass")
    axes[2].set_ylabel("Mean mass across 36x32 heads (%)")

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Moving evidence and Query together largely preserves retrieval", fontsize=14)
    fig.savefig(output_dir / "relative_fixed_48_shift.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
