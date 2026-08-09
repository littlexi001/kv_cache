from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    counts = (
        frame.groupby(
            [
                "strongest_competitor_token_id",
                "strongest_competitor_token_text",
            ]
        )
        .size()
        .sort_values(ascending=False)
    )
    competitor_id, competitor_text = counts.index[0]
    return {
        "points": int(len(frame)),
        "start_total_tokens": int(
            frame.total_tokens_before_generation.iloc[0]
        ),
        "end_total_tokens": int(
            frame.total_tokens_before_generation.iloc[-1]
        ),
        "correct_second_token_id": int(
            frame.correct_second_token_id.iloc[0]
        ),
        "correct_second_token_text": str(
            frame.correct_second_token_text.iloc[0]
        ),
        "top1_correct_rate": float(
            frame.top_is_correct.astype(bool).mean()
        ),
        "mean_correct_probability": float(
            frame.correct_second_token_probability.mean()
        ),
        "minimum_correct_probability": float(
            frame.correct_second_token_probability.min()
        ),
        "maximum_correct_probability": float(
            frame.correct_second_token_probability.max()
        ),
        "dominant_competitor_token_id": int(competitor_id),
        "dominant_competitor_token_text": str(competitor_text),
        "dominant_competitor_share": float(
            counts.iloc[0] / len(frame)
        ),
        "mean_competitor_probability": float(
            frame.strongest_competitor_probability.mean()
        ),
        "minimum_competitor_probability": float(
            frame.strongest_competitor_probability.min()
        ),
        "maximum_competitor_probability": float(
            frame.strongest_competitor_probability.max()
        ),
    }


def make_plot(
    frame: pd.DataFrame,
    summary: dict[str, Any],
    output: Path,
) -> None:
    x = frame.k_tokens.to_numpy(dtype=float)
    correct = (
        100
        * frame.correct_second_token_probability.to_numpy(
            dtype=float
        )
    )
    competitor = (
        100
        * frame.strongest_competitor_probability.to_numpy(
            dtype=float
        )
    )
    is_correct = frame.top_is_correct.astype(bool).to_numpy()

    fig, (state_ax, probability_ax) = plt.subplots(
        2,
        1,
        figsize=(15, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.6, 5.0]},
        constrained_layout=True,
    )
    state_ax.scatter(
        x,
        np.zeros_like(x),
        c=np.where(is_correct, "#56c7b1", "#ef6f6c"),
        marker="|",
        s=180,
        linewidths=2.2,
    )
    state_ax.set_yticks([])
    state_ax.set_ylim(-0.8, 0.8)
    state_ax.set_ylabel("Top-1")
    state_ax.set_title(
        "After emitting [space]: probability of the age token "
        "at the second decoding step"
    )

    probability_ax.plot(
        x,
        correct,
        color="#397dc1",
        marker="o",
        markersize=4.2,
        linewidth=1.8,
        label="P(correct answer token: 9)",
    )
    probability_ax.plot(
        x,
        competitor,
        color="#dd8a17",
        marker="o",
        markersize=4.2,
        linewidth=1.8,
        label="P(strongest competitor: 1)",
    )
    probability_ax.fill_between(
        x,
        competitor,
        correct,
        where=correct >= competitor,
        color="#56c7b1",
        alpha=0.10,
    )
    probability_ax.set_ylim(0, 100)
    probability_ax.set_xlim(x[0], x[-1])
    probability_ax.set_xlabel(
        "Prompt length before generation (K tokens)"
    )
    probability_ax.set_ylabel(
        "Second-step next-token probability (%)"
    )
    probability_ax.grid(alpha=0.28)
    probability_ax.legend(frameon=False, loc="best")
    probability_ax.text(
        0.012,
        0.47,
        (
            f"mean P(9)={100 * summary['mean_correct_probability']:.2f}%\n"
            f"mean P(1)={100 * summary['mean_competitor_probability']:.2f}%\n"
            f"9 is Top-1 at {100 * summary['top1_correct_rate']:.0f}% "
            "of checkpoints"
        ),
        transform=probability_ax.transAxes,
        ha="left",
        va="center",
        fontsize=10,
    )
    fig.savefig(output, dpi=210)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input_csv).sort_values(
        "total_tokens_before_generation"
    )
    summary = metrics(frame)
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_plot(
        frame,
        summary,
        output_dir / "post_space_p9_vs_competitor_136k_146k.png",
    )


if __name__ == "__main__":
    main()
