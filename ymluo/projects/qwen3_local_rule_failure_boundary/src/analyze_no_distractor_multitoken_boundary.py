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
    parser.add_argument("--points-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-total", type=int, default=136_000)
    parser.add_argument("--end-total", type=int, default=146_000)
    return parser.parse_args()


def token_label(value: str) -> str:
    text = str(value)
    if text == " " or not text:
        return "[space]"
    return text.replace("\n", "\\n")


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    competitor_counts = (
        frame.groupby(
            [
                "strongest_competitor_token_id",
                "strongest_competitor_token_text",
            ],
            dropna=False,
        )
        .size()
        .sort_values(ascending=False)
    )
    top_index = competitor_counts.index[0]
    competitor_text = (
        ""
        if pd.isna(top_index[1])
        else str(top_index[1])
    )
    return {
        "start_total_tokens": int(frame.total_tokens.iloc[0]),
        "end_total_tokens": int(frame.total_tokens.iloc[-1]),
        "points": int(len(frame)),
        "stride_tokens": int(
            np.median(np.diff(frame.total_tokens.to_numpy()))
        ),
        "semantic_accuracy": float(
            frame.semantic_correct.astype(bool).mean()
        ),
        "two_token_path_count": int(
            (frame.generated_token_count == 2).sum()
        ),
        "mean_nine_probability": float(
            frame.gold_exact_probability.mean()
        ),
        "minimum_nine_probability": float(
            frame.gold_exact_probability.min()
        ),
        "maximum_nine_probability": float(
            frame.gold_exact_probability.max()
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
        "dominant_competitor_token_id": int(top_index[0]),
        "dominant_competitor_token": token_label(
            competitor_text
        ),
        "dominant_competitor_share": float(
            competitor_counts.iloc[0] / len(frame)
        ),
    }


def make_plot(
    frame: pd.DataFrame,
    summary: dict[str, Any],
    output: Path,
) -> None:
    x = frame.total_tokens.to_numpy(dtype=float) / 1000.0
    nine = (
        100
        * frame.gold_exact_probability.to_numpy(dtype=float)
    )
    competitor = (
        100
        * frame.strongest_competitor_probability.to_numpy(
            dtype=float
        )
    )
    semantic_correct = frame.semantic_correct.astype(bool).to_numpy()

    fig, (state_ax, main_ax, zoom_ax) = plt.subplots(
        3,
        1,
        figsize=(15, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [0.55, 4.2, 2.1]},
        constrained_layout=True,
    )

    state_ax.scatter(
        x,
        np.zeros_like(x),
        c=np.where(semantic_correct, "#56c7b1", "#ef6f6c"),
        marker="|",
        s=180,
        linewidths=2.2,
    )
    state_ax.set_yticks([])
    state_ax.set_ylim(-0.8, 0.8)
    state_ax.set_ylabel("Semantic")
    state_ax.set_title(
        "No distractors, 136K–146K: first-token competition "
        "vs multi-token semantic answer"
    )

    main_ax.plot(
        x,
        nine,
        color="#397dc1",
        marker="o",
        markersize=3.7,
        linewidth=1.6,
        label="P(nine)",
    )
    main_ax.plot(
        x,
        competitor,
        color="#dd8a17",
        marker="o",
        markersize=3.7,
        linewidth=1.6,
        label="P(strongest competitor: [space])",
    )
    main_ax.set_ylim(0, 100)
    main_ax.set_ylabel("First-token probability (%)")
    main_ax.grid(axis="y", alpha=0.28)
    main_ax.legend(frameon=False, loc="best")
    main_ax.text(
        0.012,
        0.94,
        (
            f"semantic answer correct: "
            f"{100 * summary['semantic_accuracy']:.0f}%"
            f"  |  all {summary['points']} points use "
            "`[space] + 9`"
        ),
        transform=main_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    zoom_ax.plot(
        x,
        nine,
        color="#397dc1",
        marker="o",
        markersize=3.7,
        linewidth=1.6,
    )
    zoom_limit = max(5.0, 1.12 * float(np.max(nine)))
    zoom_ax.set_ylim(0, zoom_limit)
    zoom_ax.set_ylabel("P(nine), zoom (%)")
    zoom_ax.set_xlabel("Total sequence length (K tokens)")
    zoom_ax.grid(alpha=0.28)
    zoom_ax.set_xlim(x[0], x[-1])

    fig.savefig(output, dpi=210)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.points_csv)
    frame = frame[
        (frame.total_tokens >= args.start_total)
        & (frame.total_tokens <= args.end_total)
    ].copy()
    if frame.empty:
        raise RuntimeError("selected range contains no points")
    frame = frame.sort_values("total_tokens").reset_index(drop=True)
    summary = metrics(frame)
    (output_dir / "range_136k_146k.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_plot(
        frame,
        summary,
        output_dir / "nine_vs_competitor_136k_146k.png",
    )


if __name__ == "__main__":
    main()
