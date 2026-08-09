from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "full_rope": "Full RoPE",
    "rope_top2": "post-RoPE Top-2%",
    "semantic_top2_postscore": "pre-select / post-score",
    "local_global_raw": "Local/Global raw",
    "local_global_calibrated": "Local/Global calibrated",
    "local_global_blend25": "SAGE blend 25%",
    "local_global_blend50": "SAGE blend 50%",
    "dual_max_blend25": "SAGE dual-max 25%",
}

COLORS = {
    "full_rope": "#64748b",
    "rope_top2": "#2563eb",
    "semantic_top2_postscore": "#ef4444",
    "local_global_raw": "#a855f7",
    "local_global_calibrated": "#f59e0b",
    "local_global_blend25": "#14b8a6",
    "local_global_blend50": "#10b981",
    "dual_max_blend25": "#0f766e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.rows_csv)
    grouped = (
        rows.groupby(["target_context_tokens", "variant"], as_index=False)
        .agg(
            sample_count=("seed", "count"),
            mean_gold_nll=("gold_nll", "mean"),
            next_token_accuracy=("next_token_correct", "mean"),
            gold_evidence_token_recall=("gold_evidence_token_recall", "mean"),
            gold_chain_complete_rate=("gold_chain_complete_rate", "mean"),
            gold_evidence_attention_mass=("gold_evidence_attention_mass", "mean"),
            mean_query_seconds=("query_seconds", "mean"),
        )
    )
    grouped["gold_ppl"] = grouped["mean_gold_nll"].map(math.exp)
    baseline = rows[
        rows.variant == "full_rope"
    ][["target_context_tokens", "seed", "gold_nll"]].rename(
        columns={"gold_nll": "full_gold_nll"}
    )
    paired = rows.merge(
        baseline,
        on=["target_context_tokens", "seed"],
        how="left",
    )
    paired["delta_nll_vs_full"] = paired.gold_nll - paired.full_gold_nll
    paired_summary = (
        paired.groupby(["target_context_tokens", "variant"], as_index=False)
        .agg(
            mean_delta_nll_vs_full=("delta_nll_vs_full", "mean"),
            median_delta_nll_vs_full=("delta_nll_vs_full", "median"),
            improved_seed_fraction=(
                "delta_nll_vs_full",
                lambda values: float((values < 0).mean()),
            ),
        )
    )
    grouped = grouped.merge(
        paired_summary,
        on=["target_context_tokens", "variant"],
        how="left",
    )
    grouped.to_csv(output_dir / "aggregate.csv", index=False)
    paired.to_csv(output_dir / "paired_rows.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for variant in LABELS:
        frame = grouped[grouped.variant == variant].sort_values(
            "target_context_tokens"
        )
        if frame.empty:
            continue
        x = frame.target_context_tokens / 1024
        axes[0, 0].plot(
            x,
            frame.gold_ppl,
            marker="o",
            label=LABELS[variant],
            color=COLORS[variant],
        )
        axes[0, 1].plot(
            x,
            frame.next_token_accuracy,
            marker="o",
            label=LABELS[variant],
            color=COLORS[variant],
        )
        if variant != "full_rope":
            axes[1, 0].plot(
                x,
                frame.gold_evidence_token_recall,
                marker="o",
                label=LABELS[variant],
                color=COLORS[variant],
            )
        axes[1, 1].plot(
            x,
            100 * frame.gold_evidence_attention_mass,
            marker="o",
            label=LABELS[variant],
            color=COLORS[variant],
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Gold PPL (geometric mean)")
    axes[0, 1].set_ylabel("First-token accuracy")
    axes[1, 0].set_ylabel("Gold evidence token recall")
    axes[1, 1].set_ylabel("Gold evidence attention mass (%)")
    for axis in axes.reshape(-1):
        axis.set_xlabel("Target context (K tokens)")
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Qwen3-8B: Local RoPE / Global Semantic Retrieval Probe",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(output_dir / "local_global_rope_comparison.png", dpi=180)
    plt.close(fig)

    result = {
        "row_count": int(len(rows)),
        "lengths": sorted(
            int(value) for value in rows.target_context_tokens.unique()
        ),
        "variants": [
            variant for variant in LABELS if variant in set(rows.variant)
        ],
        "aggregate_csv": "aggregate.csv",
        "paired_csv": "paired_rows.csv",
        "figure": "local_global_rope_comparison.png",
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
