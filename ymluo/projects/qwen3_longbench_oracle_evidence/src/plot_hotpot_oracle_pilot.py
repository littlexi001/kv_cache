#!/usr/bin/env python3
"""Plot the compact condition summary produced by the oracle pilot."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = [
    ("full", "Full"),
    ("oracle_sentence", "Oracle\nspan"),
    ("oracle_document", "Oracle\ndocument"),
    ("bm25_document", "BM25\ndocument"),
    ("random_document_mean", "Random\nmatched"),
    ("query_only", "Query\nonly"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = {
        row["condition"]: row
        for row in csv.DictReader(args.summary_csv.open(encoding="utf-8"))
    }
    sample_count = int(float(rows["full"]["samples"]))
    selected = [rows[key] for key, _ in ORDER]
    labels = [label for _, label in ORDER]
    f1 = [float(row["qa_f1_percent"]) for row in selected]
    em = [float(row["exact_match_percent"]) for row in selected]
    ppl = [float(row["mean_gold_answer_ppl"]) for row in selected]
    tokens = [float(row["mean_context_tokens"]) for row in selected]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, f1, width, label="QA-F1", color="#3b82f6")
    axes[0].bar(x + width / 2, em, width, label="Exact match", color="#14b8a6")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title(f"Answer quality ({sample_count} frozen HotpotQA samples)")
    axes[0].legend(frameon=False)

    colors = ["#64748b", "#8b5cf6", "#7c3aed", "#f59e0b", "#ef4444", "#94a3b8"]
    bars = axes[1].bar(x, ppl, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_ylim(max(min(ppl) * 0.65, 0.1), max(ppl) * 3.0)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Geometric gold-answer PPL (log scale)")
    axes[1].set_title("Confidence and retained context")
    for bar, token_count in zip(bars, tokens):
        axes[1].annotate(
            f"{token_count:.0f} ctx tok",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=35,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
