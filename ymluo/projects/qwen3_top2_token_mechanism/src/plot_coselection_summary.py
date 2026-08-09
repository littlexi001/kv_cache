from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cross-corpus Top-2% co-selection summaries.")
    parser.add_argument("--war_summary", required=True)
    parser.add_argument("--monte_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def read_rows(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(int(row["layer"]), int(row["head"])): row for row in rows}


def main() -> None:
    args = parse_args()
    war = read_rows(Path(args.war_summary))
    monte = read_rows(Path(args.monte_summary))
    keys = sorted(set(war) & set(monte))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    war_score = np.asarray([float(war[key]["cluster_score"]) for key in keys])
    monte_score = np.asarray([float(monte[key]["cluster_score"]) for key in keys])
    correlation = float(spearmanr(war_score, monte_score).statistic)
    figure, axis = plt.subplots(figsize=(6.4, 5.7), constrained_layout=True)
    scatter = axis.scatter(war_score, monte_score, c=[key[0] for key in keys], s=20, alpha=0.72, cmap="viridis")
    limit = max(float(war_score.max()), float(monte_score.max())) * 1.03
    axis.plot([0, limit], [0, limit], linestyle="--", color="0.55", linewidth=1)
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("War and Peace cluster score")
    axis.set_ylabel("Monte Cristo cluster score")
    axis.set_title(f"Per-head Top-2% co-selection structure (Spearman={correlation:.3f})")
    figure.colorbar(scatter, ax=axis, label="layer")
    figure.savefig(output_dir / "cross_corpus_cluster_score.png", dpi=args.dpi)
    plt.close(figure)

    layer_count = max(key[0] for key in keys) + 1
    head_count = max(key[1] for key in keys) + 1
    matrix = np.full((layer_count, head_count), np.nan, dtype=np.float64)
    pair_matrix = np.full_like(matrix, np.nan)
    for layer, head in keys:
        matrix[layer, head] = (float(war[(layer, head)]["cluster_score"]) + float(monte[(layer, head)]["cluster_score"])) / 2
        pair_matrix[layer, head] = (
            float(war[(layer, head)]["significant_positive_pairs"])
            + float(monte[(layer, head)]["significant_positive_pairs"])
        ) / 2

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 7.0), constrained_layout=True)
    image0 = axes[0].imshow(matrix, aspect="auto", cmap="magma", interpolation="nearest")
    axes[0].set_title("Mean cluster score across two corpora")
    figure.colorbar(image0, ax=axes[0], fraction=0.046)
    image1 = axes[1].imshow(np.log10(1.0 + pair_matrix), aspect="auto", cmap="viridis", interpolation="nearest")
    axes[1].set_title("log10(1 + significant pair count)")
    figure.colorbar(image1, ax=axes[1], fraction=0.046)
    for axis in axes:
        axis.set_xlabel("attention head")
        axis.set_ylabel("layer")
        axis.set_xticks(range(head_count))
        axis.set_yticks(range(0, layer_count, 2))
    figure.savefig(output_dir / "layer_head_coselection_atlas.png", dpi=args.dpi)
    plt.close(figure)


if __name__ == "__main__":
    main()
