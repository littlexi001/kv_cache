from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot summary figures from head-count position CSV.")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--plot_dpi", type=int, default=180)
    args = parser.parse_args()

    with Path(args.summary_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = [int(row["selected_head_count"]) for row in rows]
    fractions = [float(row["fraction_of_selected_token_cases"]) for row in rows]
    token_cases = [int(row["token_cases"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=args.plot_dpi)
    bars = ax.bar(x_values, fractions, color="#4c78a8")
    ax.set_title("How many selected historical tokens have each head-count")
    ax.set_xlabel("Number of heads that selected the same historical token")
    ax.set_ylabel("Fraction of selected historical-token cases")
    ax.set_xticks(x_values)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, fractions):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "selected_token_cases_by_head_count.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=args.plot_dpi)
    bars = ax.bar(x_values, token_cases, color="#72b7b2")
    ax.set_title("Selected historical-token cases by head-count")
    ax.set_xlabel("Number of heads that selected the same historical token")
    ax.set_ylabel("Number of layer-query-token cases")
    ax.set_yscale("log")
    ax.set_xticks(x_values)
    ax.grid(True, axis="y", alpha=0.25, which="both")
    for bar, value in zip(bars, token_cases):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "selected_token_case_count_by_head_count_logy.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
