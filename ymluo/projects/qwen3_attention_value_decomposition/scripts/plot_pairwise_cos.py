from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mean pairwise cosine from value_pairwise_by_head.csv.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="", help="Default: <input parent>/plots/pairwise_cos.")
    parser.add_argument("--plot_dpi", type=int, default=180)
    parser.add_argument("--top_n", type=int, default=0, help="Optional: bar plot only top-N pairs by mean cosine.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def display_name(name: str) -> str:
    if name.startswith("top"):
        return "top" + name[3:].replace("p", ".")
    if name.startswith("tail"):
        return "tail" + name[4:].replace("p", ".")
    return name


def parse_value(value: str) -> float:
    return float(value.replace("p", "."))


def pair_label(left: str, right: str) -> str:
    return f"{display_name(left)} vs {display_name(right)}"


def vector_sort_key(name: str) -> tuple[int, float, str]:
    if name == "full":
        return (0, 0.0, name)
    if name.startswith("top"):
        try:
            return (1, parse_value(name[3:]), name)
        except ValueError:
            return (1, 0.0, name)
    if name.startswith("tail"):
        try:
            return (2, -parse_value(name[4:]), name)
        except ValueError:
            return (2, 0.0, name)
    return (3, 0.0, name)


def sort_vector_order(vector_order: list[str]) -> list[str]:
    return sorted(vector_order, key=vector_sort_key)


def aggregate_pair_cos(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], dict[tuple[str, str], float]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    vector_order: list[str] = []
    seen_vectors: set[str] = set()
    for row in rows:
        left = row["left"]
        right = row["right"]
        if left not in seen_vectors:
            seen_vectors.add(left)
            vector_order.append(left)
        if right not in seen_vectors:
            seen_vectors.add(right)
            vector_order.append(right)
        key = (left, right)
        grouped.setdefault(key, []).append(float(row["mean_cosine"]))

    summary_rows: list[dict[str, Any]] = []
    pair_mean: dict[tuple[str, str], float] = {}
    for (left, right), values in grouped.items():
        value = mean(values)
        pair_mean[(left, right)] = value
        pair_mean[(right, left)] = value
        summary_rows.append(
            {
                "left": left,
                "right": right,
                "left_display": display_name(left),
                "right_display": display_name(right),
                "pair": pair_label(left, right),
                "mean_cosine": value,
                "min_head_cosine": min(values),
                "max_head_cosine": max(values),
                "head_pair_count": len(values),
            }
        )
    summary_rows.sort(key=lambda row: row["mean_cosine"], reverse=True)
    return summary_rows, sort_vector_order(vector_order), pair_mean


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_bar(summary_rows: list[dict[str, Any]], output_path: Path, dpi: int, top_n: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary_rows[:top_n] if top_n > 0 else summary_rows
    labels = [row["pair"] for row in rows]
    values = [float(row["mean_cosine"]) for row in rows]
    height = max(4.5, 0.38 * len(rows))
    fig, ax = plt.subplots(figsize=(10, height), dpi=dpi)
    y = list(range(len(rows)))
    ax.barh(y, values, color="#2f6f9f")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean cosine over layers and heads")
    ax.set_title("Pairwise attention-value output cosine")
    ax.set_xlim(min(-1.0, min(values) - 0.05) if values else -1.0, 1.0)
    ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(value + 0.01 if value >= 0 else value - 0.01, idx, f"{value:.3f}", va="center", ha="left" if value >= 0 else "right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_heatmap(vector_order: list[str], pair_mean: dict[tuple[str, str], float], output_path: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(vector_order)
    matrix = [[1.0 if i == j else pair_mean.get((left, right), float("nan")) for j, right in enumerate(vector_order)] for i, left in enumerate(vector_order)]
    fig_size = max(5.5, 0.65 * n)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    display_order = [display_name(name) for name in vector_order]
    ax.set_xticklabels(display_order, rotation=45, ha="right")
    ax.set_yticklabels(display_order)
    ax.set_title("Mean pairwise cosine heatmap")
    for i in range(n):
        for j in range(n):
            value = matrix[i][j]
            if value == value:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="mean cosine")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else input_csv.parent / "plots" / "pairwise_cos"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_csv)
    required = {"left", "right", "mean_cosine"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{input_csv} must contain columns: {sorted(required)}")

    summary_rows, vector_order, pair_mean = aggregate_pair_cos(rows)
    summary_csv = output_dir / "pairwise_mean_cosine_summary.csv"
    write_csv(
        summary_csv,
        summary_rows,
        [
            "left",
            "right",
            "left_display",
            "right_display",
            "pair",
            "mean_cosine",
            "min_head_cosine",
            "max_head_cosine",
            "head_pair_count",
        ],
    )
    bar_path = output_dir / "pairwise_mean_cosine_bar.png"
    heatmap_path = output_dir / "pairwise_mean_cosine_heatmap.png"
    plot_bar(summary_rows, bar_path, args.plot_dpi, args.top_n)
    plot_heatmap(vector_order, pair_mean, heatmap_path, args.plot_dpi)

    summary = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv),
        "bar_plot": str(bar_path),
        "heatmap": str(heatmap_path),
        "pair_count": len(summary_rows),
        "vectors": vector_order,
        "vectors_display": [display_name(name) for name in vector_order],
    }
    (output_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
