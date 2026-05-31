from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot K/V K-means clustering summaries.")
    parser.add_argument("--cluster_summary_csv", required=True)
    parser.add_argument("--cluster_assignments_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--metrics",
        default="mean_squared_distance,largest_cluster_fraction,cluster_entropy,cluster_size_std",
    )
    parser.add_argument("--selected_heads", default="")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else float("nan")


def parse_selected_heads(value: str) -> set[tuple[str, int, int]]:
    selected: set[tuple[str, int, int]] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        cache_type, layer, head = item.split(":")
        selected.add((cache_type.lower(), int(layer), int(head)))
    return selected


def choose_representative_heads(rows: list[dict[str, Any]], metric: str) -> set[tuple[str, int, int]]:
    selected: set[tuple[str, int, int]] = set()
    for cache_type in sorted({row["cache_type"] for row in rows}):
        subset = sorted(
            [row for row in rows if row["cache_type"] == cache_type],
            key=lambda row: as_float(row, metric),
        )
        for row in subset[:3] + subset[-3:]:
            selected.add((cache_type, int(row["layer"]), int(row["head"])))
    return selected


def plot_layer_head_heatmaps(rows: list[dict[str, Any]], metrics: list[str], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    paths: list[str] = []
    for metric in metrics:
        for cache_type in sorted({row["cache_type"] for row in rows}):
            subset = [row for row in rows if row["cache_type"] == cache_type]
            max_layer = max(int(row["layer"]) for row in subset)
            max_head = max(int(row["head"]) for row in subset)
            matrix = np.full((max_layer + 1, max_head + 1), np.nan, dtype=np.float32)
            for row in subset:
                matrix[int(row["layer"]), int(row["head"])] = as_float(row, metric)

            fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
            image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
            ax.set_title(f"{cache_type.upper()} cluster {metric}")
            ax.set_xlabel("KV head")
            ax.set_ylabel("Layer")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            path = output_dir / f"{cache_type}_{metric}_layer_head_heatmap.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(str(path))
    return paths


def plot_layer_averages(rows: list[dict[str, Any]], metrics: list[str], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    for metric in metrics:
        grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            grouped[row["cache_type"]][int(row["layer"])].append(as_float(row, metric))

        fig, ax = plt.subplots(figsize=(10, 4), dpi=180)
        for cache_type, by_layer in sorted(grouped.items()):
            layers = sorted(by_layer)
            values = [sum(by_layer[layer]) / len(by_layer[layer]) for layer in layers]
            ax.plot(layers, values, marker="o", linewidth=1.5, markersize=3, label=cache_type.upper())
        ax.set_title(f"Layer-average cluster {metric}")
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"{metric}_layer_average.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


def parse_cluster_sizes(value: str) -> list[int]:
    return [int(item) for item in value.split(";") if item.strip()]


def plot_cluster_size_bars(
    rows: list[dict[str, Any]],
    selected_heads: set[tuple[str, int, int]],
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    for row in rows:
        key = (row["cache_type"], int(row["layer"]), int(row["head"]))
        if key not in selected_heads:
            continue
        sizes = parse_cluster_sizes(row.get("cluster_sizes", ""))
        if not sizes:
            continue
        cache_type, layer, head = key
        fig, ax = plt.subplots(figsize=(10, 4), dpi=180)
        ax.bar(range(len(sizes)), sizes, width=0.8)
        ax.set_title(f"{cache_type.upper()} L{layer} H{head}: cluster sizes")
        ax.set_xlabel("Cluster id")
        ax.set_ylabel("Token count")
        fig.tight_layout()
        path = output_dir / f"{cache_type}_layer_{layer:02d}_head_{head:02d}_cluster_sizes.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_assignment_traces(
    assignment_rows: list[dict[str, Any]],
    selected_heads: set[tuple[str, int, int]],
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in assignment_rows:
        key = (row["cache_type"], int(row["layer"]), int(row["head"]))
        if key in selected_heads:
            grouped[key].append(row)

    paths: list[str] = []
    for cache_type, layer, head in sorted(grouped):
        rows = sorted(grouped[(cache_type, layer, head)], key=lambda row: int(row["token_index"]))
        fig, ax = plt.subplots(figsize=(12, 3.5), dpi=180)
        ax.scatter(
            [int(row["token_index"]) for row in rows],
            [int(row["cluster"]) for row in rows],
            s=2,
            alpha=0.45,
            linewidths=0,
        )
        ax.set_title(f"{cache_type.upper()} L{layer} H{head}: cluster assignment by token")
        ax.set_xlabel("Token index")
        ax.set_ylabel("Cluster id")
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        path = output_dir / f"{cache_type}_layer_{layer:02d}_head_{head:02d}_cluster_assignments.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.cluster_summary_csv)
    output_dir = Path(args.output_dir) if args.output_dir else summary_csv.parent / "cluster_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(summary_csv)
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    paths: list[str] = []
    paths.extend(plot_layer_head_heatmaps(rows, metrics, output_dir))
    paths.extend(plot_layer_averages(rows, metrics, output_dir))

    selected_heads = (
        parse_selected_heads(args.selected_heads)
        if args.selected_heads
        else choose_representative_heads(rows, "mean_squared_distance")
    )
    paths.extend(plot_cluster_size_bars(rows, selected_heads, output_dir))

    if args.cluster_assignments_csv:
        assignments_csv = Path(args.cluster_assignments_csv)
        if assignments_csv.exists():
            paths.extend(plot_assignment_traces(read_rows(assignments_csv), selected_heads, output_dir))

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
