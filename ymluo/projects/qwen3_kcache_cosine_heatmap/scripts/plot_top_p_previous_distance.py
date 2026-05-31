from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot top-p previous-neighbor sequence-distance summaries.")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--token_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--token_bins", type=int, default=100)
    parser.add_argument("--metric", default="mean_index_distance_mean")
    parser.add_argument("--selected_heads", default="")
    parser.add_argument("--plot_all_heads", action="store_true")
    parser.add_argument("--plot_token_points", action="store_true")
    parser.add_argument("--plot_token_rank_points", action="store_true")
    parser.add_argument("--token_point_alpha", type=float, default=0.35)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_layer_head_heatmap(rows: list[dict[str, Any]], metric: str, output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    paths: list[str] = []
    cache_types = sorted({row["cache_type"] for row in rows})
    for cache_type in cache_types:
        subset = [row for row in rows if row["cache_type"] == cache_type]
        max_layer = max(int(row["layer"]) for row in subset)
        max_head = max(int(row["head"]) for row in subset)
        matrix = np.full((max_layer + 1, max_head + 1), np.nan, dtype=np.float32)
        for row in subset:
            matrix[int(row["layer"]), int(row["head"])] = as_float(row, metric)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(f"{cache_type.upper()} top-p previous distance: {metric}")
        ax.set_xlabel("KV head")
        ax.set_ylabel("Layer")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = output_dir / f"{cache_type}_{metric}_layer_head_heatmap.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_layer_average(rows: list[dict[str, Any]], metric: str, output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["cache_type"]][int(row["layer"])].append(as_float(row, metric))

    for cache_type, by_layer in grouped.items():
        layers = sorted(by_layer)
        values = [sum(by_layer[layer]) / len(by_layer[layer]) for layer in layers]
        fig, ax = plt.subplots(figsize=(10, 4), dpi=180)
        ax.plot(layers, values, marker="o", linewidth=1.5, markersize=3)
        ax.set_title(f"{cache_type.upper()} layer-average top-p previous distance")
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"{cache_type}_{metric}_layer_average.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


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


def all_heads(rows: list[dict[str, Any]]) -> set[tuple[str, int, int]]:
    return {(row["cache_type"], int(row["layer"]), int(row["head"])) for row in rows}


def head_output_dir(output_dir: Path, cache_type: str, layer: int, head: int) -> Path:
    path = output_dir / cache_type / f"layer_{layer:02d}" / f"head_{head:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_head_plot_index(output_dir: Path, selected_heads: set[tuple[str, int, int]]) -> str:
    rows = []
    for cache_type, layer, head in sorted(selected_heads):
        rows.append(
            {
                "cache_type": cache_type,
                "layer": layer,
                "head": head,
                "plot_dir": str(head_output_dir(output_dir, cache_type, layer, head)),
            }
        )
    path = output_dir / "head_plot_index.csv"
    write_csv(path, rows, ["cache_type", "layer", "head", "plot_dir"])
    return str(path)


def bin_token_rows(
    token_rows: list[dict[str, Any]],
    selected_heads: set[tuple[str, int, int]],
    token_bins: int,
    tokens_by_head: dict[tuple[str, int, int], int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int], dict[str, float]] = {}
    for row in token_rows:
        if row.get("mean_index_distance", "") == "":
            continue
        key_head = (row["cache_type"], int(row["layer"]), int(row["head"]))
        if key_head not in selected_heads:
            continue
        tokens = max(1, tokens_by_head.get(key_head, int(row.get("available_previous", "0")) + 1))
        token_index = int(row["token_index"])
        bin_index = min(token_bins - 1, int(token_index * token_bins / tokens))
        key = (*key_head, bin_index)
        payload = grouped.setdefault(
            key,
            {
                "count": 0.0,
                "mean_index_distance": 0.0,
                "mean_index_distance_percent_of_history": 0.0,
                "mean_index_distance_percent_of_context": 0.0,
            },
        )
        payload["count"] += 1
        payload["mean_index_distance"] += as_float(row, "mean_index_distance")
        payload["mean_index_distance_percent_of_history"] += as_float(
            row,
            "mean_index_distance_percent_of_history",
        )
        payload["mean_index_distance_percent_of_context"] += as_float(
            row,
            "mean_index_distance_percent_of_context",
        )

    binned_rows: list[dict[str, Any]] = []
    for (cache_type, layer, head, bin_index), payload in sorted(grouped.items()):
        count = payload["count"]
        binned_rows.append(
            {
                "cache_type": cache_type,
                "layer": layer,
                "head": head,
                "bin_index": bin_index,
                "count": int(count),
                "mean_index_distance": payload["mean_index_distance"] / count,
                "mean_index_distance_percent_of_history": payload[
                    "mean_index_distance_percent_of_history"
                ]
                / count,
                "mean_index_distance_percent_of_context": payload[
                    "mean_index_distance_percent_of_context"
                ]
                / count,
            }
        )
    return binned_rows


def tokens_by_head_from_summary(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], int]:
    result: dict[tuple[str, int, int], int] = {}
    for row in rows:
        result[(row["cache_type"], int(row["layer"]), int(row["head"]))] = int(float(row["tokens"]))
    return result


def plot_binned_token_trends(binned_rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in binned_rows:
        grouped[(row["cache_type"], int(row["layer"]), int(row["head"]))].append(row)

    for metric in ("mean_index_distance", "mean_index_distance_percent_of_history"):
        fig, ax = plt.subplots(figsize=(11, 5), dpi=180)
        for (cache_type, layer, head), rows in sorted(grouped.items()):
            rows = sorted(rows, key=lambda row: int(row["bin_index"]))
            ax.plot(
                [int(row["bin_index"]) for row in rows],
                [float(row[metric]) for row in rows],
                linewidth=1.2,
                label=f"{cache_type} L{layer} H{head}",
            )
        ax.set_title(f"Representative heads by token-position bins: {metric}")
        ax.set_xlabel("Token-position bin")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        path = output_dir / f"representative_heads_{metric}_by_token_bin.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_token_points(
    token_rows: list[dict[str, Any]],
    selected_heads: set[tuple[str, int, int]],
    output_dir: Path,
    alpha: float,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        if row.get("mean_index_distance", "") == "":
            continue
        key = (row["cache_type"], int(row["layer"]), int(row["head"]))
        if key in selected_heads:
            grouped[key].append(row)

    for cache_type, layer, head in sorted(grouped):
        rows = sorted(grouped[(cache_type, layer, head)], key=lambda row: int(row["token_index"]))
        plot_dir = head_output_dir(output_dir, cache_type, layer, head)
        token_indices = [int(row["token_index"]) for row in rows]
        metrics = [
            ("mean_index_distance", "Mean index distance"),
            ("mean_index_distance_percent_of_history", "Mean distance (% of available history)"),
            ("mean_index_distance_percent_of_context", "Mean distance (% of full context)"),
        ]
        for metric, ylabel in metrics:
            if not rows or metric not in rows[0]:
                continue
            values = [as_float(row, metric) for row in rows]
            fig, ax = plt.subplots(figsize=(12, 4), dpi=180)
            ax.scatter(token_indices, values, s=2, alpha=alpha, linewidths=0)
            ax.set_title(f"{cache_type.upper()} L{layer} H{head}: {ylabel}")
            ax.set_xlabel("Token index")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            path = plot_dir / f"{metric}_tokens.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(str(path))
    return paths


def selected_index_distances(row: dict[str, Any]) -> list[int]:
    token_index = int(row["token_index"])
    indices = [item for item in row.get("selected_indices", "").split(";") if item.strip()]
    return [token_index - int(item) for item in indices]


def plot_token_rank_points(
    token_rows: list[dict[str, Any]],
    selected_heads: set[tuple[str, int, int]],
    tokens_by_head: dict[tuple[str, int, int], int],
    output_dir: Path,
    alpha: float,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        if row.get("selected_indices", "") == "":
            continue
        key = (row["cache_type"], int(row["layer"]), int(row["head"]))
        if key in selected_heads:
            grouped[key].append(row)

    for cache_type, layer, head in sorted(grouped):
        rows = sorted(grouped[(cache_type, layer, head)], key=lambda row: int(row["token_index"]))
        plot_dir = head_output_dir(output_dir, cache_type, layer, head)
        max_rank = max((len(selected_index_distances(row)) for row in rows), default=0)
        if max_rank == 0:
            continue

        metrics = [
            ("index_distance", "Index distance", lambda distance, row: float(distance)),
            (
                "index_distance_percent_of_history",
                "Index distance (% of available history)",
                lambda distance, row: 100.0 * float(distance) / max(1, int(row["token_index"])),
            ),
            (
                "index_distance_percent_of_context",
                "Index distance (% of full context)",
                lambda distance, row: 100.0
                * float(distance)
                / max(1, tokens_by_head[(row["cache_type"], int(row["layer"]), int(row["head"]))]),
            ),
        ]

        for metric, ylabel, transform in metrics:
            fig, ax = plt.subplots(figsize=(12, 4), dpi=180)
            for rank in range(max_rank):
                xs: list[int] = []
                ys: list[float] = []
                for row in rows:
                    distances = selected_index_distances(row)
                    if rank >= len(distances):
                        continue
                    xs.append(int(row["token_index"]))
                    ys.append(transform(distances[rank], row))
                ax.scatter(xs, ys, s=2, alpha=alpha, linewidths=0, label=f"top{rank + 1}")
            ax.set_title(f"{cache_type.upper()} L{layer} H{head}: top-rank {ylabel}")
            ax.set_xlabel("Token index")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.2)
            ax.legend(markerscale=4, fontsize=8, ncol=min(max_rank, 5))
            fig.tight_layout()
            path = plot_dir / f"{metric}_by_rank_tokens.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    output_dir = Path(args.output_dir) if args.output_dir else summary_csv.parent / "top_p_previous_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_rows(summary_csv)
    paths = []
    paths.extend(plot_layer_head_heatmap(summary_rows, args.metric, output_dir))
    paths.extend(plot_layer_head_heatmap(summary_rows, "mean_index_distance_p50", output_dir))
    paths.extend(plot_layer_head_heatmap(summary_rows, "mean_index_distance_p95", output_dir))
    paths.extend(plot_layer_average(summary_rows, args.metric, output_dir))

    if args.token_csv:
        token_csv = Path(args.token_csv)
        if token_csv.exists():
            if args.plot_all_heads:
                selected_heads = all_heads(summary_rows)
            elif args.selected_heads:
                selected_heads = parse_selected_heads(args.selected_heads)
            else:
                selected_heads = choose_representative_heads(summary_rows, args.metric)
            paths.append(write_head_plot_index(output_dir, selected_heads))
            token_rows = read_rows(token_csv)
            binned_rows = bin_token_rows(
                token_rows,
                selected_heads,
                args.token_bins,
                tokens_by_head_from_summary(summary_rows),
            )
            binned_path = output_dir / "top_p_previous_distance_token_bins.csv"
            write_csv(
                binned_path,
                binned_rows,
                [
                    "cache_type",
                    "layer",
                    "head",
                    "bin_index",
                    "count",
                    "mean_index_distance",
                    "mean_index_distance_percent_of_history",
                    "mean_index_distance_percent_of_context",
                ],
            )
            paths.append(str(binned_path))
            paths.extend(plot_binned_token_trends(binned_rows, output_dir))
            if args.plot_token_points:
                paths.extend(plot_token_points(token_rows, selected_heads, output_dir, args.token_point_alpha))
            if args.plot_token_rank_points:
                paths.extend(
                    plot_token_rank_points(
                        token_rows,
                        selected_heads,
                        tokens_by_head_from_summary(summary_rows),
                        output_dir,
                        args.token_point_alpha,
                    )
                )

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
