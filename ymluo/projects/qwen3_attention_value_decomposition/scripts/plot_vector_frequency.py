from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_METRICS = "mean_norm,mean_attention_mass,mean_token_count"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot frequency comparisons from value_vectors_by_head.csv.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="", help="Default: <input parent>/plots/vector_frequency.")
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--vectors", default="", help="Optional comma-separated vector names to plot.")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--plot_dpi", type=int, default=160)
    parser.add_argument("--log_y", type=str2bool, default=False)
    return parser.parse_args()


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def display_name(name: str) -> str:
    if name.startswith("top"):
        return "top" + name[3:].replace("p", ".")
    if name.startswith("tail"):
        return "tail" + name[4:].replace("p", ".")
    return name


def parse_value(value: str) -> float:
    return float(value.replace("p", "."))


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


def parse_list(spec: str) -> list[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def read_rows(path: Path, vector_filter: set[str] | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"layer", "head", "vector", "query_count", "mean_norm", "mean_attention_mass", "mean_token_count"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        rows: list[dict[str, Any]] = []
        for row in reader:
            if vector_filter is not None and row["vector"] not in vector_filter:
                continue
            rows.append(row)
    return rows


def group_values(rows: list[dict[str, Any]], metric: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric, "")
        if value == "":
            continue
        grouped[row["vector"]].append(float(value))
    return grouped


def summary_rows(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    vectors = sorted({row["vector"] for row in rows}, key=vector_sort_key)
    output: list[dict[str, Any]] = []
    for vector in vectors:
        vector_rows = [row for row in rows if row["vector"] == vector]
        item: dict[str, Any] = {
            "vector": vector,
            "vector_display": display_name(vector),
            "layer_head_count": len(vector_rows),
            "layer_count": len({row["layer"] for row in vector_rows}),
            "head_count": len({row["head"] for row in vector_rows}),
        }
        for metric in metrics:
            values = sorted(float(row[metric]) for row in vector_rows if row.get(metric, "") != "")
            count = len(values)
            item[f"{metric}_mean"] = sum(values) / max(count, 1)
            item[f"{metric}_min"] = values[0] if values else ""
            item[f"{metric}_p50"] = values[count // 2] if values else ""
            item[f"{metric}_max"] = values[-1] if values else ""
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_overlay(grouped: dict[str, list[float]], metric: str, output_path: Path, bins: int, dpi: int, log_y: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vectors = sorted(grouped, key=vector_sort_key)
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=dpi)
    for vector in vectors:
        values = grouped[vector]
        if not values:
            continue
        ax.hist(values, bins=bins, histtype="step", linewidth=1.6, label=f"{display_name(vector)} (n={len(values)})")
    ax.set_xlabel(metric)
    ax.set_ylabel("Layer-head count")
    ax.set_title(f"Frequency comparison: {metric}")
    if log_y:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_metric_box(grouped: dict[str, list[float]], metric: str, output_path: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vectors = sorted(grouped, key=vector_sort_key)
    values = [grouped[vector] for vector in vectors]
    labels = [display_name(vector) for vector in vectors]
    width = max(8.0, 0.5 * len(vectors))
    fig, ax = plt.subplots(figsize=(width, 5.0), dpi=dpi)
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.set_ylabel(metric)
    ax.set_title(f"Layer-head distribution: {metric}")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else input_csv.parent / "plots" / "vector_frequency"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = parse_list(args.metrics)
    vector_filter = set(parse_list(args.vectors)) if args.vectors.strip() else None
    rows = read_rows(input_csv, vector_filter)
    if not rows:
        raise ValueError(f"No rows found in {input_csv}.")

    summary = summary_rows(rows, metrics)
    summary_csv = output_dir / "vector_frequency_summary.csv"
    write_csv(summary_csv, summary)

    plot_paths: list[str] = []
    for metric in metrics:
        grouped = group_values(rows, metric)
        overlay_path = output_dir / f"{metric}_frequency_overlay.png"
        box_path = output_dir / f"{metric}_boxplot.png"
        plot_metric_overlay(grouped, metric, overlay_path, args.bins, args.plot_dpi, args.log_y)
        plot_metric_box(grouped, metric, box_path, args.plot_dpi)
        plot_paths.extend([str(overlay_path), str(box_path)])

    result = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv),
        "metrics": metrics,
        "vectors": sorted({row["vector"] for row in rows}, key=vector_sort_key),
        "plot_paths": plot_paths,
    }
    (output_dir / "plot_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
