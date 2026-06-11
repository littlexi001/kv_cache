from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-token cosine histograms grouped by layer/head/pair.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="", help="Default: <input parent>/plots/pairwise_token_hist.")
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--plot_dpi", type=int, default=160)
    parser.add_argument("--pairs", default="", help="Optional comma-separated pair labels like full|top0p9,top0p9|tail0p1.")
    parser.add_argument("--layers", default="", help="Optional comma-separated layer ids.")
    parser.add_argument("--heads", default="", help="Optional comma-separated head ids.")
    return parser.parse_args()


def display_name(name: str) -> str:
    if name.startswith("top"):
        return "top" + name[3:].replace("p", ".")
    if name.startswith("tail"):
        return "tail" + name[4:].replace("p", ".")
    return name


def safe_name(name: str) -> str:
    return name.replace("|", "_vs_").replace(".", "p").replace(" ", "_")


def pair_key(row: dict[str, str]) -> str:
    return f"{row['left']}|{row['right']}"


def pair_display(pair: str) -> str:
    left, right = pair.split("|", 1)
    return f"{display_name(left)} vs {display_name(right)}"


def parse_int_filter(spec: str) -> set[int] | None:
    if not spec.strip():
        return None
    return {int(part.strip()) for part in spec.split(",") if part.strip()}


def parse_pair_filter(spec: str) -> set[str] | None:
    if not spec.strip():
        return None
    return {part.strip() for part in spec.split(",") if part.strip()}


def read_grouped_rows(path: Path, pair_filter: set[str] | None, layer_filter: set[int] | None, head_filter: set[int] | None):
    by_pair: dict[str, list[float]] = defaultdict(list)
    by_group: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"layer", "head", "left", "right", "query_index", "cosine"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row in reader:
            layer = int(row["layer"])
            head = int(row["head"])
            pair = pair_key(row)
            if pair_filter is not None and pair not in pair_filter:
                continue
            if layer_filter is not None and layer not in layer_filter:
                continue
            if head_filter is not None and head not in head_filter:
                continue
            value = float(row["cosine"])
            by_pair[pair].append(value)
            by_group[(pair, layer, head)].append(value)
    return by_pair, by_group


def summary_row(pair: str, layer: int | str, head: int | str, values: list[float]) -> dict[str, Any]:
    sorted_values = sorted(values)
    count = len(sorted_values)
    def quantile(q: float) -> float:
        if not sorted_values:
            return float("nan")
        idx = min(count - 1, max(0, round((count - 1) * q)))
        return sorted_values[idx]

    return {
        "pair": pair,
        "pair_display": pair_display(pair),
        "layer": layer,
        "head": head,
        "count": count,
        "mean": sum(sorted_values) / max(count, 1),
        "min": sorted_values[0] if sorted_values else "",
        "p05": quantile(0.05),
        "p25": quantile(0.25),
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "p95": quantile(0.95),
        "max": sorted_values[-1] if sorted_values else "",
    }


def plot_hist(values: list[float], title: str, output_path: Path, bins: int, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=dpi)
    ax.hist(values, bins=bins, range=(-1.0, 1.0), color="#2f6f9f", edgecolor="white", alpha=0.9)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Cosine")
    ax.set_ylabel("Token count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else input_csv.parent / "plots" / "pairwise_token_hist"
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_filter = parse_pair_filter(args.pairs)
    layer_filter = parse_int_filter(args.layers)
    head_filter = parse_int_filter(args.heads)
    by_pair, by_group = read_grouped_rows(input_csv, pair_filter, layer_filter, head_filter)

    summary_rows: list[dict[str, Any]] = []
    plot_count = 0
    for pair, values in sorted(by_pair.items()):
        pair_dir = output_dir / safe_name(pair)
        pair_dir.mkdir(parents=True, exist_ok=True)
        plot_hist(values, f"{pair_display(pair)} | all layers/heads", pair_dir / "all_layers_heads.png", args.bins, args.plot_dpi)
        summary_rows.append(summary_row(pair, "all", "all", values))
        plot_count += 1

    for (pair, layer, head), values in sorted(by_group.items()):
        pair_dir = output_dir / safe_name(pair)
        pair_dir.mkdir(parents=True, exist_ok=True)
        filename = f"layer_{layer:02d}_head_{head:02d}.png"
        title = f"{pair_display(pair)} | layer {layer}, head {head}"
        plot_hist(values, title, pair_dir / filename, args.bins, args.plot_dpi)
        summary_rows.append(summary_row(pair, layer, head, values))
        plot_count += 1

    summary_csv = output_dir / "pairwise_token_hist_summary.csv"
    fields = ["pair", "pair_display", "layer", "head", "count", "mean", "min", "p05", "p25", "p50", "p75", "p95", "max"]
    write_csv(summary_csv, summary_rows, fields)
    summary = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv),
        "plot_count": plot_count,
        "pairs": sorted(by_pair),
    }
    (output_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
