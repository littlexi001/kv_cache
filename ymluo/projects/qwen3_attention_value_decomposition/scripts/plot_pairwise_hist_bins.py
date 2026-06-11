from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cosine histograms from compact value_pairwise_hist_by_head.csv.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="", help="Default: <input parent>/plots/pairwise_hist_bins.")
    parser.add_argument("--plot_dpi", type=int, default=160)
    parser.add_argument("--pairs", default="", help="Optional comma-separated pairs like top0p01|tail0p1.")
    parser.add_argument("--layers", default="", help="Optional comma-separated layer ids.")
    parser.add_argument("--heads", default="", help="Optional comma-separated head ids.")
    return parser.parse_args()


def display_name(name: str) -> str:
    if name.startswith("top"):
        return "top" + name[3:].replace("p", ".")
    if name.startswith("tail"):
        return "tail" + name[4:].replace("p", ".")
    return name


def pair_key(row: dict[str, str]) -> str:
    return f"{row['left']}|{row['right']}"


def pair_display(pair: str) -> str:
    left, right = pair.split("|", 1)
    return f"{display_name(left)} vs {display_name(right)}"


def safe_name(name: str) -> str:
    return name.replace("|", "_vs_").replace(".", "p")


def parse_int_filter(spec: str) -> set[int] | None:
    if not spec.strip():
        return None
    return {int(part.strip()) for part in spec.split(",") if part.strip()}


def parse_pair_filter(spec: str) -> set[str] | None:
    if not spec.strip():
        return None
    return {part.strip() for part in spec.split(",") if part.strip()}


def read_hist_rows(path: Path, pair_filter: set[str] | None, layer_filter: set[int] | None, head_filter: set[int] | None):
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"layer", "head", "left", "right", "bin_left", "bin_right", "bin_center", "count"}
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
            grouped[(pair, str(layer), str(head))].append(
                {
                    "bin_left": float(row["bin_left"]),
                    "bin_right": float(row["bin_right"]),
                    "bin_center": float(row["bin_center"]),
                    "count": int(row["count"]),
                }
            )
            grouped[(pair, "all", "all")].append(
                {
                    "bin_left": float(row["bin_left"]),
                    "bin_right": float(row["bin_right"]),
                    "bin_center": float(row["bin_center"]),
                    "count": int(row["count"]),
                }
            )
    return grouped


def merge_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins: dict[tuple[float, float, float], int] = defaultdict(int)
    for row in rows:
        key = (row["bin_left"], row["bin_right"], row["bin_center"])
        bins[key] += int(row["count"])
    merged = [
        {"bin_left": left, "bin_right": right, "bin_center": center, "count": count}
        for (left, right, center), count in bins.items()
    ]
    return sorted(merged, key=lambda row: row["bin_center"])


def summary_row(pair: str, layer: str, head: str, bins: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["count"] for row in bins)
    mean = sum(row["bin_center"] * row["count"] for row in bins) / max(total, 1)
    return {
        "pair": pair,
        "pair_display": pair_display(pair),
        "layer": layer,
        "head": head,
        "count": total,
        "mean_from_bins": mean,
    }


def plot_hist_bins(bins: list[dict[str, Any]], title: str, output_path: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = [row["bin_center"] for row in bins]
    counts = [row["count"] for row in bins]
    width = (bins[0]["bin_right"] - bins[0]["bin_left"]) if bins else 0.03
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=dpi)
    ax.bar(centers, counts, width=width, color="#2f6f9f", edgecolor="white", align="center")
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
    output_dir = Path(args.output_dir) if args.output_dir else input_csv.parent / "plots" / "pairwise_hist_bins"
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = read_hist_rows(input_csv, parse_pair_filter(args.pairs), parse_int_filter(args.layers), parse_int_filter(args.heads))

    summary_rows: list[dict[str, Any]] = []
    plot_count = 0
    for (pair, layer, head), raw_bins in sorted(grouped.items()):
        bins = merge_bins(raw_bins)
        pair_dir = output_dir / safe_name(pair)
        pair_dir.mkdir(parents=True, exist_ok=True)
        if layer == "all" and head == "all":
            filename = "all_layers_heads.png"
            title = f"{pair_display(pair)} | all layers/heads"
        else:
            filename = f"layer_{int(layer):02d}_head_{int(head):02d}.png"
            title = f"{pair_display(pair)} | layer {layer}, head {head}"
        plot_hist_bins(bins, title, pair_dir / filename, args.plot_dpi)
        summary_rows.append(summary_row(pair, layer, head, bins))
        plot_count += 1

    summary_csv = output_dir / "pairwise_hist_bin_summary.csv"
    write_csv(summary_csv, summary_rows, ["pair", "pair_display", "layer", "head", "count", "mean_from_bins"])
    summary = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv),
        "plot_count": plot_count,
    }
    (output_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
