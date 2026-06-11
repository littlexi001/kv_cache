from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-layer/per-head cosine frequency grids from value_pairwise_hist_by_head.csv."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="", help="Default: <input parent>/plots/pairwise_cos_frequency_grid.")
    parser.add_argument("--pairs", default="", help="Comma-separated pairs like top0p01|tail0p1. Use all for all pairs.")
    parser.add_argument("--layers", default="all", help="Comma-separated layer ids, or all.")
    parser.add_argument("--heads", default="all", help="Comma-separated head ids, or all.")
    parser.add_argument("--plot_dpi", type=int, default=180)
    parser.add_argument("--max_cols", type=int, default=8, help="Number of head columns per grid.")
    parser.add_argument("--use_frequency", type=str2bool, default=True, help="Plot frequency instead of raw count.")
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


def pair_display(pair: str) -> str:
    left, right = pair.split("|", 1)
    return f"{display_name(left)} vs {display_name(right)}"


def safe_name(name: str) -> str:
    return name.replace("|", "_vs_").replace(".", "p")


def parse_int_filter(spec: str) -> set[int] | None:
    normalized = spec.strip().lower()
    if not normalized or normalized == "all":
        return None
    return {int(part.strip()) for part in normalized.split(",") if part.strip()}


def parse_pair_filter(spec: str) -> set[str] | None:
    normalized = spec.strip()
    if not normalized or normalized.lower() == "all":
        return None
    return {part.strip() for part in normalized.split(",") if part.strip()}


def pair_key(row: dict[str, str]) -> str:
    return f"{row['left']}|{row['right']}"


def read_hist_rows(
    path: Path,
    pair_filter: set[str] | None,
    layer_filter: set[int] | None,
    head_filter: set[int] | None,
) -> tuple[dict[tuple[str, int, int], list[dict[str, Any]]], list[int], list[int], list[str]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    layers: set[int] = set()
    heads: set[int] = set()
    pairs: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"layer", "head", "left", "right", "bin_center", "count", "frequency"}
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
            grouped[(pair, layer, head)].append(
                {
                    "bin_center": float(row["bin_center"]),
                    "count": int(row["count"]),
                    "frequency": float(row["frequency"]),
                }
            )
            layers.add(layer)
            heads.add(head)
            pairs.add(pair)
    return grouped, sorted(layers), sorted(heads), sorted(pairs)


def plot_pair_grid(
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]],
    pair: str,
    layers: list[int],
    heads: list[int],
    output_path: Path,
    dpi: int,
    use_frequency: bool,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(layers)
    n_cols = len(heads)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("No layers or heads matched the requested filters.")

    fig_width = max(10.0, n_cols * 1.45)
    fig_height = max(8.0, n_rows * 0.72)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=dpi, squeeze=False)
    y_key = "frequency" if use_frequency else "count"
    y_label = "freq" if use_frequency else "count"
    max_y = 0.0
    filled = 0

    for layer in layers:
        for head in heads:
            rows = grouped.get((pair, layer, head), [])
            if rows:
                max_y = max(max_y, max(float(row[y_key]) for row in rows))

    for row_index, layer in enumerate(layers):
        for col_index, head in enumerate(heads):
            ax = axes[row_index][col_index]
            rows = sorted(grouped.get((pair, layer, head), []), key=lambda item: item["bin_center"])
            if rows:
                centers = [row["bin_center"] for row in rows]
                values = [row[y_key] for row in rows]
                width = (centers[1] - centers[0]) * 0.95 if len(centers) > 1 else 0.03
                ax.bar(centers, values, width=width, color="#2f6f9f", edgecolor="none")
                filled += 1
            ax.set_xlim(-1.0, 1.0)
            if max_y > 0:
                ax.set_ylim(0.0, max_y * 1.05)
            ax.tick_params(axis="both", labelsize=5, length=1.5, pad=1)
            if row_index != n_rows - 1:
                ax.set_xticklabels([])
            if col_index != 0:
                ax.set_yticklabels([])
            if row_index == 0:
                ax.set_title(f"h{head}", fontsize=7, pad=2)
            if col_index == 0:
                ax.set_ylabel(f"L{layer}", fontsize=7, rotation=0, labelpad=10, va="center")

    fig.suptitle(f"{pair_display(pair)} cosine {y_label} by layer/head", fontsize=12)
    fig.supxlabel("cosine", fontsize=9)
    fig.supylabel(y_label, fontsize=9)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.985))
    fig.savefig(output_path)
    plt.close(fig)
    return {"pair": pair, "output": str(output_path), "panels": filled}


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else input_csv.parent / "plots" / "pairwise_cos_frequency_grid"
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped, layers, heads, pairs = read_hist_rows(
        input_csv,
        parse_pair_filter(args.pairs),
        parse_int_filter(args.layers),
        parse_int_filter(args.heads),
    )
    if not pairs:
        raise ValueError(f"No matching pairwise histogram rows found in {input_csv}")

    rows = []
    for pair in pairs:
        output_path = output_dir / f"{safe_name(pair)}_layer_head_grid.png"
        rows.append(plot_pair_grid(grouped, pair, layers, heads, output_path, args.plot_dpi, args.use_frequency))

    summary = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "pairs": pairs,
        "layers": layers,
        "heads": heads,
        "plot_count": len(rows),
        "plots": rows,
    }
    (output_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
