from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot incremental KV-cache SVD output CSV files."
    )
    parser.add_argument("--output_dir", default="outputs/kvcache_svd_profile")
    parser.add_argument("--plot_dir", default="")
    parser.add_argument("--plot_dpi", type=int, default=160)
    parser.add_argument("--max_rank", type=int, default=128)
    parser.add_argument("--cache_kinds", default="key,value")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument(
        "--heatmap_metrics",
        default=(
            "top1_singular_value,rank0_energy_fraction,"
            "u_cosine_mean,right_singular_vector_cosine_mean"
        ),
    )
    return parser.parse_args()


def parse_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"missing CSV, skipping: {path}", flush=True)
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_list(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_index_filter(spec: str) -> set[int] | None:
    normalized = spec.strip().lower()
    if normalized == "all":
        return None
    selected: set[int] = set()
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            selected.update(range(int(left), int(right) + 1))
        else:
            selected.add(int(part))
    return selected


def keep_row(row: dict[str, str], cache_kinds: set[str], layers: set[int] | None, heads: set[int] | None) -> bool:
    if row.get("cache_kind") not in cache_kinds:
        return False
    if layers is not None and int(row["layer"]) not in layers:
        return False
    if heads is not None and int(row["head"]) not in heads:
        return False
    return True


def ensure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_singular_value_plots(rows: list[dict[str, str]], plot_dir: Path, max_rank: int, dpi: int) -> None:
    plt = ensure_matplotlib()
    grouped: dict[tuple[str, int, int], dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        rank = int(row["rank"])
        if rank >= max_rank:
            continue
        key = (row["cache_kind"], int(row["layer"]), int(row["head"]))
        grouped[key][int(row["cache_length"])].append(row)

    target_dir = plot_dir / "singular_values_by_head"
    target_dir.mkdir(parents=True, exist_ok=True)
    for (cache_kind, layer, head), by_length in grouped.items():
        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        for cache_length, series in sorted(by_length.items()):
            series = sorted(series, key=lambda item: int(item["rank"]))
            ranks = [int(item["rank"]) for item in series]
            values = [float(item["singular_value"]) for item in series]
            ax.plot(ranks, values, marker="o", markersize=2, linewidth=1.2, label=str(cache_length))
        ax.set_title(f"{cache_kind} singular values L{layer} H{head}")
        ax.set_xlabel("rank")
        ax.set_ylabel("singular value")
        ax.grid(alpha=0.25)
        ax.legend(title="length")
        fig.tight_layout()
        fig.savefig(target_dir / f"{cache_kind}_layer_{layer:02d}_head_{head:02d}_singular_values.png")
        plt.close(fig)


def save_energy_plots(rows: list[dict[str, str]], plot_dir: Path, max_rank: int, dpi: int) -> None:
    plt = ensure_matplotlib()
    grouped: dict[tuple[str, int, int], dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        rank = int(row["rank"])
        if rank >= max_rank:
            continue
        key = (row["cache_kind"], int(row["layer"]), int(row["head"]))
        grouped[key][int(row["cache_length"])].append(row)

    target_dir = plot_dir / "energy_by_head"
    target_dir.mkdir(parents=True, exist_ok=True)
    for (cache_kind, layer, head), by_length in grouped.items():
        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        for cache_length, series in sorted(by_length.items()):
            series = sorted(series, key=lambda item: int(item["rank"]))
            ranks = [int(item["rank"]) for item in series]
            values = [float(item["cumulative_energy_fraction"]) for item in series]
            ax.plot(ranks, values, marker="o", markersize=2, linewidth=1.2, label=str(cache_length))
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{cache_kind} cumulative SVD energy L{layer} H{head}")
        ax.set_xlabel("rank")
        ax.set_ylabel("cumulative energy fraction")
        ax.grid(alpha=0.25)
        ax.legend(title="length")
        fig.tight_layout()
        fig.savefig(target_dir / f"{cache_kind}_layer_{layer:02d}_head_{head:02d}_cumulative_energy.png")
        plt.close(fig)


def save_cosine_plots(rows: list[dict[str, str]], plot_dir: Path, max_rank: int, dpi: int) -> None:
    plt = ensure_matplotlib()
    grouped: dict[tuple[str, int, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rank = int(row["rank"])
        if rank >= max_rank:
            continue
        key = (
            row["cache_kind"],
            int(row["layer"]),
            int(row["head"]),
            int(row["left_length"]),
            int(row["right_length"]),
        )
        grouped[key].append(row)

    target_dir = plot_dir / "cosine_by_head_pair"
    target_dir.mkdir(parents=True, exist_ok=True)
    for (cache_kind, layer, head, left_length, right_length), series in grouped.items():
        series = sorted(series, key=lambda item: int(item["rank"]))
        ranks = [int(item["rank"]) for item in series]
        u_values = [float(item["u_cosine"]) for item in series]
        right_values = [float(item["right_singular_vector_cosine"]) for item in series]
        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax.plot(ranks, u_values, marker="o", markersize=2, linewidth=1.2, label="U prefix cosine")
        ax.plot(
            ranks,
            right_values,
            marker="x",
            markersize=3,
            linewidth=1.2,
            label="right singular vector cosine",
        )
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{cache_kind} SVD vector cosine L{layer} H{head}: {left_length} vs {right_length}")
        ax.set_xlabel("rank")
        ax.set_ylabel("cosine")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            target_dir
            / f"{cache_kind}_layer_{layer:02d}_head_{head:02d}_{left_length}_vs_{right_length}_cosine.png"
        )
        plt.close(fig)


def singular_heatmap_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["cache_kind"], int(row["cache_length"]), int(row["layer"]), int(row["head"]))
        by_group[key].append(row)

    result: list[dict[str, Any]] = []
    for (cache_kind, cache_length, layer, head), series in by_group.items():
        by_rank = {int(item["rank"]): item for item in series}
        rank0 = by_rank.get(0)
        if rank0 is None:
            continue
        result.append(
            {
                "cache_kind": cache_kind,
                "cache_length": cache_length,
                "layer": layer,
                "head": head,
                "top1_singular_value": float(rank0["singular_value"]),
                "rank0_energy_fraction": float(rank0["energy_fraction"]),
            }
        )
    return result


def cosine_heatmap_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, int, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["cache_kind"],
            int(row["left_length"]),
            int(row["right_length"]),
            int(row["layer"]),
            int(row["head"]),
        )
        by_group[key].append(row)

    result: list[dict[str, Any]] = []
    for (cache_kind, left_length, right_length, layer, head), series in by_group.items():
        u_values = [float(item["u_cosine"]) for item in series]
        right_values = [float(item["right_singular_vector_cosine"]) for item in series]
        result.append(
            {
                "cache_kind": cache_kind,
                "left_length": left_length,
                "right_length": right_length,
                "layer": layer,
                "head": head,
                "u_cosine_mean": sum(u_values) / len(u_values),
                "right_singular_vector_cosine_mean": sum(right_values) / len(right_values),
            }
        )
    return result


def save_heatmaps(
    singular_rows: list[dict[str, Any]],
    cosine_rows: list[dict[str, Any]],
    plot_dir: Path,
    metrics: set[str],
    dpi: int,
) -> None:
    plt = ensure_matplotlib()
    import numpy as np

    target_dir = plot_dir / "layer_head_heatmaps"
    target_dir.mkdir(parents=True, exist_ok=True)

    heatmap_specs: list[tuple[str, tuple[Any, ...], list[dict[str, Any]]]] = []
    for row in singular_rows:
        for metric in ("top1_singular_value", "rank0_energy_fraction"):
            if metric in metrics:
                heatmap_specs.append(
                    (
                        metric,
                        (row["cache_kind"], row["cache_length"]),
                        singular_rows,
                    )
                )
    for row in cosine_rows:
        for metric in ("u_cosine_mean", "right_singular_vector_cosine_mean"):
            if metric in metrics:
                heatmap_specs.append(
                    (
                        metric,
                        (row["cache_kind"], row["left_length"], row["right_length"]),
                        cosine_rows,
                    )
                )

    seen: set[tuple[str, tuple[Any, ...]]] = set()
    for metric, group_key, source_rows in heatmap_specs:
        spec_key = (metric, group_key)
        if spec_key in seen:
            continue
        seen.add(spec_key)
        selected: list[dict[str, Any]] = []
        for row in source_rows:
            if metric not in row:
                continue
            if len(group_key) == 2:
                if (row["cache_kind"], row["cache_length"]) != group_key:
                    continue
            else:
                if (row["cache_kind"], row["left_length"], row["right_length"]) != group_key:
                    continue
            selected.append(row)
        if not selected:
            continue

        max_layer = max(int(row["layer"]) for row in selected)
        max_head = max(int(row["head"]) for row in selected)
        matrix = np.full((max_layer + 1, max_head + 1), np.nan, dtype=np.float32)
        for row in selected:
            matrix[int(row["layer"]), int(row["head"])] = float(row[metric])

        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        if len(group_key) == 2:
            title = f"{metric} {group_key[0]} len={group_key[1]}"
            stem = f"{metric}_{group_key[0]}_len_{group_key[1]}"
        else:
            title = f"{metric} {group_key[0]} {group_key[1]} vs {group_key[2]}"
            stem = f"{metric}_{group_key[0]}_{group_key[1]}_vs_{group_key[2]}"
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(target_dir / f"{stem}.png")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plot_dir = Path(args.plot_dir) if args.plot_dir else output_dir / "plots_from_csv"
    plot_dir.mkdir(parents=True, exist_ok=True)

    cache_kinds = parse_list(args.cache_kinds)
    layers = parse_index_filter(args.layers)
    heads = parse_index_filter(args.heads)
    heatmap_metrics = parse_list(args.heatmap_metrics)

    singular_rows = [
        row
        for row in parse_csv(output_dir / "singular_values.csv")
        if keep_row(row, cache_kinds, layers, heads)
    ]
    cosine_rows = [
        row
        for row in parse_csv(output_dir / "svd_vector_cosines.csv")
        if keep_row(row, cache_kinds, layers, heads)
    ]

    print(f"loaded singular rows: {len(singular_rows)}", flush=True)
    print(f"loaded cosine rows: {len(cosine_rows)}", flush=True)

    save_singular_value_plots(singular_rows, plot_dir, args.max_rank, args.plot_dpi)
    save_energy_plots(singular_rows, plot_dir, args.max_rank, args.plot_dpi)
    save_cosine_plots(cosine_rows, plot_dir, args.max_rank, args.plot_dpi)
    save_heatmaps(
        singular_heatmap_rows(singular_rows),
        cosine_heatmap_rows(cosine_rows),
        plot_dir,
        heatmap_metrics,
        args.plot_dpi,
    )
    print(f"wrote plots to: {plot_dir}", flush=True)


if __name__ == "__main__":
    main()
