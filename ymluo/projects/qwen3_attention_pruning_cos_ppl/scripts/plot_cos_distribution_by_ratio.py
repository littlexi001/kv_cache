from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RATIOS = "0.001,0.005,0.01,0.02,0.04,0.06,0.08,0.10,0.15,0.20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cosine-value distributions from cos_per_token.csv by keep ratio."
    )
    parser.add_argument("--csv_path", required=True, help="Path to cos_per_token.csv.")
    parser.add_argument("--output_dir", default="", help="Default: <csv parent>/plots/cos_distribution_by_ratio.")
    parser.add_argument("--ratios", default=DEFAULT_RATIOS)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--hist_min", type=float, default=0.0)
    parser.add_argument("--hist_max", type=float, default=1.0)
    parser.add_argument("--density", action="store_true", help="Plot density instead of raw counts.")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_ratios(spec: str) -> list[float]:
    ratios = [float(part.strip()) for part in spec.split(",") if part.strip()]
    if not ratios:
        raise ValueError("--ratios cannot be empty.")
    return ratios


def ratio_key(value: float) -> str:
    return f"{value:.12g}"


def ratio_label(value: float) -> str:
    return f"{100.0 * value:g}%"


def init_stats(ratios: list[float]) -> dict[str, dict[str, Any]]:
    return {
        ratio_key(ratio): {
            "ratio": ratio,
            "count": 0,
            "sum": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
            "values": [],
        }
        for ratio in ratios
    }


def read_values(csv_path: Path, ratios: list[float]) -> dict[str, dict[str, Any]]:
    stats = init_stats(ratios)
    wanted = set(stats)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "ratio" not in reader.fieldnames or "cosine" not in reader.fieldnames:
            raise ValueError("CSV must contain 'ratio' and 'cosine' columns.")
        for row in reader:
            key = ratio_key(float(row["ratio"]))
            if key not in wanted:
                continue
            value = float(row["cosine"])
            item = stats[key]
            item["count"] += 1
            item["sum"] += value
            item["min"] = min(item["min"], value)
            item["max"] = max(item["max"], value)
            item["values"].append(value)
    return stats


def write_summary(output_dir: Path, stats: dict[str, dict[str, Any]]) -> None:
    fields = [
        "ratio",
        "kept_percent",
        "count",
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
        "max",
    ]
    rows: list[dict[str, Any]] = []
    for item in sorted(stats.values(), key=lambda value: value["ratio"]):
        values = np.asarray(item["values"], dtype=np.float64)
        if values.size == 0:
            rows.append({"ratio": item["ratio"], "kept_percent": 100.0 * item["ratio"], "count": 0})
            continue
        rows.append(
            {
                "ratio": item["ratio"],
                "kept_percent": 100.0 * item["ratio"],
                "count": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "p01": float(np.quantile(values, 0.01)),
                "p05": float(np.quantile(values, 0.05)),
                "p25": float(np.quantile(values, 0.25)),
                "p50": float(np.quantile(values, 0.50)),
                "p75": float(np.quantile(values, 0.75)),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
                "max": float(values.max()),
            }
        )
    with (output_dir / "cos_distribution_summary_by_ratio.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_histograms(
    output_dir: Path,
    stats: dict[str, dict[str, Any]],
    bins: int,
    hist_range: tuple[float, float],
    density: bool,
    dpi: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for item in sorted(stats.values(), key=lambda value: value["ratio"]):
        values = np.asarray(item["values"], dtype=np.float64)
        if values.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
        ax.hist(values, bins=bins, range=hist_range, density=density, color="#2f78b7", alpha=0.85)
        ax.axvline(values.mean(), color="#d62728", linewidth=1.2, label=f"mean={values.mean():.4f}")
        ax.axvline(np.quantile(values, 0.50), color="#111111", linewidth=1.0, linestyle="--", label=f"p50={np.quantile(values, 0.50):.4f}")
        ax.axvline(np.quantile(values, 0.05), color="#888888", linewidth=0.9, linestyle=":", label=f"p05={np.quantile(values, 0.05):.4f}")
        ax.set_xlim(hist_range)
        ax.set_xlabel("Cosine")
        ax.set_ylabel("Density" if density else "Count")
        ax.set_title(f"Cosine distribution, keep top {ratio_label(item['ratio'])}")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = output_dir / f"cos_distribution_ratio_{ratio_key(item['ratio']).replace('.', 'p')}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
        ax.hist(values, bins=bins, range=hist_range, density=density, color="#2f78b7", alpha=0.85)
        ax.set_yscale("log")
        ax.axvline(values.mean(), color="#d62728", linewidth=1.2, label=f"mean={values.mean():.4f}")
        ax.axvline(np.quantile(values, 0.50), color="#111111", linewidth=1.0, linestyle="--", label=f"p50={np.quantile(values, 0.50):.4f}")
        ax.axvline(np.quantile(values, 0.05), color="#888888", linewidth=0.9, linestyle=":", label=f"p05={np.quantile(values, 0.05):.4f}")
        ax.set_xlim(hist_range)
        ax.set_xlabel("Cosine")
        ax.set_ylabel("Density" if density else "Count")
        ax.set_title(f"Cosine distribution, keep top {ratio_label(item['ratio'])} (log y)")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = output_dir / f"cos_distribution_ratio_{ratio_key(item['ratio']).replace('.', 'p')}_logy.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    ratios = parse_ratios(args.ratios)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / "plots" / "cos_distribution_by_ratio"
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = read_values(csv_path, ratios)
    write_summary(output_dir, stats)
    paths = plot_histograms(
        output_dir,
        stats,
        args.bins,
        (args.hist_min, args.hist_max),
        args.density,
        args.dpi,
    )
    print(f"wrote {len(paths)} plots to: {output_dir}")


if __name__ == "__main__":
    main()
