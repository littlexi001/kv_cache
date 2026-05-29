from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import torch


FILENAME_RE = re.compile(r"layer_(?P<layer>\d+)_head_(?P<head>\d+)_cosine\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cosine value histograms from saved similarity_tensors/*.pt files."
    )
    parser.add_argument("--output_dir", required=True, help="Analysis output directory containing similarity_tensors/.")
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--min", dest="min_value", type=float, default=-1.0)
    parser.add_argument("--max", dest="max_value", type=float, default=1.0)
    return parser.parse_args()


def histogram_fieldnames() -> list[str]:
    return [
        "layer",
        "head",
        "scope",
        "bin_index",
        "bin_left",
        "bin_right",
        "bin_center",
        "count",
        "total",
        "probability",
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finite_values(values: torch.Tensor) -> torch.Tensor:
    flat = values.detach().float().reshape(-1)
    return flat[torch.isfinite(flat)]


def values_for_scope(matrix: torch.Tensor, scope: str) -> torch.Tensor:
    if scope == "all":
        return finite_values(matrix)
    if scope == "offdiag":
        values = matrix.clone()
        values.fill_diagonal_(float("nan"))
        return finite_values(values)
    raise ValueError(f"Unsupported scope: {scope}")


def histogram_counts(values: torch.Tensor, bins: int, min_value: float, max_value: float) -> tuple[torch.Tensor, int]:
    finite = finite_values(values)
    total = int(finite.numel())
    if total == 0:
        return torch.zeros(bins, dtype=torch.long), 0
    return torch.histc(finite, bins=bins, min=min_value, max=max_value).to(torch.long).cpu(), total


def histogram_rows(
    counts: torch.Tensor,
    total: int,
    bins: int,
    min_value: float,
    max_value: float,
    layer: int | str,
    head: int | str,
    scope: str,
) -> list[dict[str, Any]]:
    width = (max_value - min_value) / bins
    rows: list[dict[str, Any]] = []
    for bin_index, count in enumerate(counts.tolist()):
        left = min_value + width * bin_index
        right = min_value + width * (bin_index + 1)
        rows.append(
            {
                "layer": layer,
                "head": head,
                "scope": scope,
                "bin_index": bin_index,
                "bin_left": left,
                "bin_right": right,
                "bin_center": (left + right) / 2.0,
                "count": int(count),
                "total": total,
                "probability": (int(count) / total) if total else 0.0,
            }
        )
    return rows


def load_similarity(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "similarity" in payload:
        return payload["similarity"].float()
    if isinstance(payload, torch.Tensor):
        return payload.float()
    raise TypeError(f"Unsupported tensor payload in {path}: {type(payload)!r}")


def main() -> None:
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("--bins must be positive.")
    if args.max_value <= args.min_value:
        raise ValueError("--max must be greater than --min.")

    output_dir = Path(args.output_dir)
    tensors_dir = output_dir / "similarity_tensors"
    tensor_paths = sorted(tensors_dir.glob("layer_*_head_*_cosine.pt"))
    if not tensor_paths:
        raise FileNotFoundError(f"No saved similarity tensor files found in {tensors_dir}")

    by_head_rows: list[dict[str, Any]] = []
    global_histograms: dict[str, dict[str, Any]] = {
        "all": {"counts": torch.zeros(args.bins, dtype=torch.long), "total": 0},
        "offdiag": {"counts": torch.zeros(args.bins, dtype=torch.long), "total": 0},
    }

    for path in tensor_paths:
        match = FILENAME_RE.search(path.name)
        if not match:
            continue
        layer = int(match.group("layer"))
        head = int(match.group("head"))
        matrix = load_similarity(path)
        for scope in ("all", "offdiag"):
            counts, total = histogram_counts(values_for_scope(matrix, scope), args.bins, args.min_value, args.max_value)
            by_head_rows.extend(histogram_rows(counts, total, args.bins, args.min_value, args.max_value, layer, head, scope))
            global_histograms[scope]["counts"] += counts
            global_histograms[scope]["total"] += total

    global_rows: list[dict[str, Any]] = []
    for scope, payload in global_histograms.items():
        global_rows.extend(
            histogram_rows(
                payload["counts"],
                int(payload["total"]),
                args.bins,
                args.min_value,
                args.max_value,
                "all",
                "all",
                scope,
            )
        )

    write_csv(output_dir / "histogram_by_head.csv", by_head_rows, histogram_fieldnames())
    write_csv(output_dir / "histogram_global.csv", global_rows, histogram_fieldnames())
    print(f"wrote histogram CSVs to: {output_dir}")


if __name__ == "__main__":
    main()
