#!/usr/bin/env python3
"""Summarize QKSieve mixed-bit allocations by layer, KV head, and band."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


BAND_WIDTH = 16
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more QKSieve allocations.csv files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", default="qk_balanced")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def allocation_bits(value: str) -> tuple[int, ...]:
    bits = tuple(int(item) for item in value.split("-"))
    if len(bits) != HEAD_DIM // BAND_WIDTH:
        raise ValueError(f"expected 8 bands, got {value!r}")
    return bits


def load_rows(paths: Iterable[Path], method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                if raw["method"] != method:
                    continue
                bits = allocation_bits(raw["allocation"])
                rows.append(
                    {
                        "source": path.parent.name,
                        "label": raw["label"],
                        "layer": int(raw["layer"]),
                        "kv_head": int(raw["kv_head"]),
                        "allocation": raw["allocation"],
                        "bits": bits,
                        "code_bits": int(raw["code_bits"]),
                        "metadata_bits": int(raw["metadata_bits"]),
                        "total_index_bits": int(raw["total_index_bits"]),
                    }
                )
    if not rows:
        raise ValueError(f"no rows found for method={method!r}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_head: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_head[(row["layer"], row["kv_head"])].append(row)
        by_layer[row["layer"]].append(row)

    per_head = []
    for (layer, kv_head), items in sorted(by_head.items()):
        band_means = [
            mean(item["bits"][band] for item in items) for band in range(8)
        ]
        allocations = Counter(item["allocation"] for item in items)
        per_head.append(
            {
                "layer": layer,
                "kv_head": kv_head,
                "contexts": len(items),
                "mode_allocation": allocations.most_common(1)[0][0],
                "allocation_agreement": allocations.most_common(1)[0][1]
                / len(items),
                "mean_code_bits": mean(item["code_bits"] for item in items),
                "mean_metadata_bits": mean(
                    item["metadata_bits"] for item in items
                ),
                "mean_total_index_bits": mean(
                    item["total_index_bits"] for item in items
                ),
                "mean_index_bits_per_dim": mean(
                    item["total_index_bits"] for item in items
                )
                / HEAD_DIM,
                "active_band_count": mean(
                    sum(bit > 0 for bit in item["bits"]) for item in items
                ),
                **{
                    f"band{band}_mean_bits": band_means[band]
                    for band in range(8)
                },
            }
        )

    per_layer = []
    for layer, items in sorted(by_layer.items()):
        per_layer.append(
            {
                "layer": layer,
                "kv_heads": len({item["kv_head"] for item in items}),
                "contexts": len({item["source"] for item in items}),
                "mean_index_bits_per_dim": mean(
                    item["total_index_bits"] / HEAD_DIM for item in items
                ),
                "std_index_bits_per_dim": pstdev(
                    item["total_index_bits"] / HEAD_DIM for item in items
                ),
                "active_band_count": mean(
                    sum(bit > 0 for bit in item["bits"]) for item in items
                ),
                **{
                    f"band{band}_mean_bits": mean(
                        item["bits"][band] for item in items
                    )
                    for band in range(8)
                },
            }
        )

    allocation_histogram = Counter(row["allocation"] for row in rows)
    bit_histogram = Counter(
        bit for row in rows for bit in row["bits"]
    )
    exact_context_agreement = mean(
        len({item["allocation"] for item in items}) == 1
        for items in by_head.values()
    )
    summary = {
        "method": rows[0].get("method", "qk_balanced"),
        "input_contexts": sorted({row["source"] for row in rows}),
        "layers": len(by_layer),
        "kv_heads_per_layer": sorted(
            {len({item["kv_head"] for item in items}) for items in by_layer.values()}
        ),
        "layer_head_pairs": len(by_head),
        "allocation_rows": len(rows),
        "mean_index_bits_per_dim": mean(
            row["total_index_bits"] / HEAD_DIM for row in rows
        ),
        "mean_active_bands": mean(
            sum(bit > 0 for bit in row["bits"]) for row in rows
        ),
        "exact_allocation_agreement_across_contexts": exact_context_agreement,
        "allocation_histogram": dict(allocation_histogram.most_common()),
        "band_bit_histogram": {
            str(bit): bit_histogram[bit] for bit in (0, 1, 2, 4, 8)
        },
    }
    return {
        "summary": summary,
        "per_head": per_head,
        "per_layer": per_layer,
    }


def plot(
    per_head: list[dict[str, Any]],
    per_layer: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    layers = sorted({int(row["layer"]) for row in per_head})
    heads = sorted({int(row["kv_head"]) for row in per_head})
    head_lookup = {
        (int(row["layer"]), int(row["kv_head"])): float(
            row["mean_index_bits_per_dim"]
        )
        for row in per_head
    }
    matrix = np.array(
        [[head_lookup[(layer, head)] for head in heads] for layer in layers]
    )
    fig, axis = plt.subplots(figsize=(9.2, 7.0), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xlabel("KV head")
    axis.set_ylabel("Layer")
    axis.set_title("QKSieve automatic index bits per Key dimension")
    axis.set_xticks(heads)
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("bits / Key dimension (including scale metadata)")
    fig.savefig(output_dir / "layer_head_index_bits.png", dpi=220)
    plt.close(fig)

    layer_lookup = {int(row["layer"]): row for row in per_layer}
    band_matrix = np.array(
        [
            [
                float(layer_lookup[layer][f"band{band}_mean_bits"])
                for band in range(8)
            ]
            for layer in layers
        ]
    )
    fig, axis = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    image = axis.imshow(
        band_matrix,
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=8,
    )
    axis.set_xlabel("16-D QK-balanced band")
    axis.set_ylabel("Layer")
    axis.set_title("QKSieve mean automatic bit allocation")
    axis.set_xticks(range(8), [f"{16*i}:{16*(i+1)}" for i in range(8)])
    axis.tick_params(axis="x", rotation=35)
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("mean bits / coefficient")
    fig.savefig(output_dir / "layer_band_bits.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.inputs, args.method)
    result = summarize(rows)
    result["summary"]["method"] = args.method
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_layer_head.csv", result["per_head"])
    write_csv(args.output_dir / "per_layer.csv", result["per_layer"])
    (args.output_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot(result["per_head"], result["per_layer"], args.output_dir)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
