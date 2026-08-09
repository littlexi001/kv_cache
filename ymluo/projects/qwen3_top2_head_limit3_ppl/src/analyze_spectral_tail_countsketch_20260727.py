from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from analyze_hierarchical_spectral_quantization_20260727 import (
    covariance_basis,
    quantize_groupwise,
    query_int8,
    sign_reconstruct,
)


HEAD_DIM = 128
FULL_KV_BITS = 2 * HEAD_DIM * 16
CORE_CODE_BITS = 16 * 8 + 32 * 4
CORE_METADATA_BITS = 3 * 16


def parse_ints(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("expected at least one integer")
    return result


def parse_floats(value: str) -> list[float]:
    result = sorted({float(item) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("expected at least one float")
    return result


def countsketch_matrix(
    input_dim: int,
    buckets: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bucket = torch.randint(buckets, (input_dim,), generator=generator)
    sign = torch.randint(2, (input_dim,), generator=generator) * 2 - 1
    matrix = torch.zeros(input_dim, buckets, dtype=torch.float32)
    matrix[torch.arange(input_dim), bucket] = sign.float()
    return matrix.to(device)


def quantize_sketch(values: torch.Tensor, bits: int) -> torch.Tensor:
    width = int(values.shape[-1])
    if bits == 1:
        return sign_reconstruct(values, group_size=width)
    if bits in {2, 4}:
        return quantize_groupwise(values, bits, group_size=width)
    raise ValueError("sketch bits must be 1, 2, or 4")


def selection_metrics(
    exact_scores: torch.Tensor,
    attention: torch.Tensor,
    approximate_scores: torch.Tensor,
    true_top: torch.Tensor,
    selected_fraction: float,
) -> dict[str, float]:
    selected_count = min(
        exact_scores.numel(),
        max(1, math.ceil(selected_fraction * exact_scores.numel())),
    )
    selected = torch.topk(approximate_scores, k=selected_count).indices
    mask = torch.zeros_like(exact_scores, dtype=torch.bool)
    mask[selected] = True
    exact_centered = exact_scores - exact_scores.mean()
    approximate_centered = approximate_scores - approximate_scores.mean()
    denominator = torch.linalg.vector_norm(
        exact_centered
    ) * torch.linalg.vector_norm(approximate_centered)
    pearson = (
        float(
            (
                exact_centered @ approximate_centered / denominator
            ).item()
        )
        if float(denominator.item()) > 0.0
        else 0.0
    )
    return {
        "selected_fraction": selected_count / exact_scores.numel(),
        "top2_recall": float(mask[true_top].float().mean().item()),
        "attention_mass": float(attention[selected].sum().item()),
        "score_pearson": pearson,
    }


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p10": float(torch.quantile(tensor, 0.1).item()),
        "p50": float(torch.quantile(tensor, 0.5).item()),
        "p90": float(torch.quantile(tensor, 0.9).item()),
        "minimum": float(tensor.min().item()),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["method"]), float(row["selected_fraction_target"]))
        ].append(row)
    output: list[dict[str, Any]] = []
    for (method, fraction), items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "method": method,
            "selected_fraction_target": fraction,
            "cases": len(items),
            "code_bits": float(items[0]["code_bits"]),
            "metadata_bits": float(items[0]["metadata_bits"]),
            "total_index_bits": float(items[0]["code_bits"])
            + float(items[0]["metadata_bits"]),
            "index_ratio_of_full_kv": (
                float(items[0]["code_bits"])
                + float(items[0]["metadata_bits"])
            )
            / FULL_KV_BITS,
        }
        for field in (
            "top2_recall",
            "attention_mass",
            "score_pearson",
        ):
            stats = summarize(float(item[field]) for item in items)
            result.update(
                {f"{field}_{name}": value for name, value in stats.items()}
            )
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate sub-one-bit-per-tail-dimension CountSketch residual "
            "codes on real per-head Q/K traces."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--bucket_counts", default="4,8,16,32")
    parser.add_argument("--sketch_bits", default="1,2,4")
    parser.add_argument("--seeds", default="20260727,20260728,20260729")
    parser.add_argument("--selected_fractions", default="0.02,0.03")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--max_records", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    bucket_counts = parse_ints(args.bucket_counts)
    sketch_bits = parse_ints(args.sketch_bits)
    seeds = parse_ints(args.seeds)
    selected_fractions = parse_floats(args.selected_fractions)
    if any(count <= 0 for count in bucket_counts):
        raise ValueError("bucket counts must be positive")
    if any(bits not in {1, 2, 4} for bits in sketch_bits):
        raise ValueError("sketch bits must be 1, 2, or 4")

    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", []))
    if args.max_records:
        records = records[: args.max_records]
    if not records:
        raise ValueError("trace contains no records")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    matrices = {
        (buckets, seed): countsketch_matrix(
            80, buckets, seed, device
        )
        for buckets in bucket_counts
        for seed in seeds
    }
    states: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for record_index, record in enumerate(records):
        layer = int(record["layer"])
        raw_key = record.get("key")
        if layer not in states:
            if raw_key is None:
                raise RuntimeError(f"layer {layer} is missing its initial key")
            all_key = raw_key.to(device).float()[0]
            history_count = int(all_key.shape[1]) - 1
            key = all_key[:, :history_count]
            prepared: list[dict[str, Any]] = []
            for kv_head in range(key.shape[0]):
                head_key = key[kv_head]
                basis, _ = covariance_basis(head_key[:: args.sample_stride])
                coefficients = head_key @ basis
                core_key = torch.cat(
                    (
                        quantize_groupwise(coefficients[:, :16], 8),
                        quantize_groupwise(coefficients[:, 16:48], 4),
                    ),
                    dim=-1,
                )
                tail = coefficients[:, 48:]
                sketches: dict[tuple[int, int, int], torch.Tensor] = {}
                for (buckets, seed), matrix in matrices.items():
                    exact_sketch = tail @ matrix
                    for bits in sketch_bits:
                        sketches[(buckets, seed, bits)] = quantize_sketch(
                            exact_sketch, bits
                        )
                prepared.append(
                    {
                        "head_key": head_key,
                        "basis": basis,
                        "core_key": core_key,
                        "tail": tail,
                        "sketches": sketches,
                    }
                )
            states[layer] = {
                "history_count": history_count,
                "kv_heads": int(key.shape[0]),
                "prepared": prepared,
            }

        state = states[layer]
        history_count = int(state["history_count"])
        kv_heads = int(state["kv_heads"])
        query = record["query"].to(device).float()[0, :, 0, :]
        query_heads = int(query.shape[0])
        groups = query_heads // kv_heads
        scaling = float(record["scaling"])
        top_count = max(1, math.ceil(args.top_fraction * history_count))

        for kv_head in range(kv_heads):
            prepared = state["prepared"][kv_head]
            for group in range(groups):
                query_head = kv_head * groups + group
                projected_query = query[query_head] @ prepared["basis"]
                approximate_query = query_int8(projected_query)
                exact_scores = (
                    prepared["head_key"] @ query[query_head]
                ) * scaling
                attention = torch.softmax(exact_scores, dim=-1)
                true_top = torch.topk(exact_scores, k=top_count).indices
                core_scores = (
                    prepared["core_key"] @ approximate_query[:48]
                ) * scaling

                methods: dict[str, tuple[torch.Tensor, int, int]] = {
                    "core_only": (
                        core_scores,
                        CORE_CODE_BITS,
                        CORE_METADATA_BITS,
                    )
                }
                for (buckets, seed), matrix in matrices.items():
                    query_sketch = projected_query[48:] @ matrix
                    for bits in sketch_bits:
                        sketch_scores = (
                            prepared["sketches"][(buckets, seed, bits)]
                            @ query_sketch
                        ) * scaling
                        methods[
                            f"countsketch_m{buckets}_b{bits}_s{seed}"
                        ] = (
                            core_scores + sketch_scores,
                            CORE_CODE_BITS + buckets * bits,
                            CORE_METADATA_BITS + 16,
                        )

                for method, (
                    approximate_scores,
                    code_bits,
                    metadata_bits,
                ) in methods.items():
                    for selected_fraction in selected_fractions:
                        metrics = selection_metrics(
                            exact_scores,
                            attention,
                            approximate_scores,
                            true_top,
                            selected_fraction,
                        )
                        rows.append(
                            {
                                "label": args.label,
                                "record": record_index,
                                "layer": layer,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "method": method,
                                "selected_fraction_target": selected_fraction,
                                "code_bits": code_bits,
                                "metadata_bits": metadata_bits,
                                **metrics,
                            }
                        )

        print(
            json.dumps(
                {
                    "label": args.label,
                    "record": record_index + 1,
                    "records": len(records),
                    "rows": len(rows),
                }
            ),
            flush=True,
        )

    summary = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    payload = {
        "config": {
            "trace_path": str(args.trace_path),
            "label": args.label,
            "bucket_counts": bucket_counts,
            "sketch_bits": sketch_bits,
            "seeds": seeds,
            "selected_fractions": selected_fractions,
        },
        "methods": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
