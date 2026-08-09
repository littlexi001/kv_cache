from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from analyze_hierarchical_spectral_quantization_20260727 import (
    covariance_basis,
    quantize_groupwise,
    query_int8,
    selection_metrics,
    sign_reconstruct,
)


HEAD_DIM = 128
GROUP_SIZE = 16
GROUP_COUNT = HEAD_DIM // GROUP_SIZE
FULL_KV_BITS = 2 * HEAD_DIM * 16
BIT_LEVELS = (1, 2, 4, 8)
ZERO_BIT_LEVELS = (0, 1, 2, 4, 8)


def parse_floats(value: str) -> list[float]:
    result = sorted({float(item) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("expected at least one floating-point value")
    return result


def parse_ints(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("expected at least one integer")
    return result


def parse_optional_ints(value: str) -> list[int]:
    return sorted({int(item) for item in value.split(",") if item.strip()})


def quantize_band(values: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 0:
        return torch.zeros_like(values)
    if bits == 1:
        return sign_reconstruct(values, group_size=GROUP_SIZE)
    return quantize_groupwise(values, bits, group_size=GROUP_SIZE)


def covariance(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    return values.transpose(0, 1) @ values / max(1, values.shape[0])


def distortion_table(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
    bit_levels: tuple[int, ...],
) -> tuple[list[dict[int, torch.Tensor]], list[dict[int, torch.Tensor]]]:
    key_tables: list[dict[int, torch.Tensor]] = []
    query_tables: list[dict[int, torch.Tensor]] = []
    for group_index in range(GROUP_COUNT):
        start = group_index * GROUP_SIZE
        stop = start + GROUP_SIZE
        key_band = coefficients[:, start:stop]
        query_band = calibration_queries[:, start:stop]
        query_covariance = covariance(query_band)
        key_costs: dict[int, torch.Tensor] = {}
        query_costs: dict[int, torch.Tensor] = {}
        for bits in bit_levels:
            residual = key_band - quantize_band(key_band, bits)
            residual_covariance = covariance(residual)
            key_costs[bits] = residual.float().square().mean()
            query_costs[bits] = (
                residual_covariance * query_covariance.transpose(0, 1)
            ).sum()
        key_tables.append(key_costs)
        query_tables.append(query_costs)
    return key_tables, query_tables


def allocate_bits(
    distortion: list[dict[int, torch.Tensor]],
    bit_budget_per_coordinate: int,
    bit_levels: tuple[int, ...],
    include_scale_metadata: bool = False,
) -> tuple[int, ...]:
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for group_costs in distortion:
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used_bits, (cost, allocation) in states.items():
            for bits in bit_levels:
                rate_units = bits + (
                    1 if include_scale_metadata and bits > 0 else 0
                )
                new_bits = used_bits + rate_units
                if new_bits > bit_budget_per_coordinate:
                    continue
                new_cost = cost + float(group_costs[bits].item())
                current = updated.get(new_bits)
                if current is None or new_cost < current[0]:
                    updated[new_bits] = (new_cost, allocation + (bits,))
        states = updated
    feasible = [
        (cost, -used_bits, allocation)
        for used_bits, (cost, allocation) in states.items()
    ]
    if not feasible:
        raise RuntimeError("no feasible spectral allocation")
    return min(feasible)[2]


def reconstruct(
    bands: list[dict[int, torch.Tensor]],
    allocation: tuple[int, ...],
) -> torch.Tensor:
    return torch.cat(
        [bands[index][bits] for index, bits in enumerate(allocation)],
        dim=-1,
    )


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
    }


def aggregate(
    rows: list[dict[str, Any]],
    allocations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    allocation_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in allocations:
        allocation_by_method[str(row["method"])].append(row)
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["method"]), float(row["selected_fraction_target"]))
        ].append(row)
    output = []
    for (method, fraction), items in sorted(grouped.items()):
        method_allocations = allocation_by_method[method]
        code_bits = sum(
            int(row["code_bits"]) for row in method_allocations
        ) / len(method_allocations)
        metadata_bits = sum(
            int(row["metadata_bits"]) for row in method_allocations
        ) / len(method_allocations)
        result: dict[str, Any] = {
            "method": method,
            "selected_fraction_target": fraction,
            "cases": len(items),
            "calibration_steps": int(items[0]["calibration_steps"]),
            "code_bits_mean": code_bits,
            "metadata_bits_mean": metadata_bits,
            "total_index_bits_mean": code_bits + metadata_bits,
            "index_ratio_of_full_kv": (
                code_bits + metadata_bits
            )
            / FULL_KV_BITS,
        }
        for field in (
            "top2_recall",
            "selected_attention_mass",
            "top2_attention_mass_recall",
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
            "Choose per-head spectral precision by discrete rate-distortion "
            "optimization and evaluate on held-out decode queries."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--basis_tokens", type=int, default=0)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--bit_budgets", default="21,26")
    parser.add_argument(
        "--total_rate_budgets",
        default="",
        help=(
            "Optional budgets in 16-bit units, counting both quantization "
            "codes and one FP16 scale for every enabled spectral band."
        ),
    )
    parser.add_argument("--selected_fractions", default="0.02,0.03,0.04,0.06")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--allow_zero_bits", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    bit_budgets = parse_ints(args.bit_budgets)
    total_rate_budgets = parse_optional_ints(args.total_rate_budgets)
    selected_fractions = parse_floats(args.selected_fractions)
    bit_levels = ZERO_BIT_LEVELS if args.allow_zero_bits else BIT_LEVELS
    if args.calibration_steps < 0:
        raise ValueError("calibration steps cannot be negative")
    minimum_budget = 0 if args.allow_zero_bits else GROUP_COUNT
    if min(bit_budgets) < minimum_budget or max(bit_budgets) > 8 * GROUP_COUNT:
        raise ValueError("bit budgets must fit the available group precisions")
    if total_rate_budgets and (
        min(total_rate_budgets) < 0
        or max(total_rate_budgets) > 9 * GROUP_COUNT
    ):
        raise ValueError("total-rate budgets must be between 0 and 72")

    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("trace contains no records")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[int(record["layer"])].append(record)

    rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    for layer, layer_records in sorted(by_layer.items()):
        if len(layer_records) <= args.calibration_steps:
            raise ValueError(
                f"layer {layer} does not have held-out records after calibration"
            )
        raw_key = next(
            (record.get("key") for record in layer_records if record.get("key") is not None),
            None,
        )
        if raw_key is None:
            raise ValueError(f"layer {layer} has no key state")
        all_key = raw_key.to(device).float()[0]
        history_count = int(all_key.shape[1]) - 1
        key = all_key[:, :history_count]
        kv_heads = int(key.shape[0])
        first_query = layer_records[0]["query"]
        query_heads = int(first_query.shape[1])
        groups = query_heads // kv_heads

        if args.calibration_steps > 0:
            calibration_queries = torch.stack(
                [
                    record["query"].to(device).float()[0, :, 0, :]
                    for record in layer_records[: args.calibration_steps]
                ],
                dim=0,
            )
        else:
            calibration_queries = torch.empty(
                (0, query_heads, HEAD_DIM),
                dtype=torch.float32,
                device=device,
            )
        prepared = []
        for kv_head in range(kv_heads):
            head_key = key[kv_head]
            basis_limit = (
                history_count
                if args.basis_tokens <= 0
                else min(history_count, args.basis_tokens)
            )
            basis, _ = covariance_basis(
                head_key[:basis_limit: args.sample_stride]
            )
            coefficients = head_key @ basis
            projected_calibration = (
                calibration_queries[
                    :, kv_head * groups : (kv_head + 1) * groups
                ]
                @ basis
            ).reshape(-1, HEAD_DIM)
            key_distortion, query_distortion = distortion_table(
                coefficients, projected_calibration, bit_levels
            )
            quantized_bands = []
            for group_index in range(GROUP_COUNT):
                start = group_index * GROUP_SIZE
                stop = start + GROUP_SIZE
                band = coefficients[:, start:stop]
                quantized_bands.append(
                    {
                        bits: quantize_band(band, bits)
                        for bits in bit_levels
                    }
                )

            method_allocations: dict[str, tuple[int, ...]] = {
                "fixed_group841": (8, 4, 4, 1, 1, 1, 1, 1),
                "fixed_group842": (8, 4, 4, 2, 2, 2, 2, 2),
                "uniform_int4": (4,) * GROUP_COUNT,
            }
            for budget in bit_budgets:
                method_allocations[f"auto_key_b{budget}"] = allocate_bits(
                    key_distortion, budget, bit_levels
                )
                if args.calibration_steps > 0:
                    method_allocations[f"auto_qmse_b{budget}"] = allocate_bits(
                        query_distortion, budget, bit_levels
                    )
            for budget in total_rate_budgets:
                method_allocations[
                    f"auto_key_total_b{budget}"
                ] = allocate_bits(
                    key_distortion,
                    budget,
                    bit_levels,
                    include_scale_metadata=True,
                )
                if args.calibration_steps > 0:
                    method_allocations[
                        f"auto_qmse_total_b{budget}"
                    ] = allocate_bits(
                        query_distortion,
                        budget,
                        bit_levels,
                        include_scale_metadata=True,
                    )

            reconstructed = {
                method: reconstruct(quantized_bands, allocation)
                for method, allocation in method_allocations.items()
            }
            for method, allocation in method_allocations.items():
                allocation_rows.append(
                    {
                        "label": args.label,
                        "layer": layer,
                        "kv_head": kv_head,
                        "method": method,
                        "allocation": "-".join(map(str, allocation)),
                        **{
                            f"group{index}_bits": bits
                            for index, bits in enumerate(allocation)
                        },
                        "code_bits": GROUP_SIZE * sum(allocation),
                        "metadata_bits": 16
                        * sum(bits > 0 for bits in allocation),
                    }
                )
            prepared.append(
                {
                    "head_key": head_key,
                    "basis": basis,
                    "reconstructed": reconstructed,
                }
            )

        for heldout_index, record in enumerate(
            layer_records[args.calibration_steps :],
            start=args.calibration_steps,
        ):
            query = record["query"].to(device).float()[0, :, 0, :]
            scaling = float(record["scaling"])
            top_count = max(1, math.ceil(args.top_fraction * history_count))
            for kv_head in range(kv_heads):
                state = prepared[kv_head]
                for group in range(groups):
                    query_head = kv_head * groups + group
                    head_query = query[query_head]
                    projected_query = query_int8(head_query @ state["basis"])
                    exact_scores = state["head_key"] @ head_query * scaling
                    attention = torch.softmax(exact_scores, dim=-1)
                    true_top = torch.topk(exact_scores, k=top_count).indices
                    for method, reconstructed_key in state[
                        "reconstructed"
                    ].items():
                        approximate_scores = (
                            reconstructed_key @ projected_query
                        ) * scaling
                        for fraction in selected_fractions:
                            metrics = selection_metrics(
                                exact_scores,
                                attention,
                                approximate_scores,
                                true_top,
                                fraction,
                            )
                            rows.append(
                                {
                                    "label": args.label,
                                    "layer": layer,
                                    "heldout_step": heldout_index,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "method": method,
                                    "selected_fraction_target": fraction,
                                    "calibration_steps": args.calibration_steps,
                                    **metrics,
                                }
                            )
        print(
            json.dumps(
                {
                    "label": args.label,
                    "layer": layer,
                    "layers_complete": sum(
                        int(done_layer <= layer) for done_layer in by_layer
                    ),
                    "layers": len(by_layer),
                    "rows": len(rows),
                }
            ),
            flush=True,
        )

    summary = aggregate(rows, allocation_rows)
    allocation_histograms = {}
    for method in sorted({str(row["method"]) for row in allocation_rows}):
        counter = Counter(
            str(row["allocation"])
            for row in allocation_rows
            if row["method"] == method
        )
        allocation_histograms[method] = dict(counter.most_common())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head.csv", rows)
    write_csv(args.output_dir / "allocations.csv", allocation_rows)
    write_csv(args.output_dir / "summary.csv", summary)
    output = {
        "config": {
            "trace_path": str(args.trace_path),
            "label": args.label,
            "basis_tokens": args.basis_tokens,
            "calibration_steps": args.calibration_steps,
            "bit_budgets": bit_budgets,
            "total_rate_budgets": total_rate_budgets,
            "selected_fractions": selected_fractions,
            "allow_zero_bits": args.allow_zero_bits,
            "bit_levels": bit_levels,
        },
        "allocation_histograms": allocation_histograms,
        "methods": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
