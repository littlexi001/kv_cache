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
    selection_metrics,
)


HEAD_DIM = 128
FULL_KV_BITS = 2 * HEAD_DIM * 16


def parse_floats(value: str) -> list[float]:
    result = sorted({float(item) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("expected at least one floating-point value")
    return result


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_haar_rotation(
    dimension: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(
        dimension,
        dimension,
        dtype=torch.float64,
        generator=generator,
    )
    orthogonal, triangular = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(triangular)).clamp_min(0.0) * 2.0 - 1.0
    return (orthogonal * signs).float().to(device)


def asymmetric_int4_reconstruct(values: torch.Tensor) -> torch.Tensor:
    lower = values.amin()
    upper = values.amax()
    step = ((upper - lower) / 15.0).clamp_min(1.0e-8)
    code = torch.round((values - lower) / step).clamp(0.0, 15.0)
    return code * step + lower


def prepare_rabitq_head(
    key: torch.Tensor,
    rotation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    key_centroid = key.mean(dim=0)
    centered = key - key_centroid
    norms = torch.linalg.vector_norm(centered, dim=-1).clamp_min(1.0e-8)
    rotated_unit = (centered / norms[:, None]) @ rotation.transpose(0, 1)
    codeword = torch.where(
        rotated_unit >= 0.0,
        torch.ones_like(rotated_unit),
        -torch.ones_like(rotated_unit),
    ) / math.sqrt(key.shape[-1])
    alpha = (rotated_unit * codeword).sum(dim=-1).clamp_min(1.0e-8)
    return {
        "key": key,
        "key_centroid": key_centroid,
        "norms": norms,
        "codeword": codeword,
        "alpha": alpha,
    }


def rabitq_scores(
    state: dict[str, torch.Tensor],
    query: torch.Tensor,
    query_centroid: torch.Tensor,
    rotation: torch.Tensor,
    quantize_query: bool,
) -> torch.Tensor:
    centered_query = query - query_centroid
    rotated_query = centered_query @ rotation.transpose(0, 1)
    if quantize_query:
        rotated_query = asymmetric_int4_reconstruct(rotated_query)
    normalized_estimate = (
        state["codeword"] @ rotated_query
    ) / state["alpha"]
    centered_term = state["norms"] * normalized_estimate
    query_dot_key_centroid = query @ state["key_centroid"]
    centroid_dot_key = state["key"] @ query_centroid
    centroid_dot_centroid = query_centroid @ state["key_centroid"]
    return (
        centered_term
        + query_dot_key_centroid
        + centroid_dot_key
        - centroid_dot_centroid
    )


def aggregate(
    rows: list[dict[str, Any]],
    groups_per_kv_head: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["method"]), float(row["selected_fraction_target"]))
        ].append(row)

    # The paper reports only the binary code and alpha. The official score
    # formula additionally reads the centered-key norm and c_q dot k.
    paper_bits = HEAD_DIM + 16
    executable_bits = paper_bits + 16 + 16 * groups_per_kv_head
    output = []
    for (method, fraction), items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "method": method,
            "selected_fraction_target": fraction,
            "cases": len(items),
            "calibration_steps": int(items[0]["calibration_steps"]),
            "paper_index_bits_per_kv_token": paper_bits,
            "paper_index_ratio_of_full_kv": paper_bits / FULL_KV_BITS,
            "formula_state_bits_per_kv_token": executable_bits,
            "formula_state_ratio_of_full_kv": executable_bits / FULL_KV_BITS,
            "groups_per_kv_head": groups_per_kv_head,
        }
        for field in (
            "top2_recall",
            "selected_attention_mass",
            "top2_attention_mass_recall",
            "score_pearson",
            "score_rmse",
        ):
            stats = summarize(float(item[field]) for item in items)
            result.update(
                {f"{field}_{name}": value for name, value in stats.items()}
            )
        output.append(result)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay Q/K traces with the official RaBitQ score decomposition "
            "for a same-trace comparison against the spectral index."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument(
        "--selected_fractions",
        default="0.02,0.03,0.04,0.06",
    )
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    selected_fractions = parse_floats(args.selected_fractions)
    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("trace contains no records")
    if args.calibration_steps <= 0:
        raise ValueError("RaBitQ comparison requires calibration queries")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[int(record["layer"])].append(record)

    rows: list[dict[str, Any]] = []
    groups_per_kv_head = 0
    for layer, layer_records in sorted(by_layer.items()):
        layer_records.sort(key=lambda record: int(record["step"]))
        if len(layer_records) <= args.calibration_steps:
            raise ValueError(
                f"layer {layer} does not have held-out records after calibration"
            )
        raw_key = next(
            (
                record.get("key")
                for record in layer_records
                if record.get("key") is not None
            ),
            None,
        )
        if raw_key is None:
            raise ValueError(f"layer {layer} has no key state")

        all_key = raw_key.to(device).float()[0]
        history_count = int(all_key.shape[1]) - 1
        key = all_key[:, :history_count]
        kv_heads = int(key.shape[0])
        query_heads = int(layer_records[0]["query"].shape[1])
        groups_per_kv_head = query_heads // kv_heads
        if groups_per_kv_head * kv_heads != query_heads:
            raise ValueError("query heads must be divisible by KV heads")

        calibration_queries = torch.stack(
            [
                record["query"].to(device).float()[0, :, 0, :]
                for record in layer_records[: args.calibration_steps]
            ],
            dim=0,
        )
        query_centroids = calibration_queries.mean(dim=0)
        rotation = make_haar_rotation(
            HEAD_DIM,
            args.seed + layer,
            device,
        )
        prepared = [
            prepare_rabitq_head(key[kv_head], rotation)
            for kv_head in range(kv_heads)
        ]

        top_count = max(1, math.ceil(args.top_fraction * history_count))
        for heldout_index, record in enumerate(
            layer_records[args.calibration_steps :],
            start=args.calibration_steps,
        ):
            query = record["query"].to(device).float()[0, :, 0, :]
            scaling = float(record["scaling"])
            for kv_head, state in enumerate(prepared):
                for group in range(groups_per_kv_head):
                    query_head = kv_head * groups_per_kv_head + group
                    head_query = query[query_head]
                    query_centroid = query_centroids[query_head]
                    exact_scores = state["key"] @ head_query * scaling
                    attention = torch.softmax(exact_scores, dim=-1)
                    true_top = torch.topk(exact_scores, k=top_count).indices
                    for method, quantize_query in (
                        ("rabitq_official_fp_query", False),
                        ("rabitq_paper_int4_query", True),
                    ):
                        approximate_scores = rabitq_scores(
                            state,
                            head_query,
                            query_centroid,
                            rotation,
                            quantize_query,
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
                    "layers": len(by_layer),
                    "rows": len(rows),
                }
            ),
            flush=True,
        )

    summary = aggregate(rows, groups_per_kv_head)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    output = {
        "config": {
            "trace_path": str(args.trace_path),
            "label": args.label,
            "calibration_steps": args.calibration_steps,
            "selected_fractions": selected_fractions,
            "top_fraction": args.top_fraction,
            "seed": args.seed,
            "centroid_note": (
                "The trace lacks prefill Q states, so c_q is estimated from "
                "the first calibration queries and frozen before held-out steps."
            ),
            "storage_note": (
                "paper_index counts sign codes plus alpha; formula_state also "
                "counts key norms and one c_q dot k scalar per GQA query head."
            ),
        },
        "methods": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
