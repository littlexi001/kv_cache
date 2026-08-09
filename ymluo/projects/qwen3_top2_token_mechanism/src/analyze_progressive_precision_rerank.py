from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from analyze_numeric_pruning_frontier import (
    exact_rerank,
    grouped_scores,
    logscale16_int4_dequantize,
    parse_floats,
    quantize_query_int8,
    selected_quality,
    summarize,
)
from analyze_qk_metric_lowrank import qk_metric_factors
from analyze_qk_metric_rotation_precision import uniform4_int2_dequantize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="16")
    parser.add_argument("--train_steps", type=int, default=4)
    parser.add_argument("--test_steps", type=int, default=8)
    parser.add_argument("--test_start_step", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.5)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--prefix_rank", type=int, default=48)
    parser.add_argument("--candidate_fraction", type=float, default=0.06)
    parser.add_argument("--intermediate_fractions", default="0.025,0.03,0.04")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    layers = {int(value) for value in args.layers.split(",") if value.strip()}
    intermediate_fractions = parse_floats(args.intermediate_fractions)
    if args.prefix_rank <= 0 or args.prefix_rank >= 128 or args.prefix_rank % 16:
        raise ValueError("prefix rank must be a chunk-aligned value in (0, 128)")
    if not args.top_fraction < min(intermediate_fractions):
        raise ValueError("intermediate fractions must exceed the final fraction")
    if not max(intermediate_fractions) < args.candidate_fraction:
        raise ValueError("intermediate fractions must be below the first candidate fraction")
    if args.test_start_step < args.train_steps:
        raise ValueError("test queries must not overlap calibration queries")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    head_cases = 0

    for trace_path in args.trace_paths:
        payload = torch.load(trace_path, map_location="cpu", weights_only=False)
        records_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload["records"]:
            layer = int(record["layer"])
            if layer in layers:
                records_by_layer[layer].append(record)

        for layer, records in sorted(records_by_layer.items()):
            records.sort(key=lambda row: int(row.get("step", 0)))
            needed = args.test_start_step + args.test_steps
            if len(records) < needed:
                raise ValueError(f"{trace_path} layer {layer} has only {len(records)} steps")
            key_record = next(row for row in records if row.get("key") is not None)
            key = key_record["key"].to(device).float()[0, :, :-1]
            scaling = float(key_record["scaling"])
            kv_heads, history_count, head_dim = key.shape
            query_heads = int(records[0]["query"].shape[1])
            groups = query_heads // kv_heads
            keep_count = max(1, math.ceil(args.top_fraction * history_count))
            first_count = max(keep_count, math.ceil(args.candidate_fraction * history_count))

            sampled_key = key[:, :: args.key_sample_stride]
            key_covariance = torch.einsum(
                "hnd,hne->hde", sampled_key, sampled_key
            ) / sampled_key.shape[1]
            train_query = torch.stack(
                [
                    row["query"].to(device).float()[0, :, 0]
                    for row in records[: args.train_steps]
                ]
            ).reshape(args.train_steps, kv_heads, groups, head_dim)
            query_covariance = torch.einsum(
                "thgd,thge->hde", train_query, train_query
            ) / float(args.train_steps * groups)
            isotropic_scale = query_covariance.diagonal(
                dim1=-2, dim2=-1
            ).mean(dim=-1)
            identity = torch.eye(head_dim, device=device).unsqueeze(0)
            regularized_query_covariance = (
                (1.0 - args.query_shrinkage) * query_covariance
                + args.query_shrinkage
                * isotropic_scale[:, None, None]
                * identity
            )
            query_factor, key_factor = qk_metric_factors(
                key_covariance, regularized_query_covariance, head_dim
            )
            projected_key = torch.einsum("hnd,hdr->hnr", key, key_factor)
            prefix_key = logscale16_int4_dequantize(
                projected_key[..., : args.prefix_rank]
            )
            tail_key = uniform4_int2_dequantize(
                projected_key[..., args.prefix_rank :]
            )

            for record in records[
                args.test_start_step : args.test_start_step + args.test_steps
            ]:
                query = record["query"].to(device).float()[0, :, 0]
                exact_scores = grouped_scores(query, key) * scaling
                attention = torch.softmax(exact_scores, dim=-1)
                oracle = torch.topk(
                    exact_scores, keep_count, dim=-1, sorted=False
                ).indices
                oracle_mass = torch.gather(attention, -1, oracle).sum(dim=-1)

                projected_query = torch.einsum(
                    "hgd,hdr->hgr",
                    query.reshape(kv_heads, groups, head_dim),
                    query_factor,
                ).reshape(query_heads, head_dim)
                prefix_query = quantize_query_int8(
                    projected_query[..., : args.prefix_rank]
                )
                tail_query = quantize_query_int8(
                    projected_query[..., args.prefix_rank :]
                )
                prefix_scores = grouped_scores(prefix_query, prefix_key) * scaling
                first_candidates = torch.topk(
                    prefix_scores, first_count, dim=-1, sorted=False
                ).indices

                baseline_selected = exact_rerank(
                    exact_scores, first_candidates, keep_count
                )
                recall, mass = selected_quality(
                    baseline_selected, oracle, attention, oracle_mass
                )
                metrics["prefix48_int4_candidate6_exact2"]["top2_recall"].extend(
                    recall.cpu().tolist()
                )
                metrics["prefix48_int4_candidate6_exact2"][
                    "top2_attention_mass_recall"
                ].extend(mass.cpu().tolist())

                # Reuse the already-computed prefix score to shrink the exact
                # rerank pool. This adds no index state and isolates whether
                # random FP16 K reads can be reduced safely.
                first_candidate_scores = torch.gather(
                    prefix_scores, -1, first_candidates
                )
                for fraction in intermediate_fractions:
                    intermediate_count = max(
                        keep_count, math.ceil(fraction * history_count)
                    )
                    local = torch.topk(
                        first_candidate_scores,
                        intermediate_count,
                        dim=-1,
                        sorted=False,
                    ).indices
                    intermediate = torch.gather(first_candidates, -1, local)
                    selected = exact_rerank(
                        exact_scores, intermediate, keep_count
                    )
                    recall, mass = selected_quality(
                        selected, oracle, attention, oracle_mass
                    )
                    name = f"prefix48_int4_c6_c{fraction:g}_exact2"
                    metrics[name]["top2_recall"].extend(recall.cpu().tolist())
                    metrics[name]["top2_attention_mass_recall"].extend(
                        mass.cpu().tolist()
                    )

                tail_scores = grouped_scores(tail_query, tail_key) * scaling
                refined_scores = torch.gather(
                    prefix_scores + tail_scores, -1, first_candidates
                )
                for fraction in intermediate_fractions:
                    intermediate_count = max(
                        keep_count, math.ceil(fraction * history_count)
                    )
                    local = torch.topk(
                        refined_scores,
                        intermediate_count,
                        dim=-1,
                        sorted=False,
                    ).indices
                    intermediate = torch.gather(first_candidates, -1, local)
                    selected = exact_rerank(exact_scores, intermediate, keep_count)
                    recall, mass = selected_quality(
                        selected, oracle, attention, oracle_mass
                    )
                    name = f"prefix48_int4_c6_tail80_int2_c{fraction:g}_exact2"
                    metrics[name]["top2_recall"].extend(recall.cpu().tolist())
                    metrics[name]["top2_attention_mass_recall"].extend(
                        mass.cpu().tolist()
                    )
            head_cases += args.test_steps * query_heads
            del key, projected_key, prefix_key, tail_key
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = {
        "config": vars(args)
        | {
            "trace_paths": [str(path) for path in args.trace_paths],
            "output_path": str(args.output_path),
            "layers": sorted(layers),
            "intermediate_fractions": intermediate_fractions,
        },
        "head_cases": head_cases,
        "retrieval": {
            method: {
                metric: summarize(values)
                for metric, values in sorted(metric_values.items())
            }
            for method, metric_values in sorted(metrics.items())
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
