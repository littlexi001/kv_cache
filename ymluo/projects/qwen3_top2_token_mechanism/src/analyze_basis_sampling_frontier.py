from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from analyze_numeric_pruning_frontier import (
    exact_rerank,
    grouped_scores,
    logscale16_int4_dequantize,
    parse_floats,
    parse_ints,
    quantize_query_int8,
    selected_quality,
    summarize,
)


def second_moment(sample: torch.Tensor) -> torch.Tensor:
    return torch.einsum("hnd,hne->hde", sample, sample) / float(sample.shape[1])


def block_max_norm_sample(key: torch.Tensor, block_size: int) -> torch.Tensor:
    heads, tokens, dim = key.shape
    block_count = math.ceil(tokens / block_size)
    padded_tokens = block_count * block_size
    if padded_tokens != tokens:
        key = torch.cat(
            (key, key[:, -1:].expand(-1, padded_tokens - tokens, -1)), dim=1
        )
    blocks = key.reshape(heads, block_count, block_size, dim)
    local = blocks.square().sum(dim=-1).argmax(dim=-1)
    return torch.gather(
        blocks,
        2,
        local[..., None, None].expand(-1, -1, 1, dim),
    ).squeeze(2)


def covariance_for_method(key: torch.Tensor, method: str) -> torch.Tensor:
    if method.startswith("stride"):
        stride = int(method.removeprefix("stride"))
        return second_moment(key[:, ::stride])
    if method.startswith("blockmax"):
        block_size = int(method.removeprefix("blockmax"))
        return second_moment(block_max_norm_sample(key, block_size))
    if method.startswith("hybrid"):
        block_size = int(method.removeprefix("hybrid"))
        uniform = key[:, ::block_size]
        high_norm = block_max_norm_sample(key, block_size)
        return 0.5 * second_moment(uniform) + 0.5 * second_moment(high_norm)
    if method.startswith("normmix"):
        alpha = int(method.removeprefix("normmix")) / 100.0
        uniform = key[:, ::32]
        sample_count = uniform.shape[1]
        top_norm = torch.topk(
            key.square().sum(dim=-1), sample_count, dim=-1, sorted=False
        ).indices
        top_key = torch.gather(
            key, 1, top_norm[..., None].expand(-1, -1, key.shape[-1])
        )
        return (1.0 - alpha) * second_moment(uniform) + alpha * second_moment(
            top_key
        )
    if method == "full":
        return second_moment(key)
    raise ValueError(f"unknown basis method: {method}")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="16")
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--candidate_fractions", default="0.04,0.05,0.06,0.08")
    parser.add_argument(
        "--methods",
        default="stride32,stride16,stride8,full,blockmax32,hybrid64,normmix05,normmix10",
    )
    parser.add_argument("--top_fraction", type=float, default=0.02)
    args = parser.parse_args()

    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    candidate_fractions = parse_floats(args.candidate_fractions)
    layers = set(parse_ints(args.layers))
    if args.rank % 16 or args.rank > 128:
        raise ValueError("rank must be a multiple of 16 and no larger than 128")
    if any(fraction <= args.top_fraction for fraction in candidate_fractions):
        raise ValueError("candidate fractions must exceed top fraction")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    build_ms: dict[str, list[float]] = defaultdict(list)
    case_count = 0

    for trace_path in args.trace_paths:
        payload = torch.load(trace_path, map_location="cpu", weights_only=False)
        records_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload["records"]:
            layer = int(record["layer"])
            if layer in layers:
                records_by_layer[layer].append(record)

        for layer, records in sorted(records_by_layer.items()):
            records.sort(key=lambda row: int(row.get("step", 0)))
            records = records[: args.max_steps]
            key_record = next(row for row in records if row.get("key") is not None)
            all_key = key_record["key"].to(device).float()[0]
            history_count = all_key.shape[1] - 1
            key = all_key[:, :history_count]
            scaling = float(key_record["scaling"])
            keep_count = max(1, math.ceil(args.top_fraction * history_count))

            query_rows = [
                row["query"].to(device).float()[0, :, 0, :] for row in records
            ]
            exact_rows = [grouped_scores(query, key) * scaling for query in query_rows]
            attention_rows = [torch.softmax(scores, dim=-1) for scores in exact_rows]
            oracle_rows = [
                torch.topk(scores, keep_count, dim=-1, sorted=False).indices
                for scores in exact_rows
            ]
            oracle_mass_rows = [
                torch.gather(attention, -1, oracle).sum(dim=-1)
                for attention, oracle in zip(attention_rows, oracle_rows)
            ]

            for method in methods:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                covariance = covariance_for_method(key, method)
                _, eigenvectors = torch.linalg.eigh(covariance)
                basis = eigenvectors[..., -args.rank :].contiguous()
                projected_key = torch.einsum("hnd,hdr->hnr", key, basis)
                indexed_key = logscale16_int4_dequantize(projected_key)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                build_ms[method].append((time.perf_counter() - start) * 1000.0)

                for query, exact_scores, attention, oracle, oracle_mass in zip(
                    query_rows,
                    exact_rows,
                    attention_rows,
                    oracle_rows,
                    oracle_mass_rows,
                ):
                    kv_heads = key.shape[0]
                    groups = query.shape[0] // kv_heads
                    projected_query = torch.einsum(
                        "hgd,hdr->hgr",
                        query.reshape(kv_heads, groups, query.shape[-1]),
                        basis,
                    ).reshape(query.shape[0], args.rank)
                    proxy_scores = grouped_scores(
                        quantize_query_int8(projected_query), indexed_key
                    ) * scaling
                    for fraction in candidate_fractions:
                        candidate_count = max(
                            keep_count, math.ceil(fraction * history_count)
                        )
                        candidates = torch.topk(
                            proxy_scores,
                            candidate_count,
                            dim=-1,
                            sorted=False,
                        ).indices
                        selected = exact_rerank(
                            exact_scores, candidates, keep_count
                        )
                        recall, mass = selected_quality(
                            selected, oracle, attention, oracle_mass
                        )
                        name = f"{method}_candidate{fraction:g}"
                        metrics[name]["top2_recall"].extend(recall.cpu().tolist())
                        metrics[name]["top2_attention_mass_recall"].extend(
                            mass.cpu().tolist()
                        )
                del covariance, eigenvectors, basis, projected_key, indexed_key
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            case_count += len(query_rows) * query_rows[0].shape[0]

    report = {
        "config": vars(args) | {
            "trace_paths": [str(path) for path in args.trace_paths],
            "candidate_fractions": candidate_fractions,
            "methods": methods,
        },
        "head_cases": case_count,
        "basis_build_ms": {
            method: summarize(values) for method, values in sorted(build_ms.items())
        },
        "retrieval": {
            method: {
                metric: summarize(values)
                for metric, values in sorted(metric_values.items())
            }
            for method, metric_values in sorted(metrics.items())
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

