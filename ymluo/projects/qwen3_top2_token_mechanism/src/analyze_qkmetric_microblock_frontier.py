from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from analyze_numeric_pruning_frontier import (
    grouped_scores,
    parse_floats,
    parse_ints,
    selected_quality,
    summarize,
)
from analyze_qk_metric_lowrank import qk_metric_factors


def pad_tokens(value: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    token_count = value.shape[-2]
    block_count = math.ceil(token_count / block_size)
    padded_count = block_count * block_size
    if padded_count == token_count:
        return value, block_count
    padding = torch.zeros(
        *value.shape[:-2],
        padded_count - token_count,
        value.shape[-1],
        dtype=value.dtype,
        device=value.device,
    )
    return torch.cat([value, padding], dim=-2), block_count


def microblock_summaries(
    value: torch.Tensor, block_size: int, token_count: int
) -> dict[str, torch.Tensor]:
    padded, block_count = pad_tokens(value.float(), block_size)
    heads, _, dimensions = padded.shape
    blocks = padded.reshape(heads, block_count, block_size, dimensions)
    lengths = torch.full(
        (block_count,), block_size, dtype=torch.long, device=value.device
    )
    lengths[-1] = token_count - (block_count - 1) * block_size
    mask = (
        torch.arange(block_size, device=value.device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    denominator = lengths.float().view(1, block_count, 1)
    mean = blocks.sum(dim=2) / denominator
    residual = blocks - mean.unsqueeze(2)
    residual = residual * mask.view(1, block_count, block_size, 1)
    variance = residual.square().sum(dim=2) / denominator
    residual_norm = residual.square().sum(dim=-1)
    radius = residual_norm.sqrt().amax(dim=-1)
    farthest_offset = residual_norm.argmax(dim=-1)
    maxnorm_offset = blocks.square().sum(dim=-1).masked_fill(
        ~mask.unsqueeze(0), -torch.inf
    ).argmax(dim=-1)
    block_ids = torch.arange(block_count, device=value.device).view(1, -1)
    head_ids = torch.arange(heads, device=value.device).view(-1, 1)
    farthest = blocks[head_ids, block_ids, farthest_offset]
    maxnorm = blocks[head_ids, block_ids, maxnorm_offset]

    anchor_offsets = torch.tensor(
        sorted({0, (block_size - 1) // 3, 2 * (block_size - 1) // 3, block_size - 1}),
        dtype=torch.long,
        device=value.device,
    )
    anchors = blocks.index_select(2, anchor_offsets)
    anchor_valid = anchor_offsets.view(1, 1, -1) < lengths.view(1, -1, 1)
    return {
        "mean": mean.contiguous(),
        "variance": variance.contiguous(),
        "radius": radius.contiguous(),
        "farthest": farthest.contiguous(),
        "maxnorm": maxnorm.contiguous(),
        "anchors": anchors.contiguous(),
        "anchor_valid": anchor_valid,
        "lengths": lengths,
    }


def block_candidates(
    block_scores: torch.Tensor,
    lengths: torch.Tensor,
    target_count: int,
    block_size: int,
    history_count: int,
) -> tuple[torch.Tensor, int]:
    order = block_scores.argsort(descending=True)
    cumulative = lengths[order].cumsum(dim=0)
    probe_count = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(target_count, dtype=cumulative.dtype, device=cumulative.device),
        ).item()
    ) + 1
    probe_count = min(probe_count, order.numel())
    selected_blocks = order[:probe_count]
    offsets = torch.arange(block_size, device=order.device)
    candidates = (selected_blocks.unsqueeze(1) * block_size + offsets).flatten()
    candidates = candidates[candidates < history_count]
    return candidates, probe_count


def exact_rerank(
    exact_scores: torch.Tensor, candidates: torch.Tensor, keep_count: int
) -> torch.Tensor:
    local = torch.topk(
        exact_scores[candidates], min(keep_count, candidates.numel()), sorted=False
    ).indices
    return candidates[local]


def normal_tail_probability(
    threshold: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    z = (threshold - mean) / sigma.clamp_min(1.0e-6)
    return 0.5 * torch.erfc(z / math.sqrt(2.0))


def mixture_topk_threshold(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    lengths: torch.Tensor,
    keep_count: int,
    iterations: int = 24,
) -> torch.Tensor:
    lower = (mean - 8.0 * sigma).amin()
    upper = (mean + 8.0 * sigma).amax()
    lengths_float = lengths.float()
    for _ in range(iterations):
        middle = (lower + upper) * 0.5
        expected_count = (
            normal_tail_probability(middle, mean, sigma) * lengths_float
        ).sum()
        if bool(expected_count > keep_count):
            lower = middle
        else:
            upper = middle
    return (lower + upper) * 0.5


def exceedance_block_scores(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    lengths: torch.Tensor,
    keep_count: int,
) -> dict[str, torch.Tensor]:
    threshold = mixture_topk_threshold(mean, sigma, lengths, keep_count)
    tail_probability = normal_tail_probability(threshold, mean, sigma).clamp(0.0, 1.0)
    lengths_float = lengths.float()
    expected_count = lengths_float * tail_probability
    hit_probability = -lengths_float * torch.log1p(
        -tail_probability.clamp_max(1.0 - 1.0e-7)
    )

    # For X ~ Normal(mean, sigma^2), this is log E[e^X 1(X > threshold)].
    shifted_tail = normal_tail_probability(
        threshold, mean + sigma.square(), sigma
    ).clamp_min(1.0e-30)
    log_tail_mass = (
        lengths_float.log()
        + mean
        + 0.5 * sigma.square()
        + shifted_tail.log()
    )
    return {
        "exceed_count": expected_count,
        "hit_probability": hit_probability,
        "tail_mass": log_tail_mass,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="16")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--block_sizes", default="8,16,32,64")
    parser.add_argument("--train_steps", type=int, default=4)
    parser.add_argument("--test_start_step", type=int, default=8)
    parser.add_argument("--test_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.5)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--scan_fractions", default="0.03,0.04,0.05,0.06,0.08,0.10")
    parser.add_argument("--outer_fractions", default="0.12,0.16,0.20,0.24,0.32")
    parser.add_argument("--risk_betas", default="2,3,4,5,6")
    parser.add_argument(
        "--hier_outer_score", choices=["tail_mass", "expected_max"], default="tail_mass"
    )
    parser.add_argument("--top_fraction", type=float, default=0.02)
    args = parser.parse_args()

    layers = {int(item) for item in args.layers.split(",") if item.strip()}
    block_sizes = parse_ints(args.block_sizes)
    scan_fractions = parse_floats(args.scan_fractions)
    outer_fractions = parse_floats(args.outer_fractions)
    risk_betas = parse_floats(args.risk_betas)
    if args.rank % 16 or args.rank > 128:
        raise ValueError("rank must be a multiple of 16 and at most 128")
    if any(fraction < args.top_fraction for fraction in scan_fractions):
        raise ValueError("scan fractions must cover the retained top fraction")
    if min(outer_fractions) <= max(scan_fractions):
        raise ValueError("every outer fraction must exceed every candidate fraction")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    hierarchical_by_group: dict[
        str, dict[str, dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    build_rows: list[dict[str, Any]] = []

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
                raise ValueError(f"{trace_path} layer {layer} has too few queries")
            key_record = next(row for row in records if row.get("key") is not None)
            key = key_record["key"].to(device).float()[0, :, :-1]
            scaling = float(key_record["scaling"])
            head_count, history_count, head_dim = key.shape
            query_heads = int(records[0]["query"].shape[1])
            groups = query_heads // head_count
            keep_count = max(1, math.ceil(args.top_fraction * history_count))

            sampled_key = key[:, :: args.key_sample_stride]
            key_covariance = torch.einsum(
                "hnd,hne->hde", sampled_key, sampled_key
            ) / float(sampled_key.shape[1])
            train_query = torch.stack(
                [
                    row["query"].to(device).float()[0, :, 0]
                    for row in records[: args.train_steps]
                ]
            ).reshape(args.train_steps, head_count, groups, head_dim)
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
                key_covariance, regularized_query_covariance, args.rank
            )
            projected_key = torch.einsum("hnd,hdr->hnr", key, key_factor)
            summaries = {
                block_size: microblock_summaries(
                    projected_key, block_size, history_count
                )
                for block_size in block_sizes
            }
            for summary in summaries.values():
                mean_scale = (
                    summary["mean"].abs().amax(dim=-1, keepdim=True) / 127.0
                ).clamp_min(1.0e-8)
                variance_scale = (
                    summary["variance"].amax(dim=-1, keepdim=True) / 255.0
                ).clamp_min(1.0e-8)
                summary["mean_q8_dequant"] = (
                    torch.round(summary["mean"] / mean_scale)
                    .clamp(-127, 127)
                    * mean_scale
                )
                summary["variance_q8_dequant"] = (
                    torch.round(summary["variance"] / variance_scale)
                    .clamp(0, 255)
                    * variance_scale
                )
            for block_size, summary in summaries.items():
                block_count = int(summary["lengths"].numel())
                # Mean plus four anchors is the largest tested summary.
                build_rows.append(
                    {
                        "trace": str(trace_path),
                        "layer": layer,
                        "block_size": block_size,
                        "history_tokens": history_count,
                        "block_count": block_count,
                        "mean_index_fraction_of_full_kv_bf16": (
                            block_count * args.rank * 2
                            / (history_count * head_dim * 4)
                        ),
                        "mean_plus_four_anchor_fraction_of_full_kv_bf16": (
                            block_count * args.rank * 2 * 5
                            / (history_count * head_dim * 4)
                        ),
                    }
                )

            test_records = records[
                args.test_start_step : args.test_start_step + args.test_steps
            ]
            for record in test_records:
                query = record["query"].to(device).float()[0, :, 0]
                grouped_query = query.reshape(head_count, groups, head_dim)
                projected_query = torch.einsum(
                    "hgd,hdr->hgr", grouped_query, query_factor
                )
                exact_scores = grouped_scores(query, key) * scaling
                attention = torch.softmax(exact_scores, dim=-1)
                oracle = torch.topk(
                    exact_scores, keep_count, dim=-1, sorted=False
                ).indices
                oracle_mass = torch.gather(attention, -1, oracle).sum(dim=-1)

                for block_size, summary in summaries.items():
                    expected_multiplier = torch.sqrt(
                        2.0 * summary["lengths"].float().clamp_min(2.0).log()
                    )
                    for head in range(head_count):
                        for group in range(groups):
                            query_head = head * groups + group
                            projected = projected_query[head, group]
                            center_scores = summary["mean"][head] @ projected
                            sigma = torch.sqrt(
                                summary["variance"][head] @ projected.square()
                            ).clamp_min(0.0)
                            expected_scores = center_scores + expected_multiplier * sigma
                            q8_center_scores = (
                                summary["mean_q8_dequant"][head] @ projected
                            )
                            q8_sigma = torch.sqrt(
                                summary["variance_q8_dequant"][head]
                                @ projected.square()
                            ).clamp_min(0.0)
                            q8_expected_scores = (
                                q8_center_scores + expected_multiplier * q8_sigma
                            )
                            farthest_scores = torch.maximum(
                                center_scores, summary["farthest"][head] @ projected
                            )
                            maxnorm_scores = torch.maximum(
                                center_scores, summary["maxnorm"][head] @ projected
                            )
                            anchor_scores = torch.einsum(
                                "bar,r->ba", summary["anchors"][head], projected
                            ).masked_fill(
                                ~summary["anchor_valid"][0], -torch.inf
                            ).amax(dim=-1)
                            support4_scores = torch.maximum(center_scores, anchor_scores)
                            exceedance_scores = exceedance_block_scores(
                                center_scores * scaling,
                                sigma * scaling,
                                summary["lengths"],
                                keep_count,
                            )
                            score_sets = [
                                ("mean", center_scores),
                                ("expected_max", expected_scores),
                                ("farthest", farthest_scores),
                                ("maxnorm", maxnorm_scores),
                                ("support4", support4_scores),
                            ]
                            score_sets.extend(exceedance_scores.items())
                            for score_name, block_scores in score_sets:
                                for fraction in scan_fractions:
                                    target_count = math.ceil(fraction * history_count)
                                    candidates, probe_count = block_candidates(
                                        block_scores,
                                        summary["lengths"],
                                        target_count,
                                        block_size,
                                        history_count,
                                    )
                                    selected = exact_rerank(
                                        exact_scores[query_head], candidates, keep_count
                                    )
                                    recall, mass = selected_quality(
                                        selected.unsqueeze(0),
                                        oracle[query_head].unsqueeze(0),
                                        attention[query_head].unsqueeze(0),
                                        oracle_mass[query_head].unsqueeze(0),
                                    )
                                    name = (
                                        f"b{block_size}_{score_name}_scan{fraction:g}"
                                    )
                                    metrics[name]["top2_recall"].append(
                                        float(recall.item())
                                    )
                                    metrics[name]["top2_attention_mass_recall"].append(
                                        float(mass.item())
                                    )
                                    metrics[name]["scanned_fraction"].append(
                                        candidates.numel() / history_count
                                    )
                                    metrics[name]["probe_count"].append(probe_count)

                            projected_token_scores = (
                                projected_key[head] @ projected
                            ) * scaling
                            projected_global_candidates = torch.topk(
                                projected_token_scores,
                                max(
                                    keep_count,
                                    math.ceil(max(scan_fractions) * history_count),
                                ),
                                sorted=True,
                            ).indices
                            hierarchy_scores = [
                                (
                                    "",
                                    exceedance_scores["tail_mass"]
                                    if args.hier_outer_score == "tail_mass"
                                    else expected_scores,
                                ),
                                ("_q8", q8_expected_scores),
                            ]
                            for score_suffix, outer_block_scores in hierarchy_scores:
                              for outer_fraction in outer_fractions:
                                outer_candidates, outer_probe_count = block_candidates(
                                    outer_block_scores,
                                    summary["lengths"],
                                    math.ceil(outer_fraction * history_count),
                                    block_size,
                                    history_count,
                                )
                                for candidate_fraction in scan_fractions:
                                    candidate_count = max(
                                        keep_count,
                                        math.ceil(candidate_fraction * history_count),
                                    )
                                    local = torch.topk(
                                        projected_token_scores[outer_candidates],
                                        min(candidate_count, outer_candidates.numel()),
                                        sorted=False,
                                    ).indices
                                    candidates = outer_candidates[local]
                                    selected = exact_rerank(
                                        exact_scores[query_head], candidates, keep_count
                                    )
                                    recall, mass = selected_quality(
                                        selected.unsqueeze(0),
                                        oracle[query_head].unsqueeze(0),
                                        attention[query_head].unsqueeze(0),
                                        oracle_mass[query_head].unsqueeze(0),
                                    )
                                    name = (
                                        f"b{block_size}_hier{score_suffix}"
                                        f"_outer{outer_fraction:g}"
                                        f"_cand{candidate_fraction:g}"
                                    )
                                    metrics[name]["top2_recall"].append(
                                        float(recall.item())
                                    )
                                    metrics[name]["top2_attention_mass_recall"].append(
                                        float(mass.item())
                                    )
                                    metrics[name]["outer_scanned_fraction"].append(
                                        outer_candidates.numel() / history_count
                                    )
                                    metrics[name]["candidate_fraction"].append(
                                        candidates.numel() / history_count
                                    )
                                    metrics[name]["outer_probe_count"].append(
                                        outer_probe_count
                                    )
                                    group_name = f"{trace_path.stem}/layer{layer}"
                                    group_metrics = hierarchical_by_group[group_name][name]
                                    group_metrics["top2_recall"].append(
                                        float(recall.item())
                                    )
                                    group_metrics["top2_attention_mass_recall"].append(
                                        float(mass.item())
                                    )
                                    group_metrics["outer_scanned_fraction"].append(
                                        outer_candidates.numel() / history_count
                                    )
                                    if candidates.numel() == keep_count:
                                        direct_recall, direct_mass = selected_quality(
                                            candidates.unsqueeze(0),
                                            oracle[query_head].unsqueeze(0),
                                            attention[query_head].unsqueeze(0),
                                            oracle_mass[query_head].unsqueeze(0),
                                        )
                                        group_metrics[
                                            "direct_proxy_top2_recall"
                                        ].append(float(direct_recall.item()))
                                        group_metrics[
                                            "direct_proxy_top2_attention_mass_recall"
                                        ].append(float(direct_mass.item()))

                                    initial_blocks = torch.unique(
                                        torch.div(
                                            outer_candidates,
                                            block_size,
                                            rounding_mode="floor",
                                        )
                                    )
                                    initial_mask = torch.zeros(
                                        summary["lengths"].numel(),
                                        dtype=torch.bool,
                                        device=device,
                                    )
                                    initial_mask[initial_blocks] = True
                                    initial_threshold = torch.topk(
                                        projected_token_scores[outer_candidates],
                                        min(candidate_count, outer_candidates.numel()),
                                    ).values[-1]
                                    block_upper = (
                                        center_scores * scaling
                                        + projected.norm()
                                        * summary["radius"][head]
                                        * scaling
                                    )
                                    unsafe_blocks = torch.nonzero(
                                        (~initial_mask)
                                        & (block_upper >= initial_threshold),
                                        as_tuple=False,
                                    ).flatten()
                                    certified_blocks = torch.cat(
                                        [initial_blocks, unsafe_blocks]
                                    )
                                    offsets = torch.arange(
                                        block_size, device=device
                                    ).unsqueeze(0)
                                    certified_tokens = (
                                        certified_blocks.unsqueeze(1) * block_size
                                        + offsets
                                    ).flatten()
                                    certified_tokens = certified_tokens[
                                        certified_tokens < history_count
                                    ]
                                    certified_local = torch.topk(
                                        projected_token_scores[certified_tokens],
                                        min(candidate_count, certified_tokens.numel()),
                                        sorted=False,
                                    ).indices
                                    certified_candidates = certified_tokens[
                                        certified_local
                                    ]
                                    certified_selected = exact_rerank(
                                        exact_scores[query_head],
                                        certified_candidates,
                                        keep_count,
                                    )
                                    certified_recall, certified_mass = selected_quality(
                                        certified_selected.unsqueeze(0),
                                        oracle[query_head].unsqueeze(0),
                                        attention[query_head].unsqueeze(0),
                                        oracle_mass[query_head].unsqueeze(0),
                                    )
                                    reference_proxy = projected_global_candidates[
                                        :candidate_count
                                    ]
                                    proxy_hits = (
                                        certified_candidates.unsqueeze(-1)
                                        == reference_proxy.unsqueeze(-2)
                                    ).any(dim=-1).float().mean()
                                    certified_name = (
                                        f"b{block_size}_cert_outer{outer_fraction:g}"
                                        f"_cand{candidate_fraction:g}"
                                    )
                                    certified_metrics = metrics[certified_name]
                                    certified_metrics["top2_recall"].append(
                                        float(certified_recall.item())
                                    )
                                    certified_metrics[
                                        "top2_attention_mass_recall"
                                    ].append(float(certified_mass.item()))
                                    certified_metrics[
                                        "certified_scanned_fraction"
                                    ].append(certified_tokens.numel() / history_count)
                                    certified_metrics["unsafe_block_fraction"].append(
                                        unsafe_blocks.numel()
                                        / summary["lengths"].numel()
                                    )
                                    certified_metrics["proxy_candidate_recall"].append(
                                        float(proxy_hits.item())
                                    )

                                    for risk_beta in risk_betas:
                                        risk_upper = (
                                            center_scores * scaling
                                            + risk_beta * sigma * scaling
                                        )
                                        risk_blocks = torch.nonzero(
                                            (~initial_mask)
                                            & (risk_upper >= initial_threshold),
                                            as_tuple=False,
                                        ).flatten()
                                        scanned_blocks = torch.cat(
                                            [initial_blocks, risk_blocks]
                                        )
                                        risk_tokens = (
                                            scanned_blocks.unsqueeze(1) * block_size
                                            + offsets
                                        ).flatten()
                                        risk_tokens = risk_tokens[
                                            risk_tokens < history_count
                                        ]
                                        risk_local = torch.topk(
                                            projected_token_scores[risk_tokens],
                                            min(candidate_count, risk_tokens.numel()),
                                            sorted=False,
                                        ).indices
                                        risk_candidates = risk_tokens[risk_local]
                                        risk_selected = exact_rerank(
                                            exact_scores[query_head],
                                            risk_candidates,
                                            keep_count,
                                        )
                                        risk_recall, risk_mass = selected_quality(
                                            risk_selected.unsqueeze(0),
                                            oracle[query_head].unsqueeze(0),
                                            attention[query_head].unsqueeze(0),
                                            oracle_mass[query_head].unsqueeze(0),
                                        )
                                        risk_proxy_hits = (
                                            risk_candidates.unsqueeze(-1)
                                            == reference_proxy.unsqueeze(-2)
                                        ).any(dim=-1).float().mean()
                                        risk_name = (
                                            f"b{block_size}_riskbeta{risk_beta:g}"
                                            f"_outer{outer_fraction:g}"
                                            f"_cand{candidate_fraction:g}"
                                        )
                                        risk_metrics = metrics[risk_name]
                                        risk_metrics["top2_recall"].append(
                                            float(risk_recall.item())
                                        )
                                        risk_metrics[
                                            "top2_attention_mass_recall"
                                        ].append(float(risk_mass.item()))
                                        risk_metrics["scanned_fraction"].append(
                                            risk_tokens.numel() / history_count
                                        )
                                        risk_metrics["added_block_fraction"].append(
                                            risk_blocks.numel()
                                            / summary["lengths"].numel()
                                        )
                                        risk_metrics["proxy_candidate_recall"].append(
                                            float(risk_proxy_hits.item())
                                        )

            del key, projected_key, summaries
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = {
        "config": vars(args)
        | {
            "trace_paths": [str(path) for path in args.trace_paths],
            "layers": sorted(layers),
            "block_sizes": list(block_sizes),
            "scan_fractions": list(scan_fractions),
            "outer_fractions": list(outer_fractions),
            "risk_betas": list(risk_betas),
        },
        "build": build_rows,
        "retrieval": {
            method: {
                metric: summarize(values)
                for metric, values in sorted(metric_values.items())
            }
            for method, metric_values in sorted(metrics.items())
        },
        "hierarchical_by_group": {
            group_name: {
                method: {
                    metric: summarize(values)
                    for metric, values in sorted(metric_values.items())
                }
                for method, metric_values in sorted(group_methods.items())
            }
            for group_name, group_methods in sorted(hierarchical_by_group.items())
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
