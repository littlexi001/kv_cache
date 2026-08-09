from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from analyze_numeric_pruning_frontier import (
    grouped_scores,
    logscale16_int4_dequantize,
    parse_ints,
    summarize,
)
from analyze_qk_metric_lowrank import qk_metric_factors


def pad_blocks(value: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    token_count = value.shape[0]
    block_count = math.ceil(token_count / block_size)
    padded_count = block_count * block_size
    if padded_count > token_count:
        padding = torch.zeros(
            padded_count - token_count,
            *value.shape[1:],
            dtype=value.dtype,
            device=value.device,
        )
        value = torch.cat((value, padding), dim=0)
    lengths = torch.full(
        (block_count,), block_size, dtype=torch.long, device=value.device
    )
    lengths[-1] = token_count - (block_count - 1) * block_size
    return value.reshape(block_count, block_size, *value.shape[1:]), lengths


def q8_dequantize_per_vector(value: torch.Tensor) -> torch.Tensor:
    scale = (value.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1.0e-8)
    return torch.round(value / scale).clamp(-127, 127) * scale


def output_metrics(estimate: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    relative_l2 = (
        (estimate - reference).float().norm()
        / reference.float().norm().clamp_min(1.0e-12)
    )
    cosine = F.cosine_similarity(
        estimate.float().unsqueeze(0), reference.float().unsqueeze(0)
    )[0]
    return float(relative_l2.item()), float(cosine.item())


def append_metrics(
    metrics: dict[str, dict[str, list[float]]],
    name: str,
    estimate: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    relative_l2, cosine = output_metrics(estimate, reference)
    metrics[name]["relative_l2"].append(relative_l2)
    metrics[name]["cosine"].append(cosine)


def fit_affine_proxy(
    proxy_scores: torch.Tensor,
    exact_scores: torch.Tensor,
    sample_stride: int,
    sample_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampled_proxy = proxy_scores[sample_offset::sample_stride].float()
    sampled_exact = exact_scores[sample_offset::sample_stride].float()
    proxy_centered = sampled_proxy - sampled_proxy.mean()
    exact_centered = sampled_exact - sampled_exact.mean()
    slope = (
        (proxy_centered * exact_centered).mean()
        / proxy_centered.square().mean().clamp_min(1.0e-12)
    ).clamp(0.25, 4.0)
    intercept = sampled_exact.mean() - slope * sampled_proxy.mean()
    return slope, intercept


def moment_tail_output(
    score_proxy: torch.Tensor,
    exact_scores: torch.Tensor,
    values: torch.Tensor,
    selected: torch.Tensor,
    block_size: int,
    use_exact_block_scores: bool,
    quantize_value_mean: bool,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    token_count = score_proxy.numel()
    block_scores_source = exact_scores if use_exact_block_scores else score_proxy
    score_blocks, lengths = pad_blocks(block_scores_source, block_size)
    value_blocks, _ = pad_blocks(values, block_size)
    valid = (
        torch.arange(block_size, device=values.device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    denominator = lengths.float().unsqueeze(1)
    block_score_sum = score_blocks.sum(dim=1)
    block_score_square_sum = score_blocks.square().sum(dim=1)
    block_value_sum = (value_blocks * valid.unsqueeze(-1)).sum(dim=1)
    selected_blocks = selected // block_size
    selected_count = torch.zeros_like(lengths)
    selected_count.scatter_add_(
        0, selected_blocks, torch.ones_like(selected_blocks, dtype=lengths.dtype)
    )
    block_score_sum.scatter_add_(
        0, selected_blocks, -block_scores_source[selected]
    )
    block_score_square_sum.scatter_add_(
        0, selected_blocks, -block_scores_source[selected].square()
    )
    block_value_sum.index_add_(0, selected_blocks, -values[selected])
    tail_lengths = lengths - selected_count
    tail_denominator = tail_lengths.clamp_min(1).float()
    block_mean_score = block_score_sum / tail_denominator
    block_variance = (
        block_score_square_sum / tail_denominator - block_mean_score.square()
    ).clamp_min(0.0)
    block_mean_value = block_value_sum / tail_denominator.unsqueeze(1)
    if quantize_value_mean:
        block_mean_value = q8_dequantize_per_vector(block_mean_value)

    block_log_weight = (
        tail_denominator.log() + block_mean_score + 0.5 * block_variance
    ).masked_fill(tail_lengths == 0, -torch.inf)
    anchor = torch.maximum(exact_scores[selected].max(), block_log_weight.max())
    block_weight = torch.exp(block_log_weight - anchor)
    selected_exact_weight = torch.exp(exact_scores[selected] - anchor)
    selected_numerator = torch.einsum(
        "n,nd->d", selected_exact_weight, values[selected]
    )
    tail_numerator = torch.einsum(
        "b,bd->d", block_weight, block_mean_value
    )
    total_weight = selected_exact_weight.sum() + block_weight.sum()
    denominator_only = selected_numerator / total_weight.clamp_min(1.0e-20)
    with_value_mean = (selected_numerator + tail_numerator) / total_weight.clamp_min(
        1.0e-20
    )
    estimated_selected_mass = float(
        (selected_exact_weight.sum() / total_weight.clamp_min(1.0e-20)).item()
    )
    return denominator_only, with_value_mean, estimated_selected_mass


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--rank", type=int, default=48)
    parser.add_argument("--block_sizes", default="8,16,32,64,128")
    parser.add_argument("--train_steps", type=int, default=4)
    parser.add_argument("--test_start_step", type=int, default=8)
    parser.add_argument("--test_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.5)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--affine_sample_stride", type=int, default=400)
    parser.add_argument("--candidate_fraction", type=float, default=0.06)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    args = parser.parse_args()

    block_sizes = parse_ints(args.block_sizes)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = [
        row for row in payload["records"] if int(row["layer"]) == args.layer
    ]
    records.sort(key=lambda row: int(row.get("step", 0)))
    key_record = next(row for row in records if row.get("key") is not None)
    key = key_record["key"].to(device).float()[0, :, :-1]
    value = key_record["value"].to(device).float()[0, :, :-1]
    scaling = float(key_record["scaling"])
    kv_heads, history_count, head_dim = key.shape
    query_heads = int(records[0]["query"].shape[1])
    groups = query_heads // kv_heads
    keep_count = max(1, math.ceil(args.top_fraction * history_count))
    candidate_count = max(
        keep_count, math.ceil(args.candidate_fraction * history_count)
    )

    sampled_key = key[:, :: args.key_sample_stride]
    key_covariance = torch.einsum(
        "hnd,hne->hde", sampled_key, sampled_key
    ) / float(sampled_key.shape[1])
    train_query = torch.stack(
        [row["query"].to(device).float()[0, :, 0] for row in records[: args.train_steps]]
    ).reshape(args.train_steps, kv_heads, groups, head_dim)
    query_covariance = torch.einsum(
        "thgd,thge->hde", train_query, train_query
    ) / float(args.train_steps * groups)
    isotropic_scale = query_covariance.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    identity = torch.eye(head_dim, device=device).unsqueeze(0)
    regularized_query_covariance = (
        (1.0 - args.query_shrinkage) * query_covariance
        + args.query_shrinkage * isotropic_scale[:, None, None] * identity
    )
    query_factor, key_factor = qk_metric_factors(
        key_covariance, regularized_query_covariance, args.rank
    )
    projected_key = torch.einsum("hnd,hdr->hnr", key, key_factor)
    indexed_key = logscale16_int4_dequantize(projected_key)

    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    test_records = records[
        args.test_start_step : args.test_start_step + args.test_steps
    ]
    for step_offset, record in enumerate(test_records):
        query = record["query"].to(device).float()[0, :, 0]
        grouped_query = query.reshape(kv_heads, groups, head_dim)
        projected_query = torch.einsum(
            "hgd,hdr->hgr", grouped_query, query_factor
        )
        exact_scores = grouped_scores(query, key) * scaling
        for head in range(kv_heads):
            head_value = value[head]
            for group in range(groups):
                query_head = head * groups + group
                scores = exact_scores[query_head]
                probabilities = torch.softmax(scores, dim=-1)
                reference = probabilities @ head_value
                proxy = (indexed_key[head] @ projected_query[head, group]) * scaling
                slope, intercept = fit_affine_proxy(
                    proxy,
                    scores,
                    args.affine_sample_stride,
                    step_offset % args.affine_sample_stride,
                )
                proxy = slope * proxy + intercept
                candidates = torch.topk(
                    proxy, candidate_count, sorted=False
                ).indices
                selected_local = torch.topk(
                    scores[candidates], keep_count, sorted=False
                ).indices
                selected = candidates[selected_local]
                selected_probability = probabilities[selected]
                selected_mass = selected_probability.sum()
                selected_weighted = torch.einsum(
                    "n,nd->d", selected_probability, head_value[selected]
                )
                selected_renormalized = selected_weighted / selected_mass.clamp_min(
                    1.0e-20
                )
                selected_true_denominator = selected_weighted
                append_metrics(
                    metrics, "selected_top2_renormalized", selected_renormalized, reference
                )
                append_metrics(
                    metrics,
                    "selected_top2_true_denominator",
                    selected_true_denominator,
                    reference,
                )
                metrics["selected_top2_renormalized"]["selected_mass"].append(
                    float(selected_mass.item())
                )

                for block_size in block_sizes:
                    for exact_moments in (False, True):
                        for q8_value in (False, True):
                            denominator_only, value_mean, estimated_mass = moment_tail_output(
                                proxy,
                                scores,
                                head_value,
                                selected,
                                block_size,
                                exact_moments,
                                q8_value,
                            )
                            prefix = "exact" if exact_moments else "proxy"
                            suffix = "_vq8" if q8_value else "_vfp"
                            base = f"b{block_size}_{prefix}{suffix}"
                            append_metrics(
                                metrics,
                                f"{base}_denominator_only",
                                denominator_only,
                                reference,
                            )
                            append_metrics(
                                metrics,
                                f"{base}_value_mean",
                                value_mean,
                                reference,
                            )
                            for blend in (0.25, 0.5, 0.75):
                                blended = torch.lerp(
                                    selected_renormalized, value_mean, blend
                                )
                                append_metrics(
                                    metrics,
                                    f"{base}_value_mean_blend{blend:g}",
                                    blended,
                                    reference,
                                )
                            for threshold in (0.6, 0.7, 0.8, 0.9):
                                gated_blend = 0.75 if estimated_mass >= threshold else 0.0
                                gated = torch.lerp(
                                    selected_renormalized, value_mean, gated_blend
                                )
                                append_metrics(
                                    metrics,
                                    f"{base}_value_mean_gate{threshold:g}",
                                    gated,
                                    reference,
                                )
                            for low, high in ((0.5, 0.9), (0.6, 0.9), (0.7, 0.95)):
                                adaptive_blend = 0.75 * min(
                                    1.0,
                                    max(0.0, (estimated_mass - low) / (high - low)),
                                )
                                adaptive = torch.lerp(
                                    selected_renormalized,
                                    value_mean,
                                    adaptive_blend,
                                )
                                append_metrics(
                                    metrics,
                                    f"{base}_value_mean_adaptive{low:g}_{high:g}",
                                    adaptive,
                                    reference,
                                )
                            metrics[f"{base}_value_mean"][
                                "estimated_selected_mass"
                            ].append(estimated_mass)

    result = {
        "config": {
            **vars(args),
            "trace_path": str(args.trace_path),
            "output_path": str(args.output_path),
        },
        "state": {
            "history_tokens": history_count,
            "rank": args.rank,
            "qk_index_fraction_of_full_kv_bf16": (
                history_count * args.rank / 2 + history_count * 4
            )
            / (history_count * head_dim * 4),
            "value_mean_q8_fraction_by_block": {
                str(block_size): 1.0 / (4.0 * block_size)
                for block_size in block_sizes
            },
        },
        "methods": {
            name: {metric: summarize(values) for metric, values in bucket.items()}
            for name, bucket in metrics.items()
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
