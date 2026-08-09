from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from run_head_top2_targeted_ppl_20260714 import _pca_int4_partial_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test sparse high-score attention plus a low-cost background Value."
    )
    parser.add_argument("--trace_paths", nargs="+", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--sample_fraction", type=float, default=0.0025)
    parser.add_argument("--candidate_fractions", default="0.005,0.01,0.02")
    return parser.parse_args()


def parse_fractions(spec: str) -> list[float]:
    fractions = sorted({float(item) for item in spec.split(",") if item.strip()})
    if not fractions or fractions[0] <= 0.0 or fractions[-1] > 0.25:
        raise ValueError("candidate fractions must be in (0, 0.25]")
    return fractions


def tensor_summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "max": float(tensor.max()),
    }


def output_metrics(
    estimate: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    relative_l2 = (estimate - target).norm(dim=-1) / target.norm(dim=-1).clamp_min(1.0e-8)
    cosine = F.cosine_similarity(estimate, target, dim=-1)
    return relative_l2, cosine


def evaluate_candidate_set(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    value: torch.Tensor,
    candidate_indices: torch.Tensor,
    sample_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch_count, kv_head_count, group_count, history_count = exact_scores.shape
    candidate_count = int(candidate_indices.shape[-1])
    head_dim = int(value.shape[-1])
    query_head_count = kv_head_count * group_count

    center = torch.maximum(
        exact_scores.amax(dim=-1), proxy_scores.amax(dim=-1)
    ).unsqueeze(-1)
    exact_weights = torch.exp(exact_scores - center)
    proxy_weights = torch.exp(proxy_scores - center)
    exact_total = exact_weights.sum(dim=-1)
    proxy_total = proxy_weights.sum(dim=-1)

    exact_candidate_weights = torch.gather(exact_weights, -1, candidate_indices)
    proxy_candidate_weights = torch.gather(proxy_weights, -1, candidate_indices)
    expanded_value = value.unsqueeze(2).expand(-1, -1, group_count, -1, -1)
    candidate_value = torch.gather(
        expanded_value,
        3,
        candidate_indices.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim),
    ).float()
    exact_kept_partition = exact_candidate_weights.sum(dim=-1)
    proxy_kept_partition = proxy_candidate_weights.sum(dim=-1)
    kept_numerator = (
        exact_candidate_weights.unsqueeze(-1) * candidate_value
    ).sum(dim=-2)

    full_numerator = torch.einsum("bhgn,bhnd->bhgd", exact_weights, value.float())
    full_output = full_numerator / exact_total.unsqueeze(-1).clamp_min(1.0e-30)
    sparse_output = kept_numerator / exact_kept_partition.unsqueeze(-1).clamp_min(1.0e-30)

    tail_size = history_count - candidate_count
    total_value = value.float().sum(dim=-2).unsqueeze(2)
    tail_mean = (
        total_value - candidate_value.sum(dim=-2)
    ) / max(1, tail_size)
    exact_tail_partition = (exact_total - exact_kept_partition).clamp_min(0.0)
    oracle_background_output = (
        kept_numerator + exact_tail_partition.unsqueeze(-1) * tail_mean
    ) / exact_total.unsqueeze(-1).clamp_min(1.0e-30)

    sample_count = int(sample_indices.numel())
    sample_indices_grouped = sample_indices.view(1, 1, 1, -1).expand(
        batch_count, kv_head_count, group_count, -1
    )
    sample_exact_weights = torch.gather(exact_weights, -1, sample_indices_grouped)
    sample_proxy_weights = torch.gather(proxy_weights, -1, sample_indices_grouped)
    sample_value = value.index_select(-2, sample_indices).unsqueeze(2).expand(
        -1, -1, group_count, -1, -1
    ).float()
    sample_is_candidate = (
        sample_indices_grouped.unsqueeze(-1) == candidate_indices.unsqueeze(-2)
    ).any(dim=-1)
    outside_sample = ~sample_is_candidate
    outside_count = outside_sample.sum(dim=-1).clamp_min(1)
    expansion = tail_size / outside_count

    sample_difference = sample_exact_weights - sample_proxy_weights
    difference_mean = (
        sample_difference * outside_sample
    ).sum(dim=-1) / outside_count
    proxy_tail_partition = (proxy_total - proxy_kept_partition).clamp_min(0.0)
    cv_tail_partition = (
        proxy_tail_partition + tail_size * difference_mean
    ).clamp_min(0.0)

    cv_background_output = (
        kept_numerator + cv_tail_partition.unsqueeze(-1) * tail_mean
    ) / (exact_kept_partition + cv_tail_partition).unsqueeze(-1).clamp_min(1.0e-30)

    sampled_tail_residual = (
        sample_exact_weights.unsqueeze(-1)
        * (sample_value - tail_mean.unsqueeze(-2))
        * outside_sample.unsqueeze(-1)
    ).sum(dim=-2) * expansion.unsqueeze(-1)
    residual_background_output = (
        kept_numerator
        + cv_tail_partition.unsqueeze(-1) * tail_mean
        + sampled_tail_residual
    ) / (exact_kept_partition + cv_tail_partition).unsqueeze(-1).clamp_min(1.0e-30)

    mc_tail_partition = (
        sample_exact_weights * outside_sample
    ).sum(dim=-1) * expansion
    mc_tail_numerator = (
        sample_exact_weights.unsqueeze(-1)
        * sample_value
        * outside_sample.unsqueeze(-1)
    ).sum(dim=-2) * expansion.unsqueeze(-1)
    monte_carlo_output = (
        kept_numerator + mc_tail_numerator
    ) / (exact_kept_partition + mc_tail_partition).unsqueeze(-1).clamp_min(1.0e-30)

    outputs = {
        "sparse_renorm": sparse_output,
        "oracle_mass_background": oracle_background_output,
        "cv_mass_background": cv_background_output,
        "cv_mass_sampled_residual": residual_background_output,
        "uniform_tail_monte_carlo": monte_carlo_output,
    }
    result: dict[str, torch.Tensor] = {
        "kept_mass": exact_kept_partition / exact_total.clamp_min(1.0e-30)
    }
    for name, estimate in outputs.items():
        relative_l2, cosine = output_metrics(estimate, full_output)
        result[f"{name}_relative_l2"] = relative_l2.reshape(query_head_count)
        result[f"{name}_cosine"] = cosine.reshape(query_head_count)
    return result


def main() -> None:
    args = parse_args()
    candidate_fractions = parse_fractions(args.candidate_fractions)
    device = torch.device(args.device)
    metrics: dict[tuple[str, int, str, float, str], list[float]] = defaultdict(list)
    case_count = 0

    for trace_path in args.trace_paths:
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        topic = str(trace.get("config", {}).get("topic", trace_path.stem))
        for record in trace["records"]:
            if "value" not in record:
                raise RuntimeError(f"{trace_path} does not contain Value tensors")
            query = record["query"].to(device)
            key = record["key"].to(device)
            value = record["value"].to(device)
            scaling = float(record["scaling"])
            batch_count, query_head_count, _, head_dim = query.shape
            kv_head_count = int(key.shape[1])
            group_count = query_head_count // kv_head_count
            history_count = int(key.shape[2])
            grouped_query = query.squeeze(2).reshape(
                batch_count, kv_head_count, group_count, head_dim
            )
            exact_scores = torch.einsum(
                "bhgd,bhnd->bhgn", grouped_query.float(), key.float()
            ) * scaling
            pca_state: dict[str, Any] = {}
            proxy_scores = _pca_int4_partial_scores(
                query.squeeze(2),
                key,
                pca_state,
                args.projection_dim,
                use_chunked_layout=True,
            ).reshape(batch_count, kv_head_count, group_count, history_count).float()
            proxy_scores = proxy_scores * scaling

            sample_count = min(
                history_count, max(16, math.ceil(args.sample_fraction * history_count))
            )
            stride = max(1, history_count // sample_count)
            offset = (37 * int(record["layer"])) % stride
            sample_indices = (
                offset + torch.arange(sample_count, device=device) * stride
            ).clamp_max(history_count - 1)

            max_count = math.ceil(candidate_fractions[-1] * history_count)
            exact_order = torch.topk(exact_scores, max_count, dim=-1, sorted=True).indices
            proxy_order = torch.topk(proxy_scores, max_count, dim=-1, sorted=True).indices
            for candidate_mode, order in (("exact", exact_order), ("proxy", proxy_order)):
                for fraction in candidate_fractions:
                    candidate_count = max(1, math.ceil(fraction * history_count))
                    result = evaluate_candidate_set(
                        exact_scores,
                        proxy_scores,
                        value,
                        order[..., :candidate_count],
                        sample_indices,
                    )
                    kept_mass = result.pop("kept_mass").reshape(-1).cpu().tolist()
                    metrics[(topic, int(record["layer"]), candidate_mode, fraction, "kept_mass")].extend(
                        kept_mass
                    )
                    for metric_name, tensor in result.items():
                        metrics[(topic, int(record["layer"]), candidate_mode, fraction, metric_name)].extend(
                            tensor.cpu().tolist()
                        )
            case_count += query_head_count
            print(f"[{topic}] layer={record['layer']} complete", flush=True)
            del query, key, value, exact_scores, proxy_scores, pca_state
            torch.cuda.empty_cache()

    rows = []
    for key, values in sorted(metrics.items()):
        topic, layer, candidate_mode, fraction, metric_name = key
        rows.append(
            {
                "topic": topic,
                "layer": layer,
                "candidate_mode": candidate_mode,
                "candidate_fraction": fraction,
                "metric": metric_name,
                **tensor_summary(values),
            }
        )

    aggregate: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for (topic, layer, candidate_mode, fraction, metric_name), values in metrics.items():
        del topic, layer
        aggregate[(candidate_mode, fraction, metric_name)].extend(values)
    aggregate_rows = [
        {
            "candidate_mode": candidate_mode,
            "candidate_fraction": fraction,
            "metric": metric_name,
            **tensor_summary(values),
        }
        for (candidate_mode, fraction, metric_name), values in sorted(aggregate.items())
    ]
    output = {
        "config": {
            "trace_paths": [str(path) for path in args.trace_paths],
            "projection_dim": args.projection_dim,
            "sample_fraction": args.sample_fraction,
            "candidate_fractions": candidate_fractions,
            "query_head_cases": case_count,
        },
        "aggregate": aggregate_rows,
        "by_case": rows,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
