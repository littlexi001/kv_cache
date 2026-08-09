from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from analyze_balanced_pca_int4 import (
    candidate_recall,
    dynamic_one_shot_scores,
    grouped_scores,
    quantize_per_band_logscale_int4,
    quantize_per_token_int4,
    quantize_per_token_int8,
    summarize,
)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rope_angles(
    positions: torch.Tensor, dimensions: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    inverse_frequency = 1.0 / (
        theta
        ** (
            torch.arange(
                0, dimensions, 2, dtype=torch.float32, device=positions.device
            )
            / dimensions
        )
    )
    frequency = torch.outer(positions.float(), inverse_frequency)
    embedding = torch.cat((frequency, frequency), dim=-1)
    return embedding.cos(), embedding.sin()


def inverse_rope(
    value: torch.Tensor, positions: torch.Tensor, theta: float
) -> torch.Tensor:
    cos, sin = rope_angles(positions, value.shape[-1], theta)
    while cos.ndim < value.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return value * cos - rotate_half(value) * sin


def fixed_relative_rope(value: torch.Tensor, distance: int, theta: float) -> torch.Tensor:
    position = torch.tensor([-distance], dtype=torch.float32, device=value.device)
    cos, sin = rope_angles(position, value.shape[-1], theta)
    while cos.ndim < value.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return value * cos + rotate_half(value) * sin


def zscore(value: torch.Tensor) -> torch.Tensor:
    return (value - value.mean(dim=-1, keepdim=True)) / value.std(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-6)


def quota_union(
    base_scores: torch.Tensor,
    rescue_scores: torch.Tensor,
    candidate_count: int,
    rescue_count: int,
) -> torch.Tensor:
    rescue = torch.topk(
        rescue_scores, rescue_count, dim=-1, sorted=False
    ).indices
    remaining = base_scores.clone()
    remaining.scatter_(-1, rescue, -torch.inf)
    base = torch.topk(
        remaining,
        candidate_count - rescue_count,
        dim=-1,
        sorted=False,
    ).indices
    return torch.cat((rescue, base), dim=-1)


def block_quota_union(
    base_scores: torch.Tensor,
    block_scores: torch.Tensor,
    candidate_count: int,
    rescue_count: int,
    block_size: int,
    history_count: int,
) -> torch.Tensor:
    block_count = min(
        block_scores.shape[-1], max(1, rescue_count // block_size)
    )
    blocks = torch.topk(
        block_scores, block_count, dim=-1, sorted=False
    ).indices
    offsets = torch.arange(block_size, device=blocks.device)
    rescue = (blocks.unsqueeze(-1) * block_size + offsets).flatten(-2)
    rescue = rescue.clamp_max(history_count - 1)
    rescue_capacity = rescue.shape[-1]
    remaining = base_scores.clone()
    remaining.scatter_(-1, rescue, -torch.inf)
    base = torch.topk(
        remaining,
        candidate_count - rescue_capacity,
        dim=-1,
        sorted=False,
    ).indices
    return torch.cat((rescue, base), dim=-1)


def candidate_recall_from_indices(
    candidates: torch.Tensor, exact_top: torch.Tensor
) -> torch.Tensor:
    candidates = candidates.sort(dim=-1).values
    locations = torch.searchsorted(candidates, exact_top).clamp_max(
        candidates.shape[-1] - 1
    )
    return (
        torch.gather(candidates, -1, locations) == exact_top
    ).float().mean(dim=-1)


def record_candidates(
    metrics: dict[str, dict[str, list[float]]],
    method: str,
    candidates: torch.Tensor,
    exact_scores: torch.Tensor,
    exact_top: torch.Tensor,
    value: torch.Tensor,
    keep_count: int,
    group_size: int,
    scaling: float,
) -> None:
    recall = candidate_recall_from_indices(candidates, exact_top)
    metrics[method]["top2_position_recall"].extend(
        recall.flatten().cpu().tolist()
    )
    candidate_scores = torch.gather(exact_scores, -1, candidates)
    local = torch.topk(
        candidate_scores, keep_count, dim=-1, sorted=False
    ).indices
    selected = torch.gather(candidates, -1, local)
    query_value = value.repeat_interleave(group_size, dim=0)
    selected_outputs = []
    oracle_outputs = []
    for step in range(candidates.shape[0]):
        selected_indices = selected[step]
        oracle_indices = exact_top[step]
        selected_value = torch.gather(
            query_value,
            1,
            selected_indices.unsqueeze(-1).expand(-1, -1, value.shape[-1]),
        )
        oracle_value = torch.gather(
            query_value,
            1,
            oracle_indices.unsqueeze(-1).expand(-1, -1, value.shape[-1]),
        )
        selected_weights = torch.softmax(
            torch.gather(exact_scores[step], -1, selected_indices) * scaling,
            dim=-1,
        )
        oracle_weights = torch.softmax(
            torch.gather(exact_scores[step], -1, oracle_indices) * scaling,
            dim=-1,
        )
        selected_outputs.append(
            torch.einsum("hk,hkd->hd", selected_weights, selected_value)
        )
        oracle_outputs.append(
            torch.einsum("hk,hkd->hd", oracle_weights, oracle_value)
        )
    selected_output = torch.stack(selected_outputs)
    oracle_output = torch.stack(oracle_outputs)
    cosine = F.cosine_similarity(selected_output, oracle_output, dim=-1)
    relative_l2 = (
        torch.linalg.vector_norm(selected_output - oracle_output, dim=-1)
        / torch.linalg.vector_norm(oracle_output, dim=-1).clamp_min(1.0e-12)
    )
    metrics[method]["oracle_top2_output_cosine"].extend(
        cosine.flatten().cpu().tolist()
    )
    metrics[method]["oracle_top2_output_relative_l2"].extend(
        relative_l2.flatten().cpu().tolist()
    )


@torch.inference_mode()
def evaluate_trace(
    path: Path,
    *,
    projection_dim: int,
    candidate_fraction: float,
    rescue_fractions: tuple[float, ...],
    theta: float,
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records_by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_layer[int(record["layer"])].append(record)

    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for _, records in sorted(records_by_layer.items()):
        records.sort(key=lambda row: int(row.get("step", 0)))
        if len(records) < 2:
            continue
        key_record = next((row for row in records if row.get("key") is not None), None)
        if key_record is None:
            continue
        post_key = key_record["key"].to(device).float()[0]
        value = key_record["value"].to(device).float()[0]
        history_count = int(post_key.shape[1]) - 1
        post_key = post_key[:, :history_count]
        value = value[:, :history_count]
        post_query = torch.stack(
            [row["query"].to(device).float()[0, :, 0] for row in records]
        )
        kv_heads = int(post_key.shape[0])
        query_heads = int(post_query.shape[1])
        group_size = query_heads // kv_heads
        keep_count = max(1, math.ceil(0.02 * history_count))
        candidate_count = max(
            keep_count, math.ceil(candidate_fraction * history_count)
        )

        exact_scores = grouped_scores(post_key, post_query, group_size)[1:]
        exact_top = torch.topk(
            exact_scores, keep_count, dim=-1, sorted=False
        ).indices
        sampled_key = post_key[:, ::32]
        second_moment = torch.einsum(
            "hnd,hne->hde", sampled_key, sampled_key
        ) / float(sampled_key.shape[1])
        eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
        basis = eigenvectors[..., -projection_dim:]
        projected_key = torch.einsum("hnd,hdm->hnm", post_key, basis)
        grouped_query = post_query.reshape(
            len(records), kv_heads, group_size, post_query.shape[-1]
        )
        projected_query = torch.einsum(
            "thgd,hdm->thgm", grouped_query, basis
        )
        compact_key = quantize_per_band_logscale_int4(
            projected_key, 16, 0.25
        )
        static_scores = grouped_scores(
            compact_key,
            quantize_per_token_int8(projected_query),
            group_size,
        )[1:]
        base_scores, _ = dynamic_one_shot_scores(
            compact_key,
            projected_query,
            eigenvalues[..., -projection_dim:].clamp_min(1.0e-6),
            keep_count=keep_count,
            candidate_count=candidate_count,
            band_size=16,
        )

        key_positions = torch.arange(history_count, device=device)
        query_positions = torch.arange(
            history_count, history_count + len(records), device=device
        )
        pre_key = inverse_rope(post_key, key_positions, theta)
        pre_query = inverse_rope(
            post_query.transpose(0, 1), query_positions, theta
        ).transpose(0, 1)
        pre_raw = grouped_scores(pre_key, pre_query, group_size)[1:]
        pre_cosine = grouped_scores(
            F.normalize(pre_key, dim=-1),
            F.normalize(pre_query, dim=-1),
            group_size,
        )[1:]

        half = post_key.shape[-1] // 2
        low_frequency_scores = {}
        low_key = None
        low_query = None
        for frequency_count in (4, 8, 12, 16, 24, 32):
            low_frequency = torch.cat(
                (
                    torch.arange(half - frequency_count, half, device=device),
                    torch.arange(
                        2 * half - frequency_count, 2 * half, device=device
                    ),
                )
            )
            method_key = F.normalize(
                pre_key.index_select(-1, low_frequency), dim=-1
            )
            method_query = F.normalize(
                pre_query.index_select(-1, low_frequency), dim=-1
            )
            low_frequency_scores[f"pre_lowfreq{2*frequency_count}_cosine_int4"] = (
                grouped_scores(
                    quantize_per_token_int4(method_key),
                    quantize_per_token_int8(method_query),
                    group_size,
                )[1:]
            )
            if frequency_count == 16:
                low_key = method_key
                low_query = method_query
        assert low_key is not None and low_query is not None
        low_cosine = grouped_scores(low_key, low_query, group_size)[1:]
        low_cosine_int4 = low_frequency_scores["pre_lowfreq32_cosine_int4"]
        block_low_frequency = {}
        for block_size in (8, 16, 32):
            usable_count = history_count // block_size * block_size
            block_key = low_key[:, :usable_count].reshape(
                kv_heads, -1, block_size, low_key.shape[-1]
            ).mean(dim=-2)
            block_key = F.normalize(block_key, dim=-1)
            block_low_frequency[block_size] = grouped_scores(
                quantize_per_token_int4(block_key),
                quantize_per_token_int8(low_query),
                group_size,
            )[1:]

        canonical = {}
        for distance in (128, 328, 2048, 8192):
            canonical_key = fixed_relative_rope(pre_key, distance, theta)
            canonical[f"canonical{distance}"] = grouped_scores(
                canonical_key, pre_query, group_size
            )[1:]
        multiscale = torch.stack(
            [zscore(pre_raw), zscore(pre_cosine)]
            + [zscore(scores) for scores in canonical.values()]
        ).amax(dim=0)
        rescue_scores = {
            "pre_raw": pre_raw,
            "pre_cosine": pre_cosine,
            "pre_lowfreq_cosine": low_cosine,
            "pre_lowfreq_cosine_int4": low_cosine_int4,
            **low_frequency_scores,
            **canonical,
            "pre_canonical_multiscale": multiscale,
        }

        base_candidates = torch.topk(
            base_scores, candidate_count, dim=-1, sorted=False
        ).indices
        record_candidates(
            metrics,
            "dynamic_logscale16_base",
            base_candidates,
            exact_scores,
            exact_top,
            value,
            keep_count,
            group_size,
            1.0 / math.sqrt(post_key.shape[-1]),
        )
        static_candidates = torch.topk(
            static_scores, candidate_count, dim=-1, sorted=False
        ).indices
        record_candidates(
            metrics,
            "static_logscale16_base",
            static_candidates,
            exact_scores,
            exact_top,
            value,
            keep_count,
            group_size,
            1.0 / math.sqrt(post_key.shape[-1]),
        )
        static_rescue_count = max(1, math.ceil(0.005 * history_count))
        static_rescue_candidates = quota_union(
            static_scores,
            low_cosine_int4,
            candidate_count,
            static_rescue_count,
        )
        record_candidates(
            metrics,
            "static_logscale16_plus_pre_lowfreq32_int4_0.5pct",
            static_rescue_candidates,
            exact_scores,
            exact_top,
            value,
            keep_count,
            group_size,
            1.0 / math.sqrt(post_key.shape[-1]),
        )
        for rescue_name, scores in rescue_scores.items():
            for fraction in rescue_fractions:
                rescue_count = min(
                    candidate_count - 1,
                    max(1, math.ceil(fraction * history_count)),
                )
                candidates = quota_union(
                    base_scores,
                    scores,
                    candidate_count,
                    rescue_count,
                )
                method = f"base_plus_{rescue_name}_{100*fraction:g}pct"
                record_candidates(
                    metrics,
                    method,
                    candidates,
                    exact_scores,
                    exact_top,
                    value,
                    keep_count,
                    group_size,
                    1.0 / math.sqrt(post_key.shape[-1]),
                )
        for block_size, scores in block_low_frequency.items():
            for fraction in rescue_fractions:
                rescue_count = min(
                    candidate_count - block_size,
                    max(block_size, math.floor(fraction * history_count)),
                )
                candidates = block_quota_union(
                    base_scores,
                    scores,
                    candidate_count,
                    rescue_count,
                    block_size,
                    history_count,
                )
                method = (
                    f"base_plus_pre_lowfreq_int4_block{block_size}_"
                    f"{100*fraction:g}pct"
                )
                record_candidates(
                    metrics,
                    method,
                    candidates,
                    exact_scores,
                    exact_top,
                    value,
                    keep_count,
                    group_size,
                    1.0 / math.sqrt(post_key.shape[-1]),
                )

        del post_key, post_query, pre_key, pre_query, exact_scores, exact_top
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "path": str(path),
        "candidate_fraction": candidate_fraction,
        "rescue_fractions": rescue_fractions,
        "rope_theta": theta,
        "methods": {
            method: {name: summarize(values) for name, values in values_by_name.items()}
            for method, values_by_name in sorted(metrics.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--candidate_fraction", type=float, default=0.08)
    parser.add_argument("--rescue_fractions", default="0.005,0.01,0.02")
    parser.add_argument("--rope_theta", type=float, default=5_000_000.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rescue_fractions = tuple(
        sorted({float(item) for item in args.rescue_fractions.split(",")})
    )
    report = {
        "method": "fixed-budget RoPE-free and canonical-distance candidate rescue",
        "traces": [
            evaluate_trace(
                path,
                projection_dim=args.projection_dim,
                candidate_fraction=args.candidate_fraction,
                rescue_fractions=rescue_fractions,
                theta=args.rope_theta,
                device=torch.device(args.device),
            )
            for path in args.trace_paths
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for trace in report["traces"]:
        print(trace["path"])
        ranking = sorted(
            trace["methods"].items(),
            key=lambda item: item[1]["top2_position_recall"]["mean"],
            reverse=True,
        )
        for method, values in ranking[:10]:
            print(
                method,
                f'recall={100*values["top2_position_recall"]["mean"]:.4f}%',
                f'l2={100*values["oracle_top2_output_relative_l2"]["mean"]:.4f}%',
            )


if __name__ == "__main__":
    main()
