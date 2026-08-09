from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not result or result[0] <= 0:
        raise ValueError("expected positive comma-separated integers")
    return result


def parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(sorted({float(item) for item in value.split(",") if item.strip()}))
    if not result or result[0] <= 0.0:
        raise ValueError("expected positive comma-separated numbers")
    return result


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        return {}
    return {
        "mean": float(tensor.mean()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "minimum": float(tensor.min()),
        "maximum": float(tensor.max()),
    }


def sampled_basis(key: torch.Tensor, rank: int, stride: int) -> torch.Tensor:
    sampled = key[:, ::stride].float()
    second_moment = torch.einsum("hnd,hne->hde", sampled, sampled)
    second_moment /= float(sampled.shape[1])
    _, eigenvectors = torch.linalg.eigh(second_moment)
    return eigenvectors[..., -rank:].contiguous()


def logscale16_int4_dequantize(projected_key: torch.Tensor) -> torch.Tensor:
    if projected_key.shape[-1] % 16:
        raise ValueError("log-scale INT4 rank must be divisible by 16")
    shape = projected_key.shape
    bands = projected_key.float().reshape(*shape[:-1], shape[-1] // 16, 16)
    exact_scale = bands.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    base_scale = exact_scale.amax(dim=-2, keepdim=True)
    exponent = torch.round(
        torch.log2(base_scale / exact_scale).clamp_min(0.0) / 0.25
    ).clamp(0, 15)
    scale = base_scale * torch.exp2(-0.25 * exponent)
    codes = torch.round(bands / scale).clamp(-7, 7)
    return (codes * scale).reshape(shape)


def quantize_query_int8(projected_query: torch.Tensor) -> torch.Tensor:
    scale = projected_query.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 127.0
    return torch.round(projected_query / scale).clamp(-127, 127) * scale


def grouped_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    query_heads, head_dim = query.shape
    kv_heads, tokens, key_dim = key.shape
    if head_dim != key_dim or query_heads % kv_heads:
        raise ValueError("invalid GQA query/key shapes")
    groups = query_heads // kv_heads
    return torch.einsum(
        "hgd,hnd->hgn", query.reshape(kv_heads, groups, head_dim), key
    ).reshape(query_heads, tokens)


def exact_rerank(
    exact_scores: torch.Tensor, candidate_indices: torch.Tensor, keep_count: int
) -> torch.Tensor:
    candidate_exact = torch.gather(exact_scores, -1, candidate_indices)
    local = torch.topk(candidate_exact, keep_count, dim=-1, sorted=False).indices
    return torch.gather(candidate_indices, -1, local)


def selected_quality(
    selected: torch.Tensor,
    oracle: torch.Tensor,
    attention: torch.Tensor,
    oracle_mass: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hits = (selected.unsqueeze(-1) == oracle.unsqueeze(-2)).any(dim=-1)
    recall = hits.float().mean(dim=-1)
    mass = torch.gather(attention, -1, selected).sum(dim=-1)
    return recall, mass / oracle_mass.clamp_min(1.0e-30)


def append_quality(
    metrics: dict[str, dict[str, list[float]]],
    name: str,
    selected: torch.Tensor,
    oracle: torch.Tensor,
    attention: torch.Tensor,
    oracle_mass: torch.Tensor,
    candidate_ratio: float,
    index_work_ratio: float,
) -> None:
    recall, mass_recall = selected_quality(
        selected, oracle, attention, oracle_mass
    )
    metrics[name]["top2_recall"].extend(recall.cpu().tolist())
    metrics[name]["top2_attention_mass_recall"].extend(
        mass_recall.cpu().tolist()
    )
    metrics[name]["candidate_ratio"].extend(
        [candidate_ratio] * int(recall.numel())
    )
    metrics[name]["normalized_index_work"].extend(
        [index_work_ratio] * int(recall.numel())
    )


def append_sampled_quantile_quality(
    metrics: dict[str, dict[str, list[float]]],
    name: str,
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    global_candidates: torch.Tensor,
    oracle: torch.Tensor,
    attention: torch.Tensor,
    oracle_mass: torch.Tensor,
    keep_count: int,
    sample_count: int,
    selected_fraction: float,
) -> None:
    token_count = int(proxy_scores.shape[-1])
    sample_count = min(sample_count, token_count)
    sample_indices = torch.floor(
        (torch.arange(sample_count, device=proxy_scores.device) + 0.5)
        * token_count
        / sample_count
    ).long()
    sampled = proxy_scores.index_select(-1, sample_indices)
    sample_keep = max(1, math.ceil(selected_fraction * sample_count))
    thresholds = torch.topk(sampled, sample_keep, dim=-1, sorted=True).values[:, -1]

    for head in range(proxy_scores.shape[0]):
        candidates = torch.nonzero(
            proxy_scores[head] >= thresholds[head], as_tuple=False
        ).flatten()
        if candidates.numel() < keep_count:
            candidates = global_candidates[head]
        selected = exact_rerank(
            exact_scores[head : head + 1],
            candidates.unsqueeze(0),
            keep_count,
        )
        recall, mass_recall = selected_quality(
            selected,
            oracle[head : head + 1],
            attention[head : head + 1],
            oracle_mass[head : head + 1],
        )
        global_recall = (
            candidates.unsqueeze(-1) == global_candidates[head].unsqueeze(0)
        ).any(dim=0).float().mean()
        metrics[name]["top2_recall"].append(float(recall.item()))
        metrics[name]["top2_attention_mass_recall"].append(
            float(mass_recall.item())
        )
        metrics[name]["candidate_ratio"].append(
            candidates.numel() / token_count
        )
        metrics[name]["global_candidate8_recall"].append(
            float(global_recall.item())
        )
        metrics[name]["sample_read_ratio"].append(sample_count / token_count)


def local_quota_candidates(
    scores: torch.Tensor,
    candidate_count: int,
    tile_size: int,
    multiplier: float,
) -> tuple[torch.Tensor, float]:
    heads, tokens = scores.shape
    tile_count = math.ceil(tokens / tile_size)
    padded_tokens = tile_count * tile_size
    if padded_tokens != tokens:
        scores = torch.nn.functional.pad(
            scores, (0, padded_tokens - tokens), value=-torch.inf
        )
    tiled = scores.reshape(heads, tile_count, tile_size)
    quota = min(
        tile_size,
        max(1, math.ceil(candidate_count / tile_count * multiplier)),
    )
    local = torch.topk(tiled, quota, dim=-1, sorted=False).indices
    offsets = (
        torch.arange(tile_count, device=scores.device).view(1, -1, 1) * tile_size
    )
    union = (local + offsets).reshape(heads, -1)
    valid = union < tokens
    if not bool(valid.all()):
        # Padding can only affect the last tile; replace invalid entries by index zero.
        union = union.masked_fill(~valid, 0)
    union_scores = torch.gather(scores[:, :tokens], -1, union)
    if union.shape[-1] > candidate_count:
        chosen = torch.topk(
            union_scores, candidate_count, dim=-1, sorted=False
        ).indices
        union = torch.gather(union, -1, chosen)
    return union, quota * tile_count / tokens


def certified_block_ratio(
    projected_query: torch.Tensor,
    projected_key: torch.Tensor,
    proxy_scores: torch.Tensor,
    candidate_count: int,
    block_size: int,
) -> torch.Tensor:
    kv_heads, tokens, rank = projected_key.shape
    query_heads = projected_query.shape[0]
    groups = query_heads // kv_heads
    block_count = math.ceil(tokens / block_size)
    padded_tokens = block_count * block_size
    if padded_tokens != tokens:
        pad_count = padded_tokens - tokens
        tail = projected_key[:, -1:].expand(-1, pad_count, -1)
        projected_key = torch.cat((projected_key, tail), dim=1)
    blocks = projected_key.reshape(kv_heads, block_count, block_size, rank)
    center = 0.5 * (blocks.amax(dim=2) + blocks.amin(dim=2))
    radius = 0.5 * (blocks.amax(dim=2) - blocks.amin(dim=2))
    grouped_query = projected_query.reshape(kv_heads, groups, rank)
    upper = (
        torch.einsum("hgd,hbd->hgb", grouped_query, center)
        + torch.einsum("hgd,hbd->hgb", grouped_query.abs(), radius)
    ).reshape(query_heads, block_count)
    threshold = torch.topk(
        proxy_scores, candidate_count, dim=-1, sorted=True
    ).values[:, -1]
    selected_blocks = upper >= threshold.unsqueeze(-1)
    selected_tokens = selected_blocks.sum(dim=-1) * block_size
    return selected_tokens.clamp_max(tokens).float() / tokens


def build_summary(
    metrics: dict[str, dict[str, list[float]]]
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        method: {name: summarize(values) for name, values in values_by_name.items()}
        for method, values_by_name in sorted(metrics.items())
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen training-free numerical pruning on real QKV traces."
    )
    parser.add_argument("--trace_paths", nargs="+", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="16")
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--ranks", default="32,48,64")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--candidate_fraction", type=float, default=0.08)
    parser.add_argument("--candidate_fractions", default="0.03,0.04,0.05,0.06,0.07,0.08")
    parser.add_argument("--tile_sizes", default="128,256,512,1024")
    parser.add_argument("--quota_multipliers", default="1,1.125,1.25,1.5")
    parser.add_argument("--block_sizes", default="16,32,64,128")
    parser.add_argument("--quantile_sample_counts", default="128,256,512,1024")
    parser.add_argument("--quantile_selected_fractions", default="0.10,0.12,0.14")
    args = parser.parse_args()

    ranks = parse_ints(args.ranks)
    layers = set(parse_ints(args.layers))
    tile_sizes = parse_ints(args.tile_sizes)
    quota_multipliers = parse_floats(args.quota_multipliers)
    block_sizes = parse_ints(args.block_sizes)
    quantile_sample_counts = parse_ints(args.quantile_sample_counts)
    quantile_selected_fractions = parse_floats(
        args.quantile_selected_fractions
    )
    candidate_fractions = parse_floats(args.candidate_fractions)
    if ranks[-1] > 128 or any(rank % 16 for rank in ranks):
        raise ValueError("ranks must be multiples of 16 and no larger than 128")
    if not 0.0 < args.top_fraction < args.candidate_fraction < 1.0:
        raise ValueError("fractions must satisfy 0 < top < candidate < 1")
    if any(
        not args.top_fraction < fraction < 1.0
        for fraction in candidate_fractions
    ):
        raise ValueError("candidate fractions must exceed the top fraction")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    block_metrics: dict[str, list[float]] = defaultdict(list)
    threshold_metrics: dict[str, list[float]] = defaultdict(list)
    case_count = 0

    for trace_path in args.trace_paths:
        payload = torch.load(trace_path, map_location="cpu", weights_only=False)
        topic = str(payload.get("config", {}).get("topic", trace_path.stem))
        records_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload["records"]:
            layer = int(record["layer"])
            if layer in layers:
                records_by_layer[layer].append(record)

        for layer, records in sorted(records_by_layer.items()):
            records.sort(key=lambda row: int(row.get("step", 0)))
            records = records[: args.max_steps]
            key_record = next(
                (record for record in records if record.get("key") is not None), None
            )
            if key_record is None:
                raise ValueError(f"trace {trace_path} layer {layer} has no key")
            all_key = key_record["key"].to(device).float()[0]
            history_count = int(all_key.shape[1]) - 1
            key = all_key[:, :history_count]
            scaling = float(key_record["scaling"])
            keep_count = max(1, math.ceil(args.top_fraction * history_count))
            candidate_count = max(
                keep_count, math.ceil(args.candidate_fraction * history_count)
            )

            rank_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            for rank in ranks:
                basis = sampled_basis(key, rank, args.sample_stride)
                projected_key = torch.einsum("hnd,hdr->hnr", key, basis)
                indexed_key = logscale16_int4_dequantize(projected_key)
                rank_states[rank] = (basis, indexed_key)

            previous_standardized_threshold: torch.Tensor | None = None
            for record in records:
                query = record["query"].to(device).float()[0, :, 0, :]
                kv_heads = int(key.shape[0])
                groups = int(query.shape[0]) // kv_heads
                grouped_query = query.reshape(kv_heads, groups, query.shape[-1])
                exact_scores = grouped_scores(query, key) * scaling
                attention = torch.softmax(exact_scores, dim=-1)
                oracle = torch.topk(
                    exact_scores, keep_count, dim=-1, sorted=False
                ).indices
                oracle_mass = torch.gather(attention, -1, oracle).sum(dim=-1)

                for rank, (basis, indexed_key) in rank_states.items():
                    projected_query = torch.einsum(
                        "hgd,hdr->hgr", grouped_query, basis
                    ).reshape(query.shape[0], rank)
                    projected_query = quantize_query_int8(projected_query)
                    proxy_scores = grouped_scores(projected_query, indexed_key) * scaling
                    candidates = None
                    for candidate_fraction in candidate_fractions:
                        frontier_candidate_count = max(
                            keep_count,
                            math.ceil(candidate_fraction * history_count),
                        )
                        frontier_candidates = torch.topk(
                            proxy_scores,
                            frontier_candidate_count,
                            dim=-1,
                            sorted=False,
                        ).indices
                        selected = exact_rerank(
                            exact_scores, frontier_candidates, keep_count
                        )
                        append_quality(
                            metrics,
                            f"global_rank{rank}_candidate{candidate_fraction:g}",
                            selected,
                            oracle,
                            attention,
                            oracle_mass,
                            frontier_candidate_count / history_count,
                            rank / ranks[-1],
                        )
                        if math.isclose(
                            candidate_fraction,
                            args.candidate_fraction,
                            rel_tol=0.0,
                            abs_tol=1.0e-9,
                        ):
                            candidates = frontier_candidates
                    if candidates is None:
                        candidates = torch.topk(
                            proxy_scores,
                            candidate_count,
                            dim=-1,
                            sorted=False,
                        ).indices

                    if rank != ranks[-1]:
                        continue

                    for sample_count in quantile_sample_counts:
                        for selected_fraction in quantile_selected_fractions:
                            append_sampled_quantile_quality(
                                metrics,
                                f"sampleq_s{sample_count}_f{selected_fraction:g}",
                                exact_scores,
                                proxy_scores,
                                candidates,
                                oracle,
                                attention,
                                oracle_mass,
                                keep_count,
                                sample_count,
                                selected_fraction,
                            )

                    for tile_size in tile_sizes:
                        for multiplier in quota_multipliers:
                            local_candidates, union_ratio = local_quota_candidates(
                                proxy_scores,
                                candidate_count,
                                tile_size,
                                multiplier,
                            )
                            local_selected = exact_rerank(
                                exact_scores, local_candidates, keep_count
                            )
                            append_quality(
                                metrics,
                                f"local_tile{tile_size}_x{multiplier:g}",
                                local_selected,
                                oracle,
                                attention,
                                oracle_mass,
                                union_ratio,
                                1.0,
                            )

                    for block_size in block_sizes:
                        ratio = certified_block_ratio(
                            projected_query,
                            indexed_key,
                            proxy_scores,
                            candidate_count,
                            block_size,
                        )
                        block_metrics[f"block{block_size}_certified_scan_ratio"].extend(
                            ratio.cpu().tolist()
                        )

                    score_variance = torch.einsum(
                        "hgd,hde,hge->hg",
                        grouped_query,
                        torch.einsum(
                            "hnd,hne->hde",
                            key[:, :: args.sample_stride],
                            key[:, :: args.sample_stride],
                        )
                        / key[:, :: args.sample_stride].shape[1],
                        grouped_query,
                    ).reshape(query.shape[0]).clamp_min(1.0e-12)
                    score_sigma = score_variance.sqrt() * scaling
                    current_threshold = torch.topk(
                        proxy_scores, candidate_count, dim=-1, sorted=True
                    ).values[:, -1]
                    standardized = current_threshold / score_sigma
                    if previous_standardized_threshold is not None:
                        predicted = previous_standardized_threshold * score_sigma
                        threshold_mask = proxy_scores >= predicted.unsqueeze(-1)
                        selected_ratio = threshold_mask.float().mean(dim=-1)
                        candidate_hits = torch.gather(
                            threshold_mask, -1, candidates
                        ).float().mean(dim=-1)
                        threshold_metrics["previous_standardized_selected_ratio"].extend(
                            selected_ratio.cpu().tolist()
                        )
                        threshold_metrics["previous_standardized_global_candidate_recall"].extend(
                            candidate_hits.cpu().tolist()
                        )
                    previous_standardized_threshold = standardized

                case_count += int(query.shape[0])
                print(
                    f"trace={trace_path.name} topic={topic} layer={layer} "
                    f"step={record.get('step', '?')} cases={case_count}",
                    flush=True,
                )

            del all_key, key, rank_states
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = {
        "config": {
            "trace_paths": [str(path) for path in args.trace_paths],
            "layers": sorted(layers),
            "max_steps": args.max_steps,
            "ranks": ranks,
            "top_fraction": args.top_fraction,
            "candidate_fraction": args.candidate_fraction,
            "candidate_fractions": candidate_fractions,
            "tile_sizes": tile_sizes,
            "quota_multipliers": quota_multipliers,
            "block_sizes": block_sizes,
            "quantile_sample_counts": quantile_sample_counts,
            "quantile_selected_fractions": quantile_selected_fractions,
        },
        "head_cases": case_count,
        "retrieval": build_summary(metrics),
        "certified_block_pruning": {
            name: summarize(values) for name, values in sorted(block_metrics.items())
        },
        "previous_threshold_compaction": {
            name: summarize(values)
            for name, values in sorted(threshold_metrics.items())
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
