#!/usr/bin/env python
"""Replay the JointKV CUDA path on real Qwen layer-0 Q/K/V tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import jointkv_sieve_cuda_20260802 as joint_cuda
import qabs_cuda_kernels as sparse_cuda
from analyze_qksieve_control_variate_layer0_probe_20260802 import (
    load_layer0_activations,
)
from jointkv_real_index_20260802 import (
    BASE_BITS,
    BASE_OFFSET,
    JOINT_OFFSET,
    RESIDUAL_BITS,
    RESIDUAL_OFFSET,
    build_real_index,
    prepare_packed_query,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codebook_cache", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--history_tokens", type=int, default=8192)
    parser.add_argument("--local_base_keep", type=int, default=8)
    parser.add_argument("--sparse_split_count", type=int, default=8)
    parser.add_argument("--risk_mode", choices=("qk_risk", "output_bound"), default="output_bound")
    parser.add_argument("--risk_lambda", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selected_count(tokens: int) -> int:
    return min(tokens, 1280, max(256, math.ceil(0.06 * tokens)))


def next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def residual_keep(tokens: int, base_keep: int, exact_count: int) -> int:
    base_candidates = math.ceil(tokens / 32) * base_keep
    candidate_warps = math.ceil(base_candidates / 32)
    return min(
        32,
        next_power_of_two(max(4, math.ceil(2 * exact_count / candidate_warps))),
    )


def encode_repeated(
    tokenizer: AutoTokenizer,
    text_path: Path,
    needed: int,
) -> torch.Tensor:
    text = text_path.read_text(encoding="utf-8")
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if ids.numel() < needed:
        repeats = math.ceil(needed / max(1, ids.numel()))
        ids = ids.repeat(repeats)
    return ids[:needed].contiguous()


def measure_ms(function: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    stop.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(stop)) / iterations


def sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_indices: torch.Tensor,
    split_count: int,
) -> torch.Tensor:
    group_size = query.shape[1] // key.shape[1]
    query_indices = kv_indices.repeat_interleave(group_size, dim=1).contiguous()
    counts = torch.full(
        query_indices.shape[:2],
        query_indices.shape[-1],
        dtype=torch.long,
        device=query.device,
    )
    return sparse_cuda.final_attention_ragged_self_split(
        query,
        key,
        value,
        query_indices,
        counts,
        query.shape[-1] ** -0.5,
        split_count,
    )


def relative_l2(reference: torch.Tensor, observed: torch.Tensor) -> float:
    return float(
        (observed.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1.0e-12)
    )


def unpack_signs(codes: torch.Tensor, bits: int) -> torch.Tensor:
    bit_ids = torch.arange(bits, dtype=torch.int64, device=codes.device)
    return torch.where(
        ((codes[..., None] >> bit_ids) & 1) != 0,
        1.0,
        -1.0,
    )


def selection_metrics(
    scores: torch.Tensor,
    probability: torch.Tensor,
    selected: torch.Tensor,
) -> dict[str, float]:
    mass = torch.gather(probability, -1, selected).sum(dim=-1)
    oracle = torch.topk(scores, k=selected.shape[-1], dim=-1, sorted=False).indices
    recalls = []
    for head in range(scores.shape[1]):
        for group in range(scores.shape[2]):
            recalls.append(
                float(
                    torch.isin(
                        selected[0, head, group], oracle[0, head, group]
                    ).float().mean()
                )
            )
    return {
        "topk_recall_mean": statistics.mean(recalls),
        "topk_recall_min": min(recalls),
        "attention_mass_mean": float(mass.mean()),
        "attention_mass_min": float(mass.min()),
    }


def main() -> None:
    args = parse_args()
    if args.history_tokens < 256:
        raise ValueError("history must contain at least 256 tokens")
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = encode_repeated(tokenizer, args.text, args.history_tokens)
    query_cpu, key_cpu, value_cpu, metadata = load_layer0_activations(args.model, ids)
    query = query_cpu[-1].unsqueeze(0).to(args.device, dtype=dtype).contiguous()
    key = key_cpu
    key = key.permute(1, 0, 2).unsqueeze(0).to(args.device, dtype=dtype).contiguous()
    value = value_cpu.permute(1, 0, 2).unsqueeze(0).to(
        args.device, dtype=dtype
    ).contiguous()
    cached = torch.load(args.codebook_cache, map_location="cpu", weights_only=False)
    layer_codebooks = cached["codebooks"][0]
    index = build_real_index(
        key,
        value,
        layer_codebooks,
        risk_mode=args.risk_mode,
        risk_lambda=args.risk_lambda,
    )
    exact_count = selected_count(args.history_tokens)
    local_residual_keep = residual_keep(
        args.history_tokens, args.local_base_keep, exact_count
    )
    groups = query.shape[1] // key.shape[1]
    references = torch.zeros(
        1, key.shape[1], groups, dtype=torch.float32, device=query.device
    )

    def selection() -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        packed_query = prepare_packed_query(query, index)
        query_lut = joint_cuda.build_query_lut(
            packed_query,
            base_bits=BASE_BITS,
            residual_bits=RESIDUAL_BITS,
            base_offset=BASE_OFFSET,
            residual_offset=RESIDUAL_OFFSET,
        )
        base_indices, base_scores, cluster_mass = (
            joint_cuda.base_lut_local_select_and_mass(
                index.base_codes,
                index.joint_ids,
                index.risk_codes,
                packed_query,
                query_lut,
                index.risk_lut,
                references,
                base_chunks=BASE_BITS // 8,
                joint_offset=JOINT_OFFSET,
                keep_per_warp=args.local_base_keep,
            )
        )
        shortlist_indices, shortlist_scores = (
            joint_cuda.residual_lut_local_shortlist(
                index.residual_codes,
                base_indices,
                base_scores,
                query_lut,
                residual_chunk_offset=BASE_BITS // 8,
                residual_chunks=RESIDUAL_BITS // 8,
                keep_per_warp=local_residual_keep,
            )
        )
        local = torch.topk(
            shortlist_scores, k=exact_count, dim=-1, sorted=False
        ).indices
        selected = torch.gather(shortlist_indices, 2, local)
        return selected, packed_query, query_lut, cluster_mass

    selected, packed_query, query_lut, cluster_mass = selection()
    sparse_output = sparse_attention(
        query, key, value, selected, args.sparse_split_count
    )
    omitted_mass = joint_cuda.subtract_selected_mass(
        cluster_mass.clone(),
        index.base_codes,
        index.joint_ids,
        packed_query,
        selected,
        references,
        bits=BASE_BITS,
        probe_offset=BASE_OFFSET,
        joint_offset=JOINT_OFFSET,
    )
    tail_output = joint_cuda.tail_blend(
        sparse_output,
        omitted_mass,
        index.value_centroids,
        exact_count / args.history_tokens,
    )

    expanded_key = key.repeat_interleave(groups, dim=1)
    expanded_value = value.repeat_interleave(groups, dim=1)
    full_scores = torch.einsum(
        "bhd,bhnd->bhn", query.float(), expanded_key.float()
    ) * (query.shape[-1] ** -0.5)
    full_probability = torch.softmax(full_scores, dim=-1)
    full_output = torch.einsum(
        "bhn,bhnd->bhd", full_probability, expanded_value.float()
    )
    selected_per_query = selected.repeat_interleave(groups, dim=1)
    selected_mass = torch.gather(
        full_probability, 2, selected_per_query
    ).sum(dim=-1)
    oracle = torch.topk(
        full_scores, k=exact_count, dim=-1, sorted=False
    ).indices
    recalls = []
    for head in range(query.shape[1]):
        recalls.append(
            float(
                torch.isin(selected_per_query[0, head], oracle[0, head])
                .float()
                .mean()
            )
        )
    top1 = full_scores.argmax(dim=-1)
    top1_covered = torch.stack(
        [
            (selected_per_query[0, head] == top1[0, head]).any()
            for head in range(query.shape[1])
        ]
    ).float()

    # Diagnose where quality is lost: score coding, GQA sharing, or local
    # shortlist.  These tensors are used only for diagnosis, not speed timing.
    grouped_full_scores = full_scores.reshape(
        1, key.shape[1], groups, args.history_tokens
    )
    grouped_probability = full_probability.reshape_as(grouped_full_scores)
    base_signs = unpack_signs(index.base_codes[0], BASE_BITS)
    residual_signs = unpack_signs(index.residual_codes[0], RESIDUAL_BITS)
    base_probe = packed_query[0, :, :, BASE_OFFSET : BASE_OFFSET + BASE_BITS].float()
    residual_probe = packed_query[
        0, :, :, RESIDUAL_OFFSET : RESIDUAL_OFFSET + RESIDUAL_BITS
    ].float()
    joint_table = packed_query[
        0, :, :, JOINT_OFFSET : JOINT_OFFSET + 64
    ].float()
    direct_base = torch.einsum("hnb,hgb->hgn", base_signs, base_probe)
    direct_base = direct_base + torch.gather(
        joint_table,
        2,
        index.joint_ids[0, :, None, :].long().expand(-1, groups, -1),
    )
    direct_refined = direct_base + torch.einsum(
        "hnb,hgb->hgn", residual_signs, residual_probe
    )
    per_query_proxy = torch.topk(
        direct_refined, k=exact_count, dim=-1, sorted=False
    ).indices.unsqueeze(0)
    shared_proxy_ids = torch.topk(
        direct_refined.max(dim=1).values,
        k=exact_count,
        dim=-1,
        sorted=False,
    ).indices
    shared_proxy = shared_proxy_ids[:, None, :].expand(-1, groups, -1).unsqueeze(0)
    centered_proxy_scores = direct_refined - torch.logsumexp(
        direct_refined, dim=-1, keepdim=True
    )
    centered_shared_proxy_ids = torch.topk(
        centered_proxy_scores.max(dim=1).values,
        k=exact_count,
        dim=-1,
        sorted=False,
    ).indices
    centered_shared_proxy = (
        centered_shared_proxy_ids[:, None, :]
        .expand(-1, groups, -1)
        .unsqueeze(0)
    )
    centered_exact_scores = grouped_full_scores[0] - torch.logsumexp(
        grouped_full_scores[0], dim=-1, keepdim=True
    )
    centered_exact_ids = torch.topk(
        centered_exact_scores.max(dim=1).values,
        k=exact_count,
        dim=-1,
        sorted=False,
    ).indices
    centered_exact_shared = (
        centered_exact_ids[:, None, :].expand(-1, groups, -1).unsqueeze(0)
    )
    local_shared = selected[:, :, None, :].expand(-1, -1, groups, -1)
    proxy_per_query_metrics = selection_metrics(
        grouped_full_scores, grouped_probability, per_query_proxy
    )
    proxy_shared_metrics = selection_metrics(
        grouped_full_scores, grouped_probability, shared_proxy
    )
    centered_proxy_shared_metrics = selection_metrics(
        grouped_full_scores, grouped_probability, centered_shared_proxy
    )
    centered_exact_shared_metrics = selection_metrics(
        grouped_full_scores, grouped_probability, centered_exact_shared
    )
    local_shared_metrics = selection_metrics(
        grouped_full_scores, grouped_probability, local_shared
    )
    diagnostic_base_indices, _, _ = joint_cuda.base_lut_local_select_and_mass(
        index.base_codes,
        index.joint_ids,
        index.risk_codes,
        packed_query,
        query_lut,
        index.risk_lut,
        references,
        base_chunks=BASE_BITS // 8,
        joint_offset=JOINT_OFFSET,
        keep_per_warp=args.local_base_keep,
    )
    correct_shared_scores = direct_refined.max(dim=1).values
    base_candidate_scores = torch.gather(
        correct_shared_scores, 1, diagnostic_base_indices[0]
    )
    base_global_local = torch.topk(
        base_candidate_scores, k=exact_count, dim=-1, sorted=False
    ).indices
    base_global_ids = torch.gather(
        diagnostic_base_indices[0], 1, base_global_local
    )
    base_global_shared = base_global_ids[:, None, :].expand(-1, groups, -1).unsqueeze(0)
    corrected_warp_scores = base_candidate_scores.reshape(
        key.shape[1], -1, 32
    )
    corrected_warp_positions = torch.topk(
        corrected_warp_scores,
        k=local_residual_keep,
        dim=-1,
        sorted=False,
    ).indices
    corrected_warp_ids = torch.gather(
        diagnostic_base_indices[0].reshape(key.shape[1], -1, 32),
        2,
        corrected_warp_positions,
    ).reshape(key.shape[1], -1)
    corrected_short_scores = torch.gather(
        correct_shared_scores, 1, corrected_warp_ids
    )
    corrected_final_positions = torch.topk(
        corrected_short_scores, k=exact_count, dim=-1, sorted=False
    ).indices
    corrected_ids = torch.gather(
        corrected_warp_ids, 1, corrected_final_positions
    )
    corrected_shared = corrected_ids[:, None, :].expand(-1, groups, -1).unsqueeze(0)
    base_candidate_global_metrics = selection_metrics(
        grouped_full_scores, grouped_probability, base_global_shared
    )
    corrected_local_metrics = selection_metrics(
        grouped_full_scores, grouped_probability, corrected_shared
    )
    correlations = []
    for head in range(key.shape[1]):
        for group in range(groups):
            correlations.append(
                float(
                    torch.corrcoef(
                        torch.stack(
                            (
                                direct_refined[head, group],
                                grouped_full_scores[0, head, group],
                            )
                        )
                    )[0, 1]
                )
            )

    selected_exact_scores = torch.gather(
        full_scores, 2, selected_per_query
    ).float()
    selected_denominator = selected_exact_scores.exp().sum(dim=-1).reshape(
        1, key.shape[1], groups
    )
    tail_denominator = omitted_mass.clamp_min(0).sum(dim=-1)
    tail_numerator = torch.einsum(
        "bhgc,hcd->bhgd", omitted_mass.clamp_min(0), index.value_centroids
    )
    selected_numerator = (
        sparse_output[:, 0].float().reshape(1, key.shape[1], groups, -1)
        * selected_denominator[..., None]
    )
    mass_tail_output = (
        selected_numerator + tail_numerator
    ) / (selected_denominator + tail_denominator).clamp_min(1.0e-8)[..., None]
    mass_tail_output = mass_tail_output.reshape(1, query.shape[1], query.shape[-1])

    def full_attention() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query.unsqueeze(2), expanded_key, expanded_value, is_causal=False
        )

    def selected_only_complete() -> torch.Tensor:
        current, _, _, _ = selection()
        return sparse_attention(
            query, key, value, current, args.sparse_split_count
        )

    def tail_complete() -> torch.Tensor:
        current, current_query, _, current_mass = selection()
        current_sparse = sparse_attention(
            query, key, value, current, args.sparse_split_count
        )
        current_mass = joint_cuda.subtract_selected_mass(
            current_mass,
            index.base_codes,
            index.joint_ids,
            current_query,
            current,
            references,
            bits=BASE_BITS,
            probe_offset=BASE_OFFSET,
            joint_offset=JOINT_OFFSET,
        )
        return joint_cuda.tail_blend(
            current_sparse,
            current_mass,
            index.value_centroids,
            exact_count / args.history_tokens,
        )

    iterations = min(args.iterations, 20 if args.history_tokens >= 32768 else args.iterations)
    warmup = min(args.warmup, iterations)
    full_ms = measure_ms(full_attention, warmup, iterations)
    selected_ms = measure_ms(selected_only_complete, warmup, iterations)
    tail_ms = measure_ms(tail_complete, warmup, iterations)
    base_candidates = math.ceil(args.history_tokens / 32) * args.local_base_keep
    shortlist = math.ceil(base_candidates / 32) * local_residual_keep
    physical_index_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            index.base_codes,
            index.residual_codes,
            index.joint_ids,
            index.risk_codes,
        )
    )
    payload = {
        "schema": "jointkv-real-layer-cuda-v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "setup": {
            "model": str(args.model),
            "codebook_cache": str(args.codebook_cache),
            "text": str(args.text),
            "history_tokens": args.history_tokens,
            "layer": 0,
            "query_heads": int(query.shape[1]),
            "kv_heads": int(key.shape[1]),
            "gqa_groups": groups,
            "head_dim": int(query.shape[-1]),
            "dtype": args.dtype,
            "exact_tokens_per_kv_head": exact_count,
            "exact_fraction": exact_count / args.history_tokens,
            "local_base_keep_per_32": args.local_base_keep,
            "base_candidates_per_kv_head": base_candidates,
            "local_residual_keep_per_32": local_residual_keep,
            "shortlist_per_kv_head": shortlist,
            "risk_mode": args.risk_mode,
            "quality_uses_real_post_rope_qkv": True,
            "index_values_are_random": False,
            "index_build_in_complete_attention_timing": False,
        },
        "index": {
            "build_seconds": index.build_seconds,
            "logical_bits_per_token_kv_head": index.logical_bits_per_token_head,
            "physical_bytes_per_token_kv_head": index.physical_bytes_per_token_head,
            "physical_primary_index_bytes": physical_index_bytes,
            "physical_primary_index_mib": physical_index_bytes / 2**20,
        },
        "quality": {
            "exact_topk_recall_mean": statistics.mean(recalls),
            "exact_topk_recall_min": min(recalls),
            "selected_full_attention_mass_mean": float(selected_mass.mean()),
            "selected_full_attention_mass_min": float(selected_mass.min()),
            "exact_top1_covered_fraction": float(top1_covered.mean()),
            "selected_only_output_relative_l2": relative_l2(
                full_output, sparse_output[:, 0]
            ),
            "tail_output_relative_l2": relative_l2(
                full_output, tail_output[:, 0]
            ),
            "mass_normalized_tail_output_relative_l2": relative_l2(
                full_output, mass_tail_output
            ),
            "proxy_score_pearson_mean": statistics.mean(correlations),
            "proxy_score_pearson_min": min(correlations),
            "global_per_query_proxy": proxy_per_query_metrics,
            "global_shared_gqa_proxy": proxy_shared_metrics,
            "global_centered_shared_gqa_proxy": centered_proxy_shared_metrics,
            "exact_centered_shared_gqa_upper_bound": centered_exact_shared_metrics,
            "base_local_candidates_then_correct_global_score": (
                base_candidate_global_metrics
            ),
            "correct_same_group_residual_local_proxy": corrected_local_metrics,
            "cuda_local_shared_proxy": local_shared_metrics,
            "finite_selected_only": bool(torch.isfinite(sparse_output).all()),
            "finite_tail": bool(torch.isfinite(tail_output).all()),
        },
        "speed": {
            "scope": "one real layer decode Attention; direct complete-call timing; preexpanded Full SDPA; index build excluded",
            "full_ms": full_ms,
            "selected_only_ms": selected_ms,
            "tail_ms": tail_ms,
            "selected_only_speedup": full_ms / selected_ms,
            "tail_speedup": full_ms / tail_ms,
        },
        "model_metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
