#!/usr/bin/env python
"""Validate per-query JointKV selection on one real Qwen attention layer."""

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
from analyze_qaware_binarypc_blockmean_layer0_20260802 import (
    load_layer0_activations,
)
from jointkv_global_threshold_cuda_20260802 import (
    allocate_query_workspace,
    allocate_workspace,
    project_query_lut_out,
    sampled_threshold_compact_out,
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
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument("--overfetch", type=float, default=1.0)
    parser.add_argument("--capacity_multiplier", type=float, default=2.5)
    parser.add_argument("--sparse_split_count", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selected_count(tokens: int) -> int:
    return min(tokens, 1280, max(256, math.ceil(0.06 * tokens)))


def encode_repeated(
    tokenizer: AutoTokenizer,
    text_path: Path,
    needed: int,
) -> torch.Tensor:
    text = text_path.read_text(encoding="utf-8")
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if ids.numel() < needed:
        ids = ids.repeat(math.ceil(needed / max(1, ids.numel())))
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


def relative_l2(reference: torch.Tensor, observed: torch.Tensor) -> float:
    return float(
        (observed.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1.0e-12)
    )


def main() -> None:
    args = parse_args()
    if args.history_tokens < 257:
        raise ValueError("history must contain a prefix and one query token")
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = encode_repeated(tokenizer, args.text, args.history_tokens)
    query_cpu, key_cpu, value_cpu, metadata = load_layer0_activations(args.model, ids)
    query = query_cpu[-1].unsqueeze(0).to(args.device, dtype=dtype).contiguous()
    key = key_cpu.permute(1, 0, 2).unsqueeze(0).to(
        args.device, dtype=dtype
    ).contiguous()
    value = value_cpu.permute(1, 0, 2).unsqueeze(0).to(
        args.device, dtype=dtype
    ).contiguous()
    prefix_key = key[:, :, :-1].contiguous()
    prefix_value = value[:, :, :-1].contiguous()
    prefix_tokens = prefix_key.shape[2]
    cached = torch.load(args.codebook_cache, map_location="cpu", weights_only=False)
    index = build_real_index(
        prefix_key,
        prefix_value,
        cached["codebooks"][0],
        risk_mode="qk_risk",
        risk_lambda=0.0,
    )
    target_count = selected_count(prefix_tokens)
    capacity = min(
        prefix_tokens,
        max(target_count, math.ceil(target_count * args.capacity_multiplier)),
    )
    groups = query.shape[1] // key.shape[1]
    packed_query = prepare_packed_query(query, index)
    expanded_query_matrix = (
        index.query_matrix.repeat_interleave(groups, dim=0)
        .to(query.dtype)
        .contiguous()
    )
    fused_query_matrix = index.query_matrix.to(query.dtype).contiguous()
    fused_query_workspace = allocate_query_workspace(query, fused_query_matrix)

    def prepare_query_bmm() -> torch.Tensor:
        return torch.bmm(
            query[0].unsqueeze(1), expanded_query_matrix
        ).squeeze(1).reshape(
            1, key.shape[1], groups, expanded_query_matrix.shape[-1]
        )

    def prepare_query_fused() -> tuple[torch.Tensor, torch.Tensor]:
        return project_query_lut_out(
            query,
            fused_query_matrix,
            fused_query_workspace,
            base_offset=BASE_OFFSET,
            residual_offset=RESIDUAL_OFFSET,
            base_chunks=BASE_BITS // 8,
            residual_chunks=RESIDUAL_BITS // 8,
        )

    query_lut = joint_cuda.build_query_lut(
        packed_query,
        base_bits=BASE_BITS,
        residual_bits=RESIDUAL_BITS,
        base_offset=BASE_OFFSET,
        residual_offset=RESIDUAL_OFFSET,
    )
    workspace = allocate_workspace(packed_query, capacity)
    fused_packed_query, fused_query_lut = prepare_query_fused()
    fused_packed_max_error = float(
        (fused_packed_query.float() - packed_query.float()).abs().max()
    )
    fused_lut_max_error = float(
        (fused_query_lut - query_lut).abs().max()
    )

    def select() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return sampled_threshold_compact_out(
            packed_query,
            query_lut,
            index.base_codes,
            index.residual_codes,
            index.joint_ids,
            workspace,
            sample_count=args.sample_count,
            selected_fraction=target_count / prefix_tokens,
            overfetch=args.overfetch,
            joint_offset=JOINT_OFFSET,
        )

    indices, counts, thresholds, overflow = select()
    sparse_output = sparse_cuda.final_attention_ragged_self_split(
        query,
        key,
        value,
        indices,
        counts,
        query.shape[-1] ** -0.5,
        args.sparse_split_count,
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

    attention_mass = []
    recalls = []
    top1_covered = []
    prefix_scores = full_scores[..., :prefix_tokens]
    oracle = torch.topk(
        prefix_scores, k=target_count, dim=-1, sorted=False
    ).indices
    for head in range(query.shape[1]):
        count = int(counts[0, head])
        selected = indices[0, head, :count]
        # The sparse attention kernel always adds the final self token.
        selected_with_self = torch.cat(
            (
                selected,
                torch.tensor(
                    [args.history_tokens - 1],
                    dtype=torch.int64,
                    device=selected.device,
                ),
            )
        ).unique()
        attention_mass.append(
            float(full_probability[0, head].index_select(0, selected_with_self).sum())
        )
        recalls.append(
            float(torch.isin(selected, oracle[0, head]).float().mean())
            if count
            else 0.0
        )
        top1_covered.append(
            float(torch.isin(full_scores[0, head].argmax()[None], selected_with_self)[0])
        )

    def full_attention() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query.unsqueeze(2), expanded_key, expanded_value, is_causal=False
        )

    def selection_only() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return select()

    def query_projection_only() -> torch.Tensor:
        return prepare_packed_query(query, index)

    def query_projection_bmm_only() -> torch.Tensor:
        return prepare_query_bmm()

    def fused_query_encoding_only() -> tuple[torch.Tensor, torch.Tensor]:
        return prepare_query_fused()

    def lut_only() -> torch.Tensor:
        return joint_cuda.build_query_lut(
            packed_query,
            base_bits=BASE_BITS,
            residual_bits=RESIDUAL_BITS,
            base_offset=BASE_OFFSET,
            residual_offset=RESIDUAL_OFFSET,
        )

    def sparse_only() -> torch.Tensor:
        return sparse_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            indices,
            counts,
            query.shape[-1] ** -0.5,
            args.sparse_split_count,
        )

    def complete() -> torch.Tensor:
        current_indices, current_counts, _, _ = select()
        return sparse_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            current_indices,
            current_counts,
            query.shape[-1] ** -0.5,
            args.sparse_split_count,
        )

    def complete_with_query_encoding() -> torch.Tensor:
        current_query = prepare_packed_query(query, index)
        current_lut = joint_cuda.build_query_lut(
            current_query,
            base_bits=BASE_BITS,
            residual_bits=RESIDUAL_BITS,
            base_offset=BASE_OFFSET,
            residual_offset=RESIDUAL_OFFSET,
        )
        current_indices, current_counts, _, _ = sampled_threshold_compact_out(
            current_query,
            current_lut,
            index.base_codes,
            index.residual_codes,
            index.joint_ids,
            workspace,
            sample_count=args.sample_count,
            selected_fraction=target_count / prefix_tokens,
            overfetch=args.overfetch,
            joint_offset=JOINT_OFFSET,
        )
        return sparse_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            current_indices,
            current_counts,
            query.shape[-1] ** -0.5,
            args.sparse_split_count,
        )

    def complete_with_bmm_query_encoding() -> torch.Tensor:
        current_query = prepare_query_bmm()
        current_lut = joint_cuda.build_query_lut(
            current_query,
            base_bits=BASE_BITS,
            residual_bits=RESIDUAL_BITS,
            base_offset=BASE_OFFSET,
            residual_offset=RESIDUAL_OFFSET,
        )
        current_indices, current_counts, _, _ = sampled_threshold_compact_out(
            current_query,
            current_lut,
            index.base_codes,
            index.residual_codes,
            index.joint_ids,
            workspace,
            sample_count=args.sample_count,
            selected_fraction=target_count / prefix_tokens,
            overfetch=args.overfetch,
            joint_offset=JOINT_OFFSET,
        )
        return sparse_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            current_indices,
            current_counts,
            query.shape[-1] ** -0.5,
            args.sparse_split_count,
        )

    def complete_with_fused_query_encoding() -> torch.Tensor:
        current_query, current_lut = prepare_query_fused()
        current_indices, current_counts, _, _ = sampled_threshold_compact_out(
            current_query,
            current_lut,
            index.base_codes,
            index.residual_codes,
            index.joint_ids,
            workspace,
            sample_count=args.sample_count,
            selected_fraction=target_count / prefix_tokens,
            overfetch=args.overfetch,
            joint_offset=JOINT_OFFSET,
        )
        return sparse_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            current_indices,
            current_counts,
            query.shape[-1] ** -0.5,
            args.sparse_split_count,
        )

    iterations = min(args.iterations, 20 if args.history_tokens >= 32768 else args.iterations)
    warmup = min(args.warmup, iterations)
    full_ms = measure_ms(full_attention, warmup, iterations)
    query_projection_ms = measure_ms(query_projection_only, warmup, iterations)
    query_projection_bmm_ms = measure_ms(
        query_projection_bmm_only, warmup, iterations
    )
    fused_query_encoding_ms = measure_ms(
        fused_query_encoding_only, warmup, iterations
    )
    lut_ms = measure_ms(lut_only, warmup, iterations)
    selection_ms = measure_ms(selection_only, warmup, iterations)
    sparse_ms = measure_ms(sparse_only, warmup, iterations)
    complete_ms = measure_ms(complete, warmup, iterations)
    complete_with_query_ms = measure_ms(
        complete_with_query_encoding, warmup, iterations
    )
    complete_with_bmm_query_ms = measure_ms(
        complete_with_bmm_query_encoding, warmup, iterations
    )
    complete_with_fused_query_ms = measure_ms(
        complete_with_fused_query_encoding, warmup, iterations
    )
    count_values = counts.float().flatten()
    payload = {
        "schema": "jointkv-global-real-layer-cuda-v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "setup": {
            "model": str(args.model),
            "text": str(args.text),
            "history_tokens_including_query": args.history_tokens,
            "indexed_prefix_tokens": prefix_tokens,
            "target_candidates_per_query_head": target_count,
            "target_fraction": target_count / prefix_tokens,
            "sample_count": args.sample_count,
            "overfetch": args.overfetch,
            "capacity": capacity,
            "dtype": args.dtype,
            "selector": "per-query sampled global threshold; no local warp quota",
            "tail_correction": False,
            "exact_attention": True,
        },
        "index": {
            "build_seconds": index.build_seconds,
            "physical_bytes_per_token_kv_head": index.physical_bytes_per_token_head,
            "logical_bits_per_token_kv_head": index.logical_bits_per_token_head,
        },
        "quality": {
            "selected_attention_mass_mean": statistics.mean(attention_mass),
            "selected_attention_mass_min": min(attention_mass),
            "oracle_topk_recall_mean": statistics.mean(recalls),
            "oracle_topk_recall_min": min(recalls),
            "top1_covered_fraction": statistics.mean(top1_covered),
            "attention_output_relative_l2": relative_l2(full_output, sparse_output),
            "finite": bool(torch.isfinite(sparse_output).all()),
            "fused_packed_query_max_abs_error": fused_packed_max_error,
            "fused_query_lut_max_abs_error": fused_lut_max_error,
        },
        "candidate_counts": {
            "mean": float(count_values.mean()),
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean_fraction": float(count_values.mean() / prefix_tokens),
            "overflow_heads": int(overflow.sum()),
        },
        "speed": {
            "scope": "one real layer decode Attention; direct CUDA event timing; index build excluded",
            "full_attention_ms": full_ms,
            "query_projection_ms": query_projection_ms,
            "query_projection_bmm_ms": query_projection_bmm_ms,
            "fused_query_projection_and_lut_ms": fused_query_encoding_ms,
            "query_lut_ms": lut_ms,
            "selector_ms": selection_ms,
            "exact_sparse_attention_ms": sparse_ms,
            "complete_sparse_ms": complete_ms,
            "complete_speedup": full_ms / complete_ms,
            "complete_with_query_encoding_ms": complete_with_query_ms,
            "complete_with_query_encoding_speedup": full_ms / complete_with_query_ms,
            "complete_with_bmm_query_encoding_ms": complete_with_bmm_query_ms,
            "complete_with_bmm_query_encoding_speedup": (
                full_ms / complete_with_bmm_query_ms
            ),
            "complete_with_fused_query_encoding_ms": complete_with_fused_query_ms,
            "complete_with_fused_query_encoding_speedup": (
                full_ms / complete_with_fused_query_ms
            ),
        },
        "thresholds": {
            "mean": float(thresholds.mean()),
            "min": float(thresholds.min()),
            "max": float(thresholds.max()),
        },
        "model_metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
