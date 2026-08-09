from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoConfig

import qabs_cuda_kernels
from run_head_top2_targeted_ppl_20260714 import (
    _configured_rope_phase_tables,
    _inverse_standard_rope,
    _standard_rope_phase_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-count", type=int, default=32768)
    parser.add_argument("--candidate-count", type=int, default=2272)
    parser.add_argument("--target-count", type=int, default=1280)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rope-theta", type=float, default=5_000_000.0)
    parser.add_argument(
        "--model-config",
        default="",
        help="Optional local model/config path for the exact configured RoPE.",
    )
    parser.add_argument("--original-max-position", type=int, default=0)
    parser.add_argument("--global-max-position", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    return parser.parse_args()


def materialized_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    indices: torch.Tensor,
    query_position: int,
    rope_theta: float,
) -> torch.Tensor:
    batch_count, query_head_count, head_dim = query.shape
    kv_head_count = key.shape[1]
    group_count = query_head_count // kv_head_count
    grouped_indices = indices.reshape(
        batch_count,
        kv_head_count,
        group_count,
        indices.shape[-1],
    )
    gathered = torch.gather(
        key.unsqueeze(2).expand(
            batch_count,
            kv_head_count,
            group_count,
            key.shape[-2],
            head_dim,
        ),
        3,
        grouped_indices.unsqueeze(-1).expand(
            *grouped_indices.shape,
            head_dim,
        ),
    )
    query_grouped = query.reshape(
        batch_count,
        kv_head_count,
        group_count,
        head_dim,
    )
    query_positions = torch.full(
        query_grouped.shape[:-1],
        query_position,
        dtype=torch.float32,
        device=query.device,
    )
    key_pre = _inverse_standard_rope(gathered, grouped_indices, rope_theta)
    query_pre = _inverse_standard_rope(
        query_grouped,
        query_positions,
        rope_theta,
    )
    return torch.einsum(
        "bhgkd,bhgd->bhgk",
        key_pre,
        query_pre,
    ).reshape(batch_count, query_head_count, indices.shape[-1])


def relative_phase_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    indices: torch.Tensor,
    phase_cosine: torch.Tensor,
    phase_sine: torch.Tensor,
    query_position: int,
) -> torch.Tensor:
    """Materialize q_post^T R_(t-i) k_post using the supplied phase table."""

    batch_count, query_head_count, head_dim = query.shape
    kv_head_count = key.shape[1]
    group_count = query_head_count // kv_head_count
    half = head_dim // 2
    grouped_indices = indices.reshape(
        batch_count,
        kv_head_count,
        group_count,
        indices.shape[-1],
    )
    gathered = torch.gather(
        key.unsqueeze(2).expand(
            batch_count,
            kv_head_count,
            group_count,
            key.shape[-2],
            head_dim,
        ),
        3,
        grouped_indices.unsqueeze(-1).expand(
            *grouped_indices.shape,
            head_dim,
        ),
    ).float()
    distances = query_position - grouped_indices
    cosine = phase_cosine.index_select(
        0,
        distances.reshape(-1),
    ).reshape(*distances.shape, half)
    sine = phase_sine.index_select(
        0,
        distances.reshape(-1),
    ).reshape(*distances.shape, half)
    key_first = gathered[..., :half]
    key_second = gathered[..., half:]
    rotated = torch.cat(
        (
            key_first * cosine - key_second * sine,
            key_second * cosine + key_first * sine,
        ),
        dim=-1,
    )
    query_grouped = query.float().reshape(
        batch_count,
        kv_head_count,
        group_count,
        head_dim,
    )
    return torch.einsum(
        "bhgkd,bhgd->bhgk",
        rotated,
        query_grouped,
    ).reshape(batch_count, query_head_count, indices.shape[-1])


def benchmark_ms(
    function: Callable[[], torch.Tensor],
    warmup: int,
    repetitions: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        function()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)) / repetitions


def full_rerank_pipeline(
    proxy_scores: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    phase_cosine: torch.Tensor,
    phase_sine: torch.Tensor,
    query_position: int,
    target_count: int,
) -> torch.Tensor:
    pool = torch.topk(
        proxy_scores,
        min(proxy_scores.shape[-1], 2 * target_count),
        dim=-1,
        sorted=False,
    ).indices
    pre_scores = qabs_cuda_kernels.candidate_prerope_scores(
        query,
        key,
        pool,
        phase_cosine,
        phase_sine,
        query_position,
    )
    chosen = torch.topk(
        pre_scores,
        target_count,
        dim=-1,
        sorted=False,
    ).indices
    return pool.gather(-1, chosen)


def boundary75_rerank_pipeline(
    proxy_scores: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    phase_cosine: torch.Tensor,
    phase_sine: torch.Tensor,
    query_position: int,
    target_count: int,
) -> torch.Tensor:
    pool = torch.topk(
        proxy_scores,
        min(proxy_scores.shape[-1], 2 * target_count),
        dim=-1,
        sorted=True,
    ).indices
    core_count = min(target_count - 1, round(0.75 * target_count))
    boundary_pool = pool[..., core_count:]
    pre_scores = qabs_cuda_kernels.candidate_prerope_scores(
        query,
        key,
        boundary_pool,
        phase_cosine,
        phase_sine,
        query_position,
    )
    chosen = torch.topk(
        pre_scores,
        target_count - core_count,
        dim=-1,
        sorted=False,
    ).indices
    return torch.cat(
        (pool[..., :core_count], boundary_pool.gather(-1, chosen)),
        dim=-1,
    )


def dual_mass_rerank_pipeline(
    proxy_scores: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    phase_cosine: torch.Tensor,
    phase_sine: torch.Tensor,
    query_position: int,
    target_count: int,
) -> torch.Tensor:
    pool = torch.topk(
        proxy_scores,
        min(proxy_scores.shape[-1], 2 * target_count),
        dim=-1,
        sorted=False,
    ).indices
    scaling = float(query.shape[-1]) ** -0.5
    pre_scores = qabs_cuda_kernels.candidate_prerope_scores(
        query,
        key,
        pool,
        phase_cosine,
        phase_sine,
        query_position,
        scaling,
    )
    post_scores = qabs_cuda_kernels.candidate_compact_scores(
        query,
        key,
        pool,
        scaling,
    )
    mixture_mass = (
        torch.softmax(pre_scores, dim=-1)
        + torch.softmax(post_scores, dim=-1)
    )
    chosen = torch.topk(
        mixture_mass,
        target_count,
        dim=-1,
        sorted=False,
    ).indices
    return pool.gather(-1, chosen)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.query_heads % args.kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260731)
    query = torch.randn(
        (1, args.query_heads, args.head_dim),
        generator=generator,
        dtype=torch.float16,
        device=device,
    )
    key = torch.randn(
        (1, args.kv_heads, args.history_count, args.head_dim),
        generator=generator,
        dtype=torch.float16,
        device=device,
    )
    indices = torch.randint(
        0,
        args.history_count,
        (1, args.query_heads, args.candidate_count),
        generator=generator,
        dtype=torch.long,
        device=device,
    )
    proxy_scores = torch.randn(
        (1, args.query_heads, args.history_count),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    model_config = None
    if args.model_config:
        model_config = AutoConfig.from_pretrained(
            args.model_config,
            local_files_only=True,
            trust_remote_code=True,
        )
        if (
            args.original_max_position > 0
            and args.global_max_position > args.original_max_position
        ):
            model_config.max_position_embeddings = args.global_max_position
            model_config.rope_scaling = {
                "type": "yarn",
                "factor": (
                    args.global_max_position
                    / float(args.original_max_position)
                ),
                "original_max_position_embeddings": (
                    args.original_max_position
                ),
            }
        phase_cosine, phase_sine = _configured_rope_phase_tables(
            args.history_count + 1,
            args.head_dim,
            {
                "model_config": model_config,
                "rope_theta": args.rope_theta,
            },
            device,
        )
        half = args.head_dim // 2
        if phase_cosine.shape[-1] == args.head_dim:
            phase_cosine = phase_cosine[..., :half]
            phase_sine = phase_sine[..., :half]
        elif phase_cosine.shape[-1] != half:
            raise RuntimeError(
                "configured phase width must equal the head dimension "
                "or its number of rotation pairs"
            )
        phase_norm = (
            phase_cosine.square() + phase_sine.square()
        ).sqrt().clamp_min(1.0e-12)
        phase_cosine = (phase_cosine / phase_norm).contiguous()
        phase_sine = (phase_sine / phase_norm).contiguous()
        reference = relative_phase_reference(
            query,
            key,
            indices,
            phase_cosine,
            phase_sine,
            args.history_count,
        )
    else:
        phase_cosine, phase_sine = _standard_rope_phase_tables(
            args.history_count + 1,
            args.head_dim,
            args.rope_theta,
            device,
        )
        reference = materialized_reference(
            query,
            key,
            indices,
            args.history_count,
            args.rope_theta,
        )
    fused = qabs_cuda_kernels.candidate_prerope_scores(
        query,
        key,
        indices,
        phase_cosine,
        phase_sine,
        args.history_count,
    )
    absolute_error = (reference - fused).abs()
    topk_count = min(args.target_count, args.candidate_count)
    reference_topk = torch.topk(reference, topk_count, dim=-1).indices
    fused_topk = torch.topk(fused, topk_count, dim=-1).indices
    topk_overlap = (
        (
            reference_topk.unsqueeze(-1)
            == fused_topk.unsqueeze(-2)
        )
        .any(dim=-1)
        .float()
        .mean()
    )
    materialized_ms = benchmark_ms(
        lambda: materialized_reference(
            query,
            key,
            indices,
            args.history_count,
            args.rope_theta,
        ),
        args.warmup,
        args.repetitions,
    )
    fused_ms = benchmark_ms(
        lambda: qabs_cuda_kernels.candidate_prerope_scores(
            query,
            key,
            indices,
            phase_cosine,
            phase_sine,
            args.history_count,
        ),
        args.warmup,
        args.repetitions,
    )
    target_count = min(args.target_count, args.history_count // 2)
    full_pipeline_ms = benchmark_ms(
        lambda: full_rerank_pipeline(
            proxy_scores,
            query,
            key,
            phase_cosine,
            phase_sine,
            args.history_count,
            target_count,
        ),
        args.warmup,
        args.repetitions,
    )
    boundary75_pipeline_ms = benchmark_ms(
        lambda: boundary75_rerank_pipeline(
            proxy_scores,
            query,
            key,
            phase_cosine,
            phase_sine,
            args.history_count,
            target_count,
        ),
        args.warmup,
        args.repetitions,
    )
    dual_mass_pipeline_ms = benchmark_ms(
        lambda: dual_mass_rerank_pipeline(
            proxy_scores,
            query,
            key,
            phase_cosine,
            phase_sine,
            args.history_count,
            target_count,
        ),
        args.warmup,
        args.repetitions,
    )
    payload = {
        "schema": "qksieve_prerope_candidate_cuda_validation_v1",
        "history_count": args.history_count,
        "candidate_count": args.candidate_count,
        "query_heads": args.query_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "rope_theta": args.rope_theta,
        "model_config": args.model_config or None,
        "original_max_position": args.original_max_position or None,
        "global_max_position": args.global_max_position or None,
        "rope_scaling": (
            getattr(model_config, "rope_scaling", None)
            if model_config is not None
            else None
        ),
        "max_absolute_error": float(absolute_error.max().item()),
        "mean_absolute_error": float(absolute_error.mean().item()),
        "topk_overlap": float(topk_overlap.item()),
        "materialized_reference_ms": materialized_ms,
        "fused_cuda_ms": fused_ms,
        "speedup": materialized_ms / fused_ms,
        "target_count": target_count,
        "full_rerank_pipeline_ms": full_pipeline_ms,
        "boundary75_rerank_pipeline_ms": boundary75_pipeline_ms,
        "boundary75_pipeline_speedup": (
            full_pipeline_ms / boundary75_pipeline_ms
        ),
        "dual_mass_rerank_pipeline_ms": dual_mass_pipeline_ms,
        "dual_mass_overhead_vs_prerope": (
            dual_mass_pipeline_ms / full_pipeline_ms
        ),
        "phase_table_bytes": (
            phase_cosine.numel() * phase_cosine.element_size()
            + phase_sine.numel() * phase_sine.element_size()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
