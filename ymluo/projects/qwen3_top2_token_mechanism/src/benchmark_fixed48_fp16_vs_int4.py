from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import qabs_cuda_kernels


def pack_int4(projected_key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scales = projected_key.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    codes = torch.round(projected_key / scales).clamp(-7, 7).to(torch.int16) + 7
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    batch, heads, tokens, dimensions = projected_key.shape
    chunked = packed.reshape(batch, heads, tokens, dimensions // 16, 8)
    return chunked.permute(0, 1, 3, 2, 4).contiguous(), scales


def pack_group16_int4(
    projected_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, tokens, dimensions = projected_key.shape
    bands = projected_key.reshape(batch, heads, tokens, dimensions // 16, 16)
    scales = bands.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    codes = torch.round(bands / scales).clamp(-7, 7).to(torch.int16) + 7
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    packed_chunked = packed.permute(0, 1, 3, 2, 4).contiguous()
    scale_chunked = scales.squeeze(-1).permute(0, 1, 3, 2).contiguous()
    return packed_chunked, scale_chunked


def pack_logscale16_int4(
    projected_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, tokens, dimensions = projected_key.shape
    bands = projected_key.reshape(batch, heads, tokens, dimensions // 16, 16)
    exact_scale = bands.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    base_scale = exact_scale.amax(dim=-2, keepdim=True)
    exponent = torch.round(
        torch.log2(base_scale / exact_scale).clamp_min(0.0) / 0.25
    ).clamp(0, 15).to(torch.uint8)
    scale = base_scale * torch.exp2(-0.25 * exponent.float())
    codes = torch.round(bands / scale).clamp(-7, 7).to(torch.int16) + 7
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    packed_chunked = packed.permute(0, 1, 3, 2, 4).contiguous()
    exponent = exponent.squeeze(-1)
    packed_exponents = (
        exponent[..., 0::2] | (exponent[..., 1::2] << 4)
    ).contiguous()
    return packed_chunked, base_scale.squeeze(-2).contiguous(), packed_exponents


def dequantize_logscale16_key(
    packed_key_chunked: torch.Tensor,
    base_scale: torch.Tensor,
    packed_exponents: torch.Tensor,
) -> torch.Tensor:
    chunk_count = packed_key_chunked.shape[2]
    packed = packed_key_chunked.permute(0, 1, 3, 2, 4)
    low = (packed & 0x0F).to(torch.float32) - 7.0
    high = (packed >> 4).to(torch.float32) - 7.0
    key_codes = torch.stack((low, high), dim=-1).flatten(-2)

    exponent = torch.empty(
        (*packed_exponents.shape[:-1], chunk_count),
        dtype=torch.uint8,
        device=packed_exponents.device,
    )
    exponent[..., 0::2] = packed_exponents & 0x0F
    exponent[..., 1::2] = packed_exponents >> 4
    band_scale = base_scale.unsqueeze(-2).float() * torch.exp2(
        -0.25 * exponent.float().unsqueeze(-1)
    )
    return (key_codes * band_scale).flatten(-2)


def reference_logscale16_scores(
    projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    base_scale: torch.Tensor,
    packed_exponents: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    batch, kv_heads, groups, _ = projected_query.shape
    key = dequantize_logscale16_key(
        packed_key_chunked, base_scale, packed_exponents
    )[..., :rank]

    query_scale = (
        projected_query.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
        / 127.0
    )
    query_codes = (
        torch.round(projected_query.float() / query_scale)
        .clamp(-127, 127)
        .to(torch.int8)
    )
    query = query_codes.float() * query_scale
    return torch.einsum(
        "bhgr,bhkr->bhgk", query[..., :rank], key
    ).reshape(batch, kv_heads * groups, key.shape[-2])


def measure_ms(callback, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        callback()
    torch.cuda.synchronize()
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    started.record()
    for _ in range(repeats):
        callback()
    ended.record()
    torch.cuda.synchronize()
    return float(started.elapsed_time(ended) / repeats)


@torch.inference_mode()
def benchmark_length(
    token_count: int,
    rank: int,
    top_fraction: float,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    batch, kv_heads, groups, projection_dim = 1, 8, 4, 64
    device = torch.device("cuda")
    projected_key = torch.randn(
        (batch, kv_heads, token_count, projection_dim),
        dtype=torch.float16,
        device=device,
    )
    projected_query = torch.randn(
        (batch, kv_heads, groups, projection_dim),
        dtype=torch.float16,
        device=device,
    )
    packed, scales = pack_int4(projected_key)
    group16_packed, group16_scales = pack_group16_int4(projected_key)
    logscale16_packed, logscale16_base, logscale16_exponents = (
        pack_logscale16_int4(projected_key)
    )
    candidate_count = max(1, int(round(token_count * top_fraction)))
    sampled_capacity = math.ceil(0.20 * token_count)

    def fp16_scores() -> torch.Tensor:
        return torch.matmul(
            projected_query[..., :rank],
            projected_key[..., :rank].transpose(-1, -2),
        ).reshape(batch, kv_heads * groups, token_count)

    def int4_scores() -> torch.Tensor:
        query_scale = (
            projected_query.float()
            .abs()
            .amax(dim=-1, keepdim=True)
            .clamp_min(1.0e-8)
            / 127.0
        )
        query_codes = (
            torch.round(projected_query.float() / query_scale)
            .clamp(-127, 127)
            .to(torch.int8)
        )
        scores = qabs_cuda_kernels.pca_int4_chunked_prefix_scores(
            query_codes.contiguous(),
            packed,
            scales,
            token_count,
            rank,
        )
        return scores * query_scale.reshape(batch, kv_heads * groups, 1)

    def group16_int4_scores() -> torch.Tensor:
        query_scale = (
            projected_query.float()
            .abs()
            .amax(dim=-1, keepdim=True)
            .clamp_min(1.0e-8)
            / 127.0
        )
        query_codes = (
            torch.round(projected_query.float() / query_scale)
            .clamp(-127, 127)
            .to(torch.int8)
        )
        scores = qabs_cuda_kernels.pca_int4_chunked_group16_prefix_scores(
            query_codes.contiguous(),
            group16_packed,
            group16_scales,
            token_count,
            rank,
        )
        return scores * query_scale.reshape(batch, kv_heads * groups, 1)

    def logscale16_int4_scores() -> torch.Tensor:
        query_scale = (
            projected_query.float()
            .abs()
            .amax(dim=-1, keepdim=True)
            .clamp_min(1.0e-8)
            / 127.0
        )
        query_codes = (
            torch.round(projected_query.float() / query_scale)
            .clamp(-127, 127)
            .to(torch.int8)
        )
        scores = qabs_cuda_kernels.pca_int4_chunked_logscale16_prefix_scores(
            query_codes.contiguous(),
            logscale16_packed,
            logscale16_base,
            logscale16_exponents,
            token_count,
            rank,
        )
        return scores * query_scale.reshape(batch, kv_heads * groups, 1)

    def fp16_select() -> torch.Tensor:
        return torch.topk(
            fp16_scores(), candidate_count, dim=-1, sorted=False
        ).indices

    def int4_select() -> torch.Tensor:
        return torch.topk(
            int4_scores(), candidate_count, dim=-1, sorted=False
        ).indices

    def group16_int4_select() -> torch.Tensor:
        return torch.topk(
            group16_int4_scores(), candidate_count, dim=-1, sorted=False
        ).indices

    def logscale16_int4_select() -> torch.Tensor:
        return torch.topk(
            logscale16_int4_scores(), candidate_count, dim=-1, sorted=False
        ).indices

    def fused_sampled_quantile_select() -> tuple[torch.Tensor, ...]:
        query_scale = (
            projected_query.float()
            .abs()
            .amax(dim=-1, keepdim=True)
            .clamp_min(1.0e-8)
            / 127.0
        )
        query_codes = (
            torch.round(projected_query.float() / query_scale)
            .clamp(-127, 127)
            .to(torch.int8)
        )
        return qabs_cuda_kernels.pca_int4_logscale16_sampled_quantile_candidates(
            query_codes.contiguous(),
            logscale16_packed,
            logscale16_base,
            logscale16_exponents,
            token_count,
            sample_count=256,
            selected_fraction=0.12,
            candidate_capacity=sampled_capacity,
        )

    # Trigger extension loading before timing and validate output contracts.
    fp16_output = fp16_scores()
    int4_output = int4_scores()
    group16_int4_output = group16_int4_scores()
    logscale16_int4_output = logscale16_int4_scores()
    if (
        fp16_output.shape != int4_output.shape
        or fp16_output.shape != group16_int4_output.shape
        or fp16_output.shape != logscale16_int4_output.shape
    ):
        raise RuntimeError(
            f"score shape mismatch: {fp16_output.shape} vs {int4_output.shape}"
        )
    logscale16_reference = reference_logscale16_scores(
        projected_query,
        logscale16_packed,
        logscale16_base,
        logscale16_exponents,
        rank,
    )
    logscale16_abs_error = (
        logscale16_int4_output.float() - logscale16_reference.float()
    ).abs()
    logscale16_max_abs_error = float(logscale16_abs_error.max().item())
    if not torch.allclose(
        logscale16_int4_output.float(),
        logscale16_reference.float(),
        rtol=1.0e-5,
        atol=1.0e-4,
    ):
        raise RuntimeError(
            "compact log-scale CUDA score does not match explicit dequantization: "
            f"max_abs_error={logscale16_max_abs_error:.6g}"
        )

    global_indices = torch.topk(
        logscale16_int4_output, candidate_count, dim=-1, sorted=False
    ).indices
    fused_indices, _, fused_counts, _, fused_overflow = (
        fused_sampled_quantile_select()
    )
    fused_recalls = []
    fused_count_values = []
    for head in range(global_indices.shape[1]):
        count = min(int(fused_counts[0, head]), sampled_capacity)
        fused_count_values.append(count)
        selected = fused_indices[0, head, :count]
        recall = (
            selected.unsqueeze(-1) == global_indices[0, head].unsqueeze(0)
        ).any(dim=0).float().mean()
        fused_recalls.append(float(recall.item()))

    fp16_score_ms = measure_ms(fp16_scores, warmup, repeats)
    int4_score_ms = measure_ms(int4_scores, warmup, repeats)
    group16_int4_score_ms = measure_ms(group16_int4_scores, warmup, repeats)
    logscale16_int4_score_ms = measure_ms(
        logscale16_int4_scores, warmup, repeats
    )
    fp16_select_ms = measure_ms(fp16_select, warmup, repeats)
    int4_select_ms = measure_ms(int4_select, warmup, repeats)
    group16_int4_select_ms = measure_ms(group16_int4_select, warmup, repeats)
    logscale16_int4_select_ms = measure_ms(
        logscale16_int4_select, warmup, repeats
    )
    fused_sampled_quantile_ms = measure_ms(
        fused_sampled_quantile_select, warmup, repeats
    )

    fp16_rank_bytes = projected_key[..., :rank].numel() * 2
    int4_shared64_bytes = packed.numel() + scales.numel() * scales.element_size()
    int4_rank_bytes = batch * kv_heads * token_count * (rank // 2 + 2)
    group16_shared64_bytes = (
        group16_packed.numel()
        + group16_scales.numel() * group16_scales.element_size()
    )
    logscale16_shared64_bytes = (
        logscale16_packed.numel()
        + logscale16_base.numel() * logscale16_base.element_size()
        + logscale16_exponents.numel()
    )
    return {
        "tokens": token_count,
        "rank": rank,
        "top_fraction": top_fraction,
        "candidate_count": candidate_count,
        "fp16_score_ms": fp16_score_ms,
        "int4_score_ms": int4_score_ms,
        "int4_score_speedup": fp16_score_ms / int4_score_ms,
        "group16_int4_score_ms": group16_int4_score_ms,
        "group16_int4_score_speedup": fp16_score_ms / group16_int4_score_ms,
        "logscale16_int4_score_ms": logscale16_int4_score_ms,
        "logscale16_int4_score_speedup": fp16_score_ms
        / logscale16_int4_score_ms,
        "fp16_score_topk_ms": fp16_select_ms,
        "int4_score_topk_ms": int4_select_ms,
        "int4_score_topk_speedup": fp16_select_ms / int4_select_ms,
        "group16_int4_score_topk_ms": group16_int4_select_ms,
        "group16_int4_score_topk_speedup": fp16_select_ms / group16_int4_select_ms,
        "logscale16_int4_score_topk_ms": logscale16_int4_select_ms,
        "logscale16_int4_score_topk_speedup": fp16_select_ms
        / logscale16_int4_select_ms,
        "fused_sampled_quantile_ms": fused_sampled_quantile_ms,
        "fused_speedup_vs_logscale16_score_topk": logscale16_int4_select_ms
        / fused_sampled_quantile_ms,
        "fused_global_candidate_recall_mean": sum(fused_recalls)
        / len(fused_recalls),
        "fused_global_candidate_recall_min": min(fused_recalls),
        "fused_candidate_fraction_mean": sum(fused_count_values)
        / len(fused_count_values)
        / token_count,
        "fused_candidate_fraction_max": max(fused_count_values) / token_count,
        "fused_overflow_rate": float(fused_overflow.float().mean().item()),
        "logscale16_reference_max_abs_error": logscale16_max_abs_error,
        "logscale16_reference_mean_abs_error": float(
            logscale16_abs_error.mean().item()
        ),
        "fp16_rank_index_mib": fp16_rank_bytes / (1024**2),
        "int4_shared64_index_mib": int4_shared64_bytes / (1024**2),
        "int4_rank_index_mib": int4_rank_bytes / (1024**2),
        "int4_shared64_over_fp16_rank": int4_shared64_bytes / fp16_rank_bytes,
        "int4_rank_over_fp16_rank": int4_rank_bytes / fp16_rank_bytes,
        "group16_shared64_index_mib": group16_shared64_bytes / (1024**2),
        "group16_shared64_over_fp16_rank": group16_shared64_bytes
        / fp16_rank_bytes,
        "logscale16_shared64_index_mib": logscale16_shared64_bytes / (1024**2),
        "logscale16_shared64_over_fp16_rank": logscale16_shared64_bytes
        / fp16_rank_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed-rank FP16 matmul with deployed INT4 PCA scoring."
    )
    parser.add_argument("--lengths", default="8192,32768,65536,131072")
    parser.add_argument("--rank", type=int, default=48)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rank <= 0 or args.rank > 64 or args.rank % 16:
        raise ValueError("rank must be a positive multiple of 16 no larger than 64")

    torch.manual_seed(20260719)
    results = []
    for raw_length in args.lengths.split(","):
        token_count = int(raw_length)
        results.append(
            benchmark_length(
                token_count,
                args.rank,
                args.top_fraction,
                args.warmup,
                args.repeats,
            )
        )
        torch.cuda.empty_cache()
    report = {
        "hardware": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
