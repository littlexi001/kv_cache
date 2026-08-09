from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import qabs_cuda_kernels


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    value: torch.Tensor, positions: torch.Tensor, theta: float
) -> torch.Tensor:
    dimensions = value.shape[-1]
    inverse_frequency = 1.0 / (
        theta
        ** (
            torch.arange(
                0, dimensions, 2, dtype=torch.float32, device=value.device
            )
            / dimensions
        )
    )
    angle = torch.outer(positions.float(), inverse_frequency)
    embedding = torch.cat((angle, angle), dim=-1)
    while embedding.ndim < value.ndim:
        embedding = embedding.unsqueeze(0)
    return value * embedding.cos() + rotate_half(value) * embedding.sin()


def quantize_int4(value: torch.Tensor) -> torch.Tensor:
    scale = value.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    return torch.round(value / scale).clamp(-7, 7) * scale


def quantize_int8(value: torch.Tensor) -> torch.Tensor:
    scale = value.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 127.0
    return torch.round(value / scale).clamp(-127, 127) * scale


def unpack_key(
    packed: torch.Tensor, scales: torch.Tensor, key_count: int
) -> torch.Tensor:
    token_major = packed[..., :key_count, :].permute(0, 1, 3, 2, 4)
    low = (token_major & 0x0F).float() - 7.0
    high = (token_major >> 4).float() - 7.0
    codes = torch.stack((low, high), dim=-1).flatten(-3)
    return codes * scales[..., :key_count, :].float()


def measure_ms(callback, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        callback()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        callback()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


@torch.inference_mode()
def benchmark_length(
    tokens: int, warmup: int, repeats: int, theta: float
) -> dict[str, float | int]:
    batch, kv_heads, groups, head_dim = 1, 8, 4, 128
    device = torch.device("cuda")
    positions = torch.arange(tokens, device=device)
    pre_key = torch.randn(
        (batch, kv_heads, tokens, head_dim),
        dtype=torch.float16,
        device=device,
    )
    post_key = apply_rope(pre_key, positions, theta).to(torch.float16)
    pre_query = torch.randn(
        (batch, kv_heads * groups, head_dim),
        dtype=torch.float16,
        device=device,
    )
    query_position = tokens
    post_query = apply_rope(
        pre_query.unsqueeze(2),
        torch.tensor([query_position], device=device),
        theta,
    ).squeeze(2).to(torch.float16)
    packed = torch.empty(
        (batch, kv_heads, 2, tokens, 8), dtype=torch.uint8, device=device
    )
    scales = torch.empty(
        (batch, kv_heads, tokens, 1), dtype=torch.float16, device=device
    )
    packed_int2 = torch.empty(
        (batch, kv_heads, tokens, 8), dtype=torch.uint8, device=device
    )

    def pack() -> torch.Tensor:
        return qabs_cuda_kernels.pre_rope_lowfreq_int4_pack_into(
            post_key, packed, scales, 0, theta
        )

    def pack_int2() -> torch.Tensor:
        return qabs_cuda_kernels.pre_rope_lowfreq_int2_fixed_pack_into(
            post_key, packed_int2, 0, theta, 1.5
        )

    pack()
    pack_int2()
    rescue_count = max(1, math.ceil(0.005 * tokens))

    def scores() -> torch.Tensor:
        return qabs_cuda_kernels.pre_rope_lowfreq_int4_scores(
            post_query,
            packed,
            scales,
            tokens,
            query_position,
            theta,
        )

    def score_topk() -> torch.Tensor:
        return torch.topk(
            scores(), rescue_count, dim=-1, sorted=False
        ).indices

    def scores_int2() -> torch.Tensor:
        return qabs_cuda_kernels.pre_rope_lowfreq_int2_fixed_scores(
            post_query,
            packed_int2,
            tokens,
            query_position,
            theta,
        )

    def score_topk_int2() -> torch.Tensor:
        return torch.topk(
            scores_int2(), rescue_count, dim=-1, sorted=False
        ).indices

    def score_topk_int2_oldest50() -> torch.Tensor:
        scan_count = max(1, math.ceil(0.5 * tokens))
        partial_scores = qabs_cuda_kernels.pre_rope_lowfreq_int2_fixed_scores(
            post_query,
            packed_int2,
            scan_count,
            query_position,
            theta,
        )
        return torch.topk(
            partial_scores, rescue_count, dim=-1, sorted=False
        ).indices

    base_count = max(1, math.ceil(0.08 * tokens))
    base_indices = torch.topk(
        torch.randn(batch, kv_heads * groups, tokens, device=device),
        base_count,
        dim=-1,
        sorted=False,
    ).indices
    cached_rescue = score_topk_int2()

    def append_cached() -> tuple[torch.Tensor, torch.Tensor]:
        return qabs_cuda_kernels.append_rescue_candidates(
            base_indices, cached_rescue, tokens
        )

    def refresh_and_append() -> tuple[torch.Tensor, torch.Tensor]:
        return qabs_cuda_kernels.append_rescue_candidates(
            base_indices, score_topk_int2(), tokens
        )

    fused_scores = scores()
    int2_score_topk_ms = measure_ms(score_topk_int2, warmup, repeats)
    int2_oldest50_score_topk_ms = measure_ms(
        score_topk_int2_oldest50, warmup, repeats
    )
    append_cached_ms = measure_ms(append_cached, warmup, repeats)
    refresh_and_append_ms = measure_ms(refresh_and_append, warmup, repeats)
    result: dict[str, float | int] = {
        "tokens": tokens,
        "rescue_count": rescue_count,
        "prefill_pack_ms": measure_ms(pack, max(1, warmup // 2), max(1, repeats // 10)),
        "score_ms": measure_ms(scores, warmup, repeats),
        "score_topk_ms": measure_ms(score_topk, warmup, repeats),
        "int2_prefill_pack_ms": measure_ms(
            pack_int2, max(1, warmup // 2), max(1, repeats // 10)
        ),
        "int2_score_ms": measure_ms(scores_int2, warmup, repeats),
        "int2_score_topk_ms": int2_score_topk_ms,
        "int2_oldest50_score_topk_ms": int2_oldest50_score_topk_ms,
        "append_cached_ms": append_cached_ms,
        "refresh_and_append_ms": refresh_and_append_ms,
        "refresh4_amortized_overhead_ms": (
            refresh_and_append_ms + 3.0 * append_cached_ms
        ) / 4.0,
        "oldest50_refresh4_amortized_overhead_ms": (
            int2_oldest50_score_topk_ms
            + append_cached_ms
            + 3.0 * append_cached_ms
        )
        / 4.0,
        "index_mib": (packed.numel() + scales.numel() * scales.element_size())
        / (1024**2),
        "index_over_full_kv": (
            packed.numel() + scales.numel() * scales.element_size()
        )
        / (batch * kv_heads * tokens * head_dim * 2 * 2),
        "int2_index_mib": packed_int2.numel() / (1024**2),
        "int2_index_over_full_kv": packed_int2.numel()
        / (batch * kv_heads * tokens * head_dim * 2 * 2),
    }
    if tokens <= 8192:
        low_indices = torch.cat(
            (
                torch.arange(48, 64, device=device),
                torch.arange(112, 128, device=device),
            )
        )
        reference_key = quantize_int4(
            F.normalize(pre_key.index_select(-1, low_indices).float(), dim=-1)
        )
        reference_query = quantize_int8(
            F.normalize(pre_query.index_select(-1, low_indices).float(), dim=-1)
        ).reshape(batch, kv_heads, groups, 32)
        reference_scores = torch.einsum(
            "bhgd,bhnd->bhgn", reference_query, reference_key
        ).reshape(batch, kv_heads * groups, tokens)
        fused_key = unpack_key(packed, scales, tokens)
        key_relative_l2 = float(
            (fused_key - reference_key).norm().item()
            / reference_key.norm().clamp_min(1.0e-8).item()
        )
        reference_top = torch.topk(
            reference_scores, rescue_count, dim=-1, sorted=False
        ).indices
        fused_top = torch.topk(
            fused_scores, rescue_count, dim=-1, sorted=False
        ).indices
        recall = (
            reference_top.unsqueeze(-1) == fused_top.unsqueeze(-2)
        ).any(dim=-1).float().mean()
        normalized_key = F.normalize(
            pre_key.index_select(-1, low_indices).float(), dim=-1
        )
        clip = 1.5 / math.sqrt(32)
        int2_codes = torch.round(
            ((normalized_key / clip).clamp(-1, 1) + 1.0) * 1.5
        ).clamp(0, 3)
        int2_key = (int2_codes / 1.5 - 1.0) * clip
        int2_reference_scores = torch.einsum(
            "bhgd,bhnd->bhgn", reference_query, int2_key
        ).reshape(batch, kv_heads * groups, tokens)
        int2_reference_top = torch.topk(
            int2_reference_scores, rescue_count, dim=-1, sorted=False
        ).indices
        int2_fused_top = torch.topk(
            scores_int2(), rescue_count, dim=-1, sorted=False
        ).indices
        int2_recall = (
            int2_reference_top.unsqueeze(-1) == int2_fused_top.unsqueeze(-2)
        ).any(dim=-1).float().mean()
        result.update(
            {
                "reference_key_relative_l2": key_relative_l2,
                "reference_topk_recall": float(recall.item()),
                "int2_reference_topk_recall": float(int2_recall.item()),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="8192,32768,65536,131072")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--rope_theta", type=float, default=5_000_000.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260720)
    lengths = [int(item) for item in args.lengths.split(",")]
    report = {
        "hardware": torch.cuda.get_device_name(),
        "rope_theta": args.rope_theta,
        "results": [],
    }
    for tokens in lengths:
        report["results"].append(
            benchmark_length(tokens, args.warmup, args.repeats, args.rope_theta)
        )
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
