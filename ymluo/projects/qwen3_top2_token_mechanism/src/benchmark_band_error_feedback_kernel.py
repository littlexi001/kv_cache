from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import qabs_cuda_kernels


def pack_int4(projected_key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scales = projected_key.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    codes = torch.round(projected_key / scales).clamp(-7, 7).to(torch.int16) + 7
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    batch, heads, tokens, dimensions = projected_key.shape
    chunked = packed.reshape(batch, heads, tokens, dimensions // 16, 8)
    return chunked.permute(0, 1, 3, 2, 4).contiguous(), scales.to(projected_key.dtype)


def reference_update(
    current: torch.Tensor,
    anchor: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, groups, dimensions = current.shape
    chunks = dimensions // 16
    residual = current.float() - anchor.float()
    energy = (residual.square() * weights.unsqueeze(2)).reshape(
        batch, heads, groups, chunks, 16
    ).sum(dim=(2, 4))
    selected = energy.argmax(dim=-1)
    packed_index = selected[:, :, None, None, None].expand(
        batch, heads, 1, packed.shape[3], 8
    )
    selected_packed = torch.gather(packed, 2, packed_index).squeeze(2)
    low = (selected_packed & 0x0F).float() - 7.0
    high = (selected_packed >> 4).float() - 7.0
    key = torch.stack((low, high), dim=-1).flatten(-2) * scales.float()
    residual_chunks = residual.reshape(batch, heads, groups, chunks, 16)
    query_index = selected[:, :, None, None, None].expand(
        batch, heads, groups, 1, 16
    )
    selected_residual = torch.gather(residual_chunks, 3, query_index).squeeze(3)
    scores = torch.einsum("bhgd,bhnd->bhgn", selected_residual, key)
    return selected, scores.reshape(batch, heads * groups, key.shape[2])


def reference_anchor_update(
    current: torch.Tensor, anchor: torch.Tensor, selected: torch.Tensor
) -> None:
    batch, heads, groups, dimensions = current.shape
    chunks = dimensions // 16
    current_chunks = current.reshape(batch, heads, groups, chunks, 16)
    anchor_chunks = anchor.reshape(batch, heads, groups, chunks, 16)
    chunk_index = selected[:, :, None, None, None].expand(
        batch, heads, groups, 1, 16
    )
    selected_current = torch.gather(current_chunks, 3, chunk_index)
    anchor_chunks.scatter_(3, chunk_index, selected_current)


def multistep_correctness(
    *,
    initial: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    token_count: int,
    steps: int = 16,
) -> dict[str, float]:
    batch, kv_heads, groups, _ = initial.shape
    reference_anchor = initial.clone()
    kernel_anchor = initial.clone()
    reference_cache = torch.zeros(
        (batch, kv_heads * groups, token_count),
        dtype=torch.float16,
        device=initial.device,
    )
    kernel_cache = torch.zeros_like(reference_cache)
    selected = torch.empty(
        (batch, kv_heads), dtype=torch.int32, device=initial.device
    )
    gate = torch.empty(
        (batch, kv_heads), dtype=torch.float32, device=initial.device
    )
    current = initial.clone()
    selected_matches = []
    topk_matches = []
    for _ in range(steps):
        current = current + 0.03 * torch.randn_like(current)
        reference_selected, reference_delta = reference_update(
            current, reference_anchor, packed, scales, weights
        )
        reference_cache.add_(reference_delta.to(reference_cache.dtype))
        reference_anchor_update(current, reference_anchor, reference_selected)
        qabs_cuda_kernels.pca_int4_chunked_band_error_feedback(
            current,
            packed,
            scales,
            weights,
            kernel_anchor,
            selected,
            gate,
            kernel_cache,
            token_count,
        )
        selected_matches.append((selected == reference_selected).float().mean())
        candidate_count = max(1, int(token_count * 0.02))
        reference_topk = torch.topk(
            reference_cache, candidate_count, dim=-1, sorted=False
        ).indices
        kernel_topk = torch.topk(
            kernel_cache, candidate_count, dim=-1, sorted=False
        ).indices
        topk_matches.append(
            (kernel_topk.unsqueeze(-1) == reference_topk.unsqueeze(-2))
            .any(dim=-1)
            .float()
            .mean()
        )
    difference = kernel_cache.float() - reference_cache.float()
    relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        reference_cache.float()
    )
    return {
        "multistep_selected_band_match": float(torch.stack(selected_matches).mean()),
        "multistep_top2pct_overlap": float(torch.stack(topk_matches).mean()),
        "multistep_relative_l2": float(relative_l2),
    }


def time_calls(callback, repeats: int) -> float:
    for _ in range(5):
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


def benchmark_length(
    token_count: int, repeats: int, device: torch.device
) -> dict[str, float | int]:
    batch, kv_heads, groups, dimensions = 1, 8, 4, 64
    key = torch.randn(
        (batch, kv_heads, token_count, dimensions),
        dtype=torch.float16,
        device=device,
    )
    packed, scales = pack_int4(key)
    weights = torch.rand((batch, kv_heads, dimensions), device=device)
    anchor = torch.randn(
        (batch, kv_heads, groups, dimensions), dtype=torch.float16, device=device
    )
    current = anchor + 0.05 * torch.randn_like(anchor)
    score_cache = torch.zeros(
        (batch, kv_heads * groups, token_count), dtype=torch.float16, device=device
    )
    selected = torch.empty((batch, kv_heads), dtype=torch.int32, device=device)
    gate = torch.empty((batch, kv_heads), dtype=torch.float32, device=device)

    reference_selected, reference_scores = reference_update(
        current, anchor, packed, scales, weights
    )
    kernel_anchor = anchor.clone()
    qabs_cuda_kernels.pca_int4_chunked_band_error_feedback(
        current,
        packed,
        scales,
        weights,
        kernel_anchor,
        selected,
        gate,
        score_cache,
        token_count,
    )
    torch.cuda.synchronize()
    difference = score_cache.float() - reference_scores
    relative_l2 = float(
        (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference_scores)).item()
    )
    selected_match = float((selected == reference_selected).float().mean().item())
    active_mask = torch.zeros(
        (batch, kv_heads), dtype=torch.uint8, device=device
    )
    active_mask[:, ::2] = 1
    masked_anchor = anchor.clone()
    masked_cache = torch.zeros_like(score_cache)
    qabs_cuda_kernels.pca_int4_chunked_band_error_feedback_masked(
        current,
        packed,
        scales,
        weights,
        masked_anchor,
        active_mask,
        selected,
        gate,
        masked_cache,
        token_count,
    )
    active_query_mask = active_mask.repeat_interleave(groups, dim=1).bool()
    inactive_cache_unchanged = float(
        (masked_cache[~active_query_mask] == 0).all().item()
    )
    inactive_anchor_unchanged = float(
        (masked_anchor[:, 1::2] == anchor[:, 1::2]).all().item()
    )
    active_cache_relative_l2 = float(
        (
            torch.linalg.vector_norm(
                masked_cache[active_query_mask].float()
                - reference_scores[active_query_mask]
            )
            / torch.linalg.vector_norm(reference_scores[active_query_mask]).clamp_min(
                1.0e-12
            )
        ).item()
    )
    multistep = multistep_correctness(
        initial=anchor,
        packed=packed,
        scales=scales,
        weights=weights,
        token_count=token_count,
    )

    band_anchor = anchor.clone()
    band_cache = torch.zeros_like(score_cache)
    band_ms = time_calls(
        lambda: qabs_cuda_kernels.pca_int4_chunked_band_error_feedback(
            current,
            packed,
            scales,
            weights,
            band_anchor,
            selected,
            gate,
            band_cache,
            token_count,
        ),
        repeats,
    )

    fixed_cache = torch.zeros_like(score_cache)
    fixed_ms = time_calls(
        lambda: qabs_cuda_kernels.pca_int4_chunked_contiguous_delta_add(
            current,
            anchor,
            packed,
            scales,
            fixed_cache,
            token_count,
            48,
            16,
        ),
        repeats,
    )

    spectral_anchor = anchor.clone()
    spectral_cache = torch.zeros_like(score_cache)
    refresh_mask = torch.empty((batch, kv_heads), dtype=torch.uint8, device=device)
    refresh_indices = torch.empty((batch, kv_heads), dtype=torch.int32, device=device)
    spectral_ms = time_calls(
        lambda: qabs_cuda_kernels.pca_int4_chunked_spectral_gated_delta_add(
            current,
            anchor,
            packed,
            scales,
            weights,
            spectral_anchor,
            refresh_mask,
            gate,
            refresh_indices,
            spectral_cache,
            token_count,
            48,
            0.0,
            2,
        ),
        repeats,
    )
    return {
        "tokens": token_count,
        "selected_band_match": selected_match,
        "relative_l2_vs_fp32_reference": relative_l2,
        "masked_inactive_cache_unchanged": inactive_cache_unchanged,
        "masked_inactive_anchor_unchanged": inactive_anchor_unchanged,
        "masked_active_relative_l2": active_cache_relative_l2,
        "band_error_feedback_ms": band_ms,
        "fixed_delta16_ms": fixed_ms,
        "spectral_top2_ms": spectral_ms,
        "bandef_vs_spectral_speedup": spectral_ms / band_ms,
        "bandef_over_fixed16": band_ms / fixed_ms,
        **multistep,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="32768,65536,131072")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    results = [
        benchmark_length(int(value), args.repeats, device)
        for value in args.lengths.split(",")
        if value
    ]
    report = {
        "hardware": torch.cuda.get_device_name(device),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
