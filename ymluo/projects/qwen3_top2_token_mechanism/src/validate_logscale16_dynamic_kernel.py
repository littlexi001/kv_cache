from __future__ import annotations

import json

import torch

import qabs_cuda_kernels
from benchmark_fixed48_fp16_vs_int4 import (
    dequantize_logscale16_key,
    pack_logscale16_int4,
)


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(20260720)
    device = torch.device("cuda")
    batch, kv_heads, groups, tokens, dimensions = 1, 8, 4, 1024, 64
    capacity = tokens + 32

    key = torch.randn(
        (batch, kv_heads, tokens, dimensions),
        dtype=torch.float16,
        device=device,
    )
    packed_reference, base_reference, exponent_reference = pack_logscale16_int4(key)
    packed_capacity = torch.zeros(
        (batch, kv_heads, dimensions // 16, capacity, 8),
        dtype=torch.uint8,
        device=device,
    )
    base_capacity = torch.ones(
        (batch, kv_heads, capacity, 1),
        dtype=torch.float16,
        device=device,
    )
    exponent_capacity = torch.zeros(
        (batch, kv_heads, capacity, dimensions // 32),
        dtype=torch.uint8,
        device=device,
    )
    qabs_cuda_kernels.pca_int4_logscale16_pack_into(
        key,
        packed_capacity,
        base_capacity,
        exponent_capacity,
        0,
    )
    torch.cuda.synchronize()
    packed = packed_capacity[..., :tokens, :]
    base_scale = base_capacity[..., :tokens, :]
    packed_exponents = exponent_capacity[..., :tokens, :]
    packed_mismatch = int((packed != packed_reference).sum().item())
    exponent_mismatch = int(
        (packed_exponents != exponent_reference).sum().item()
    )
    base_max_abs_error = float(
        (base_scale.float() - base_reference.float()).abs().max().item()
    )
    packed_mismatch_rate = packed_mismatch / packed_reference.numel()
    exponent_mismatch_rate = exponent_mismatch / exponent_reference.numel()
    reference_key = dequantize_logscale16_key(
        packed_reference, base_reference, exponent_reference
    )
    fused_key = dequantize_logscale16_key(
        packed, base_scale, packed_exponents
    )
    key_relative_l2 = float(
        (fused_key - reference_key).norm().item()
        / reference_key.norm().clamp_min(1.0e-8).item()
    )
    probe_query = torch.randn(
        (batch, kv_heads, groups, dimensions),
        dtype=torch.float32,
        device=device,
    )
    reference_scores = torch.einsum(
        "bhgd,bhkd->bhgk", probe_query, reference_key
    )
    fused_scores = torch.einsum("bhgd,bhkd->bhgk", probe_query, fused_key)
    top_count = max(1, int(round(tokens * 0.02)))
    reference_top = torch.topk(reference_scores, top_count, dim=-1).indices
    fused_top = torch.topk(fused_scores, top_count, dim=-1).indices
    pack_top2_recall = float(
        (
            reference_top.unsqueeze(-1) == fused_top.unsqueeze(-2)
        ).any(dim=-1).float().mean().item()
    )
    if (
        packed_mismatch_rate > 0.005
        or exponent_mismatch_rate > 0.005
        or key_relative_l2 > 0.01
        or pack_top2_recall < 0.995
        or base_max_abs_error != 0.0
    ):
        raise RuntimeError(
            "fused compact log-scale pack mismatch: "
            f"codes={packed_mismatch_rate:.3%}, "
            f"exponents={exponent_mismatch_rate:.3%}, "
            f"key_l2={key_relative_l2:.3%}, recall={pack_top2_recall:.3%}"
        )

    anchor = torch.randn(
        (batch, kv_heads, groups, dimensions),
        dtype=torch.float16,
        device=device,
    )
    query = anchor + 0.1 * torch.randn_like(anchor)
    anchor_before = anchor.clone()
    spectral_weights = torch.rand(
        (batch, kv_heads, dimensions), dtype=torch.float32, device=device
    ).clamp_min_(1.0e-4)
    active_mask = torch.ones(
        (batch, kv_heads), dtype=torch.uint8, device=device
    )
    selected_chunk = torch.empty(
        (batch, kv_heads), dtype=torch.int32, device=device
    )
    gate_signal = torch.empty(
        (batch, kv_heads), dtype=torch.float32, device=device
    )
    score_cache = torch.randn(
        (batch, kv_heads * groups, capacity),
        dtype=torch.float16,
        device=device,
    )
    score_before = score_cache.clone()

    qabs_cuda_kernels.pca_int4_chunked_logscale16_band_error_feedback_masked(
        query.contiguous(),
        packed_capacity,
        base_capacity,
        exponent_capacity,
        spectral_weights,
        anchor,
        active_mask,
        selected_chunk,
        gate_signal,
        score_cache,
        tokens,
    )
    torch.cuda.synchronize()

    expected = score_before[..., :tokens].float()
    dequantized_key = dequantize_logscale16_key(
        packed, base_scale, packed_exponents
    )
    for kv_head in range(kv_heads):
        chunk = int(selected_chunk[0, kv_head].item())
        start, end = chunk * 16, (chunk + 1) * 16
        query_delta = (
            query[0, kv_head, :, start:end]
            - anchor_before[0, kv_head, :, start:end]
        )
        head_start = kv_head * groups
        expected[:, head_start : head_start + groups] += torch.einsum(
            "gd,kd->gk",
            query_delta.float(),
            dequantized_key[0, kv_head, :, start:end],
        ).unsqueeze(0)

    absolute_error = (score_cache[..., :tokens].float() - expected).abs()
    max_abs_error = float(absolute_error.max().item())
    mean_abs_error = float(absolute_error.mean().item())
    if not torch.allclose(
        score_cache[..., :tokens].float(), expected, rtol=2.0e-3, atol=2.0e-3
    ):
        raise RuntimeError(
            f"dynamic compact log-scale update mismatch: {max_abs_error:.6g}"
        )

    print(
        json.dumps(
            {
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                "pack_code_mismatch": packed_mismatch,
                "pack_code_mismatch_rate": packed_mismatch_rate,
                "pack_exponent_mismatch": exponent_mismatch,
                "pack_exponent_mismatch_rate": exponent_mismatch_rate,
                "pack_base_max_abs_error": base_max_abs_error,
                "pack_key_relative_l2": key_relative_l2,
                "pack_top2_recall": pack_top2_recall,
                "selected_chunks": selected_chunk.cpu().tolist(),
                "gate_signal_mean": float(gate_signal.mean().item()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
