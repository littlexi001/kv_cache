from __future__ import annotations

import argparse
import json
import math
import time

import torch

import qabs_cuda_kernels


def measure_ms(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=131072)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    torch.manual_seed(20260717)
    device = torch.device("cuda")
    query_heads = 32
    kv_heads = 8
    head_dim = 128
    maximum_count = math.ceil(0.08 * args.history_tokens)
    query = torch.randn(
        1, query_heads, head_dim, dtype=torch.float16, device=device
    )
    key = torch.randn(
        1,
        kv_heads,
        args.history_tokens,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    candidate_indices = torch.randint(
        0,
        args.history_tokens,
        (1, query_heads, maximum_count),
        dtype=torch.long,
        device=device,
    )
    fractions = torch.tensor(
        [0.02, 0.03, 0.04, 0.06, 0.08], device=device
    )
    pattern = fractions[torch.arange(query_heads, device=device) % len(fractions)]
    candidate_counts = torch.ceil(pattern * args.history_tokens).long().unsqueeze(0)
    scaling = head_dim**-0.5
    dense_function = lambda: qabs_cuda_kernels.candidate_compact_scores(
        query, key, candidate_indices, scaling
    )
    ragged_function = lambda: qabs_cuda_kernels.candidate_compact_scores_ragged(
        query, key, candidate_indices, candidate_counts, scaling
    )
    random_priority = torch.rand_like(candidate_indices, dtype=torch.float32)
    candidate_valid = random_priority < pattern.view(1, -1, 1) / 0.08
    masked_function = lambda: qabs_cuda_kernels.candidate_compact_scores_masked(
        query, key, candidate_indices, candidate_valid, scaling
    )
    dense = dense_function()
    ragged = ragged_function()
    masked = masked_function()
    rank = torch.arange(maximum_count, device=device).view(1, 1, -1)
    active = rank < candidate_counts.unsqueeze(-1)
    max_abs_error = float((dense[active] - ragged[active]).abs().max().item())
    inactive_is_negative_infinity = bool(torch.isneginf(ragged[~active]).all())
    masked_max_abs_error = float(
        (dense[candidate_valid] - masked[candidate_valid]).abs().max().item()
    )
    masked_inactive_is_negative_infinity = bool(
        torch.isneginf(masked[~candidate_valid]).all()
    )
    dense_ms = measure_ms(dense_function, args.warmup, args.iterations)
    ragged_ms = measure_ms(ragged_function, args.warmup, args.iterations)
    masked_ms = measure_ms(masked_function, args.warmup, args.iterations)
    proxy_scores = torch.randn(
        1, query_heads, maximum_count, device=device, dtype=torch.float32
    )
    error_sigma = torch.full(
        (1, query_heads), 0.1, device=device, dtype=torch.float32
    )
    final_count = math.ceil(0.02 * args.history_tokens)
    narrow_valid, narrow_counts, _ = qabs_cuda_kernels.uncertainty_band_mask(
        proxy_scores, error_sigma, final_count, 0.0
    )
    wide_valid, wide_counts, _ = qabs_cuda_kernels.uncertainty_band_mask(
        proxy_scores, error_sigma, final_count, 1.0
    )
    true_top_indices = torch.topk(
        proxy_scores, k=final_count, dim=-1, sorted=False
    ).indices
    top_is_valid = bool(torch.gather(narrow_valid, -1, true_top_indices).all())
    histogram_monotone = bool((wide_counts >= narrow_counts).all())
    histogram_function = lambda: qabs_cuda_kernels.uncertainty_band_mask(
        proxy_scores, error_sigma, final_count, 1.0
    )
    histogram_ms = measure_ms(histogram_function, args.warmup, args.iterations)
    full_proxy_scores = torch.randn(
        1, query_heads, args.history_tokens, device=device, dtype=torch.float32
    )
    direct_function = lambda: qabs_cuda_kernels.direct_uncertainty_candidates(
        full_proxy_scores,
        error_sigma,
        final_count,
        maximum_count,
        1.0,
    )
    direct_indices, direct_counts, _, direct_overflow = direct_function()
    direct_top_indices = torch.topk(
        full_proxy_scores, k=final_count, dim=-1, sorted=False
    ).indices
    direct_top_is_valid = all(
        bool(
            torch.isin(
                direct_top_indices[0, head],
                direct_indices[0, head, : direct_counts[0, head]],
            ).all()
        )
        for head in range(query_heads)
    )
    direct_ms = measure_ms(direct_function, args.warmup, args.iterations)
    sample_count = max(16, math.ceil(0.0025 * args.history_tokens))
    sample_stride = max(1, args.history_tokens // sample_count)
    sample_indices_1d = torch.arange(sample_count, device=device) * sample_stride
    sample_indices = sample_indices_1d.view(1, 1, -1).expand(
        1, query_heads, -1
    )
    exact_sample = qabs_cuda_kernels.candidate_compact_scores(
        query, key, sample_indices, scaling
    )
    proxy_sample = full_proxy_scores.index_select(-1, sample_indices_1d) * scaling
    sample_error = exact_sample - proxy_sample
    sample_error = sample_error - sample_error.mean(dim=-1, keepdim=True)
    reference_sigma = sample_error.square().mean(dim=-1).sqrt().clamp_min(1.0e-8)
    fused_sigma_function = lambda: qabs_cuda_kernels.sample_error_sigma(
        query,
        key,
        full_proxy_scores,
        sample_count,
        0,
        scaling,
    )
    fused_sigma = fused_sigma_function()
    fused_sigma_max_abs_error = float(
        (reference_sigma - fused_sigma).abs().max().item()
    )
    fused_sigma_ms = measure_ms(
        fused_sigma_function, args.warmup, args.iterations
    )
    value_indices = candidate_indices[..., : final_count + 1].contiguous()
    historical_scores = torch.randn(
        1, query_heads, final_count, device=device
    ).sort(dim=-1, descending=True).values
    value_scores = torch.cat(
        (historical_scores, torch.randn(1, query_heads, 1, device=device)),
        dim=-1,
    )
    value_counts = torch.full(
        (1, query_heads),
        final_count + 1,
        dtype=torch.long,
        device=device,
    )
    full_value_function = lambda: qabs_cuda_kernels.final_attention_from_scores_ragged(
        key, value_indices, value_scores, value_counts, 1.0
    )
    mass99_value_function = lambda: qabs_cuda_kernels.final_attention_from_scores_ragged(
        key, value_indices, value_scores, value_counts, 0.99
    )
    mass95_value_function = lambda: qabs_cuda_kernels.final_attention_from_scores_ragged(
        key, value_indices, value_scores, value_counts, 0.95
    )
    full_value_output = full_value_function()
    mass99_value_output = mass99_value_function()
    mass95_value_output = mass95_value_function()
    expanded_value = key.repeat_interleave(query_heads // kv_heads, dim=1)
    gathered_value = torch.gather(
        expanded_value,
        2,
        value_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim),
    )
    reference_value_output = torch.einsum(
        "bhs,bhsd->bhd", torch.softmax(value_scores, dim=-1), gathered_value.float()
    )
    full_value_max_abs_error = float(
        (full_value_output[:, 0].float() - reference_value_output).abs().max().item()
    )
    mass99_value_relative_l2 = float(
        (
            (mass99_value_output.float() - full_value_output.float()).norm()
            / full_value_output.float().norm().clamp_min(1.0e-8)
        ).item()
    )
    mass95_value_relative_l2 = float(
        (
            (mass95_value_output.float() - full_value_output.float()).norm()
            / full_value_output.float().norm().clamp_min(1.0e-8)
        ).item()
    )
    full_value_ms = measure_ms(
        full_value_function, args.warmup, args.iterations
    )
    mass99_value_ms = measure_ms(
        mass99_value_function, args.warmup, args.iterations
    )
    mass95_value_ms = measure_ms(
        mass95_value_function, args.warmup, args.iterations
    )
    payload = {
        "history_tokens": args.history_tokens,
        "maximum_candidate_fraction": 0.08,
        "average_candidate_fraction": float(pattern.mean().item()),
        "max_abs_error_active": max_abs_error,
        "inactive_is_negative_infinity": inactive_is_negative_infinity,
        "dense_ms": dense_ms,
        "ragged_ms": ragged_ms,
        "ragged_speedup": dense_ms / ragged_ms,
        "masked_average_candidate_fraction": float(
            candidate_valid.float().mean().item() * 0.08
        ),
        "masked_max_abs_error_active": masked_max_abs_error,
        "masked_inactive_is_negative_infinity": (
            masked_inactive_is_negative_infinity
        ),
        "masked_ms": masked_ms,
        "masked_speedup": dense_ms / masked_ms,
        "histogram_top2_is_valid": top_is_valid,
        "histogram_wider_band_is_monotone": histogram_monotone,
        "histogram_ms": histogram_ms,
        "histogram_zero_width_candidate_fraction": float(
            narrow_counts.float().mean().item() / args.history_tokens
        ),
        "direct_top2_is_valid": direct_top_is_valid,
        "direct_overflow": bool(direct_overflow.any()),
        "direct_candidate_fraction": float(
            direct_counts.float().mean().item() / args.history_tokens
        ),
        "direct_selection_ms": direct_ms,
        "fused_sigma_max_abs_error": fused_sigma_max_abs_error,
        "fused_sigma_ms": fused_sigma_ms,
        "full_value_max_abs_error": full_value_max_abs_error,
        "full_value_ms": full_value_ms,
        "mass99_value_ms": mass99_value_ms,
        "mass99_value_speedup": full_value_ms / mass99_value_ms,
        "mass99_value_relative_l2": mass99_value_relative_l2,
        "mass95_value_ms": mass95_value_ms,
        "mass95_value_speedup": full_value_ms / mass95_value_ms,
        "mass95_value_relative_l2": mass95_value_relative_l2,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if (
        max_abs_error != 0.0
        or not inactive_is_negative_infinity
        or masked_max_abs_error != 0.0
        or not masked_inactive_is_negative_infinity
        or not top_is_valid
        or not histogram_monotone
        or not direct_top_is_valid
        or bool(direct_overflow.any())
        or fused_sigma_max_abs_error > 1.0e-5
        or full_value_max_abs_error > 1.0e-3
    ):
        raise RuntimeError("ragged candidate scores do not match dense scores")


if __name__ == "__main__":
    main()
