from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

import qabs_cuda_kernels as cuda_kernels


def elapsed_ms(operation, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        operation()
        finished.record()
        finished.synchronize()
        values.append(float(started.elapsed_time(finished)))
    return float(statistics.median(values))


def valid_candidate_sets(
    indices: torch.Tensor,
    counts: torch.Tensor,
) -> list[list[int]]:
    indices_cpu = indices.cpu()
    counts_cpu = counts.cpu()
    result = []
    for row in range(indices_cpu.shape[0] * indices_cpu.shape[1]):
        batch = row // indices_cpu.shape[1]
        head = row % indices_cpu.shape[1]
        count = int(counts_cpu[batch, head])
        result.append(sorted(indices_cpu[batch, head, :count].tolist()))
    return result


def benchmark_length(
    history_tokens: int,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | bool]:
    batch_count = 1
    kv_head_count = 8
    group_count = 4
    projection_dim = 48
    chunk_count = projection_dim // 16
    selected_fraction = 0.06
    candidate_capacity = min(
        history_tokens,
        max(1, round(0.10 * history_tokens)),
    )
    device = torch.device("cuda")

    torch.manual_seed(20260724 + history_tokens)
    projected_key = torch.randn(
        batch_count,
        kv_head_count,
        history_tokens,
        projection_dim,
        dtype=torch.float16,
        device=device,
    )
    packed_key = torch.empty(
        batch_count,
        kv_head_count,
        chunk_count,
        history_tokens,
        8,
        dtype=torch.uint8,
        device=device,
    )
    scales = torch.empty(
        batch_count,
        kv_head_count,
        history_tokens,
        1,
        dtype=torch.float16,
        device=device,
    )
    exponents = torch.empty(
        batch_count,
        kv_head_count,
        history_tokens,
        (chunk_count + 1) // 2,
        dtype=torch.uint8,
        device=device,
    )
    cuda_kernels.pca_int4_logscale16_pack_into(
        projected_key,
        packed_key,
        scales,
        exponents,
        0,
    )
    projected_query = torch.randint(
        -127,
        128,
        (
            batch_count,
            kv_head_count,
            group_count,
            projection_dim,
        ),
        dtype=torch.int8,
        device=device,
    )

    def run(write_proxy_scores: bool, use_dp4a: bool = False):
        return cuda_kernels.pca_int4_logscale16_sampled_quantile_candidates(
            projected_query,
            packed_key,
            scales,
            exponents,
            history_tokens,
            256,
            selected_fraction,
            candidate_capacity,
            use_dp4a=use_dp4a,
            write_proxy_scores=write_proxy_scores,
        )

    with_proxy = run(True)
    without_proxy = run(False)
    dp4a_candidates = run(False, use_dp4a=True)
    torch.cuda.synchronize()
    counts_equal = bool(torch.equal(with_proxy[2], without_proxy[2]))
    boundaries_max_abs_error = float(
        (with_proxy[3] - without_proxy[3]).abs().max().item()
    )
    overflow_equal = bool(torch.equal(with_proxy[4], without_proxy[4]))
    candidate_sets_equal = (
        valid_candidate_sets(with_proxy[0], with_proxy[2])
        == valid_candidate_sets(without_proxy[0], without_proxy[2])
    )
    dp4a_candidate_sets_equal = (
        valid_candidate_sets(without_proxy[0], without_proxy[2])
        == valid_candidate_sets(dp4a_candidates[0], dp4a_candidates[2])
    )
    dp4a_counts_equal = bool(torch.equal(without_proxy[2], dp4a_candidates[2]))

    head_dim = 128
    query_head_count = kv_head_count * group_count
    query = torch.randn(
        batch_count,
        query_head_count,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    key = torch.randn(
        batch_count,
        kv_head_count,
        history_tokens + 1,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    value = torch.randn_like(key)
    scaling = head_dim**-0.5
    current_key = key[..., -1, :].repeat_interleave(group_count, dim=1)
    self_scores = (
        query.float() * current_key.float()
    ).sum(dim=-1, keepdim=True) * scaling

    def current_pipeline() -> torch.Tensor:
        indices, _, counts, _, _ = run(False)
        return cuda_kernels.final_attention_ragged_self(
            query,
            key,
            value,
            indices,
            counts,
            scaling,
        )

    def current_dp4a_pipeline() -> torch.Tensor:
        indices, _, counts, _, _ = run(False, use_dp4a=True)
        return cuda_kernels.final_attention_ragged_self(
            query,
            key,
            value,
            indices,
            counts,
            scaling,
        )

    def exact_scan_pipeline() -> torch.Tensor:
        indices, exact_scores, counts, _, _ = (
            cuda_kernels.pca_int4_logscale16_sampled_quantile_exact_candidates(
                projected_query,
                query,
                key,
                packed_key,
                scales,
                exponents,
                history_tokens,
                256,
                selected_fraction,
                candidate_capacity,
                scaling,
            )
        )
        return cuda_kernels.final_attention_from_scores_ragged_self(
            value,
            indices,
            exact_scores,
            counts,
            self_scores,
        )

    exact_indices, exact_scores, exact_counts, exact_boundaries, exact_overflow = (
        cuda_kernels.pca_int4_logscale16_sampled_quantile_exact_candidates(
            projected_query,
            query,
            key,
            packed_key,
            scales,
            exponents,
            history_tokens,
            256,
            selected_fraction,
            candidate_capacity,
            scaling,
        )
    )
    exact_candidate_sets_equal = (
        valid_candidate_sets(without_proxy[0], without_proxy[2])
        == valid_candidate_sets(exact_indices, exact_counts)
    )
    exact_dp4a_candidate_sets_equal = (
        valid_candidate_sets(dp4a_candidates[0], dp4a_candidates[2])
        == valid_candidate_sets(exact_indices, exact_counts)
    )
    exact_counts_equal = bool(torch.equal(without_proxy[2], exact_counts))
    exact_boundaries_max_abs_error = float(
        (without_proxy[3] - exact_boundaries).abs().max().item()
    )
    exact_overflow_equal = bool(torch.equal(without_proxy[4], exact_overflow))
    exact_candidate_reference = cuda_kernels.final_attention_ragged_self(
        query,
        key,
        value,
        exact_indices,
        exact_counts,
        scaling,
    )
    exact_candidate_consumed = (
        cuda_kernels.final_attention_from_scores_ragged_self(
            value,
            exact_indices,
            exact_scores,
            exact_counts,
            self_scores,
        )
    )
    current_output = current_pipeline()
    current_dp4a_output = current_dp4a_pipeline()
    exact_scan_output = exact_scan_pipeline()
    torch.cuda.synchronize()
    pipeline_max_abs_error = float(
        (current_output.float() - exact_scan_output.float()).abs().max().item()
    )
    dp4a_pipeline_max_abs_error = float(
        (
            current_dp4a_output.float()
            - exact_scan_output.float()
        )
        .abs()
        .max()
        .item()
    )
    exact_consumer_max_abs_error = float(
        (
            exact_candidate_reference.float()
            - exact_candidate_consumed.float()
        )
        .abs()
        .max()
        .item()
    )
    with_proxy_ms = elapsed_ms(lambda: run(True), warmup, repeats)
    without_proxy_ms = elapsed_ms(lambda: run(False), warmup, repeats)
    current_pipeline_ms = elapsed_ms(current_pipeline, warmup, repeats)
    current_dp4a_pipeline_ms = elapsed_ms(
        current_dp4a_pipeline,
        warmup,
        repeats,
    )
    exact_scan_pipeline_ms = elapsed_ms(exact_scan_pipeline, warmup, repeats)
    return {
        "history_tokens": history_tokens,
        "candidate_capacity": candidate_capacity,
        "with_proxy_ms": with_proxy_ms,
        "without_proxy_ms": without_proxy_ms,
        "speedup": with_proxy_ms / without_proxy_ms,
        "counts_equal": counts_equal,
        "boundaries_max_abs_error": boundaries_max_abs_error,
        "overflow_equal": overflow_equal,
        "candidate_sets_equal": candidate_sets_equal,
        "dp4a_candidate_sets_equal": dp4a_candidate_sets_equal,
        "dp4a_counts_equal": dp4a_counts_equal,
        "current_pipeline_ms": current_pipeline_ms,
        "current_dp4a_pipeline_ms": current_dp4a_pipeline_ms,
        "exact_scan_pipeline_ms": exact_scan_pipeline_ms,
        "exact_scan_pipeline_speedup": (
            current_pipeline_ms / exact_scan_pipeline_ms
        ),
        "pipeline_max_abs_error": pipeline_max_abs_error,
        "dp4a_pipeline_max_abs_error": dp4a_pipeline_max_abs_error,
        "exact_candidate_sets_equal": exact_candidate_sets_equal,
        "exact_dp4a_candidate_sets_equal": exact_dp4a_candidate_sets_equal,
        "exact_counts_equal": exact_counts_equal,
        "exact_boundaries_max_abs_error": exact_boundaries_max_abs_error,
        "exact_overflow_equal": exact_overflow_equal,
        "exact_consumer_max_abs_error": exact_consumer_max_abs_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="8192,16000")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = [
        benchmark_length(int(item), args.warmup, args.repeats)
        for item in args.lengths.split(",")
        if item.strip()
    ]
    payload = {"results": results}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
