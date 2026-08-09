from __future__ import annotations

import argparse
import json
import time

import torch

import qabs_cuda_kernels
from run_head_top2_targeted_ppl_20260714 import _pca_int4_partial_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and benchmark a 16-dimension chunk-major PCA INT4 scan."
    )
    parser.add_argument("--history_tokens", type=int, default=131072)
    parser.add_argument("--prefix_dims", default="16,32,48,64")
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", default="")
    return parser.parse_args()


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
    args = parse_args()
    torch.manual_seed(20260717)
    device = torch.device("cuda")
    batch_count = 1
    kv_heads = 8
    groups = 4
    query_heads = kv_heads * groups
    head_dim = 128
    projection_dim = 64
    capacity = args.history_tokens + 2048
    query = torch.randn(
        batch_count, query_heads, head_dim, dtype=torch.float16, device=device
    )
    key = torch.randn(
        batch_count,
        kv_heads,
        args.history_tokens,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    state: dict[str, object] = {}
    _pca_int4_partial_scores(
        query,
        key,
        state,
        projection_dim,
        basis_descending=True,
        score_prefix_dim=16,
    )
    packed = state["packed"]
    if packed.shape[-2:] != (capacity, projection_dim // 2):
        raise RuntimeError("unexpected packed index shape")
    packed_chunked = (
        packed.reshape(batch_count, kv_heads, capacity, 4, 8)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    query_codes = state["last_projected_query_codes"].contiguous()
    scales = state["scales"]
    prefix_dims = tuple(int(value) for value in args.prefix_dims.split(","))
    results = []
    for prefix_dim in prefix_dims:
        row_function = lambda: qabs_cuda_kernels.pca_int4_prefix_scores(
            query_codes,
            packed,
            scales,
            args.history_tokens,
            prefix_dim,
        )
        chunked_function = lambda: qabs_cuda_kernels.pca_int4_chunked_prefix_scores(
            query_codes,
            packed_chunked,
            scales,
            args.history_tokens,
            prefix_dim,
        )
        row_scores = row_function()
        row_ms = measure_ms(row_function, args.warmup, args.iterations)
        result = {
            "prefix_dim": prefix_dim,
            "row_major_ms": row_ms,
            "chunked_supported": prefix_dim % 16 == 0,
        }
        if prefix_dim % 16 == 0:
            chunked_scores = chunked_function()
            max_abs_error = float((row_scores - chunked_scores).abs().max().item())
            mean_abs_error = float((row_scores - chunked_scores).abs().mean().item())
            if max_abs_error != 0.0:
                raise RuntimeError(
                    f"chunked prefix {prefix_dim} differs from row-major: {max_abs_error}"
                )
            chunked_ms = measure_ms(
                chunked_function, args.warmup, args.iterations
            )
            result.update(
                {
                    "chunked_ms": chunked_ms,
                    "chunked_speedup": row_ms / chunked_ms,
                    "max_abs_error": max_abs_error,
                    "mean_abs_error": mean_abs_error,
                }
            )
        results.append(result)
    candidate_count = max(1, int(0.20 * args.history_tokens))
    candidate_indices = torch.randint(
        0,
        args.history_tokens,
        (batch_count, query_heads, candidate_count),
        dtype=torch.long,
        device=device,
    )
    candidate_results = []
    for start_dim, end_dim in ((16, 64), (32, 64)):
        row_function = lambda: qabs_cuda_kernels.pca_int4_candidate_range_scores(
            query_codes,
            packed,
            scales,
            candidate_indices,
            args.history_tokens,
            start_dim,
            end_dim,
        )
        chunked_function = (
            lambda: qabs_cuda_kernels.pca_int4_chunked_candidate_range_scores(
                query_codes,
                packed_chunked,
                scales,
                candidate_indices,
                args.history_tokens,
                start_dim,
                end_dim,
            )
        )
        row_scores = row_function()
        chunked_scores = chunked_function()
        max_abs_error = float((row_scores - chunked_scores).abs().max().item())
        if max_abs_error != 0.0:
            raise RuntimeError(
                f"chunked candidate range {start_dim}:{end_dim} differs: "
                f"{max_abs_error}"
            )
        row_ms = measure_ms(row_function, args.warmup, args.iterations)
        chunked_ms = measure_ms(chunked_function, args.warmup, args.iterations)
        candidate_results.append(
            {
                "start_dim": start_dim,
                "end_dim": end_dim,
                "candidate_count": candidate_count,
                "row_major_ms": row_ms,
                "chunked_ms": chunked_ms,
                "chunked_speedup": row_ms / chunked_ms,
                "max_abs_error": max_abs_error,
            }
        )
    payload = {
        "history_tokens": args.history_tokens,
        "index_bytes_each_layout": packed.numel() * packed.element_size(),
        "results": results,
        "candidate_results": candidate_results,
    }
    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output)
    if args.output:
        output_path = __import__("pathlib").Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
