from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import hierarchical_spectral_cuda_20260727 as hierarchical_cuda


def parse_ints(value: str) -> list[int]:
    output = sorted({int(item) for item in value.split(",") if item.strip()})
    if not output:
        raise ValueError("expected at least one integer")
    return output


def measure_ms(
    function: Callable[[], object],
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started) / iterations


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int16)
    high = (packed >> 4).to(torch.int16)
    low = torch.where(low < 8, low, low - 16)
    high = torch.where(high < 8, high, high - 16)
    return torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], 16)


def unpack_sign(packed: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return torch.where(bits.bool(), 1.0, -1.0).reshape(
        *packed.shape[:-1], 80
    )


def make_inputs(
    history_tokens: int,
    candidate_fraction: float,
) -> dict[str, torch.Tensor]:
    batch_count = 1
    kv_heads = 8
    query_groups = 4
    query_heads = kv_heads * query_groups
    device = torch.device("cuda")
    query_codes = torch.randint(
        -127,
        128,
        (batch_count, kv_heads, query_groups, 128),
        dtype=torch.int8,
        device=device,
    )
    query_scales = (
        0.002
        + 0.02
        * torch.rand(
            batch_count,
            kv_heads,
            query_groups,
            8,
            dtype=torch.float16,
            device=device,
        )
    )
    core_int8 = torch.randint(
        -127,
        128,
        (batch_count, kv_heads, history_tokens, 16),
        dtype=torch.int8,
        device=device,
    )
    middle_int4 = torch.randint(
        0,
        256,
        (batch_count, kv_heads, 2, history_tokens, 8),
        dtype=torch.uint8,
        device=device,
    )
    tail_sign = torch.randint(
        0,
        256,
        (batch_count, kv_heads, history_tokens, 10),
        dtype=torch.uint8,
        device=device,
    )
    key_scales = (
        0.002
        + 0.02
        * torch.rand(
            batch_count,
            kv_heads,
            history_tokens,
            4,
            dtype=torch.float16,
            device=device,
        )
    )
    candidate_count = max(1, int(candidate_fraction * history_tokens))
    base = (
        torch.arange(candidate_count, device=device, dtype=torch.long)
        * history_tokens
        // candidate_count
    )
    candidate_indices = base.view(1, 1, -1).expand(
        batch_count, query_heads, -1
    ).contiguous()
    return {
        "query_codes": query_codes,
        "query_scales": query_scales,
        "core_int8": core_int8,
        "middle_int4": middle_int4,
        "tail_sign": tail_sign,
        "key_scales": key_scales,
        "candidate_indices": candidate_indices,
    }


@torch.inference_mode()
def validate_correctness() -> dict[str, float]:
    tensors = make_inputs(257, 0.3)
    cuda_core = hierarchical_cuda.core_scores(
        tensors["query_codes"],
        tensors["query_scales"],
        tensors["core_int8"],
        tensors["middle_int4"],
        tensors["key_scales"],
    )
    cuda_tail = hierarchical_cuda.tail_candidate_scores(
        tensors["query_codes"],
        tensors["query_scales"],
        tensors["tail_sign"],
        tensors["key_scales"],
        tensors["candidate_indices"],
    )

    query = (
        tensors["query_codes"].float()
        * tensors["query_scales"].float().repeat_interleave(16, dim=-1)
    )
    core = (
        tensors["core_int8"].float()
        * tensors["key_scales"][..., 0:1].float()
    )
    middle = unpack_int4(tensors["middle_int4"]).float()
    middle = (
        middle.permute(0, 1, 3, 2, 4)
        * tensors["key_scales"][..., 1:3]
        .float()
        .unsqueeze(-1)
    ).reshape(1, 8, 257, 32)
    exact_core = torch.einsum(
        "bhgd,bhnd->bhgn",
        query[..., :48],
        torch.cat((core, middle), dim=-1),
    ).reshape_as(cuda_core)

    tail = (
        unpack_sign(tensors["tail_sign"])
        * tensors["key_scales"][..., 3:4].float()
    )
    exact_tail = torch.einsum(
        "bhgd,bhnd->bhgn",
        query[..., 48:],
        tail,
    ).reshape(1, 32, 257)
    selected_tail = torch.gather(
        exact_tail,
        dim=-1,
        index=tensors["candidate_indices"],
    )
    return {
        "core_max_abs_error": float(
            (cuda_core - exact_core).abs().max().item()
        ),
        "core_mean_abs_error": float(
            (cuda_core - exact_core).abs().mean().item()
        ),
        "tail_max_abs_error": float(
            (cuda_tail - selected_tail).abs().max().item()
        ),
        "tail_mean_abs_error": float(
            (cuda_tail - selected_tail).abs().mean().item()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark packed 8/4/1 spectral scan CUDA kernels."
    )
    parser.add_argument(
        "--lengths",
        default="8192,16384,32768,65536,131072",
    )
    parser.add_argument("--candidate_fraction", type=float, default=0.30)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--full_iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(20260727)
    if not 0.0 < args.candidate_fraction <= 1.0:
        raise ValueError("candidate fraction must be in (0, 1]")
    correctness = validate_correctness()
    if (
        correctness["core_max_abs_error"] > 5.0e-3
        or correctness["tail_max_abs_error"] > 5.0e-3
    ):
        raise RuntimeError(f"packed CUDA correctness failure: {correctness}")

    rows = []
    for history_tokens in parse_ints(args.lengths):
        tensors = make_inputs(history_tokens, args.candidate_fraction)

        def core_function() -> torch.Tensor:
            return hierarchical_cuda.core_scores(
                tensors["query_codes"],
                tensors["query_scales"],
                tensors["core_int8"],
                tensors["middle_int4"],
                tensors["key_scales"],
            )

        def tail_function() -> torch.Tensor:
            return hierarchical_cuda.tail_candidate_scores(
                tensors["query_codes"],
                tensors["query_scales"],
                tensors["tail_sign"],
                tensors["key_scales"],
                tensors["candidate_indices"],
            )

        def progressive_function() -> tuple[torch.Tensor, torch.Tensor]:
            return core_function(), tail_function()

        core_ms = measure_ms(core_function, args.warmup, args.iterations)
        tail_ms = measure_ms(tail_function, args.warmup, args.iterations)
        progressive_ms = measure_ms(
            progressive_function, args.warmup, args.iterations
        )

        query = torch.randn(
            1, 32, 1, 128, dtype=torch.float16, device="cuda"
        )
        key = torch.randn(
            1,
            8,
            history_tokens,
            128,
            dtype=torch.float16,
            device="cuda",
        )
        value = torch.randn_like(key)

        def full_attention() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                enable_gqa=True,
            )

        full_ms = measure_ms(
            full_attention,
            min(args.warmup, 10),
            args.full_iterations,
        )
        candidate_count = int(tensors["candidate_indices"].shape[-1])
        code_scan_bits = 16 * 8 + 32 * 4 + 80 * args.candidate_fraction
        code_index_bits = 16 * 8 + 32 * 4 + 80
        rows.append(
            {
                "history_tokens": history_tokens,
                "candidate_count": candidate_count,
                "candidate_fraction": args.candidate_fraction,
                "core_ms": core_ms,
                "tail_candidate_ms": tail_ms,
                "progressive_two_kernel_ms": progressive_ms,
                "full_sdpa_ms": full_ms,
                "full_sdpa_over_progressive_scan": (
                    full_ms / progressive_ms
                ),
                "logical_code_scan_bits_per_token": code_scan_bits,
                "logical_code_index_bits_per_token": code_index_bits,
                "logical_scale_bits_per_token": 64,
                "logical_total_index_bytes_per_token": (
                    code_index_bits + 64
                )
                / 8,
            }
        )
        del tensors, query, key, value
        torch.cuda.empty_cache()

    output = {
        "correctness": correctness,
        "scope": (
            "Retrieval scan microbenchmark only. Candidate selection, exact "
            "sparse attention, the remaining model forward, and index build "
            "are intentionally excluded."
        ),
        "rows": rows,
    }
    text = json.dumps(output, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
