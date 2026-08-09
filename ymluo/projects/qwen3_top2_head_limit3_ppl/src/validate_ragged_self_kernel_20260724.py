from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import qabs_cuda_kernels as cuda_kernels
from run_head_top2_targeted_ppl_20260714 import (
    _append_self_to_ragged_candidates,
)


def elapsed_ms(operation, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return float(statistics.median(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=16000)
    parser.add_argument("--candidate_fraction", type=float, default=0.06)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(20260724)
    device = torch.device("cuda")
    batch = 1
    query_heads = 32
    kv_heads = 8
    head_dim = 128
    capacity = max(1, round(args.history_tokens * args.candidate_fraction))
    value = torch.randn(
        batch,
        kv_heads,
        args.history_tokens + 1,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    indices = torch.randint(
        0,
        args.history_tokens,
        (batch, query_heads, capacity),
        dtype=torch.long,
        device=device,
    )
    scores = torch.randn(
        batch,
        query_heads,
        capacity,
        dtype=torch.float32,
        device=device,
    )
    counts = torch.randint(
        max(1, capacity // 2),
        capacity + 1,
        (batch, query_heads),
        dtype=torch.long,
        device=device,
    )
    self_indices = torch.full(
        (batch, query_heads, 1),
        args.history_tokens,
        dtype=torch.long,
        device=device,
    )
    self_scores = torch.randn(
        batch,
        query_heads,
        1,
        dtype=torch.float32,
        device=device,
    )

    packed_indices, packed_scores, packed_counts = (
        _append_self_to_ragged_candidates(
            indices,
            scores,
            counts,
            self_indices,
            self_scores,
        )
    )
    reference = cuda_kernels.final_attention_from_scores_ragged(
        value,
        packed_indices,
        packed_scores,
        packed_counts,
    )
    fused = cuda_kernels.final_attention_from_scores_ragged_self(
        value,
        indices,
        scores,
        counts,
        self_scores,
    )
    torch.cuda.synchronize()
    max_abs_error = float((reference.float() - fused.float()).abs().max().item())
    if max_abs_error > 2.0e-3:
        raise AssertionError(f"fused kernel mismatch: {max_abs_error}")

    def old_path() -> torch.Tensor:
        old_indices, old_scores, old_counts = _append_self_to_ragged_candidates(
            indices,
            scores,
            counts,
            self_indices,
            self_scores,
        )
        return cuda_kernels.final_attention_from_scores_ragged(
            value,
            old_indices,
            old_scores,
            old_counts,
        )

    def fused_path() -> torch.Tensor:
        return cuda_kernels.final_attention_from_scores_ragged_self(
            value,
            indices,
            scores,
            counts,
            self_scores,
        )

    old_ms = elapsed_ms(old_path, args.warmup, args.repeats)
    fused_ms = elapsed_ms(fused_path, args.warmup, args.repeats)

    query = torch.randn(
        batch,
        query_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    key = torch.randn(
        batch,
        kv_heads,
        args.history_tokens + 1,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    scaling = head_dim**-0.5

    def staged_qkv_path() -> torch.Tensor:
        exact_scores = cuda_kernels.candidate_compact_scores_ragged(
            query,
            key,
            indices,
            counts,
            scaling,
        )
        current_key = key[..., -1, :].repeat_interleave(
            query_heads // kv_heads,
            dim=1,
        )
        exact_self_scores = (
            query.float() * current_key.float()
        ).sum(dim=-1, keepdim=True) * scaling
        return cuda_kernels.final_attention_from_scores_ragged_self(
            value,
            indices,
            exact_scores,
            counts,
            exact_self_scores,
        )

    def fused_qkv_path() -> torch.Tensor:
        return cuda_kernels.final_attention_ragged_self(
            query,
            key,
            value,
            indices,
            counts,
            scaling,
        )

    def warp_qkv_path() -> torch.Tensor:
        return cuda_kernels.final_attention_ragged_self_warp(
            query,
            key,
            value,
            indices,
            counts,
            scaling,
        )

    staged_qkv = staged_qkv_path()
    fused_qkv = fused_qkv_path()
    warp_qkv = warp_qkv_path()
    torch.cuda.synchronize()
    qkv_max_abs_error = float(
        (staged_qkv.float() - fused_qkv.float()).abs().max().item()
    )
    if qkv_max_abs_error > 2.0e-3:
        raise AssertionError(f"QK+V fused kernel mismatch: {qkv_max_abs_error}")
    warp_qkv_max_abs_error = float(
        (staged_qkv.float() - warp_qkv.float()).abs().max().item()
    )
    if warp_qkv_max_abs_error > 2.0e-3:
        raise AssertionError(
            f"warp QK+V fused kernel mismatch: {warp_qkv_max_abs_error}"
        )
    staged_qkv_ms = elapsed_ms(staged_qkv_path, args.warmup, args.repeats)
    fused_qkv_ms = elapsed_ms(fused_qkv_path, args.warmup, args.repeats)
    warp_qkv_ms = elapsed_ms(warp_qkv_path, args.warmup, args.repeats)

    result = {
        "history_tokens": args.history_tokens,
        "candidate_tokens": capacity,
        "candidate_fraction": args.candidate_fraction,
        "max_abs_error": max_abs_error,
        "old_pack_and_attention_ms": old_ms,
        "fused_attention_ms": fused_ms,
        "microkernel_speedup": old_ms / fused_ms,
        "qkv_max_abs_error": qkv_max_abs_error,
        "staged_exact_qk_and_attention_ms": staged_qkv_ms,
        "fused_exact_qk_and_attention_ms": fused_qkv_ms,
        "qkv_microkernel_speedup": staged_qkv_ms / fused_qkv_ms,
        "warp_qkv_max_abs_error": warp_qkv_max_abs_error,
        "warp_exact_qk_and_attention_ms": warp_qkv_ms,
        "warp_qkv_microkernel_speedup": staged_qkv_ms / warp_qkv_ms,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
