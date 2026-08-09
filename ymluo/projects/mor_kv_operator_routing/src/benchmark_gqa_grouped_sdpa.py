from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a real GQA physical-union gather+SDPA reference path. "
            "This removes unselected K/V computation but uses one SDPA call per KV head."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sequence_lengths", default="4096,16384,32768")
    parser.add_argument("--active_blocks", default="2,8,12")
    parser.add_argument("--block_tokens", type=int, default=256)
    parser.add_argument("--query_heads", type=int, default=16)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def evenly_spaced_blocks(total_blocks: int, active_blocks: int) -> list[int]:
    count = min(total_blocks, active_blocks)
    if count <= 0:
        raise ValueError("active block count must be positive")
    return sorted(
        set(
            int(item)
            for item in np.linspace(0, total_blocks - 1, num=count, dtype=np.int64)
        )
    )


def token_indices_for_blocks(
    blocks: Sequence[int], sequence_length: int, block_tokens: int, device: torch.device
) -> torch.Tensor:
    indices: list[int] = []
    for block in blocks:
        start = int(block) * block_tokens
        end = min((int(block) + 1) * block_tokens, sequence_length)
        indices.extend(range(start, end))
    return torch.tensor(indices, dtype=torch.long, device=device)


def dense_gqa_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    group_size = query.shape[1] // key.shape[1]
    repeated_key = key.repeat_interleave(group_size, dim=1)
    repeated_value = value.repeat_interleave(group_size, dim=1)
    return F.scaled_dot_product_attention(
        query, repeated_key, repeated_value, is_causal=False
    )


def grouped_gqa_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices_by_kv_head: Sequence[torch.Tensor],
) -> torch.Tensor:
    group_size = query.shape[1] // key.shape[1]
    outputs: list[torch.Tensor] = []
    for kv_head, indices in enumerate(indices_by_kv_head):
        query_group = query[:, kv_head * group_size : (kv_head + 1) * group_size]
        gathered_key = key[:, kv_head : kv_head + 1].index_select(2, indices)
        gathered_value = value[:, kv_head : kv_head + 1].index_select(2, indices)
        outputs.append(
            F.scaled_dot_product_attention(
                query_group,
                gathered_key.expand(-1, group_size, -1, -1),
                gathered_value.expand(-1, group_size, -1, -1),
                is_causal=False,
            )
        )
    return torch.cat(outputs, dim=1)


def pack_ragged_indices(
    indices_by_kv_head: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not indices_by_kv_head:
        raise ValueError("indices_by_kv_head must be non-empty")
    max_length = max(int(indices.numel()) for indices in indices_by_kv_head)
    if max_length <= 0:
        raise ValueError("each physical KV head must retain at least one token")
    device = indices_by_kv_head[0].device
    padded = torch.zeros(
        (len(indices_by_kv_head), max_length), dtype=torch.long, device=device
    )
    valid = torch.zeros(
        (len(indices_by_kv_head), max_length), dtype=torch.bool, device=device
    )
    for head, indices in enumerate(indices_by_kv_head):
        length = int(indices.numel())
        padded[head, :length] = indices
        valid[head, :length] = True
    return padded, valid


def batched_ragged_gqa_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices_by_kv_head: Sequence[torch.Tensor],
) -> torch.Tensor:
    repeated_key, repeated_value, attention_mask = prepare_batched_ragged_kv(
        key, value, indices_by_kv_head, query.shape[1]
    )
    return prepacked_gqa_sdpa(query, repeated_key, repeated_value, attention_mask)


def prepare_batched_ragged_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    indices_by_kv_head: Sequence[torch.Tensor],
    query_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group_size = query_heads // key.shape[1]
    padded, valid = pack_ragged_indices(indices_by_kv_head)
    gather_index = padded[None, :, :, None].expand(
        key.shape[0], -1, -1, key.shape[-1]
    )
    gathered_key = torch.gather(key, 2, gather_index)
    gathered_value = torch.gather(value, 2, gather_index)
    repeated_key = gathered_key.repeat_interleave(group_size, dim=1)
    repeated_value = gathered_value.repeat_interleave(group_size, dim=1)
    attention_mask = valid.repeat_interleave(group_size, dim=0)[None, :, None, :]
    return repeated_key, repeated_value, attention_mask


def prepacked_gqa_sdpa(
    query: torch.Tensor,
    repeated_key: torch.Tensor,
    repeated_value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query,
        repeated_key,
        repeated_value,
        attn_mask=attention_mask,
        is_causal=False,
    )


def cuda_time_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("kernel benchmark requires CUDA")
    if args.query_heads % args.kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", 0)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    torch.manual_seed(args.seed)
    rows = []
    for sequence_length in [int(item) for item in args.sequence_lengths.split(",")]:
        query = torch.randn(
            1, args.query_heads, 1, args.head_dim, device=device, dtype=dtype
        )
        key = torch.randn(
            1, args.kv_heads, sequence_length, args.head_dim, device=device, dtype=dtype
        )
        value = torch.randn_like(key)
        dense_ms = cuda_time_ms(
            lambda: dense_gqa_sdpa(query, key, value), args.warmup, args.repeats
        )
        total_blocks = math.ceil(sequence_length / args.block_tokens)
        for requested_blocks in [int(item) for item in args.active_blocks.split(",")]:
            blocks = evenly_spaced_blocks(total_blocks, requested_blocks)
            indices = token_indices_for_blocks(
                blocks, sequence_length, args.block_tokens, device
            )
            indices_by_head = [indices for _ in range(args.kv_heads)]
            sparse_fn = lambda: grouped_gqa_sdpa(
                query, key, value, indices_by_head
            )
            sparse_ms = cuda_time_ms(sparse_fn, args.warmup, args.repeats)
            batched_fn = lambda: batched_ragged_gqa_sdpa(
                query, key, value, indices_by_head
            )
            batched_ms = cuda_time_ms(batched_fn, args.warmup, args.repeats)
            pack_fn = lambda: prepare_batched_ragged_kv(
                key, value, indices_by_head, args.query_heads
            )
            pack_ms = cuda_time_ms(pack_fn, args.warmup, args.repeats)
            packed_key, packed_value, packed_mask = pack_fn()
            attention_only_fn = lambda: prepacked_gqa_sdpa(
                query, packed_key, packed_value, packed_mask
            )
            attention_only_ms = cuda_time_ms(
                attention_only_fn, args.warmup, args.repeats
            )
            amortized = {
                str(steps): (pack_ms + steps * attention_only_ms) / steps
                for steps in [1, 4, 16, 64]
            }
            rows.append(
                {
                    "sequence_length": sequence_length,
                    "total_blocks": total_blocks,
                    "active_blocks": len(blocks),
                    "active_tokens": int(indices.numel()),
                    "dense_ms": dense_ms,
                    "grouped_gqa_ms": sparse_ms,
                    "grouped_speedup": dense_ms / sparse_ms,
                    "batched_ragged_gqa_ms": batched_ms,
                    "batched_ragged_speedup": dense_ms / batched_ms,
                    "pack_once_ms": pack_ms,
                    "prepacked_attention_ms": attention_only_ms,
                    **{
                        f"amortized_{steps}_step_ms": value
                        for steps, value in amortized.items()
                    },
                    **{
                        f"amortized_{steps}_step_speedup": dense_ms / value
                        for steps, value in amortized.items()
                    },
                    "kv_fraction": indices.numel() / sequence_length,
                }
            )
        full_indices = torch.arange(sequence_length, device=device)
        dense_output = dense_gqa_sdpa(query, key, value)
        grouped_full = grouped_gqa_sdpa(
            query, key, value, [full_indices for _ in range(args.kv_heads)]
        )
        batched_full = batched_ragged_gqa_sdpa(
            query, key, value, [full_indices for _ in range(args.kv_heads)]
        )
        max_error = float((dense_output - grouped_full).abs().max().item())
        if max_error > 5.0e-3:
            raise AssertionError(f"full-index grouped path mismatch: {max_error}")
        batched_error = float((dense_output - batched_full).abs().max().item())
        if batched_error > 5.0e-3:
            raise AssertionError(f"full-index batched path mismatch: {batched_error}")
        del query, key, value, dense_output, grouped_full, batched_full
        torch.cuda.empty_cache()

    with (output_dir / "kernel_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "backend": "group-loop versus one-call padded-ragged physical-GQA gather + SDPA",
        "query_heads": args.query_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "rows": rows,
        "note": (
            "Grouped timing uses one index_select+SDPA per physical KV head. Batched-ragged "
            "timing pads physical groups, gathers once into a tensor, and uses one SDPA call. "
            "Amortized columns reuse one packed physical union for 1/4/16/64 query steps. "
            "These are real reduced-compute references, not fused production kernels."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
