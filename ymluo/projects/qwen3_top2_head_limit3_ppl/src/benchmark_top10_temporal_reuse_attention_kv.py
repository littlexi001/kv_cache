#!/usr/bin/env python3
"""Section-65 style attention/KV subsystem benchmark for top10 temporal reuse.

This benchmark intentionally excludes tokenizer, HF model forward, MLP, lm_head,
and decode-loop overhead. It times only operations that touch the historical KV:

* full attention over the full KV cache
* query/K scoring used for top10 refresh
* top-k selection
* selected KV gather/compact
* attention over selected KV

The measured unit is one transformer layer. Reported totals are multiplied by
``--layer_count`` so the CSV approximates the whole model's attention/KV
subsystem under the same per-layer dimensions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value}")


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def cuda_time_ms(fn: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(repeat)


def make_active_indices(
    *,
    batch_count: int,
    head_count: int,
    history_count: int,
    top_count: int,
    sink_tokens: int,
    recent_tokens: int,
    include_self: bool,
    device: torch.device,
) -> torch.Tensor:
    sink_count = min(max(0, sink_tokens), history_count)
    recent_count = min(max(0, recent_tokens), max(0, history_count - sink_count))
    remote_start = sink_count
    remote_end = max(remote_start, history_count - recent_count)
    remote_available = max(0, remote_end - remote_start)
    remote_count = min(top_count, remote_available)

    if remote_count > 0:
        # Deterministic spread across the remote region. This avoids accidental
        # overlap with protected sink/recent tokens and keeps active length fixed.
        remote = torch.linspace(
            remote_start,
            remote_end - 1,
            steps=remote_count,
            device=device,
            dtype=torch.float32,
        ).round().to(torch.long)
        remote = torch.unique_consecutive(remote)
        if remote.numel() < remote_count:
            fill = torch.arange(remote_start, remote_end, device=device, dtype=torch.long)
            remote = fill[:remote_count]
        else:
            remote = remote[:remote_count]
    else:
        remote = torch.empty(0, device=device, dtype=torch.long)

    parts = []
    if sink_count > 0:
        parts.append(torch.arange(0, sink_count, device=device, dtype=torch.long))
    if remote.numel() > 0:
        parts.append(remote)
    if recent_count > 0:
        parts.append(torch.arange(history_count - recent_count, history_count, device=device, dtype=torch.long))
    if include_self:
        parts.append(torch.tensor([history_count], device=device, dtype=torch.long))

    base = torch.cat(parts) if parts else torch.tensor([history_count], device=device, dtype=torch.long)
    base = torch.unique(base, sorted=True)
    return base.view(1, 1, -1).expand(batch_count, head_count, -1).contiguous()


def attention_full(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scaling: float) -> torch.Tensor:
    scores = torch.matmul(query[:, :, None, :], key.transpose(2, 3)).squeeze(2) * scaling
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.sum(weights[:, :, :, None] * value, dim=2)


def gather_selected(key: torch.Tensor, value: torch.Tensor, active_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = key.shape[-1]
    gather_index = active_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_key = torch.gather(key, dim=2, index=gather_index)
    selected_value = torch.gather(value, dim=2, index=gather_index)
    return selected_key, selected_value


def attention_selected(query: torch.Tensor, selected_key: torch.Tensor, selected_value: torch.Tensor, scaling: float) -> torch.Tensor:
    scores = torch.matmul(query[:, :, None, :], selected_key.transpose(2, 3)).squeeze(2) * scaling
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.sum(weights[:, :, :, None] * selected_value, dim=2)


def maybe_cuda_final_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    active_indices: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    import qabs_cuda_kernels

    valid = torch.ones_like(active_indices, dtype=torch.bool)
    return qabs_cuda_kernels.final_attention(query, key, value, active_indices, valid, scaling).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--full_kv_len", type=int, default=4097, help="History KV plus the current/self token.")
    parser.add_argument("--batch_count", type=int, default=1)
    parser.add_argument("--layer_count", type=int, default=28)
    parser.add_argument("--head_count", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--top_fraction", type=float, default=0.10)
    parser.add_argument("--sink_tokens", type=int, default=64)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--include_self", type=str2bool, default=True)
    parser.add_argument("--refresh_intervals", default="1,16,64")
    parser.add_argument("--steps", default="1,16,64,256,1024")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cuda_final_attention", type=str2bool, default=False)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("This benchmark requires CUDA because it uses CUDA event timing.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    history_count = args.full_kv_len - 1 if args.include_self else args.full_kv_len
    if history_count <= 0:
        raise ValueError("--full_kv_len must leave at least one history token")

    top_count = max(1, math.ceil(args.top_fraction * history_count))
    scaling = 1.0 / math.sqrt(args.head_dim)
    refresh_intervals = parse_int_list(args.refresh_intervals)
    step_values = parse_int_list(args.steps)

    query = torch.randn(args.batch_count, args.head_count, args.head_dim, device=device, dtype=dtype)
    key = torch.randn(args.batch_count, args.head_count, args.full_kv_len, args.head_dim, device=device, dtype=dtype)
    value = torch.randn_like(key)
    key_history = key[:, :, :history_count, :]
    scores_for_topk = torch.randn(args.batch_count, args.head_count, history_count, device=device, dtype=dtype)
    active_indices = make_active_indices(
        batch_count=args.batch_count,
        head_count=args.head_count,
        history_count=history_count,
        top_count=top_count,
        sink_tokens=args.sink_tokens,
        recent_tokens=args.recent_tokens,
        include_self=args.include_self,
        device=device,
    )
    active_kv_len = int(active_indices.shape[-1])

    selected_key, selected_value = gather_selected(key, value, active_indices)
    torch.cuda.synchronize()

    def full_attention_fn() -> torch.Tensor:
        return attention_full(query, key, value, scaling)

    def scoring_fn() -> torch.Tensor:
        return torch.matmul(query[:, :, None, :], key_history.transpose(2, 3)).squeeze(2) * scaling

    def topk_fn() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.topk(scores_for_topk, k=top_count, dim=-1, sorted=False)

    def gather_fn() -> tuple[torch.Tensor, torch.Tensor]:
        return gather_selected(key, value, active_indices)

    if args.use_cuda_final_attention:
        def selected_attention_fn() -> torch.Tensor:
            return maybe_cuda_final_attention(query, key, value, active_indices, scaling)
    else:
        def selected_attention_fn() -> torch.Tensor:
            return attention_selected(query, selected_key, selected_value, scaling)

    timings_one_layer = {
        "full_attention_ms": cuda_time_ms(full_attention_fn, args.warmup, args.repeat),
        "scoring_ms": cuda_time_ms(scoring_fn, args.warmup, args.repeat),
        "topk_ms": cuda_time_ms(topk_fn, args.warmup, args.repeat),
        "gather_ms": cuda_time_ms(gather_fn, args.warmup, args.repeat),
        "selected_attention_ms": cuda_time_ms(selected_attention_fn, args.warmup, args.repeat),
    }
    timings = {name: value * args.layer_count for name, value in timings_one_layer.items()}
    method_gather_ms = 0.0 if args.use_cuda_final_attention else timings["gather_ms"]

    rows: list[dict[str, object]] = []

    full_per_step = timings["full_attention_ms"]
    for steps in step_values:
        rows.append(
            {
                "method": "full_attention",
                "full_kv_len": args.full_kv_len,
                "active_kv_len": args.full_kv_len,
                "page_size": 1,
                "selected_pages": args.full_kv_len,
                "steps": steps,
                "router_time": 0.0,
                "scoring_time": 0.0,
                "topk_time": 0.0,
                "gather_time": 0.0,
                "summary_read_time": 0.0,
                "attention_time": full_per_step * steps,
                "total_time": full_per_step * steps,
                "overhead_share": 0.0,
                "speedup_vs_full_attention": 1.0,
                "refresh_interval": 0,
                "refresh_count": 0,
                "reuse_count": steps,
                "per_full_attention_ms": full_per_step,
                "per_refresh_total_ms": 0.0,
                "per_reuse_total_ms": full_per_step,
                "dtype": args.dtype,
                "layer_count": args.layer_count,
                "head_count": args.head_count,
                "head_dim": args.head_dim,
                "batch_count": args.batch_count,
                "top_fraction": args.top_fraction,
                "sink_tokens": args.sink_tokens,
                "recent_tokens": args.recent_tokens,
                "use_cuda_final_attention": args.use_cuda_final_attention,
            }
        )

    refresh_overhead = timings["scoring_ms"] + timings["topk_ms"] + method_gather_ms
    refresh_attention = timings["selected_attention_ms"]
    # For temporal reuse, selected K/V is compacted once at refresh and reused
    # for the whole interval. Reuse steps should not pay full gather again.
    # Newly generated tail KV can be maintained as a contiguous decode cache; it
    # is not the remote top10 gather cost being benchmarked here.
    reuse_overhead = 0.0
    reuse_attention = timings["selected_attention_ms"]
    for interval in refresh_intervals:
        if interval <= 0:
            continue
        for steps in step_values:
            refresh_count = math.ceil(steps / interval)
            reuse_count = max(0, steps - refresh_count)
            scoring_time = timings["scoring_ms"] * refresh_count
            topk_time = timings["topk_ms"] * refresh_count
            gather_time = method_gather_ms * refresh_count
            attention_time = refresh_attention * refresh_count + reuse_attention * reuse_count
            total_time = scoring_time + topk_time + gather_time + attention_time
            overhead_time = scoring_time + topk_time + gather_time
            full_total = full_per_step * steps
            rows.append(
                {
                    "method": (
                        f"top10_temporal_reuse_k{interval}_indexed_attention"
                        if args.use_cuda_final_attention
                        else f"top10_temporal_reuse_k{interval}_gather_compact"
                    ),
                    "full_kv_len": args.full_kv_len,
                    "active_kv_len": active_kv_len,
                    "page_size": 1,
                    "selected_pages": active_kv_len,
                    "steps": steps,
                    "router_time": 0.0,
                    "scoring_time": scoring_time,
                    "topk_time": topk_time,
                    "gather_time": gather_time,
                    "summary_read_time": 0.0,
                    "attention_time": attention_time,
                    "total_time": total_time,
                    "overhead_share": overhead_time / total_time if total_time > 0 else 0.0,
                    "speedup_vs_full_attention": full_total / total_time if total_time > 0 else 0.0,
                    "refresh_interval": interval,
                    "refresh_count": refresh_count,
                    "reuse_count": reuse_count,
                    "per_full_attention_ms": full_per_step,
                    "per_refresh_total_ms": refresh_overhead + refresh_attention,
                    "per_reuse_total_ms": reuse_overhead + reuse_attention,
                    "dtype": args.dtype,
                    "layer_count": args.layer_count,
                    "head_count": args.head_count,
                    "head_dim": args.head_dim,
                    "batch_count": args.batch_count,
                    "top_fraction": args.top_fraction,
                    "sink_tokens": args.sink_tokens,
                    "recent_tokens": args.recent_tokens,
                    "use_cuda_final_attention": args.use_cuda_final_attention,
                }
            )

    csv_path = output_dir / "attention_kv_subsystem_benchmark.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "args": vars(args),
        "history_count": history_count,
        "top_count": top_count,
        "active_kv_len": active_kv_len,
        "timings_one_layer_ms": timings_one_layer,
        "timings_scaled_by_layer_count_ms": timings,
        "method_gather_ms": method_gather_ms,
        "csv_path": str(csv_path),
        "notes": [
            "Section-65 attention/KV subsystem only; excludes HF forward, MLP, lm_head, tokenizer, and Python decode loop.",
            "Rows are measured for one layer then multiplied by layer_count.",
            "page_size=1 because this token-level top10 method is not page/block based.",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
