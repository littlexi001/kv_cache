#!/usr/bin/env python3
"""Section-65 speed benchmark for typed page-gather KV retrieval.

This benchmark isolates the attention/KV subsystem. It does not call
HuggingFace generate(), tokenizer, MLP, or lm_head. The measured method is the
serving-side shape of typed page routing:

1. score pages once for a new query,
2. top-k/select pages to fit a KV budget,
3. gather/compact selected KV once,
4. attend to the compact KV for multiple decode steps.

Full attention is measured as warm-cache attention over the existing full KV.
Per-layer attention and gather timings are multiplied by layer_count; query
router/scoring/top-k are not multiplied because they are query-time operations,
not per-layer transformer attention.
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
    return float(start.elapsed_time(end)) / max(1, float(repeat))


def gqa_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scaling: float) -> torch.Tensor:
    batch, query_heads, head_dim = query.shape
    kv_heads = key.shape[1]
    if query_heads % kv_heads != 0:
        raise ValueError(f"query_heads={query_heads} must be divisible by kv_heads={kv_heads}")
    group = query_heads // kv_heads
    grouped_query = query.view(batch, kv_heads, group, head_dim)
    scores = torch.einsum("bkgd,bkld->bkgl", grouped_query, key) * scaling
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.einsum("bkgl,bkld->bkgd", weights, value)
    return out.reshape(batch, query_heads, head_dim)


def gather_kv(key: torch.Tensor, value: torch.Tensor, active_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    index = active_indices.view(1, 1, -1, 1).expand(key.shape[0], key.shape[1], -1, key.shape[-1])
    return torch.gather(key, dim=2, index=index), torch.gather(value, dim=2, index=index)


def make_active_indices(
    *,
    history_count: int,
    budget_tokens: int,
    page_size: int,
    sink_tokens: int,
    recent_tokens: int,
    include_self: bool,
    device: torch.device,
) -> tuple[torch.Tensor, int, int, int]:
    context_budget = min(max(1, budget_tokens), history_count)
    sink_count = min(max(0, sink_tokens), context_budget)
    recent_count = min(max(0, recent_tokens), max(0, context_budget - sink_count))
    selected_token_budget = max(0, context_budget - sink_count - recent_count)
    selected_pages = math.ceil(selected_token_budget / max(1, page_size)) if selected_token_budget > 0 else 0
    selected_token_count = min(selected_token_budget, selected_pages * page_size)

    parts: list[torch.Tensor] = []
    if sink_count > 0:
        parts.append(torch.arange(0, sink_count, device=device, dtype=torch.long))

    remote_start = sink_count
    remote_end = max(remote_start, history_count - recent_count)
    remote_available = max(0, remote_end - remote_start)
    if selected_token_count > 0 and remote_available > 0:
        # Deterministic page-aligned selection spread through the remote area.
        page_count = max(1, math.ceil(remote_available / max(1, page_size)))
        take_pages = min(selected_pages, page_count)
        if take_pages == 1:
            page_ids = torch.tensor([page_count // 2], device=device, dtype=torch.long)
        else:
            page_ids = torch.linspace(0, page_count - 1, steps=take_pages, device=device).round().to(torch.long)
            page_ids = torch.unique_consecutive(page_ids)[:take_pages]
        remote_indices = []
        for page_id in page_ids.tolist():
            start = remote_start + page_id * page_size
            end = min(start + page_size, remote_end)
            if start < end:
                remote_indices.append(torch.arange(start, end, device=device, dtype=torch.long))
        if remote_indices:
            remote = torch.cat(remote_indices)[:selected_token_count]
            parts.append(remote)

    if recent_count > 0:
        parts.append(torch.arange(history_count - recent_count, history_count, device=device, dtype=torch.long))
    if include_self:
        parts.append(torch.tensor([history_count], device=device, dtype=torch.long))

    active = torch.unique(torch.cat(parts), sorted=True) if parts else torch.tensor([history_count], device=device)
    if active.numel() > context_budget + int(include_self):
        active = active[: context_budget + int(include_self)]
    return active.contiguous(), selected_pages, selected_token_count, int(active.numel())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--full_kv_lens", default="8192,16384,32768")
    parser.add_argument("--budgets", default="256,512,1024,2048")
    parser.add_argument("--steps", default="1,16,64,256,1024")
    parser.add_argument("--page_size", type=int, default=256)
    parser.add_argument("--sink_tokens", type=int, default=64)
    parser.add_argument("--recent_tokens", type=int, default=256)
    parser.add_argument("--batch_count", type=int, default=1)
    parser.add_argument("--layer_count", type=int, default=32)
    parser.add_argument("--query_head_count", type=int, default=32)
    parser.add_argument("--kv_head_count", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for CUDA event timing.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_kv_lens = parse_int_list(args.full_kv_lens)
    budgets = parse_int_list(args.budgets)
    step_values = parse_int_list(args.steps)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    scaling = 1.0 / math.sqrt(args.head_dim)
    torch.manual_seed(args.seed)

    rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []

    for full_kv_len in full_kv_lens:
        history_count = full_kv_len - 1
        query = torch.randn(args.batch_count, args.query_head_count, args.head_dim, device=device, dtype=dtype)
        key = torch.randn(args.batch_count, args.kv_head_count, full_kv_len, args.head_dim, device=device, dtype=dtype)
        value = torch.randn_like(key)
        page_count = max(1, math.ceil(history_count / args.page_size))
        page_query = torch.randn(args.head_dim, device=device, dtype=dtype)
        page_embeddings = torch.randn(page_count, args.head_dim, device=device, dtype=dtype)
        page_scores = torch.empty(page_count, device=device, dtype=dtype)
        torch.cuda.synchronize()

        def full_attention_fn() -> torch.Tensor:
            return gqa_attention(query, key, value, scaling)

        full_attention_one_layer_ms = cuda_time_ms(full_attention_fn, args.warmup, args.repeat)
        full_attention_ms = full_attention_one_layer_ms * args.layer_count

        for steps in step_values:
            rows.append(
                {
                    "method": "full_attention",
                    "full_kv_len": full_kv_len,
                    "active_kv_len": full_kv_len,
                    "page_size": args.page_size,
                    "selected_pages": page_count,
                    "steps": steps,
                    "router_time": 0.0,
                    "scoring_time": 0.0,
                    "topk_time": 0.0,
                    "gather_time": 0.0,
                    "summary_read_time": 0.0,
                    "attention_time": full_attention_ms * steps,
                    "total_time": full_attention_ms * steps,
                    "overhead_share": 0.0,
                    "speedup_vs_full_attention": 1.0,
                    "refresh_count": 0,
                    "per_full_attention_ms": full_attention_ms,
                    "per_selected_attention_ms": "",
                    "per_query_overhead_ms": 0.0,
                    "dtype": args.dtype,
                    "layer_count": args.layer_count,
                    "query_head_count": args.query_head_count,
                    "kv_head_count": args.kv_head_count,
                    "head_dim": args.head_dim,
                    "batch_count": args.batch_count,
                    "sink_tokens": args.sink_tokens,
                    "recent_tokens": args.recent_tokens,
                }
            )

        for budget in budgets:
            active_indices, selected_pages, selected_token_count, active_kv_len = make_active_indices(
                history_count=history_count,
                budget_tokens=budget,
                page_size=args.page_size,
                sink_tokens=args.sink_tokens,
                recent_tokens=args.recent_tokens,
                include_self=True,
                device=device,
            )
            selected_key, selected_value = gather_kv(key, value, active_indices)
            torch.cuda.synchronize()

            def page_scoring_fn() -> torch.Tensor:
                return torch.matmul(page_embeddings, page_query)

            def page_topk_fn() -> torch.Tensor:
                if selected_pages <= 0:
                    return page_scores
                return torch.topk(page_scores, k=min(selected_pages, page_scores.numel()), dim=0, sorted=False).indices

            def gather_fn() -> tuple[torch.Tensor, torch.Tensor]:
                return gather_kv(key, value, active_indices)

            def selected_attention_fn() -> torch.Tensor:
                return gqa_attention(query, selected_key, selected_value, scaling)

            scoring_ms = cuda_time_ms(page_scoring_fn, args.warmup, args.repeat)
            page_scores = page_scoring_fn()
            torch.cuda.synchronize()
            topk_ms = 0.0 if selected_pages <= 0 else cuda_time_ms(page_topk_fn, args.warmup, args.repeat)
            gather_one_layer_ms = cuda_time_ms(gather_fn, args.warmup, args.repeat)
            selected_attention_one_layer_ms = cuda_time_ms(selected_attention_fn, args.warmup, args.repeat)

            gather_ms = gather_one_layer_ms * args.layer_count
            selected_attention_ms = selected_attention_one_layer_ms * args.layer_count
            overhead_ms = scoring_ms + topk_ms + gather_ms
            timing_rows.append(
                {
                    "full_kv_len": full_kv_len,
                    "budget_tokens": budget,
                    "active_kv_len": active_kv_len,
                    "page_count": page_count,
                    "selected_pages": selected_pages,
                    "selected_page_tokens": selected_token_count,
                    "full_attention_one_layer_ms": full_attention_one_layer_ms,
                    "selected_attention_one_layer_ms": selected_attention_one_layer_ms,
                    "gather_one_layer_ms": gather_one_layer_ms,
                    "page_scoring_ms": scoring_ms,
                    "page_topk_ms": topk_ms,
                    "full_attention_scaled_ms": full_attention_ms,
                    "selected_attention_scaled_ms": selected_attention_ms,
                    "gather_scaled_ms": gather_ms,
                }
            )

            for steps in step_values:
                attention_time = selected_attention_ms * steps
                total_time = overhead_ms + attention_time
                full_total = full_attention_ms * steps
                rows.append(
                    {
                        "method": "typed_page_gather_once",
                        "full_kv_len": full_kv_len,
                        "active_kv_len": active_kv_len,
                        "page_size": args.page_size,
                        "selected_pages": selected_pages,
                        "steps": steps,
                        "router_time": 0.0,
                        "scoring_time": scoring_ms,
                        "topk_time": topk_ms,
                        "gather_time": gather_ms,
                        "summary_read_time": 0.0,
                        "attention_time": attention_time,
                        "total_time": total_time,
                        "overhead_share": overhead_ms / total_time if total_time > 0 else 0.0,
                        "speedup_vs_full_attention": full_total / total_time if total_time > 0 else 0.0,
                        "refresh_count": 1,
                        "per_full_attention_ms": full_attention_ms,
                        "per_selected_attention_ms": selected_attention_ms,
                        "per_query_overhead_ms": overhead_ms,
                        "dtype": args.dtype,
                        "layer_count": args.layer_count,
                        "query_head_count": args.query_head_count,
                        "kv_head_count": args.kv_head_count,
                        "head_dim": args.head_dim,
                        "batch_count": args.batch_count,
                        "sink_tokens": args.sink_tokens,
                        "recent_tokens": args.recent_tokens,
                    }
                )

        del query, key, value, page_query, page_embeddings
        torch.cuda.empty_cache()

    csv_path = output_dir / "attention_kv_subsystem_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    timing_path = output_dir / "primitive_timings.csv"
    with timing_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
        writer.writeheader()
        writer.writerows(timing_rows)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "args": vars(args),
        "csv_path": str(csv_path),
        "primitive_timings_path": str(timing_path),
        "notes": [
            "Section-65 attention/KV subsystem benchmark.",
            "Excludes tokenizer, HF generate, full transformer MLP, lm_head, and Python decode loop.",
            "Full attention is warm-cache attention over existing full KV; no prompt re-prefill is included.",
            "typed_page_gather_once pays page scoring/top-k/gather once per query and amortizes it over decode steps.",
            "Scoring/top-k are query-time operations and are not multiplied by layer_count; attention/gather are per-layer and scaled by layer_count.",
            "Synthetic page scoring uses page embedding dot query to approximate learned/semantic page scorer cost; text CPU heuristics are excluded from attention/KV subsystem timing.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
