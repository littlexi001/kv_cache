from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    lengths: tuple[int, ...]
    steps_list: tuple[int, ...]
    page_size: int
    selected_pages_list: tuple[int, ...]
    interval_list: tuple[int, ...]
    recent_tokens: int
    batch_size: int
    layers: int
    q_heads: int
    kv_heads: int
    head_dim: int
    dtype: str
    warmup: int
    seed: int


@dataclass
class TimingRow:
    method: str
    full_len: int
    steps: int
    selected_pages: int
    interval: int
    active_kv_start: int
    active_ratio: float
    reroutes: int
    router_scoring_topk_ms: float
    gather_compact_ms: float
    overhead_ms: float
    attention_ms: float
    total_ms: float
    speedup_vs_full_attention: float
    speedup_vs_full_total: float
    overhead_share: float


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Recent-plus attention/KV subsystem timing. Includes router/scoring/top-k/"
            "KV gather/compact overhead and amortized multi-step attention."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--lengths", default="4096,8192,16384,20000,32768")
    parser.add_argument("--steps_list", default="1,64,256,1024")
    parser.add_argument("--page_size", type=int, default=1024)
    parser.add_argument("--selected_pages_list", default="2,3,4")
    parser.add_argument("--interval_list", default="0,128")
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--q_heads", type=int, default=0)
    parser.add_argument("--kv_heads", type=int, default=0)
    parser.add_argument("--head_dim", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026070606)
    args = parser.parse_args()

    layers = args.layers
    q_heads = args.q_heads
    kv_heads = args.kv_heads
    head_dim = args.head_dim
    if layers <= 0 or q_heads <= 0 or kv_heads <= 0 or head_dim <= 0:
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        layers = layers or int(getattr(model_config, "num_hidden_layers"))
        q_heads = q_heads or int(getattr(model_config, "num_attention_heads"))
        kv_heads = kv_heads or int(getattr(model_config, "num_key_value_heads", q_heads))
        head_dim = head_dim or int(getattr(model_config, "head_dim", model_config.hidden_size // q_heads))

    return Config(
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        lengths=parse_ints(args.lengths),
        steps_list=parse_ints(args.steps_list),
        page_size=args.page_size,
        selected_pages_list=parse_ints(args.selected_pages_list),
        interval_list=parse_ints(args.interval_list),
        recent_tokens=args.recent_tokens,
        batch_size=args.batch_size,
        layers=layers,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=args.dtype,
        warmup=args.warmup,
        seed=args.seed,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed_ms(fn: Any) -> float:
    cuda_sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    cuda_sync()
    return float(start.elapsed_time(end))


def warmup(fn: Any, count: int) -> None:
    for _ in range(max(0, count)):
        fn()
    cuda_sync()


def has_native_gqa(device: torch.device, dtype: torch.dtype) -> bool:
    q = torch.randn(1, 4, 1, 16, device=device, dtype=dtype)
    k = torch.randn(1, 2, 8, 16, device=device, dtype=dtype)
    v = torch.randn(1, 2, 8, 16, device=device, dtype=dtype)
    try:
        _ = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, enable_gqa=True)
        cuda_sync()
        return True
    except TypeError:
        return False


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, native_gqa: bool, q_heads: int, kv_heads: int) -> torch.Tensor:
    if native_gqa and q_heads != kv_heads:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, enable_gqa=True)
    if q_heads != kv_heads:
        repeat = q_heads // kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def attention_loop(
    q_bank: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    base_len: int,
    steps: int,
    native_gqa: bool,
    q_heads: int,
    kv_heads: int,
    reset_interval: int = 0,
) -> None:
    layers = q_bank.shape[1]
    for step in range(steps):
        offset = step if reset_interval <= 0 else step % reset_interval
        kv_len = base_len + offset
        for layer_idx in range(layers):
            _ = sdpa(
                q_bank[step, layer_idx],
                k_cache[layer_idx, :, :, :kv_len, :],
                v_cache[layer_idx, :, :, :kv_len, :],
                native_gqa=native_gqa,
                q_heads=q_heads,
                kv_heads=kv_heads,
            )


def router_score_topk(
    router: nn.Module,
    router_features: torch.Tensor,
    page_query: torch.Tensor,
    page_embeddings: torch.Tensor,
    selected_pages: int,
) -> torch.Tensor:
    _ = router(router_features)
    scores = page_embeddings @ page_query
    return torch.topk(scores, k=selected_pages).indices.sort().values


def gather_recent_plus_compact(
    k_source: torch.Tensor,
    v_source: torch.Tensor,
    k_target: torch.Tensor,
    v_target: torch.Tensor,
    page_indices: torch.Tensor,
    full_len: int,
    page_size: int,
    recent_tokens: int,
) -> int:
    layers, batch, heads, _, dim = k_source.shape
    old_len = max(0, full_len - recent_tokens)
    old_pages = old_len // page_size
    selected_old_len = int(page_indices.numel()) * page_size
    if old_pages > 0 and selected_old_len > 0:
        usable_old_len = old_pages * page_size
        k_pages = k_source[:, :, :, :usable_old_len, :].reshape(layers, batch, heads, old_pages, page_size, dim)
        v_pages = v_source[:, :, :, :usable_old_len, :].reshape(layers, batch, heads, old_pages, page_size, dim)
        gathered_k = k_pages.index_select(3, page_indices).reshape(layers, batch, heads, selected_old_len, dim)
        gathered_v = v_pages.index_select(3, page_indices).reshape(layers, batch, heads, selected_old_len, dim)
        k_target[:, :, :, :selected_old_len, :].copy_(gathered_k)
        v_target[:, :, :, :selected_old_len, :].copy_(gathered_v)
    recent_len = min(recent_tokens, full_len)
    if recent_len > 0:
        k_target[:, :, :, selected_old_len : selected_old_len + recent_len, :].copy_(
            k_source[:, :, :, full_len - recent_len : full_len, :]
        )
        v_target[:, :, :, selected_old_len : selected_old_len + recent_len, :].copy_(
            v_source[:, :, :, full_len - recent_len : full_len, :]
        )
    return selected_old_len + recent_len


def make_row(
    method: str,
    full_len: int,
    steps: int,
    selected_pages: int,
    interval: int,
    active_kv_start: int,
    router_ms: float,
    gather_ms: float,
    attention_ms: float,
    full_attention_ms: float,
) -> TimingRow:
    reroutes = 0 if method == "full_attention" else (1 if interval <= 0 else max(1, math.ceil(steps / interval)))
    overhead_ms = 0.0 if method == "full_attention" else reroutes * (router_ms + gather_ms)
    total_ms = overhead_ms + attention_ms
    return TimingRow(
        method=method,
        full_len=full_len,
        steps=steps,
        selected_pages=selected_pages,
        interval=interval,
        active_kv_start=active_kv_start,
        active_ratio=active_kv_start / full_len if full_len else 0.0,
        reroutes=reroutes,
        router_scoring_topk_ms=0.0 if method == "full_attention" else reroutes * router_ms,
        gather_compact_ms=0.0 if method == "full_attention" else reroutes * gather_ms,
        overhead_ms=overhead_ms,
        attention_ms=attention_ms,
        total_ms=total_ms,
        speedup_vs_full_attention=full_attention_ms / attention_ms if attention_ms else 0.0,
        speedup_vs_full_total=full_attention_ms / total_ms if total_ms else 0.0,
        overhead_share=overhead_ms / total_ms if total_ms else 0.0,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = resolve_dtype(config.dtype)
    device = torch.device("cuda")
    torch.manual_seed(config.seed)
    native_gqa = has_native_gqa(device, dtype)

    max_steps = max(config.steps_list)
    max_len = max(config.lengths)
    max_compact_len = max(config.selected_pages_list) * config.page_size + config.recent_tokens + max_steps + 8
    full_capacity = max_len + max_steps + 8

    k_full = torch.randn(
        config.layers,
        config.batch_size,
        config.kv_heads,
        full_capacity,
        config.head_dim,
        device=device,
        dtype=dtype,
    )
    v_full = torch.randn_like(k_full)
    k_compact = torch.empty(
        config.layers,
        config.batch_size,
        config.kv_heads,
        max_compact_len,
        config.head_dim,
        device=device,
        dtype=dtype,
    )
    v_compact = torch.empty_like(k_compact)
    q_bank = torch.randn(
        max_steps,
        config.layers,
        config.batch_size,
        config.q_heads,
        1,
        config.head_dim,
        device=device,
        dtype=dtype,
    )
    router = nn.Sequential(nn.Linear(34, 64), nn.ReLU(), nn.Linear(64, 10)).to(device=device, dtype=dtype)
    router_features = torch.randn(1, 34, device=device, dtype=dtype)

    rows: list[TimingRow] = []
    start_wall = time.perf_counter()

    for full_len in config.lengths:
        num_old_pages = max(1, (max(0, full_len - config.recent_tokens)) // config.page_size)
        page_embeddings = torch.randn(num_old_pages, config.head_dim, device=device, dtype=dtype)
        page_query = torch.randn(config.head_dim, device=device, dtype=dtype)

        full_attention_by_steps: dict[int, float] = {}
        for steps in config.steps_list:
            warmup(
                lambda steps=steps, full_len=full_len: attention_loop(
                    q_bank,
                    k_full,
                    v_full,
                    full_len,
                    steps,
                    native_gqa=native_gqa,
                    q_heads=config.q_heads,
                    kv_heads=config.kv_heads,
                ),
                config.warmup,
            )
            attention_ms = elapsed_ms(
                lambda steps=steps, full_len=full_len: attention_loop(
                    q_bank,
                    k_full,
                    v_full,
                    full_len,
                    steps,
                    native_gqa=native_gqa,
                    q_heads=config.q_heads,
                    kv_heads=config.kv_heads,
                )
            )
            full_attention_by_steps[steps] = attention_ms
            rows.append(
                make_row(
                    "full_attention",
                    full_len,
                    steps,
                    selected_pages=0,
                    interval=0,
                    active_kv_start=full_len,
                    router_ms=0.0,
                    gather_ms=0.0,
                    attention_ms=attention_ms,
                    full_attention_ms=attention_ms,
                )
            )

        for selected_pages in config.selected_pages_list:
            selected = min(selected_pages, num_old_pages)
            selected_indices = router_score_topk(router, router_features, page_query, page_embeddings, selected)
            warmup(
                lambda selected=selected: router_score_topk(
                    router, router_features, page_query, page_embeddings, selected
                ),
                config.warmup,
            )
            router_ms = elapsed_ms(
                lambda selected=selected: router_score_topk(
                    router, router_features, page_query, page_embeddings, selected
                )
            )
            active_start = gather_recent_plus_compact(
                k_full,
                v_full,
                k_compact,
                v_compact,
                selected_indices,
                full_len,
                config.page_size,
                config.recent_tokens,
            )
            warmup(
                lambda selected_indices=selected_indices, full_len=full_len: gather_recent_plus_compact(
                    k_full,
                    v_full,
                    k_compact,
                    v_compact,
                    selected_indices,
                    full_len,
                    config.page_size,
                    config.recent_tokens,
                ),
                config.warmup,
            )
            gather_ms = elapsed_ms(
                lambda selected_indices=selected_indices, full_len=full_len: gather_recent_plus_compact(
                    k_full,
                    v_full,
                    k_compact,
                    v_compact,
                    selected_indices,
                    full_len,
                    config.page_size,
                    config.recent_tokens,
                )
            )
            for interval in config.interval_list:
                reset_interval = 0 if interval <= 0 else interval
                method = f"recent_plus_k{selected_pages}_once" if interval <= 0 else f"recent_plus_k{selected_pages}_interval{interval}"
                for steps in config.steps_list:
                    warmup(
                        lambda steps=steps, active_start=active_start, reset_interval=reset_interval: attention_loop(
                            q_bank,
                            k_compact,
                            v_compact,
                            active_start,
                            steps,
                            native_gqa=native_gqa,
                            q_heads=config.q_heads,
                            kv_heads=config.kv_heads,
                            reset_interval=reset_interval,
                        ),
                        config.warmup,
                    )
                    attention_ms = elapsed_ms(
                        lambda steps=steps, active_start=active_start, reset_interval=reset_interval: attention_loop(
                            q_bank,
                            k_compact,
                            v_compact,
                            active_start,
                            steps,
                            native_gqa=native_gqa,
                            q_heads=config.q_heads,
                            kv_heads=config.kv_heads,
                            reset_interval=reset_interval,
                        )
                    )
                    rows.append(
                        make_row(
                            method,
                            full_len,
                            steps,
                            selected_pages=selected_pages,
                            interval=interval,
                            active_kv_start=active_start,
                            router_ms=router_ms,
                            gather_ms=gather_ms,
                            attention_ms=attention_ms,
                            full_attention_ms=full_attention_by_steps[steps],
                        )
                    )
        torch.cuda.empty_cache()

    wall_seconds = time.perf_counter() - start_wall
    write_csv(output_dir / "recent_plus_attention_subsystem_timing.csv", [asdict(row) for row in rows])
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "native_gqa": native_gqa,
                "wall_seconds": wall_seconds,
                "rows": [asdict(row) for row in rows],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("method,full_len,steps,active_kv_start,active_ratio,overhead_ms,attention_ms,total_ms,speedup_vs_full_total")
    for row in rows:
        print(
            f"{row.method},{row.full_len},{row.steps},{row.active_kv_start},"
            f"{row.active_ratio:.4f},{row.overhead_ms:.4f},{row.attention_ms:.4f},"
            f"{row.total_ms:.4f},{row.speedup_vs_full_total:.4f}"
        )
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
