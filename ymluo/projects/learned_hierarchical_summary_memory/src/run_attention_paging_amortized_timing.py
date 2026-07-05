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
    full_len: int
    steps_list: tuple[int, ...]
    page_size: int
    selected_pages_list: tuple[int, ...]
    interval_list: tuple[int, ...]
    batch_size: int
    layers: int
    heads: int
    head_dim: int
    dtype: str
    seed: int


@dataclass
class TimingRow:
    method: str
    steps: int
    full_len: int
    active_kv_start: int
    page_size: int
    selected_pages: int
    interval: int
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
    parser = argparse.ArgumentParser(description="Attention-only paging/gather amortization benchmark.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--full_len", type=int, default=19455)
    parser.add_argument("--steps_list", default="1,16,64,256,1024")
    parser.add_argument("--page_size", type=int, default=1024)
    parser.add_argument("--selected_pages_list", default="2,4,5")
    parser.add_argument("--interval_list", default="128,1024")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--heads", type=int, default=0)
    parser.add_argument("--head_dim", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=2026070502)
    args = parser.parse_args()

    layers = args.layers
    heads = args.heads
    head_dim = args.head_dim
    if layers <= 0 or heads <= 0 or head_dim <= 0:
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        layers = layers or int(getattr(model_config, "num_hidden_layers"))
        heads = heads or int(getattr(model_config, "num_attention_heads"))
        head_dim = head_dim or int(getattr(model_config, "head_dim", model_config.hidden_size // heads))

    return Config(
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        full_len=args.full_len,
        steps_list=parse_ints(args.steps_list),
        page_size=args.page_size,
        selected_pages_list=parse_ints(args.selected_pages_list),
        interval_list=parse_ints(args.interval_list),
        batch_size=args.batch_size,
        layers=layers,
        heads=heads,
        head_dim=head_dim,
        dtype=args.dtype,
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


def attention_loop(
    q_bank: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    base_len: int,
    steps: int,
    reset_interval: int = 0,
) -> None:
    layers = q_bank.shape[1]
    for step in range(steps):
        offset = step if reset_interval <= 0 else step % reset_interval
        kv_len = base_len + offset
        for layer_idx in range(layers):
            _ = F.scaled_dot_product_attention(
                q_bank[step, layer_idx],
                k_cache[layer_idx, :, :, :kv_len, :],
                v_cache[layer_idx, :, :, :kv_len, :],
                dropout_p=0.0,
                is_causal=False,
            )


def score_pages_and_router(
    router: nn.Module,
    router_features: torch.Tensor,
    page_query: torch.Tensor,
    page_embeddings: torch.Tensor,
    selected_pages: int,
) -> torch.Tensor:
    _ = router(router_features)
    scores = page_embeddings @ page_query
    return torch.topk(scores, k=selected_pages).indices.sort().values


def gather_pages(
    k_source: torch.Tensor,
    v_source: torch.Tensor,
    k_target: torch.Tensor,
    v_target: torch.Tensor,
    page_indices: torch.Tensor,
    page_size: int,
) -> int:
    layers, batch, heads, _, dim = k_source.shape
    num_pages = k_source.shape[3] // page_size
    usable = num_pages * page_size
    selected_len = int(page_indices.numel()) * page_size
    k_pages = k_source[:, :, :, :usable, :].reshape(layers, batch, heads, num_pages, page_size, dim)
    v_pages = v_source[:, :, :, :usable, :].reshape(layers, batch, heads, num_pages, page_size, dim)
    gathered_k = k_pages.index_select(3, page_indices).reshape(layers, batch, heads, selected_len, dim)
    gathered_v = v_pages.index_select(3, page_indices).reshape(layers, batch, heads, selected_len, dim)
    k_target[:, :, :, :selected_len, :].copy_(gathered_k)
    v_target[:, :, :, :selected_len, :].copy_(gathered_v)
    return selected_len


def make_row(
    method: str,
    steps: int,
    config: Config,
    active_kv_start: int,
    selected_pages: int,
    interval: int,
    overhead_ms: float,
    attention_ms: float,
    full_attention_ms: float,
) -> TimingRow:
    total_ms = overhead_ms + attention_ms
    return TimingRow(
        method=method,
        steps=steps,
        full_len=config.full_len,
        active_kv_start=active_kv_start,
        page_size=config.page_size,
        selected_pages=selected_pages,
        interval=interval,
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

    max_steps = max(config.steps_list)
    full_capacity = config.full_len + max_steps + max(config.interval_list, default=0) + 8
    k_full = torch.randn(
        config.layers,
        config.batch_size,
        config.heads,
        full_capacity,
        config.head_dim,
        device=device,
        dtype=dtype,
    )
    v_full = torch.randn_like(k_full)
    q_bank = torch.randn(
        max_steps,
        config.layers,
        config.batch_size,
        config.heads,
        1,
        config.head_dim,
        device=device,
        dtype=dtype,
    )
    num_pages = config.full_len // config.page_size
    page_embeddings = torch.randn(num_pages, config.head_dim, device=device, dtype=dtype)
    page_query = torch.randn(config.head_dim, device=device, dtype=dtype)
    router = nn.Sequential(
        nn.Linear(29, 48),
        nn.ReLU(),
        nn.Linear(48, 8),
    ).to(device=device, dtype=dtype)
    router_features = torch.randn(1, 29, device=device, dtype=dtype)

    # Warm up representative kernels.
    attention_loop(q_bank, k_full, v_full, config.full_len, min(4, max_steps))
    cuda_sync()

    full_attention_by_steps: dict[int, float] = {}
    rows: list[TimingRow] = []
    for steps in config.steps_list:
        full_ms = elapsed_ms(lambda steps=steps: attention_loop(q_bank, k_full, v_full, config.full_len, steps))
        full_attention_by_steps[steps] = full_ms
        rows.append(make_row("full_attention", steps, config, config.full_len, 0, 0, 0.0, full_ms, full_ms))

    for selected_pages in config.selected_pages_list:
        selected_len = selected_pages * config.page_size
        target_capacity = selected_len + max(max_steps, max(config.interval_list, default=0)) + 8
        k_target = torch.empty(
            config.layers,
            config.batch_size,
            config.heads,
            target_capacity,
            config.head_dim,
            device=device,
            dtype=dtype,
        )
        v_target = torch.empty_like(k_target)

        def one_time_prepare() -> None:
            page_indices = score_pages_and_router(router, router_features, page_query, page_embeddings, selected_pages)
            gather_pages(k_full, v_full, k_target, v_target, page_indices, config.page_size)

        for _ in range(5):
            one_time_prepare()
        cuda_sync()
        one_time_overhead = elapsed_ms(one_time_prepare)
        # Keep the selected pages materialized for all one-time amortization rows.
        one_time_prepare()
        cuda_sync()

        for steps in config.steps_list:
            attn_ms = elapsed_ms(
                lambda steps=steps: attention_loop(q_bank, k_target, v_target, selected_len, steps)
            )
            rows.append(
                make_row(
                    f"page_once_{selected_pages}p",
                    steps,
                    config,
                    selected_len,
                    selected_pages,
                    0,
                    one_time_overhead,
                    attn_ms,
                    full_attention_by_steps[steps],
                )
            )

        for interval in config.interval_list:
            if interval <= 0:
                continue
            interval_overhead = one_time_overhead * math.ceil(max_steps / interval)
            for steps in config.steps_list:
                repeats = math.ceil(steps / interval)
                overhead_ms = one_time_overhead * repeats
                attn_ms = elapsed_ms(
                    lambda steps=steps, interval=interval: attention_loop(
                        q_bank, k_target, v_target, selected_len, steps, reset_interval=interval
                    )
                )
                rows.append(
                    make_row(
                        f"page_interval_{interval}_{selected_pages}p",
                        steps,
                        config,
                        selected_len,
                        selected_pages,
                        interval,
                        overhead_ms,
                        attn_ms,
                        full_attention_by_steps[steps],
                    )
                )
            # Make interval_overhead visible in summary.json for the longest sequence.
            _ = interval_overhead

        del k_target, v_target
        torch.cuda.empty_cache()

    write_csv(output_dir / "paging_amortized_timing.csv", [asdict(row) for row in rows])
    (output_dir / "summary.json").write_text(
        json.dumps({"config": asdict(config), "rows": [asdict(row) for row in rows]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("method,steps,active_kv_start,overhead_ms,attention_ms,total_ms,speedup_vs_full_total,overhead_share")
    for row in rows:
        print(
            f"{row.method},{row.steps},{row.active_kv_start},"
            f"{row.overhead_ms:.3f},{row.attention_ms:.3f},{row.total_ms:.3f},"
            f"{row.speedup_vs_full_total:.3f},{row.overhead_share:.4f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
