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
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    lengths: tuple[int, ...]
    batch_size: int
    q_len: int
    heads: int
    kv_heads: int
    head_dim: int
    dtype: str
    warmup: int
    iters: int
    seed: int


@dataclass
class TimingRow:
    mode: str
    batch_size: int
    q_len: int
    heads: int
    kv_heads: int
    head_dim: int
    kv_len: int
    dtype: str
    warmup: int
    iters: int
    avg_ms: float
    tokens_per_second: float
    kv_bytes: int
    effective_kv_gb_per_second: float
    speedup_vs_max_len: float


def parse_lengths(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Isolated attention/KV kernel timing for decode-like q_len=1.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument(
        "--lengths",
        default="1532,1616,2095,2456,3081,3536,4116,4850,8192,10338,16384,19455,32768",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--q_len", type=int, default=1)
    parser.add_argument("--heads", type=int, default=0)
    parser.add_argument("--kv_heads", type=int, default=0)
    parser.add_argument("--head_dim", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026070501)
    args = parser.parse_args()

    heads = args.heads
    kv_heads = args.kv_heads
    head_dim = args.head_dim
    if heads <= 0 or kv_heads <= 0 or head_dim <= 0:
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        heads = heads or int(getattr(model_config, "num_attention_heads"))
        kv_heads = kv_heads or int(getattr(model_config, "num_key_value_heads", heads))
        head_dim = head_dim or int(getattr(model_config, "head_dim", model_config.hidden_size // heads))

    return Config(
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        lengths=parse_lengths(args.lengths),
        batch_size=args.batch_size,
        q_len=args.q_len,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=args.dtype,
        warmup=args.warmup,
        iters=args.iters,
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


def sdpa_call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, enable_gqa: bool) -> torch.Tensor:
    if enable_gqa:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, enable_gqa=True)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


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


def benchmark_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, warmup: int, iters: int, enable_gqa: bool) -> float:
    for _ in range(warmup):
        _ = sdpa_call(q, k, v, enable_gqa=enable_gqa)
    cuda_sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = sdpa_call(q, k, v, enable_gqa=enable_gqa)
    end.record()
    cuda_sync()
    return float(start.elapsed_time(end)) / max(1, iters)


def make_row(
    config: Config,
    mode: str,
    kv_len: int,
    avg_ms: float,
    kv_heads_for_bytes: int,
    max_len_ms: float,
) -> TimingRow:
    bytes_per_elem = torch.tensor([], dtype=resolve_dtype(config.dtype)).element_size()
    kv_bytes = 2 * config.batch_size * kv_heads_for_bytes * kv_len * config.head_dim * bytes_per_elem
    seconds = avg_ms / 1000.0
    return TimingRow(
        mode=mode,
        batch_size=config.batch_size,
        q_len=config.q_len,
        heads=config.heads,
        kv_heads=config.kv_heads,
        head_dim=config.head_dim,
        kv_len=kv_len,
        dtype=config.dtype,
        warmup=config.warmup,
        iters=config.iters,
        avg_ms=avg_ms,
        tokens_per_second=(config.batch_size * config.q_len) / seconds if seconds else 0.0,
        kv_bytes=kv_bytes,
        effective_kv_gb_per_second=(kv_bytes / seconds / 1e9) if seconds else 0.0,
        speedup_vs_max_len=max_len_ms / avg_ms if avg_ms else 0.0,
    )


def run_mode(config: Config, mode: str, dtype: torch.dtype) -> list[TimingRow]:
    device = torch.device("cuda")
    rows_raw: list[tuple[int, float, int]] = []
    max_len = max(config.lengths)
    torch.manual_seed(config.seed)
    native_gqa = has_native_gqa(device, dtype)

    for kv_len in sorted(config.lengths):
        q = torch.randn(
            config.batch_size,
            config.heads,
            config.q_len,
            config.head_dim,
            device=device,
            dtype=dtype,
        )
        if mode == "sdpa_repeated_kv":
            k_heads = config.heads
            enable_gqa = False
        elif mode == "sdpa_gqa":
            k_heads = config.kv_heads
            enable_gqa = native_gqa and config.kv_heads != config.heads
        else:
            raise ValueError(mode)
        k = torch.randn(config.batch_size, k_heads, kv_len, config.head_dim, device=device, dtype=dtype)
        v = torch.randn(config.batch_size, k_heads, kv_len, config.head_dim, device=device, dtype=dtype)
        if mode == "sdpa_gqa" and not native_gqa and config.kv_heads != config.heads:
            repeats = config.heads // config.kv_heads
            k_for_kernel = k.repeat_interleave(repeats, dim=1)
            v_for_kernel = v.repeat_interleave(repeats, dim=1)
        else:
            k_for_kernel = k
            v_for_kernel = v
        avg_ms = benchmark_sdpa(q, k_for_kernel, v_for_kernel, config.warmup, config.iters, enable_gqa=enable_gqa)
        rows_raw.append((kv_len, avg_ms, k_heads))
        del q, k, v, k_for_kernel, v_for_kernel
        torch.cuda.empty_cache()

    max_len_ms = next(avg_ms for kv_len, avg_ms, _ in rows_raw if kv_len == max_len)
    return [make_row(config, mode, kv_len, avg_ms, k_heads, max_len_ms) for kv_len, avg_ms, k_heads in rows_raw]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2026070501)

    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = resolve_dtype(config.dtype)
    rows: list[TimingRow] = []

    start_wall = time.perf_counter()
    for mode in ("sdpa_repeated_kv", "sdpa_gqa"):
        rows.extend(run_mode(config, mode, dtype))
    wall_seconds = time.perf_counter() - start_wall

    write_csv(output_dir / "attention_kernel_timing.csv", [asdict(row) for row in rows])
    payload = {
        "config": asdict(config),
        "wall_seconds": wall_seconds,
        "rows": [asdict(row) for row in rows],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("mode,kv_len,avg_ms,speedup_vs_max_len,effective_kv_gb_s")
    for row in rows:
        print(
            f"{row.mode},{row.kv_len},{row.avg_ms:.6f},"
            f"{row.speedup_vs_max_len:.3f},{row.effective_kv_gb_per_second:.2f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
