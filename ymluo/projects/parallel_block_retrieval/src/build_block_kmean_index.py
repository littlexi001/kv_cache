from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from profile_real_qk import (
    QKCapture,
    captured_qk,
    parse_pairs,
    resolve_dtype,
    run_base_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build real pre-RoPE block K-mean and segment-centroid indices."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pairs", default="3:10,21:8,6:7,16:14")
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--batch_blocks", type=int, default=32)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def concentration(keys: torch.Tensor) -> torch.Tensor:
    mean_norm = keys.float().mean(dim=1).norm(dim=-1)
    token_norm = keys.float().norm(dim=-1).mean(dim=1).clamp_min(1.0e-8)
    return mean_norm / token_norm


def distribution(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    return {
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "p05": float(np.percentile(flat, 5)),
        "p95": float(np.percentile(flat, 95)),
    }


def main() -> None:
    args = parse_args()
    if args.segments <= 0 or args.batch_blocks <= 0:
        raise ValueError("segments and batch_blocks must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    block_count, block_tokens = map(int, blocks.shape)
    if block_tokens % args.segments:
        raise ValueError("block token count must be divisible by segments")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=(
            resolve_dtype(args.dtype) if device.type == "cuda" else torch.float32
        ),
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    num_query_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    repeat_groups = num_query_heads // num_kv_heads
    head_dim = int(
        getattr(model.config, "head_dim", model.config.hidden_size // num_query_heads)
    )
    pair_specs: list[dict[str, int]] = []
    for layer, query_head in parse_pairs(args.pairs):
        pair_specs.append(
            {
                "layer": layer,
                "query_head": query_head,
                "kv_head": query_head // repeat_groups,
            }
        )
    profile_count = len(pair_specs)
    capture = QKCapture(model, sorted({item["layer"] for item in pair_specs}))

    pre_mean = np.lib.format.open_memmap(
        output_dir / "pre_k_mean.npy",
        mode="w+",
        dtype=np.float16,
        shape=(block_count, profile_count, head_dim),
    )
    pre_segments = np.lib.format.open_memmap(
        output_dir / "pre_k_segment_mean.npy",
        mode="w+",
        dtype=np.float16,
        shape=(block_count, args.segments, profile_count, head_dim),
    )
    post_mean = np.lib.format.open_memmap(
        output_dir / "post_local_k_mean.npy",
        mode="w+",
        dtype=np.float16,
        shape=(block_count, profile_count, head_dim),
    )
    pre_concentration = np.lib.format.open_memmap(
        output_dir / "pre_concentration.npy",
        mode="w+",
        dtype=np.float16,
        shape=(block_count, profile_count),
    )
    post_concentration = np.lib.format.open_memmap(
        output_dir / "post_local_concentration.npy",
        mode="w+",
        dtype=np.float16,
        shape=(block_count, profile_count),
    )

    segment_tokens = block_tokens // args.segments
    started = time.perf_counter()
    for start in range(0, block_count, args.batch_blocks):
        end = min(block_count, start + args.batch_blocks)
        input_ids = torch.from_numpy(
            np.asarray(blocks[start:end], dtype=np.int64)
        ).to(device)
        run_base_model(model, capture, input_ids)
        _, pre_keys = captured_qk(model, capture, pair_specs, "pre_rope_block_qk")
        _, post_keys = captured_qk(model, capture, pair_specs, "post_rope_record_qk")

        batch_pre_mean = pre_keys.float().mean(dim=1)
        batch_post_mean = post_keys.float().mean(dim=1)
        batch_segments = pre_keys.float().reshape(
            end - start,
            args.segments,
            segment_tokens,
            profile_count,
            head_dim,
        ).mean(dim=2)
        pre_mean[start:end] = batch_pre_mean.to(torch.float16).cpu().numpy()
        pre_segments[start:end] = batch_segments.to(torch.float16).cpu().numpy()
        post_mean[start:end] = batch_post_mean.to(torch.float16).cpu().numpy()
        pre_concentration[start:end] = concentration(pre_keys).to(torch.float16).cpu().numpy()
        post_concentration[start:end] = concentration(post_keys).to(torch.float16).cpu().numpy()

        if (start // args.batch_blocks + 1) % 20 == 0 or end == block_count:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "blocks": end,
                        "total_blocks": block_count,
                        "tokens_per_second": end * block_tokens / max(elapsed, 1.0e-9),
                    }
                ),
                flush=True,
            )

    for array in (
        pre_mean,
        pre_segments,
        post_mean,
        pre_concentration,
        post_concentration,
    ):
        array.flush()
    elapsed = time.perf_counter() - started
    payload: dict[str, Any] = {
        "source": "real Qwen block-local K centroids",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "model": args.model_name_or_path,
        "corpus_dir": str(corpus_dir),
        "blocks": block_count,
        "block_tokens": block_tokens,
        "tokens": block_count * block_tokens,
        "pair_specs": pair_specs,
        "profiles": profile_count,
        "head_dim": head_dim,
        "segments": args.segments,
        "dtype": "float16",
        "pre_rope_concentration": distribution(pre_concentration),
        "post_local_rope_concentration": distribution(post_concentration),
        "elapsed_seconds": elapsed,
        "tokens_per_second": block_count * block_tokens / max(elapsed, 1.0e-9),
        "pre_k_mean_path": "pre_k_mean.npy",
        "pre_k_segment_mean_path": "pre_k_segment_mean.npy",
        "post_local_k_mean_path": "post_local_k_mean.npy",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    capture.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
