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
    read_jsonl,
    resolve_dtype,
    run_base_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile raw pre-RoPE K only for blocks named by candidate rows."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--candidate_rows_path", required=True)
    parser.add_argument("--candidate_field", default="anchor_candidates")
    parser.add_argument("--candidate_limit", type=int, default=512)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--pairs", default="3:10,21:9,6:13,6:12")
    parser.add_argument("--batch_blocks", type=int, default=2)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    return parser.parse_args()


def candidate_block_ids(
    rows: list[dict[str, Any]], field: str, limit: int
) -> list[int]:
    return sorted(
        {
            int(block_id)
            for row in rows
            for block_id in row[field][:limit]
        }
    )


def main() -> None:
    args = parse_args()
    if args.batch_blocks <= 0 or args.candidate_limit <= 0:
        raise ValueError("batch_blocks and candidate_limit must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    corpus_dir = Path(args.corpus_dir)
    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    rows = read_jsonl(Path(args.candidate_rows_path))
    block_ids = candidate_block_ids(rows, args.candidate_field, args.candidate_limit)
    if not block_ids:
        raise ValueError("candidate rows produced no blocks")

    model_dtype = resolve_dtype(args.dtype) if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=model_dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    num_query_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // num_query_heads))
    repeat_groups = num_query_heads // num_kv_heads
    pair_specs = []
    for layer, query_head in parse_pairs(args.pairs):
        pair_specs.append(
            {
                "layer": layer,
                "query_head": query_head,
                "kv_head": query_head // repeat_groups,
            }
        )
    capture = QKCapture(model, sorted({int(item["layer"]) for item in pair_specs}))
    raw_path = profile_dir / "raw_k.npy"
    raw = np.lib.format.open_memmap(
        raw_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(block_ids), int(blocks.shape[1]), len(pair_specs), head_dim),
    )
    started = time.perf_counter()
    for start in range(0, len(block_ids), args.batch_blocks):
        batch_ids = block_ids[start : start + args.batch_blocks]
        input_ids = torch.from_numpy(
            np.asarray(blocks[batch_ids], dtype=np.int64)
        ).to(device)
        run_base_model(model, capture, input_ids)
        _, keys = captured_qk(model, capture, pair_specs, "pre_rope_block_qk")
        raw[start : start + len(batch_ids)] = keys.to(torch.float16).cpu().numpy()
        if (start // args.batch_blocks + 1) % 20 == 0 or start + len(batch_ids) == len(block_ids):
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "blocks": start + len(batch_ids),
                        "total_blocks": len(block_ids),
                        "tokens_per_second": (start + len(batch_ids))
                        * int(blocks.shape[1])
                        / max(elapsed, 1.0e-9),
                    }
                ),
                flush=True,
            )
    raw.flush()
    np.save(profile_dir / "block_ids.npy", np.asarray(block_ids, dtype=np.int32))
    elapsed = time.perf_counter() - started
    summary = {
        "source": "sparse candidate-only real Qwen block-local pre-RoPE K",
        "contains_synthetic_vectors": False,
        "context_mode": "block_local",
        "profile_space": "pre_rope_block_qk",
        "corpus_dir": str(corpus_dir),
        "candidate_rows_path": args.candidate_rows_path,
        "candidate_field": args.candidate_field,
        "candidate_limit": args.candidate_limit,
        "num_blocks": len(block_ids),
        "block_tokens": int(blocks.shape[1]),
        "num_tokens": len(block_ids) * int(blocks.shape[1]),
        "pair_specs": pair_specs,
        "head_dim": head_dim,
        "dtype": "float16",
        "device": str(device),
        "elapsed_seconds": elapsed,
        "tokens_per_second": len(block_ids) * int(blocks.shape[1]) / max(elapsed, 1.0e-9),
        "raw_k_path": str(raw_path),
        "block_ids_path": str(profile_dir / "block_ids.npy"),
    }
    (profile_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    capture.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
