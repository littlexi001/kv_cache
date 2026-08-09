from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from profile_all_head_qk import (
    AllHeadCapture,
    merge_query_profile_shards,
    profile_all_head_queries,
)
from profile_real_qk import barrier, read_jsonl, resolve_dtype, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile new queries while reusing an existing all-head K-SVD basis/index."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--base_profile_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--query_vector_tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, _local_rank, device = setup_distributed()
    corpus_dir = Path(args.corpus_dir)
    base_profile_dir = Path(args.base_profile_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)

    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    basis_payload = torch.load(
        base_profile_dir / "basis.pt", map_location="cpu", weights_only=False
    )
    layers = [int(layer) for layer in basis_payload["layers"]]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    num_query_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    if int(basis_payload["num_query_heads"]) != num_query_heads:
        raise ValueError("base profile query-head count does not match model")
    if int(basis_payload["num_kv_heads"]) != num_kv_heads:
        raise ValueError("base profile KV-head count does not match model")
    capture = AllHeadCapture(model, layers)
    local_indices = list(range(rank, len(queries), world_size))
    shard_path = output_dir / f"query_profiles_rank{rank:03d}.pt"
    profile_all_head_queries(
        model=model,
        tokenizer=AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True),
        capture=capture,
        blocks=blocks,
        queries=queries,
        layers=layers,
        basis=basis_payload["basis"],
        query_vector_tokens=args.query_vector_tokens,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        device=device,
        output_path=shard_path,
        query_indices=local_indices,
    )
    capture.close()
    barrier(world_size)
    output_path = output_dir / "query_profiles.pt"
    if rank == 0:
        merge_query_profile_shards(
            shard_paths=[
                output_dir / f"query_profiles_rank{shard_rank:03d}.pt"
                for shard_rank in range(world_size)
            ],
            queries=queries,
            output_path=output_path,
        )
        summary = {
            "source": "query-only all-head profile reusing a frozen K-SVD index",
            "queries": len(queries),
            "base_profile_dir": str(base_profile_dir),
            "query_profiles_path": str(output_path),
            "layers": layers,
            "query_vector_tokens": args.query_vector_tokens,
            "contains_synthetic_vectors": False,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
    barrier(world_size)
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

