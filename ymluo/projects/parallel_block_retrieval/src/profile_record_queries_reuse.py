from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from profile_real_qk import QKCapture, profile_queries, read_jsonl, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile new queries while reusing an existing record-level QK basis/index."
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
    if not torch.cuda.is_available():
        raise RuntimeError("query profiling requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    corpus_dir = Path(args.corpus_dir)
    base_profile_dir = Path(args.base_profile_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    basis_payload = torch.load(
        base_profile_dir / "basis.pt", map_location="cpu", weights_only=False
    )
    pair_specs = [dict(item) for item in basis_payload["pair_specs"]]
    profile_space = str(basis_payload["profile_space"])
    base_summary = json.loads(
        (base_profile_dir / "summary.json").read_text(encoding="utf-8")
    )
    query_vector_mode = str(base_summary["query_vector_mode"])
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    capture = QKCapture(model, sorted({int(item["layer"]) for item in pair_specs}))
    output_path = output_dir / "query_profiles.pt"
    profile_queries(
        model=model,
        tokenizer=AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True),
        capture=capture,
        blocks=blocks,
        queries=queries,
        pair_specs=pair_specs,
        basis=basis_payload["basis"],
        query_vector_tokens=args.query_vector_tokens,
        device=device,
        output_path=output_path,
        profile_space=profile_space,
        query_vector_mode=query_vector_mode,
    )
    capture.close()
    summary = {
        "source": "query-only record QK profile reusing a frozen K-SVD index",
        "queries": len(queries),
        "base_profile_dir": str(base_profile_dir),
        "query_profiles_path": str(output_path),
        "profile_space": profile_space,
        "query_vector_mode": query_vector_mode,
        "query_vector_tokens": args.query_vector_tokens,
        "contains_synthetic_vectors": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

