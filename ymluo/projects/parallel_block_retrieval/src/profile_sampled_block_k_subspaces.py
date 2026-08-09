from __future__ import annotations

import argparse
import json
import statistics
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


RANKS = (1, 2, 4, 8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure uncentered and residual intrinsic rank of sampled real block K."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--pairs", default="3:10,21:8,6:7,16:14")
    parser.add_argument("--sample_blocks", type=int, default=1024)
    parser.add_argument("--batch_blocks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def rank_for(cumulative: torch.Tensor, threshold: float) -> torch.Tensor:
    return (cumulative < threshold).sum(dim=-1) + 1


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def main() -> None:
    args = parse_args()
    if args.sample_blocks <= 0 or args.batch_blocks <= 0:
        raise ValueError("sample_blocks and batch_blocks must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    corpus_dir = Path(args.corpus_dir)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    sample_count = min(args.sample_blocks, len(blocks))
    block_ids = np.sort(rng.choice(len(blocks), size=sample_count, replace=False))

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
    pair_specs = [
        {
            "layer": layer,
            "query_head": query_head,
            "kv_head": query_head // repeat_groups,
        }
        for layer, query_head in parse_pairs(args.pairs)
    ]
    capture = QKCapture(model, sorted({item["layer"] for item in pair_specs}))

    metrics: dict[str, list[list[float]]] = {}
    for name in (
        "uncentered_rank90",
        "uncentered_rank95",
        "residual_rank90",
        "residual_rank95",
        "uncentered_effective_rank",
        "residual_effective_rank",
    ):
        metrics[name] = [[] for _ in pair_specs]
    for prefix in ("uncentered", "residual"):
        for rank in RANKS:
            metrics[f"{prefix}_energy_rank{rank}"] = [[] for _ in pair_specs]

    started = time.perf_counter()
    for start in range(0, sample_count, args.batch_blocks):
        ids = block_ids[start : start + args.batch_blocks]
        input_ids = torch.from_numpy(np.asarray(blocks[ids], dtype=np.int64)).to(device)
        run_base_model(model, capture, input_ids)
        _, keys = captured_qk(model, capture, pair_specs, "pre_rope_block_qk")
        for prefix, matrix in (
            ("uncentered", keys.float()),
            ("residual", keys.float() - keys.float().mean(dim=1, keepdim=True)),
        ):
            arranged = matrix.permute(0, 2, 1, 3).reshape(
                -1, matrix.shape[1], matrix.shape[3]
            )
            singular = torch.linalg.svdvals(arranged)
            energy = singular.square()
            probability = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1.0e-30)
            cumulative = probability.cumsum(dim=-1)
            effective = torch.exp(
                -(probability * torch.log(probability.clamp_min(1.0e-30))).sum(dim=-1)
            )
            batch_size = len(ids)
            probability = probability.reshape(batch_size, len(pair_specs), -1)
            cumulative = cumulative.reshape(batch_size, len(pair_specs), -1)
            effective = effective.reshape(batch_size, len(pair_specs))
            for profile in range(len(pair_specs)):
                metrics[f"{prefix}_rank90"][profile].extend(
                    rank_for(cumulative[:, profile], 0.90).cpu().tolist()
                )
                metrics[f"{prefix}_rank95"][profile].extend(
                    rank_for(cumulative[:, profile], 0.95).cpu().tolist()
                )
                metrics[f"{prefix}_effective_rank"][profile].extend(
                    effective[:, profile].cpu().tolist()
                )
                for rank in RANKS:
                    metrics[f"{prefix}_energy_rank{rank}"][profile].extend(
                        cumulative[:, profile, min(rank, cumulative.shape[-1]) - 1]
                        .cpu()
                        .tolist()
                    )
        completed = min(start + args.batch_blocks, sample_count)
        print(json.dumps({"sampled_blocks": completed, "total": sample_count}), flush=True)

    elapsed = time.perf_counter() - started
    profiles = []
    for profile, spec in enumerate(pair_specs):
        item: dict[str, Any] = {"profile": profile, **spec}
        for name, per_profile in metrics.items():
            item[name] = summarize(per_profile[profile])
        profiles.append(item)
    payload = {
        "source": "sampled real block-local pre-RoPE K singular spectra",
        "contains_synthetic_vectors": False,
        "model": args.model_name_or_path,
        "corpus_dir": str(corpus_dir),
        "corpus_blocks": int(len(blocks)),
        "block_tokens": int(blocks.shape[1]),
        "sample_blocks": sample_count,
        "seed": args.seed,
        "pair_specs": pair_specs,
        "profiles": profiles,
        "elapsed_seconds": elapsed,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    capture.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
