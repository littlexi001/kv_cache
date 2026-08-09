from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from profile_real_qk import (
    QKCapture,
    captured_qk,
    parse_pairs,
    resolve_dtype,
    run_base_model,
    shard_bounds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sharded pre-RoPE SVD32 K index for shared XSum memory."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--pairs", default="3:10,21:8,6:7,16:14")
    parser.add_argument("--svd_rank", type=int, default=32)
    parser.add_argument("--calibration_blocks", type=int, default=8192)
    parser.add_argument("--batch_blocks", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def pair_specs_for_model(
    model: AutoModelForCausalLM, pair_text: str
) -> list[dict[str, int]]:
    query_heads = int(model.config.num_attention_heads)
    kv_heads = int(model.config.num_key_value_heads)
    repeat_groups = query_heads // kv_heads
    specs = []
    for layer, query_head in parse_pairs(pair_text):
        specs.append(
            {
                "layer": layer,
                "query_head": query_head,
                "kv_head": query_head // repeat_groups,
            }
        )
    return specs


@torch.inference_mode()
def fit_covariance_basis(
    *,
    model: AutoModelForCausalLM,
    capture: QKCapture,
    blocks: np.ndarray,
    pair_specs: list[dict[str, int]],
    calibration_blocks: int,
    batch_blocks: int,
    svd_rank: int,
    device: torch.device,
) -> dict[str, Any]:
    count = min(calibration_blocks, len(blocks))
    profiles = len(pair_specs)
    head_dim = int(model.config.head_dim)
    sums = torch.zeros(profiles, head_dim, dtype=torch.float64, device=device)
    cross = torch.zeros(profiles, head_dim, head_dim, dtype=torch.float64, device=device)
    tokens = 0
    for start in range(0, count, batch_blocks):
        input_ids = torch.from_numpy(
            np.asarray(blocks[start : start + batch_blocks], dtype=np.int64)
        ).to(device)
        run_base_model(model, capture, input_ids)
        _, keys = captured_qk(model, capture, pair_specs, "pre_rope_block_qk")
        matrix = keys.reshape(-1, profiles, head_dim).float()
        sums += matrix.sum(dim=0).to(torch.float64)
        cross += torch.einsum("npd,npe->pde", matrix, matrix).to(torch.float64)
        tokens += int(matrix.shape[0])
    means = sums / tokens
    covariance = cross - tokens * torch.einsum("pd,pe->pde", means, means)
    basis_parts = []
    retained = []
    eigenvalues = []
    for profile in range(profiles):
        values, vectors = torch.linalg.eigh(covariance[profile])
        order = torch.argsort(values, descending=True)
        values = values.index_select(0, order).clamp_min(0)
        vectors = vectors.index_select(1, order)
        basis_parts.append(vectors[:, :svd_rank].to(torch.float32).cpu())
        retained.append(float((values[:svd_rank].sum() / values.sum().clamp_min(1e-30)).item()))
        eigenvalues.append([float(item) for item in values.cpu().tolist()])
    return {
        "basis": torch.stack(basis_parts),
        "mean": means.to(torch.float32).cpu(),
        "retained_energy": retained,
        "eigenvalues": eigenvalues,
        "calibration_blocks": count,
        "calibration_tokens": tokens,
        "basis_fit": "online centered K covariance eigendecomposition",
    }


@torch.inference_mode()
def write_projected_blocks(
    *,
    model: AutoModelForCausalLM,
    capture: QKCapture,
    blocks: np.ndarray,
    pair_specs: list[dict[str, int]],
    basis: torch.Tensor,
    output_path: Path,
    batch_blocks: int,
    device: torch.device,
    log_every: int,
    rank: int,
) -> tuple[float, int]:
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(blocks), blocks.shape[1], len(pair_specs), basis.shape[-1]),
    )
    started = time.perf_counter()
    batches = math.ceil(len(blocks) / batch_blocks)
    for batch_index, start in enumerate(range(0, len(blocks), batch_blocks)):
        input_ids = torch.from_numpy(
            np.asarray(blocks[start : start + batch_blocks], dtype=np.int64)
        ).to(device)
        run_base_model(model, capture, input_ids)
        _, keys = captured_qk(model, capture, pair_specs, "pre_rope_block_qk")
        projected = torch.einsum("btpd,pdr->btpr", keys, basis)
        output[start : start + len(projected)] = projected.to(torch.float16).cpu().numpy()
        if (batch_index + 1) % log_every == 0 or batch_index + 1 == batches:
            elapsed = time.perf_counter() - started
            done = min(start + len(projected), len(blocks)) * int(blocks.shape[1])
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "batch": batch_index + 1,
                        "batches": batches,
                        "tokens": done,
                        "tokens_per_second": done / max(elapsed, 1e-9),
                    }
                ),
                flush=True,
            )
    output.flush()
    elapsed = time.perf_counter() - started
    return elapsed, int(output.nbytes)


def main() -> None:
    args = parse_args()
    rank, world_size, _, device = setup_distributed()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    source_blocks = np.load(data_dir / "source_blocks.npy", mmap_mode="r")

    dtype = resolve_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    pair_specs = pair_specs_for_model(model, args.pairs)
    capture = QKCapture(model, sorted({int(item["layer"]) for item in pair_specs}))

    basis_path = output_dir / "basis.pt"
    if rank == 0:
        payload = fit_covariance_basis(
            model=model,
            capture=capture,
            blocks=base_blocks,
            pair_specs=pair_specs,
            calibration_blocks=args.calibration_blocks,
            batch_blocks=args.batch_blocks,
            svd_rank=args.svd_rank,
            device=device,
        )
        payload.update(
            {
                "pair_specs": pair_specs,
                "svd_rank": args.svd_rank,
                "profile_space": "pre_rope_block_qk",
                "contains_synthetic_vectors": False,
            }
        )
        torch.save(payload, basis_path)
    barrier(world_size)
    payload = torch.load(basis_path, map_location="cpu", weights_only=False)
    basis = payload["basis"].to(device=device, dtype=dtype)

    block_start, block_end = shard_bounds(len(base_blocks), rank, world_size)
    shard = np.asarray(base_blocks[block_start:block_end])
    shard_path = output_dir / f"base_svd_k_rank{rank:03d}.npy"
    torch.cuda.reset_peak_memory_stats(device)
    elapsed, index_bytes = write_projected_blocks(
        model=model,
        capture=capture,
        blocks=shard,
        pair_specs=pair_specs,
        basis=basis,
        output_path=shard_path,
        batch_blocks=args.batch_blocks,
        device=device,
        log_every=args.log_every,
        rank=rank,
    )
    shard_summary = {
        "rank": rank,
        "world_size": world_size,
        "block_start": block_start,
        "block_end": block_end,
        "blocks": block_end - block_start,
        "tokens": (block_end - block_start) * int(base_blocks.shape[1]),
        "path": str(shard_path),
        "index_bytes": index_bytes,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (block_end - block_start)
        * int(base_blocks.shape[1])
        / max(elapsed, 1e-9),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    (output_dir / f"shard_rank{rank:03d}.json").write_text(
        json.dumps(shard_summary, indent=2), encoding="utf-8"
    )
    barrier(world_size)

    if rank == 0:
        flat_sources = np.asarray(source_blocks).reshape(-1, source_blocks.shape[-1])
        source_path = output_dir / "source_svd_k.npy"
        source_elapsed, source_bytes = write_projected_blocks(
            model=model,
            capture=capture,
            blocks=flat_sources,
            pair_specs=pair_specs,
            basis=basis,
            output_path=source_path,
            batch_blocks=args.batch_blocks,
            device=device,
            log_every=args.log_every,
            rank=rank,
        )
        shards = [
            json.loads(
                (output_dir / f"shard_rank{item:03d}.json").read_text(
                    encoding="utf-8"
                )
            )
            for item in range(world_size)
        ]
        summary = {
            "source": "real XSum pre-RoPE block-local SVD K index",
            "data_summary": data_summary,
            "model_name_or_path": args.model_name_or_path,
            "pair_specs": pair_specs,
            "svd_rank": args.svd_rank,
            "profile_space": "pre_rope_block_qk",
            "profile_limit": "each 64-token block is independently encoded",
            "retained_energy": payload["retained_energy"],
            "calibration_blocks": payload["calibration_blocks"],
            "calibration_tokens": payload["calibration_tokens"],
            "base_blocks": len(base_blocks),
            "base_tokens": int(base_blocks.size),
            "source_blocks": int(flat_sources.shape[0]),
            "source_index_path": str(source_path),
            "source_index_bytes": source_bytes,
            "source_elapsed_seconds": source_elapsed,
            "total_base_index_bytes": sum(int(item["index_bytes"]) for item in shards),
            "world_size": world_size,
            "shards": shards,
            "contains_synthetic_vectors": False,
            "selection_uses_target": False,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    barrier(world_size)
    capture.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
