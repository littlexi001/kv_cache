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

from profile_real_qk import QKCapture, captured_qk, resolve_dtype, run_base_model, shard_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile block-mean and sampled token K geometry on past-only 10M memory."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--kv_heads", default="0,4")
    parser.add_argument(
        "--token_profiles",
        default="0:0,4:4,8:0,12:4,16:0,20:4,24:0,27:4",
        help="Comma-separated layer:kv_head profiles retaining token-level projected K.",
    )
    parser.add_argument("--svd_rank", type=int, default=32)
    parser.add_argument("--calibration_blocks", type=int, default=1024)
    parser.add_argument("--batch_blocks", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def parse_ints(spec: str) -> list[int]:
    return sorted({int(item.strip()) for item in spec.split(",") if item.strip()})


def parse_token_profiles(spec: str) -> list[tuple[int, int]]:
    output = []
    for item in spec.split(","):
        if not item.strip():
            continue
        layer, kv_head = (int(value) for value in item.split(":"))
        output.append((layer, kv_head))
    return output


def build_specs(
    model: AutoModelForCausalLM, kv_heads: list[int]
) -> list[dict[str, int]]:
    layers = int(model.config.num_hidden_layers)
    query_heads = int(model.config.num_attention_heads)
    key_heads = int(model.config.num_key_value_heads)
    repeats = query_heads // key_heads
    if not kv_heads or min(kv_heads) < 0 or max(kv_heads) >= key_heads:
        raise ValueError("kv_heads are outside the model range")
    return [
        {
            "layer": layer,
            "kv_head": kv_head,
            "query_head": kv_head * repeats,
        }
        for layer in range(layers)
        for kv_head in kv_heads
    ]


@torch.inference_mode()
def fit_profile_bases(
    *,
    model: AutoModelForCausalLM,
    capture: QKCapture,
    blocks: np.ndarray,
    specs: list[dict[str, int]],
    count: int,
    batch_blocks: int,
    svd_rank: int,
    device: torch.device,
) -> dict[str, Any]:
    count = min(count, len(blocks))
    profiles = len(specs)
    head_dim = int(model.config.head_dim)
    sums = torch.zeros(profiles, head_dim, dtype=torch.float32, device=device)
    cross = torch.zeros(profiles, head_dim, head_dim, dtype=torch.float32, device=device)
    tokens = 0
    for start in range(0, count, batch_blocks):
        input_ids = torch.from_numpy(
            np.asarray(blocks[start : start + batch_blocks], dtype=np.int64)
        ).to(device)
        run_base_model(model, capture, input_ids)
        _, keys = captured_qk(model, capture, specs, "pre_rope_block_qk")
        matrix = keys.reshape(-1, profiles, head_dim).float()
        sums += matrix.sum(dim=0)
        cross += torch.einsum("npd,npe->pde", matrix, matrix)
        tokens += int(matrix.shape[0])
    means = sums / tokens
    covariance = cross - tokens * torch.einsum("pd,pe->pde", means, means)
    bases = []
    retained = []
    for profile in range(profiles):
        values, vectors = torch.linalg.eigh(covariance[profile])
        order = torch.argsort(values, descending=True)
        values = values.index_select(0, order).clamp_min(0)
        vectors = vectors.index_select(1, order)
        bases.append(vectors[:, :svd_rank].cpu())
        retained.append(
            float((values[:svd_rank].sum() / values.sum().clamp_min(1e-30)).item())
        )
    return {
        "basis": torch.stack(bases),
        "mean": means.cpu(),
        "retained_energy": retained,
        "calibration_blocks": count,
        "calibration_tokens_per_profile": tokens,
        "basis_fit": "per layer/kv-head centered pre-RoPE K covariance in float32",
    }


@torch.inference_mode()
def write_shard(
    *,
    model: AutoModelForCausalLM,
    capture: QKCapture,
    blocks: np.ndarray,
    specs: list[dict[str, int]],
    basis: torch.Tensor,
    projected_mean: torch.Tensor,
    token_profile_indices: list[int],
    mean_path: Path,
    coherence_path: Path,
    token_path: Path,
    batch_blocks: int,
    device: torch.device,
    log_every: int,
    rank: int,
) -> dict[str, Any]:
    block_tokens = int(blocks.shape[1])
    profiles = len(specs)
    svd_rank = int(basis.shape[-1])
    means = np.lib.format.open_memmap(
        mean_path, mode="w+", dtype=np.float16, shape=(len(blocks), profiles, svd_rank)
    )
    coherence = np.lib.format.open_memmap(
        coherence_path, mode="w+", dtype=np.float16, shape=(len(blocks), profiles)
    )
    token_keys = np.lib.format.open_memmap(
        token_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(blocks), block_tokens, len(token_profile_indices), svd_rank),
    )
    token_indices = torch.tensor(token_profile_indices, dtype=torch.long, device=device)
    started = time.perf_counter()
    batches = math.ceil(len(blocks) / batch_blocks)
    for batch_index, start in enumerate(range(0, len(blocks), batch_blocks)):
        input_ids = torch.from_numpy(
            np.asarray(blocks[start : start + batch_blocks], dtype=np.int64)
        ).to(device)
        run_base_model(model, capture, input_ids)
        _, keys = captured_qk(model, capture, specs, "pre_rope_block_qk")
        projected = torch.einsum("btpd,pdr->btpr", keys, basis)
        centered = projected.float() - projected_mean[None, None, :, :]
        block_mean = centered.mean(dim=1)
        unit = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        block_coherence = unit.mean(dim=1).norm(dim=-1)
        take = len(projected)
        means[start : start + take] = block_mean.to(torch.float16).cpu().numpy()
        coherence[start : start + take] = block_coherence.to(torch.float16).cpu().numpy()
        token_keys[start : start + take] = (
            centered.index_select(2, token_indices).to(torch.float16).cpu().numpy()
        )
        if (batch_index + 1) % log_every == 0 or batch_index + 1 == batches:
            elapsed = time.perf_counter() - started
            done = min(start + take, len(blocks)) * block_tokens
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
    means.flush()
    coherence.flush()
    token_keys.flush()
    elapsed = time.perf_counter() - started
    return {
        "blocks": len(blocks),
        "tokens": int(blocks.size),
        "mean_index_bytes": int(means.nbytes),
        "coherence_index_bytes": int(coherence.nbytes),
        "token_index_bytes": int(token_keys.nbytes),
        "elapsed_seconds": elapsed,
        "tokens_per_second": int(blocks.size) / max(elapsed, 1e-9),
    }


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    if not data_summary.get("past_only"):
        raise ValueError("past-only data required")
    blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    specs = build_specs(model, parse_ints(args.kv_heads))
    token_pairs = parse_token_profiles(args.token_profiles)
    spec_lookup = {(item["layer"], item["kv_head"]): index for index, item in enumerate(specs)}
    token_profile_indices = [spec_lookup[pair] for pair in token_pairs]
    capture = QKCapture(model, list(range(int(model.config.num_hidden_layers))))

    basis_path = output_dir / "basis.pt"
    if rank == 0:
        payload = fit_profile_bases(
            model=model,
            capture=capture,
            blocks=blocks,
            specs=specs,
            count=args.calibration_blocks,
            batch_blocks=args.batch_blocks,
            svd_rank=args.svd_rank,
            device=device,
        )
        payload.update(
            {
                "pair_specs": specs,
                "token_profile_indices": token_profile_indices,
                "token_profile_specs": [specs[index] for index in token_profile_indices],
                "svd_rank": args.svd_rank,
                "profile_space": "pre_rope_block_local",
                "contains_synthetic_vectors": False,
            }
        )
        torch.save(payload, basis_path)
    barrier(world_size)
    payload = torch.load(basis_path, map_location="cpu", weights_only=False)
    basis = payload["basis"].to(device=device, dtype=resolve_dtype(args.dtype))
    projected_mean = torch.einsum(
        "pd,pdr->pr", payload["mean"].to(device).float(), basis.float()
    )

    block_start, block_end = shard_bounds(len(blocks), rank, world_size)
    shard_blocks = np.asarray(blocks[block_start:block_end])
    mean_path = output_dir / f"block_mean_rank{rank:03d}.npy"
    coherence_path = output_dir / f"block_coherence_rank{rank:03d}.npy"
    token_path = output_dir / f"sample_token_k_rank{rank:03d}.npy"
    torch.cuda.reset_peak_memory_stats(device)
    shard_stats = write_shard(
        model=model,
        capture=capture,
        blocks=shard_blocks,
        specs=specs,
        basis=basis,
        projected_mean=projected_mean,
        token_profile_indices=token_profile_indices,
        mean_path=mean_path,
        coherence_path=coherence_path,
        token_path=token_path,
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
        "mean_path": str(mean_path),
        "coherence_path": str(coherence_path),
        "token_path": str(token_path),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        **shard_stats,
    }
    (output_dir / f"shard_rank{rank:03d}.json").write_text(
        json.dumps(shard_summary, indent=2), encoding="utf-8"
    )
    barrier(world_size)
    if rank == 0:
        shards = [
            json.loads((output_dir / f"shard_rank{item:03d}.json").read_text(encoding="utf-8"))
            for item in range(world_size)
        ]
        summary = {
            "source": "PG19 past-only block-mean and sampled-token pre-RoPE K index",
            "data_summary": data_summary,
            "model_name_or_path": args.model_name_or_path,
            "pair_specs": specs,
            "profile_count": len(specs),
            "token_profile_indices": token_profile_indices,
            "token_profile_specs": [specs[index] for index in token_profile_indices],
            "svd_rank": args.svd_rank,
            "retained_energy": payload["retained_energy"],
            "calibration_blocks": payload["calibration_blocks"],
            "base_blocks": len(blocks),
            "base_tokens": int(blocks.size),
            "block_tokens": int(blocks.shape[1]),
            "total_mean_index_bytes": sum(item["mean_index_bytes"] for item in shards),
            "total_coherence_index_bytes": sum(
                item["coherence_index_bytes"] for item in shards
            ),
            "total_sample_token_index_bytes": sum(item["token_index_bytes"] for item in shards),
            "world_size": world_size,
            "shards": shards,
            "profile_limit": "two KV heads per layer; eight profiles retain token-level K",
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
