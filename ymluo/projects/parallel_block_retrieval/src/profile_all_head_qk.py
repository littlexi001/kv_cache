from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from profile_real_qk import (
    barrier,
    content_query_positions,
    read_jsonl,
    resolve_dtype,
    setup_distributed,
    shard_bounds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a GQA-aware all-layer/all-head pre-RoPE SVD index without "
            "duplicating shared KV heads."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--svd_rank", type=int, default=32)
    parser.add_argument("--calibration_blocks", type=int, default=32)
    parser.add_argument("--query_vector_tokens", type=int, default=16)
    parser.add_argument(
        "--skip_query_profiles",
        action="store_true",
        help="Build only the K-SVD basis/index; profile leakage-free queries separately.",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(num_layers))
    output: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            output.update(range(start, end + 1))
        else:
            output.add(int(item))
    layers = sorted(output)
    if not layers or layers[0] < 0 or layers[-1] >= num_layers:
        raise ValueError(f"layers must be within [0, {num_layers})")
    return layers


class AllHeadCapture:
    def __init__(self, model: AutoModelForCausalLM, layers: list[int]) -> None:
        self.q: dict[int, torch.Tensor] = {}
        self.k: dict[int, torch.Tensor] = {}
        self.capture_q = False
        self.capture_k = False
        self.q_positions: torch.Tensor | None = None
        self.handles: list[Any] = []
        for layer in layers:
            attention = model.model.layers[layer].self_attn
            self.handles.append(attention.q_norm.register_forward_hook(self._q_hook(layer)))
            self.handles.append(attention.k_norm.register_forward_hook(self._k_hook(layer)))

    def _q_hook(self, layer: int):
        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            if not self.capture_q:
                return
            if self.q_positions is None:
                self.q[layer] = output.detach()
            else:
                self.q[layer] = output.index_select(1, self.q_positions).detach()

        return capture

    def _k_hook(self, layer: int):
        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            if self.capture_k:
                self.k[layer] = output.detach()

        return capture

    def configure(
        self,
        *,
        capture_q: bool,
        capture_k: bool,
        q_positions: torch.Tensor | None = None,
    ) -> None:
        self.q.clear()
        self.k.clear()
        self.capture_q = capture_q
        self.capture_k = capture_k
        self.q_positions = q_positions

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def run_base_model(
    model: AutoModelForCausalLM,
    capture: AllHeadCapture,
    input_ids: torch.Tensor,
) -> None:
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    with torch.inference_mode():
        model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )


def fit_all_head_basis(
    *,
    model: AutoModelForCausalLM,
    capture: AllHeadCapture,
    blocks: np.ndarray,
    records: list[dict[str, Any]],
    layers: list[int],
    calibration_blocks: int,
    svd_rank: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    calibration_tokens = min(calibration_blocks, int(blocks.shape[0])) * int(blocks.shape[1])
    calibration = {
        layer: torch.empty(
            calibration_tokens,
            num_kv_heads,
            head_dim,
            dtype=torch.float16,
        )
        for layer in layers
    }
    cursor = 0
    for record in records:
        if cursor >= calibration_tokens:
            break
        block_start = int(record["block_start"])
        block_count = int(record["block_count"])
        input_array = np.array(
            blocks[block_start : block_start + block_count],
            dtype=np.int64,
            copy=True,
        ).reshape(1, -1)
        input_ids = torch.from_numpy(input_array).to(device=device, non_blocking=True)
        capture.configure(capture_q=False, capture_k=True)
        run_base_model(model, capture, input_ids)
        take = min(calibration_tokens - cursor, int(input_ids.shape[1]))
        for layer in layers:
            calibration[layer][cursor : cursor + take].copy_(
                capture.k[layer][0, :take].to(dtype=torch.float16, device="cpu")
            )
        cursor += take
    if cursor != calibration_tokens:
        raise RuntimeError(f"calibration captured {cursor}/{calibration_tokens} tokens")

    basis = torch.empty(
        len(layers), num_kv_heads, head_dim, svd_rank, dtype=torch.float32
    )
    means = torch.empty(len(layers), num_kv_heads, head_dim, dtype=torch.float32)
    retained_energy = torch.empty(len(layers), num_kv_heads, dtype=torch.float32)
    eigenvalues = torch.empty(len(layers), num_kv_heads, head_dim, dtype=torch.float32)
    for layer_index, layer in enumerate(layers):
        for kv_head in range(num_kv_heads):
            matrix = calibration[layer][:, kv_head].to(device=device, dtype=torch.float32)
            mean = matrix.mean(dim=0)
            centered = matrix - mean
            covariance = centered.transpose(0, 1) @ centered
            values, vectors = torch.linalg.eigh(covariance)
            order = torch.argsort(values, descending=True)
            values = values.index_select(0, order).clamp_min(0.0)
            vectors = vectors.index_select(1, order)
            basis[layer_index, kv_head].copy_(vectors[:, :svd_rank].cpu())
            means[layer_index, kv_head].copy_(mean.cpu())
            eigenvalues[layer_index, kv_head].copy_(values.cpu())
            retained_energy[layer_index, kv_head] = (
                values[:svd_rank].sum() / values.sum().clamp_min(1.0e-30)
            ).cpu()
        print(
            json.dumps(
                {
                    "stage": "basis",
                    "layer": layer,
                    "mean_retained_energy": float(retained_energy[layer_index].mean()),
                }
            ),
            flush=True,
        )
    return {
        "basis": basis,
        "mean": means,
        "retained_energy": retained_energy,
        "eigenvalues": eigenvalues,
        "layers": layers,
        "svd_rank": svd_rank,
        "calibration_blocks": calibration_tokens // int(blocks.shape[1]),
        "calibration_tokens": calibration_tokens,
        "basis_fit": "centered K covariance eigendecomposition, equivalent to right-singular K-SVD subspace",
    }


def profile_all_head_queries(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    capture: AllHeadCapture,
    blocks: np.ndarray,
    queries: list[dict[str, Any]],
    layers: list[int],
    basis: torch.Tensor,
    query_vector_tokens: int,
    num_query_heads: int,
    num_kv_heads: int,
    device: torch.device,
    output_path: Path,
    query_indices: list[int] | None = None,
) -> None:
    repeat_groups = num_query_heads // num_kv_heads
    kv_map = torch.arange(num_query_heads, device=device) // repeat_groups
    basis_device = basis.to(device=device, dtype=next(model.parameters()).dtype)
    selected_indices = query_indices if query_indices is not None else list(range(len(queries)))
    query_count = len(selected_indices)
    svd_rank = int(basis.shape[-1])
    projected = torch.zeros(
        query_count,
        query_vector_tokens,
        len(layers),
        num_query_heads,
        svd_rank,
        dtype=torch.float16,
    )
    mask = torch.zeros(query_count, query_vector_tokens, dtype=torch.bool)
    token_positions: list[list[int]] = []

    selected_queries: list[dict[str, Any]] = []
    for local_query_index, query_index in enumerate(selected_indices):
        query = queries[query_index]
        selected_queries.append(query)
        question = str(query["question"])
        block_start = int(query["block_start"])
        block_count = int(query["block_count"])
        context_array = np.array(
            blocks[block_start : block_start + block_count],
            dtype=np.int64,
            copy=True,
        ).reshape(1, -1)
        context_ids = torch.from_numpy(context_array)
        prompt = f"\nQuestion: {question}\nAnswer:"
        question_start = len("\nQuestion: ")
        prompt_ids, local_positions = content_query_positions(
            tokenizer,
            prompt,
            content_start=question_start,
            content_end=question_start + len(question),
            requested=query_vector_tokens,
        )
        token_ids = torch.cat([context_ids, prompt_ids], dim=1).to(device)
        positions = [int(context_ids.shape[1]) + item for item in local_positions]
        position_tensor = torch.tensor(positions, device=device, dtype=torch.long)
        capture.configure(capture_q=True, capture_k=False, q_positions=position_tensor)
        run_base_model(model, capture, token_ids)

        for layer_index, layer in enumerate(layers):
            query_values = capture.q[layer][0]
            query_basis = basis_device[layer_index].index_select(0, kv_map)
            query_projected = torch.einsum("thd,hdr->thr", query_values, query_basis)
            query_projected = F.normalize(query_projected.float(), dim=-1)
            projected[local_query_index, : len(positions), layer_index].copy_(
                query_projected.to(dtype=torch.float16, device="cpu")
            )
        mask[local_query_index, : len(positions)] = True
        token_positions.append(positions)
        print(
            json.dumps(
                {
                    "stage": "queries",
                    "query_id": query_index,
                    "local_query": local_query_index + 1,
                    "local_queries": query_count,
                }
            ),
            flush=True,
        )

    torch.save(
        {
            "svd_q": projected,
            "mask": mask,
            "token_positions": token_positions,
            "query_indices": selected_indices,
            "queries": selected_queries,
            "layers": layers,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "profile_space": "pre_rope_record_qk",
            "query_vector_mode": "question_content",
            "normalized": True,
        },
        output_path,
    )


def merge_query_profile_shards(
    *,
    shard_paths: list[Path],
    queries: list[dict[str, Any]],
    output_path: Path,
) -> None:
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in shard_paths]
    sample = payloads[0]
    local_shape = sample["svd_q"].shape
    projected = torch.zeros(
        len(queries), *local_shape[1:], dtype=sample["svd_q"].dtype
    )
    mask = torch.zeros(len(queries), sample["mask"].shape[1], dtype=torch.bool)
    token_positions: list[list[int] | None] = [None for _ in queries]
    seen: set[int] = set()
    for payload in payloads:
        indices = [int(item) for item in payload["query_indices"]]
        for local_index, query_index in enumerate(indices):
            if query_index in seen:
                raise RuntimeError(f"duplicate query profile for query {query_index}")
            seen.add(query_index)
            projected[query_index].copy_(payload["svd_q"][local_index])
            mask[query_index].copy_(payload["mask"][local_index])
            token_positions[query_index] = payload["token_positions"][local_index]
    if seen != set(range(len(queries))):
        missing = sorted(set(range(len(queries))) - seen)
        raise RuntimeError(f"missing query profiles: {missing[:10]}")
    torch.save(
        {
            "svd_q": projected,
            "mask": mask,
            "token_positions": token_positions,
            "query_indices": list(range(len(queries))),
            "queries": queries,
            "layers": sample["layers"],
            "num_query_heads": sample["num_query_heads"],
            "num_kv_heads": sample["num_kv_heads"],
            "profile_space": sample["profile_space"],
            "query_vector_mode": sample["query_vector_mode"],
            "normalized": sample["normalized"],
        },
        output_path,
    )


def main() -> None:
    args = parse_args()
    rank, world_size, _local_rank, device = setup_distributed()
    corpus_dir = Path(args.corpus_dir)
    profile_dir = Path(args.profile_dir)
    if rank == 0:
        profile_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)

    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    records_path = corpus_dir / "records.jsonl"
    if records_path.exists():
        records = read_jsonl(records_path)
        record_mode = "corpus_records"
    else:
        records = [
            {"record_id": block_id, "block_start": block_id, "block_count": 1}
            for block_id in range(int(blocks.shape[0]))
        ]
        record_mode = "block_local_fallback"
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    block_count, block_tokens = int(blocks.shape[0]), int(blocks.shape[1])
    dtype = resolve_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    config = model.config
    num_layers = len(model.model.layers)
    num_query_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // num_query_heads))
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if args.svd_rank <= 0 or args.svd_rank > head_dim:
        raise ValueError(f"svd_rank must be within [1, {head_dim}]")
    layers = parse_layers(args.layers, num_layers)
    capture = AllHeadCapture(model, layers)

    basis_path = profile_dir / "basis.pt"
    query_path = profile_dir / "query_profiles.pt"
    if rank == 0:
        basis_payload = fit_all_head_basis(
            model=model,
            capture=capture,
            blocks=blocks,
            records=records,
            layers=layers,
            calibration_blocks=args.calibration_blocks,
            svd_rank=args.svd_rank,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=device,
        )
        basis_payload.update(
            {
                "contains_synthetic_vectors": False,
                "model_name_or_path": args.model_name_or_path,
                "head_dim": head_dim,
                "num_query_heads": num_query_heads,
                "num_kv_heads": num_kv_heads,
            }
        )
        torch.save(basis_payload, basis_path)
    barrier(world_size)
    basis_payload = torch.load(basis_path, map_location="cpu", weights_only=False)
    if not args.skip_query_profiles:
        query_shard_path = profile_dir / f"query_profiles_rank{rank:03d}.pt"
        local_query_indices = list(range(rank, len(queries), world_size))
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
            output_path=query_shard_path,
            query_indices=local_query_indices,
        )
        barrier(world_size)
        if rank == 0:
            merge_query_profile_shards(
                shard_paths=[
                    profile_dir / f"query_profiles_rank{item:03d}.pt"
                    for item in range(world_size)
                ],
                queries=queries,
                output_path=query_path,
            )
        barrier(world_size)
    basis = basis_payload["basis"].to(device=device, dtype=dtype)

    block_start, block_end = shard_bounds(block_count, rank, world_size)
    local_blocks = block_end - block_start
    layer_paths: dict[int, Path] = {
        layer: profile_dir / f"svd_k_layer{layer:03d}_rank{rank:03d}.npy" for layer in layers
    }
    outputs = {
        layer: np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float16,
            shape=(local_blocks, block_tokens, num_kv_heads, args.svd_rank),
        )
        for layer, path in layer_paths.items()
    }

    overlapping_records = [
        record
        for record in records
        if int(record["block_start"]) < block_end
        and int(record["block_start"]) + int(record["block_count"]) > block_start
    ]
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    written_blocks = 0
    for record_index, record in enumerate(overlapping_records):
        record_start = int(record["block_start"])
        record_blocks = int(record["block_count"])
        record_end = record_start + record_blocks
        input_array = np.array(
            blocks[record_start:record_end], dtype=np.int64, copy=True
        ).reshape(1, record_blocks * block_tokens)
        input_ids = torch.from_numpy(input_array).to(device=device, non_blocking=True)
        capture.configure(capture_q=False, capture_k=True)
        run_base_model(model, capture, input_ids)

        overlap_start = max(record_start, block_start)
        overlap_end = min(record_end, block_end)
        count = overlap_end - overlap_start
        record_offset = overlap_start - record_start
        local_offset = overlap_start - block_start
        for layer_index, layer in enumerate(layers):
            record_keys = capture.k[layer].reshape(
                record_blocks, block_tokens, num_kv_heads, head_dim
            )
            selected = record_keys[record_offset : record_offset + count]
            projected = torch.einsum("bthd,hdr->bthr", selected, basis[layer_index])
            projected = F.normalize(projected.float(), dim=-1)
            outputs[layer][local_offset : local_offset + count] = (
                projected.to(dtype=torch.float16, device="cpu").numpy()
            )
        written_blocks += count
        if (record_index + 1) % args.log_every == 0 or record_index + 1 == len(overlapping_records):
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "stage": "index",
                        "rank": rank,
                        "record": record_index + 1,
                        "records": len(overlapping_records),
                        "blocks": written_blocks,
                        "tokens_per_second": written_blocks * block_tokens / max(elapsed, 1.0e-9),
                    }
                ),
                flush=True,
            )
    if written_blocks != local_blocks:
        raise RuntimeError(f"rank {rank} wrote {written_blocks}/{local_blocks} blocks")
    for output in outputs.values():
        output.flush()
    elapsed = time.perf_counter() - started
    shard_summary = {
        "rank": rank,
        "world_size": world_size,
        "block_start": block_start,
        "block_end": block_end,
        "local_blocks": local_blocks,
        "local_tokens": local_blocks * block_tokens,
        "layer_k_paths": {str(layer): str(path) for layer, path in layer_paths.items()},
        "elapsed_seconds": elapsed,
        "tokens_per_second": local_blocks * block_tokens / max(elapsed, 1.0e-9),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "index_bytes": local_blocks * block_tokens * len(layers) * num_kv_heads * args.svd_rank * 2,
    }
    (profile_dir / f"shard_rank{rank:03d}.json").write_text(
        json.dumps(shard_summary, indent=2), encoding="utf-8"
    )
    barrier(world_size)

    if rank == 0:
        shards = [
            json.loads(
                (profile_dir / f"shard_rank{item:03d}.json").read_text(encoding="utf-8")
            )
            for item in range(world_size)
        ]
        retained = basis_payload["retained_energy"]
        summary = {
            "source": "real all-layer/all-head Qwen3 pre-RoPE Q/K with GQA-shared K",
            "model_name_or_path": args.model_name_or_path,
            "profile_space": "pre_rope_record_qk",
            "contains_synthetic_vectors": False,
            "normalized_svd_k": True,
            "num_blocks": block_count,
            "block_tokens": block_tokens,
            "num_tokens": block_count * block_tokens,
            "num_queries": len(queries),
            "record_mode": record_mode,
            "records": len(records),
            "layers": layers,
            "num_layers": len(layers),
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "svd_rank": args.svd_rank,
            "calibration_blocks": basis_payload["calibration_blocks"],
            "calibration_tokens": basis_payload["calibration_tokens"],
            "mean_retained_energy": float(retained.mean()),
            "min_retained_energy": float(retained.min()),
            "max_retained_energy": float(retained.max()),
            "query_vector_tokens": args.query_vector_tokens,
            "query_vector_mode": (
                "question_content" if not args.skip_query_profiles else None
            ),
            "query_profiles_built": not args.skip_query_profiles,
            "profile_world_size": world_size,
            "dtype": args.dtype,
            "basis_path": str(basis_path),
            "query_profiles_path": (
                str(query_path) if not args.skip_query_profiles else None
            ),
            "shards": shards,
            "total_index_bytes": sum(int(shard["index_bytes"]) for shard in shards),
        }
        (profile_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    capture.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
