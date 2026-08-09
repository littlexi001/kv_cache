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
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile real Qwen Q/K vectors and build a true K-SVD index for a block corpus."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--pairs", default="3:10,21:8,6:7,16:14", help="Comma-separated layer:query_head pairs.")
    parser.add_argument("--svd_rank", type=int, default=32)
    parser.add_argument("--calibration_blocks", type=int, default=32)
    parser.add_argument("--batch_blocks", type=int, default=8)
    parser.add_argument("--query_vector_tokens", type=int, default=1)
    parser.add_argument(
        "--skip_query_profiles",
        action="store_true",
        help="Build only the K index; profile leakage-free step queries separately.",
    )
    parser.add_argument(
        "--query_vector_mode",
        choices=["prompt_tail", "question_content"],
        default="prompt_tail",
    )
    parser.add_argument(
        "--profile_space",
        choices=["pre_rope_block_qk", "pre_rope_record_qk", "post_rope_record_qk"],
        default="post_rope_record_qk",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Real Q/K profiling requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def parse_pairs(spec: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        fields = item.split(":")
        if len(fields) != 2:
            raise ValueError(f"Invalid pair {item!r}; expected layer:query_head")
        pair = (int(fields[0]), int(fields[1]))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    if not pairs:
        raise ValueError("At least one layer:query_head pair is required")
    return pairs


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def shard_bounds(total: int, rank: int, world_size: int) -> tuple[int, int]:
    return (total * rank) // world_size, (total * (rank + 1)) // world_size


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class QKCapture:
    def __init__(self, model: AutoModelForCausalLM, layers: list[int]) -> None:
        self.q: dict[int, torch.Tensor] = {}
        self.k: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []
        for layer in layers:
            attention = model.model.layers[layer].self_attn
            self.handles.append(attention.q_norm.register_forward_hook(self._hook(self.q, layer)))
            self.handles.append(attention.k_norm.register_forward_hook(self._hook(self.k, layer)))

    @staticmethod
    def _hook(target: dict[int, torch.Tensor], layer: int):
        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            target[layer] = output.detach()

        return capture

    def clear(self) -> None:
        self.q.clear()
        self.k.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def run_base_model(
    model: AutoModelForCausalLM,
    capture: QKCapture,
    input_ids: torch.Tensor,
) -> None:
    capture.clear()
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    with torch.inference_mode():
        model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )


def stack_keys(
    capture: QKCapture,
    pair_specs: list[dict[str, int]],
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for pair in pair_specs:
        layer_keys = capture.k[pair["layer"]]
        parts.append(layer_keys[:, :, pair["kv_head"], :])
    return torch.stack(parts, dim=2).contiguous()


def stack_queries(
    capture: QKCapture,
    pair_specs: list[dict[str, int]],
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for pair in pair_specs:
        layer_queries = capture.q[pair["layer"]]
        parts.append(layer_queries[:, :, pair["query_head"], :])
    return torch.stack(parts, dim=2).contiguous()


def stack_post_rope_qk(
    model: AutoModelForCausalLM,
    capture: QKCapture,
    pair_specs: list[dict[str, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    first_layer = pair_specs[0]["layer"]
    sample = capture.q[first_layer]
    token_count = int(sample.shape[1])
    position_ids = torch.arange(token_count, device=sample.device, dtype=torch.long)[None, :]
    cos, sin = model.model.rotary_emb(sample, position_ids)
    layer_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in sorted({pair["layer"] for pair in pair_specs}):
        q_pre = capture.q[layer].transpose(1, 2)
        k_pre = capture.k[layer].transpose(1, 2)
        layer_cache[layer] = apply_rotary_pos_emb(q_pre, k_pre, cos, sin)

    query_parts: list[torch.Tensor] = []
    key_parts: list[torch.Tensor] = []
    for pair in pair_specs:
        q_post, k_post = layer_cache[pair["layer"]]
        query_parts.append(q_post[:, pair["query_head"], :, :])
        key_parts.append(k_post[:, pair["kv_head"], :, :])
    queries = torch.stack(query_parts, dim=2).contiguous()
    keys = torch.stack(key_parts, dim=2).contiguous()
    return queries, keys


def captured_qk(
    model: AutoModelForCausalLM,
    capture: QKCapture,
    pair_specs: list[dict[str, int]],
    profile_space: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if profile_space in {"pre_rope_block_qk", "pre_rope_record_qk"}:
        return stack_queries(capture, pair_specs), stack_keys(capture, pair_specs)
    if profile_space == "post_rope_record_qk":
        return stack_post_rope_qk(model, capture, pair_specs)
    raise ValueError(f"Unsupported profile_space: {profile_space}")


def fit_basis(
    model: AutoModelForCausalLM,
    capture: QKCapture,
    blocks: np.ndarray,
    records: list[dict[str, Any]],
    pair_specs: list[dict[str, int]],
    calibration_blocks: int,
    svd_rank: int,
    device: torch.device,
    profile_space: str,
) -> dict[str, Any]:
    count = min(calibration_blocks, int(blocks.shape[0]))
    if profile_space == "pre_rope_block_qk":
        input_ids = torch.from_numpy(np.asarray(blocks[:count], dtype=np.int64)).to(device)
        run_base_model(model, capture, input_ids)
        _, keys = captured_qk(model, capture, pair_specs, profile_space)
        key_matrix = keys.reshape(-1, keys.shape[2], keys.shape[3]).float()
    else:
        remaining_tokens = count * int(blocks.shape[1])
        key_parts: list[torch.Tensor] = []
        for record in records:
            block_start = int(record["block_start"])
            block_count = int(record["block_count"])
            record_ids = np.asarray(
                blocks[block_start : block_start + block_count], dtype=np.int64
            ).reshape(1, -1)
            input_ids = torch.from_numpy(record_ids).to(device)
            run_base_model(model, capture, input_ids)
            _, record_keys = captured_qk(model, capture, pair_specs, profile_space)
            take = min(remaining_tokens, int(record_keys.shape[1]))
            key_parts.append(record_keys[0, :take].float())
            remaining_tokens -= take
            if remaining_tokens <= 0:
                break
        if remaining_tokens > 0:
            raise RuntimeError("Not enough record tokens to fit the requested SVD basis")
        key_matrix = torch.cat(key_parts, dim=0)
    _, profile_count, head_dim = key_matrix.shape
    basis_parts: list[torch.Tensor] = []
    mean_parts: list[torch.Tensor] = []
    energy_parts: list[float] = []
    singular_values: list[list[float]] = []
    for profile in range(profile_count):
        matrix = key_matrix[:, profile, :]
        mean = matrix.mean(dim=0)
        centered = matrix - mean
        _, values, vh = torch.linalg.svd(centered, full_matrices=False)
        rank = min(svd_rank, int(vh.shape[0]))
        basis_parts.append(vh[:rank].transpose(0, 1).contiguous())
        mean_parts.append(mean)
        total_energy = values.square().sum().clamp_min(1.0e-30)
        energy_parts.append(float((values[:rank].square().sum() / total_energy).item()))
        singular_values.append([float(item) for item in values.detach().cpu().tolist()])
    return {
        "basis": torch.stack(basis_parts).cpu(),
        "mean": torch.stack(mean_parts).cpu(),
        "retained_energy": energy_parts,
        "singular_values": singular_values,
        "calibration_blocks": count,
        "calibration_tokens": count * int(blocks.shape[1]),
        "basis_fit": "centered K SVD; retrieval projects raw q and k onto the fitted right-singular subspace",
        "profile_space": profile_space,
    }


def query_positions(token_count: int, requested: int) -> list[int]:
    count = min(max(1, requested), token_count)
    return list(range(token_count - count, token_count))


def content_query_positions(
    tokenizer: AutoTokenizer,
    text: str,
    *,
    content_start: int,
    content_end: int,
    requested: int,
) -> tuple[torch.Tensor, list[int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"][0].tolist()]
    candidates: list[int] = []
    fallback: list[int] = []
    for token_index, (start, end) in enumerate(offsets):
        if end <= content_start or start >= content_end:
            continue
        fallback.append(token_index)
        piece = text[max(start, content_start) : min(end, content_end)]
        if any(character.isalnum() for character in piece):
            candidates.append(token_index)
    positions = candidates or fallback
    if not positions:
        raise RuntimeError("Question content did not produce any query token positions")
    return encoded["input_ids"], positions[-max(1, requested) :]


def profile_queries(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    capture: QKCapture,
    blocks: np.ndarray,
    queries: list[dict[str, Any]],
    pair_specs: list[dict[str, int]],
    basis: torch.Tensor,
    query_vector_tokens: int,
    device: torch.device,
    output_path: Path,
    profile_space: str,
    query_vector_mode: str,
) -> None:
    profile_count, head_dim, svd_rank = basis.shape
    query_count = len(queries)
    raw = torch.zeros(query_count, query_vector_tokens, profile_count, head_dim, dtype=torch.float16)
    projected = torch.zeros(query_count, query_vector_tokens, profile_count, svd_rank, dtype=torch.float16)
    mask = torch.zeros(query_count, query_vector_tokens, dtype=torch.bool)
    token_positions: list[list[int]] = []
    basis_device = basis.to(device=device, dtype=torch.float32)
    for query_index, query_row in enumerate(queries):
        question = str(query_row["question"])
        if profile_space in {"pre_rope_record_qk", "post_rope_record_qk"}:
            block_start = int(query_row["block_start"])
            block_count = int(query_row["block_count"])
            context_ids = torch.from_numpy(
                np.asarray(blocks[block_start : block_start + block_count], dtype=np.int64).reshape(1, -1)
            )
            prompt = f"\nQuestion: {question}\nAnswer:"
            if query_vector_mode == "question_content":
                question_start = len("\nQuestion: ")
                prompt_ids, local_positions = content_query_positions(
                    tokenizer,
                    prompt,
                    content_start=question_start,
                    content_end=question_start + len(question),
                    requested=query_vector_tokens,
                )
            else:
                prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
                    "input_ids"
                ]
                local_positions = query_positions(int(prompt_ids.shape[1]), query_vector_tokens)
            token_ids = torch.cat([context_ids, prompt_ids], dim=1).to(device)
            query_offset = int(context_ids.shape[1])
        else:
            if query_vector_mode == "question_content":
                question_ids, local_positions = content_query_positions(
                    tokenizer,
                    question,
                    content_start=0,
                    content_end=len(question),
                    requested=query_vector_tokens,
                )
            else:
                question_ids = tokenizer(question, add_special_tokens=False, return_tensors="pt")[
                    "input_ids"
                ]
                local_positions = query_positions(int(question_ids.shape[1]), query_vector_tokens)
            token_ids = question_ids.to(device)
            query_offset = 0
        if token_ids.shape[1] == 0:
            raise RuntimeError(f"Query {query_index} tokenized to an empty sequence")
        run_base_model(model, capture, token_ids)
        q_stacked, _ = captured_qk(model, capture, pair_specs, profile_space)
        q_all = q_stacked[0]
        positions = [query_offset + item for item in local_positions]
        chosen = q_all.index_select(0, torch.tensor(positions, device=device, dtype=torch.long)).float()
        chosen_projected = torch.einsum("spd,pdr->spr", chosen, basis_device)
        count = chosen.shape[0]
        raw[query_index, :count] = chosen.to(torch.float16).cpu()
        projected[query_index, :count] = chosen_projected.to(torch.float16).cpu()
        mask[query_index, :count] = True
        token_positions.append(positions)
    torch.save(
        {
            "raw_q": raw,
            "svd_q": projected,
            "mask": mask,
            "token_positions": token_positions,
            "pair_specs": pair_specs,
            "queries": queries,
            "profile_space": profile_space,
            "query_vector_mode": query_vector_mode,
        },
        output_path,
    )


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = setup_distributed()
    pair_tuples = parse_pairs(args.pairs)
    corpus_dir = Path(args.corpus_dir)
    profile_dir = Path(args.profile_dir)
    if rank == 0:
        profile_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)

    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    records = read_jsonl(corpus_dir / "records.jsonl")
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    block_count, block_tokens = (int(blocks.shape[0]), int(blocks.shape[1]))
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
    repeat_groups = num_query_heads // num_kv_heads
    head_dim = int(getattr(config, "head_dim", config.hidden_size // num_query_heads))
    if args.svd_rank <= 0 or args.svd_rank > head_dim:
        raise ValueError(f"svd_rank must be in [1, {head_dim}]")

    pair_specs: list[dict[str, int]] = []
    for layer, query_head in pair_tuples:
        if not 0 <= layer < num_layers:
            raise ValueError(f"Layer {layer} is outside [0, {num_layers})")
        if not 0 <= query_head < num_query_heads:
            raise ValueError(f"Query head {query_head} is outside [0, {num_query_heads})")
        pair_specs.append(
            {
                "layer": layer,
                "query_head": query_head,
                "kv_head": query_head // repeat_groups,
            }
        )

    capture = QKCapture(model, sorted({pair["layer"] for pair in pair_specs}))
    basis_path = profile_dir / "basis.pt"
    if rank == 0:
        basis_payload = fit_basis(
            model,
            capture,
            blocks,
            records,
            pair_specs,
            args.calibration_blocks,
            args.svd_rank,
            device,
            args.profile_space,
        )
        basis_payload.update(
            {
                "pair_specs": pair_specs,
                "head_dim": head_dim,
                "svd_rank": args.svd_rank,
                "profile_space": args.profile_space,
                "contains_synthetic_vectors": False,
            }
        )
        torch.save(basis_payload, basis_path)
        if not args.skip_query_profiles:
            profile_queries(
                model=model,
                tokenizer=AutoTokenizer.from_pretrained(
                    args.model_name_or_path, use_fast=True
                ),
                capture=capture,
                blocks=blocks,
                queries=queries,
                pair_specs=pair_specs,
                basis=basis_payload["basis"],
                query_vector_tokens=args.query_vector_tokens,
                device=device,
                output_path=profile_dir / "query_profiles.pt",
                profile_space=args.profile_space,
                query_vector_mode=args.query_vector_mode,
            )
        print(
            json.dumps(
                {
                    "basis_path": str(basis_path),
                    "pair_specs": pair_specs,
                    "retained_energy": basis_payload["retained_energy"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    barrier(world_size)
    basis_payload = torch.load(basis_path, map_location="cpu", weights_only=False)
    basis = basis_payload["basis"].to(device=device, dtype=dtype)

    block_start, block_end = shard_bounds(block_count, rank, world_size)
    local_blocks = block_end - block_start
    profile_count = len(pair_specs)
    raw_path = profile_dir / f"raw_k_rank{rank:03d}.npy"
    svd_path = profile_dir / f"svd_k_rank{rank:03d}.npy"
    raw_out = np.lib.format.open_memmap(
        raw_path,
        mode="w+",
        dtype=np.float16,
        shape=(local_blocks, block_tokens, profile_count, head_dim),
    )
    svd_out = np.lib.format.open_memmap(
        svd_path,
        mode="w+",
        dtype=np.float16,
        shape=(local_blocks, block_tokens, profile_count, args.svd_rank),
    )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if args.profile_space == "pre_rope_block_qk":
        batches = math.ceil(local_blocks / args.batch_blocks)
        for batch_index, local_offset in enumerate(range(0, local_blocks, args.batch_blocks)):
            count = min(args.batch_blocks, local_blocks - local_offset)
            global_start = block_start + local_offset
            input_array = np.asarray(blocks[global_start : global_start + count], dtype=np.int64)
            input_ids = torch.from_numpy(input_array).to(device=device, non_blocking=True)
            run_base_model(model, capture, input_ids)
            _, keys = captured_qk(model, capture, pair_specs, args.profile_space)
            projected = torch.einsum("btpd,pdr->btpr", keys, basis)
            raw_out[local_offset : local_offset + count] = keys.to(torch.float16).cpu().numpy()
            svd_out[local_offset : local_offset + count] = projected.to(torch.float16).cpu().numpy()
            if (batch_index + 1) % args.log_every == 0 or batch_index + 1 == batches:
                elapsed = time.perf_counter() - started
                done_tokens = (local_offset + count) * block_tokens
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "batch": batch_index + 1,
                            "batches": batches,
                            "tokens": done_tokens,
                            "tokens_per_second": done_tokens / max(elapsed, 1.0e-9),
                        }
                    ),
                    flush=True,
                )
    else:
        overlapping_records = [
            record
            for record in records
            if int(record["block_start"]) < block_end
            and int(record["block_start"]) + int(record["block_count"]) > block_start
        ]
        written_blocks = 0
        for record_index, record in enumerate(overlapping_records):
            record_start = int(record["block_start"])
            record_blocks = int(record["block_count"])
            record_end = record_start + record_blocks
            input_array = np.asarray(
                blocks[record_start:record_end], dtype=np.int64
            ).reshape(1, record_blocks * block_tokens)
            input_ids = torch.from_numpy(input_array).to(device=device, non_blocking=True)
            run_base_model(model, capture, input_ids)
            _, record_keys = captured_qk(model, capture, pair_specs, args.profile_space)
            record_keys = record_keys.reshape(
                record_blocks, block_tokens, profile_count, head_dim
            )

            overlap_start = max(record_start, block_start)
            overlap_end = min(record_end, block_end)
            count = overlap_end - overlap_start
            record_offset = overlap_start - record_start
            local_offset = overlap_start - block_start
            selected_keys = record_keys[record_offset : record_offset + count]
            projected = torch.einsum("btpd,pdr->btpr", selected_keys, basis)
            raw_out[local_offset : local_offset + count] = selected_keys.to(torch.float16).cpu().numpy()
            svd_out[local_offset : local_offset + count] = projected.to(torch.float16).cpu().numpy()
            written_blocks += count
            if (record_index + 1) % args.log_every == 0 or record_index + 1 == len(overlapping_records):
                elapsed = time.perf_counter() - started
                done_tokens = written_blocks * block_tokens
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "record": record_index + 1,
                            "records": len(overlapping_records),
                            "tokens": done_tokens,
                            "tokens_per_second": done_tokens / max(elapsed, 1.0e-9),
                        }
                    ),
                    flush=True,
                )
        if written_blocks != local_blocks:
            raise RuntimeError(
                f"Rank {rank} wrote {written_blocks}/{local_blocks} record-profile blocks"
            )
    raw_out.flush()
    svd_out.flush()
    elapsed = time.perf_counter() - started
    shard_summary = {
        "rank": rank,
        "world_size": world_size,
        "block_start": block_start,
        "block_end": block_end,
        "local_blocks": local_blocks,
        "local_tokens": local_blocks * block_tokens,
        "raw_k_path": str(raw_path),
        "svd_k_path": str(svd_path),
        "elapsed_seconds": elapsed,
        "tokens_per_second": local_blocks * block_tokens / max(elapsed, 1.0e-9),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    (profile_dir / f"shard_rank{rank:03d}.json").write_text(
        json.dumps(shard_summary, indent=2), encoding="utf-8"
    )
    barrier(world_size)

    if rank == 0:
        shards = [
            json.loads((profile_dir / f"shard_rank{item:03d}.json").read_text(encoding="utf-8"))
            for item in range(world_size)
        ]
        summary = {
            "source": "real Qwen3 forward pass over real LongBench records",
            "model_name_or_path": args.model_name_or_path,
            "profile_space": args.profile_space,
            "profile_space_reason": (
                "actual post-RoPE Q/K from full causal record prefill, then partitioned into blocks"
                if args.profile_space == "post_rope_record_qk"
                else (
                    "position-independent pre-RoPE Q/K from full causal record prefill, then "
                    "partitioned into blocks"
                    if args.profile_space == "pre_rope_record_qk"
                    else "position-independent pre-RoPE Q/K from independently profiled blocks"
                )
            ),
            "contains_synthetic_vectors": False,
            "num_blocks": block_count,
            "block_tokens": block_tokens,
            "num_tokens": block_count * block_tokens,
            "num_queries": len(queries),
            "pair_specs": pair_specs,
            "head_dim": head_dim,
            "svd_rank": args.svd_rank,
            "calibration_blocks": basis_payload["calibration_blocks"],
            "calibration_tokens": basis_payload["calibration_tokens"],
            "retained_energy": basis_payload["retained_energy"],
            "query_vector_tokens": args.query_vector_tokens,
            "query_vector_mode": (
                args.query_vector_mode if not args.skip_query_profiles else None
            ),
            "query_profiles_built": not args.skip_query_profiles,
            "profile_world_size": world_size,
            "dtype": args.dtype,
            "shards": shards,
            "basis_path": str(basis_path),
            "query_profiles_path": (
                str(profile_dir / "query_profiles.pt")
                if not args.skip_query_profiles
                else None
            ),
        }
        (profile_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    capture.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
