from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_stepwise_set_utility import build_span_query
from profile_all_head_qk import AllHeadCapture, run_base_model
from profile_real_qk import (
    barrier,
    content_query_positions,
    read_jsonl,
    resolve_dtype,
    setup_distributed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile leakage-free pre-RoPE Q for explicit reasoning-step states."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--base_profile_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", default="train,dev,test")
    parser.add_argument("--task_types", default="multihop")
    parser.add_argument("--exclude_query_ids", default="")
    parser.add_argument("--query_vector_tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    return parser.parse_args()


def step_state_text(step: dict[str, Any]) -> str:
    return build_span_query(
        step,
        [str(item) for item in step["compact_state_before"]],
    )


@torch.inference_mode()
def profile_local_steps(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    capture: AllHeadCapture,
    steps: list[dict[str, Any]],
    step_indices: list[int],
    layers: list[int],
    basis: torch.Tensor,
    query_vector_tokens: int,
    num_query_heads: int,
    num_kv_heads: int,
    device: torch.device,
    output_path: Path,
) -> None:
    repeat_groups = num_query_heads // num_kv_heads
    kv_map = torch.arange(num_query_heads, device=device) // repeat_groups
    basis_device = basis.to(device=device, dtype=next(model.parameters()).dtype)
    projected = torch.zeros(
        len(step_indices),
        query_vector_tokens,
        len(layers),
        num_query_heads,
        int(basis.shape[-1]),
        dtype=torch.float16,
    )
    mask = torch.zeros(len(step_indices), query_vector_tokens, dtype=torch.bool)
    token_positions = []
    selected_steps = []
    for local_index, step_index in enumerate(step_indices):
        step = steps[step_index]
        selected_steps.append(step)
        state = step_state_text(step)
        prompt = f"\nCurrent reasoning state: {state}\nRetrieve evidence for the next step:"
        content_start = len("\nCurrent reasoning state: ")
        prompt_ids, positions = content_query_positions(
            tokenizer,
            prompt,
            content_start=content_start,
            content_end=content_start + len(state),
            requested=query_vector_tokens,
        )
        position_tensor = torch.tensor(positions, device=device, dtype=torch.long)
        capture.configure(capture_q=True, capture_k=False, q_positions=position_tensor)
        run_base_model(model, capture, prompt_ids.to(device))
        for layer_index, layer in enumerate(layers):
            query_values = capture.q[layer][0]
            query_basis = basis_device[layer_index].index_select(0, kv_map)
            query_projected = torch.einsum("thd,hdr->thr", query_values, query_basis)
            query_projected = F.normalize(query_projected.float(), dim=-1)
            projected[local_index, : len(positions), layer_index].copy_(
                query_projected.to(dtype=torch.float16, device="cpu")
            )
        mask[local_index, : len(positions)] = True
        token_positions.append(positions)
        print(
            json.dumps(
                {
                    "rank": dist.get_rank() if dist.is_initialized() else 0,
                    "query_id": int(step["query_id"]),
                    "step_index": int(step["step_index"]),
                    "local_step": local_index + 1,
                    "local_steps": len(step_indices),
                }
            ),
            flush=True,
        )
    torch.save(
        {
            "svd_q": projected,
            "mask": mask,
            "token_positions": token_positions,
            "step_indices": step_indices,
            "steps": selected_steps,
            "layers": layers,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "profile_space": "pre_rope_step_state_q",
            "query_vector_mode": "last_state_content_tokens_without_source",
            "normalized": True,
        },
        output_path,
    )


def merge_shards(shard_paths: list[Path], steps: list[dict[str, Any]], output_path: Path) -> None:
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in shard_paths]
    sample = payloads[0]
    output_shape = (len(steps), *sample["svd_q"].shape[1:])
    projected = torch.zeros(output_shape, dtype=sample["svd_q"].dtype)
    mask = torch.zeros(len(steps), sample["mask"].shape[1], dtype=torch.bool)
    token_positions: list[list[int] | None] = [None for _ in steps]
    seen = set()
    for payload in payloads:
        for local_index, step_index in enumerate(payload["step_indices"]):
            step_index = int(step_index)
            if step_index in seen:
                raise RuntimeError(f"duplicate step profile: {step_index}")
            seen.add(step_index)
            projected[step_index].copy_(payload["svd_q"][local_index])
            mask[step_index].copy_(payload["mask"][local_index])
            token_positions[step_index] = payload["token_positions"][local_index]
    if seen != set(range(len(steps))):
        raise RuntimeError("step profile shards are incomplete")
    torch.save(
        {
            "svd_q": projected,
            "mask": mask,
            "token_positions": token_positions,
            "step_indices": list(range(len(steps))),
            "steps": steps,
            "layers": sample["layers"],
            "num_query_heads": sample["num_query_heads"],
            "num_kv_heads": sample["num_kv_heads"],
            "profile_space": sample["profile_space"],
            "query_vector_mode": sample["query_vector_mode"],
            "normalized": True,
        },
        output_path,
    )


def main() -> None:
    args = parse_args()
    rank, world_size, _local_rank, device = setup_distributed()
    base_profile_dir = Path(args.base_profile_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)
    allowed_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    allowed_tasks = {item.strip() for item in args.task_types.split(",") if item.strip()}
    excluded_ids = {
        int(item.strip()) for item in args.exclude_query_ids.split(",") if item.strip()
    }
    steps = [
        row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) in allowed_splits
        and str(row["task_type"]) in allowed_tasks
        and int(row["query_id"]) not in excluded_ids
    ]
    basis_payload = torch.load(
        base_profile_dir / "basis.pt", map_location="cpu", weights_only=False
    )
    layers = [int(item) for item in basis_payload["layers"]]
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
    capture = AllHeadCapture(model, layers)
    shard_path = output_dir / f"step_q_rank{rank:03d}.pt"
    profile_started = time.perf_counter()
    profile_local_steps(
        model=model,
        tokenizer=AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True),
        capture=capture,
        steps=steps,
        step_indices=list(range(rank, len(steps), world_size)),
        layers=layers,
        basis=basis_payload["basis"],
        query_vector_tokens=args.query_vector_tokens,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        device=device,
        output_path=shard_path,
    )
    local_profile_seconds = time.perf_counter() - profile_started
    elapsed_tensor = torch.tensor(local_profile_seconds, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    profile_wall_seconds = float(elapsed_tensor.item())
    capture.close()
    barrier(world_size)
    output_path = output_dir / "step_query_profiles.pt"
    if rank == 0:
        merge_shards(
            [output_dir / f"step_q_rank{item:03d}.pt" for item in range(world_size)],
            steps,
            output_path,
        )
        summary = {
            "source": "leakage-free step state pre-RoPE Q projected into frozen K-SVD basis",
            "contains_source_context": False,
            "contains_synthetic_vectors": False,
            "base_profile_dir": str(base_profile_dir),
            "step_queries_path": str(args.step_queries_path),
            "step_profiles_path": str(output_path),
            "steps": len(steps),
            "splits": sorted(allowed_splits),
            "task_types": sorted(allowed_tasks),
            "excluded_query_ids": sorted(excluded_ids),
            "layers": layers,
            "query_vector_tokens": args.query_vector_tokens,
            "world_size": world_size,
            "profile_wall_seconds": profile_wall_seconds,
            "steps_per_second": len(steps) / max(profile_wall_seconds, 1.0e-9),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    barrier(world_size)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
