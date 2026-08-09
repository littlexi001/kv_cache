from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from profile_all_head_qk import AllHeadCapture, run_base_model
from profile_real_qk import resolve_dtype, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace token-identity-sensitive native generation Q with a fixed retrieval "
            "probe conditioned on each generated reasoning state."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--base_profile_dir", required=True)
    parser.add_argument("--trajectory_profile", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--probe_tokens", type=int, default=4)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    return parser.parse_args()


def build_probe_text(
    *,
    question: str,
    step_question: str,
    generated_state: str,
    evidence: str | None,
) -> str:
    sections = [
        f"Original question: {question}",
        f"Current atomic question: {step_question}",
    ]
    if evidence is not None:
        sections.append(f"Previously read evidence:\n{evidence}")
    sections.extend(
        [
            f"Current reasoning state: {generated_state or '(empty)'}",
            "Retrieve evidence needed for the next reasoning step:",
        ]
    )
    return "\n\n".join(sections)


def project_probe_q(
    capture: AllHeadCapture,
    *,
    layers: list[int],
    basis: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    repeat_groups = num_query_heads // num_kv_heads
    kv_map = torch.arange(num_query_heads, device=basis.device) // repeat_groups
    output = []
    for layer_index, layer in enumerate(layers):
        query = capture.q[layer][0].float()
        query_basis = basis[layer_index].index_select(0, kv_map)
        projected = torch.einsum("thd,hdr->thr", query, query_basis)
        projected = F.normalize(projected, dim=-1).mean(dim=0)
        output.append(F.normalize(projected, dim=-1).to(dtype=torch.float16, device="cpu"))
    return torch.stack(output)


@torch.inference_mode()
def profile_probe(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    capture: AllHeadCapture,
    text: str,
    layers: list[int],
    basis: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    probe_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(
        device
    )
    count = min(max(1, probe_tokens), int(input_ids.shape[1]))
    positions = torch.arange(input_ids.shape[1] - count, input_ids.shape[1], device=device)
    capture.configure(capture_q=True, capture_k=False, q_positions=positions)
    run_base_model(model, capture, input_ids)
    return project_probe_q(
        capture,
        layers=layers,
        basis=basis,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
    )


def merge_shards(
    *,
    source: dict[str, Any],
    shard_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    state_only = torch.zeros_like(source["svd_q"])
    evidence = torch.zeros_like(source["svd_q"])
    seen = set()
    for path in shard_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for local_index, trajectory_index in enumerate(payload["trajectory_indices"]):
            trajectory_index = int(trajectory_index)
            if trajectory_index in seen:
                raise RuntimeError(f"duplicate trajectory {trajectory_index}")
            seen.add(trajectory_index)
            count = int(source["mask"][trajectory_index].sum().item())
            state_only[trajectory_index, :count].copy_(
                payload["state_only_probe_q"][local_index, :count]
            )
            evidence[trajectory_index, :count].copy_(
                payload["evidence_probe_q"][local_index, :count]
            )
    if seen != set(range(len(source["trajectories"]))):
        raise RuntimeError("probe shards are incomplete")
    output = dict(source)
    output["native_generation_q"] = source["svd_q"]
    output["state_only_probe_q"] = state_only
    output["evidence_probe_q"] = evidence
    output["probe_profile_space"] = "fixed_suffix_pre_rope_q_projected_to_frozen_k_svd"
    torch.save(output, output_path)
    return {
        "trajectories": len(source["trajectories"]),
        "states": int(source["mask"].sum().item()),
        "state_only_finite": bool(torch.isfinite(state_only[source["mask"]]).all()),
        "evidence_finite": bool(torch.isfinite(evidence[source["mask"]]).all()),
        "state_only_norm_mean": float(
            state_only[source["mask"]].float().norm(dim=-1).mean().item()
        ),
        "evidence_norm_mean": float(
            evidence[source["mask"]].float().norm(dim=-1).mean().item()
        ),
    }


def main() -> None:
    args = parse_args()
    rank, world_size, _local_rank, device = setup_distributed()
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    source = torch.load(args.trajectory_profile, map_location="cpu", weights_only=False)
    trajectories = source["trajectories"]
    blocks = np.load(Path(args.corpus_dir) / "blocks.npy", mmap_mode="r")
    basis_payload = torch.load(
        Path(args.base_profile_dir) / "basis.pt", map_location="cpu", weights_only=False
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    capture = AllHeadCapture(model, layers)
    basis = basis_payload["basis"].to(device=device, dtype=torch.float32)
    num_query_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    trajectory_indices = list(range(rank, len(trajectories), world_size))
    max_states = int(source["svd_q"].shape[1])
    shape = (len(trajectory_indices), max_states, len(layers), num_query_heads, int(basis.shape[-1]))
    state_only = torch.zeros(shape, dtype=torch.float16)
    evidence_q = torch.zeros(shape, dtype=torch.float16)
    started = time.perf_counter()
    for local_index, trajectory_index in enumerate(trajectory_indices):
        trajectory = trajectories[trajectory_index]
        evidence = "\n".join(
            tokenizer.decode(blocks[int(block_id)].tolist(), skip_special_tokens=True)
            for block_id in trajectory["hop1_gold_block_ids"]
        )
        count = int(source["mask"][trajectory_index].sum().item())
        for state_index in range(count):
            generated_state = str(trajectory["state_metadata"][state_index]["generated_text"])
            common = {
                "question": str(trajectory["question"]),
                "step_question": str(trajectory["step_question"]),
                "generated_state": generated_state,
            }
            state_only[local_index, state_index].copy_(
                profile_probe(
                    model=model,
                    tokenizer=tokenizer,
                    capture=capture,
                    text=build_probe_text(**common, evidence=None),
                    layers=layers,
                    basis=basis,
                    num_query_heads=num_query_heads,
                    num_kv_heads=num_kv_heads,
                    probe_tokens=args.probe_tokens,
                    device=device,
                )
            )
            evidence_q[local_index, state_index].copy_(
                profile_probe(
                    model=model,
                    tokenizer=tokenizer,
                    capture=capture,
                    text=build_probe_text(**common, evidence=evidence),
                    layers=layers,
                    basis=basis,
                    num_query_heads=num_query_heads,
                    num_kv_heads=num_kv_heads,
                    probe_tokens=args.probe_tokens,
                    device=device,
                )
            )
        print(
            json.dumps(
                {
                    "rank": rank,
                    "trajectory": local_index + 1,
                    "local_trajectories": len(trajectory_indices),
                    "query_id": int(trajectory["query_id"]),
                    "states": count,
                }
            ),
            flush=True,
        )
    capture.close()
    shard_path = output_dir / f"probe_q_rank{rank:03d}.pt"
    torch.save(
        {
            "trajectory_indices": trajectory_indices,
            "state_only_probe_q": state_only,
            "evidence_probe_q": evidence_q,
        },
        shard_path,
    )
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        output_path = output_dir / "trajectory_probe_q_profiles.pt"
        summary = merge_shards(
            source=source,
            shard_paths=[output_dir / f"probe_q_rank{item:03d}.pt" for item in range(world_size)],
            output_path=output_path,
        )
        summary.update(
            {
                "source": "fixed retrieval probe over real generated reasoning states",
                "contains_synthetic_vectors": False,
                "trajectory_profile": args.trajectory_profile,
                "base_profile_dir": args.base_profile_dir,
                "probe_tokens": args.probe_tokens,
                "output_path": str(output_path),
                "world_size": world_size,
                "wall_seconds": time.perf_counter() - started,
            }
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
