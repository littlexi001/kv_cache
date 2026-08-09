from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from generate_head_distortion_teacher import read_jsonl, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile frozen per-layer/KV-head post-RoPE K PCA bases."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_queries", type=int, default=64)
    parser.add_argument("--max_context_tokens", type=int, default=4096)
    parser.add_argument("--samples_per_query", type=int, default=128)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def basis_from_moments(
    sums: torch.Tensor, grams: torch.Tensor, count: int, target_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    means = sums / max(count, 1)
    covariance = grams / max(count, 1) - torch.einsum(
        "...d,...e->...de", means, means
    )
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    rank = min(target_rank, covariance.shape[-1])
    basis = eigenvectors[..., -rank:]
    retained = eigenvalues[..., -rank:].clamp_min(0).sum(dim=-1) / eigenvalues.clamp_min(
        0
    ).sum(dim=-1).clamp_min(1.0e-12)
    return basis, retained


class MomentCapture:
    def __init__(
        self,
        model: Any,
        samples_per_query: int,
        device: torch.device,
    ) -> None:
        self.samples_per_query = samples_per_query
        self.device = device
        layers = len(model.model.layers)
        kv_heads = model.config.num_key_value_heads
        head_dim = model.model.layers[0].self_attn.head_dim
        self.sums = torch.zeros(
            (layers, kv_heads, head_dim), dtype=torch.float32, device=device
        )
        self.grams = torch.zeros(
            (layers, kv_heads, head_dim, head_dim), dtype=torch.float32, device=device
        )
        self.counts = torch.zeros(layers, dtype=torch.long, device=device)
        self.handles = []
        for layer, decoder_layer in enumerate(model.model.layers):
            self.handles.append(
                decoder_layer.self_attn.register_forward_pre_hook(
                    self._hook(layer), with_kwargs=True
                )
            )

    def _hook(self, layer: int):
        def capture(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get("position_embeddings")
            if hidden_states is None or position_embeddings is None:
                raise ValueError("Qwen attention hook did not receive required inputs")
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            keys = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            _, keys = apply_rotary_pos_emb(keys, keys, *position_embeddings)
            length = keys.shape[2]
            sample_count = min(self.samples_per_query, length)
            positions = torch.linspace(
                0, length - 1, steps=sample_count, device=keys.device
            ).round().long().unique()
            sample = keys[0, :, positions].float()
            self.sums[layer] += sample.sum(dim=1)
            self.grams[layer] += torch.einsum("ksd,kse->kde", sample, sample)
            self.counts[layer] += sample.shape[1]

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("post-RoPE basis profiling requires CUDA")
    rank, world_size, _, device = setup_distributed()
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    if args.max_queries > 0:
        queries = queries[: args.max_queries]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    capture = MomentCapture(model, args.samples_per_query, device)
    local_queries = 0
    try:
        for query_index in range(rank, len(queries), world_size):
            query = queries[query_index]
            context = np.asarray(
                blocks[
                    int(query["block_start"]) : int(query["block_start"])
                    + int(query["block_count"])
                ],
                dtype=np.int64,
            ).reshape(-1)
            context = context[: args.max_context_tokens]
            input_ids = torch.from_numpy(np.array(context, copy=True)).long()[None].to(device)
            with torch.inference_mode():
                model.model(input_ids=input_ids, use_cache=False, return_dict=True)
            local_queries += 1
    finally:
        capture.close()

    if world_size > 1:
        dist.all_reduce(capture.sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(capture.grams, op=dist.ReduceOp.SUM)
        dist.all_reduce(capture.counts, op=dist.ReduceOp.SUM)
    if not torch.all(capture.counts == capture.counts[0]):
        raise ValueError("layers accumulated different sample counts")

    if rank == 0:
        basis, retained = basis_from_moments(
            capture.sums,
            capture.grams,
            int(capture.counts[0].item()),
            args.rank,
        )
        payload = {
            "basis": basis.float().cpu(),
            "retained_energy": retained.float().cpu(),
            "metadata": {
                "model_name_or_path": args.model_name_or_path,
                "profile_space": "post_rope_k",
                "queries": len(queries),
                "context_tokens": args.max_context_tokens,
                "samples_per_query": args.samples_per_query,
                "samples_per_layer": int(capture.counts[0].item()),
                "rank": basis.shape[-1],
                "world_size": world_size,
                "mean_retained_energy": float(retained.mean().item()),
                "min_retained_energy": float(retained.min().item()),
                "max_retained_energy": float(retained.max().item()),
                "record_uids": [str(query.get("record_uid", "")) for query in queries],
            },
        }
        torch.save(payload, output_dir / "postrope_k_basis.pt")
        (output_dir / "summary.json").write_text(
            json.dumps(payload["metadata"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
