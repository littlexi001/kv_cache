from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from evaluate_past_only_10m_text_retrieval import scope_metrics
from evaluate_xsum_10m_dynamic_text_retrieval import reciprocal_rank_fusion, top_indices
from profile_real_qk import QKCapture, captured_qk, resolve_dtype, run_base_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pre-RoPE block K-mean and token-max retrieval on past-only memory."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--state_suffix_tokens", default="64,128,256,512")
    parser.add_argument("--topks", default="8,64,512")
    parser.add_argument("--ranking_depth", type=int, default=512)
    parser.add_argument("--query_q_tokens", type=int, default=8)
    parser.add_argument("--mean_chunk_blocks", type=int, default=8192)
    parser.add_argument("--token_chunk_blocks", type=int, default=1024)
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
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
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values:
        raise ValueError("integer list cannot be empty")
    return values


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def load_sharded_tensor(
    summary: dict[str, Any], *, path_key: str, device: torch.device
) -> torch.Tensor:
    first = np.load(summary["shards"][0][path_key], mmap_mode="r")
    shape = (int(summary["base_blocks"]),) + tuple(first.shape[1:])
    output = torch.empty(shape, dtype=torch.float16, device=device)
    for shard in summary["shards"]:
        array = np.load(shard[path_key], mmap_mode="r")
        start = int(shard["block_start"])
        end = int(shard["block_end"])
        output[start:end].copy_(torch.from_numpy(np.array(array, copy=True)).to(device))
    return output


@torch.inference_mode()
def block_mean_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    *,
    chunk_blocks: int,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(keys), chunk_blocks):
        chunk = keys[start : start + chunk_blocks]
        similarity = torch.einsum("qpd,bpd->qbp", query.float(), chunk.float())
        parts.append(similarity.amax(dim=0).transpose(0, 1).cpu())
    return torch.cat(parts, dim=1)


@torch.inference_mode()
def token_max_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    *,
    chunk_blocks: int,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(keys), chunk_blocks):
        chunk = F.normalize(keys[start : start + chunk_blocks].float(), dim=-1)
        similarity = torch.einsum("qpd,btpd->qbtp", query.float(), chunk)
        parts.append(similarity.amax(dim=(0, 2)).transpose(0, 1).cpu())
    return torch.cat(parts, dim=1)


def aggregate_rankings(
    per_profile: np.ndarray, *, depth: int, rrf_k: float
) -> tuple[dict[str, list[int]], list[list[int]]]:
    profile_rankings = [top_indices(scores, depth) for scores in per_profile]
    return (
        {
            "max": top_indices(per_profile.max(axis=0), depth),
            "profile_rrf": reciprocal_rank_fusion(
                profile_rankings, depth=depth, rrf_k=rrf_k
            ),
        },
        profile_rankings,
    )


def summarize(rows: list[dict[str, Any]], topks: list[int]) -> list[dict[str, Any]]:
    output = []
    for suffix in sorted({int(row["prefix_tokens"]) for row in rows}):
        for method in sorted({str(row["method"]) for row in rows}):
            group = [
                row
                for row in rows
                if int(row["prefix_tokens"]) == suffix and row["method"] == method
            ]
            item: dict[str, Any] = {
                "state_suffix_tokens": suffix,
                "method": method,
                "queries": len(group),
                "mean_capture_seconds": mean(float(row["query_capture_seconds"]) for row in group),
                "mean_score_seconds": mean(float(row["score_seconds"]) for row in group),
            }
            for topk in topks:
                for metric in (
                    "same_scope_any",
                    "same_scope_fraction",
                    "same_scope_within_4k_any",
                    "same_scope_within_16k_any",
                ):
                    key = f"{metric}_at_{topk}"
                    item[key] = mean(float(row[key]) for row in group)
            output.append(item)
    return output


def profile_summary(
    rows: list[dict[str, Any]], specs: list[dict[str, int]]
) -> list[dict[str, Any]]:
    output = []
    for family in sorted({str(row["family"]) for row in rows}):
        for suffix in sorted({int(row["prefix_tokens"]) for row in rows}):
            group = [
                row
                for row in rows
                if row["family"] == family and int(row["prefix_tokens"]) == suffix
            ]
            for profile_id in sorted({int(row["profile_id"]) for row in group}):
                profile_group = [row for row in group if int(row["profile_id"]) == profile_id]
                spec = specs[profile_id]
                output.append(
                    {
                        "family": family,
                        "state_suffix_tokens": suffix,
                        "profile_id": profile_id,
                        "layer": int(spec["layer"]),
                        "kv_head": int(spec["kv_head"]),
                        "queries": len(profile_group),
                        "same_scope_any_at_8": mean(
                            float(row["same_scope_any_at_8"]) for row in profile_group
                        ),
                        "same_scope_fraction_at_8": mean(
                            float(row["same_scope_fraction_at_8"]) for row in profile_group
                        ),
                    }
                )
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    data_dir = Path(args.data_dir)
    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    profile = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    metadata = {
        int(row["query_id"]): row for row in read_jsonl(data_dir / "metadata.jsonl")
    }
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    block_scope_ids = np.load(data_dir / "base_block_scope_ids.npy", mmap_mode="r")
    block_original_centers = np.load(
        data_dir / "base_block_original_centers.npy", mmap_mode="r"
    )
    suffixes = parse_ints(args.state_suffix_tokens)
    topks = parse_ints(args.topks)
    if max(topks) > args.ranking_depth:
        raise ValueError("ranking depth must cover requested top-k")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    specs = profile["pair_specs"]
    capture = QKCapture(model, sorted({int(item["layer"]) for item in specs}))
    basis_payload = torch.load(profile_dir / "basis.pt", map_location="cpu", weights_only=False)
    basis = basis_payload["basis"].to(device=device, dtype=resolve_dtype(args.dtype))
    token_profile_indices = [int(item) for item in profile["token_profile_indices"]]
    token_indices = torch.tensor(token_profile_indices, dtype=torch.long, device=device)

    torch.cuda.synchronize(device)
    load_started = time.perf_counter()
    mean_raw = load_sharded_tensor(profile, path_key="mean_path", device=device)
    mean_normalized = F.normalize(mean_raw.float(), dim=-1).to(torch.float16)
    token_raw = load_sharded_tensor(profile, path_key="token_path", device=device)
    coherence = load_sharded_tensor(profile, path_key="coherence_path", device=device)
    torch.cuda.synchronize(device)
    index_load_seconds = time.perf_counter() - load_started

    local_query_ids = [query_id for query_id in range(len(queries)) if query_id % world_size == rank]
    rows = []
    diagnostic_rows = []
    for query_id in local_query_ids:
        query_scope = int(metadata[query_id]["book_index"])
        local_start = int(metadata[query_id]["local_context_start_token"])
        for suffix in suffixes:
            input_ids = torch.from_numpy(
                np.asarray(queries[query_id, -suffix:], dtype=np.int64)
            )[None, :].to(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            run_base_model(model, capture, input_ids)
            query_raw, _ = captured_qk(model, capture, specs, "pre_rope_block_qk")
            keep = min(args.query_q_tokens, int(query_raw.shape[1]))
            query_projected = torch.einsum(
                "qpd,pdr->qpr", query_raw[0, -keep:], basis
            )
            query_normalized = F.normalize(query_projected.float(), dim=-1)
            query_sample = query_normalized.index_select(1, token_indices)
            torch.cuda.synchronize(device)
            capture_seconds = time.perf_counter() - started

            torch.cuda.synchronize(device)
            started = time.perf_counter()
            mean56 = block_mean_scores(
                query_normalized, mean_normalized, chunk_blocks=args.mean_chunk_blocks
            ).numpy()
            mean8 = mean56[np.asarray(token_profile_indices, dtype=np.int64)]
            token8 = token_max_scores(
                query_sample, token_raw, chunk_blocks=args.token_chunk_blocks
            ).numpy()
            torch.cuda.synchronize(device)
            score_seconds = time.perf_counter() - started

            families = {
                "qk_mean56_cosine": (mean56, list(range(len(specs)))),
                "qk_mean8_cosine": (mean8, token_profile_indices),
                "qk_token8_cosine": (token8, token_profile_indices),
            }
            for family, (per_profile, profile_ids) in families.items():
                aggregated, individual = aggregate_rankings(
                    per_profile, depth=args.ranking_depth, rrf_k=args.rrf_k
                )
                for aggregation, ranking in aggregated.items():
                    rows.append(
                        {
                            "query_id": query_id,
                            "memory_tokens": int(data_summary["memory_tokens"]),
                            "memory_blocks": len(block_scope_ids),
                            "prefix_tokens": suffix,
                            "method": f"{family}_{aggregation}",
                            "query_capture_seconds": capture_seconds,
                            "score_seconds": score_seconds,
                            "top_block_ids": ranking[: max(topks)],
                            "selection_uses_target": False,
                            **scope_metrics(
                                ranking,
                                query_scope=query_scope,
                                local_start=local_start,
                                block_scope_ids=block_scope_ids,
                                block_original_centers=block_original_centers,
                                topks=topks,
                            ),
                        }
                    )
                for local_profile, ranking in enumerate(individual):
                    metrics = scope_metrics(
                        ranking,
                        query_scope=query_scope,
                        local_start=local_start,
                        block_scope_ids=block_scope_ids,
                        block_original_centers=block_original_centers,
                        topks=[8],
                    )
                    diagnostic_rows.append(
                        {
                            "query_id": query_id,
                            "prefix_tokens": suffix,
                            "family": family,
                            "profile_id": int(profile_ids[local_profile]),
                            **metrics,
                        }
                    )

    for name, data in (("rows", rows), ("profile_rows", diagnostic_rows)):
        with (output_dir / f"{name}_rank{rank:03d}.jsonl").open("w", encoding="utf-8") as handle:
            for row in data:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    runtime = {
        "rank": rank,
        "queries": len(local_query_ids),
        "index_load_seconds": index_load_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    (output_dir / f"runtime_rank{rank:03d}.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )
    barrier(world_size)
    if rank == 0:
        all_rows = [
            row
            for shard in range(world_size)
            for row in read_jsonl(output_dir / f"rows_rank{shard:03d}.jsonl")
        ]
        all_profiles = [
            row
            for shard in range(world_size)
            for row in read_jsonl(output_dir / f"profile_rows_rank{shard:03d}.jsonl")
        ]
        all_rows.sort(
            key=lambda row: (int(row["query_id"]), int(row["prefix_tokens"]), row["method"])
        )
        all_profiles.sort(
            key=lambda row: (
                row["family"],
                int(row["prefix_tokens"]),
                int(row["profile_id"]),
                int(row["query_id"]),
            )
        )
        for name, data in (("rows", all_rows), ("profile_rows", all_profiles)):
            with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
                for row in data:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        coherence_cpu = coherence.float().cpu().numpy()
        summary = {
            "source": "PG19 past-only pre-RoPE K-mean versus token-max retrieval",
            "data_summary": data_summary,
            "profile_summary": profile,
            "state_suffix_tokens": suffixes,
            "topks": topks,
            "ranking_depth": args.ranking_depth,
            "query_q_tokens": args.query_q_tokens,
            "index_load_seconds": index_load_seconds,
            "world_size": world_size,
            "coherence": {
                "global_mean": float(np.mean(coherence_cpu)),
                "global_median": float(np.median(coherence_cpu)),
                "global_p10": float(np.quantile(coherence_cpu, 0.1)),
                "global_p90": float(np.quantile(coherence_cpu, 0.9)),
                "per_profile_mean": np.mean(coherence_cpu, axis=0).tolist(),
            },
            "retrieval_quality": summarize(all_rows, topks),
            "profile_quality": profile_summary(all_profiles, specs),
            "contains_synthetic_vectors": False,
            "selection_uses_target": False,
            "past_only": True,
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
