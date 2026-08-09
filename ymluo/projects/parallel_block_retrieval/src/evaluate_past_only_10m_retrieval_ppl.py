from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from evaluate_xsum_10m_retrieval_ppl import locality_window_selection
from evaluate_xsum_news_ppl_retrieval import target_nll
from profile_real_qk import resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reader PPL for retrieval from a real 10M past-only PG19 memory."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--text_rows", required=True)
    parser.add_argument("--additional_rows")
    parser.add_argument(
        "--base_methods",
        default="bm25,e5,bm25_e5_rrf",
        help="Comma-separated methods read from text_rows; may be empty.",
    )
    parser.add_argument(
        "--additional_methods",
        default="",
        help="Comma-separated methods read from additional_rows.",
    )
    parser.add_argument(
        "--additional_memory_tokens",
        type=int,
        help="Optional memory scale used to filter additional_rows.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--state_suffix_tokens", type=int, default=512)
    parser.add_argument("--retrieval_blocks", type=int, default=8)
    parser.add_argument("--locality_windows", default="4:2")
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260717)
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


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def parse_layouts(spec: str, budget: int) -> list[tuple[int, int]]:
    layouts = []
    for item in spec.split(","):
        if not item.strip():
            continue
        length, count = (int(value) for value in item.split(":"))
        if length * count != budget:
            raise ValueError("every locality layout must fill retrieval_blocks")
        layouts.append((length, count))
    return layouts


def selected_context(selection: list[int], base_blocks: np.ndarray) -> np.ndarray:
    if not selection:
        return np.empty(0, dtype=np.int32)
    return np.stack(
        [np.asarray(base_blocks[int(block_id)], dtype=np.int32) for block_id in sorted(set(selection))]
    ).reshape(-1)


def bootstrap_ci(values: list[float], *, samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def selection_scope_metrics(
    selection: list[int],
    *,
    query_scope: int,
    local_start: int,
    block_scope_ids: np.ndarray,
    block_original_centers: np.ndarray,
) -> dict[str, Any]:
    if not selection:
        return {
            "same_scope_fraction": 0.0,
            "same_scope_any": False,
            "same_scope_within_4k_any": False,
            "same_scope_within_16k_any": False,
        }
    selected = np.asarray(selection, dtype=np.int64)
    scopes = np.asarray(block_scope_ids[selected], dtype=np.int64)
    positions = np.asarray(block_original_centers[selected], dtype=np.int64)
    same = scopes == query_scope
    if np.any(same & (positions >= local_start)):
        raise RuntimeError("past-only invariant violated")
    distances = local_start - positions
    return {
        "same_scope_fraction": float(np.mean(same)),
        "same_scope_any": bool(np.any(same)),
        "same_scope_within_4k_any": bool(np.any(same & (distances <= 4096))),
        "same_scope_within_16k_any": bool(np.any(same & (distances <= 16384))),
    }


def summarize(rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    query_only = {
        int(row["query_id"]): float(row["mean_nll"])
        for row in rows
        if row["method"] == "query_only"
    }
    output = []
    for index, method in enumerate(sorted({str(row["method"]) for row in rows})):
        group = [row for row in rows if row["method"] == method]
        micro_nll = sum(float(row["total_nll"]) for row in group) / sum(
            int(row["target_tokens"]) for row in group
        )
        deltas = [
            float(row["mean_nll"]) - query_only[int(row["query_id"])] for row in group
        ]
        output.append(
            {
                "method": method,
                "queries": len(group),
                "retrieved_tokens": int(group[0]["retrieved_tokens"]),
                "micro_nll": micro_nll,
                "ppl": math.exp(min(micro_nll, 20.0)),
                "mean_forward_seconds": mean(float(row["forward_seconds"]) for row in group),
                "mean_delta_nll_vs_query_only": mean(deltas),
                "delta_vs_query_only_bootstrap95": bootstrap_ci(
                    deltas, samples=bootstrap_samples, seed=seed + index
                ),
                "same_scope_any": mean(float(row["same_scope_any"]) for row in group),
                "mean_same_scope_fraction": mean(
                    float(row["same_scope_fraction"]) for row in group
                ),
                "same_scope_within_4k_any": mean(
                    float(row["same_scope_within_4k_any"]) for row in group
                ),
                "same_scope_within_16k_any": mean(
                    float(row["same_scope_within_16k_any"]) for row in group
                ),
            }
        )
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)
    data_dir = Path(args.data_dir)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    if not data_summary.get("past_only") or data_summary.get("source_blocks") != 0:
        raise ValueError("past-only dataset without source blocks required")
    metadata = {
        int(row["query_id"]): row for row in read_jsonl(data_dir / "metadata.jsonl")
    }
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    targets = np.load(data_dir / "targets.npy", mmap_mode="r")
    block_scope_ids = np.load(data_dir / "base_block_scope_ids.npy", mmap_mode="r")
    block_original_centers = np.load(
        data_dir / "base_block_original_centers.npy", mmap_mode="r"
    )
    layouts = parse_layouts(args.locality_windows, args.retrieval_blocks)
    text_rows = [
        row
        for row in read_jsonl(args.text_rows)
        if int(row["prefix_tokens"]) == args.state_suffix_tokens
    ]
    base_methods = [item.strip() for item in args.base_methods.split(",") if item.strip()]
    text_lookup = {
        (int(row["query_id"]), str(row["method"])): row
        for row in text_rows
        if str(row["method"]) in base_methods
    }
    additional_methods = [
        item.strip() for item in args.additional_methods.split(",") if item.strip()
    ]
    if additional_methods:
        if not args.additional_rows:
            raise ValueError("additional_rows is required when additional_methods are set")
        extra_rows = [
            row
            for row in read_jsonl(args.additional_rows)
            if int(row["prefix_tokens"]) == args.state_suffix_tokens
            and str(row["method"]) in additional_methods
            and (
                args.additional_memory_tokens is None
                or int(row.get("memory_tokens", -1)) == args.additional_memory_tokens
            )
        ]
        text_lookup.update(
            {(int(row["query_id"]), str(row["method"])): row for row in extra_rows}
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    local_query_ids = [query_id for query_id in range(len(queries)) if query_id % world_size == rank]
    rows = []
    for query_id in local_query_ids:
        retrieval_methods = [*base_methods, *additional_methods]
        rankings = {
            method: [
                int(item) for item in text_lookup[(query_id, method)]["top_block_ids"]
            ]
            for method in retrieval_methods
        }
        selections: dict[str, list[int]] = {
            "query_only": [],
            "random512": sorted(
                random.Random(args.seed + query_id).sample(
                    range(len(base_blocks)), args.retrieval_blocks
                )
            ),
        }
        for method, ranking in rankings.items():
            selections[method] = ranking[: args.retrieval_blocks]
            for window_length, window_count in layouts:
                selections[f"{method}_local_{window_count}x{window_length}"] = (
                    locality_window_selection(
                        ranking,
                        total_blocks=len(base_blocks),
                        window_length=window_length,
                        window_count=window_count,
                        rrf_k=args.rrf_k,
                    )
                )
        query = np.asarray(
            queries[query_id, -args.state_suffix_tokens :], dtype=np.int32
        )
        target = np.asarray(targets[query_id], dtype=np.int32)
        query_scope = int(metadata[query_id]["book_index"])
        local_start = int(metadata[query_id]["local_context_start_token"])
        for method, selection in selections.items():
            context = selected_context(selection, base_blocks)
            mean_nll, total_nll, target_tokens, seconds, model_input_tokens = target_nll(
                model, context, query, target, device
            )
            rows.append(
                {
                    "query_id": query_id,
                    "method": method,
                    "selected_block_ids": selection,
                    "retrieved_tokens": len(selection) * int(data_summary["block_tokens"]),
                    "model_input_tokens": model_input_tokens,
                    "mean_nll": mean_nll,
                    "total_nll": total_nll,
                    "target_tokens": target_tokens,
                    "forward_seconds": seconds,
                    "selection_uses_target": False,
                    **selection_scope_metrics(
                        selection,
                        query_scope=query_scope,
                        local_start=local_start,
                        block_scope_ids=block_scope_ids,
                        block_original_centers=block_original_centers,
                    ),
                }
            )

    shard_path = output_dir / f"rows_rank{rank:03d}.jsonl"
    with shard_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    barrier(world_size)
    if rank == 0:
        all_rows = [
            row
            for shard in range(world_size)
            for row in read_jsonl(output_dir / f"rows_rank{shard:03d}.jsonl")
        ]
        all_rows.sort(key=lambda row: (int(row["query_id"]), str(row["method"])))
        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "source": f"{data_summary['source']} retrieval-conditioned reader PPL",
            "data_summary": data_summary,
            "state_suffix_tokens": args.state_suffix_tokens,
            "retrieval_blocks": args.retrieval_blocks,
            "retrieval_tokens": args.retrieval_blocks * int(data_summary["block_tokens"]),
            "world_size": world_size,
            "locality_window_layouts": [
                {"window_length": length, "window_count": count}
                for length, count in layouts
            ],
            "past_only": True,
            "predefined_source": False,
            "contains_synthetic_text": False,
            "selection_uses_target": False,
            "quality": summarize(
                all_rows, bootstrap_samples=args.bootstrap_samples, seed=args.seed
            ),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
