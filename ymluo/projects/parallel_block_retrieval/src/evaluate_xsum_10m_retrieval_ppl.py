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

from evaluate_xsum_news_ppl_retrieval import target_nll
from profile_real_qk import resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 512-token reader PPL from 10M XSum retrieval selections."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--text_rows", required=True)
    parser.add_argument("--qk_rows")
    parser.add_argument("--qk_method", default="qk_kcentered_cosine_head_rrf")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--memory_tokens", type=int, default=10_000_000)
    parser.add_argument("--prefix_tokens", type=int, default=64)
    parser.add_argument("--retrieval_blocks", type=int, default=8)
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument(
        "--locality_windows",
        default="",
        help="Optional comma-separated window_length:window_count layouts, e.g. 8:1,4:2,2:4.",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--seed", type=int, default=20260716)
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


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def reciprocal_rank_fusion(
    rankings: list[list[int]], *, budget: int, rrf_k: float
) -> list[int]:
    scores: dict[int, float] = {}
    best: dict[int, int] = {}
    for ranking in rankings:
        for rank, block_id in enumerate(ranking, start=1):
            block_id = int(block_id)
            scores[block_id] = scores.get(block_id, 0.0) + 1.0 / (rrf_k + rank)
            best[block_id] = min(best.get(block_id, rank), rank)
    return sorted(scores, key=lambda item: (-scores[item], best[item], item))[:budget]


def parse_window_layouts(spec: str, budget: int) -> list[tuple[int, int]]:
    layouts = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        length, count = (int(value) for value in item.split(":"))
        if length <= 0 or count <= 0 or length * count != budget:
            raise ValueError("every locality layout must contain positive values and fill the budget")
        layouts.append((length, count))
    return layouts


def locality_window_selection(
    ranking: list[int],
    *,
    total_blocks: int,
    window_length: int,
    window_count: int,
    rrf_k: float,
) -> list[int]:
    weights = {
        int(block_id): 1.0 / (rrf_k + rank)
        for rank, block_id in enumerate(ranking, start=1)
    }
    starts = {
        max(0, min(total_blocks - window_length, int(block_id) - offset))
        for block_id in ranking
        for offset in range(window_length)
    }
    scored = sorted(
        (
            (
                sum(weights.get(block_id, 0.0) for block_id in range(start, start + window_length)),
                start,
            )
            for start in starts
        ),
        key=lambda item: (-item[0], item[1]),
    )
    windows: list[set[int]] = []
    for _, start in scored:
        candidate = set(range(start, start + window_length))
        if any(candidate & chosen for chosen in windows):
            continue
        windows.append(candidate)
        if len(windows) >= window_count:
            break
    return sorted(block_id for window in windows for block_id in window)


def selected_context(
    selection: list[int],
    *,
    base_blocks: np.ndarray,
    source_blocks: np.ndarray,
    base_count: int,
) -> np.ndarray:
    pieces = []
    for block_id in sorted(set(int(item) for item in selection)):
        if block_id < base_count:
            pieces.append(np.asarray(base_blocks[block_id], dtype=np.int32))
        else:
            source_index = block_id - base_count
            pieces.append(np.asarray(source_blocks[source_index], dtype=np.int32))
    if not pieces:
        return np.empty(0, dtype=np.int32)
    return np.stack(pieces).reshape(-1)


def bootstrap_mean_ci(values: list[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(5000, len(array)))
    samples = array[indices].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_sample_method = {
        (int(row["query_id"]), str(row["method"])): row for row in rows
    }
    query_only = {
        query_id: float(row["mean_nll"])
        for (query_id, method), row in by_sample_method.items()
        if method == "query_only"
    }
    oracle = {
        query_id: float(row["mean_nll"])
        for (query_id, method), row in by_sample_method.items()
        if method == "oracle_source512"
    }
    output = []
    for method in sorted({str(row["method"]) for row in rows}):
        group = [row for row in rows if str(row["method"]) == method]
        total_nll = sum(float(row["total_nll"]) for row in group)
        total_tokens = sum(int(row["target_tokens"]) for row in group)
        micro_nll = total_nll / total_tokens
        delta_query = [
            float(row["mean_nll"]) - query_only[int(row["query_id"])] for row in group
        ]
        delta_oracle = [
            float(row["mean_nll"]) - oracle[int(row["query_id"])] for row in group
        ]
        output.append(
            {
                "method": method,
                "queries": len(group),
                "retrieved_tokens": int(group[0]["retrieved_tokens"]),
                "micro_nll": micro_nll,
                "ppl": math.exp(min(micro_nll, 20.0)),
                "mean_forward_seconds": mean(
                    float(row["forward_seconds"]) for row in group
                ),
                "mean_source_block_recall": mean(
                    float(row["source_block_recall"]) for row in group
                ),
                "source_any_hit": mean(bool(row["source_any_hit"]) for row in group),
                "source_last_hit": mean(bool(row["source_last_hit"]) for row in group),
                "mean_delta_nll_vs_query_only": mean(delta_query),
                "delta_vs_query_only_bootstrap95": bootstrap_mean_ci(
                    delta_query, seed + len(output)
                ),
                "mean_delta_nll_vs_oracle": mean(delta_oracle),
                "delta_vs_oracle_bootstrap95": bootstrap_mean_ci(
                    delta_oracle, seed + 100 + len(output)
                ),
            }
        )
    return output


@torch.inference_mode()
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
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    targets = np.load(data_dir / "targets.npy", mmap_mode="r")
    source_count = int(data_summary["source_blocks"])
    locality_layouts = parse_window_layouts(
        args.locality_windows, args.retrieval_blocks
    )
    total_blocks = args.memory_tokens // int(data_summary["block_tokens"])
    base_count = total_blocks - source_count

    text_rows = [
        row
        for row in read_jsonl(args.text_rows)
        if int(row["memory_tokens"]) == args.memory_tokens
        and int(row["prefix_tokens"]) == args.prefix_tokens
    ]
    text_lookup = {
        (int(row["query_id"]), str(row["method"])): row for row in text_rows
    }
    qk_lookup: dict[tuple[int, str], dict[str, Any]] = {}
    if args.qk_rows:
        qk_lookup = {
            (int(row["query_id"]), str(row["method"])): row
            for row in read_jsonl(args.qk_rows)
            if int(row["memory_tokens"]) == args.memory_tokens
            and int(row["prefix_tokens"]) == args.prefix_tokens
        }

    dtype = resolve_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    local_query_ids = [item for item in range(len(queries)) if item % world_size == rank]
    rows = []
    for query_id in local_query_ids:
        selections: dict[str, list[int]] = {
            "query_only": [],
            "random512": sorted(
                random.Random(args.seed + query_id).sample(
                    range(total_blocks), args.retrieval_blocks
                )
            ),
            "oracle_source512": list(range(base_count, base_count + source_count)),
        }
        for method in ("bm25", "e5", "bm25_e5_rrf"):
            ranking = [
                int(item)
                for item in text_lookup[(query_id, method)]["top_block_ids"]
            ]
            selections[method] = ranking[: args.retrieval_blocks]
            for window_length, window_count in locality_layouts:
                selections[f"{method}_local_{window_count}x{window_length}"] = (
                    locality_window_selection(
                        ranking,
                        total_blocks=total_blocks,
                        window_length=window_length,
                        window_count=window_count,
                        rrf_k=args.rrf_k,
                    )
                )
        if qk_lookup:
            qk_ranking = [
                int(item)
                for item in qk_lookup[(query_id, args.qk_method)]["top_block_ids"]
            ]
            selections[args.qk_method] = qk_ranking[: args.retrieval_blocks]
            selections["bm25_e5_qk_rrf"] = reciprocal_rank_fusion(
                [
                    [
                        int(item)
                        for item in text_lookup[(query_id, "bm25")]["top_block_ids"]
                    ],
                    [
                        int(item)
                        for item in text_lookup[(query_id, "e5")]["top_block_ids"]
                    ],
                    qk_ranking,
                ],
                budget=args.retrieval_blocks,
                rrf_k=args.rrf_k,
            )

        gold = set(range(base_count, base_count + source_count))
        for method, selection in selections.items():
            context = selected_context(
                selection,
                base_blocks=base_blocks,
                source_blocks=source_blocks[query_id],
                base_count=base_count,
            )
            mean_nll, total_nll, target_tokens, seconds, model_input_tokens = target_nll(
                model,
                context,
                np.asarray(queries[query_id], dtype=np.int32),
                np.asarray(targets[query_id], dtype=np.int32),
                device,
            )
            selected = set(selection)
            rows.append(
                {
                    "query_id": query_id,
                    "method": method,
                    "selected_block_ids": selection,
                    "retrieved_tokens": len(selection)
                    * int(data_summary["block_tokens"]),
                    "model_input_tokens": model_input_tokens,
                    "mean_nll": mean_nll,
                    "total_nll": total_nll,
                    "target_tokens": target_tokens,
                    "forward_seconds": seconds,
                    "source_block_recall": len(selected & gold) / source_count,
                    "source_any_hit": bool(selected & gold),
                    "source_last_hit": base_count + source_count - 1 in selected,
                    "selection_uses_target": False,
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
            "memory_tokens": args.memory_tokens,
            "prefix_tokens": args.prefix_tokens,
            "retrieval_blocks": args.retrieval_blocks,
            "retrieval_tokens": args.retrieval_blocks
            * int(data_summary["block_tokens"]),
            "world_size": world_size,
            "qk_method": args.qk_method if qk_lookup else None,
            "locality_window_layouts": [
                {"window_length": length, "window_count": count}
                for length, count in locality_layouts
            ],
            "contains_synthetic_text": False,
            "selection_uses_target": False,
            "quality": summarize(all_rows, args.seed),
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
