from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.stats import binomtest, spearmanr
from transformers import AutoModelForCausalLM

from profile_real_qk import resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map target-NLL utility over real PG19 candidate windows and test "
            "whether utility observed on one generated segment persists to the next."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--text_rows", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--memory_tokens", type=int, default=9_900_032)
    parser.add_argument("--prefix_tokens", type=int, default=64)
    parser.add_argument("--target_split_tokens", type=int, default=64)
    parser.add_argument("--candidate_depth", type=int, default=64)
    parser.add_argument("--random_windows", type=int, default=64)
    parser.add_argument("--window_blocks", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260716)
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
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def window_start_for_block(
    block_id: int,
    *,
    base_count: int,
    source_count: int,
    window_blocks: int,
) -> int:
    if block_id < base_count:
        return max(0, min(base_count - window_blocks, block_id - window_blocks // 2))
    source_offset = block_id - base_count
    return base_count + max(
        0,
        min(source_count - window_blocks, source_offset - window_blocks // 2),
    )


def context_for_window(
    start: int,
    *,
    window_blocks: int,
    base_blocks: np.ndarray,
    source_blocks: np.ndarray,
    base_count: int,
) -> np.ndarray:
    pieces = []
    for block_id in range(start, start + window_blocks):
        if block_id < base_count:
            pieces.append(np.asarray(base_blocks[block_id], dtype=np.int32))
        else:
            pieces.append(
                np.asarray(source_blocks[block_id - base_count], dtype=np.int32)
            )
    return np.stack(pieces).reshape(-1)


def build_candidates(
    rankings: dict[str, list[int]],
    *,
    query_id: int,
    candidate_depth: int,
    random_windows: int,
    total_blocks: int,
    base_count: int,
    source_count: int,
    window_blocks: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidates: dict[int, dict[str, Any]] = {}

    def add(block_id: int, origin: str, origin_rank: int) -> None:
        start = window_start_for_block(
            block_id,
            base_count=base_count,
            source_count=source_count,
            window_blocks=window_blocks,
        )
        item = candidates.setdefault(start, {"window_start": start, "origins": {}})
        previous = item["origins"].get(origin)
        item["origins"][origin] = (
            origin_rank if previous is None else min(previous, origin_rank)
        )

    for method, ranking in rankings.items():
        for origin_rank, block_id in enumerate(ranking[:candidate_depth], start=1):
            add(int(block_id), method, origin_rank)

    rng = random.Random(seed + query_id)
    random_starts = rng.sample(
        range(0, base_count - window_blocks + 1),
        min(random_windows, base_count - window_blocks + 1),
    )
    for origin_rank, start in enumerate(random_starts, start=1):
        item = candidates.setdefault(start, {"window_start": start, "origins": {}})
        item["origins"]["random"] = origin_rank

    for block_id in range(base_count, total_blocks):
        add(block_id, "oracle_source", block_id - base_count + 1)

    gold = set(range(base_count, total_blocks))
    output = []
    for item in candidates.values():
        blocks = list(range(item["window_start"], item["window_start"] + window_blocks))
        item["block_ids"] = blocks
        item["source_overlap"] = len(set(blocks) & gold)
        output.append(item)
    return sorted(output, key=lambda item: int(item["window_start"]))


@torch.inference_mode()
def batch_target_nll(
    model: AutoModelForCausalLM,
    contexts: list[np.ndarray],
    query_ids: np.ndarray,
    target_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    if not contexts:
        return []
    query = torch.from_numpy(np.asarray(query_ids, dtype=np.int64))
    target = torch.from_numpy(np.asarray(target_ids, dtype=np.int64))
    prompt_tokens = int(len(contexts[0]) + query.numel())
    target_tokens = int(target.numel())
    output: list[float] = []
    for start in range(0, len(contexts), batch_size):
        batch_contexts = contexts[start : start + batch_size]
        input_ids = torch.stack(
            [
                torch.cat(
                    [
                        torch.from_numpy(np.asarray(context, dtype=np.int64)),
                        query,
                        target,
                    ]
                )
                for context in batch_contexts
            ]
        ).to(device)
        hidden = model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.long),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        positions = torch.arange(
            prompt_tokens - 1,
            prompt_tokens + target_tokens - 1,
            device=device,
        )
        selected = hidden.index_select(1, positions)
        batch_targets = input_ids[:, prompt_tokens : prompt_tokens + target_tokens]
        losses = []
        # Chunk the LM head to avoid materializing batch x target x vocabulary logits.
        for row_hidden, row_targets in zip(selected, batch_targets):
            logits = model.lm_head(row_hidden).float()
            losses.append(F.cross_entropy(logits, row_targets, reduction="mean"))
        output.extend(float(value.item()) for value in losses)
    return output


def bootstrap_ci(values: list[float], *, samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def selected_quality(
    selected: list[dict[str, Any]],
    *,
    baseline_key: str,
    nll_key: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    deltas = [float(row[baseline_key]) - float(row[nll_key]) for row in selected]
    nll = mean(float(row[nll_key]) for row in selected)
    baseline_nll = mean(float(row[baseline_key]) for row in selected)
    return {
        "queries": len(selected),
        "baseline_mean_nll": baseline_nll,
        "baseline_ppl": math.exp(min(baseline_nll, 20.0)),
        "mean_nll": nll,
        "ppl": math.exp(min(nll, 20.0)),
        "mean_nll_improvement": mean(deltas),
        "improvement_bootstrap95": bootstrap_ci(
            deltas, samples=bootstrap_samples, seed=seed
        ),
        "positive_utility_rate": mean(delta > 0 for delta in deltas),
        "mean_source_overlap": mean(int(row["source_overlap"]) for row in selected),
    }


def choose_per_query(
    rows: list[dict[str, Any]],
    query_ids: list[int],
    predicate: Any,
    key: Any,
) -> list[dict[str, Any]]:
    selected = []
    for query_id in query_ids:
        group = [row for row in rows if int(row["query_id"]) == query_id and predicate(row)]
        if group:
            selected.append(max(group, key=key))
    return selected


def summarize(
    rows: list[dict[str, Any]],
    *,
    candidate_depth: int,
    random_windows: int,
    window_blocks: int,
    block_tokens: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    query_ids = sorted({int(row["query_id"]) for row in rows})
    retrieval_methods = ("bm25", "e5", "bm25_e5_rrf")
    correlations = []
    sparsity = []
    for query_id in query_ids:
        group = [row for row in rows if int(row["query_id"]) == query_id]
        retrieval = [
            row
            for row in group
            if any(method in row["origins"] for method in retrieval_methods)
        ]
        random_group = [row for row in group if "random" in row["origins"]]
        delta_a = [float(row["delta_nll_a"]) for row in retrieval]
        delta_b = [float(row["delta_nll_b"]) for row in retrieval]
        correlation = spearmanr(delta_a, delta_b) if len(retrieval) > 1 else None
        random_q95_a = float(
            np.quantile([float(row["delta_nll_a"]) for row in random_group], 0.95)
        )
        random_q95_b = float(
            np.quantile([float(row["delta_nll_b"]) for row in random_group], 0.95)
        )
        top_a = max(retrieval, key=lambda row: float(row["delta_nll_a"]))
        b_order = sorted(retrieval, key=lambda row: float(row["delta_nll_b"]), reverse=True)
        top_b_decile = {
            int(row["window_start"])
            for row in b_order[: max(1, math.ceil(len(b_order) / 10))]
        }
        positive_b = np.maximum(np.asarray(delta_b, dtype=np.float64), 0.0)
        effective_support = (
            float(positive_b.sum() ** 2 / np.square(positive_b).sum())
            if np.square(positive_b).sum() > 0
            else 0.0
        )
        correlations.append(
            {
                "query_id": query_id,
                "spearman_delta_a_vs_b": (
                    float(correlation.statistic)
                    if correlation is not None and math.isfinite(correlation.statistic)
                    else None
                ),
                "top_a_is_top_b_decile": int(top_a["window_start"]) in top_b_decile,
                "top_a_delta_b": float(top_a["delta_nll_b"]),
            }
        )
        events_a = [row for row in retrieval if float(row["delta_nll_a"]) > random_q95_a]
        events_b = [row for row in retrieval if float(row["delta_nll_b"]) > random_q95_b]
        sparsity.append(
            {
                "query_id": query_id,
                "retrieval_windows": len(retrieval),
                "positive_a": sum(value > 0 for value in delta_a),
                "positive_b": sum(value > 0 for value in delta_b),
                "above_random95_a": len(events_a),
                "above_random95_b": len(events_b),
                "above_random95_b_non_source": sum(
                    int(row["source_overlap"]) == 0 for row in events_b
                ),
                "positive_b_effective_support": effective_support,
            }
        )

    retrieval_predicate = lambda row: any(
        method in row["origins"] for method in retrieval_methods
    )
    selections: dict[str, list[dict[str, Any]]] = {
        "best_A_retrieval_union": choose_per_query(
            rows,
            query_ids,
            retrieval_predicate,
            lambda row: float(row["delta_nll_a"]),
        ),
        "best_A_non_source_union": choose_per_query(
            rows,
            query_ids,
            lambda row: retrieval_predicate(row) and int(row["source_overlap"]) == 0,
            lambda row: float(row["delta_nll_a"]),
        ),
        "oracle_B_retrieval_union": choose_per_query(
            rows,
            query_ids,
            retrieval_predicate,
            lambda row: float(row["delta_nll_b"]),
        ),
        "best_A_random": choose_per_query(
            rows,
            query_ids,
            lambda row: "random" in row["origins"],
            lambda row: float(row["delta_nll_a"]),
        ),
    }
    for method in retrieval_methods:
        selections[f"static_{method}_top1"] = choose_per_query(
            rows,
            query_ids,
            lambda row, name=method: name in row["origins"],
            lambda row, name=method: -int(row["origins"][name]),
        )
        selections[f"best_A_{method}"] = choose_per_query(
            rows,
            query_ids,
            lambda row, name=method: name in row["origins"],
            lambda row: float(row["delta_nll_a"]),
        )

    selection_quality = {
        name: selected_quality(
            selected,
            baseline_key="baseline_nll_b",
            nll_key="mean_nll_b",
            bootstrap_samples=bootstrap_samples,
            seed=seed + index,
        )
        for index, (name, selected) in enumerate(selections.items())
    }

    def paired_selection(left: str, right: str, comparison_seed: int) -> dict[str, Any]:
        left_lookup = {
            int(row["query_id"]): float(row["mean_nll_b"])
            for row in selections[left]
        }
        right_lookup = {
            int(row["query_id"]): float(row["mean_nll_b"])
            for row in selections[right]
        }
        shared = sorted(set(left_lookup) & set(right_lookup))
        improvements = [left_lookup[item] - right_lookup[item] for item in shared]
        return {
            "left": left,
            "right": right,
            "queries": len(shared),
            "mean_nll_improvement_right_over_left": mean(improvements),
            "improvement_bootstrap95": bootstrap_ci(
                improvements,
                samples=bootstrap_samples,
                seed=comparison_seed,
            ),
            "right_wins": sum(value > 0 for value in improvements),
            "left_wins": sum(value < 0 for value in improvements),
        }

    selection_comparisons = [
        paired_selection(
            "static_e5_top1", "best_A_retrieval_union", seed + 500
        ),
        paired_selection(
            "static_bm25_e5_rrf_top1", "best_A_retrieval_union", seed + 501
        ),
        paired_selection("best_A_random", "best_A_retrieval_union", seed + 502),
        paired_selection(
            "static_e5_top1", "best_A_non_source_union", seed + 503
        ),
    ]

    method_rank_correlations = []
    for method in retrieval_methods:
        per_query_a = []
        per_query_b = []
        for query_id in query_ids:
            group = [
                row
                for row in rows
                if int(row["query_id"]) == query_id and method in row["origins"]
            ]
            if len(group) < 2:
                continue
            negative_rank = [-int(row["origins"][method]) for row in group]
            corr_a = spearmanr(negative_rank, [float(row["delta_nll_a"]) for row in group])
            corr_b = spearmanr(negative_rank, [float(row["delta_nll_b"]) for row in group])
            if math.isfinite(corr_a.statistic):
                per_query_a.append(float(corr_a.statistic))
            if math.isfinite(corr_b.statistic):
                per_query_b.append(float(corr_b.statistic))
        method_rank_correlations.append(
            {
                "method": method,
                "mean_spearman_negative_rank_vs_delta_a": mean(per_query_a),
                "mean_spearman_negative_rank_vs_delta_b": mean(per_query_b),
            }
        )

    valid_correlations = [
        float(row["spearman_delta_a_vs_b"])
        for row in correlations
        if row["spearman_delta_a_vs_b"] is not None
    ]
    top_decile_hits = sum(bool(row["top_a_is_top_b_decile"]) for row in correlations)
    return {
        "queries": len(query_ids),
        "candidate_depth_per_retriever": candidate_depth,
        "random_windows_per_query": random_windows,
        "window_blocks": window_blocks,
        "window_tokens": window_blocks * block_tokens,
        "mean_unique_retrieval_windows": mean(
            int(row["retrieval_windows"]) for row in sparsity
        ),
        "temporal_utility_persistence": {
            "mean_per_query_spearman_delta_a_vs_b": mean(valid_correlations),
            "spearman_bootstrap95_across_queries": bootstrap_ci(
                valid_correlations,
                samples=bootstrap_samples,
                seed=seed + 600,
            ),
            "top_A_is_top_B_decile_rate": mean(
                bool(row["top_a_is_top_b_decile"]) for row in correlations
            ),
            "top_A_in_top_B_decile_binomial_p_vs_0p1": float(
                binomtest(top_decile_hits, len(correlations), 0.1, alternative="greater").pvalue
            ),
            "top_A_positive_on_B_rate": mean(
                float(row["top_a_delta_b"]) > 0 for row in correlations
            ),
        },
        "utility_sparsity": {
            "mean_positive_B_fraction": mean(
                int(row["positive_b"]) / int(row["retrieval_windows"])
                for row in sparsity
            ),
            "mean_windows_above_random95_B": mean(
                int(row["above_random95_b"]) for row in sparsity
            ),
            "mean_fraction_above_random95_B": mean(
                int(row["above_random95_b"]) / int(row["retrieval_windows"])
                for row in sparsity
            ),
            "mean_non_source_windows_above_random95_B": mean(
                int(row["above_random95_b_non_source"]) for row in sparsity
            ),
            "mean_positive_B_effective_support": mean(
                float(row["positive_b_effective_support"]) for row in sparsity
            ),
            "mean_positive_B_effective_support_fraction": mean(
                float(row["positive_b_effective_support"])
                / int(row["retrieval_windows"])
                for row in sparsity
            ),
        },
        "selection_quality_on_future_B": selection_quality,
        "paired_selection_comparisons_on_future_B": selection_comparisons,
        "retriever_rank_utility_correlation": method_rank_correlations,
        "per_query_temporal_correlation": correlations,
        "per_query_sparsity": sparsity,
    }


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
    block_tokens = int(data_summary["block_tokens"])
    source_count = int(data_summary["source_blocks"])
    total_blocks = args.memory_tokens // block_tokens
    base_count = total_blocks - source_count
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    source_blocks = np.load(data_dir / "source_blocks.npy", mmap_mode="r")
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    targets = np.load(data_dir / "targets.npy", mmap_mode="r")
    if not 0 < args.target_split_tokens < targets.shape[1]:
        raise ValueError("target_split_tokens must split the stored target")
    if source_count and args.window_blocks > source_count:
        raise ValueError("window_blocks cannot exceed source blocks")

    text_rows = [
        row
        for row in read_jsonl(args.text_rows)
        if int(row["memory_tokens"]) == args.memory_tokens
        and int(row["prefix_tokens"]) == args.prefix_tokens
    ]
    text_lookup = {
        (int(row["query_id"]), str(row["method"])): row for row in text_rows
    }

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    local_query_ids = [item for item in range(len(queries)) if item % world_size == rank]
    local_rows = []
    started = time.perf_counter()
    for query_id in local_query_ids:
        rankings = {
            method: [
                int(item)
                for item in text_lookup[(query_id, method)]["top_block_ids"]
            ]
            for method in ("bm25", "e5", "bm25_e5_rrf")
        }
        candidates = build_candidates(
            rankings,
            query_id=query_id,
            candidate_depth=args.candidate_depth,
            random_windows=args.random_windows,
            total_blocks=total_blocks,
            base_count=base_count,
            source_count=source_count,
            window_blocks=args.window_blocks,
            seed=args.seed,
        )
        contexts = [
            context_for_window(
                int(item["window_start"]),
                window_blocks=args.window_blocks,
                base_blocks=base_blocks,
                source_blocks=source_blocks[query_id],
                base_count=base_count,
            )
            for item in candidates
        ]
        query = np.asarray(queries[query_id, : args.prefix_tokens], dtype=np.int32)
        target_a = np.asarray(
            targets[query_id, : args.target_split_tokens], dtype=np.int32
        )
        target_b = np.asarray(
            targets[query_id, args.target_split_tokens :], dtype=np.int32
        )
        query_b = np.concatenate([query, target_a])
        empty = np.empty(0, dtype=np.int32)
        baseline_a = batch_target_nll(
            model, [empty], query, target_a, device, 1
        )[0]
        baseline_b = batch_target_nll(
            model, [empty], query_b, target_b, device, 1
        )[0]
        nll_a = batch_target_nll(
            model, contexts, query, target_a, device, args.batch_size
        )
        nll_b = batch_target_nll(
            model, contexts, query_b, target_b, device, args.batch_size
        )
        for item, value_a, value_b in zip(candidates, nll_a, nll_b):
            local_rows.append(
                {
                    "query_id": query_id,
                    **item,
                    "baseline_nll_a": baseline_a,
                    "baseline_nll_b": baseline_b,
                    "mean_nll_a": value_a,
                    "mean_nll_b": value_b,
                    "delta_nll_a": baseline_a - value_a,
                    "delta_nll_b": baseline_b - value_b,
                    "candidate_retrieval_uses_target": False,
                    "segment_a_is_observed_before_evaluating_b": True,
                    "selection_uses_future_segment_b": False,
                }
            )

    shard_path = output_dir / f"rows_rank{rank:03d}.jsonl"
    with shard_path.open("w", encoding="utf-8") as handle:
        for row in local_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    elapsed = time.perf_counter() - started
    (output_dir / f"runtime_rank{rank:03d}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "queries": len(local_query_ids),
                "candidate_rows": len(local_rows),
                "elapsed_seconds": elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    barrier(world_size)

    if rank == 0:
        all_rows = [
            row
            for shard in range(world_size)
            for row in read_jsonl(output_dir / f"rows_rank{shard:03d}.jsonl")
        ]
        all_rows.sort(key=lambda row: (int(row["query_id"]), int(row["window_start"])))
        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        runtimes = [
            json.loads(
                (output_dir / f"runtime_rank{shard:03d}.json").read_text(
                    encoding="utf-8"
                )
            )
            for shard in range(world_size)
        ]
        summary = {
            "source": f"{data_summary['source']} candidate utility landscape",
            "protocol": {
                "candidate_retrieval_uses_target": False,
                "segment_a_tokens": args.target_split_tokens,
                "segment_b_tokens": int(targets.shape[1] - args.target_split_tokens),
                "segment_a_is_observed_before_evaluating_b": True,
                "selection_uses_future_segment_b": False,
                "oracle_B_is_diagnostic_only": True,
                "contains_synthetic_text": False,
            },
            "memory_tokens": args.memory_tokens,
            "world_size": world_size,
            "max_rank_elapsed_seconds": max(
                float(item["elapsed_seconds"]) for item in runtimes
            ),
            "analysis": summarize(
                all_rows,
                candidate_depth=args.candidate_depth,
                random_windows=args.random_windows,
                window_blocks=args.window_blocks,
                block_tokens=block_tokens,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
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
