from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from run_lexical_block_retrieval import bm25_matrix, decode_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure temporal Q/ranking stability and evidence-switch dynamics over a "
            "real 10M all-head K index, with a matched dynamic BM25 baseline."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--base_profile_dir", required=True)
    parser.add_argument("--trajectory_profile", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--q_field",
        default="svd_q",
        help="Tensor field to score, e.g. svd_q, state_only_probe_q, or evidence_probe_q.",
    )
    parser.add_argument(
        "--selected_heads",
        default=(
            "3:10,14:15,13:6,8:4,21:8,16:15,20:6,14:6,11:3,26:7,"
            "11:5,16:13,20:15,14:14,14:7,16:8"
        ),
        help="Comma-separated layer:query_head pairs frozen before this experiment.",
    )
    parser.add_argument("--head_topk", type=int, default=64)
    parser.add_argument("--head_vote_depth", type=int, default=16)
    parser.add_argument("--final_blocks", type=int, default=39)
    parser.add_argument("--rrf_constant", type=int, default=60)
    parser.add_argument("--block_batch", type=int, default=32)
    parser.add_argument("--skip_bm25", action="store_true")
    parser.add_argument(
        "--score_prior_dir",
        default="",
        help="Optional exact train-only head/block mean and std profile for score z-normalization.",
    )
    parser.add_argument("--prior_fold", type=int, default=0)
    parser.add_argument("--prior_std_epsilon", type=float, default=1.0e-4)
    return parser.parse_args()


def parse_heads(spec: str) -> list[tuple[int, int]]:
    heads = []
    seen = set()
    for item in spec.split(","):
        layer_text, head_text = item.strip().split(":", maxsplit=1)
        pair = (int(layer_text), int(head_text))
        if pair not in seen:
            seen.add(pair)
            heads.append(pair)
    if not heads:
        raise ValueError("selected_heads cannot be empty")
    return heads


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def percentile(values: Iterable[float], q: float) -> float:
    values = list(values)
    return float(np.percentile(values, q)) if values else math.nan


def jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set, right_set = set(int(item) for item in left), set(int(item) for item in right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def rrf(ids: np.ndarray, depth: int, target: int, constant: int) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for head_ids in ids:
        for rank, block_id in enumerate(head_ids[:depth], start=1):
            scores[int(block_id)] += 1.0 / (constant + rank)
    return [
        block_id
        for block_id, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
            :target
        ]
    ]


def flatten_states(
    payload: dict[str, Any], q_field: str
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if q_field not in payload:
        raise KeyError(f"trajectory profile has no Q field {q_field!r}")
    q = payload[q_field]
    mask = payload["mask"]
    trajectories = payload["trajectories"]
    states = []
    metadata = []
    for trajectory_index, trajectory in enumerate(trajectories):
        count = int(mask[trajectory_index].sum().item())
        for state_index in range(count):
            states.append(q[trajectory_index, state_index])
            state_row = dict(trajectory["state_metadata"][state_index])
            state_row.update(
                {
                    "flat_state_index": len(metadata),
                    "trajectory_index": trajectory_index,
                    "query_id": int(trajectory["query_id"]),
                    "split": str(trajectory["split"]),
                    "question": str(trajectory["question"]),
                    "step_question": str(trajectory["step_question"]),
                    "bridge_target": str(trajectory["bridge_target"]),
                    "bridge_generation_hit": bool(
                        trajectory["generation"]["raw_target_hit"]
                    ),
                    "first_complete_state": trajectory["generation"]["first_complete_state"],
                    "hop1_gold_block_ids": [
                        int(item) for item in trajectory["hop1_gold_block_ids"]
                    ],
                    "hop2_gold_block_ids": [
                        int(item) for item in trajectory["hop2_gold_block_ids"]
                    ],
                }
            )
            metadata.append(state_row)
    return torch.stack(states), metadata


def gold_scores_for_head(
    *,
    q: torch.Tensor,
    metadata: Sequence[dict[str, Any]],
    array: np.ndarray,
    block_start: int,
    block_end: int,
    kv_head: int,
    field: str,
    device: torch.device,
    prior_mean: np.ndarray | None = None,
    prior_std: np.ndarray | None = None,
    prior_std_epsilon: float = 1.0e-4,
    reduce_across_ranks: bool = True,
    validate: bool = True,
) -> torch.Tensor:
    output = torch.full((len(metadata),), -torch.inf, dtype=torch.float32, device=device)
    for state_index, row in enumerate(metadata):
        local_scores = []
        for block_id in row[field]:
            block_id = int(block_id)
            if block_start <= block_id < block_end:
                key = torch.from_numpy(
                    np.array(array[block_id - block_start, :, kv_head], copy=True)
                ).to(device=device, dtype=torch.float16)
                score = torch.mv(key, q[state_index]).max().float()
                if prior_mean is not None and prior_std is not None:
                    score = (score - float(prior_mean[block_id])) / max(
                        float(prior_std[block_id]), prior_std_epsilon
                    )
                local_scores.append(score)
        if local_scores:
            output[state_index] = torch.stack(local_scores).max()
    if reduce_across_ranks and dist.is_initialized():
        dist.all_reduce(output, op=dist.ReduceOp.MAX)
    if validate and not torch.isfinite(output).all():
        raise RuntimeError(f"some {field} blocks were absent from all K-index shards")
    return output


def scan_one_head(
    *,
    q: torch.Tensor,
    gold1: torch.Tensor,
    gold2: torch.Tensor,
    array: np.ndarray,
    block_start: int,
    kv_head: int,
    block_batch: int,
    topk: int,
    device: torch.device,
    prior_mean: np.ndarray | None = None,
    prior_std: np.ndarray | None = None,
    prior_std_epsilon: float = 1.0e-4,
    reduce_across_ranks: bool = True,
    one_based_ranks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state_count = q.shape[0]
    local_blocks = int(array.shape[0])
    best_scores = torch.full((state_count, topk), -torch.inf, dtype=torch.float32, device=device)
    best_ids = torch.full((state_count, topk), -1, dtype=torch.int64, device=device)
    rank1 = torch.zeros(state_count, dtype=torch.int64, device=device)
    rank2 = torch.zeros(state_count, dtype=torch.int64, device=device)
    for local_start in range(0, local_blocks, block_batch):
        count = min(block_batch, local_blocks - local_start)
        key = torch.from_numpy(
            np.array(array[local_start : local_start + count, :, kv_head], copy=True)
        ).to(device=device, dtype=torch.float16)
        scores = torch.einsum("sd,btd->sbt", q, key).amax(dim=-1).float()
        if prior_mean is not None and prior_std is not None:
            global_start = block_start + local_start
            batch_mean = torch.from_numpy(
                np.array(prior_mean[global_start : global_start + count], copy=True)
            ).to(device=device, dtype=torch.float32)
            batch_std = torch.from_numpy(
                np.array(prior_std[global_start : global_start + count], copy=True)
            ).to(device=device, dtype=torch.float32)
            scores = (scores - batch_mean[None]) / batch_std.clamp_min(prior_std_epsilon)[
                None
            ]
        rank1 += (scores > gold1[:, None]).sum(dim=1)
        rank2 += (scores > gold2[:, None]).sum(dim=1)
        ids = torch.arange(
            block_start + local_start,
            block_start + local_start + count,
            dtype=torch.int64,
            device=device,
        )[None].expand(state_count, -1)
        merged_scores = torch.cat([best_scores, scores], dim=1)
        merged_ids = torch.cat([best_ids, ids], dim=1)
        best_scores, positions = torch.topk(merged_scores, k=topk, dim=1)
        best_ids = merged_ids.gather(1, positions)
    if reduce_across_ranks and dist.is_initialized():
        dist.all_reduce(rank1, op=dist.ReduceOp.SUM)
        dist.all_reduce(rank2, op=dist.ReduceOp.SUM)
    if one_based_ranks:
        rank1 = rank1 + 1
        rank2 = rank2 + 1
    return best_scores, best_ids, rank1, rank2


def merge_local_topk(
    left_scores: torch.Tensor,
    left_ids: torch.Tensor,
    right_scores: torch.Tensor,
    right_ids: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.cat([left_scores, right_scores], dim=1)
    ids = torch.cat([left_ids, right_ids], dim=1)
    best_scores, positions = torch.topk(scores, k=topk, dim=1)
    return best_scores, ids.gather(1, positions)


def gather_global_topk(
    scores: torch.Tensor, ids: torch.Tensor, topk: int, rank: int, world_size: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if world_size == 1:
        return scores.cpu(), ids.cpu()
    gathered_scores = [torch.empty_like(scores) for _ in range(world_size)]
    gathered_ids = [torch.empty_like(ids) for _ in range(world_size)]
    dist.all_gather(gathered_scores, scores)
    dist.all_gather(gathered_ids, ids)
    if rank != 0:
        return None, None
    all_scores = torch.cat(gathered_scores, dim=1)
    all_ids = torch.cat(gathered_ids, dim=1)
    top_scores, positions = torch.topk(all_scores, k=topk, dim=1)
    return top_scores.cpu(), all_ids.gather(1, positions).cpu()


def build_transition_metrics(
    *,
    full_q: torch.Tensor,
    selected_q: torch.Tensor,
    metadata: Sequence[dict[str, Any]],
    top_ids: np.ndarray,
    final_sets: Sequence[Sequence[int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transitions = []
    by_trajectory: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        by_trajectory[int(row["trajectory_index"])].append(index)
    for trajectory_index, state_ids in by_trajectory.items():
        for previous, current in zip(state_ids, state_ids[1:]):
            selected_cos = (selected_q[previous] * selected_q[current]).sum(dim=-1).numpy()
            all_cos = (full_q[previous] * full_q[current]).sum(dim=-1).reshape(-1).numpy()
            delta = np.maximum(1.0 - all_cos.astype(np.float64), 0.0)
            delta_sum = max(float(delta.sum()), 1.0e-12)
            probability = delta / delta_sum
            effective_heads = float(np.exp(-(probability * np.log(probability + 1.0e-12)).sum()))
            top16_share = float(np.sort(delta)[-min(16, len(delta)) :].sum() / delta_sum)
            head_jaccards = [
                jaccard(top_ids[previous, head, :16], top_ids[current, head, :16])
                for head in range(top_ids.shape[1])
            ]
            transitions.append(
                {
                    "trajectory_index": trajectory_index,
                    "query_id": int(metadata[current]["query_id"]),
                    "from_state": int(metadata[previous]["state_index"]),
                    "to_state": int(metadata[current]["state_index"]),
                    "bridge_progress": float(metadata[current]["bridge_progress"]),
                    "bridge_complete": bool(metadata[current]["bridge_complete"]),
                    "selected_q_cosine_mean": float(selected_cos.mean()),
                    "selected_q_cosine_min": float(selected_cos.min()),
                    "all_q_cosine_mean": float(all_cos.mean()),
                    "delta_top16_head_share": top16_share,
                    "delta_effective_heads": effective_heads,
                    "head_top16_jaccard_mean": float(np.mean(head_jaccards)),
                    "rrf39_jaccard": jaccard(final_sets[previous], final_sets[current]),
                }
            )
    summary = {
        "transitions": len(transitions),
        "selected_q_cosine_mean": mean(row["selected_q_cosine_mean"] for row in transitions),
        "selected_q_cosine_p05": percentile(
            (row["selected_q_cosine_mean"] for row in transitions), 5
        ),
        "all_q_cosine_mean": mean(row["all_q_cosine_mean"] for row in transitions),
        "head_top16_jaccard_mean": mean(
            row["head_top16_jaccard_mean"] for row in transitions
        ),
        "head_top16_jaccard_median": percentile(
            (row["head_top16_jaccard_mean"] for row in transitions), 50
        ),
        "rrf39_jaccard_mean": mean(row["rrf39_jaccard"] for row in transitions),
        "rrf39_jaccard_median": percentile(
            (row["rrf39_jaccard"] for row in transitions), 50
        ),
        "rrf39_stable_ge_0p5_rate": mean(
            row["rrf39_jaccard"] >= 0.5 for row in transitions
        ),
        "rrf39_change_le_0p25_rate": mean(
            row["rrf39_jaccard"] <= 0.25 for row in transitions
        ),
        "delta_top16_head_share_mean": mean(
            row["delta_top16_head_share"] for row in transitions
        ),
        "delta_effective_heads_mean": mean(
            row["delta_effective_heads"] for row in transitions
        ),
    }
    return transitions, summary


def bm25_metrics(
    *,
    corpus_dir: Path,
    model_name_or_path: str,
    metadata: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    decode_started = time.perf_counter()
    block_texts = decode_blocks(tokenizer, blocks)
    dynamic_queries = [
        f"{row['step_question']} {row['generated_text']}".strip() for row in metadata
    ]
    scores, index_summary = bm25_matrix(
        block_texts,
        dynamic_queries,
        min_df=2,
        max_df=0.98,
        k1=1.2,
        b=0.75,
    )
    rows = []
    previous_by_trajectory: dict[int, list[int]] = defaultdict(list)
    for state_index, row in enumerate(metadata):
        order = np.lexsort((np.arange(scores.shape[1]), -scores[state_index]))
        top39 = order[:39].tolist()
        gold1 = min(
            int(np.flatnonzero(order == int(block_id))[0]) + 1
            for block_id in row["hop1_gold_block_ids"]
        )
        gold2 = min(
            int(np.flatnonzero(order == int(block_id))[0]) + 1
            for block_id in row["hop2_gold_block_ids"]
        )
        rows.append(
            {
                "top39": top39,
                "hop1_rank": gold1,
                "hop2_rank": gold2,
                "hop1_hit39": gold1 <= 39,
                "hop2_hit39": gold2 <= 39,
            }
        )
        previous_by_trajectory[int(row["trajectory_index"])].append(state_index)
    transition_jaccards = []
    for state_ids in previous_by_trajectory.values():
        transition_jaccards.extend(
            jaccard(rows[left]["top39"], rows[right]["top39"])
            for left, right in zip(state_ids, state_ids[1:])
        )
    summary = {
        "index": index_summary,
        "decode_seconds": time.perf_counter() - decode_started,
        "dynamic_states": len(rows),
        "top39_consecutive_jaccard_mean": mean(transition_jaccards),
        "top39_consecutive_jaccard_median": percentile(transition_jaccards, 50),
    }
    return rows, summary


def summarize_trajectories(
    *,
    metadata: Sequence[dict[str, Any]],
    head_rank1: np.ndarray,
    head_rank2: np.ndarray,
    final_sets: Sequence[Sequence[int]],
    bm25_rows: Sequence[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_trajectory: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        by_trajectory[int(row["trajectory_index"])].append(index)
    rows = []
    for trajectory_index, state_ids in sorted(by_trajectory.items()):
        first = state_ids[0]
        gold1 = set(metadata[first]["hop1_gold_block_ids"])
        gold2 = set(metadata[first]["hop2_gold_block_ids"])
        q_hop1_hits = [bool(gold1 & set(final_sets[state])) for state in state_ids]
        q_hop2_hits = [bool(gold2 & set(final_sets[state])) for state in state_ids]
        any_head_hop2 = [bool((head_rank2[state] <= 16).any()) for state in state_ids]
        completion = metadata[first]["first_complete_state"]
        precomplete = [
            state
            for state in state_ids
            if completion is None or int(metadata[state]["state_index"]) < int(completion)
        ]
        first_q_hop2 = next(
            (int(metadata[state]["state_index"]) for state, hit in zip(state_ids, q_hop2_hits) if hit),
            None,
        )
        first_any_head_hop2 = next(
            (
                int(metadata[state]["state_index"])
                for state, hit in zip(state_ids, any_head_hop2)
                if hit
            ),
            None,
        )
        row = {
            "trajectory_index": trajectory_index,
            "query_id": int(metadata[first]["query_id"]),
            "bridge_generation_hit": bool(metadata[first]["bridge_generation_hit"]),
            "bridge_complete_state": completion,
            "states": len(state_ids),
            "q_rrf_hop1_initial_hit39": q_hop1_hits[0],
            "q_rrf_hop2_initial_hit39": q_hop2_hits[0],
            "q_rrf_hop2_ever_hit39": any(q_hop2_hits),
            "q_rrf_hop2_precomplete_hit39": any(q_hop2_hits[state_ids.index(i)] for i in precomplete),
            "q_anyhead_hop2_initial_hit16": any_head_hop2[0],
            "q_anyhead_hop2_ever_hit16": any(any_head_hop2),
            "q_anyhead_hop2_precomplete_hit16": any(
                any_head_hop2[state_ids.index(i)] for i in precomplete
            ),
            "q_first_rrf_hop2_state": first_q_hop2,
            "q_first_anyhead_hop2_state": first_any_head_hop2,
            "q_rrf_lead_vs_bridge": (
                first_q_hop2 - int(completion)
                if first_q_hop2 is not None and completion is not None
                else None
            ),
            "q_anyhead_lead_vs_bridge": (
                first_any_head_hop2 - int(completion)
                if first_any_head_hop2 is not None and completion is not None
                else None
            ),
            "q_hop1_best_head_rank_initial": int(head_rank1[first].min()),
            "q_hop2_best_head_rank_initial": int(head_rank2[first].min()),
            "q_hop2_best_head_rank_best_state": int(head_rank2[state_ids].min()),
        }
        if bm25_rows is not None:
            bm25_hop2 = [bool(bm25_rows[state]["hop2_hit39"]) for state in state_ids]
            bm25_pre = [bool(bm25_rows[state]["hop2_hit39"]) for state in precomplete]
            row.update(
                {
                    "bm25_hop2_initial_hit39": bm25_hop2[0],
                    "bm25_hop2_ever_hit39": any(bm25_hop2),
                    "bm25_hop2_precomplete_hit39": any(bm25_pre),
                    "q_rrf_precomplete_unique_vs_bm25": (
                        row["q_rrf_hop2_precomplete_hit39"] and not any(bm25_pre)
                    ),
                    "q_anyhead_precomplete_unique_vs_bm25": (
                        row["q_anyhead_hop2_precomplete_hit16"] and not any(bm25_pre)
                    ),
                }
            )
        rows.append(row)

    summary: dict[str, Any] = {"trajectories": len(rows)}
    boolean_fields = [
        "bridge_generation_hit",
        "q_rrf_hop1_initial_hit39",
        "q_rrf_hop2_initial_hit39",
        "q_rrf_hop2_ever_hit39",
        "q_rrf_hop2_precomplete_hit39",
        "q_anyhead_hop2_initial_hit16",
        "q_anyhead_hop2_ever_hit16",
        "q_anyhead_hop2_precomplete_hit16",
        "bm25_hop2_initial_hit39",
        "bm25_hop2_ever_hit39",
        "bm25_hop2_precomplete_hit39",
        "q_rrf_precomplete_unique_vs_bm25",
        "q_anyhead_precomplete_unique_vs_bm25",
    ]
    for field in boolean_fields:
        if field in rows[0]:
            summary[f"{field}_rate"] = mean(bool(row[field]) for row in rows)
    for bridge_hit in (False, True):
        group = [row for row in rows if bool(row["bridge_generation_hit"]) == bridge_hit]
        if group:
            summary[f"bridge_{'correct' if bridge_hit else 'wrong'}"] = {
                "trajectories": len(group),
                "q_rrf_hop2_ever_hit39_rate": mean(
                    row["q_rrf_hop2_ever_hit39"] for row in group
                ),
                "q_anyhead_hop2_ever_hit16_rate": mean(
                    row["q_anyhead_hop2_ever_hit16"] for row in group
                ),
                "bm25_hop2_ever_hit39_rate": (
                    mean(row["bm25_hop2_ever_hit39"] for row in group)
                    if bm25_rows is not None
                    else None
                ),
            }
    return rows, summary


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    selected_heads = parse_heads(args.selected_heads)
    prior_means = None
    prior_stds = None
    prior_model_by_head: dict[tuple[int, int], int] = {}
    if args.score_prior_dir:
        prior_dir = Path(args.score_prior_dir)
        prior_means = np.load(prior_dir / "exact_train_mean.npy", mmap_mode="r")
        prior_stds = np.load(prior_dir / "exact_train_std.npy", mmap_mode="r")
        with (prior_dir / "models.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["fold"]) == args.prior_fold:
                    prior_model_by_head[(int(row["layer"]), int(row["query_head"]))] = int(
                        row["model_index"]
                    )
        missing = [head for head in selected_heads if head not in prior_model_by_head]
        if missing:
            raise ValueError(
                f"score prior fold {args.prior_fold} has no model for selected heads {missing}"
            )
        if prior_means.shape != prior_stds.shape:
            raise ValueError("score prior mean/std shapes differ")
    profile_dir = Path(args.base_profile_dir)
    profile_summary = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    shards = sorted(profile_summary["shards"], key=lambda row: int(row["block_start"]))
    if world_size > len(shards):
        raise ValueError(
            f"scoring world_size {world_size} cannot exceed K-index shards {len(shards)}"
        )
    assigned_shards = shards[rank::world_size]
    if not assigned_shards:
        raise RuntimeError(f"rank {rank} has no assigned K-index shard")
    trajectory = torch.load(args.trajectory_profile, map_location="cpu", weights_only=False)
    full_q, metadata = flatten_states(trajectory, args.q_field)
    layers = [int(item) for item in trajectory["layers"]]
    layer_to_index = {layer: index for index, layer in enumerate(layers)}
    selected_q_cpu = torch.stack(
        [full_q[:, layer_to_index[layer], head] for layer, head in selected_heads], dim=1
    )
    selected_q = selected_q_cpu.to(device=device, dtype=torch.float16)
    num_query_heads = int(trajectory["num_query_heads"])
    num_kv_heads = int(trajectory["num_kv_heads"])
    repeat_groups = num_query_heads // num_kv_heads
    top_ids_by_head = []
    top_scores_by_head = []
    ranks1 = []
    ranks2 = []
    array_cache: dict[tuple[int, int], np.ndarray] = {}
    started = time.perf_counter()
    for head_index, (layer, query_head) in enumerate(selected_heads):
        kv_head = query_head // repeat_groups
        q = selected_q[:, head_index]
        prior_mean = None
        prior_std = None
        if prior_means is not None and prior_stds is not None:
            prior_model_index = prior_model_by_head[(layer, query_head)]
            prior_mean = prior_means[prior_model_index]
            prior_std = prior_stds[prior_model_index]
        gold1 = torch.full((len(metadata),), -torch.inf, dtype=torch.float32, device=device)
        gold2 = torch.full_like(gold1, -torch.inf)
        head_arrays = []
        for shard_index, shard in enumerate(assigned_shards):
            cache_key = (shards.index(shard), layer)
            if cache_key not in array_cache:
                source = Path(shard["layer_k_paths"][str(layer)])
                array_cache[cache_key] = np.load(profile_dir / source.name, mmap_mode="r")
            array = array_cache[cache_key]
            block_start = int(shard["block_start"])
            block_end = int(shard["block_end"])
            head_arrays.append((array, block_start))
            gold1 = torch.maximum(
                gold1,
                gold_scores_for_head(
                    q=q,
                    metadata=metadata,
                    array=array,
                    block_start=block_start,
                    block_end=block_end,
                    kv_head=kv_head,
                    field="hop1_gold_block_ids",
                    device=device,
                    prior_mean=prior_mean,
                    prior_std=prior_std,
                    prior_std_epsilon=args.prior_std_epsilon,
                    reduce_across_ranks=False,
                    validate=False,
                ),
            )
            gold2 = torch.maximum(
                gold2,
                gold_scores_for_head(
                    q=q,
                    metadata=metadata,
                    array=array,
                    block_start=block_start,
                    block_end=block_end,
                    kv_head=kv_head,
                    field="hop2_gold_block_ids",
                    device=device,
                    prior_mean=prior_mean,
                    prior_std=prior_std,
                    prior_std_epsilon=args.prior_std_epsilon,
                    reduce_across_ranks=False,
                    validate=False,
                ),
            )
        if dist.is_initialized():
            dist.all_reduce(gold1, op=dist.ReduceOp.MAX)
            dist.all_reduce(gold2, op=dist.ReduceOp.MAX)
        if not torch.isfinite(gold1).all() or not torch.isfinite(gold2).all():
            raise RuntimeError("some gold blocks were absent from all assigned K-index shards")

        local_scores = torch.full(
            (len(metadata), args.head_topk), -torch.inf, dtype=torch.float32, device=device
        )
        local_ids = torch.full(
            (len(metadata), args.head_topk), -1, dtype=torch.int64, device=device
        )
        rank1 = torch.zeros(len(metadata), dtype=torch.int64, device=device)
        rank2 = torch.zeros_like(rank1)
        for array, block_start in head_arrays:
            shard_scores, shard_ids, shard_rank1, shard_rank2 = scan_one_head(
                q=q,
                gold1=gold1,
                gold2=gold2,
                array=array,
                block_start=block_start,
                kv_head=kv_head,
                block_batch=args.block_batch,
                topk=args.head_topk,
                device=device,
                prior_mean=prior_mean,
                prior_std=prior_std,
                prior_std_epsilon=args.prior_std_epsilon,
                reduce_across_ranks=False,
                one_based_ranks=False,
            )
            local_scores, local_ids = merge_local_topk(
                local_scores,
                local_ids,
                shard_scores,
                shard_ids,
                args.head_topk,
            )
            rank1 += shard_rank1
            rank2 += shard_rank2
        if dist.is_initialized():
            dist.all_reduce(rank1, op=dist.ReduceOp.SUM)
            dist.all_reduce(rank2, op=dist.ReduceOp.SUM)
        rank1 += 1
        rank2 += 1
        global_scores, global_ids = gather_global_topk(
            local_scores, local_ids, args.head_topk, rank, world_size
        )
        if rank == 0:
            assert global_scores is not None and global_ids is not None
            top_scores_by_head.append(global_scores)
            top_ids_by_head.append(global_ids)
            ranks1.append(rank1.cpu())
            ranks2.append(rank2.cpu())
        print(
            json.dumps(
                {
                    "rank": rank,
                    "head": head_index + 1,
                    "heads": len(selected_heads),
                    "layer": layer,
                    "query_head": query_head,
                    "assigned_shards": len(assigned_shards),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    if rank == 0:
        top_ids = torch.stack(top_ids_by_head, dim=1).numpy()
        top_scores = torch.stack(top_scores_by_head, dim=1).numpy()
        head_rank1 = torch.stack(ranks1, dim=1).numpy()
        head_rank2 = torch.stack(ranks2, dim=1).numpy()
        final_sets = [
            rrf(ids, args.head_vote_depth, args.final_blocks, args.rrf_constant)
            for ids in top_ids
        ]
        transitions, transition_summary = build_transition_metrics(
            full_q=full_q,
            selected_q=selected_q_cpu,
            metadata=metadata,
            top_ids=top_ids,
            final_sets=final_sets,
        )
        bm25_rows = None
        bm25_summary = None
        if not args.skip_bm25:
            bm25_rows, bm25_summary = bm25_metrics(
                corpus_dir=Path(args.corpus_dir),
                model_name_or_path=args.model_name_or_path,
                metadata=metadata,
            )
        trajectory_rows, trajectory_summary = summarize_trajectories(
            metadata=metadata,
            head_rank1=head_rank1,
            head_rank2=head_rank2,
            final_sets=final_sets,
            bm25_rows=bm25_rows,
        )
        torch.save(
            {
                "top_ids": torch.from_numpy(top_ids),
                "top_scores": torch.from_numpy(top_scores),
                "hop1_head_ranks": torch.from_numpy(head_rank1),
                "hop2_head_ranks": torch.from_numpy(head_rank2),
                "selected_heads": selected_heads,
                "state_metadata": metadata,
                "final_rrf_block_ids": final_sets,
            },
            output_dir / "retrieval_dynamics.pt",
        )
        with (output_dir / "transitions.jsonl").open("w", encoding="utf-8") as handle:
            for row in transitions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (output_dir / "trajectory_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in trajectory_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "source": "trajectory-state retrieval dynamics over real 10M Qwen K",
            "trajectory_profile_space": trajectory.get("profile_space"),
            "contains_synthetic_vectors": False,
            "contains_synthetic_text": False,
            "contains_source_context": trajectory.get("contains_source_context"),
            "uses_oracle_bridge_state": bool(
                trajectory.get("uses_oracle_bridge_state", False)
            ),
            "selection_uses_gold_first_hop_evidence": bool(
                trajectory.get("contains_source_context", True)
            ),
            "selection_uses_gold_second_hop_evidence": False,
            "num_tokens": int(profile_summary["num_tokens"]),
            "num_blocks": int(profile_summary["num_blocks"]),
            "states": len(metadata),
            "q_field": args.q_field,
            "selected_heads": [
                {"layer": layer, "query_head": head} for layer, head in selected_heads
            ],
            "selected_head_source": (
                "frozen 16-head query-responsiveness gate from the 2WikiMQA-heldout LODO fold"
            ),
            "score_normalization": (
                {
                    "mode": "train-only per-head per-block z-score",
                    "prior_dir": args.score_prior_dir,
                    "prior_fold": args.prior_fold,
                    "std_epsilon": args.prior_std_epsilon,
                }
                if args.score_prior_dir
                else {"mode": "raw max-QK"}
            ),
            "head_vote_depth": args.head_vote_depth,
            "final_blocks": args.final_blocks,
            "transition": transition_summary,
            "trajectory": trajectory_summary,
            "bm25": bm25_summary,
            "world_size": world_size,
            "k_index_shards": len(shards),
            "shards_per_rank": [len(shards[item::world_size]) for item in range(world_size)],
            "qk_scan_wall_seconds": time.perf_counter() - started,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
