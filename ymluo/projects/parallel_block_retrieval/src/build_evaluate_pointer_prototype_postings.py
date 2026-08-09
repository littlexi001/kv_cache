from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

from analyze_state_pointer_query_manifold import pointer_token_indices
from profile_real_qk import barrier, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build train-only head/prototype block postings from real SVD K and measure "
            "held-out gold recall at fixed candidate budgets."
        )
    )
    parser.add_argument("--base_profile_dir", required=True)
    parser.add_argument("--step_profile", required=True)
    parser.add_argument("--prototype_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--test_splits", default="test")
    parser.add_argument("--posting_depth", type=int, default=2048)
    parser.add_argument("--block_batch", type=int, default=32)
    parser.add_argument("--route_prototypes", default="1,2,4")
    parser.add_argument("--candidate_budgets", default="39,195,391,1953")
    parser.add_argument("--rrf_constant", type=int, default=60)
    parser.add_argument("--reuse_postings", action="store_true")
    parser.add_argument(
        "--max_heads",
        type=int,
        default=0,
        help="Use only the first train-frozen heads; zero keeps every indexed head.",
    )
    parser.add_argument("--save_candidate_method", default="weighted_rrf_idf")
    parser.add_argument("--save_route_prototypes", type=int, default=1)
    return parser.parse_args()


def parse_ints(spec: str) -> list[int]:
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def merge_topk(
    left_scores: torch.Tensor,
    left_ids: torch.Tensor,
    right_scores: torch.Tensor,
    right_ids: torch.Tensor,
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.cat([left_scores, right_scores], dim=1)
    ids = torch.cat([left_ids, right_ids], dim=1)
    best_scores, positions = torch.topk(scores, k=depth, dim=1)
    return best_scores, ids.gather(1, positions)


@torch.inference_mode()
def local_prototype_postings(
    *,
    centers: torch.Tensor,
    arrays: list[tuple[np.ndarray, int]],
    kv_head: int,
    posting_depth: int,
    block_batch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = centers.to(device=device, dtype=torch.float16)
    scores = torch.full(
        (len(centers), posting_depth), -torch.inf, dtype=torch.float32, device=device
    )
    ids = torch.full(
        (len(centers), posting_depth), -1, dtype=torch.int64, device=device
    )
    for array, block_start in arrays:
        for local_start in range(0, len(array), block_batch):
            count = min(block_batch, len(array) - local_start)
            key = torch.from_numpy(
                np.array(array[local_start : local_start + count, :, kv_head], copy=True)
            ).to(device=device, dtype=torch.float16)
            batch_scores = torch.einsum("pd,btd->pbt", centers, key).amax(dim=-1).float()
            batch_ids = torch.arange(
                block_start + local_start,
                block_start + local_start + count,
                dtype=torch.int64,
                device=device,
            )[None].expand(len(centers), -1)
            scores, ids = merge_topk(
                scores, ids, batch_scores, batch_ids, posting_depth
            )
    return scores.cpu(), ids.cpu()


def build_postings(
    *,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    output_dir: Path,
) -> Path | None:
    profile_dir = Path(args.base_profile_dir)
    profile_summary = json.loads((profile_dir / "summary.json").read_text())
    shards = profile_summary["shards"]
    assigned_shards = [shard for index, shard in enumerate(shards) if index % world_size == rank]
    if not assigned_shards:
        raise ValueError("world size cannot exceed K-index shard count")

    prototype_payload = torch.load(args.prototype_path, map_location="cpu", weights_only=False)
    centers = F.normalize(prototype_payload["centers"].float(), dim=-1)
    selected_heads = [tuple(int(item) for item in pair) for pair in prototype_payload["selected_heads"]]
    if args.max_heads > 0:
        if args.max_heads > len(selected_heads):
            raise ValueError("max_heads exceeds prototype head count")
        selected_heads = selected_heads[: args.max_heads]
        centers = centers[: args.max_heads]
    num_query_heads = int(profile_summary["num_query_heads"])
    num_kv_heads = int(profile_summary["num_kv_heads"])
    repeat_groups = num_query_heads // num_kv_heads
    local_scores = []
    local_ids = []
    array_cache: dict[tuple[int, int], np.ndarray] = {}
    started = time.perf_counter()
    for head_index, (layer, query_head) in enumerate(selected_heads):
        arrays = []
        for shard_index, shard in enumerate(assigned_shards):
            cache_key = (shard_index, layer)
            if cache_key not in array_cache:
                source = Path(shard["layer_k_paths"][str(layer)])
                array_cache[cache_key] = np.load(profile_dir / source.name, mmap_mode="r")
            arrays.append((array_cache[cache_key], int(shard["block_start"])))
        scores, ids = local_prototype_postings(
            centers=centers[head_index],
            arrays=arrays,
            kv_head=query_head // repeat_groups,
            posting_depth=args.posting_depth,
            block_batch=args.block_batch,
            device=device,
        )
        local_scores.append(scores.to(dtype=torch.float16))
        local_ids.append(ids.to(dtype=torch.int32))
        print(
            json.dumps(
                {
                    "rank": rank,
                    "head": head_index + 1,
                    "heads": len(selected_heads),
                    "layer": layer,
                    "query_head": query_head,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    local_path = output_dir / f"prototype_postings_rank{rank:03d}.pt"
    torch.save(
        {
            "scores": torch.stack(local_scores),
            "ids": torch.stack(local_ids),
            "selected_heads": selected_heads,
            "assigned_shards": [int(shards.index(shard)) for shard in assigned_shards],
        },
        local_path,
    )
    barrier(world_size)
    if rank != 0:
        return None

    payloads = [
        torch.load(
            output_dir / f"prototype_postings_rank{item:03d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for item in range(world_size)
    ]
    global_scores = payloads[0]["scores"].float()
    global_ids = payloads[0]["ids"].long()
    for payload in payloads[1:]:
        scores = torch.cat([global_scores, payload["scores"].float()], dim=-1)
        ids = torch.cat([global_ids, payload["ids"].long()], dim=-1)
        global_scores, positions = torch.topk(scores, k=args.posting_depth, dim=-1)
        global_ids = ids.gather(-1, positions)
    postings_path = output_dir / "prototype_postings.pt"
    torch.save(
        {
            "scores": global_scores.to(dtype=torch.float16),
            "ids": global_ids.to(dtype=torch.int32),
            "selected_heads": selected_heads,
            "prototype_path": args.prototype_path,
            "posting_depth": args.posting_depth,
            "num_blocks": int(profile_summary["num_blocks"]),
            "source": "train-only state-pointer prototype support postings over real SVD K",
        },
        postings_path,
    )
    return postings_path


def update_max(target: np.ndarray, ids: np.ndarray, values: np.ndarray) -> None:
    valid = ids >= 0
    np.maximum.at(target, ids[valid], values[valid])


def update_sum(target: np.ndarray, ids: np.ndarray, values: np.ndarray) -> None:
    valid = ids >= 0
    np.add.at(target, ids[valid], values[valid])


def evaluate_postings(args: argparse.Namespace, postings_path: Path) -> dict[str, Any]:
    step_payload = torch.load(args.step_profile, map_location="cpu", weights_only=False)
    prototype_payload = torch.load(args.prototype_path, map_location="cpu", weights_only=False)
    postings = torch.load(postings_path, map_location="cpu", weights_only=False)
    centers = F.normalize(prototype_payload["centers"].float(), dim=-1).numpy()
    posting_scores = postings["scores"].float().numpy()
    posting_ids = postings["ids"].long().numpy()
    selected_heads = [tuple(int(item) for item in pair) for pair in postings["selected_heads"]]
    if args.max_heads > 0:
        if args.max_heads > len(selected_heads):
            raise ValueError("max_heads exceeds indexed head count")
        selected_heads = selected_heads[: args.max_heads]
        centers = centers[: args.max_heads]
        posting_scores = posting_scores[: args.max_heads]
        posting_ids = posting_ids[: args.max_heads]
    layers = [int(item) for item in step_payload["layers"]]
    layer_to_index = {layer: index for index, layer in enumerate(layers)}
    test_splits = {item.strip() for item in args.test_splits.split(",") if item.strip()}
    route_counts = parse_ints(args.route_prototypes)
    budgets = parse_ints(args.candidate_budgets)
    if max(budgets) > int(postings["num_blocks"]):
        raise ValueError("candidate budget exceeds number of blocks")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    test_indices = [
        index
        for index, step in enumerate(step_payload["steps"])
        if str(step["split"]) in test_splits
    ]
    token_indices = {
        index: pointer_token_indices(
            tokenizer=tokenizer,
            step=step_payload["steps"][index],
            token_positions=step_payload["token_positions"][index],
        )
        for index in test_indices
    }
    num_blocks = int(postings["num_blocks"])
    valid_posting_ids = posting_ids[posting_ids >= 0]
    document_frequency = np.bincount(valid_posting_ids, minlength=num_blocks).astype(
        np.float32
    )
    posting_lists = float(np.prod(posting_ids.shape[:-1]))
    block_idf = np.log((posting_lists + 1.0) / (document_frequency + 1.0)) + 1.0
    methods = (
        "lipschitz_lower",
        "cosine_product",
        "cosine_product_idf",
        "weighted_rrf",
        "weighted_rrf_idf",
    )
    hits = {
        (routes, method, budget, step_index): 0
        for routes in route_counts
        for method in methods
        for budget in budgets
        for step_index in (0, 1)
    }
    counts = {0: 0, 1: 0}
    union_hits = {
        (routes, step_index): 0
        for routes in route_counts
        for step_index in (0, 1)
    }
    union_sizes = {routes: [] for routes in route_counts}
    unique_candidates = {
        (routes, method, budget): []
        for routes in route_counts
        for method in methods
        for budget in budgets
    }
    started = time.perf_counter()
    saved_candidates = []
    saved_candidate_steps = []
    for ordinal, index in enumerate(test_indices):
        step = step_payload["steps"][index]
        step_index = int(step["step_index"])
        counts[step_index] += 1
        gold = {int(item) for item in step["target_block_ids"]}
        for routes in route_counts:
            accumulators = {
                "lipschitz_lower": np.full(num_blocks, -np.inf, dtype=np.float32),
                "cosine_product": np.full(num_blocks, -np.inf, dtype=np.float32),
                "cosine_product_idf": np.full(
                    num_blocks, -np.inf, dtype=np.float32
                ),
                "weighted_rrf": np.zeros(num_blocks, dtype=np.float32),
                "weighted_rrf_idf": np.zeros(num_blocks, dtype=np.float32),
            }
            union_mask = np.zeros(num_blocks, dtype=np.bool_)
            for head_index, (layer, query_head) in enumerate(selected_heads):
                query = step_payload["svd_q"][
                    index,
                    token_indices[index],
                    layer_to_index[layer],
                    query_head,
                ].float().numpy()
                similarity = query @ centers[head_index].T
                top = np.argpartition(similarity, -routes, axis=1)[:, -routes:]
                top_similarity = np.take_along_axis(similarity, top, axis=1)
                order = np.argsort(-top_similarity, axis=1)
                top = np.take_along_axis(top, order, axis=1)
                top_similarity = np.take_along_axis(top_similarity, order, axis=1)
                ids = posting_ids[head_index, top].reshape(-1, args.posting_depth)
                support = posting_scores[head_index, top].reshape(-1, args.posting_depth)
                valid_ids = ids[ids >= 0]
                union_mask[valid_ids] = True
                route_similarity = top_similarity.reshape(-1, 1)
                distance = np.sqrt(np.maximum(2.0 - 2.0 * route_similarity, 0.0))
                update_max(
                    accumulators["lipschitz_lower"],
                    ids.ravel(),
                    (support - distance).ravel(),
                )
                update_max(
                    accumulators["cosine_product"],
                    ids.ravel(),
                    (support * np.maximum(route_similarity, 0.0)).ravel(),
                )
                update_max(
                    accumulators["cosine_product_idf"],
                    ids.ravel(),
                    (
                        support
                        * np.maximum(route_similarity, 0.0)
                        * block_idf[np.maximum(ids, 0)]
                    ).ravel(),
                )
                posting_rank = np.arange(1, args.posting_depth + 1, dtype=np.float32)[None]
                rrf_value = np.maximum(route_similarity, 0.0) / (
                    args.rrf_constant + posting_rank
                )
                update_sum(
                    accumulators["weighted_rrf"],
                    ids.ravel(),
                    np.broadcast_to(rrf_value, ids.shape).ravel(),
                )
                update_sum(
                    accumulators["weighted_rrf_idf"],
                    ids.ravel(),
                    (
                        np.broadcast_to(rrf_value, ids.shape)
                        * block_idf[np.maximum(ids, 0)]
                    ).ravel(),
                )
            union_hits[(routes, step_index)] += int(
                any(union_mask[block_id] for block_id in gold)
            )
            union_sizes[routes].append(int(union_mask.sum()))
            for method, scores in accumulators.items():
                maximum = max(budgets)
                candidate = np.argpartition(scores, -maximum)[-maximum:]
                candidate = candidate[np.argsort(-scores[candidate])]
                if (
                    routes == args.save_route_prototypes
                    and method == args.save_candidate_method
                ):
                    saved_candidates.append(candidate.copy())
                    saved_candidate_steps.append(index)
                for budget in budgets:
                    chosen = candidate[:budget]
                    hits[(routes, method, budget, step_index)] += int(
                        bool(gold & set(int(item) for item in chosen))
                    )
                    unique_candidates[(routes, method, budget)].append(
                        int(np.isfinite(scores[chosen]).sum())
                        if method != "weighted_rrf"
                        else int((scores[chosen] > 0).sum())
                    )
        if (ordinal + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "stage": "evaluate",
                        "steps": ordinal + 1,
                        "total": len(test_indices),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    rows = []
    for routes in route_counts:
        for method in methods:
            for budget in budgets:
                hop1 = hits[(routes, method, budget, 0)] / counts[0]
                hop2 = hits[(routes, method, budget, 1)] / counts[1]
                rows.append(
                    {
                        "route_prototypes": routes,
                        "score_method": method,
                        "candidate_budget": budget,
                        "candidate_fraction": budget / num_blocks,
                        "hop1_gold_recall": hop1,
                        "hop2_gold_recall": hop2,
                        "macro_step_recall": (hop1 + hop2) / 2,
                        "mean_nonempty_candidates": mean(
                            unique_candidates[(routes, method, budget)]
                        ),
                    }
                )
    candidate_path = Path(args.output_dir) / "routed_candidates.pt"
    if len(saved_candidates) != len(test_indices):
        raise RuntimeError("saved candidate configuration was not evaluated")
    torch.save(
        {
            "candidate_ids": torch.from_numpy(np.stack(saved_candidates)).to(
                dtype=torch.int32
            ),
            "step_profile_indices": saved_candidate_steps,
            "candidate_budget": max(budgets),
            "score_method": args.save_candidate_method,
            "route_prototypes": args.save_route_prototypes,
            "selected_heads": selected_heads,
            "source": "prototype-routed held-out candidates before exact QK reranking",
        },
        candidate_path,
    )
    return {
        "source": "held-out state-pointer routing through train-only prototype block postings",
        "contains_synthetic_vectors": False,
        "selection_uses_test_gold": False,
        "test_splits": sorted(test_splits),
        "test_steps": len(test_indices),
        "step_counts": counts,
        "num_blocks": num_blocks,
        "posting_depth": args.posting_depth,
        "selected_heads": selected_heads,
        "max_heads": len(selected_heads),
        "prototype_path": args.prototype_path,
        "postings_path": str(postings_path),
        "routed_candidates_path": str(candidate_path),
        "evaluation_seconds": time.perf_counter() - started,
        "random_recall_by_budget": {
            str(budget): budget / num_blocks for budget in budgets
        },
        "posting_document_frequency": {
            "mean": float(document_frequency.mean()),
            "p95": float(np.quantile(document_frequency, 0.95)),
            "max": float(document_frequency.max()),
            "unseen_block_rate": float((document_frequency == 0).mean()),
        },
        "routed_posting_union": {
            str(routes): {
                "mean_blocks": mean(union_sizes[routes]),
                "mean_fraction": mean(union_sizes[routes]) / num_blocks,
                "hop1_gold_recall": union_hits[(routes, 0)] / counts[0],
                "hop2_gold_recall": union_hits[(routes, 1)] / counts[1],
            }
            for routes in route_counts
        },
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    rank, world_size, _local_rank, device = setup_distributed()
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)
    build_started = time.perf_counter()
    existing_postings = output_dir / "prototype_postings.pt"
    if args.reuse_postings:
        if not existing_postings.exists():
            raise FileNotFoundError(existing_postings)
        postings_path = existing_postings if rank == 0 else None
    else:
        postings_path = build_postings(
            args=args,
            rank=rank,
            world_size=world_size,
            device=device,
            output_dir=output_dir,
        )
    barrier(world_size)
    posting_build_wall_seconds = time.perf_counter() - build_started
    if rank == 0:
        assert postings_path is not None
        summary = evaluate_postings(args, postings_path)
        summary["world_size"] = world_size
        summary["posting_build_wall_seconds"] = posting_build_wall_seconds
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    barrier(world_size)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
