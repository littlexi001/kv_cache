import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


FAMILIES = ["exact", "paraphrase", "conflict", "bridge2", "rule_chain4"]


@dataclass
class SyntheticCases:
    family: List[str]
    support_ids: torch.Tensor
    support_mask: torch.Tensor
    hard_ids: torch.Tensor
    hard_mask: torch.Tensor
    query_embeddings: torch.Tensor
    block_embeddings: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_tokens", type=int, default=10_000_000)
    parser.add_argument("--block_tokens", type=int, default=256)
    parser.add_argument("--target_tokens", type=int, default=10_000)
    parser.add_argument("--num_queries", type=int, default=512)
    parser.add_argument("--dense_dim", type=int, default=128)
    parser.add_argument("--svd_rank", type=int, default=64)
    parser.add_argument("--qabs_dims", type=int, default=8)
    parser.add_argument("--hard_negatives", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--risk_gap2", type=float, default=0.18)
    parser.add_argument("--risk_min_top1", type=float, default=1.25)
    parser.add_argument("--methods", type=str, default="lexical,qabs8,svd64,hybrid,risk_hybrid_lazy")
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def setup_distributed() -> Tuple[int, int, int, torch.device, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if is_distributed and not dist.is_initialized():
        backend = "nccl" if has_cuda else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank, device, is_distributed


def barrier_and_sync(device: torch.device, is_distributed: bool) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if is_distributed:
        dist.barrier()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def shard_bounds(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    start = (total * rank) // world_size
    end = (total * (rank + 1)) // world_size
    return start, end


def choose_unique_supports(rng: random.Random, used: set, num_blocks: int, count: int) -> List[int]:
    ids = []
    while len(ids) < count:
        candidate = rng.randrange(num_blocks)
        if candidate not in used:
            used.add(candidate)
            ids.append(candidate)
    return ids


def make_synthetic_cases(
    *,
    num_blocks: int,
    num_queries: int,
    dense_dim: int,
    hard_negatives: int,
    seed: int,
) -> SyntheticCases:
    rng = random.Random(seed)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    query_embeddings = F.normalize(torch.randn(num_queries, dense_dim, generator=gen), dim=1)
    # Concentrate a bit more signal in the first half, matching a low-rank K-SVD basis.
    query_embeddings[:, : dense_dim // 2] *= 1.7
    query_embeddings = F.normalize(query_embeddings, dim=1)

    block_embeddings = F.normalize(torch.randn(num_blocks, dense_dim, generator=gen), dim=1)

    max_supports = 4
    support_ids = torch.full((num_queries, max_supports), -1, dtype=torch.long)
    support_mask = torch.zeros((num_queries, max_supports), dtype=torch.bool)
    hard_ids = torch.full((num_queries, hard_negatives + 4), -1, dtype=torch.long)
    hard_mask = torch.zeros((num_queries, hard_negatives + 4), dtype=torch.bool)
    family: List[str] = []

    used_supports: set = set()
    for q in range(num_queries):
        fam = FAMILIES[q % len(FAMILIES)]
        family.append(fam)
        support_count = 4 if fam == "rule_chain4" else 2 if fam == "bridge2" else 1
        supports = choose_unique_supports(rng, used_supports, num_blocks, support_count)
        for j, block_id in enumerate(supports):
            support_ids[q, j] = block_id
            support_mask[q, j] = True
            noise = 0.08 * torch.randn(dense_dim, generator=gen)
            block_embeddings[block_id] = F.normalize(query_embeddings[q] + noise, dim=0)

        local_hards: List[int] = []
        while len(local_hards) < hard_negatives:
            block_id = rng.randrange(num_blocks)
            if block_id not in supports:
                local_hards.append(block_id)
        if fam in {"conflict", "rule_chain4"}:
            for _ in range(4):
                block_id = rng.randrange(num_blocks)
                if block_id not in supports:
                    local_hards.append(block_id)
                    noise = 0.55 * torch.randn(dense_dim, generator=gen)
                    block_embeddings[block_id] = F.normalize(0.55 * query_embeddings[q] + noise, dim=0)
        for j, block_id in enumerate(local_hards[: hard_negatives + 4]):
            hard_ids[q, j] = block_id
            hard_mask[q, j] = True

    block_embeddings = F.normalize(block_embeddings, dim=1)
    return SyntheticCases(
        family=family,
        support_ids=support_ids,
        support_mask=support_mask,
        hard_ids=hard_ids,
        hard_mask=hard_mask,
        query_embeddings=query_embeddings,
        block_embeddings=block_embeddings,
    )


def make_lexical_scores(
    cases: SyntheticCases,
    *,
    local_start: int,
    local_end: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    num_queries = cases.query_embeddings.shape[0]
    local_n = local_end - local_start
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 17 + local_start)
    scores = 0.025 * torch.rand(num_queries, local_n, generator=gen)

    for q, fam in enumerate(cases.family):
        support_boosts: Sequence[float]
        if fam == "exact":
            support_boosts = [1.50]
            hard_boost = 1.00
        elif fam == "paraphrase":
            support_boosts = [0.35]
            hard_boost = 1.10
        elif fam == "conflict":
            support_boosts = [1.10]
            hard_boost = 1.20
        elif fam == "bridge2":
            support_boosts = [1.20, 0.45]
            hard_boost = 1.10
        else:
            support_boosts = [0.82, 0.75, 0.70, 0.65]
            hard_boost = 1.12

        for j, boost in enumerate(support_boosts):
            block_id = int(cases.support_ids[q, j].item())
            if local_start <= block_id < local_end:
                scores[q, block_id - local_start] = boost

        for j in range(cases.hard_ids.shape[1]):
            if not bool(cases.hard_mask[q, j].item()):
                continue
            block_id = int(cases.hard_ids[q, j].item())
            if local_start <= block_id < local_end:
                scores[q, block_id - local_start] = hard_boost + 0.04 * (j % 3)

    return scores.to(device=device, non_blocking=True)


def local_topk(scores: torch.Tensor, k: int, local_start: int) -> Tuple[torch.Tensor, torch.Tensor]:
    k = min(k, scores.shape[1])
    vals, idx = torch.topk(scores, k=k, dim=1)
    return vals, idx.to(torch.long) + local_start


def distributed_merge_topk(
    local_vals: torch.Tensor,
    local_ids: torch.Tensor,
    final_k: int,
    is_distributed: bool,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not is_distributed:
        return local_topk_from_candidates(local_vals, local_ids, final_k)

    vals_parts = [torch.empty_like(local_vals) for _ in range(world_size)]
    ids_parts = [torch.empty_like(local_ids) for _ in range(world_size)]
    dist.all_gather(vals_parts, local_vals.contiguous())
    dist.all_gather(ids_parts, local_ids.contiguous())
    all_vals = torch.cat(vals_parts, dim=1)
    all_ids = torch.cat(ids_parts, dim=1)
    return local_topk_from_candidates(all_vals, all_ids, final_k)


def local_topk_from_candidates(vals: torch.Tensor, ids: torch.Tensor, final_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    top_vals, pos = torch.topk(vals, k=min(final_k, vals.shape[1]), dim=1)
    top_ids = torch.gather(ids, 1, pos)
    return top_vals, top_ids


def qabs_scores(query: torch.Tensor, blocks: torch.Tensor, dims: int) -> torch.Tensor:
    num_queries = query.shape[0]
    out = torch.empty(num_queries, blocks.shape[0], device=blocks.device, dtype=blocks.dtype)
    qabs_idx = torch.topk(query.abs(), k=dims, dim=1).indices
    for q in range(num_queries):
        idx = qabs_idx[q]
        out[q] = blocks[:, idx].matmul(query[q, idx])
    return out


def compute_risk_mask_from_lexical(
    lex_scores: torch.Tensor,
    local_start: int,
    budget_blocks: int,
    risk_gap2: float,
    risk_min_top1: float,
    is_distributed: bool,
    world_size: int,
) -> torch.Tensor:
    probe_k = max(8, min(budget_blocks, lex_scores.shape[1]))
    local_vals, local_ids = local_topk(lex_scores, probe_k, local_start)
    global_vals, _ = distributed_merge_topk(local_vals, local_ids, probe_k, is_distributed, world_size)
    top1 = global_vals[:, 0]
    top2 = global_vals[:, 1] if global_vals.shape[1] > 1 else torch.zeros_like(top1)
    gap2 = top1 - top2
    return (top1 < risk_min_top1) | (gap2 < risk_gap2)


def evaluate_retrieval(
    final_ids: torch.Tensor,
    cases: SyntheticCases,
    method: str,
    risk_trigger_rate: Optional[float] = None,
) -> Dict[str, object]:
    ids_cpu = final_ids.detach().cpu()
    num_queries = ids_cpu.shape[0]
    any_support = 0
    all_support = 0
    clean_answer_proxy = 0
    support_frac_sum = 0.0
    contamination = 0
    by_family: Dict[str, Dict[str, float]] = {}

    for q in range(num_queries):
        retrieved = set(int(x) for x in ids_cpu[q].tolist())
        supports = [
            int(cases.support_ids[q, j].item())
            for j in range(cases.support_ids.shape[1])
            if bool(cases.support_mask[q, j].item())
        ]
        hards = [
            int(cases.hard_ids[q, j].item())
            for j in range(cases.hard_ids.shape[1])
            if bool(cases.hard_mask[q, j].item())
        ]
        recalled = sum(1 for x in supports if x in retrieved)
        has_hard = any(x in retrieved for x in hards)
        any_ok = recalled > 0
        all_ok = recalled == len(supports)
        any_support += int(any_ok)
        all_support += int(all_ok)
        clean_answer_proxy += int(all_ok and not has_hard)
        support_frac_sum += recalled / max(1, len(supports))
        contamination += int(has_hard)

        fam = cases.family[q]
        stats = by_family.setdefault(
            fam,
            {"n": 0.0, "any_support_recall": 0.0, "all_support_recall": 0.0, "support_fraction": 0.0},
        )
        stats["n"] += 1.0
        stats["any_support_recall"] += float(any_ok)
        stats["all_support_recall"] += float(all_ok)
        stats["support_fraction"] += recalled / max(1, len(supports))

    for stats in by_family.values():
        n = max(1.0, stats["n"])
        stats["any_support_recall"] /= n
        stats["all_support_recall"] /= n
        stats["support_fraction"] /= n

    row: Dict[str, object] = {
        "method": method,
        "any_support_recall": any_support / num_queries,
        "all_support_recall": all_support / num_queries,
        "support_fraction": support_frac_sum / num_queries,
        "answer_proxy_acc": all_support / num_queries,
        "clean_answer_proxy_acc": clean_answer_proxy / num_queries,
        "hard_negative_contamination": contamination / num_queries,
        "by_family": by_family,
    }
    if risk_trigger_rate is not None:
        row["risk_trigger_rate"] = risk_trigger_rate
    return row


def run_method_once(
    method: str,
    *,
    query: torch.Tensor,
    blocks: torch.Tensor,
    block_lowrank: torch.Tensor,
    query_lowrank: torch.Tensor,
    lex_scores: torch.Tensor,
    local_start: int,
    budget_blocks: int,
    qabs_dims: int,
    risk_gap2: float,
    risk_min_top1: float,
    is_distributed: bool,
    world_size: int,
) -> Tuple[torch.Tensor, Optional[float]]:
    risk_rate: Optional[float] = None
    if method == "lexical":
        local_vals, local_ids = local_topk(lex_scores, budget_blocks, local_start)
    elif method == "qabs8":
        scores = qabs_scores(query, blocks, qabs_dims)
        local_vals, local_ids = local_topk(scores, budget_blocks, local_start)
    elif method == "svd64":
        scores = query_lowrank.matmul(block_lowrank.t())
        local_vals, local_ids = local_topk(scores, budget_blocks, local_start)
    elif method == "hybrid":
        sem = query_lowrank.matmul(block_lowrank.t())
        scores = 0.70 * lex_scores + 1.60 * sem
        local_vals, local_ids = local_topk(scores, budget_blocks, local_start)
    elif method == "risk_hybrid_lazy":
        risk_mask = compute_risk_mask_from_lexical(
            lex_scores,
            local_start,
            budget_blocks,
            risk_gap2,
            risk_min_top1,
            is_distributed,
            world_size,
        )
        risk_rate = float(risk_mask.float().mean().item())
        local_vals = torch.empty(
            query.shape[0],
            min(budget_blocks, lex_scores.shape[1]),
            device=query.device,
            dtype=lex_scores.dtype,
        )
        local_ids = torch.empty(query.shape[0], min(budget_blocks, lex_scores.shape[1]), device=query.device, dtype=torch.long)

        safe_idx = (~risk_mask).nonzero(as_tuple=False).flatten()
        risk_idx = risk_mask.nonzero(as_tuple=False).flatten()
        if safe_idx.numel() > 0:
            vals, ids = local_topk(lex_scores.index_select(0, safe_idx), budget_blocks, local_start)
            local_vals.index_copy_(0, safe_idx, vals)
            local_ids.index_copy_(0, safe_idx, ids)
        if risk_idx.numel() > 0:
            sem = query_lowrank.index_select(0, risk_idx).matmul(block_lowrank.t())
            hybrid = 0.70 * lex_scores.index_select(0, risk_idx) + 1.60 * sem
            vals, ids = local_topk(hybrid, budget_blocks, local_start)
            local_vals.index_copy_(0, risk_idx, vals)
            local_ids.index_copy_(0, risk_idx, ids)
    else:
        raise ValueError(f"Unknown method: {method}")

    _, final_ids = distributed_merge_topk(local_vals, local_ids, budget_blocks, is_distributed, world_size)
    return final_ids, risk_rate


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device, is_distributed = setup_distributed()
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    num_blocks = math.ceil(args.seq_tokens / args.block_tokens)
    budget_blocks = max(1, args.target_tokens // args.block_tokens)
    local_start, local_end = shard_bounds(num_blocks, rank, world_size)

    if rank == 0:
        print(
            json.dumps(
                {
                    "seq_tokens": args.seq_tokens,
                    "block_tokens": args.block_tokens,
                    "target_tokens": args.target_tokens,
                    "num_blocks": num_blocks,
                    "budget_blocks": budget_blocks,
                    "num_queries": args.num_queries,
                    "world_size": world_size,
                    "device": str(device),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    cases = make_synthetic_cases(
        num_blocks=num_blocks,
        num_queries=args.num_queries,
        dense_dim=args.dense_dim,
        hard_negatives=args.hard_negatives,
        seed=args.seed,
    )

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    block_embeddings = cases.block_embeddings[local_start:local_end].to(device=device, dtype=dtype, non_blocking=True)
    query_embeddings = cases.query_embeddings.to(device=device, dtype=dtype, non_blocking=True)
    lex_scores = make_lexical_scores(cases, local_start=local_start, local_end=local_end, seed=args.seed, device=device).to(dtype)

    rank_dim = min(args.svd_rank, args.dense_dim)
    block_lowrank = block_embeddings[:, :rank_dim].contiguous()
    query_lowrank = query_embeddings[:, :rank_dim].contiguous()

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    method_results: List[Dict[str, object]] = []
    family_rows: List[Dict[str, object]] = []

    for method in methods:
        for _ in range(args.warmup):
            barrier_and_sync(device, is_distributed)
            run_method_once(
                method,
                query=query_embeddings,
                blocks=block_embeddings,
                block_lowrank=block_lowrank,
                query_lowrank=query_lowrank,
                lex_scores=lex_scores,
                local_start=local_start,
                budget_blocks=budget_blocks,
                qabs_dims=args.qabs_dims,
                risk_gap2=args.risk_gap2,
                risk_min_top1=args.risk_min_top1,
                is_distributed=is_distributed,
                world_size=world_size,
            )

        elapsed: List[float] = []
        final_ids: Optional[torch.Tensor] = None
        risk_rate: Optional[float] = None
        for _ in range(args.repeats):
            barrier_and_sync(device, is_distributed)
            start = time.perf_counter()
            final_ids, risk_rate = run_method_once(
                method,
                query=query_embeddings,
                blocks=block_embeddings,
                block_lowrank=block_lowrank,
                query_lowrank=query_lowrank,
                lex_scores=lex_scores,
                local_start=local_start,
                budget_blocks=budget_blocks,
                qabs_dims=args.qabs_dims,
                risk_gap2=args.risk_gap2,
                risk_min_top1=args.risk_min_top1,
                is_distributed=is_distributed,
                world_size=world_size,
            )
            barrier_and_sync(device, is_distributed)
            elapsed.append(time.perf_counter() - start)

        if rank == 0 and final_ids is not None:
            eval_row = evaluate_retrieval(final_ids, cases, method, risk_rate)
            elapsed_mean = statistics.mean(elapsed)
            elapsed_std = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
            flat_row = {
                "method": method,
                "elapsed_s_mean": elapsed_mean,
                "elapsed_s_std": elapsed_std,
                "queries_per_s": args.num_queries / elapsed_mean,
                "blocks_per_s": (args.num_queries * num_blocks) / elapsed_mean,
                "any_support_recall": eval_row["any_support_recall"],
                "all_support_recall": eval_row["all_support_recall"],
                "support_fraction": eval_row["support_fraction"],
                "answer_proxy_acc": eval_row["answer_proxy_acc"],
                "clean_answer_proxy_acc": eval_row["clean_answer_proxy_acc"],
                "hard_negative_contamination": eval_row["hard_negative_contamination"],
            }
            if "risk_trigger_rate" in eval_row:
                flat_row["risk_trigger_rate"] = eval_row["risk_trigger_rate"]
            method_results.append(flat_row)

            for fam, stats in eval_row["by_family"].items():
                family_rows.append(
                    {
                        "method": method,
                        "family": fam,
                        "n": stats["n"],
                        "any_support_recall": stats["any_support_recall"],
                        "all_support_recall": stats["all_support_recall"],
                        "support_fraction": stats["support_fraction"],
                    }
                )

            print(json.dumps(flat_row, ensure_ascii=False), flush=True)

    if rank == 0:
        config = {
            "seq_tokens": args.seq_tokens,
            "block_tokens": args.block_tokens,
            "target_tokens": args.target_tokens,
            "num_blocks": num_blocks,
            "budget_blocks": budget_blocks,
            "num_queries": args.num_queries,
            "dense_dim": args.dense_dim,
            "svd_rank": rank_dim,
            "qabs_dims": args.qabs_dims,
            "world_size": world_size,
            "device": str(device),
            "methods": methods,
            "seed": args.seed,
        }
        summary = {"config": config, "method_results": method_results}
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(
            out_dir / "method_results.csv",
            method_results,
            [
                "method",
                "elapsed_s_mean",
                "elapsed_s_std",
                "queries_per_s",
                "blocks_per_s",
                "any_support_recall",
                "all_support_recall",
                "support_fraction",
                "answer_proxy_acc",
                "clean_answer_proxy_acc",
                "hard_negative_contamination",
                "risk_trigger_rate",
            ],
        )
        write_csv(
            out_dir / "family_results.csv",
            family_rows,
            ["method", "family", "n", "any_support_recall", "all_support_recall", "support_fraction"],
        )

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
