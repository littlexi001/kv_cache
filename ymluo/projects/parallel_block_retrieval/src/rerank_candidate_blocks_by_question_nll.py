from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rerank_records_by_question_nll import resolve_dtype
from run_hybrid_block_retrieval import candidate_ids_for_query, rrf_order
from run_lexical_block_retrieval import (
    evaluate_selection,
    group_for_context,
    read_jsonl,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank 10M-corpus block candidates by answer-free Qwen question NLL."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--lexical_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--global_candidates", type=int, default=782)
    parser.add_argument("--top_record_candidates", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--base_retrieval_results")
    parser.add_argument("--base_method", default="deep_ql_record39_svd32")
    parser.add_argument("--base_quotas", default="19,29,33,35")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("block question-NLL reranking requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, world_size, torch.device("cuda", local_rank)


@torch.inference_mode()
def batched_question_nll(
    model: AutoModelForCausalLM,
    block_ids: torch.Tensor,
    prefix_ids: torch.Tensor,
    question_ids: torch.Tensor,
) -> torch.Tensor:
    batch = block_ids.shape[0]
    prefix = prefix_ids[None].expand(batch, -1)
    question = question_ids[None].expand(batch, -1)
    input_ids = torch.cat([block_ids, prefix, question], dim=1)
    prompt_tokens = block_ids.shape[1] + prefix_ids.numel()
    hidden = model.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        return_dict=True,
    ).last_hidden_state[:, prompt_tokens - 1 : -1]
    logits = model.lm_head(hidden).float()
    targets = question[:, None].expand(-1, logits.shape[1], -1)
    token_losses = F.cross_entropy(
        logits.transpose(1, 2), targets[:, 0], reduction="none"
    )
    return token_losses.mean(dim=1)


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / max(float(values.std()), 1.0e-6)


def quota_combine(
    primary: list[int], secondary: list[int], primary_quota: int, target: int
) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for source, quota in ((primary, primary_quota), (secondary, target - primary_quota)):
        added = 0
        for block_id in source:
            if added >= quota or len(output) >= target:
                break
            if block_id not in seen:
                output.append(block_id)
                seen.add(block_id)
                added += 1
    for source in (primary, secondary):
        for block_id in source:
            if len(output) >= target:
                break
            if block_id not in seen:
                output.append(block_id)
                seen.add(block_id)
    return output


def load_base_rankings(path: str | None, method: str) -> dict[int, list[int]]:
    if not path:
        return {}
    output: dict[int, list[int]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == method:
                output[int(row["query_id"])] = [
                    int(item) for item in json.loads(row["ranked_block_ids"])
                ]
    if not output:
        raise ValueError(f"base method {method} not found")
    return output


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    corpus_dir = Path(args.corpus_dir)
    lexical_dir = Path(args.lexical_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    records = read_jsonl(corpus_dir / "records.jsonl")
    block_scores = np.load(lexical_dir / "block_scores.npy", mmap_mode="r")
    record_scores = np.load(lexical_dir / "record_scores.npy", mmap_mode="r")
    base_rankings = load_base_rankings(args.base_retrieval_results, args.base_method)
    base_quotas = sorted(
        {int(item) for item in args.base_quotas.split(",") if item.strip()}
    )

    block_to_record = np.empty(blocks.shape[0], dtype=np.int32)
    source_record_by_start: dict[int, int] = {}
    for record_id, record in enumerate(records):
        start = int(record["block_start"])
        end = start + int(record["block_count"])
        block_to_record[start:end] = record_id
        source_record_by_start[start] = record_id

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    prefix_ids = torch.tensor(
        tokenizer("\nQuestion:", add_special_tokens=False)["input_ids"],
        dtype=torch.long,
        device=device,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    local_rows: list[dict[str, Any]] = []
    local_diagnostics: list[dict[str, Any]] = []
    local_candidate_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for query_index in range(rank, len(queries), world_size):
        query = queries[query_index]
        candidate_ids, ranked_records = candidate_ids_for_query(
            block_scores=block_scores[query_index],
            record_scores=record_scores[query_index],
            records=records,
            global_candidates=args.global_candidates,
            top_record_candidates=args.top_record_candidates,
        )
        question_ids = torch.tensor(
            tokenizer(" " + str(query["question"]), add_special_tokens=False)["input_ids"],
            dtype=torch.long,
            device=device,
        )
        losses: list[np.ndarray] = []
        for start in range(0, len(candidate_ids), args.batch_size):
            batch_ids = candidate_ids[start : start + args.batch_size]
            block_array = np.array(blocks[batch_ids], dtype=np.int64, copy=True)
            block_tensor = torch.from_numpy(block_array).long().to(device)
            losses.append(
                batched_question_nll(
                    model, block_tensor, prefix_ids, question_ids
                ).float().cpu().numpy()
            )
        nll = np.concatenate(losses)
        candidate_array = np.asarray(candidate_ids, dtype=np.int64)
        nll_order = candidate_array[
            np.lexsort((candidate_array, nll))
        ].tolist()
        lexical = np.asarray(block_scores[query_index, candidate_ids], dtype=np.float32)
        rrf_ranked = rrf_order(candidate_ids, lexical, -nll)
        fused = zscore(lexical) + zscore(-nll)
        fused_ranked = candidate_array[
            np.lexsort((candidate_array, -fused))
        ].tolist()
        source_record = source_record_by_start[int(query["block_start"])]
        predicted_record = int(ranked_records[0])
        methods = {
            "block_question_nll": nll_order[: args.target_blocks],
            "block_question_nll_bm25_rrf": rrf_ranked[: args.target_blocks],
            "block_question_nll_bm25_z": fused_ranked[: args.target_blocks],
        }
        candidate_position = {
            block_id: position for position, block_id in enumerate(candidate_ids)
        }
        routed_records = list(ranked_records[: args.top_record_candidates])
        aggregate_top3: list[float] = []
        aggregate_top8: list[float] = []
        record_blockql_rankings: dict[int, list[int]] = {}
        for record_id in routed_records:
            record = records[record_id]
            record_start = int(record["block_start"])
            record_ids = list(
                range(record_start, record_start + int(record["block_count"]))
            )
            record_ids.sort(
                key=lambda block_id: (
                    float(nll[candidate_position[block_id]]),
                    block_id,
                )
            )
            record_blockql_rankings[record_id] = record_ids
            ordered_nll = np.asarray(
                [nll[candidate_position[block_id]] for block_id in record_ids],
                dtype=np.float32,
            )
            aggregate_top3.append(float(ordered_nll[: min(3, len(ordered_nll))].mean()))
            aggregate_top8.append(float(ordered_nll[: min(8, len(ordered_nll))].mean()))

        record_bm25 = np.asarray(
            [record_scores[query_index, record_id] for record_id in routed_records],
            dtype=np.float32,
        )
        for aggregate_name, aggregate_values in (
            ("top3", np.asarray(aggregate_top3, dtype=np.float32)),
            ("top8", np.asarray(aggregate_top8, dtype=np.float32)),
        ):
            record_ids_array = np.asarray(routed_records, dtype=np.int64)
            ql_order = record_ids_array[
                np.lexsort((record_ids_array, aggregate_values))
            ].tolist()
            record_fused = zscore(record_bm25) + zscore(-aggregate_values)
            fused_record_order = record_ids_array[
                np.lexsort((record_ids_array, -record_fused))
            ].tolist()
            for route_name, route_order in (
                ("ql", ql_order),
                ("bm25_z", fused_record_order),
            ):
                chosen_record = route_order[0]
                methods[f"blockql_record_{aggregate_name}_{route_name}"] = quota_combine(
                    record_blockql_rankings[chosen_record],
                    nll_order,
                    min(
                        args.target_blocks,
                        len(record_blockql_rankings[chosen_record]),
                    ),
                    args.target_blocks,
                )
        if query_index in base_rankings:
            for quota in base_quotas:
                methods[f"base{quota}_blockql{args.target_blocks - quota}"] = quota_combine(
                    base_rankings[query_index], nll_order, quota, args.target_blocks
                )
        for method, ranked in methods.items():
            local_rows.append(
                evaluate_selection(
                    method=method,
                    query=query,
                    ranked_ids=ranked,
                    context_ids=group_for_context(ranked, block_to_record),
                    predicted_record=predicted_record,
                    source_record=source_record,
                    record_margin=0.0,
                )
            )
        gold = {int(item) for item in query.get("gold_block_ids", [])}
        local_diagnostics.append(
            {
                "query_id": query_index,
                "dataset": query["dataset"],
                "candidate_blocks": len(candidate_ids),
                "candidate_any_oracle": float(bool(gold & set(candidate_ids))),
                "candidate_all_oracle": float(gold <= set(candidate_ids)),
                "mean_block_question_nll": float(nll.mean()),
                "min_block_question_nll": float(nll.min()),
            }
        )
        block_rank = np.empty(len(candidate_ids), dtype=np.int64)
        block_rank[np.lexsort((candidate_array, -lexical))] = np.arange(len(candidate_ids))
        nll_rank = np.empty(len(candidate_ids), dtype=np.int64)
        nll_rank[np.lexsort((candidate_array, nll))] = np.arange(len(candidate_ids))
        gold = {int(item) for item in query.get("gold_block_ids", [])}
        source_record = source_record_by_start[int(query["block_start"])]
        record_rank_by_id = {
            int(record_id): record_rank
            for record_rank, record_id in enumerate(ranked_records)
        }
        for position, block_id in enumerate(candidate_ids):
            record_id = int(block_to_record[block_id])
            record = records[record_id]
            local_candidate_rows.append(
                {
                    "query_id": query_index,
                    "dataset": query["dataset"],
                    "block_id": block_id,
                    "record_id": record_id,
                    "question_nll": float(nll[position]),
                    "block_bm25_score": float(lexical[position]),
                    "block_bm25_rank": int(block_rank[position]) + 1,
                    "block_nll_rank": int(nll_rank[position]) + 1,
                    "record_bm25_score": float(record_scores[query_index, record_id]),
                    "record_bm25_rank": int(record_rank_by_id[record_id]) + 1,
                    "block_position_fraction": (
                        (block_id - int(record["block_start"]))
                        / max(int(record["block_count"]) - 1, 1)
                    ),
                    "record_blocks": int(record["block_count"]),
                    "is_gold": float(block_id in gold),
                    "is_source_record": float(record_id == source_record),
                }
            )
        print(
            json.dumps(
                {"rank": rank, "query_id": query_index, "candidates": len(candidate_ids)}
            ),
            flush=True,
        )

    gathered_rows: list[list[dict[str, Any]] | None] = [None] * world_size
    gathered_diagnostics: list[list[dict[str, Any]] | None] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered_rows, local_rows)
        dist.all_gather_object(gathered_diagnostics, local_diagnostics)
    else:
        gathered_rows[0] = local_rows
        gathered_diagnostics[0] = local_diagnostics
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / f"candidate_scores_rank{rank:03d}.csv",
        local_candidate_rows,
        list(local_candidate_rows[0]),
    )
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        rows = [row for part in gathered_rows if part for row in part]
        rows.sort(key=lambda row: (row["method"], int(row["query_id"])))
        diagnostics = [row for part in gathered_diagnostics if part for row in part]
        diagnostics.sort(key=lambda row: int(row["query_id"]))
        write_csv(output_dir / "query_results.csv", rows, list(rows[0]))
        write_csv(
            output_dir / "candidate_diagnostics.csv",
            diagnostics,
            list(diagnostics[0]),
        )
        summaries = []
        for method in sorted({str(row["method"]) for row in rows}):
            group = [row for row in rows if row["method"] == method]
            summaries.append(
                {
                    "method": method,
                    "queries": len(group),
                    "answer_block_recall": statistics.fmean(
                        float(row["answer_block_recall"]) for row in group
                    ),
                    "answer_block_mrr": statistics.fmean(
                        float(row["answer_block_mrr"]) for row in group
                    ),
                    "source_record_recall": statistics.fmean(
                        float(row["source_record_recall"]) for row in group
                    ),
                }
            )
        write_csv(output_dir / "method_summary.csv", summaries, list(summaries[0]))
        summary = {
            "source": "answer-free Qwen block question-likelihood reranking",
            "contains_synthetic_vectors": False,
            "queries": len(queries),
            "target_blocks": args.target_blocks,
            "global_candidates": args.global_candidates,
            "top_record_candidates": args.top_record_candidates,
            "mean_candidate_blocks": statistics.fmean(
                int(row["candidate_blocks"]) for row in diagnostics
            ),
            "candidate_any_oracle": statistics.fmean(
                float(row["candidate_any_oracle"]) for row in diagnostics
            ),
            "candidate_all_oracle": statistics.fmean(
                float(row["candidate_all_oracle"]) for row in diagnostics
            ),
            "world_size": world_size,
            "wall_seconds": elapsed,
            "methods": summaries,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
