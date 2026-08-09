from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from profile_real_qk import QKCapture, read_jsonl, resolve_dtype
from run_global_bridge_controller_single import (
    extract_novel_entity_from_memory,
    greedy_chat,
    lexical_window,
    memory_text,
    normalize_bridge_query,
    parse_search_action,
)
from run_global_dynamic_svd_kv_single import capture_query_ids, distributed_retrieve
from run_dynamic_kv_multisample import token_f1
from run_iterative_condition_retrieval import BM25Index
from run_lexical_block_retrieval import decode_blocks
from run_real_qk_retrieval import load_index, setup_distributed
from run_single_query_dynamic_kv_generation import answer_hit
from verified_step_state import (
    AtomicFact,
    fact_chain_connects,
    parse_step_action,
    verified_facts,
    verified_step_prompt,
)


MULTIHOP_DATASETS = ("2wikimqa", "hotpotqa", "musique")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen-policy batch evaluation of global Q/K retrieval followed by "
            "model-generated bridge searches."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--datasets", default=",".join(MULTIHOP_DATASETS))
    parser.add_argument("--query_ids", default="")
    parser.add_argument("--exclude_query_ids", default="0,6")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument(
        "--initial_retriever",
        choices=["qk", "question_bm25", "bm25_record_qk"],
        default="qk",
    )
    parser.add_argument("--svd_rank", type=int, default=32)
    parser.add_argument("--candidate_blocks", type=int, default=512)
    parser.add_argument("--target_blocks", type=int, default=3)
    parser.add_argument("--query_tokens", type=int, default=16)
    parser.add_argument("--dynamic_query_tokens", type=int, default=3)
    parser.add_argument("--exclude_block_prefix_tokens", type=int, default=0)
    parser.add_argument("--block_chunk", type=int, default=256)
    parser.add_argument("--hop1_block", type=int, default=-1)
    parser.add_argument("--anchor_initial_query", action="store_true")
    parser.add_argument("--coarse_reserve_blocks", type=int, default=0)
    parser.add_argument("--bridge_max_new_tokens", type=int, default=32)
    parser.add_argument("--answer_max_new_tokens", type=int, default=64)
    parser.add_argument("--search_hops", type=int, default=1)
    parser.add_argument(
        "--controller_mode",
        choices=["forced_search", "verified_state"],
        default="forced_search",
    )
    parser.add_argument(
        "--bridge_channels",
        choices=["model_only", "model2_det1"],
        default="model_only",
    )
    parser.add_argument(
        "--final_prompt_mode",
        choices=["bound_focus", "evidence_only"],
        default="bound_focus",
    )
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    return parser.parse_args()


def parse_id_set(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def select_queries(
    queries: Sequence[dict[str, Any]],
    *,
    datasets: set[str],
    query_ids: set[int],
    excluded_ids: set[int],
    max_queries: int,
) -> list[dict[str, Any]]:
    selected = [
        item
        for item in queries
        if (not datasets or str(item["dataset"]) in datasets)
        and (not query_ids or int(item["query_id"]) in query_ids)
        and int(item["query_id"]) not in excluded_ids
    ]
    selected.sort(key=lambda item: int(item["query_id"]))
    if max_queries > 0:
        selected = selected[:max_queries]
    return selected


def ranked_block_ids(scores: np.ndarray) -> list[int]:
    ids = np.arange(scores.shape[0], dtype=np.int64)
    return [int(item) for item in np.lexsort((ids, -scores)).tolist()]


def first_rank_in_record(ranked: Sequence[int], block_start: int, block_count: int) -> int:
    block_end = block_start + block_count
    return min(
        index + 1
        for index, block_id in enumerate(ranked)
        if block_start <= int(block_id) < block_end
    )


def optional_rank_in_record(
    ranked: Sequence[int], block_start: int, block_count: int
) -> int | None:
    block_end = block_start + block_count
    return next(
        (
            index + 1
            for index, block_id in enumerate(ranked)
            if block_start <= int(block_id) < block_end
        ),
        None,
    )


def gold_ranks(ranked: Sequence[int], gold_block_ids: Sequence[int]) -> dict[str, int]:
    positions = {int(block_id): index + 1 for index, block_id in enumerate(ranked)}
    return {str(block_id): positions[int(block_id)] for block_id in gold_block_ids}


def selected_hits(
    selected: Sequence[int],
    *,
    block_start: int,
    block_count: int,
    gold_block_ids: Sequence[int],
) -> tuple[bool, bool]:
    block_end = block_start + block_count
    return (
        any(block_start <= int(item) < block_end for item in selected),
        any(int(item) in gold_block_ids for item in selected),
    )


def record_range_for_block(
    records: Sequence[dict[str, Any]], block_id: int
) -> tuple[int, int]:
    for record in records:
        start = int(record["block_start"])
        end = start + int(record["block_count"])
        if start <= block_id < end:
            return start, end
    raise ValueError(f"block {block_id} is not covered by records.jsonl")


def merge_channel_selections(
    primary: Sequence[int],
    secondary: Sequence[int],
    *,
    primary_quota: int,
    target_blocks: int,
) -> list[int]:
    selected: list[int] = []
    for block_id in list(primary)[:primary_quota]:
        if int(block_id) not in selected:
            selected.append(int(block_id))
    for block_id in secondary:
        if int(block_id) not in selected:
            selected.append(int(block_id))
        if len(selected) >= target_blocks:
            return selected
    for block_id in list(primary)[primary_quota:]:
        if int(block_id) not in selected:
            selected.append(int(block_id))
        if len(selected) >= target_blocks:
            break
    return selected


def mean_field(rows: Sequence[dict[str, Any]], field: str) -> float:
    values = [float(item[field]) for item in rows if item.get(field) is not None]
    return statistics.fmean(values) if values else 0.0


def median_field(rows: Sequence[dict[str, Any]], field: str) -> float:
    values = [float(item[field]) for item in rows if item.get(field) is not None]
    return statistics.median(values) if values else 0.0


def percentile_field(
    rows: Sequence[dict[str, Any]], field: str, percentile: float
) -> float:
    values = [float(item[field]) for item in rows if item.get(field) is not None]
    return float(np.percentile(values, percentile)) if values else 0.0


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in rows if item.get("controller_error") is None]
    qk_valid = [
        item
        for item in valid
        if item.get("initial_retriever") in {"qk", "bm25_record_qk"}
    ]

    def rate(field: str, source: Sequence[dict[str, Any]] = valid) -> float:
        values = [bool(item[field]) for item in source if item.get(field) is not None]
        return sum(values) / len(values) if values else 0.0

    def all_query_rate(field: str) -> float:
        return (
            sum(bool(item.get(field, False)) for item in rows) / len(rows)
            if rows
            else 0.0
        )

    return {
        "queries": len(rows),
        "valid_queries": len(valid),
        "controller_errors": len(rows) - len(valid),
        "controller_success_rate": len(valid) / len(rows) if rows else 0.0,
        "question_bm25_record_recall_at_3": rate("question_bm25_record_hit"),
        "question_bm25_gold_recall_at_3": rate("question_bm25_gold_hit"),
        "initial_record_recall_at_3": rate("hop1_record_hit"),
        "initial_gold_recall_at_3": rate("hop1_gold_hit"),
        "qk_record_recall_at_3": rate("hop1_record_hit", qk_valid),
        "qk_gold_recall_at_3": rate("hop1_gold_hit", qk_valid),
        "qk_record_recall_at_512": rate(
            "hop1_record_in_coarse_candidates", qk_valid
        ),
        "any_bridge_record_recall_at_3": rate("any_search_record_hit"),
        "any_bridge_gold_recall_at_3": rate("any_search_gold_hit"),
        "final_bridge_gold_recall_at_3": rate("final_search_gold_hit"),
        "answer_hit_rate": rate("answer_hit"),
        "mean_answer_f1": mean_field(valid, "answer_f1"),
        "all_query_any_bridge_gold_recall_at_3": all_query_rate(
            "any_search_gold_hit"
        ),
        "all_query_answer_hit_rate": all_query_rate("answer_hit"),
        "all_query_mean_answer_f1": mean_field(rows, "answer_f1"),
        "verified_early_final_rate": rate("verified_early_final"),
        "mean_verifier_rejections": mean_field(valid, "verifier_rejections"),
        "mean_qk_capture_seconds": mean_field(qk_valid, "qk_capture_seconds"),
        "mean_qk_retrieval_seconds": mean_field(qk_valid, "qk_retrieval_seconds"),
        "mean_initial_retrieval_seconds": mean_field(
            valid, "initial_retrieval_seconds"
        ),
        "mean_bridge_bm25_seconds": mean_field(valid, "bridge_bm25_seconds"),
        "mean_bridge_generation_seconds": mean_field(
            valid, "bridge_generation_seconds"
        ),
        "mean_answer_generation_seconds": mean_field(
            valid, "answer_generation_seconds"
        ),
        "mean_online_seconds": mean_field(valid, "online_seconds"),
        "median_online_seconds": median_field(valid, "online_seconds"),
        "p95_online_seconds": percentile_field(valid, "online_seconds", 95.0),
        "median_qk_retrieval_seconds": median_field(
            qk_valid, "qk_retrieval_seconds"
        ),
        "p95_qk_retrieval_seconds": percentile_field(
            qk_valid, "qk_retrieval_seconds", 95.0
        ),
    }


def summarize_all(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    datasets = sorted({str(item["dataset"]) for item in rows})
    return {
        "overall": summarize_rows(rows),
        "by_dataset": {
            dataset: summarize_rows(
                [item for item in rows if str(item["dataset"]) == dataset]
            )
            for dataset in datasets
        },
    }


def bridge_prompt(
    memory: str,
    question: str,
    previous_queries: Sequence[str],
) -> str:
    previous_text = (
        "\nPrevious lookup queries: " + " | ".join(previous_queries)
        if previous_queries
        else ""
    )
    return (
        "Memory:\n"
        f"{memory}\n\n"
        f"Question: {question}"
        f"{previous_text}\n"
        "Perform one more lookup step before answering. Resolve one relational "
        "description, such as a spouse, parent, child, author, director, organization, "
        "or location, to the exact proper name stated in the current memory. Prefer a "
        "proper name that has not already been searched. Output exactly one line in the "
        "form `SEARCH: <exact proper name> <still-unresolved relation>`. Do not answer "
        "the original question and do not explain."
    )


def final_prompt(
    memory: str,
    question: str,
    previous_queries: Sequence[str],
    focus_entity: str,
) -> str:
    return (
        "Memory:\n"
        f"{memory}\n\n"
        f"Question: {question}\n"
        f"Intermediate lookups used: {' | '.join(previous_queries)}\n"
        f"Focus entity for the final unresolved relation: {focus_entity}\n"
        "Answer the original question using the memory. Match paraphrased relation evidence: "
        "for example, location may be expressed with born, died, buried, interred, or "
        "entombed, and authorship may be expressed with wrote, authored, or created. Use the "
        "shortest directly supported entity or location phrase. The evidence must describe "
        "the focus entity, not another person or organization mentioned nearby. Ignore "
        "locations whose grammatical subject is not the focus entity. Output only the "
        "concise final answer."
    )


def evidence_only_final_prompt(memory: str, question: str) -> str:
    return (
        "Memory:\n"
        f"{memory}\n\n"
        f"Question: {question}\n"
        "Answer using only directly supported evidence in Memory. Resolve every relation "
        "in the Question, but do not assume that any previously proposed intermediate "
        "entity was correct. Output only the shortest supported answer. If Memory is "
        "insufficient, output UNKNOWN."
    )


def main() -> None:
    args = parse_args()
    if args.search_hops < 1:
        raise ValueError("search_hops must be at least one")
    rank, world_size, _local_rank, device = setup_distributed()
    requires_qk_index = args.initial_retriever in {"qk", "bm25_record_qk"}
    if requires_qk_index and world_size <= 1:
        raise ValueError("This experiment requires distributed index shards")

    corpus_dir = Path(args.corpus_dir)
    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    summary = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    raw_keys = None
    svd_keys = None
    local_block_ids = None
    basis = None
    if requires_qk_index:
        raw_keys, svd_keys, local_block_ids, _ = load_index(
            profile_dir, summary, rank, world_size, device
        )
        basis_payload = torch.load(
            profile_dir / "basis.pt", map_location="cpu", weights_only=False
        )
        basis = basis_payload["basis"].to(device=device, dtype=torch.float16)
    pair_specs = [dict(item) for item in summary["pair_specs"]]
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    records = read_jsonl(corpus_dir / "records.jsonl")
    queries = select_queries(
        read_jsonl(corpus_dir / "queries.jsonl"),
        datasets={item.strip() for item in args.datasets.split(",") if item.strip()},
        query_ids=parse_id_set(args.query_ids),
        excluded_ids=parse_id_set(args.exclude_query_ids),
        max_queries=args.max_queries,
    )
    if not queries:
        raise ValueError("no queries matched the requested batch")

    model = None
    tokenizer = None
    capture = None
    block_texts = None
    bm25 = None
    bm25_build_seconds = None
    question_bm25_seconds = None
    question_bm25_scores = None
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=resolve_dtype(args.dtype),
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        if requires_qk_index:
            capture = QKCapture(
                model, sorted({int(item["layer"]) for item in pair_specs})
            )
        bm25_started = time.perf_counter()
        block_texts = decode_blocks(tokenizer, blocks)
        bm25 = BM25Index(block_texts, min_df=1, max_df=1.0, k1=1.2, b=0.75)
        bm25_build_seconds = time.perf_counter() - bm25_started
        question_bm25_started = time.perf_counter()
        question_bm25_scores = bm25.score([str(item["question"]) for item in queries])
        question_bm25_seconds = time.perf_counter() - question_bm25_started

    results: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for query_offset, query in enumerate(queries):
        gold_block_ids = [int(item) for item in query["gold_block_ids"]]
        block_start = int(query["block_start"])
        block_count = int(query["block_count"])
        question_query = None
        qk_capture_seconds = None
        question_ranked_before_retrieval = (
            ranked_block_ids(question_bm25_scores[query_offset])
            if rank == 0
            else None
        )
        routed_record_range = (
            record_range_for_block(records, question_ranked_before_retrieval[0])
            if rank == 0 and args.initial_retriever == "bm25_record_qk"
            else None
        )
        if rank == 0 and args.initial_retriever in {"qk", "bm25_record_qk"}:
            question_ids = tokenizer(
                str(query["question"]), add_special_tokens=False
            )["input_ids"]
            torch.cuda.synchronize(device)
            capture_started = time.perf_counter()
            question_query = capture_query_ids(
                model, capture, pair_specs, question_ids, args.query_tokens, device
            )
            torch.cuda.synchronize(device)
            qk_capture_seconds = time.perf_counter() - capture_started

        if args.initial_retriever in {"qk", "bm25_record_qk"}:
            hop1_selected, hop1_event = distributed_retrieve(
                query=question_query,
                basis=basis,
                raw_keys=raw_keys,
                svd_keys=svd_keys,
                local_block_ids=local_block_ids,
                args=args,
                rank=rank,
                world_size=world_size,
                device=device,
                gold_block_id=gold_block_ids[0],
                allowed_block_range=routed_record_range,
            )
        else:
            hop1_selected = (
                ranked_block_ids(question_bm25_scores[query_offset])[: args.target_blocks]
                if rank == 0
                else []
            )
            hop1_event = None
        if rank != 0:
            continue

        hop1_record_hit, hop1_gold_hit = selected_hits(
            hop1_selected,
            block_start=block_start,
            block_count=block_count,
            gold_block_ids=gold_block_ids,
        )
        question_ranked = question_ranked_before_retrieval
        question_selected = question_ranked[: args.target_blocks]
        question_record_hit, question_gold_hit = selected_hits(
            question_selected,
            block_start=block_start,
            block_count=block_count,
            gold_block_ids=gold_block_ids,
        )
        coarse_candidates = (
            [int(item) for item in hop1_event["coarse_candidate_ids"]]
            if hop1_event is not None
            else []
        )
        exact_candidates = (
            [int(item) for item in hop1_event["exact_ranked_candidate_ids"]]
            if hop1_event is not None
            else []
        )
        initial_retrieval_seconds = (
            float(hop1_event["retrieval_seconds"])
            + (
                float(question_bm25_seconds) / len(queries)
                if args.initial_retriever == "bm25_record_qk"
                else 0.0
            )
            if hop1_event is not None
            else float(question_bm25_seconds) / len(queries)
        )
        row: dict[str, Any] = {
            "query_id": int(query["query_id"]),
            "dataset": str(query["dataset"]),
            "question": str(query["question"]),
            "answers": list(query["answers"]),
            "block_start": block_start,
            "block_count": block_count,
            "gold_block_ids": gold_block_ids,
            "question_bm25_selected": question_selected,
            "question_bm25_record_hit": question_record_hit,
            "question_bm25_gold_hit": question_gold_hit,
            "question_bm25_record_rank": first_rank_in_record(
                question_ranked, block_start, block_count
            ),
            "question_bm25_gold_ranks": gold_ranks(question_ranked, gold_block_ids),
            "hop1_selected": hop1_selected,
            "hop1_record_hit": hop1_record_hit,
            "hop1_gold_hit": hop1_gold_hit,
            "hop1_record_in_coarse_candidates": (
                optional_rank_in_record(coarse_candidates, block_start, block_count)
                is not None
                if hop1_event is not None
                else None
            ),
            "hop1_record_coarse_best_rank": optional_rank_in_record(
                coarse_candidates, block_start, block_count
            ),
            "hop1_record_exact_best_rank": optional_rank_in_record(
                exact_candidates, block_start, block_count
            ),
            "qk_capture_seconds": qk_capture_seconds,
            "qk_retrieval_seconds": (
                float(hop1_event["retrieval_seconds"])
                if hop1_event is not None
                else None
            ),
            "initial_retrieval_seconds": initial_retrieval_seconds,
            "routed_record_block_range": (
                list(routed_record_range) if routed_record_range is not None else None
            ),
            "search_trace": [],
            "controller_error": None,
            "selection_uses_gold": False,
            "initial_retriever": args.initial_retriever,
            "controller_mode": args.controller_mode,
        }

        current_selected = list(hop1_selected)
        current_memory = memory_text(tokenizer, blocks, current_selected)
        previous_queries: list[str] = []
        bridge_generation_seconds = 0.0
        bridge_bm25_seconds = 0.0
        verified_final_answer = None
        fact_ledger: list[AtomicFact] = []
        try:
            for search_hop in range(args.search_hops):
                action_prompt = (
                    verified_step_prompt(
                        current_memory,
                        str(query["question"]),
                        previous_queries,
                        fact_ledger,
                    )
                    if args.controller_mode == "verified_state"
                    else bridge_prompt(
                        current_memory, str(query["question"]), previous_queries
                    )
                )
                output, generated_tokens, generation_seconds = greedy_chat(
                    model,
                    tokenizer,
                    action_prompt,
                    (
                        max(args.bridge_max_new_tokens, 64)
                        if args.controller_mode == "verified_state"
                        else args.bridge_max_new_tokens
                    ),
                    device,
                )
                bridge_generation_seconds += generation_seconds
                action_facts = []
                verified_fact_count = 0
                action_kind = "search"
                final_verified = False
                verifier_rejected = False
                if args.controller_mode == "verified_state":
                    action = parse_step_action(output)
                    action_facts = [
                        {
                            "subject": fact.subject,
                            "relation": fact.relation,
                            "object": fact.object,
                            "evidence": fact.evidence,
                        }
                        for fact in action.facts
                    ]
                    supported_action_facts = verified_facts(
                        action, current_memory, require_evidence=True
                    )
                    verified_fact_count = len(supported_action_facts)
                    candidate_ledger = list(fact_ledger)
                    for fact in supported_action_facts:
                        if fact not in candidate_ledger:
                            candidate_ledger.append(fact)
                    action_kind = action.kind
                    if action.kind == "final":
                        final_verified = (
                            len(supported_action_facts) == len(action.facts)
                            and fact_chain_connects(
                                candidate_ledger,
                                action.value,
                                str(query["question"]),
                            )
                        )
                        if final_verified:
                            fact_ledger = candidate_ledger
                            record_hit, gold_hit = selected_hits(
                                current_selected,
                                block_start=block_start,
                                block_count=block_count,
                                gold_block_ids=gold_block_ids,
                            )
                            row["search_trace"].append(
                                {
                                    "search_hop": search_hop + 1,
                                    "model_output": output,
                                    "action_kind": "final",
                                    "action_facts": action_facts,
                                    "verified_fact_count": verified_fact_count,
                                    "fact_ledger_size": len(fact_ledger),
                                    "final_verified": True,
                                    "verifier_rejected": False,
                                    "generation_tokens": generated_tokens,
                                    "generation_seconds": generation_seconds,
                                    "bm25_seconds": 0.0,
                                    "selected_blocks": list(current_selected),
                                    "record_hit": record_hit,
                                    "gold_hit": gold_hit,
                                }
                            )
                            verified_final_answer = action.value
                            break
                        verifier_rejected = True
                    fact_ledger = candidate_ledger
                    raw_query = action.value
                else:
                    raw_query = parse_search_action(output)
                exclusion_text = str(query["question"]) + " " + " ".join(
                    previous_queries
                )
                normalized_query = normalize_bridge_query(raw_query, exclusion_text)
                fallback_entity = None
                if search_hop > 0 and normalized_query == raw_query:
                    fallback_entity = extract_novel_entity_from_memory(
                        current_memory, exclusion_text
                    )
                    if fallback_entity is not None:
                        normalized_query = (
                            f"{fallback_entity} {query['question']}"
                        )
                focus_entity = fallback_entity
                if focus_entity is None and normalized_query != raw_query:
                    focus_entity = normalized_query

                deterministic_entity = None
                deterministic_query = None
                if args.bridge_channels == "model2_det1":
                    deterministic_entity = extract_novel_entity_from_memory(
                        current_memory, exclusion_text
                    )
                    if deterministic_entity is not None:
                        deterministic_query = (
                            f"{deterministic_entity} {query['question']}"
                        )
                channel_queries = [normalized_query]
                if deterministic_query is not None:
                    channel_queries.append(deterministic_query)
                bm25_started = time.perf_counter()
                channel_scores = bm25.score(channel_queries)
                model_selected, model_ranked, model_policy = lexical_window(
                    channel_scores[0],
                    args.target_blocks,
                    documents=block_texts,
                    focus_entity=focus_entity,
                )
                deterministic_selected: list[int] = []
                deterministic_ranked: list[int] = []
                if deterministic_query is not None:
                    (
                        deterministic_selected,
                        deterministic_ranked,
                        _deterministic_policy,
                    ) = lexical_window(
                        channel_scores[1],
                        args.target_blocks,
                        documents=block_texts,
                        focus_entity=deterministic_entity,
                    )
                    next_selected = merge_channel_selections(
                        model_selected,
                        deterministic_selected,
                        primary_quota=2,
                        target_blocks=args.target_blocks,
                    )
                    selection_policy = "model2_det1"
                    lexical_ranked = model_ranked
                    focus_entity = " | ".join(
                        item
                        for item in (
                            focus_entity or normalized_query,
                            deterministic_entity,
                        )
                        if item
                    ) or None
                else:
                    next_selected = model_selected
                    lexical_ranked = model_ranked
                    selection_policy = model_policy
                query_seconds = time.perf_counter() - bm25_started
                bridge_bm25_seconds += query_seconds
                record_hit, gold_hit = selected_hits(
                    next_selected,
                    block_start=block_start,
                    block_count=block_count,
                    gold_block_ids=gold_block_ids,
                )
                row["search_trace"].append(
                    {
                        "search_hop": search_hop + 1,
                        "model_output": output,
                        "action_kind": action_kind,
                        "action_facts": action_facts,
                        "verified_fact_count": verified_fact_count,
                        "fact_ledger_size": len(fact_ledger),
                        "final_verified": final_verified,
                        "verifier_rejected": verifier_rejected,
                        "raw_query": raw_query,
                        "normalized_query": normalized_query,
                        "fallback_entity": fallback_entity,
                        "focus_entity": focus_entity,
                        "selection_policy": selection_policy,
                        "deterministic_entity": deterministic_entity,
                        "deterministic_query": deterministic_query,
                        "model_selected_blocks": model_selected,
                        "deterministic_selected_blocks": deterministic_selected,
                        "generation_tokens": generated_tokens,
                        "generation_seconds": generation_seconds,
                        "bm25_seconds": query_seconds,
                        "selected_blocks": next_selected,
                        "record_hit": record_hit,
                        "gold_hit": gold_hit,
                        "record_rank": first_rank_in_record(
                            lexical_ranked, block_start, block_count
                        ),
                        "gold_ranks": gold_ranks(lexical_ranked, gold_block_ids),
                        "deterministic_gold_ranks": (
                            gold_ranks(deterministic_ranked, gold_block_ids)
                            if deterministic_ranked
                            else None
                        ),
                        "lexical_top10": lexical_ranked[:10],
                    }
                )
                previous_queries.append(normalized_query)
                current_selected = next_selected
                current_memory = memory_text(tokenizer, blocks, current_selected)
        except (RuntimeError, ValueError) as error:
            row["controller_error"] = f"{type(error).__name__}: {error}"

        if verified_final_answer is not None:
            answer_text = verified_final_answer
            answer_tokens = 0
            answer_seconds = 0.0
        else:
            search_actions = [
                item
                for item in row["search_trace"]
                if item.get("action_kind") == "search"
            ]
            final_focus_entity = (
                str(search_actions[-1]["focus_entity"] or previous_queries[-1])
                if search_actions
                else "unknown"
            )
            answer_text, answer_tokens, answer_seconds = greedy_chat(
                model,
                tokenizer,
                (
                    evidence_only_final_prompt(
                        current_memory, str(query["question"])
                    )
                    if args.final_prompt_mode == "evidence_only"
                    else final_prompt(
                        current_memory,
                        str(query["question"]),
                        previous_queries,
                        final_focus_entity,
                    )
                ),
                args.answer_max_new_tokens,
                device,
            )
        search_trace = row["search_trace"]
        row.update(
            {
                "any_search_record_hit": any(
                    bool(item["record_hit"]) for item in search_trace
                ),
                "any_search_gold_hit": any(bool(item["gold_hit"]) for item in search_trace),
                "final_search_gold_hit": (
                    bool(search_trace[-1]["gold_hit"]) if search_trace else False
                ),
                "bridge_generation_seconds": bridge_generation_seconds,
                "bridge_bm25_seconds": bridge_bm25_seconds,
                "answer_text": answer_text,
                "answer_generation_tokens": answer_tokens,
                "answer_generation_seconds": answer_seconds,
                "verified_early_final": verified_final_answer is not None,
                "verifier_rejections": sum(
                    bool(item.get("verifier_rejected")) for item in search_trace
                ),
                "verified_fact_ledger": [
                    {
                        "subject": fact.subject,
                        "relation": fact.relation,
                        "object": fact.object,
                        "evidence": fact.evidence,
                    }
                    for fact in fact_ledger
                ],
                "answer_hit": answer_hit(answer_text, query["answers"]),
                "answer_f1": max(
                    token_f1(answer_text, str(reference))
                    for reference in query["answers"]
                ),
            }
        )
        row["online_seconds"] = (
            float(row.get("qk_capture_seconds") or 0.0)
            + float(row["initial_retrieval_seconds"])
            + float(row["bridge_generation_seconds"])
            + float(row["bridge_bm25_seconds"])
            + float(row["answer_generation_seconds"])
        )
        results.append(row)
        with (output_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    "progress": f"{query_offset + 1}/{len(queries)}",
                    "query_id": row["query_id"],
                    "dataset": row["dataset"],
                    "hop1_record_hit": row["hop1_record_hit"],
                    "bridge_gold_hit": row["any_search_gold_hit"],
                    "answer_hit": row["answer_hit"],
                    "answer_f1": row["answer_f1"],
                    "error": row["controller_error"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if rank == 0:
        run_seconds = time.perf_counter() - run_started
        aggregate = {
            "source": "frozen initial retrieval plus bridge BM25 batch diagnostic",
            "contains_synthetic_vectors": False,
            "selection_uses_gold": False,
            "num_tokens": int(summary["num_tokens"]),
            "num_blocks": int(summary["num_blocks"]),
            "world_size": world_size,
            "query_ids": [int(item["query_id"]) for item in queries],
            "excluded_query_ids": sorted(parse_id_set(args.exclude_query_ids)),
            "search_hops": args.search_hops,
            "initial_retriever": args.initial_retriever,
            "controller_mode": args.controller_mode,
            "bridge_channels": args.bridge_channels,
            "final_prompt_mode": args.final_prompt_mode,
            "bm25_build_seconds": bm25_build_seconds,
            "question_bm25_batch_seconds": question_bm25_seconds,
            "batch_run_seconds": run_seconds,
            "batch_queries_per_second": len(results) / max(run_seconds, 1.0e-9),
            **summarize_all(results),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
        if capture is not None:
            capture.close()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
