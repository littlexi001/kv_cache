from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_longmemeval_10m_hierarchical_bm25 import (
    interval_blocks,
    selection_metrics,
)
from evaluate_past_only_100m_hierarchical_bm25 import CompactBM25, rank_candidates
from evaluate_xsum_10m_dynamic_text_retrieval import decode_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve, generate an evidence-conditioned state without gold, and refresh "
            "a bounded LongMemEval 10M session frontier."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--session_depth", type=int, default=3)
    parser.add_argument("--initial_topk", type=int, default=8)
    parser.add_argument("--frontier_blocks", type=int, default=12)
    parser.add_argument("--state_tokens", type=int, default=32)
    parser.add_argument("--state_prefixes", default="8,16,32")
    parser.add_argument(
        "--state_prompt_mode",
        choices=["compact", "temporal_multivalue"],
        default="compact",
    )
    parser.add_argument(
        "--question_types",
        default="",
        help="Optional comma-separated LongMemEval question types.",
    )
    parser.add_argument("--decode_batch_size", type=int, default=4096)
    parser.add_argument("--max_queries", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_ints(spec: str) -> list[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) <= 0:
        raise ValueError("state prefixes must be positive")
    return values


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def format_context(block_ids: list[int], block_texts: list[str]) -> str:
    return "\n\n".join(
        f"[Memory page {index + 1}]\n{block_texts[block_id]}"
        for index, block_id in enumerate(block_ids)
    )


def state_prompt(question: str, context: str, mode: str) -> str:
    if mode == "temporal_multivalue":
        instruction = (
            "You are maintaining a provenance-preserving state for iterative memory "
            "retrieval. The question may ask for a current, latest, or updated value. For "
            "each relevant relation, list every distinct value explicitly present in the "
            "pages together with its date and source page when available. Never collapse "
            "conflicting values into one fact. Mark values that may be stale and write "
            "LATEST UNRESOLVED unless the pages establish which relevant write is newest. "
            "Keep the entity and relation terms needed by another retrieval step. Do not "
            "guess and do not give a final answer. Be terse."
        )
    else:
        instruction = (
            "You are maintaining a compact state for iterative memory retrieval. Read the "
            "memory pages and question. Write only explicitly supported facts, entities, "
            "relations, dates, and unresolved information slots that another retrieval step "
            "should search for. Do not guess. Do not give a final answer. Be terse."
        )
    return (
        f"{instruction}\n\n"
        f"Question: {question}\n\n"
        f"Memory pages:\n{context}\n\n"
        "Retrieval state:"
    )


@torch.inference_mode()
def generate_state(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], float]:
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(device)
    attention_mask = torch.ones_like(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    token_ids = output[0, input_ids.shape[1] :].tolist()
    while token_ids and token_ids[-1] in {
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    }:
        token_ids.pop()
    return [int(item) for item in token_ids], elapsed


def paired_binary(
    rows: list[dict[str, Any]], baseline: str, treatment: str, metric: str
) -> dict[str, Any]:
    base = {
        int(row["query_id"]): bool(row[metric])
        for row in rows
        if row["method"] == baseline and not row["is_abstention"]
    }
    new = {
        int(row["query_id"]): bool(row[metric])
        for row in rows
        if row["method"] == treatment and not row["is_abstention"]
    }
    ids = sorted(set(base) & set(new))
    wins = sum(not base[qid] and new[qid] for qid in ids)
    losses = sum(base[qid] and not new[qid] for qid in ids)
    return {
        "queries": len(ids),
        "baseline_rate": mean(float(base[qid]) for qid in ids),
        "treatment_rate": mean(float(new[qid]) for qid in ids),
        "delta": mean(float(new[qid]) - float(base[qid]) for qid in ids),
        "wins": wins,
        "losses": losses,
        "ties": len(ids) - wins - losses,
        "two_sided_binomial_p": (
            float(binomtest(wins, wins + losses, 0.5).pvalue)
            if wins + losses
            else 1.0
        ),
    }


def main() -> None:
    args = parse_args()
    prefixes = parse_ints(args.state_prefixes)
    if max(prefixes) > args.state_tokens:
        raise ValueError("state prefix exceeds generated state length")
    if not 0 < args.initial_topk <= args.frontier_blocks:
        raise ValueError("require 0 < initial_topk <= frontier_blocks")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    queries = read_jsonl(data_dir / "queries.jsonl")
    allowed_types = {
        item.strip() for item in args.question_types.split(",") if item.strip()
    }
    if allowed_types:
        queries = [
            row for row in queries if str(row["question_type"]) in allowed_types
        ]
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    sessions = read_jsonl(data_dir / "session_manifest.jsonl")
    owners = read_jsonl(data_dir / "owner_manifest.jsonl")
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    block_owner_ids = np.asarray(
        np.load(data_dir / "base_block_owner_ids.npy", mmap_mode="r"), dtype=np.int64
    )
    block_session_rows = np.asarray(
        np.load(data_dir / "base_block_session_rows.npy", mmap_mode="r"), dtype=np.int64
    )
    block_tokens = int(data_summary["block_tokens"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    started = time.perf_counter()
    block_texts = decode_blocks(tokenizer, base_blocks, args.decode_batch_size)
    decode_seconds = time.perf_counter() - started
    print("building block BM25", flush=True)
    started = time.perf_counter()
    block_index = CompactBM25(
        block_texts, min_df=1, max_df=0.995, k1=1.2, b=0.75
    )
    block_index_seconds = time.perf_counter() - started

    owner_ids = [int(row["owner_row"]) for row in owners]
    owner_to_index = {owner_id: index for index, owner_id in enumerate(owner_ids)}
    session_blocks = []
    session_texts = []
    session_owner_indices = np.empty(len(sessions), dtype=np.int64)
    for session in sessions:
        block_ids = interval_blocks(
            int(session["start_token"]), int(session["end_token"]), block_tokens
        )
        session_blocks.append(block_ids)
        session_texts.append(" ".join(block_texts[int(item)] for item in block_ids))
        session_owner_indices[int(session["session_row"])] = owner_to_index[
            int(session["owner_row"])
        ]
    print("building session BM25", flush=True)
    started = time.perf_counter()
    session_index = CompactBM25(
        session_texts, min_df=1, max_df=1.0, k1=1.2, b=0.75
    )
    session_index_seconds = time.perf_counter() - started
    del session_texts
    gc.collect()
    sessions_by_owner = [
        np.flatnonzero(session_owner_indices == index).astype(np.int64)
        for index in range(len(owners))
    ]

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype) if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    rows = []
    states = []
    for query_index, query in enumerate(queries):
        query_id = int(query["query_id"])
        question = str(query["question"])
        owner_index = owner_to_index[int(query["owner_row"])]
        owner_sessions = sessions_by_owner[owner_index]

        def retrieve(text: str, rank_depth: int) -> tuple[list[int], list[int], float]:
            session_query = session_index.query_vector(text)
            block_query = block_index.query_vector(text)
            started_at = time.perf_counter()
            session_scores = session_index.score_candidates(session_query, owner_sessions)
            selected_sessions = rank_candidates(
                owner_sessions, session_scores, args.session_depth
            )
            candidates = np.unique(
                np.concatenate([session_blocks[item] for item in selected_sessions])
            )
            block_scores = block_index.score_candidates(block_query, candidates)
            ranking = rank_candidates(candidates, block_scores, rank_depth)
            return ranking, selected_sessions, time.perf_counter() - started_at

        static_ranking, static_sessions, static_seconds = retrieve(
            question, args.frontier_blocks
        )
        initial = static_ranking[: args.initial_topk]
        prompt = state_prompt(
            question, format_context(initial, block_texts), args.state_prompt_mode
        )
        state_token_ids, generation_seconds = generate_state(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.state_tokens,
            device=device,
        )
        state_text = tokenizer.decode(state_token_ids, skip_special_tokens=True).strip()
        frontier = list(initial)
        seen = set(frontier)
        refresh_rows = []
        total_refresh_seconds = 0.0
        for prefix in prefixes:
            prefix_text = tokenizer.decode(
                state_token_ids[:prefix], skip_special_tokens=True
            ).strip()
            ranking, selected_sessions, query_seconds = retrieve(
                f"{question}\n{prefix_text}", args.initial_topk
            )
            total_refresh_seconds += query_seconds
            added = []
            for block_id in ranking:
                if block_id in seen:
                    continue
                if len(frontier) >= args.frontier_blocks:
                    break
                seen.add(block_id)
                frontier.append(block_id)
                added.append(block_id)
            refresh_rows.append(
                {
                    "prefix_tokens": prefix,
                    "selected_session_rows": selected_sessions,
                    "top_block_ids": ranking,
                    "added_block_ids": added,
                    "frontier_blocks": len(frontier),
                    "query_seconds": query_seconds,
                }
            )

        reference = str(query["answer"])
        normalized_reference = normalize(reference)
        normalized_question = normalize(question)
        normalized_state = normalize(state_text)
        state_mentions_answer = (
            len(normalized_reference) >= 3
            and normalized_reference in normalized_state
        )
        state_adds_answer_not_in_question = (
            state_mentions_answer and normalized_reference not in normalized_question
        )
        states.append(
            {
                "query_id": query_id,
                "question_id": str(query["question_id"]),
                "question_type": str(query["question_type"]),
                "is_abstention": bool(query["is_abstention"]),
                "state_text": state_text,
                "state_token_ids": state_token_ids,
                "generated_tokens": len(state_token_ids),
                "generation_seconds": generation_seconds,
                "state_mentions_reference_posthoc": state_mentions_answer,
                "state_adds_reference_not_in_question_posthoc": (
                    state_adds_answer_not_in_question
                ),
                "initial_block_ids": initial,
                "static_session_rows": static_sessions,
                "refreshes": refresh_rows,
                "dynamic_frontier_block_ids": frontier,
                "selection_uses_answer": False,
            }
        )

        for method, ranking, query_seconds in (
            ("static_top8", static_ranking[: args.initial_topk], static_seconds),
            ("static_top12", static_ranking[: args.frontier_blocks], static_seconds),
            ("evidence_state_dynamic_top12", frontier, total_refresh_seconds),
        ):
            rows.append(
                {
                    "query_id": query_id,
                    "question_id": str(query["question_id"]),
                    "question_type": str(query["question_type"]),
                    "is_abstention": bool(query["is_abstention"]),
                    "method": method,
                    "selected_blocks": len(ranking),
                    "working_set_tokens": len(ranking) * block_tokens,
                    "top_block_ids": ranking,
                    "query_seconds": query_seconds,
                    "selection_uses_answer": False,
                    **selection_metrics(
                        ranking,
                        query=query,
                        block_session_rows=block_session_rows,
                        block_owner_ids=block_owner_ids,
                        topks=[args.initial_topk, args.frontier_blocks],
                    ),
                }
            )
        print(
            json.dumps(
                {
                    "completed": query_index + 1,
                    "queries": len(queries),
                    "query_id": query_id,
                    "frontier_blocks": len(frontier),
                    "generation_seconds": round(generation_seconds, 4),
                    "state_mentions_reference": state_mentions_answer,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    with (output_dir / "states.jsonl").open("w", encoding="utf-8") as handle:
        for row in states:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    quality = []
    for method in ("static_top8", "static_top12", "evidence_state_dynamic_top12"):
        group = [row for row in rows if row["method"] == method]
        positive = [row for row in group if not row["is_abstention"]]
        k = args.initial_topk if method == "static_top8" else args.frontier_blocks
        quality.append(
            {
                "method": method,
                "queries": len(group),
                "positive_queries": len(positive),
                "mean_working_set_tokens": mean(
                    float(row["working_set_tokens"]) for row in group
                ),
                "mean_query_milliseconds": 1000
                * float(mean(float(row["query_seconds"]) for row in group)),
                "exact_block_any": mean(
                    float(row[f"exact_block_any_at_{k}"]) for row in positive
                ),
                "all_evidence_sessions": mean(
                    float(row[f"all_evidence_sessions_at_{k}"]) for row in positive
                ),
                "evidence_session_recall": mean(
                    float(row[f"evidence_session_recall_at_{k}"]) for row in positive
                ),
            }
        )
    summary = {
        "source": "LongMemEval 10M evidence-conditioned natural state refresh",
        "data_summary": data_summary,
        "protocol": {
            "selection_uses_answer": False,
            "initial_topk": args.initial_topk,
            "frontier_blocks": args.frontier_blocks,
            "state_tokens": args.state_tokens,
            "state_prefixes": prefixes,
            "state_prompt_mode": args.state_prompt_mode,
            "question_types": sorted(allowed_types),
            "state_reads_only_initial_retrieval": True,
            "reference_used_only_for_posthoc_state_audit_and_metrics": True,
            "final_answer_reader_not_run": True,
        },
        "decode_seconds": decode_seconds,
        "block_index_seconds": block_index_seconds,
        "session_index_seconds": session_index_seconds,
        "mean_state_generation_seconds": mean(
            float(row["generation_seconds"]) for row in states
        ),
        "states_mentioning_reference_posthoc": sum(
            bool(row["state_mentions_reference_posthoc"]) for row in states
        ),
        "states_adding_reference_not_in_question_posthoc": sum(
            bool(row["state_adds_reference_not_in_question_posthoc"]) for row in states
        ),
        "quality": quality,
        "dynamic_vs_static_top12": {
            "exact_block_any": paired_binary(
                rows,
                "static_top12",
                "evidence_state_dynamic_top12",
                f"exact_block_any_at_{args.frontier_blocks}",
            ),
            "all_evidence_sessions": paired_binary(
                rows,
                "static_top12",
                "evidence_state_dynamic_top12",
                f"all_evidence_sessions_at_{args.frontier_blocks}",
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
