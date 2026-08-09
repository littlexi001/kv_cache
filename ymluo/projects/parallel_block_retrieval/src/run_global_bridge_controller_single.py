from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from profile_real_qk import QKCapture, read_jsonl, resolve_dtype
from run_global_dynamic_svd_kv_single import (
    capture_query_ids,
    decode_snippet,
    distributed_retrieve,
    selected_memory_ids,
)
from run_iterative_condition_retrieval import BM25Index
from run_lexical_block_retrieval import decode_blocks
from run_real_qk_retrieval import load_index, setup_distributed
from run_single_query_dynamic_kv_generation import answer_hit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage global retrieval with a model-generated bridge query."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--query_id", type=int, default=0)
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
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return parser.parse_args()


def render_chat(tokenizer: Any, content: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.inference_mode()
def greedy_chat(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    content: str,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, int, float]:
    prompt_ids = render_chat(tokenizer, content)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
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
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    generated = output[0, input_ids.shape[1] :].tolist()
    return (
        tokenizer.decode(generated, skip_special_tokens=True).strip(),
        len(generated),
        elapsed,
    )


def parse_search_action(text: str) -> str:
    match = re.search(r"(?:^|\n)\s*SEARCH\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"model did not emit a SEARCH action: {text!r}")
    query = match.group(1).splitlines()[0].strip().strip("`*\"' .")
    if not query:
        raise ValueError("model emitted an empty SEARCH action")
    return query


def normalize_bridge_query(search_query: str, original_question: str) -> str:
    named_spans = re.findall(
        r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*)+\b",
        search_query,
    )
    novel = [
        span
        for span in named_spans
        if span.casefold() not in original_question.casefold()
    ]
    if not novel:
        return search_query
    novel.sort(key=lambda item: (-len(item.split()), -len(item), item.casefold()))
    return novel[0]


def extract_novel_entity_from_memory(memory: str, exclusion_text: str) -> str | None:
    excluded = exclusion_text.casefold()
    context_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", excluded)
        if len(token) >= 3
    }
    relation_groups = [
        {"wife", "husband", "spouse", "married"},
        {"parent", "father", "mother", "child", "son", "daughter"},
        {"brother", "sister", "sibling"},
        {"author", "writer", "wrote", "written"},
        {"director", "directed", "film"},
        {"member", "chair", "organization"},
    ]
    active_relation_terms = set().union(
        *(group for group in relation_groups if group & context_tokens)
    )
    if not active_relation_terms:
        active_relation_terms = set().union(*relation_groups)
    candidates: list[tuple[int, int, int, int, int, int, str]] = []
    for sentence_index, sentence in enumerate(re.split(r"(?<=[.!?])\s+|\n+", memory)):
        sentence_tokens = set(re.findall(r"[a-z0-9]+", sentence.casefold()))
        overlap = len(context_tokens & sentence_tokens)
        relation_bonus = len(active_relation_terms & sentence_tokens)
        relation_matches = list(
            re.finditer(
                r"\b(?:" + "|".join(sorted(active_relation_terms)) + r")\b",
                sentence,
                flags=re.IGNORECASE,
            )
        )
        for span_match in re.finditer(
            r"\b[A-Z][A-Za-z0-9-]*(?:\s+(?:of\s+)?[A-Z][A-Za-z0-9-]*)+\b",
            sentence,
        ):
            span = span_match.group(0)
            if span.casefold() in excluded or span.startswith("Passage "):
                continue
            distances = [span_match.start() - item.end() for item in relation_matches]
            after_relation = int(any(0 <= item <= 120 for item in distances))
            absolute_distance = min((abs(item) for item in distances), default=10_000)
            candidates.append(
                (
                    after_relation,
                    relation_bonus,
                    overlap,
                    -absolute_distance,
                    len(span.split()),
                    -sentence_index,
                    span,
                )
            )
    if not candidates:
        return None
    return max(candidates)[6]


def memory_text(tokenizer: Any, blocks: np.ndarray, block_ids: Sequence[int]) -> str:
    ids = selected_memory_ids(
        blocks,
        block_ids,
        tokenizer("\n\n", add_special_tokens=False)["input_ids"],
    )
    return tokenizer.decode(ids, skip_special_tokens=True)


def lexical_window(
    scores: np.ndarray,
    target_blocks: int,
    documents: Sequence[str] | None = None,
    focus_entity: str | None = None,
) -> tuple[list[int], list[int], str]:
    ids = np.arange(scores.shape[0], dtype=np.int64)
    ranked = np.lexsort((ids, -scores)).tolist()
    if documents is not None and focus_entity:
        title_pattern = re.compile(
            r"passage\s+\d+\s*:\s*" + re.escape(focus_entity.casefold())
        )
        for block_id in ranked[:100]:
            if title_pattern.search(documents[int(block_id)].casefold()):
                selected = [
                    item
                    for item in range(int(block_id), int(block_id) + target_blocks)
                    if item < len(scores)
                ]
                return selected, [int(item) for item in ranked], "entity_title_window"
    return (
        [int(item) for item in ranked[:target_blocks]],
        [int(item) for item in ranked],
        "lexical_topk",
    )


def main() -> None:
    args = parse_args()
    if args.search_hops < 1:
        raise ValueError("search_hops must be at least one")
    rank, world_size, _local_rank, device = setup_distributed()
    if world_size <= 1:
        raise ValueError("This experiment requires distributed index shards")

    corpus_dir = Path(args.corpus_dir)
    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    summary = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    raw_keys, svd_keys, local_block_ids, _ = load_index(
        profile_dir, summary, rank, world_size, device
    )
    basis_payload = torch.load(profile_dir / "basis.pt", map_location="cpu", weights_only=False)
    basis = basis_payload["basis"].to(device=device, dtype=torch.float16)
    pair_specs = [dict(item) for item in summary["pair_specs"]]
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    query = next(item for item in queries if int(item["query_id"]) == args.query_id)
    gold_block_ids = [int(item) for item in query["gold_block_ids"]]
    if not gold_block_ids:
        raise ValueError("query must provide at least one gold block for evaluation")

    model = None
    tokenizer = None
    capture = None
    question_query = None
    block_texts = None
    bm25 = None
    bm25_build_seconds = None
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=resolve_dtype(args.dtype),
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        capture = QKCapture(model, sorted({int(item["layer"]) for item in pair_specs}))
        question_ids = tokenizer(str(query["question"]), add_special_tokens=False)["input_ids"]
        question_query = capture_query_ids(
            model, capture, pair_specs, question_ids, args.query_tokens, device
        )
        bm25_started = time.perf_counter()
        block_texts = decode_blocks(tokenizer, blocks)
        bm25 = BM25Index(block_texts, min_df=1, max_df=1.0, k1=1.2, b=0.75)
        bm25_build_seconds = time.perf_counter() - bm25_started

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
    )

    payload = None
    if rank == 0:
        current_selected = list(hop1_selected)
        current_memory = memory_text(tokenizer, blocks, current_selected)
        search_trace: list[dict[str, Any]] = []
        previous_queries: list[str] = []
        lexical_seconds = 0.0
        for search_hop in range(args.search_hops):
            previous_text = (
                "\nPrevious lookup queries: " + " | ".join(previous_queries)
                if previous_queries
                else ""
            )
            bridge_prompt = (
                "Memory:\n"
                f"{current_memory}\n\n"
                f"Question: {query['question']}"
                f"{previous_text}\n"
                "Perform one more lookup step before answering. Resolve one relational "
                "description, such as a spouse, parent, child, author, director, organization, "
                "or location, to the exact proper name stated in the current memory. Prefer a "
                "proper name that has not already been searched. Output exactly one line in the "
                "form `SEARCH: <exact proper name> <still-unresolved relation>`. Do not answer "
                "the original question and do not explain."
            )
            bridge_output, bridge_tokens, bridge_generation_seconds = greedy_chat(
                model,
                tokenizer,
                bridge_prompt,
                args.bridge_max_new_tokens,
                device,
            )
            raw_bridge_query = parse_search_action(bridge_output)
            exclusion_text = str(query["question"]) + " " + " ".join(previous_queries)
            bridge_query = normalize_bridge_query(raw_bridge_query, exclusion_text)
            fallback_entity = None
            if search_hop > 0 and bridge_query == raw_bridge_query:
                fallback_entity = extract_novel_entity_from_memory(
                    current_memory, exclusion_text
                )
                if fallback_entity is not None:
                    bridge_query = f"{fallback_entity} {query['question']}"
            search_focus_entity = fallback_entity
            if search_focus_entity is None and bridge_query != raw_bridge_query:
                search_focus_entity = bridge_query

            lexical_started = time.perf_counter()
            lexical_scores = bm25.score([bridge_query])[0]
            next_selected, lexical_ranked, selection_policy = lexical_window(
                lexical_scores,
                args.target_blocks,
                documents=block_texts,
                focus_entity=search_focus_entity,
            )
            lexical_seconds += time.perf_counter() - lexical_started
            search_trace.append(
                {
                    "search_hop": search_hop + 1,
                    "model_output": bridge_output,
                    "raw_query": raw_bridge_query,
                    "normalized_query": bridge_query,
                    "fallback_entity": fallback_entity,
                    "focus_entity": search_focus_entity,
                    "selection_policy": selection_policy,
                    "generation_tokens": bridge_tokens,
                    "generation_seconds": bridge_generation_seconds,
                    "generation_tokens_per_second": (
                        bridge_tokens / max(bridge_generation_seconds, 1.0e-9)
                    ),
                    "selected_blocks": next_selected,
                    "lexical_top10": lexical_ranked[:10],
                }
            )
            previous_queries.append(bridge_query)
            current_selected = next_selected
            current_memory = memory_text(tokenizer, blocks, current_selected)

        hop2_selected = current_selected
        hop2_memory = current_memory
        final_focus_entity = (
            search_trace[-1]["fallback_entity"]
            or search_trace[-1]["normalized_query"]
        )
        final_prompt = (
            "Memory:\n"
            f"{hop2_memory}\n\n"
            f"Question: {query['question']}\n"
            f"Intermediate lookups used: {' | '.join(previous_queries)}\n"
            f"Focus entity for the final unresolved relation: {final_focus_entity}\n"
            "Answer the original question using the memory. Match paraphrased relation evidence: "
            "for example, location may be expressed with born, died, buried, interred, or "
            "entombed, and authorship may be expressed with wrote, authored, or created. Use the "
            "shortest directly supported entity or location phrase. The evidence must describe "
            "the focus entity, not another person or organization mentioned nearby. Ignore "
            "locations whose grammatical subject is not the focus entity. Output only the "
            "concise final answer."
        )
        answer_text, answer_tokens, answer_generation_seconds = greedy_chat(
            model,
            tokenizer,
            final_prompt,
            args.answer_max_new_tokens,
            device,
        )
        gold_ranks = {
            str(block_id): lexical_ranked.index(block_id) + 1 for block_id in gold_block_ids
        }
        final_search = search_trace[-1]
        payload = {
            "source": "10M distributed SVD32/raw128 first hop plus runtime bridge BM25 second hop",
            "contains_synthetic_vectors": False,
            "query_id": int(query["query_id"]),
            "question": query["question"],
            "answers": query["answers"],
            "num_tokens": int(summary["num_tokens"]),
            "num_blocks": int(summary["num_blocks"]),
            "world_size": world_size,
            "hop1_selected": hop1_selected,
            "hop1_event": hop1_event,
            "search_hops": args.search_hops,
            "search_trace": search_trace,
            "bridge_model_output": final_search["model_output"],
            "raw_bridge_query": final_search["raw_query"],
            "bridge_query": final_search["normalized_query"],
            "bridge_generation_tokens": final_search["generation_tokens"],
            "final_focus_entity": final_focus_entity,
            "hop2_selected": hop2_selected,
            "hop2_lexical_top10": final_search["lexical_top10"],
            "gold_block_ids": gold_block_ids,
            "gold_lexical_ranks": gold_ranks,
            "best_gold_lexical_rank": min(gold_ranks.values()),
            "gold_selected_hop2": any(item in hop2_selected for item in gold_block_ids),
            "answer_text": answer_text,
            "answer_generation_tokens": answer_tokens,
            "answer_generation_seconds": answer_generation_seconds,
            "answer_generation_tokens_per_second": (
                answer_tokens / max(answer_generation_seconds, 1.0e-9)
            ),
            "answer_hit": answer_hit(answer_text, query["answers"]),
            "bm25_build_seconds": bm25_build_seconds,
            "bm25_query_seconds": lexical_seconds,
            "hop1_snippets": {
                str(item): decode_snippet(tokenizer, blocks, item) for item in hop1_selected
            },
            "hop2_snippets": {
                str(item): decode_snippet(tokenizer, blocks, item) for item in hop2_selected
            },
            "selection_uses_gold": False,
        }
        (output_dir / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        capture.close()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
