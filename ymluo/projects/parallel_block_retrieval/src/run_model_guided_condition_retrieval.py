from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from profile_real_qk import read_jsonl
from run_iterative_condition_retrieval import BM25Index, rank_blocks, split_sentences
from run_lexical_block_retrieval import (
    decode_blocks,
    evaluate_selection,
    group_for_context,
    write_csv,
)


INVALID_STATUS_PATTERNS = (
    r"\breject(?:ed|ion)?\b",
    r"\bdiscard(?:ed)?\b",
    r"\bwithdrawn\b",
    r"\bvoid(?:ed)?\b",
    r"\bannul(?:led|ed)?\b",
    r"\brescind(?:ed)?\b",
    r"\bsuperseded\b",
    r"\bobsolete\b",
    r"\bretired\b",
    r"\bdecommissioned\b",
    r"\bexpired\b",
    r"\blapsed\b",
    r"\bcancel(?:ed|led)?\b",
    r"\bunexecuted\b",
    r"\bunsigned\b",
    r"\bnonbinding\b",
    r"\bprovisional\b",
    r"\bpreliminary\b",
    r"\btentative\b",
    r"\bhypothetical\b",
    r"\bsimulat(?:ed|ion)\b",
    r"\bfictional\b",
    r"\bmock[- ]?up\b",
    r"\btraining example\b",
    r"\bnever (?:approved|adopted|ratified|enacted|took effect)\b",
    r"\bwithout (?:approval|authorization|identifying|specifying)\b",
    r"\b(?:no|not) (?:approved|operative|usable|deployed|binding|real)\b",
    r"\b(?:no|not|lacks?|omits?|missing|blank|empty)\b.*\b(?:value|entry|field|force|name|detail|display|color|wording|unit)\b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distributed model-guided validation with anchor-state condition retrieval."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--index_cache_dir")
    parser.add_argument("--build_index_only", action="store_true")
    parser.add_argument("--target_blocks", type=int, default=3)
    parser.add_argument("--candidate_blocks", type=int, default=16)
    parser.add_argument("--candidate_sentence_scan", type=int, default=96)
    parser.add_argument("--anchor_candidate_blocks", type=int, default=8)
    parser.add_argument("--anchor_terms", type=int, default=1)
    parser.add_argument(
        "--anchor_only_mode",
        choices=["always", "identifier", "never"],
        default="always",
    )
    parser.add_argument("--bm25_weight", type=float, default=0.25)
    parser.add_argument("--choice_weight", type=float, default=0.0)
    parser.add_argument("--invalid_status_penalty", type=float, default=3.0)
    parser.add_argument(
        "--completion_thresholds",
        default="0,2,3,4",
        help="Comma-separated completion margins below which a second hop is used.",
    )
    parser.add_argument("--max_followup_tokens", type=int, default=16)
    parser.add_argument(
        "--diagnostic_model_followup",
        action="store_true",
        help="Also generate a free-text next-hop query for diagnostics.",
    )
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--splits", default="", help="Optional comma-separated data splits.")
    parser.add_argument("--task_types", default="", help="Optional comma-separated task types.")
    parser.add_argument("--record_routing_csv")
    parser.add_argument("--record_routing_field", default="risk_record")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--min_df", type=int, default=1)
    parser.add_argument("--max_df", type=float, default=1.0)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


def setup_distributed(device_mode: str) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = device_mode == "cuda" or (
        device_mode == "auto" and torch.cuda.is_available()
    )
    if world_size > 1:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def standardize(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    scale = max(float(values.std()), 1.0e-6)
    return ((values - float(values.mean())) / scale).astype(np.float32)


def normalized_terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def invalid_status_flags(sentences: list[str]) -> np.ndarray:
    return np.asarray(
        [
            float(any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in INVALID_STATUS_PATTERNS))
            for sentence in sentences
        ],
        dtype=np.float32,
    )


def candidate_sentences(
    scores: np.ndarray,
    sentence_blocks: np.ndarray,
    *,
    candidate_blocks: int,
    sentence_scan: int,
    excluded_blocks: set[int] | None = None,
) -> list[int]:
    excluded = excluded_blocks or set()
    scan = min(sentence_scan, len(scores))
    if scan == len(scores):
        order = np.argsort(-scores, kind="stable")
    else:
        prefix = np.argpartition(-scores, scan - 1)[:scan]
        order = prefix[np.argsort(-scores[prefix], kind="stable")]
    selected: list[int] = []
    seen_blocks: set[int] = set()
    for sentence_id in order:
        block_id = int(sentence_blocks[int(sentence_id)])
        if block_id in excluded or block_id in seen_blocks:
            continue
        selected.append(int(sentence_id))
        seen_blocks.add(block_id)
        if len(selected) >= candidate_blocks:
            break
    return selected


def multichannel_candidate_sentences(
    index: BM25Index,
    text: str,
    base_scores: np.ndarray,
    sentence_blocks: np.ndarray,
    *,
    candidate_blocks_count: int,
    anchor_candidate_blocks: int,
    anchor_terms: int,
    sentence_scan: int,
    excluded_blocks: set[int] | None = None,
    anchor_text_override: str = "",
) -> tuple[list[int], str, np.ndarray, np.ndarray]:
    anchor_text = anchor_text_override or index.rare_query_text(
        text, max_terms=anchor_terms
    )
    anchor_scores = index.score([anchor_text])[0]
    priority = candidate_sentences(
        anchor_scores,
        sentence_blocks,
        candidate_blocks=min(anchor_candidate_blocks, candidate_blocks_count),
        sentence_scan=sentence_scan,
        excluded_blocks=excluded_blocks,
    )
    selected = list(priority)
    seen_blocks = {int(sentence_blocks[item]) for item in selected}
    base = candidate_sentences(
        base_scores,
        sentence_blocks,
        candidate_blocks=candidate_blocks_count,
        sentence_scan=sentence_scan,
        excluded_blocks=(excluded_blocks or set()) | seen_blocks,
    )
    for sentence_id in base:
        block_id = int(sentence_blocks[sentence_id])
        if block_id in seen_blocks:
            continue
        selected.append(sentence_id)
        seen_blocks.add(block_id)
        if len(selected) >= candidate_blocks_count:
            break
    return selected, anchor_text, anchor_scores, np.maximum(base_scores, anchor_scores)


def relevance_prompt(tokenizer: Any, question: str, search_need: str, sentence: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Judge whether a memory sentence is a current, valid premise that directly "
                "helps satisfy the stated search need. Rejected, tentative, missing, or "
                "nonbinding statements are invalid. Retired, obsolete, canceled, simulated, "
                "example-only, blank, or absent records are also invalid. A sentence about a "
                "different named subject is not useful. A current positive sentence that maps an "
                "identifier, alias, or call sign to a real entity is useful when that identifier "
                "appears in the question, even if another premise is still needed. Answer only "
                "Yes or No."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Current search need: {search_need}\n"
                f"Memory sentence: {sentence}\n"
                "Is this a valid and useful premise?"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def choice_prompt(
    tokenizer: Any,
    question: str,
    search_need: str,
    sentences: list[str],
    labels: list[str],
) -> str:
    candidates = "\n".join(
        f"[{label}] {sentence}" for label, sentence in zip(labels, sentences, strict=True)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Choose the single memory sentence that is the most current, authoritative, "
                "non-hypothetical premise for the search need. Prefer an enacted or operative "
                "fact over drafts, mock-ups, simulations, expired proposals, missing fields, or "
                "statements about another subject. Output only its letter."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\nCurrent search need: {search_need}\n"
                f"Candidates:\n{candidates}\nBest candidate:"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.inference_mode()
def score_candidate_choice(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    *,
    question: str,
    search_need: str,
    sentences: list[str],
) -> np.ndarray:
    if len(sentences) > 26:
        raise ValueError("candidate choice supports at most 26 sentences")
    if not sentences:
        return np.empty(0, dtype=np.float32)
    labels = [chr(ord("A") + item) for item in range(len(sentences))]
    label_ids: list[int] = []
    for label in labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"choice label must be one token: {label} -> {token_ids}")
        label_ids.append(token_ids[0])
    prompt = choice_prompt(tokenizer, question, search_need, sentences, labels)
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    logits = model(**encoded, use_cache=False).logits[0, -1].float()
    choice_logits = logits[torch.tensor(label_ids, dtype=torch.long, device=device)]
    return choice_logits.cpu().numpy().astype(np.float32, copy=False)


@torch.inference_mode()
def score_relevance(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    *,
    question: str,
    search_need: str,
    sentences: list[str],
    yes_token_id: int,
    no_token_id: int,
) -> np.ndarray:
    if not sentences:
        return np.empty(0, dtype=np.float32)
    prompts = [
        relevance_prompt(tokenizer, question, search_need, sentence)
        for sentence in sentences
    ]
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    logits = model(**encoded, use_cache=False).logits[:, -1, :].float()
    pair = torch.stack((logits[:, yes_token_id], logits[:, no_token_id]), dim=-1)
    log_probabilities = torch.log_softmax(pair, dim=-1)
    margins = log_probabilities[:, 0] - log_probabilities[:, 1]
    return margins.cpu().numpy().astype(np.float32, copy=False)


def followup_prompt(tokenizer: Any, question: str, premise: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Turn a question and one retrieved premise into the shortest search phrase "
                "for the next missing premise. If the premise maps an identifier to a newly "
                "introduced entity, replace the identifier with that entity in the search phrase; "
                "do not repeat the old identifier. Copy the new entity name exactly. If the "
                "premise already answers the question, output NONE. Output only the phrase or NONE."
            ),
        },
        {
            "role": "user",
            "content": (
                "Question: What color is the beacon on the craft called CODE-100?\n"
                "Retrieved premise: The registry maps CODE-100 to North Star Vessel."
            ),
        },
        {"role": "assistant", "content": "North Star Vessel beacon color"},
        {
            "role": "user",
            "content": f"Question: {question}\nRetrieved premise: {premise}",
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def completion_prompt(tokenizer: Any, question: str, premise: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Decide whether the retrieved premise by itself contains the requested final "
                "answer to the question. A premise that only maps an alias, identifies an "
                "intermediate entity, or provides one hop is incomplete. Answer only Yes or No."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\nRetrieved premise: {premise}\nComplete answer present?",
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.inference_mode()
def score_completion(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    *,
    question: str,
    premise: str,
    yes_token_id: int,
    no_token_id: int,
) -> float:
    prompt = completion_prompt(tokenizer, question, premise)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    logits = model(**encoded, use_cache=False).logits[0, -1].float()
    pair = torch.stack((logits[yes_token_id], logits[no_token_id]))
    log_probabilities = torch.log_softmax(pair, dim=-1)
    return float((log_probabilities[0] - log_probabilities[1]).item())


@torch.inference_mode()
def generate_followup(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    *,
    question: str,
    premise: str,
    max_new_tokens: int,
) -> str:
    prompt = followup_prompt(tokenizer, question, premise)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    continuation = generated[0, encoded["input_ids"].shape[1] :]
    text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    text = text.splitlines()[0].strip().strip('"\'') if text else ""
    for prefix in ("Search phrase:", "Phrase:", "Next:"):
        if text.casefold().startswith(prefix.casefold()):
            text = text[len(prefix) :].strip()
    if not text or text.casefold().startswith("none"):
        return ""
    return text


def rerank_candidates(
    *,
    sentence_ids: list[int],
    sentence_scores: np.ndarray,
    model_scores: np.ndarray,
    sentence_blocks: np.ndarray,
    bm25_weight: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    lexical = np.asarray([sentence_scores[item] for item in sentence_ids], dtype=np.float32)
    combined = standardize(model_scores) + bm25_weight * standardize(lexical)
    order = np.lexsort(
        (
            np.asarray([int(sentence_blocks[item]) for item in sentence_ids]),
            -lexical,
            -combined,
        )
    )
    blocks = [int(sentence_blocks[sentence_ids[int(item)]]) for item in order]
    diagnostics = [
        {
            "sentence_id": int(sentence_ids[int(item)]),
            "block_id": int(sentence_blocks[sentence_ids[int(item)]]),
            "bm25_score": float(lexical[int(item)]),
            "model_margin": float(model_scores[int(item)]),
            "combined_score": float(combined[int(item)]),
        }
        for item in order
    ]
    return blocks, diagnostics


def complete_ranking(prefix: list[int], fallback: list[int]) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for block_id in [*prefix, *fallback]:
        if block_id in seen:
            continue
        output.append(block_id)
        seen.add(block_id)
    return output


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        group = [row for row in rows if row["method"] == method]
        output.append(
            {
                "method": method,
                "queries": len(group),
                "mean_selected_blocks": statistics.fmean(
                    len(json.loads(str(row["selected_block_ids"]))) for row in group
                ),
                "source_record_recall": statistics.fmean(
                    float(row["source_record_recall"]) for row in group
                ),
                "record_top1_recall": statistics.fmean(
                    float(row["record_top1_recall"]) for row in group
                ),
                "answer_block_recall": statistics.fmean(
                    float(row["answer_block_recall"]) for row in group
                ),
                "answer_block_mrr": statistics.fmean(
                    float(row["answer_block_mrr"]) for row in group
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    completion_thresholds = sorted(
        {float(item) for item in args.completion_thresholds.split(",") if item.strip()}
    )
    if not completion_thresholds:
        raise ValueError("at least one completion threshold is required")
    rank, world_size, local_rank, device = setup_distributed(args.device)
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(f"Yes/No must be single tokens, got {yes_ids=} and {no_ids=}")

    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    allowed_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    if allowed_splits:
        queries = [query for query in queries if str(query.get("split", "")) in allowed_splits]
    allowed_tasks = {item.strip() for item in args.task_types.split(",") if item.strip()}
    if allowed_tasks:
        queries = [
            query for query in queries if str(query.get("task_type", "")) in allowed_tasks
        ]
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    records = read_jsonl(corpus_dir / "records.jsonl")
    routed_records: dict[int, int] = {}
    if args.record_routing_csv:
        with Path(args.record_routing_csv).open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                routed_records[int(row["query_id"])] = int(row[args.record_routing_field])
    index_started = time.perf_counter()
    cache_path = (
        Path(args.index_cache_dir) / "sentence_bm25.joblib"
        if args.index_cache_dir
        else None
    )
    blocks_path = corpus_dir / "blocks.npy"
    blocks_stat = blocks_path.stat()
    cache_metadata = {
        "version": 1,
        "blocks_shape": tuple(int(size) for size in blocks.shape),
        "blocks_file_size": blocks_stat.st_size,
        "blocks_file_mtime_ns": blocks_stat.st_mtime_ns,
        "model_name_or_path": args.model_name_or_path,
        "min_df": args.min_df,
        "max_df": args.max_df,
        "k1": args.k1,
        "b": args.b,
    }
    cache_payload = joblib.load(cache_path) if cache_path and cache_path.exists() else None
    cache_hit = bool(
        cache_payload and cache_payload.get("metadata") == cache_metadata
    )
    if cache_hit:
        sentences = cache_payload["sentences"]
        sentence_blocks = cache_payload["sentence_blocks"]
        index = cache_payload["index"]
    else:
        if world_size == 1 or rank == 0:
            block_texts = decode_blocks(tokenizer, blocks)
            sentences, sentence_blocks = split_sentences(block_texts)
            index = BM25Index(
                sentences,
                min_df=args.min_df,
                max_df=args.max_df,
                k1=args.k1,
                b=args.b,
            )
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(
                    {
                        "metadata": cache_metadata,
                        "sentences": sentences,
                        "sentence_blocks": sentence_blocks,
                        "index": index,
                    },
                    cache_path,
                    compress=3,
                )
        if world_size > 1:
            dist.barrier()
            if rank != 0:
                if cache_path is None:
                    raise ValueError("distributed cache build requires --index_cache_dir")
                cache_payload = joblib.load(cache_path)
                sentences = cache_payload["sentences"]
                sentence_blocks = cache_payload["sentence_blocks"]
                index = cache_payload["index"]
    if args.build_index_only:
        if cache_path is None:
            raise ValueError("--build_index_only requires --index_cache_dir")
        if rank == 0:
            index_summary = {
                "blocks": int(blocks.shape[0]),
                "sentences": len(sentences),
                "index_cache_path": str(cache_path),
                "index_cache_hit": cache_hit,
                "wall_seconds": time.perf_counter() - index_started,
                "metadata": cache_metadata,
            }
            (output_dir / "index_build_summary.json").write_text(
                json.dumps(index_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(index_summary, ensure_ascii=False, indent=2))
        if world_size > 1:
            dist.destroy_process_group()
        return
    questions = [str(query["question"]) for query in queries]
    local_query_indices = list(range(rank, len(queries), world_size))
    lexical_score_started = time.perf_counter()
    local_query_scores = index.score([questions[item] for item in local_query_indices])
    lexical_score_seconds = time.perf_counter() - lexical_score_started

    block_count = int(blocks.shape[0])
    block_to_record = np.empty(block_count, dtype=np.int32)
    source_record_by_start: dict[int, int] = {}
    for record_id, record in enumerate(records):
        start = int(record["block_start"])
        end = start + int(record["block_count"])
        block_to_record[start:end] = record_id
        source_record_by_start[start] = record_id
    all_block_ids = set(range(block_count))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if world_size > 1:
        dist.barrier()
    started = time.perf_counter()

    local_rows: list[dict[str, Any]] = []
    local_diagnostics: list[dict[str, Any]] = []
    for query_index, local_sentence_scores in zip(
        local_query_indices, local_query_scores, strict=True
    ):
        query = queries[query_index]
        question = questions[query_index]
        sentence_scores = local_sentence_scores.copy()
        routed_record = routed_records.get(int(query["query_id"]))
        excluded_for_query: set[int] = set()
        if routed_record is not None:
            routed = records[routed_record]
            routed_start = int(routed["block_start"])
            routed_end = routed_start + int(routed["block_count"])
            allowed_blocks = set(range(routed_start, routed_end))
            excluded_for_query = all_block_ids - allowed_blocks
            allowed_sentences = np.isin(sentence_blocks, list(allowed_blocks))
            sentence_scores[~allowed_sentences] = -np.inf
        base_block_ranking = rank_blocks(sentence_scores, sentence_blocks, block_count)
        (
            first_sentence_ids,
            first_anchor_text,
            first_anchor_scores,
            _first_combined_scores,
        ) = multichannel_candidate_sentences(
            index,
            question,
            sentence_scores,
            sentence_blocks,
            candidate_blocks_count=args.candidate_blocks,
            anchor_candidate_blocks=args.anchor_candidate_blocks,
            anchor_terms=args.anchor_terms,
            sentence_scan=args.candidate_sentence_scan,
            excluded_blocks=excluded_for_query,
        )
        anchor_only_ids = [
            sentence_id
            for sentence_id in first_sentence_ids
            if float(first_anchor_scores[sentence_id]) > 0.0
            and normalized_terms(first_anchor_text) <= normalized_terms(sentences[sentence_id])
        ][: args.anchor_candidate_blocks]
        anchor_has_identifier = any(character.isdigit() for character in first_anchor_text)
        use_anchor_only = args.anchor_only_mode == "always" or (
            args.anchor_only_mode == "identifier" and anchor_has_identifier
        )
        if use_anchor_only and anchor_only_ids:
            first_sentence_ids = anchor_only_ids
            first_lexical_scores = first_anchor_scores
            first_search_need = (
                "First resolve this rare query anchor and select only a premise tied to the "
                f"exact anchor: {first_anchor_text}"
            )
        else:
            first_lexical_scores = _first_combined_scores
            first_search_need = question
        first_model_scores = score_relevance(
            model,
            tokenizer,
            device,
            question=question,
            search_need=first_search_need,
            sentences=[sentences[item] for item in first_sentence_ids],
            yes_token_id=yes_ids[0],
            no_token_id=no_ids[0],
        )
        first_candidate_texts = [sentences[item] for item in first_sentence_ids]
        first_choice_scores = np.zeros(len(first_sentence_ids), dtype=np.float32)
        if args.choice_weight != 0.0:
            first_choice_scores = score_candidate_choice(
                model,
                tokenizer,
                device,
                question=question,
                search_need=first_search_need,
                sentences=first_candidate_texts,
            )
        first_invalid_flags = invalid_status_flags(first_candidate_texts)
        first_fused_scores = (
            standardize(first_model_scores)
            + args.choice_weight * standardize(first_choice_scores)
            - args.invalid_status_penalty * first_invalid_flags
        )

        model_only_blocks, model_only_diagnostics = rerank_candidates(
            sentence_ids=first_sentence_ids,
            sentence_scores=first_lexical_scores,
            model_scores=first_fused_scores,
            sentence_blocks=sentence_blocks,
            bm25_weight=0.0,
        )
        hybrid_blocks, hybrid_diagnostics = rerank_candidates(
            sentence_ids=first_sentence_ids,
            sentence_scores=first_lexical_scores,
            model_scores=first_fused_scores,
            sentence_blocks=sentence_blocks,
            bm25_weight=args.bm25_weight,
        )
        model_only_ranking = complete_ranking(model_only_blocks, base_block_ranking)
        hybrid_ranking = complete_ranking(hybrid_blocks, base_block_ranking)

        first_block = hybrid_ranking[0]
        first_diagnostic = next(
            item for item in hybrid_diagnostics if int(item["block_id"]) == first_block
        )
        first_sentence_id = int(first_diagnostic["sentence_id"])
        first_premise = sentences[first_sentence_id]
        completion_margin = score_completion(
            model,
            tokenizer,
            device,
            question=question,
            premise=first_premise,
            yes_token_id=yes_ids[0],
            no_token_id=no_ids[0],
        )
        generated_followup = ""
        if args.diagnostic_model_followup:
            generated_followup = generate_followup(
                model,
                tokenizer,
                device,
                question=question,
                premise=first_premise,
                max_new_tokens=args.max_followup_tokens,
            )
        feedback_terms = index.rare_novel_text(
            first_premise,
            exclude_text=question,
            max_terms=4,
        )
        followup = (
            f"{feedback_terms} {question}".strip()
            if feedback_terms
            else generated_followup
        )

        second_block: int | None = None
        second_diagnostics: list[dict[str, Any]] = []
        iterative_prefix = [first_block]
        if followup:
            followup_scores = index.score([followup])[0]
            (
                second_sentence_ids,
                second_anchor_text,
                _second_anchor_scores,
                second_lexical_scores,
            ) = multichannel_candidate_sentences(
                index,
                followup,
                followup_scores,
                sentence_blocks,
                candidate_blocks_count=args.candidate_blocks,
                anchor_candidate_blocks=args.anchor_candidate_blocks,
                anchor_terms=args.anchor_terms,
                sentence_scan=args.candidate_sentence_scan,
                excluded_blocks=excluded_for_query | {first_block},
                anchor_text_override=feedback_terms,
            )
            second_model_scores = score_relevance(
                model,
                tokenizer,
                device,
                question=question,
                search_need=followup,
                sentences=[sentences[item] for item in second_sentence_ids],
                yes_token_id=yes_ids[0],
                no_token_id=no_ids[0],
            )
            second_candidate_texts = [sentences[item] for item in second_sentence_ids]
            second_choice_scores = np.zeros(len(second_sentence_ids), dtype=np.float32)
            if args.choice_weight != 0.0:
                second_choice_scores = score_candidate_choice(
                    model,
                    tokenizer,
                    device,
                    question=question,
                    search_need=followup,
                    sentences=second_candidate_texts,
                )
            second_invalid_flags = invalid_status_flags(second_candidate_texts)
            second_fused_scores = (
                standardize(second_model_scores)
                + args.choice_weight * standardize(second_choice_scores)
                - args.invalid_status_penalty * second_invalid_flags
            )
            second_blocks, second_diagnostics = rerank_candidates(
                sentence_ids=second_sentence_ids,
                sentence_scores=second_lexical_scores,
                model_scores=second_fused_scores,
                sentence_blocks=sentence_blocks,
                bm25_weight=args.bm25_weight,
            )
            if second_blocks:
                second_block = second_blocks[0]
                iterative_prefix.append(second_block)
        source_record = source_record_by_start[int(query["block_start"])]
        method_rankings = {
            "model_validity_only": model_only_ranking,
            f"model_bm25_hybrid_{args.bm25_weight:g}": hybrid_ranking,
        }
        for threshold in completion_thresholds:
            prefix = iterative_prefix if completion_margin < threshold else [first_block]
            method_rankings[
                f"model_iterative_dynamic_{args.bm25_weight:g}_threshold_{threshold:g}"
            ] = prefix
            risk_prefix = [first_block]
            if completion_margin < threshold:
                risk_prefix = hybrid_blocks[:2]
                if second_block is not None:
                    risk_prefix.append(second_block)
            method_rankings[
                f"model_risk_top2_{args.bm25_weight:g}_threshold_{threshold:g}"
            ] = complete_ranking(risk_prefix, [])[: args.target_blocks]
        for method, ranking in method_rankings.items():
            ranked = ranking[: args.target_blocks]
            predicted_record = int(block_to_record[ranked[0]])
            row = evaluate_selection(
                method=method,
                query=query,
                ranked_ids=ranked,
                context_ids=group_for_context(ranked, block_to_record),
                predicted_record=predicted_record,
                source_record=source_record,
                record_margin=0.0,
            )
            row.update(
                {
                    "first_condition_block": first_block,
                    "second_condition_block": second_block if second_block is not None else -1,
                    "followup_query": followup,
                    "generated_followup_query": generated_followup,
                    "feedback_terms": feedback_terms,
                    "completion_margin": completion_margin,
                }
            )
            local_rows.append(row)
        local_diagnostics.append(
            {
                "query_id": int(query["query_id"]),
                "task_type": query.get("task_type", ""),
                "split": query.get("split", ""),
                "question": question,
                "routed_record": routed_record if routed_record is not None else -1,
                "first_premise": first_premise,
                "first_anchor_text": first_anchor_text,
                "anchor_only_used": use_anchor_only and bool(anchor_only_ids),
                "first_search_need": first_search_need,
                "completion_margin": completion_margin,
                "completion_thresholds": completion_thresholds,
                "followup_query": followup,
                "generated_followup_query": generated_followup,
                "feedback_terms": feedback_terms,
                "first_condition_block": first_block,
                "second_condition_block": second_block if second_block is not None else -1,
                "second_anchor_text": second_anchor_text if followup else "",
                "first_candidates": hybrid_diagnostics,
                "first_candidates_model_only": model_only_diagnostics,
                "first_binary_margins": first_model_scores.tolist(),
                "first_choice_logits": first_choice_scores.tolist(),
                "first_invalid_status_flags": first_invalid_flags.tolist(),
                "second_candidates": second_diagnostics,
            }
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    local_elapsed = time.perf_counter() - started
    local_peak_gib = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    gathered_rows: list[list[dict[str, Any]]] = [list() for _ in range(world_size)]
    gathered_diagnostics: list[list[dict[str, Any]]] = [list() for _ in range(world_size)]
    gathered_stats: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    local_stats = {
        "rank": rank,
        "local_rank": local_rank,
        "queries": len(local_diagnostics),
        "lexical_score_seconds": lexical_score_seconds,
        "elapsed_seconds": local_elapsed,
        "peak_memory_gib": local_peak_gib,
    }
    if world_size > 1:
        dist.all_gather_object(gathered_rows, local_rows)
        dist.all_gather_object(gathered_diagnostics, local_diagnostics)
        dist.all_gather_object(gathered_stats, local_stats)
    else:
        gathered_rows[0] = local_rows
        gathered_diagnostics[0] = local_diagnostics
        gathered_stats[0] = local_stats

    if rank == 0:
        rows = sorted(
            [row for part in gathered_rows for row in part],
            key=lambda row: (int(row["query_id"]), str(row["method"])),
        )
        diagnostics = sorted(
            [row for part in gathered_diagnostics for row in part],
            key=lambda row: int(row["query_id"]),
        )
        summaries = summarize(rows)
        write_csv(output_dir / "query_results.csv", rows, list(rows[0]))
        with (output_dir / "route_diagnostics.jsonl").open("w", encoding="utf-8") as f:
            for row in diagnostics:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_csv(output_dir / "method_summary.csv", summaries, list(summaries[0]))
        summary = {
            "source": (
                "Qwen validity scoring plus exact-IDF anchors and novel-entity feedback"
            ),
            "contains_synthetic_vectors": False,
            "queries": len(queries),
            "blocks": block_count,
            "sentences": len(sentences),
            "candidate_blocks": args.candidate_blocks,
            "anchor_candidate_blocks": args.anchor_candidate_blocks,
            "anchor_terms": args.anchor_terms,
            "anchor_only_mode": args.anchor_only_mode,
            "target_blocks": args.target_blocks,
            "retrieved_token_limit": args.target_blocks * int(blocks.shape[1]),
            "bm25_weight": args.bm25_weight,
            "choice_weight": args.choice_weight,
            "invalid_status_penalty": args.invalid_status_penalty,
            "record_routing_csv": args.record_routing_csv or "",
            "record_routing_field": args.record_routing_field,
            "index_cache_path": str(cache_path) if cache_path else "",
            "index_cache_hit": cache_hit,
            "completion_thresholds": completion_thresholds,
            "diagnostic_model_followup": args.diagnostic_model_followup,
            "world_size": world_size,
            "rank_stats": gathered_stats,
            "wall_seconds": max(float(item["elapsed_seconds"]) for item in gathered_stats if item),
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
