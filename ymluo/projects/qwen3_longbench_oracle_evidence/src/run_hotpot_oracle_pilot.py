#!/usr/bin/env python3
"""Run a small, auditable LongBench HotpotQA gold-evidence pilot.

The program joins LongBench rows to the original HotpotQA validation set and
uses the original sentence-level supporting-fact annotations.  It deliberately
does not derive evidence from the gold answer string.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import string
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evidence_alignment import align_support_sentence


LONG_BENCH_PREFIX = (
    "Answer the question based on the given passages. Only give me the answer "
    "and do not output any other words.\n\nThe following are given passages.\n"
)
LONG_BENCH_SUFFIX = (
    "\n\nAnswer the question based on the given passages. Only give me the answer "
    "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
)
WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Document:
    title: str
    sentences: tuple[str, ...]
    index: int
    char_start: int = -1
    char_end: int = -1


@dataclass
class EligibleExample:
    sample_id: str
    question: str
    answers: list[str]
    reported_length: int
    full_context: str
    documents: list[Document]
    support_pairs: list[tuple[str, int]]
    support_titles: list[str]
    support_sentences: list[tuple[str, int, str]]
    support_alignment_records: list[dict[str, Any]]
    evidence_position_fraction: float
    evidence_position_bin: str
    full_prompt_tokens: int
    original_hotpot_id: str
    hotpot_type: str
    hotpot_level: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--longbench_jsonl", required=True, type=Path)
    parser.add_argument("--hotpot_parquet", default="", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--sample_seed", type=int, default=20260802)
    parser.add_argument(
        "--exclude_manifest",
        default="",
        type=Path,
        help="Optional prior sample_manifest.jsonl whose sample IDs are excluded.",
    )
    parser.add_argument(
        "--alignment_mode",
        choices=("sentence_exact", "sentence_semantic", "support_title"),
        default="sentence_exact",
    )
    parser.add_argument(
        "--sample_strategy",
        choices=("position_length", "type_position_random"),
        default="position_length",
    )
    parser.add_argument("--min_prompt_tokens", type=int, default=6000)
    parser.add_argument("--max_prompt_tokens", type=int, default=16384)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--random_seeds", default="0,1,2")
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn_implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--skip_gold_nll", action="store_true")
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def normalize_question(text: str) -> str:
    return normalize_space(text).casefold()


def normalize_title(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).replace("_", " ")
    return normalize_space(normalized).casefold()


def normalize_answer(text: str) -> str:
    lowered = str(text).lower()
    lowered = "".join(ch for ch in lowered if ch not in set(string.punctuation))
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    return " ".join(lowered.split())


def qa_f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(answer).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / max(1, len(pred_tokens))
    recall = same / max(1, len(gold_tokens))
    return 2.0 * precision * recall / max(precision + recall, 1.0e-12)


def official_score(prediction: str, answers: Sequence[str]) -> float:
    return max((qa_f1(prediction, answer) for answer in answers), default=0.0)


def normalized_exact_match(prediction: str, answers: Sequence[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(answer) for answer in answers)


def contains_answer(text: str, answers: Sequence[str]) -> bool:
    haystack = normalize_answer(text)
    return any(
        bool(needle) and re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack)
        for needle in (normalize_answer(answer) for answer in answers)
    )


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def raw_prompt(question: str, context: str) -> str:
    return LONG_BENCH_PREFIX + context + LONG_BENCH_SUFFIX.format(input=question)


def chat_prompt(tokenizer: Any, question: str, context: str) -> str:
    content = raw_prompt(question, context)
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Frozen Qwen3 fallback used by the existing LongBench infrastructure.
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def parse_struct_pairs(value: Any, first: str, second: str) -> list[tuple[Any, Any]]:
    """Read a HF parquet struct represented as dict-of-lists or list-of-dicts."""
    if isinstance(value, dict):
        return list(zip(list(value.get(first) or []), list(value.get(second) or [])))
    if isinstance(value, list):
        output = []
        for item in value:
            if isinstance(item, dict):
                output.append((item[first], item[second]))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                output.append((item[0], item[1]))
        return output
    raise TypeError(f"unsupported struct value: {type(value)!r}")


def parse_documents(value: Any) -> list[Document]:
    pairs = parse_struct_pairs(value, "title", "sentences")
    output = []
    for index, (title, sentences) in enumerate(pairs):
        if isinstance(sentences, str):
            sentences = [sentences]
        output.append(
            Document(
                title=str(title),
                sentences=tuple(str(sentence) for sentence in sentences),
                index=index,
            )
        )
    return output


def parse_longbench_passages(context: str) -> list[Document]:
    marker = re.compile(r"(?m)^Passage\s+(\d+):\s*\r?\n")
    matches = list(marker.finditer(context))
    output: list[Document] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        block = context[match.end() : end].strip()
        lines = block.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        if len(lines) < 2 or not lines[0].strip():
            continue
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        output.append(
            Document(
                title=title,
                sentences=(body,),
                index=index,
                char_start=match.start(),
                char_end=end,
            )
        )
    return output


def load_hotpot_rows(path_text: str | Path) -> list[dict[str, Any]]:
    path = Path(path_text) if str(path_text) not in {"", "."} else None
    if path is None or not path.exists():
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id="hotpotqa/hotpot_qa",
                filename="distractor/validation-00000-of-00001.parquet",
                repo_type="dataset",
            )
        )
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the official HotpotQA parquet") from exc
    return parquet.read_table(path).to_pylist()


def load_longbench_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def serialize_document(document: Document, sentences: Sequence[str] | None = None) -> str:
    body = normalize_space(" ".join(sentences if sentences is not None else document.sentences))
    return f"[{document.title}]\n{body}"


def serialize_documents(documents: Sequence[Document]) -> str:
    return "\n\n".join(serialize_document(document) for document in documents)


def serialize_support_sentences(example: EligibleExample) -> str:
    by_title: dict[str, list[tuple[int, str]]] = {}
    for title, sentence_id, sentence in example.support_sentences:
        by_title.setdefault(title, []).append((sentence_id, sentence))
    documents_by_title = {
        normalize_title(document.title): document for document in example.documents
    }
    blocks = []
    for title in sorted(
        by_title, key=lambda item: documents_by_title[normalize_title(item)].index
    ):
        sentences = [sentence for _, sentence in sorted(by_title[title])]
        document = documents_by_title[normalize_title(title)]
        blocks.append(serialize_document(document, sentences))
    return "\n\n".join(blocks)


def evidence_position(full_context: str, evidence_sentences: Sequence[str]) -> float:
    haystack = normalize_space(full_context).casefold()
    positions = []
    for sentence in evidence_sentences:
        position = haystack.find(normalize_space(sentence).casefold())
        if position >= 0:
            positions.append(position + len(sentence) / 2.0)
    if not positions or not haystack:
        return -1.0
    return float(sum(positions) / len(positions) / len(haystack))


def evidence_document_position(full_context: str, documents: Sequence[Document]) -> float:
    centers = [
        (document.char_start + document.char_end) / 2.0
        for document in documents
        if document.char_start >= 0 and document.char_end > document.char_start
    ]
    if not centers or not full_context:
        return -1.0
    return float(sum(centers) / len(centers) / len(full_context))


def best_support_sentence_similarity(source_sentence: str, document: Document) -> float:
    body = " ".join(document.sentences)
    candidates = [
        candidate.strip()
        for candidate in re.split(r"\n+|(?<=[.!?])\s+", body)
        if candidate.strip()
    ]
    if not candidates:
        return 0.0
    source = normalize_space(source_sentence).casefold()
    return max(
        SequenceMatcher(None, source, normalize_space(candidate).casefold()).ratio()
        for candidate in candidates
    )


def position_bin(fraction: float) -> str:
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "middle"
    return "late"


def build_eligible_examples(
    longbench_rows: Sequence[dict[str, Any]],
    hotpot_rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[list[EligibleExample], list[dict[str, Any]]]:
    hotpot_by_question: dict[str, list[dict[str, Any]]] = {}
    for row in hotpot_rows:
        hotpot_by_question.setdefault(normalize_question(row["question"]), []).append(row)

    eligible: list[EligibleExample] = []
    audit: list[dict[str, Any]] = []
    for row_index, row in enumerate(longbench_rows):
        sample_id = str(row.get("_id", row_index))
        question = str(row["input"])
        matches = hotpot_by_question.get(normalize_question(question), [])
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "question": question,
            "reported_length": int(row.get("length", 0) or 0),
            "match_count": len(matches),
            "eligible": False,
            "rejection_reason": "",
        }
        if len(matches) != 1:
            record["rejection_reason"] = "question_join_not_unique"
            audit.append(record)
            continue
        source = matches[0]
        source_documents = parse_documents(source["context"])
        source_document_by_title = {
            normalize_title(document.title): document for document in source_documents
        }
        documents = parse_longbench_passages(str(row["context"]))
        record["longbench_passage_count"] = len(documents)
        if not documents:
            record["rejection_reason"] = "longbench_passage_parse_failed"
            audit.append(record)
            continue
        longbench_by_title: dict[str, list[Document]] = {}
        for document in documents:
            longbench_by_title.setdefault(normalize_title(document.title), []).append(document)
        support_pairs = [
            (str(title), int(sentence_id))
            for title, sentence_id in parse_struct_pairs(
                source["supporting_facts"], "title", "sent_id"
            )
        ]
        source_support_sentences: list[tuple[str, int, str]] = []
        invalid_pair = False
        for title, sentence_id in support_pairs:
            document = source_document_by_title.get(normalize_title(title))
            if document is None or not (0 <= sentence_id < len(document.sentences)):
                invalid_pair = True
                break
            source_support_sentences.append(
                (title, sentence_id, document.sentences[sentence_id])
            )
        if invalid_pair or not source_support_sentences:
            record["rejection_reason"] = "support_index_invalid"
            audit.append(record)
            continue
        support_titles_unordered = list(dict.fromkeys(title for title, _ in support_pairs))
        missing_titles = [
            title
            for title in support_titles_unordered
            if len(longbench_by_title.get(normalize_title(title), [])) == 0
        ]
        duplicate_titles = [
            title
            for title in support_titles_unordered
            if len(longbench_by_title.get(normalize_title(title), [])) > 1
        ]
        record["missing_support_title_count"] = len(missing_titles)
        record["duplicate_support_title_count"] = len(duplicate_titles)
        if missing_titles:
            record["missing_support_titles"] = missing_titles
            record["rejection_reason"] = "support_title_missing_from_longbench"
            audit.append(record)
            continue
        if duplicate_titles:
            record["duplicate_support_titles"] = duplicate_titles
            record["rejection_reason"] = "support_title_not_unique"
            audit.append(record)
            continue
        matched_support_documents = [
            longbench_by_title[normalize_title(title)][0]
            for title in support_titles_unordered
        ]
        matched_support_documents.sort(key=lambda document: document.index)
        support_titles = [document.title for document in matched_support_documents]
        support_alignment_records: list[dict[str, Any]] = []
        current_support_sentences: list[tuple[str, int, str]] = []
        for title, sentence_id, source_sentence in source_support_sentences:
            matched_document = longbench_by_title[normalize_title(title)][0]
            alignment = align_support_sentence(
                source_sentence, " ".join(matched_document.sentences)
            )
            support_alignment_records.append(
                {
                    "title": matched_document.title,
                    "source_sentence_id": sentence_id,
                    **alignment.to_dict(),
                }
            )
            if alignment.matched and alignment.matched_text:
                current_support_sentences.append(
                    (matched_document.title, sentence_id, alignment.matched_text)
                )
        context_normalized = normalize_space(str(row["context"])).casefold()
        missing_sentences = [
            sentence
            for _, _, sentence in source_support_sentences
            if normalize_space(sentence).casefold() not in context_normalized
        ]
        record["support_fact_count"] = len(source_support_sentences)
        record["exact_support_sentence_match_count"] = (
            len(source_support_sentences) - len(missing_sentences)
        )
        record["missing_support_count"] = len(missing_sentences)
        similarity_values = []
        for title, _, sentence in source_support_sentences:
            matched_document = longbench_by_title[normalize_title(title)][0]
            similarity_values.append(
                best_support_sentence_similarity(sentence, matched_document)
            )
        record["minimum_support_sentence_similarity"] = min(similarity_values)
        record["mean_support_sentence_similarity"] = sum(similarity_values) / len(
            similarity_values
        )
        record["support_alignment_records"] = support_alignment_records
        record["semantic_support_match_count"] = sum(
            int(item["matched"]) for item in support_alignment_records
        )
        record["semantic_support_failure_codes"] = [
            item["failure_code"]
            for item in support_alignment_records
            if not item["matched"]
        ]
        if args.alignment_mode == "sentence_exact" and missing_sentences:
            record["rejection_reason"] = "support_sentence_missing_from_longbench"
            audit.append(record)
            continue
        if (
            args.alignment_mode == "sentence_semantic"
            and len(current_support_sentences) != len(source_support_sentences)
        ):
            record["rejection_reason"] = "support_sentence_semantic_alignment_failed"
            audit.append(record)
            continue
        support_sentences = (
            current_support_sentences
            if args.alignment_mode == "sentence_semantic"
            else source_support_sentences
        )
        if len(support_titles) < 2:
            record["rejection_reason"] = "fewer_than_two_support_documents"
            audit.append(record)
            continue
        prompt_text = chat_prompt(tokenizer, question, str(row["context"]))
        full_prompt_tokens = len(token_ids(tokenizer, prompt_text))
        record["full_prompt_tokens"] = full_prompt_tokens
        if full_prompt_tokens < args.min_prompt_tokens:
            record["rejection_reason"] = "prompt_too_short"
            audit.append(record)
            continue
        if full_prompt_tokens > args.max_prompt_tokens:
            record["rejection_reason"] = "prompt_too_long"
            audit.append(record)
            continue
        if args.alignment_mode in {"sentence_exact", "sentence_semantic"}:
            fraction = evidence_position(
                str(row["context"]), [sentence for _, _, sentence in support_sentences]
            )
        else:
            fraction = evidence_document_position(
                str(row["context"]), matched_support_documents
            )
        if fraction < 0:
            record["rejection_reason"] = "evidence_position_unresolved"
            audit.append(record)
            continue
        example = EligibleExample(
            sample_id=sample_id,
            question=question,
            answers=[str(answer) for answer in row["answers"]],
            reported_length=int(row.get("length", 0) or 0),
            full_context=str(row["context"]),
            documents=documents,
            support_pairs=support_pairs,
            support_titles=support_titles,
            support_sentences=support_sentences,
            support_alignment_records=support_alignment_records,
            evidence_position_fraction=fraction,
            evidence_position_bin=position_bin(fraction),
            full_prompt_tokens=full_prompt_tokens,
            original_hotpot_id=str(source.get("id", "")),
            hotpot_type=str(source.get("type", "")),
            hotpot_level=str(source.get("level", "")),
        )
        record.update(
            {
                "eligible": True,
                "rejection_reason": "",
                "evidence_position_fraction": fraction,
                "evidence_position_bin": example.evidence_position_bin,
                "support_titles": support_titles,
                "original_hotpot_id": example.original_hotpot_id,
                "alignment_mode": args.alignment_mode,
            }
        )
        eligible.append(example)
        audit.append(record)
    return eligible, audit


def freeze_sample(
    eligible: Sequence[EligibleExample],
    count: int,
    sample_seed: int,
    sample_strategy: str,
) -> list[EligibleExample]:
    if sample_strategy == "position_length":
        bins: dict[str, list[EligibleExample]] = {
            "early": [],
            "middle": [],
            "late": [],
        }
        for example in eligible:
            bins[example.evidence_position_bin].append(example)
        for values in bins.values():
            values.sort(
                key=lambda item: (
                    -item.full_prompt_tokens,
                    stable_hash(f"{sample_seed}:{item.sample_id}"),
                )
            )
        ordered_keys: list[Any] = ["early", "middle", "late"]
    elif sample_strategy == "type_position_random":
        bins = {}
        for example in eligible:
            key = (example.hotpot_type, example.evidence_position_bin)
            bins.setdefault(key, []).append(example)
        for key, values in bins.items():
            values.sort(
                key=lambda item: stable_hash(
                    f"{sample_seed}:{key[0]}:{key[1]}:{item.sample_id}"
                )
            )
        type_order = {"bridge": 0, "comparison": 1}
        position_order = {"early": 0, "middle": 1, "late": 2}
        ordered_keys = sorted(
            bins,
            key=lambda key: (
                type_order.get(key[0], 99),
                position_order.get(key[1], 99),
                str(key),
            ),
        )
    else:
        raise ValueError(f"unsupported sample strategy: {sample_strategy}")
    selected: list[EligibleExample] = []
    cursors = {key: 0 for key in bins}
    while len(selected) < count:
        progress = False
        for key in ordered_keys:
            cursor = cursors[key]
            if cursor < len(bins[key]) and len(selected) < count:
                selected.append(bins[key][cursor])
                cursors[key] += 1
                progress = True
        if not progress:
            break
    if len(selected) < count:
        raise RuntimeError(f"only {len(selected)} eligible samples for requested {count}")
    return selected


def bm25_scores(question: str, documents: Sequence[Document]) -> list[float]:
    tokenized_docs = [
        [token.lower() for token in WORD_RE.findall(document.title + " " + " ".join(document.sentences))]
        for document in documents
    ]
    query_tokens = [token.lower() for token in WORD_RE.findall(question)]
    n_docs = len(documents)
    doc_frequency = Counter()
    for tokens in tokenized_docs:
        doc_frequency.update(set(tokens))
    lengths = [len(tokens) for tokens in tokenized_docs]
    average_length = max(sum(lengths) / max(1, n_docs), 1.0)
    scores = []
    for tokens, length in zip(tokenized_docs, lengths):
        counts = Counter(tokens)
        score = 0.0
        for term in query_tokens:
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            df = doc_frequency.get(term, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.5 * (1.0 - 0.75 + 0.75 * length / average_length)
            score += idf * frequency * 2.5 / denominator
        scores.append(score)
    return scores


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def token_budget_prefix(
    tokenizer: Any, text: str, target_tokens: int
) -> tuple[str, int, int]:
    """Return a deterministic natural-text prefix within 5% of a token budget."""
    ids = token_ids(tokenizer, text)
    if not ids:
        raise RuntimeError("random distractor pool is empty")
    repetitions = max(1, math.ceil(target_tokens / len(ids)))
    if repetitions > 1:
        text = "\n\n".join(text for _ in range(repetitions))
        ids = token_ids(tokenizer, text)
    context = tokenizer.decode(
        ids[:target_tokens],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    actual_tokens = len(token_ids(tokenizer, context))
    tolerance = max(1, math.ceil(0.05 * target_tokens))
    if abs(actual_tokens - target_tokens) > tolerance:
        raise RuntimeError(
            "decoded random control exceeds 5% token-budget tolerance: "
            f"actual={actual_tokens}, target={target_tokens}"
        )
    return context, actual_tokens, repetitions


def build_condition_contexts(
    example: EligibleExample,
    tokenizer: Any,
    random_seeds: Sequence[int],
    random_document_pool: Sequence[Document] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    support_set = {normalize_title(title) for title in example.support_titles}
    support_documents = [
        doc for doc in example.documents if normalize_title(doc.title) in support_set
    ]
    oracle_sentence = serialize_support_sentences(example)
    oracle_document = serialize_documents(support_documents)
    contexts: dict[str, str] = {
        "full": example.full_context,
        "oracle_sentence": oracle_sentence,
        "oracle_document": oracle_document,
        "query_only": "",
    }
    support_count = len(support_documents)
    scores = bm25_scores(example.question, example.documents)
    bm25_ids = sorted(
        range(len(example.documents)), key=lambda index: (-scores[index], index)
    )[:support_count]
    contexts["bm25_document"] = serialize_documents(
        [example.documents[index] for index in sorted(bm25_ids)]
    )

    target_tokens = len(token_ids(tokenizer, oracle_document))
    random_source = random_document_pool or example.documents
    non_support = [
        document
        for document in random_source
        if normalize_title(document.title) not in support_set
    ]
    if not non_support:
        raise RuntimeError(f"no non-support documents for {example.sample_id}")
    random_choices: dict[str, Any] = {}
    for seed in random_seeds:
        ordered = sorted(
            non_support,
            key=lambda document: stable_hash(
                f"{example.sample_id}:{seed}:random-document:{document.title}"
            ),
        )
        pool = serialize_documents(ordered)
        random_context, actual_tokens, repetitions = token_budget_prefix(
            tokenizer, pool, target_tokens
        )
        name = f"random_document_seed{seed}"
        contexts[name] = random_context
        random_choices[name] = {
            "source_title_order": [document.title for document in ordered],
            "context_tokens": actual_tokens,
            "target_context_tokens": target_tokens,
            "absolute_budget_error": abs(actual_tokens - target_tokens),
            "relative_budget_error": abs(actual_tokens - target_tokens)
            / max(1, target_tokens),
            "is_token_truncated_natural_prefix": True,
            "source_pool_repetitions": repetitions,
            "contains_answer": contains_answer(random_context, example.answers),
        }
    audit = {
        "support_titles": example.support_titles,
        "support_pairs": example.support_pairs,
        "support_sentences": example.support_sentences,
        "support_alignment_records": example.support_alignment_records,
        "bm25_titles": [example.documents[index].title for index in sorted(bm25_ids)],
        "random_choices": random_choices,
        "random_pool_document_count": len(non_support),
        "oracle_sentence_tokens": len(token_ids(tokenizer, oracle_sentence)),
        "oracle_document_tokens": target_tokens,
        "full_context_tokens": len(token_ids(tokenizer, example.full_context)),
    }
    return contexts, audit


def model_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


@torch.inference_mode()
def forward_last(
    model: Any, input_ids: torch.Tensor, past_key_values: Any | None = None
) -> tuple[torch.Tensor, Any]:
    outputs = model.model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state[:, -1:, :]
    logits = model.lm_head(hidden)[:, -1, :].float()
    return logits, outputs.past_key_values


def eos_ids(tokenizer: Any) -> set[int]:
    output: set[int] = set()
    if tokenizer.eos_token_id is not None:
        output.add(int(tokenizer.eos_token_id))
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            output.add(token_id)
    return output


@torch.inference_mode()
def greedy_generate(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, list[int], float]:
    started = time.perf_counter()
    logits, cache = forward_last(
        model, torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    )
    generated: list[int] = []
    stops = eos_ids(tokenizer)
    for step in range(max_new_tokens):
        next_id = int(torch.argmax(logits, dim=-1).item())
        if next_id in stops:
            break
        generated.append(next_id)
        if step + 1 == max_new_tokens:
            break
        logits, cache = forward_last(
            model,
            torch.tensor([[next_id]], dtype=torch.long, device=device),
            cache,
        )
    elapsed = time.perf_counter() - started
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    del cache, logits
    return text, generated, elapsed


@torch.inference_mode()
def gold_answer_nll(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    answer: str,
    device: torch.device,
) -> tuple[float, int, float]:
    surfaces = [answer]
    if normalize_answer(answer) in {"yes", "no"}:
        surfaces.append(answer.strip().capitalize())
    candidate_ids = []
    for surface in dict.fromkeys(surfaces):
        ids = token_ids(tokenizer, surface)
        if ids:
            candidate_ids.append(ids)
    if not candidate_ids:
        return float("nan"), 0, 0.0
    started = time.perf_counter()
    logits, cache = forward_last(
        model, torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    )
    # LongBench scores yes/no case-insensitively.  Qwen often emits ``Yes``
    # while the dataset stores ``yes``; use the lowest NLL over those equivalent
    # one-token surfaces instead of assigning a spurious ~20-nat penalty.
    if all(len(ids) == 1 for ids in candidate_ids):
        log_probabilities = torch.log_softmax(logits, dim=-1)[0]
        losses = [float(-log_probabilities[ids[0]].item()) for ids in candidate_ids]
        elapsed = time.perf_counter() - started
        best = min(range(len(losses)), key=losses.__getitem__)
        del cache, logits
        return losses[best], 1, elapsed
    answer_ids = candidate_ids[0]
    losses: list[float] = []
    for index, answer_id in enumerate(answer_ids):
        losses.append(float(-torch.log_softmax(logits, dim=-1)[0, answer_id].item()))
        if index + 1 < len(answer_ids):
            logits, cache = forward_last(
                model,
                torch.tensor([[answer_id]], dtype=torch.long, device=device),
                cache,
            )
    elapsed = time.perf_counter() - started
    del cache, logits
    return sum(losses) / len(losses), len(answer_ids), elapsed


def json_safe_example(example: EligibleExample) -> dict[str, Any]:
    return {
        "sample_id": example.sample_id,
        "original_hotpot_id": example.original_hotpot_id,
        "question": example.question,
        "answers": example.answers,
        "reported_length": example.reported_length,
        "full_prompt_tokens": example.full_prompt_tokens,
        "evidence_position_fraction": example.evidence_position_fraction,
        "evidence_position_bin": example.evidence_position_bin,
        "hotpot_type": example.hotpot_type,
        "hotpot_level": example.hotpot_level,
        "support_titles": example.support_titles,
        "support_pairs": example.support_pairs,
        "support_alignment_match_types": [
            record["match_type"] for record in example.support_alignment_records
        ],
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def excluded_sample_ids(path_text: str | Path) -> set[str]:
    path = Path(path_text) if str(path_text) not in {"", "."} else None
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(f"exclude manifest not found: {path}")
    return {
        str(json.loads(line)["sample_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> None:
    args = parse_args()
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("shard_index must be in [0, shard_count)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random_seeds = [int(item) for item in args.random_seeds.split(",") if item.strip()]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, use_fast=True
    )
    longbench_rows = load_longbench_rows(args.longbench_jsonl)
    hotpot_rows = load_hotpot_rows(args.hotpot_parquet)
    eligible, alignment_audit = build_eligible_examples(
        longbench_rows, hotpot_rows, tokenizer, args
    )
    excluded_ids = excluded_sample_ids(args.exclude_manifest)
    eligible_before_exclusion = len(eligible)
    eligible = [example for example in eligible if example.sample_id not in excluded_ids]
    write_jsonl(args.output_dir / "alignment_audit.jsonl", alignment_audit)
    selected = freeze_sample(
        eligible, args.max_samples, args.sample_seed, args.sample_strategy
    )
    write_jsonl(
        args.output_dir / "sample_manifest.jsonl",
        [json_safe_example(example) for example in selected],
    )
    manifest_hash = hashlib.sha256(
        (args.output_dir / "sample_manifest.jsonl").read_bytes()
    ).hexdigest()
    run_meta = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "manifest_sha256": manifest_hash,
        "eligible_before_exclusion": eligible_before_exclusion,
        "eligible_count": len(eligible),
        "excluded_sample_count": eligible_before_exclusion - len(eligible),
        "selected_count": len(selected),
        "model_revision": Path(args.model_name_or_path).name,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    (args.output_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shard_examples = [
        example
        for index, example in enumerate(selected)
        if index % args.shard_count == args.shard_index
    ]
    evidence_rows = []
    prepared: list[tuple[EligibleExample, dict[str, str]]] = []
    random_pool_by_hash: dict[str, Document] = {}
    for selected_example in selected:
        for document in selected_example.documents:
            document_hash = hashlib.sha256(
                serialize_document(document).encode("utf-8")
            ).hexdigest()
            random_pool_by_hash.setdefault(document_hash, document)
    random_document_pool = list(random_pool_by_hash.values())
    for example in shard_examples:
        contexts, evidence_audit = build_condition_contexts(
            example, tokenizer, random_seeds, random_document_pool
        )
        evidence_rows.append({"sample_id": example.sample_id, **evidence_audit})
        prepared.append((example, contexts))
    write_jsonl(args.output_dir / "evidence_mapping.jsonl", evidence_rows)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "eligible": len(eligible),
                    "selected": len(selected),
                    "shard_selected": len(shard_examples),
                    "manifest_sha256": manifest_hash,
                },
                ensure_ascii=False,
            )
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-8B inference")
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=model_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    predictions_path = args.output_dir / "predictions.jsonl"
    if predictions_path.exists():
        predictions_path.unlink()
    completed = 0
    for example, contexts in prepared:
        for condition, context in contexts.items():
            prompt_text = chat_prompt(tokenizer, example.question, context)
            prompt_ids = token_ids(tokenizer, prompt_text)
            if len(prompt_ids) > args.max_prompt_tokens:
                raise RuntimeError(
                    f"condition {condition} for {example.sample_id} exceeds token limit: {len(prompt_ids)}"
                )
            prediction, generated_ids, generation_seconds = greedy_generate(
                model, tokenizer, prompt_ids, args.max_new_tokens, device
            )
            nll = float("nan")
            answer_token_count = 0
            nll_seconds = 0.0
            if not args.skip_gold_nll:
                nll, answer_token_count, nll_seconds = gold_answer_nll(
                    model, tokenizer, prompt_ids, example.answers[0], device
                )
            record = {
                "sample_id": example.sample_id,
                "original_hotpot_id": example.original_hotpot_id,
                "condition": condition,
                "question": example.question,
                "answers": example.answers,
                "prediction": prediction,
                "generated_token_ids": generated_ids,
                "official_qa_f1": official_score(prediction, example.answers),
                "normalized_exact_match": normalized_exact_match(prediction, example.answers),
                "prediction_contains_answer": contains_answer(prediction, example.answers),
                "selected_context_contains_answer": contains_answer(context, example.answers),
                "context_tokens": len(token_ids(tokenizer, context)),
                "prompt_tokens": len(prompt_ids),
                "full_prompt_tokens": example.full_prompt_tokens,
                "compression_ratio": len(prompt_ids) / example.full_prompt_tokens,
                "gold_answer_mean_nll": nll,
                "gold_answer_ppl": math.exp(min(nll, 30.0)) if math.isfinite(nll) else float("nan"),
                "gold_answer_tokens": answer_token_count,
                "generation_seconds": generation_seconds,
                "nll_seconds": nll_seconds,
                "evidence_position_fraction": example.evidence_position_fraction,
                "evidence_position_bin": example.evidence_position_bin,
                "hotpot_type": example.hotpot_type,
                "hotpot_level": example.hotpot_level,
                "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                "context_preview": context[:240],
            }
            append_jsonl(predictions_path, record)
            completed += 1
            if completed % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "sample_id": example.sample_id,
                            "condition": condition,
                            "f1": record["official_qa_f1"],
                            "em": record["normalized_exact_match"],
                            "prediction": prediction,
                            "prompt_tokens": len(prompt_ids),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            gc.collect()
            torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": "complete",
                "rows": completed,
                "predictions": str(predictions_path),
                "manifest_sha256": manifest_hash,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
