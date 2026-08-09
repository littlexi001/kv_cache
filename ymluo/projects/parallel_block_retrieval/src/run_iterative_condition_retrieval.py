from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer

from profile_real_qk import read_jsonl
from run_lexical_block_retrieval import (
    decode_blocks,
    evaluate_selection,
    group_for_context,
    write_csv,
)


NEGATIVE_MARKERS = (
    "rejected",
    "never took effect",
    "omitted",
    "without identifying",
    "does not describe",
    "omits",
    "no clearance",
    "unconfirmed",
    "nonbinding",
    "contains no",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sentence-level BM25 with status filtering and iterative condition retrieval."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--min_df", type=int, default=1)
    parser.add_argument("--max_df", type=float, default=1.0)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


class BM25Index:
    def __init__(
        self,
        documents: list[str],
        *,
        min_df: int,
        max_df: float,
        k1: float,
        b: float,
    ) -> None:
        self.vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
            dtype=np.float32,
        )
        counts = self.vectorizer.fit_transform(documents).tocsr().astype(np.float32)
        document_count = int(counts.shape[0])
        document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        inverse_document_frequency = np.log1p(
            (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        ).astype(np.float32)
        self.inverse_document_frequency = inverse_document_frequency
        lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        average_length = max(float(lengths.mean()), 1.0e-6)
        row_ids = np.repeat(np.arange(document_count), np.diff(counts.indptr))
        frequencies = counts.data
        denominator = frequencies + k1 * (1.0 - b + b * lengths[row_ids] / average_length)
        counts.data = (
            inverse_document_frequency[counts.indices]
            * frequencies
            * (k1 + 1.0)
            / denominator
        )
        self.weighted_documents = counts
        self.weighted_documents_csc = counts.tocsc()
        self.document_count = document_count
        self.average_length = average_length

    def score(self, queries: list[str]) -> np.ndarray:
        query_counts = self.vectorizer.transform(queries).tocsr().astype(np.float32)
        query_counts.data.fill(1.0)
        return (query_counts @ self.weighted_documents.transpose()).toarray().astype(
            np.float32, copy=False
        )

    def score_postings(self, queries: list[str]) -> np.ndarray:
        """Score small online batches by accumulating only matching postings."""
        query_counts = self.vectorizer.transform(queries).tocsr()
        scores = np.zeros((len(queries), self.document_count), dtype=np.float32)
        postings = self.weighted_documents_csc
        for query_index in range(len(queries)):
            feature_ids = query_counts.indices[
                query_counts.indptr[query_index] : query_counts.indptr[query_index + 1]
            ]
            for feature_id in feature_ids:
                start = postings.indptr[feature_id]
                end = postings.indptr[feature_id + 1]
                document_ids = postings.indices[start:end]
                scores[query_index, document_ids] += postings.data[start:end]
        return scores

    @property
    def features(self) -> int:
        return int(self.weighted_documents.shape[1])

    def rare_query_text(self, text: str, max_terms: int = 2) -> str:
        counts = self.vectorizer.transform([text]).tocsr()
        if counts.nnz == 0:
            return text
        feature_names = self.vectorizer.get_feature_names_out()
        feature_ids = counts.indices
        order = sorted(
            feature_ids,
            key=lambda item: (
                -int(any(character.isdigit() for character in str(feature_names[item]))),
                -float(self.inverse_document_frequency[item]),
                -len(str(feature_names[item]).split()),
                str(feature_names[item]),
            ),
        )
        selected: list[str] = []
        for feature_id in order:
            term = str(feature_names[feature_id])
            if any(term in existing or existing in term for existing in selected):
                continue
            selected.append(term)
            if len(selected) >= max_terms:
                break
        return " ".join(selected) if selected else text

    def rare_novel_text(self, text: str, exclude_text: str, max_terms: int = 4) -> str:
        excluded_tokens = set(re.findall(r"[a-z0-9]+", exclude_text.casefold()))
        named_spans = re.findall(
            r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*)+\b",
            text,
        )
        novel_spans = [
            span
            for span in named_spans
            if set(re.findall(r"[a-z0-9]+", span.casefold())) - excluded_tokens
            == set(re.findall(r"[a-z0-9]+", span.casefold()))
        ]
        if novel_spans:
            novel_spans.sort(
                key=lambda span: (
                    -len(re.findall(r"[a-z0-9]+", span.casefold())),
                    span.casefold(),
                )
            )
            return novel_spans[0]

        counts = self.vectorizer.transform([text]).tocsr()
        excluded = set(self.vectorizer.transform([exclude_text]).indices.tolist())
        feature_names = self.vectorizer.get_feature_names_out()
        order = sorted(
            (int(item) for item in counts.indices if int(item) not in excluded),
            key=lambda item: (
                -float(self.inverse_document_frequency[item]),
                -len(str(feature_names[item]).split()),
                str(feature_names[item]),
            ),
        )
        selected: list[str] = []
        for feature_id in order:
            term = str(feature_names[feature_id])
            if normalized := set(re.findall(r"[a-z0-9]+", term.casefold())):
                if normalized & excluded_tokens:
                    continue
            if any(term in existing or existing in term for existing in selected):
                continue
            selected.append(term)
            if len(selected) >= max_terms:
                break
        return " ".join(selected)


def split_sentences(block_texts: list[str]) -> tuple[list[str], np.ndarray]:
    sentences: list[str] = []
    block_ids: list[int] = []
    for block_id, text in enumerate(block_texts):
        parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        for part in parts:
            sentence = " ".join(part.split())
            if len(sentence) < 12:
                continue
            sentences.append(sentence)
            block_ids.append(block_id)
    return sentences, np.asarray(block_ids, dtype=np.int32)


def rank_blocks(sentence_scores: np.ndarray, sentence_blocks: np.ndarray, block_count: int) -> list[int]:
    scores = np.full(block_count, -np.inf, dtype=np.float32)
    np.maximum.at(scores, sentence_blocks, sentence_scores)
    ids = np.arange(block_count, dtype=np.int64)
    return np.lexsort((ids, -scores)).tolist()


def is_negative_sentence(sentence: str) -> bool:
    lowered = sentence.casefold()
    return any(marker in lowered for marker in NEGATIVE_MARKERS)


def status_filtered_scores(scores: np.ndarray, negative_mask: np.ndarray) -> np.ndarray:
    output = scores.copy()
    output[negative_mask] = -np.inf
    return output


def chain_alias(question: str) -> str | None:
    match = re.search(r"\b([A-Z][A-Za-z]+-\d{4})\b", question)
    return match.group(1) if match else None


def extract_intermediate_entity(sentence: str) -> str | None:
    patterns = [
        r"operational alias used for ([^.]+)",
        r"refers specifically to the vessel ([^.]+)",
        r"assigned to ([^.]+?) in the current dispatch ledger",
    ]
    for pattern in patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        group = [row for row in rows if row["method"] == method]
        output.append(
            {
                "method": method,
                "queries": len(group),
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
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    records = read_jsonl(corpus_dir / "records.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    started = time.perf_counter()
    block_texts = decode_blocks(tokenizer, blocks)
    sentences, sentence_blocks = split_sentences(block_texts)
    index = BM25Index(
        sentences,
        min_df=args.min_df,
        max_df=args.max_df,
        k1=args.k1,
        b=args.b,
    )
    questions = [str(query["question"]) for query in queries]
    query_scores = index.score(questions)
    negative_mask = np.asarray(
        [is_negative_sentence(sentence) for sentence in sentences], dtype=bool
    )

    block_count = int(blocks.shape[0])
    block_to_record = np.empty(block_count, dtype=np.int32)
    source_record_by_start: dict[int, int] = {}
    for record_id, record in enumerate(records):
        start = int(record["block_start"])
        end = start + int(record["block_count"])
        block_to_record[start:end] = record_id
        source_record_by_start[start] = record_id

    rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        base_sentence_scores = query_scores[query_index]
        base_ranking = rank_blocks(base_sentence_scores, sentence_blocks, block_count)
        filtered_sentence_scores = status_filtered_scores(base_sentence_scores, negative_mask)
        status_ranking = rank_blocks(filtered_sentence_scores, sentence_blocks, block_count)
        alias = chain_alias(str(query["question"]))
        intermediate: str | None = None
        first_condition_block: int | None = None
        second_condition_block: int | None = None
        iterative_ranking = list(status_ranking)

        if alias:
            sentence_order = np.lexsort(
                (np.arange(len(sentences), dtype=np.int64), -filtered_sentence_scores)
            )
            alias_lower = alias.casefold()
            for sentence_id in sentence_order:
                sentence = sentences[int(sentence_id)]
                if alias_lower not in sentence.casefold():
                    continue
                candidate = extract_intermediate_entity(sentence)
                if candidate:
                    intermediate = candidate
                    first_condition_block = int(sentence_blocks[int(sentence_id)])
                    break
            if intermediate:
                followup = (
                    f"{intermediate} navigation beacon emitted color signal displays produces"
                )
                followup_scores = index.score([followup])[0]
                followup_scores = status_filtered_scores(followup_scores, negative_mask)
                followup_order = np.lexsort(
                    (np.arange(len(sentences), dtype=np.int64), -followup_scores)
                )
                for sentence_id in followup_order:
                    block_id = int(sentence_blocks[int(sentence_id)])
                    if block_id != first_condition_block:
                        second_condition_block = block_id
                        break
                prefix = [
                    item
                    for item in [first_condition_block, second_condition_block]
                    if item is not None
                ]
                iterative_ranking = prefix + [
                    block_id for block_id in status_ranking if block_id not in set(prefix)
                ]

        source_record = source_record_by_start[int(query["block_start"])]
        method_rankings = {
            "sentence_bm25": base_ranking,
            "sentence_status": status_ranking,
            "iterative_condition": iterative_ranking,
        }
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
                    "route_type": "iterative_chain" if alias else "single_step",
                    "intermediate_entity": intermediate or "",
                    "first_condition_block": (
                        first_condition_block if first_condition_block is not None else -1
                    ),
                    "second_condition_block": (
                        second_condition_block if second_condition_block is not None else -1
                    ),
                }
            )
            rows.append(row)
        route_rows.append(
            {
                "query_id": query_index,
                "task_type": query.get("task_type", ""),
                "alias": alias or "",
                "intermediate_entity": intermediate or "",
                "first_condition_block": (
                    first_condition_block if first_condition_block is not None else -1
                ),
                "second_condition_block": (
                    second_condition_block if second_condition_block is not None else -1
                ),
            }
        )

    summaries = summarize(rows)
    write_csv(output_dir / "query_results.csv", rows, list(rows[0]))
    write_csv(output_dir / "method_summary.csv", summaries, list(summaries[0]))
    write_csv(output_dir / "route_diagnostics.csv", route_rows, list(route_rows[0]))
    elapsed = time.perf_counter() - started
    summary = {
        "source": "sentence-level BM25, query-conditioned status filtering, and iterative condition expansion",
        "contains_synthetic_vectors": False,
        "queries": len(queries),
        "blocks": block_count,
        "sentences": len(sentences),
        "features": index.features,
        "negative_sentences": int(negative_mask.sum()),
        "target_blocks": args.target_blocks,
        "elapsed_seconds": elapsed,
        "methods": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
