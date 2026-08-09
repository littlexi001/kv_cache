from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from transformers import AutoTokenizer

from profile_real_qk import read_jsonl
from profile_step_state_q import step_state_text
from run_iterative_condition_retrieval import BM25Index
from run_lexical_block_retrieval import decode_blocks


def normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def lexical_stem(term: str) -> str:
    if len(term) > 4 and term.endswith("ied"):
        return term[:-3] + "y"
    if len(term) > 4 and term.endswith("ing"):
        return term[:-3]
    if len(term) > 3 and term.endswith("ed"):
        return term[:-2]
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term


def step_anchor_text(step: dict[str, Any]) -> str:
    state = [str(item) for item in step.get("compact_state_before", []) if str(item)]
    if state:
        return state[-1].split(":", 1)[-1].strip()
    return str(step.get("lookup_key", "")).strip()


class AnchorInvertedIndex:
    def __init__(self, documents: Sequence[str]) -> None:
        self.documents = documents
        postings: dict[str, list[int]] = {}
        for block_id, document in enumerate(documents):
            for term in set(normalized_tokens(document)):
                postings.setdefault(term, []).append(block_id)
        self.postings = postings
        self.document_count = len(documents)

    def _candidate_intersection(self, terms: Sequence[str]) -> set[int]:
        posting_lists = [self.postings[term] for term in terms if term in self.postings]
        if not posting_lists:
            return set()
        posting_lists.sort(key=len)
        candidates = set(posting_lists[0])
        for posting in posting_lists[1:]:
            candidates.intersection_update(posting)
            if not candidates:
                break
        return candidates

    def search(self, anchor: str, query: str, budget: int) -> list[int]:
        anchor_terms = list(dict.fromkeys(normalized_tokens(anchor)))
        posting_lists = [self.postings[term] for term in anchor_terms if term in self.postings]
        if not posting_lists:
            return []
        posting_lists.sort(key=len)
        candidates = set(posting_lists[0])
        for posting in posting_lists[1:]:
            candidates.intersection_update(posting)
            if not candidates:
                break
        if not candidates:
            candidates = set(posting_lists[0])
        anchor_phrase = " ".join(anchor_terms)
        query_terms = set(normalized_tokens(query)) - set(anchor_terms)
        scored = []
        for block_id in candidates:
            document_terms = normalized_tokens(self.documents[block_id])
            document_set = set(document_terms)
            phrase_hit = anchor_phrase in " ".join(document_terms)
            relation_overlap = len(query_terms & document_set)
            scored.append((int(phrase_hit), relation_overlap, block_id))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [item[2] for item in scored[:budget]]

    def search_alias_aware(self, anchor: str, query: str, budget: int) -> list[int]:
        anchor_terms = list(dict.fromkeys(normalized_tokens(anchor)))
        exact_ranked = self.search(anchor, query, budget)
        if len(anchor_terms) < 2 or len(anchor_terms[-1]) < 4:
            return exact_ranked
        alias_candidates = self._candidate_intersection([anchor_terms[-1]]) - set(
            exact_ranked
        )
        if not alias_candidates:
            return exact_ranked
        query_terms = set(normalized_tokens(query)) - set(anchor_terms)
        query_stems = {lexical_stem(term) for term in query_terms}
        scored = []
        for block_id in alias_candidates:
            document_terms = normalized_tokens(self.documents[block_id])
            document_set = set(document_terms)
            document_stems = {lexical_stem(term) for term in document_set}
            relation_score = sum(
                math.log((self.document_count + 1.0) / (len(self.postings[term]) + 1.0))
                for term in query_terms & document_set
                if term in self.postings
            )
            stem_overlap = len(query_stems & document_stems)
            score = relation_score + 2.0 * stem_overlap
            scored.append((score, block_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return (exact_ranked + [item[1] for item in scored])[:budget]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate operator-aware lexical and Q/K candidate union on 10M blocks."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument(
        "--qk_rows_path",
        default="",
        help="Optional full-index Q/K rows. Omit for lexical/anchor-only candidate creation.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--task_types", default="multihop")
    parser.add_argument("--exclude_query_ids", default="")
    parser.add_argument("--candidate_blocks", type=int, default=512)
    parser.add_argument("--lexical_share", type=int, default=256)
    parser.add_argument(
        "--lexical_backend",
        choices=["auto", "matrix", "postings"],
        default="auto",
        help="Use postings for small online batches and matrix scoring for large batches.",
    )
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--anchor_alias_fallback", action="store_true")
    return parser.parse_args()


def stable_rank(scores: np.ndarray, budget: int) -> list[int]:
    block_ids = np.arange(len(scores), dtype=np.int64)
    if budget >= len(scores):
        return np.lexsort((block_ids, -scores)).tolist()
    candidate_ids = np.argpartition(scores, -budget)[-budget:]
    order = np.lexsort((candidate_ids, -scores[candidate_ids]))
    return candidate_ids[order].astype(np.int64).tolist()


def rank_or_zero(values: Sequence[int], target: int) -> int:
    try:
        return values.index(target) + 1
    except ValueError:
        return 0


def reciprocal_rank_fusion(
    ranked_groups: Sequence[Sequence[int]],
    budget: int,
    rrf_k: float,
) -> list[int]:
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for group in ranked_groups:
        for rank, block_id_value in enumerate(group, start=1):
            block_id = int(block_id_value)
            scores[block_id] = scores.get(block_id, 0.0) + 1.0 / (rrf_k + rank)
            best_rank[block_id] = min(best_rank.get(block_id, rank), rank)
    return sorted(scores, key=lambda item: (-scores[item], best_rank[item], item))[:budget]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(row["split"], row["step_type"]) for row in rows})
    for split, step_type in keys:
        group = [
            row for row in rows if row["split"] == split and row["step_type"] == step_type
        ]
        item: dict[str, Any] = {
            "split": split,
            "step_type": step_type,
            "steps": len(group),
            "mean_lexical_seconds": statistics.fmean(
                float(row["lexical_seconds"]) for row in group
            ),
            "mean_anchor_seconds": statistics.fmean(
                float(row["anchor_seconds"]) for row in group
            ),
        }
        for method in (
            "anchor",
            "lexical",
            "qk_exact",
            "hybrid_anchor_qk",
            "hybrid_union",
            "hybrid_rrf",
        ):
            for budget in (1, 3, 16, 512):
                item[f"{method}_recall_at_{budget}"] = statistics.fmean(
                    0 < int(row[f"{method}_rank"]) <= budget for row in group
                )
        output.append(item)
    return output


def main() -> None:
    args = parse_args()
    if not 0 < args.lexical_share <= args.candidate_blocks:
        raise ValueError("lexical_share must be within the candidate budget")
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    allowed_tasks = {item.strip() for item in args.task_types.split(",") if item.strip()}
    excluded_ids = {
        int(item.strip()) for item in args.exclude_query_ids.split(",") if item.strip()
    }
    steps = [
        row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) in allowed_splits
        and str(row["task_type"]) in allowed_tasks
        and int(row["query_id"]) not in excluded_ids
    ]
    steps.sort(key=lambda row: (int(row["query_id"]), int(row["step_index"])))
    qk_rows = (
        {
            (int(row["query_id"]), int(row["step_index"])): row
            for row in read_jsonl(Path(args.qk_rows_path))
        }
        if args.qk_rows_path
        else {}
    )
    if args.qk_rows_path:
        missing = [
            (int(step["query_id"]), int(step["step_index"]))
            for step in steps
            if (int(step["query_id"]), int(step["step_index"])) not in qk_rows
        ]
        if missing:
            raise ValueError(f"missing Q/K rows for {len(missing)} steps")

    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    build_started = time.perf_counter()
    block_texts = decode_blocks(tokenizer, blocks)
    bm25 = BM25Index(block_texts, min_df=1, max_df=1.0, k1=1.2, b=0.75)
    build_seconds = time.perf_counter() - build_started
    anchor_build_started = time.perf_counter()
    anchor_index = AnchorInvertedIndex(block_texts)
    anchor_build_seconds = time.perf_counter() - anchor_build_started
    step_query_texts = [step_state_text(step) for step in steps]
    lexical_backend = args.lexical_backend
    if lexical_backend == "auto":
        lexical_backend = "postings" if len(step_query_texts) <= 64 else "matrix"
    lexical_batch_started = time.perf_counter()
    lexical_score_matrix = (
        bm25.score_postings(step_query_texts)
        if lexical_backend == "postings"
        else bm25.score(step_query_texts)
    )
    lexical_batch_seconds = time.perf_counter() - lexical_batch_started
    amortized_lexical_seconds = lexical_batch_seconds / max(1, len(steps))

    rows = []
    qk_share = args.candidate_blocks - args.lexical_share
    for step_offset, step in enumerate(steps):
        query_text = step_query_texts[step_offset]
        lexical_scores = lexical_score_matrix[step_offset]
        lexical = stable_rank(lexical_scores, args.candidate_blocks)
        lexical_seconds = amortized_lexical_seconds
        anchor_text = step_anchor_text(step)
        anchor_started = time.perf_counter()
        use_alias = args.anchor_alias_fallback and str(step["step_type"]) == "resolve_answer_from_bridge"
        anchor = (
            anchor_index.search_alias_aware(
                anchor_text, query_text, args.candidate_blocks
            )
            if use_alias
            else anchor_index.search(anchor_text, query_text, args.candidate_blocks)
        )
        anchor_seconds = time.perf_counter() - anchor_started
        key = (int(step["query_id"]), int(step["step_index"]))
        qk_row = qk_rows.get(key, {})
        qk_exact = [int(item) for item in qk_row.get("exact_candidates", [])]
        if not qk_exact:
            qk_exact = [int(item) for item in qk_row.get("exact_top16", [])]
        target = int(step["target_block_ids"][0])
        union = list(
            dict.fromkeys(lexical[: args.lexical_share] + qk_exact[:qk_share])
        )[: args.candidate_blocks]
        hybrid_anchor_qk = list(dict.fromkeys(anchor + qk_exact))[
            : args.candidate_blocks
        ]
        hybrid_rrf = reciprocal_rank_fusion(
            [lexical, qk_exact], args.candidate_blocks, args.rrf_k
        )
        lexical_top_scores = [float(lexical_scores[item]) for item in lexical[:16]]
        lexical_top1_gap = (
            lexical_top_scores[0] - lexical_top_scores[1]
            if len(lexical_top_scores) >= 2
            else lexical_top_scores[0] if lexical_top_scores else 0.0
        )
        row = {
            "query_id": key[0],
            "step_index": key[1],
            "split": str(step["split"]),
            "step_type": str(step["step_type"]),
            "step_operator": str(step["step_operator"]),
            "target_block_id": target,
            "selection_uses_gold": False,
            "query_text": query_text,
            "anchor_text": anchor_text,
            "anchor_seconds": anchor_seconds,
            "lexical_seconds": lexical_seconds,
            "query_term_count": len(normalized_tokens(query_text)),
            "anchor_candidate_count": len(anchor),
            "anchor_lexical_top3_overlap": len(set(anchor[:3]) & set(lexical[:3])),
            "anchor_lexical_top16_overlap": len(set(anchor[:16]) & set(lexical[:16])),
            "lexical_top_scores": lexical_top_scores,
            "lexical_top1_gap": lexical_top1_gap,
            "anchor_rank": rank_or_zero(anchor, target),
            "lexical_rank": rank_or_zero(lexical, target),
            "qk_exact_rank": int(qk_row.get("exact_rank", 0)),
            "hybrid_anchor_qk_rank": rank_or_zero(hybrid_anchor_qk, target),
            "hybrid_union_rank": rank_or_zero(union, target),
            "hybrid_rrf_rank": rank_or_zero(hybrid_rrf, target),
            "anchor_candidates": anchor,
            "lexical_candidates": lexical,
            "qk_exact_candidates": qk_exact,
            "hybrid_anchor_qk_candidates": hybrid_anchor_qk,
            "hybrid_union_candidates": union,
            "hybrid_rrf_candidates": hybrid_rrf,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "step": len(rows),
                    "steps": len(steps),
                    "query_id": key[0],
                    "step_index": key[1],
                    "anchor_rank": row["anchor_rank"],
                    "lexical_rank": row["lexical_rank"],
                    "qk_exact_rank": row["qk_exact_rank"],
                    "hybrid_union_rank": row["hybrid_union_rank"],
                }
            ),
            flush=True,
        )

    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "operator-state BM25 and distributed step-Q/K candidate fusion",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "num_blocks": int(blocks.shape[0]),
        "steps": len(rows),
        "candidate_blocks": args.candidate_blocks,
        "lexical_share": args.lexical_share,
        "qk_share": qk_share,
        "bm25_build_seconds": build_seconds,
        "bm25_batch_seconds": lexical_batch_seconds,
        "bm25_backend": lexical_backend,
        "bm25_features": bm25.features,
        "anchor_build_seconds": anchor_build_seconds,
        "anchor_terms": len(anchor_index.postings),
        "anchor_alias_fallback": args.anchor_alias_fallback,
        "qk_rows_path": args.qk_rows_path or None,
        "summaries": summarize(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
