from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from transformers import AutoModel, AutoTokenizer

from analyze_longmemeval_10m_utility_gate import (
    paired_bootstrap,
    read_jsonl,
    state_features,
    summarize_variant,
)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate-content utility gates across independent 10M shards."
    )
    parser.add_argument("--data_pattern", required=True)
    parser.add_argument("--selection_pattern", required=True)
    parser.add_argument("--reader_pattern", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--embedding_model", default="intfloat/e5-base-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding_batch_size", type=int, default=256)
    parser.add_argument("--embedding_max_length", type=int, default=128)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def terms(text: str) -> set[str]:
    return {
        value
        for value in re.findall(r"[a-z0-9]+", text.lower())
        if value not in STOPWORDS and (len(value) > 1 or value.isdigit())
    }


def entity_terms(text: str) -> set[str]:
    proper = re.findall(r"\b[A-Z][A-Za-z0-9_-]+\b", text)
    numbers = re.findall(r"\b\d+(?:[.:/-]\d+)*\b", text)
    return {value.lower() for value in proper + numbers if value.lower() not in STOPWORDS}


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


@torch.inference_mode()
def encode_texts(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    prefix: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    output = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            [prefix + text for text in texts[start : start + batch_size]],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        output.append(F.normalize(pooled.float(), dim=1).cpu().numpy())
    return np.concatenate(output, axis=0)


def page_set_features(
    prefix: str,
    pages: list[set[str]],
    *,
    question: set[str],
    state: set[str],
    state_novel: set[str],
    entities: set[str],
    initial_union: set[str],
) -> dict[str, float]:
    output: dict[str, float] = {f"{prefix}_pages": float(len(pages))}
    page_union = set().union(*pages) if pages else set()
    for name, source in (
        ("question", question),
        ("state", state),
        ("state_novel", state_novel),
        ("entities", entities),
    ):
        coverages = [len(source & page) / max(1, len(source)) for page in pages]
        jaccards = [len(source & page) / max(1, len(source | page)) for page in pages]
        output[f"{prefix}_{name}_union_coverage"] = len(source & page_union) / max(
            1, len(source)
        )
        output[f"{prefix}_{name}_max_coverage"] = max(coverages, default=0.0)
        output[f"{prefix}_{name}_mean_coverage"] = safe_mean(coverages)
        output[f"{prefix}_{name}_max_jaccard"] = max(jaccards, default=0.0)
    novelty = [len(page - initial_union) / max(1, len(page)) for page in pages]
    redundancy = [len(page & initial_union) / max(1, len(page | initial_union)) for page in pages]
    output[f"{prefix}_mean_novelty_to_initial"] = safe_mean(novelty)
    output[f"{prefix}_max_novelty_to_initial"] = max(novelty, default=0.0)
    output[f"{prefix}_mean_jaccard_initial"] = safe_mean(redundancy)
    output[f"{prefix}_max_jaccard_initial"] = max(redundancy, default=0.0)
    output[f"{prefix}_union_terms"] = float(len(page_union))
    output[f"{prefix}_mean_page_terms"] = safe_mean(float(len(page)) for page in pages)
    return output


def dense_page_features(
    prefix: str,
    vectors: list[np.ndarray],
    *,
    question: np.ndarray,
    state: np.ndarray,
    initial: np.ndarray,
) -> dict[str, float]:
    output: dict[str, float] = {}
    if not vectors:
        for suffix in (
            "question_max",
            "question_mean",
            "state_max",
            "state_mean",
            "state_minus_question_max",
            "initial_max_mean",
            "initial_centroid_mean",
            "initial_novelty_min",
        ):
            output[f"{prefix}_e5_{suffix}"] = 0.0
        return output
    matrix = np.stack(vectors)
    question_scores = matrix @ question
    state_scores = matrix @ state
    initial_scores = matrix @ initial.T
    initial_centroid = initial.mean(axis=0)
    initial_centroid /= max(float(np.linalg.norm(initial_centroid)), 1e-8)
    centroid_scores = matrix @ initial_centroid
    output.update(
        {
            f"{prefix}_e5_question_max": float(question_scores.max()),
            f"{prefix}_e5_question_mean": float(question_scores.mean()),
            f"{prefix}_e5_state_max": float(state_scores.max()),
            f"{prefix}_e5_state_mean": float(state_scores.mean()),
            f"{prefix}_e5_state_minus_question_max": float(
                (state_scores - question_scores).max()
            ),
            f"{prefix}_e5_initial_max_mean": float(initial_scores.max(axis=1).mean()),
            f"{prefix}_e5_initial_centroid_mean": float(centroid_scores.mean()),
            f"{prefix}_e5_initial_novelty_min": float((1.0 - centroid_scores).min()),
        }
    )
    return output


def candidate_features(
    record: dict[str, Any], *, include_dense: bool
) -> dict[str, float | str]:
    state = record["state"]
    static = record["selection_static"]
    dynamic = record["selection_dynamic"]
    output = state_features(
        state,
        static,
        dynamic,
        include_text_flags=True,
        include_question_type=True,
    )
    question_text = str(record["question"])
    state_text = str(state["state_text"])
    question_terms = terms(question_text)
    state_terms = terms(state_text)
    state_novel = state_terms - question_terms
    entities = entity_terms(question_text) | entity_terms(state_text)

    block_terms: dict[int, set[str]] = record["block_terms"]
    initial_ids = set(int(value) for value in state["initial_block_ids"])
    static_ids = set(int(value) for value in static["top_block_ids"])
    dynamic_ids = set(int(value) for value in dynamic["top_block_ids"])
    initial_union = set().union(*(block_terms[value] for value in initial_ids))
    dynamic_only = [block_terms[value] for value in sorted(dynamic_ids - static_ids)]
    static_only = [block_terms[value] for value in sorted(static_ids - dynamic_ids)]
    dynamic_extra = [block_terms[value] for value in sorted(dynamic_ids - initial_ids)]
    static_extra = [block_terms[value] for value in sorted(static_ids - initial_ids)]

    groups = {
        "dynamic_only": dynamic_only,
        "static_only": static_only,
        "dynamic_extra": dynamic_extra,
        "static_extra": static_extra,
    }
    for prefix, pages in groups.items():
        output.update(
            page_set_features(
                prefix,
                pages,
                question=question_terms,
                state=state_terms,
                state_novel=state_novel,
                entities=entities,
                initial_union=initial_union,
            )
        )
    for suffix in (
        "question_union_coverage",
        "question_max_coverage",
        "state_union_coverage",
        "state_max_coverage",
        "state_novel_union_coverage",
        "state_novel_max_coverage",
        "entities_union_coverage",
        "entities_max_coverage",
        "mean_novelty_to_initial",
        "max_novelty_to_initial",
        "mean_jaccard_initial",
        "max_jaccard_initial",
        "union_terms",
        "mean_page_terms",
    ):
        output[f"only_dynamic_minus_static_{suffix}"] = float(
            output[f"dynamic_only_{suffix}"]
        ) - float(output[f"static_only_{suffix}"])
        output[f"extra_dynamic_minus_static_{suffix}"] = float(
            output[f"dynamic_extra_{suffix}"]
        ) - float(output[f"static_extra_{suffix}"])
    output["question_state_overlap"] = len(question_terms & state_terms) / max(
        1, len(question_terms)
    )
    output["state_novel_terms"] = float(len(state_novel))
    output["state_entity_terms"] = float(len(entities))
    if include_dense:
        block_embeddings: dict[int, np.ndarray] = record["block_embeddings"]
        initial_vectors = np.stack([block_embeddings[value] for value in sorted(initial_ids)])
        dense_groups = {
            "dynamic_only": [block_embeddings[value] for value in sorted(dynamic_ids - static_ids)],
            "static_only": [block_embeddings[value] for value in sorted(static_ids - dynamic_ids)],
            "dynamic_extra": [block_embeddings[value] for value in sorted(dynamic_ids - initial_ids)],
            "static_extra": [block_embeddings[value] for value in sorted(static_ids - initial_ids)],
        }
        for prefix, vectors in dense_groups.items():
            output.update(
                dense_page_features(
                    prefix,
                    vectors,
                    question=record["question_embedding"],
                    state=record["state_embedding"],
                    initial=initial_vectors,
                )
            )
        for suffix in (
            "question_max",
            "question_mean",
            "state_max",
            "state_mean",
            "state_minus_question_max",
            "initial_max_mean",
            "initial_centroid_mean",
            "initial_novelty_min",
        ):
            output[f"only_dynamic_minus_static_e5_{suffix}"] = float(
                output[f"dynamic_only_e5_{suffix}"]
            ) - float(output[f"static_only_e5_{suffix}"])
            output[f"extra_dynamic_minus_static_e5_{suffix}"] = float(
                output[f"dynamic_extra_e5_{suffix}"]
            ) - float(output[f"static_extra_e5_{suffix}"])
    return output


def fit_predict(
    records: list[dict[str, Any]],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    include_state_text: bool,
    include_dense: bool,
) -> tuple[np.ndarray, float]:
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(
        [candidate_features(records[index], include_dense=include_dense) for index in train_indices]
    )
    x_test = vectorizer.transform(
        [candidate_features(records[index], include_dense=include_dense) for index in test_indices]
    )
    if include_state_text:
        text_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=3,
            max_features=3_000,
            sublinear_tf=True,
        )
        text_train = text_vectorizer.fit_transform(
            [str(records[index]["state"]["state_text"]) for index in train_indices]
        )
        text_test = text_vectorizer.transform(
            [str(records[index]["state"]["state_text"]) for index in test_indices]
        )
        x_train = sparse.hstack([x_train, text_train], format="csr")
        x_test = sparse.hstack([x_test, text_test], format="csr")
    target = np.asarray([records[index]["nll_delta"] for index in train_indices])
    lower, upper = np.quantile(target, [0.025, 0.975])
    target = np.clip(target, lower, upper)
    groups = np.asarray([records[index]["partition"] for index in train_indices])
    splits = list(
        GroupKFold(n_splits=min(7, len(set(groups)))).split(x_train, target, groups)
    )
    model = RidgeCV(
        alphas=np.asarray([0.1, 1.0, 10.0, 100.0, 1_000.0]),
        cv=splits,
        scoring="neg_mean_squared_error",
    )
    model.fit(x_train, target)
    return np.asarray(model.predict(x_test)), float(model.alpha_)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    device = torch.device(args.device)
    embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, use_fast=True)
    embedding_model = AutoModel.from_pretrained(args.embedding_model).to(device).eval()
    records: list[dict[str, Any]] = []
    for partition in range(args.partitions):
        data_dir = Path(args.data_pattern.format(partition=partition))
        selection_dir = Path(args.selection_pattern.format(partition=partition))
        reader_dir = Path(args.reader_pattern.format(partition=partition))
        queries = {str(row["question_id"]): row for row in read_jsonl(data_dir / "queries.jsonl")}
        states = {
            str(row["question_id"]): row for row in read_jsonl(selection_dir / "states.jsonl")
        }
        selection: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in read_jsonl(selection_dir / "rows.jsonl"):
            if row["method"] in {"static_top12", "evidence_state_dynamic_top12"}:
                selection[str(row["method"])][str(row["question_id"])] = row
        reader: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in read_jsonl(reader_dir / "rows.jsonl"):
            reader[str(row["method"])][str(row["question_id"])] = row

        block_ids = set()
        for question_id, state in states.items():
            block_ids.update(int(value) for value in state["initial_block_ids"])
            for method in ("static_top12", "evidence_state_dynamic_top12"):
                block_ids.update(
                    int(value) for value in selection[method][question_id]["top_block_ids"]
                )
        ordered_ids = sorted(block_ids)
        base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
        decoded = tokenizer.batch_decode(
            np.asarray(base_blocks[ordered_ids]).tolist(), skip_special_tokens=True
        )
        block_terms = {block_id: terms(text) for block_id, text in zip(ordered_ids, decoded)}
        block_matrix = encode_texts(
            embedding_model,
            embedding_tokenizer,
            decoded,
            prefix="passage: ",
            batch_size=args.embedding_batch_size,
            max_length=args.embedding_max_length,
            device=device,
        )
        block_embeddings = {
            block_id: block_matrix[index] for index, block_id in enumerate(ordered_ids)
        }
        ordered_question_ids = sorted(states)
        question_matrix = encode_texts(
            embedding_model,
            embedding_tokenizer,
            [str(queries[question_id]["question"]) for question_id in ordered_question_ids],
            prefix="query: ",
            batch_size=args.embedding_batch_size,
            max_length=args.embedding_max_length,
            device=device,
        )
        state_matrix = encode_texts(
            embedding_model,
            embedding_tokenizer,
            [str(states[question_id]["state_text"]) for question_id in ordered_question_ids],
            prefix="query: ",
            batch_size=args.embedding_batch_size,
            max_length=args.embedding_max_length,
            device=device,
        )
        question_embeddings = {
            question_id: question_matrix[index]
            for index, question_id in enumerate(ordered_question_ids)
        }
        state_embeddings = {
            question_id: state_matrix[index]
            for index, question_id in enumerate(ordered_question_ids)
        }
        for question_id, state in states.items():
            static_reader = reader["static_top12"][question_id]
            dynamic_reader = reader["evidence_state_dynamic_top12"][question_id]
            records.append(
                {
                    "question_id": question_id,
                    "partition": partition,
                    "question": queries[question_id]["question"],
                    "question_type": state["question_type"],
                    "is_abstention": bool(state["is_abstention"]),
                    "state": state,
                    "selection_static": selection["static_top12"][question_id],
                    "selection_dynamic": selection["evidence_state_dynamic_top12"][question_id],
                    "reader_static": static_reader,
                    "reader_dynamic": dynamic_reader,
                    "block_terms": block_terms,
                    "block_embeddings": block_embeddings,
                    "question_embedding": question_embeddings[question_id],
                    "state_embedding": state_embeddings[question_id],
                    "nll_delta": float(dynamic_reader["reference_nll"])
                    - float(static_reader["reference_nll"]),
                }
            )
    records.sort(key=lambda row: (row["partition"], row["question_id"]))
    if len(records) != 500 or len({row["question_id"] for row in records}) != 500:
        raise RuntimeError("expected 500 unique questions")

    groups = np.asarray([row["partition"] for row in records])
    variants = [
        "candidate_lexical",
        "candidate_lexical_plus_state_text",
        "candidate_lexical_e5",
        "candidate_lexical_e5_plus_state_text",
    ]
    predictions = {variant: np.zeros(len(records)) for variant in variants}
    alphas = {variant: [] for variant in variants}
    for held_out in range(args.partitions):
        train_indices = np.flatnonzero(groups != held_out)
        test_indices = np.flatnonzero(groups == held_out)
        for variant in variants:
            prediction, alpha = fit_predict(
                records,
                train_indices,
                test_indices,
                include_state_text=variant.endswith("plus_state_text"),
                include_dense="_e5" in variant,
            )
            predictions[variant][test_indices] = prediction
            alphas[variant].append(alpha)

    static_nll = np.asarray([row["reader_static"]["reference_nll"] for row in records])
    dynamic_nll = np.asarray([row["reader_dynamic"]["reference_nll"] for row in records])
    output = {
        "protocol": {
            "memory_scope": "eight independent real 10M-token shards",
            "outer_validation": "leave-one-10M-shard-out",
            "test_features": "trajectory metadata plus lexical/E5 state-candidate and displacement-page interactions",
            "selection_uses_answer_at_test": False,
        },
        "queries": len(records),
        "always_static_mean_reference_nll": float(static_nll.mean()),
        "always_dynamic_mean_reference_nll": float(dynamic_nll.mean()),
        "variants": {
            variant: {
                "fold_selected_alphas": alphas[variant],
                **summarize_variant(
                    records,
                    predictions[variant],
                    samples=args.bootstrap_samples,
                    seed=args.seed + 10 * index,
                ),
            }
            for index, variant in enumerate(variants)
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    rows_path = output_path.with_suffix(".rows.jsonl")
    with rows_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            row = {
                "question_id": record["question_id"],
                "partition": record["partition"],
                "nll_delta": record["nll_delta"],
            }
            for variant in variants:
                row[f"{variant}_predicted_nll_delta"] = float(predictions[variant][index])
                row[f"{variant}_use_dynamic"] = bool(predictions[variant][index] < 0)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
