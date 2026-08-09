from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import AutoModel, AutoTokenizer


DEPTHS = (3, 8, 16, 32)
TRANSITIONS = tuple(zip(DEPTHS[:-1], DEPTHS[1:]))
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect no-target, no-reader features describing how an expanded Top-K "
            "candidate set complements the current workset."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--retrieval_rows", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--decode_tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--embedding_model", default="intfloat/e5-base-v2")
    parser.add_argument("--memory_tokens", type=int, default=100_000_000)
    parser.add_argument("--state_suffix_tokens", default="128,512")
    parser.add_argument("--scope_depths", default="3,8,16,32")
    parser.add_argument("--retrieval_blocks", type=int, default=8)
    parser.add_argument("--embedding_batch_size", type=int, default=256)
    parser.add_argument("--embedding_max_length", type=int, default=128)
    parser.add_argument(
        "--candidate_query_end_offset_tokens",
        type=int,
        default=-1,
        help=(
            "Tokens to omit from the state when computing candidate features. "
            "-1 reuses the retrieval offset; 0 includes the full observed state."
        ),
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_queries", type=int, default=0)
    return parser.parse_args()


def parse_ints(spec: str) -> list[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) <= 0:
        raise ValueError("integer list must contain positive values")
    return values


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def scalar_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_std": 0.0,
        }
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_max": float(array.max()),
        f"{prefix}_min": float(array.min()),
        f"{prefix}_std": float(array.std()),
    }


def lexical_tokens(text: str, *, content_only: bool) -> set[str]:
    tokens = {item.lower() for item in TOKEN_PATTERN.findall(text)}
    if not content_only:
        return tokens
    return {
        token
        for token in tokens
        if len(token) > 1 and token not in ENGLISH_STOP_WORDS and not token.isdigit()
    }


def safe_fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def set_jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return safe_fraction(len(left & right), len(union)) if union else 1.0


def pairwise_off_diagonal(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) < 2:
        return np.empty(0, dtype=np.float32)
    similarities = embeddings @ embeddings.T
    return similarities[np.triu_indices(len(embeddings), k=1)]


def projection_explained(query: np.ndarray, embeddings: np.ndarray) -> float:
    if not len(embeddings):
        return 0.0
    _, singular_values, right = np.linalg.svd(embeddings, full_matrices=False)
    threshold = max(float(singular_values.max()) * 1.0e-5, 1.0e-8)
    basis = right[singular_values > threshold]
    return float(np.square(basis @ query).sum()) if len(basis) else 0.0


def residual_statistics(
    added: np.ndarray, current: np.ndarray, query: np.ndarray
) -> dict[str, float]:
    if not len(added):
        output = {}
        output.update(scalar_stats(np.empty(0), "dense_added_max_current_similarity"))
        output.update(scalar_stats(np.empty(0), "dense_added_residual_norm"))
        output.update(scalar_stats(np.empty(0), "dense_added_query_aligned_residual"))
        return output
    if not len(current):
        maximum_similarity = np.zeros(len(added), dtype=np.float32)
        residual = added
    else:
        maximum_similarity = (added @ current.T).max(axis=1)
        _, singular_values, right = np.linalg.svd(current, full_matrices=False)
        threshold = max(float(singular_values.max()) * 1.0e-5, 1.0e-8)
        basis = right[singular_values > threshold]
        residual = added - (added @ basis.T) @ basis if len(basis) else added
    residual_norm = np.linalg.norm(residual, axis=1)
    aligned = (residual @ query) / np.maximum(residual_norm, 1.0e-8)
    output = {}
    output.update(
        scalar_stats(maximum_similarity, "dense_added_max_current_similarity")
    )
    output.update(scalar_stats(residual_norm, "dense_added_residual_norm"))
    output.update(scalar_stats(aligned, "dense_added_query_aligned_residual"))
    return output


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


def decode_selected_blocks(
    tokenizer: Any, base_blocks: np.ndarray, block_ids: Sequence[int], batch_size: int = 512
) -> dict[int, str]:
    output: dict[int, str] = {}
    for start in range(0, len(block_ids), batch_size):
        ids = block_ids[start : start + batch_size]
        token_rows = np.asarray(base_blocks[np.asarray(ids, dtype=np.int64)], dtype=np.int64)
        texts = tokenizer.batch_decode(token_rows.tolist(), skip_special_tokens=True)
        output.update({int(block_id): text for block_id, text in zip(ids, texts)})
    return output


def build_features(
    *,
    query_text: str,
    query_embedding: np.ndarray,
    previous_ids: list[int],
    expanded_ids: list[int],
    block_texts: dict[int, str],
    block_embeddings: dict[int, np.ndarray],
    block_scope_ids: np.ndarray,
    previous_scope_ids: list[int],
    expanded_scope_ids: list[int],
) -> dict[str, float]:
    previous_set = set(previous_ids)
    expanded_set = set(expanded_ids)
    retained_ids = sorted(previous_set & expanded_set)
    added_ids = sorted(expanded_set - previous_set)
    dropped_ids = sorted(previous_set - expanded_set)

    def matrix(ids: Sequence[int]) -> np.ndarray:
        if not ids:
            return np.empty((0, len(query_embedding)), dtype=np.float32)
        return np.stack([block_embeddings[int(item)] for item in ids]).astype(np.float32)

    previous = matrix(previous_ids)
    expanded = matrix(expanded_ids)
    added = matrix(added_ids)
    dropped = matrix(dropped_ids)
    features: dict[str, float] = {
        "set_retained_count": float(len(retained_ids)),
        "set_added_count": float(len(added_ids)),
        "set_dropped_count": float(len(dropped_ids)),
        "set_topk_jaccard": set_jaccard(previous_set, expanded_set),
        "set_replacement_fraction": safe_fraction(len(added_ids), len(expanded_set)),
    }

    query_scores: dict[str, np.ndarray] = {}
    for name, values in (
        ("current", previous),
        ("expanded", expanded),
        ("added", added),
        ("dropped", dropped),
    ):
        scores = values @ query_embedding if len(values) else np.empty(0)
        query_scores[name] = scores
        features.update(scalar_stats(scores, f"dense_query_affinity_{name}"))
        pairwise = pairwise_off_diagonal(values)
        features.update(scalar_stats(pairwise, f"dense_set_similarity_{name}"))

    previous_centroid = previous.mean(axis=0) if len(previous) else np.zeros_like(query_embedding)
    expanded_centroid = expanded.mean(axis=0) if len(expanded) else np.zeros_like(query_embedding)
    previous_centroid /= max(float(np.linalg.norm(previous_centroid)), 1.0e-8)
    expanded_centroid /= max(float(np.linalg.norm(expanded_centroid)), 1.0e-8)
    features.update(
        {
            "dense_query_affinity_mean_gain": mean(query_scores["expanded"])
            - mean(query_scores["current"]),
            "dense_query_affinity_max_gain": (
                float(query_scores["expanded"].max()) if len(query_scores["expanded"]) else 0.0
            )
            - (float(query_scores["current"].max()) if len(query_scores["current"]) else 0.0),
            "dense_centroid_query_current": float(previous_centroid @ query_embedding),
            "dense_centroid_query_expanded": float(expanded_centroid @ query_embedding),
            "dense_centroid_query_gain": float(
                (expanded_centroid - previous_centroid) @ query_embedding
            ),
            "dense_centroid_shift": float(
                1.0 - np.clip(previous_centroid @ expanded_centroid, -1.0, 1.0)
            ),
            "dense_projection_query_current": projection_explained(query_embedding, previous),
            "dense_projection_query_expanded": projection_explained(query_embedding, expanded),
        }
    )
    features["dense_projection_query_gain"] = (
        features["dense_projection_query_expanded"]
        - features["dense_projection_query_current"]
    )
    features.update(residual_statistics(added, previous, query_embedding))
    if len(added) and len(dropped):
        features.update(
            scalar_stats(added @ dropped.T, "dense_added_dropped_similarity")
        )
    else:
        features.update(scalar_stats(np.empty(0), "dense_added_dropped_similarity"))

    for content_only, label in ((False, "all"), (True, "content")):
        query_terms = lexical_tokens(query_text, content_only=content_only)
        current_terms_by_block = [
            lexical_tokens(block_texts[item], content_only=content_only)
            for item in previous_ids
        ]
        expanded_terms_by_block = [
            lexical_tokens(block_texts[item], content_only=content_only)
            for item in expanded_ids
        ]
        added_terms_by_block = [
            lexical_tokens(block_texts[item], content_only=content_only)
            for item in added_ids
        ]
        current_terms = set().union(*current_terms_by_block) if current_terms_by_block else set()
        expanded_terms = set().union(*expanded_terms_by_block) if expanded_terms_by_block else set()
        added_terms = set().union(*added_terms_by_block) if added_terms_by_block else set()
        uncovered = query_terms - current_terms
        current_coverage = safe_fraction(len(query_terms & current_terms), len(query_terms))
        expanded_coverage = safe_fraction(len(query_terms & expanded_terms), len(query_terms))
        added_query_overlaps = np.asarray(
            [safe_fraction(len(query_terms & terms), len(query_terms)) for terms in added_terms_by_block]
        )
        added_current_jaccards = np.asarray(
            [set_jaccard(terms, current_terms) for terms in added_terms_by_block]
        )
        prefix = f"lexical_{label}"
        features.update(
            {
                f"{prefix}_query_terms": float(len(query_terms)),
                f"{prefix}_query_coverage_current": current_coverage,
                f"{prefix}_query_coverage_expanded": expanded_coverage,
                f"{prefix}_query_coverage_gain": expanded_coverage - current_coverage,
                f"{prefix}_uncovered_query_recovery": safe_fraction(
                    len(uncovered & added_terms), len(uncovered)
                ),
                f"{prefix}_added_term_novelty": safe_fraction(
                    len(added_terms - current_terms), len(added_terms)
                ),
                f"{prefix}_current_expanded_jaccard": set_jaccard(
                    current_terms, expanded_terms
                ),
            }
        )
        features.update(scalar_stats(added_query_overlaps, f"{prefix}_added_query_overlap"))
        features.update(scalar_stats(added_current_jaccards, f"{prefix}_added_current_jaccard"))

    previous_scopes = {int(block_scope_ids[item]) for item in previous_ids}
    expanded_scopes = {int(block_scope_ids[item]) for item in expanded_ids}
    added_route_scopes = set(expanded_scope_ids) - set(previous_scope_ids)
    features.update(
        {
            "scope_current_selected_count": float(len(previous_scopes)),
            "scope_expanded_selected_count": float(len(expanded_scopes)),
            "scope_selected_count_gain": float(len(expanded_scopes) - len(previous_scopes)),
            "scope_added_route_count": float(len(added_route_scopes)),
            "scope_added_blocks_from_new_route_fraction": safe_fraction(
                sum(int(block_scope_ids[item]) in added_route_scopes for item in added_ids),
                len(added_ids),
            ),
            "scope_expanded_blocks_from_new_route_fraction": safe_fraction(
                sum(int(block_scope_ids[item]) in added_route_scopes for item in expanded_ids),
                len(expanded_ids),
            ),
        }
    )
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("non-finite complementarity feature")
    return features


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffixes = parse_ints(args.state_suffix_tokens)
    depths = parse_ints(args.scope_depths)
    if tuple(depths) != DEPTHS:
        raise ValueError(f"scope_depths must be {DEPTHS}")

    data_dir = Path(args.data_dir)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    if not data_summary.get("past_only") or data_summary.get("source_blocks") != 0:
        raise ValueError("requires past-only data without predefined source blocks")
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    block_scope_ids = np.asarray(
        np.load(data_dir / "base_block_scope_ids.npy", mmap_mode="r"), dtype=np.int64
    )
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    query_count = len(queries) if args.max_queries <= 0 else min(len(queries), args.max_queries)

    retrieval_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in read_jsonl(args.retrieval_rows):
        if int(row["memory_tokens"]) != args.memory_tokens:
            continue
        method = str(row["method"])
        if not method.startswith("hier_bm25_scope"):
            continue
        depth_text = method.removeprefix("hier_bm25_scope")
        if not depth_text.isdigit():
            continue
        key = (int(row["query_id"]), int(row["prefix_tokens"]), int(depth_text))
        if key[0] < query_count and key[1] in suffixes and key[2] in depths:
            retrieval_lookup[key] = row
    expected = query_count * len(suffixes) * len(depths)
    if len(retrieval_lookup) != expected:
        raise RuntimeError(f"expected {expected} retrieval rows, found {len(retrieval_lookup)}")
    if any(
        bool(row["selection_uses_target"])
        or int(row.get("query_end_offset_tokens", 0)) <= 0
        for row in retrieval_lookup.values()
    ):
        raise RuntimeError("retrieval protocol is not strict pre-probe past-only")

    unique_block_ids = sorted(
        {
            int(block_id)
            for row in retrieval_lookup.values()
            for block_id in row["top_block_ids"][: args.retrieval_blocks]
        }
    )
    decode_tokenizer = AutoTokenizer.from_pretrained(args.decode_tokenizer, use_fast=True)
    started = time.perf_counter()
    block_texts = decode_selected_blocks(decode_tokenizer, base_blocks, unique_block_ids)
    query_texts: dict[tuple[int, int], str] = {}
    for query_id in range(query_count):
        for suffix in suffixes:
            retrieval_offset = int(
                retrieval_lookup[(query_id, suffix, depths[0])][
                    "query_end_offset_tokens"
                ]
            )
            offset = (
                retrieval_offset
                if args.candidate_query_end_offset_tokens < 0
                else args.candidate_query_end_offset_tokens
            )
            if offset < 0 or offset >= suffix:
                raise ValueError("candidate query end offset must lie in [0, suffix)")
            end = -offset if offset else None
            state = np.asarray(queries[query_id, -suffix:end], dtype=np.int64)
            query_texts[(query_id, suffix)] = decode_tokenizer.decode(
                state.tolist(), skip_special_tokens=True
            )
    decode_seconds = time.perf_counter() - started

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, use_fast=True)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoModel.from_pretrained(args.embedding_model, torch_dtype=dtype).eval().to(device)
    started = time.perf_counter()
    encoded_blocks = encode_texts(
        model,
        embedding_tokenizer,
        [block_texts[item] for item in unique_block_ids],
        prefix="passage: ",
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        device=device,
    )
    embedding_tokenizer.truncation_side = "left"
    query_keys = sorted(query_texts)
    encoded_queries = encode_texts(
        model,
        embedding_tokenizer,
        [query_texts[item] for item in query_keys],
        prefix="query: ",
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        device=device,
    )
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    embedding_seconds = time.perf_counter() - started
    block_embeddings = {
        block_id: encoded_blocks[index] for index, block_id in enumerate(unique_block_ids)
    }
    query_embeddings = {key: encoded_queries[index] for index, key in enumerate(query_keys)}

    valid_scopes = sorted({int(item) for item in block_scope_ids if int(item) >= 0})
    scope_row_to_id = {row: scope for row, scope in enumerate(valid_scopes)}
    rows = []
    started = time.perf_counter()
    for query_id in range(query_count):
        for suffix in suffixes:
            for previous_depth, expanded_depth in TRANSITIONS:
                previous = retrieval_lookup[(query_id, suffix, previous_depth)]
                expanded = retrieval_lookup[(query_id, suffix, expanded_depth)]
                previous_ids = [int(item) for item in previous["top_block_ids"][: args.retrieval_blocks]]
                expanded_ids = [int(item) for item in expanded["top_block_ids"][: args.retrieval_blocks]]
                previous_scope_ids = [
                    scope_row_to_id[int(item)] for item in previous["selected_scope_rows"]
                ]
                expanded_scope_ids = [
                    scope_row_to_id[int(item)] for item in expanded["selected_scope_rows"]
                ]
                rows.append(
                    {
                        "query_id": query_id,
                        "state_suffix_tokens": suffix,
                        "previous_depth": previous_depth,
                        "expanded_depth": expanded_depth,
                        "previous_block_ids": previous_ids,
                        "expanded_block_ids": expanded_ids,
                        "features": build_features(
                            query_text=query_texts[(query_id, suffix)],
                            query_embedding=query_embeddings[(query_id, suffix)],
                            previous_ids=previous_ids,
                            expanded_ids=expanded_ids,
                            block_texts=block_texts,
                            block_embeddings=block_embeddings,
                            block_scope_ids=block_scope_ids,
                            previous_scope_ids=previous_scope_ids,
                            expanded_scope_ids=expanded_scope_ids,
                        ),
                        "retrieval_query_end_offset_tokens": int(
                            previous["query_end_offset_tokens"]
                        ),
                        "candidate_query_end_offset_tokens": (
                            int(previous["query_end_offset_tokens"])
                            if args.candidate_query_end_offset_tokens < 0
                            else args.candidate_query_end_offset_tokens
                        ),
                        "candidate_texts_observed": True,
                        "reader_forward_used": False,
                        "expanded_workset_reader_forward_used": False,
                        "future_target_used": False,
                        "selection_uses_target": False,
                    }
                )
    feature_seconds = time.perf_counter() - started
    summary = {
        "source": "candidate-conditioned complementarity features for scope expansion",
        "protocol": {
            "queries": query_count,
            "memory_tokens": args.memory_tokens,
            "states": suffixes,
            "transitions": [f"{left}->{right}" for left, right in TRANSITIONS],
            "retrieval_blocks": args.retrieval_blocks,
            "retrieval_excludes_observed_64_tokens": True,
            "candidate_query_end_offset_tokens": (
                64
                if args.candidate_query_end_offset_tokens < 0
                else args.candidate_query_end_offset_tokens
            ),
            "candidate_features_include_observed_64_tokens": (
                args.candidate_query_end_offset_tokens == 0
            ),
            "candidate_query_embedding_truncation_side": "left",
            "candidate_texts_observed": True,
            "reader_forward_used": False,
            "expanded_workset_reader_forward_used": False,
            "future_target_used": False,
            "selection_uses_target": False,
        },
        "embedding_model": args.embedding_model,
        "rows": len(rows),
        "unique_candidate_blocks": len(unique_block_ids),
        "feature_count": len(rows[0]["features"]),
        "decode_seconds": decode_seconds,
        "embedding_seconds": embedding_seconds,
        "feature_seconds": feature_seconds,
    }
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
