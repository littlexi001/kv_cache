from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

from evaluate_xsum_news_ppl_retrieval import encode_e5
from run_iterative_condition_retrieval import BM25Index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect target-free incremental-refresh features by scoring only the "
            "previous RRF Top512 frontier under the current generation state."
        )
    )
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--retrieval_rows", required=True)
    parser.add_argument("--e5_base_embeddings", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--embedding_model_name_or_path", default="intfloat/e5-base-v2")
    parser.add_argument("--embedding_batch_size", type=int, default=128)
    parser.add_argument("--embedding_max_length", type=int, default=96)
    parser.add_argument("--decode_batch_size", type=int, default=2048)
    parser.add_argument("--frontier_blocks", type=int, default=512)
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def decode_blocks(tokenizer: Any, blocks: np.ndarray, batch_size: int) -> list[str]:
    texts: list[str] = []
    for start in range(0, len(blocks), batch_size):
        batch = np.asarray(blocks[start : start + batch_size], dtype=np.int64)
        texts.extend(tokenizer.batch_decode(batch.tolist(), skip_special_tokens=True))
    return texts


def weighted_external_matrix(
    index: BM25Index,
    documents: list[str],
    *,
    k1: float,
    b: float,
) -> sparse.csr_matrix:
    counts = index.vectorizer.transform(documents).tocsr().astype(np.float32)
    lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    row_ids = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
    frequencies = counts.data.copy()
    denominator = frequencies + k1 * (
        1.0 - b + b * lengths[row_ids] / max(index.average_length, 1.0e-6)
    )
    counts.data = (
        index.inverse_document_frequency[counts.indices]
        * frequencies
        * (k1 + 1.0)
        / denominator
    )
    return counts


def bm25_frontier_scores(
    index: BM25Index,
    source_matrix: sparse.csr_matrix,
    query_text: str,
    candidate_ids: list[int],
    base_count: int,
) -> np.ndarray:
    query = index.vectorizer.transform([query_text]).tocsr().astype(np.float32)
    query.data.fill(1.0)
    scores = np.zeros(len(candidate_ids), dtype=np.float32)
    base_positions = [
        index for index, block_id in enumerate(candidate_ids) if block_id < base_count
    ]
    source_positions = [
        index for index, block_id in enumerate(candidate_ids) if block_id >= base_count
    ]
    if base_positions:
        ids = [candidate_ids[index] for index in base_positions]
        result = query @ index.weighted_documents[ids].transpose()
        scores[base_positions] = np.asarray(result.toarray()).reshape(-1)
    if source_positions:
        ids = [candidate_ids[index] - base_count for index in source_positions]
        result = query @ source_matrix[ids].transpose()
        scores[source_positions] = np.asarray(result.toarray()).reshape(-1)
    return scores


def e5_frontier_scores(
    base_embeddings: torch.Tensor,
    source_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    candidate_ids: list[int],
    base_count: int,
) -> np.ndarray:
    scores = torch.empty(
        len(candidate_ids), device=query_embedding.device, dtype=torch.float32
    )
    base_positions = [
        index for index, block_id in enumerate(candidate_ids) if block_id < base_count
    ]
    source_positions = [
        index for index, block_id in enumerate(candidate_ids) if block_id >= base_count
    ]
    if base_positions:
        ids = torch.as_tensor(
            [candidate_ids[index] for index in base_positions],
            device=query_embedding.device,
            dtype=torch.long,
        )
        positions = torch.as_tensor(
            base_positions, device=query_embedding.device, dtype=torch.long
        )
        scores[positions] = base_embeddings[ids].float() @ query_embedding.float()
    if source_positions:
        ids = torch.as_tensor(
            [candidate_ids[index] - base_count for index in source_positions],
            device=query_embedding.device,
            dtype=torch.long,
        )
        positions = torch.as_tensor(
            source_positions, device=query_embedding.device, dtype=torch.long
        )
        scores[positions] = source_embeddings[ids].float() @ query_embedding.float()
    return scores.cpu().numpy()


def rank_ids(scores: np.ndarray, candidate_ids: list[int]) -> list[int]:
    ids = np.asarray(candidate_ids, dtype=np.int64)
    order = np.lexsort((ids, -np.asarray(scores, dtype=np.float64)))
    return ids[order].tolist()


def rrf_scores(
    rankings: list[list[int]], candidate_ids: list[int], rrf_k: float
) -> np.ndarray:
    positions = {block_id: index for index, block_id in enumerate(candidate_ids)}
    output = np.zeros(len(candidate_ids), dtype=np.float64)
    for ranking in rankings:
        for rank, block_id in enumerate(ranking, start=1):
            output[positions[block_id]] += 1.0 / (rrf_k + rank)
    return output


def set_jaccard(left: list[int], right: list[int], depth: int) -> float:
    a = set(left[:depth])
    b = set(right[:depth])
    return len(a & b) / len(a | b) if a or b else 1.0


def score_geometry(prefix: str, scores: np.ndarray) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    ordered = np.sort(values)[::-1]
    standard_deviation = max(float(values.std()), 1.0e-8)
    standardized = (values - float(values.mean())) / standard_deviation
    probabilities = np.exp(standardized - standardized.max())
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-30))))
    sorted_probabilities = np.sort(probabilities)[::-1]
    return {
        f"{prefix}_score_mean": float(values.mean()),
        f"{prefix}_score_std": float(values.std()),
        f"{prefix}_top1_score": float(ordered[0]),
        f"{prefix}_top1_z": float((ordered[0] - values.mean()) / standard_deviation),
        f"{prefix}_top1_top2_margin_z": float(
            (ordered[0] - ordered[1]) / standard_deviation
        ),
        f"{prefix}_top8_top9_margin_z": float(
            (ordered[7] - ordered[8]) / standard_deviation
        ),
        f"{prefix}_normalized_entropy": float(entropy / math.log(len(values))),
        f"{prefix}_top8_softmax_mass": float(sorted_probabilities[:8].sum()),
        f"{prefix}_positive_fraction": float((values > 0).mean()),
    }


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else 0.0


def retrieval_lookup(
    rows: list[dict[str, Any]], memory_tokens: int
) -> dict[tuple[int, int, str], dict[str, Any]]:
    selected = [row for row in rows if int(row["memory_tokens"]) == memory_tokens]
    if any(bool(row["selection_uses_target"]) for row in selected):
        raise ValueError("retrieval candidate rows use target information")
    return {
        (int(row["query_id"]), int(row["prefix_tokens"]), str(row["method"])): row
        for row in selected
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    memory_tokens = max(int(item) for item in data_summary["memory_scales_tokens"])
    block_tokens = int(data_summary["block_tokens"])
    source_count = int(data_summary["source_blocks"])
    memory_blocks = memory_tokens // block_tokens
    base_count = memory_blocks - source_count

    retrieval_rows = read_jsonl(args.retrieval_rows)
    retrieval = retrieval_lookup(retrieval_rows, memory_tokens)
    query_ids = sorted({key[0] for key in retrieval})
    prefixes = sorted({key[1] for key in retrieval})
    required_methods = {"bm25", "e5", "bm25_e5_rrf"}
    if {key[2] for key in retrieval} != required_methods:
        raise ValueError("expected BM25, E5, and RRF retrieval rows")

    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")[:base_count]
    source_blocks = np.load(data_dir / "source_blocks.npy", mmap_mode="r")
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    offline_started = time.perf_counter()
    base_texts = decode_blocks(qwen_tokenizer, base_blocks, args.decode_batch_size)
    source_texts = [
        decode_blocks(qwen_tokenizer, source_blocks[query_id], args.decode_batch_size)
        for query_id in query_ids
    ]
    query_texts = {
        (query_id, prefix): qwen_tokenizer.decode(
            np.asarray(queries[query_id, :prefix], dtype=np.int64).tolist(),
            skip_special_tokens=True,
        )
        for query_id in query_ids
        for prefix in prefixes
    }
    bm25_started = time.perf_counter()
    bm25 = BM25Index(base_texts, min_df=1, max_df=1.0, k1=1.2, b=0.75)
    bm25_build_seconds = time.perf_counter() - bm25_started
    source_matrices = [
        weighted_external_matrix(bm25, texts, k1=1.2, b=0.75)
        for texts in source_texts
    ]

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    e5_tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model_name_or_path, use_fast=True
    )
    e5_model = AutoModel.from_pretrained(
        args.embedding_model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    e5_model.eval()
    base_embedding_array = np.load(args.e5_base_embeddings, mmap_mode="r")[:base_count]
    base_embeddings = torch.as_tensor(
        np.asarray(base_embedding_array), device=device, dtype=dtype
    )
    flat_sources = [text for group in source_texts for text in group]
    source_embeddings = encode_e5(
        e5_model,
        e5_tokenizer,
        flat_sources,
        prefix="passage: ",
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        device=device,
    ).reshape(len(query_ids), source_count, -1).to(dtype)
    query_keys = sorted(query_texts)
    query_embedding_tensor = encode_e5(
        e5_model,
        e5_tokenizer,
        [query_texts[key] for key in query_keys],
        prefix="query: ",
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        device=device,
    ).to(dtype)
    query_embeddings = {
        key: query_embedding_tensor[index] for index, key in enumerate(query_keys)
    }
    torch.cuda.synchronize(device)
    offline_seconds = time.perf_counter() - offline_started

    rows: list[dict[str, Any]] = []
    online_seconds: list[float] = []
    for query_id in query_ids:
        source = set(range(base_count, memory_blocks))
        for prefix_index in range(len(prefixes) - 1):
            previous_prefix = prefixes[prefix_index]
            current_prefix = prefixes[prefix_index + 1]
            previous_global = retrieval[
                (query_id, previous_prefix, "bm25_e5_rrf")
            ]
            current_global = retrieval[(query_id, current_prefix, "bm25_e5_rrf")]
            candidate_ids = [
                int(item)
                for item in previous_global["top_block_ids"][: args.frontier_blocks]
            ]
            if len(candidate_ids) != args.frontier_blocks:
                raise ValueError("previous frontier is shorter than requested")

            torch.cuda.synchronize(device)
            started = time.perf_counter()
            candidate_scores: dict[int, dict[str, np.ndarray]] = {}
            candidate_rankings: dict[int, dict[str, list[int]]] = {}
            for prefix in (previous_prefix, current_prefix):
                bm25_scores = bm25_frontier_scores(
                    bm25,
                    source_matrices[query_id],
                    query_texts[(query_id, prefix)],
                    candidate_ids,
                    base_count,
                )
                e5_scores = e5_frontier_scores(
                    base_embeddings,
                    source_embeddings[query_id],
                    query_embeddings[(query_id, prefix)],
                    candidate_ids,
                    base_count,
                )
                bm25_ranking = rank_ids(bm25_scores, candidate_ids)
                e5_ranking = rank_ids(e5_scores, candidate_ids)
                fused_scores = rrf_scores(
                    [bm25_ranking, e5_ranking], candidate_ids, args.rrf_k
                )
                fused_ranking = rank_ids(fused_scores, candidate_ids)
                candidate_scores[prefix] = {
                    "bm25": bm25_scores,
                    "e5": e5_scores,
                    "rrf": fused_scores,
                }
                candidate_rankings[prefix] = {
                    "bm25": bm25_ranking,
                    "e5": e5_ranking,
                    "rrf": fused_ranking,
                }
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            online_seconds.append(elapsed)

            current_scores = candidate_scores[current_prefix]
            previous_scores = candidate_scores[previous_prefix]
            current_rankings = candidate_rankings[current_prefix]
            previous_rankings = candidate_rankings[previous_prefix]
            restricted_top8 = set(current_rankings["rrf"][:8])
            full_top8 = {
                int(item) for item in current_global["top_block_ids"][:8]
            }
            frontier = set(candidate_ids)
            previous_tokens = np.asarray(
                queries[query_id, :previous_prefix], dtype=np.int64
            )
            current_tokens = np.asarray(
                queries[query_id, :current_prefix], dtype=np.int64
            )
            previous_token_set = set(previous_tokens.tolist())
            current_token_set = set(current_tokens.tolist())
            token_union = previous_token_set | current_token_set
            query_embedding_cosine = float(
                torch.dot(
                    query_embeddings[(query_id, previous_prefix)].float(),
                    query_embeddings[(query_id, current_prefix)].float(),
                ).item()
            )
            features: dict[str, float] = {
                "previous_prefix_tokens": float(previous_prefix),
                "current_prefix_tokens": float(current_prefix),
                "prefix_ratio": float(current_prefix / previous_prefix),
                "new_tokens": float(current_prefix - previous_prefix),
                "new_unique_tokens": float(
                    len(current_token_set - previous_token_set)
                ),
                "query_token_set_jaccard": float(
                    len(previous_token_set & current_token_set) / len(token_union)
                    if token_union
                    else 1.0
                ),
                "query_e5_cosine": query_embedding_cosine,
                "current_bm25_e5_top8_jaccard": set_jaccard(
                    current_rankings["bm25"], current_rankings["e5"], 8
                ),
                "current_bm25_e5_top64_jaccard": set_jaccard(
                    current_rankings["bm25"], current_rankings["e5"], 64
                ),
                "temporal_rrf_top8_jaccard": set_jaccard(
                    previous_rankings["rrf"], current_rankings["rrf"], 8
                ),
                "temporal_rrf_top64_jaccard": set_jaccard(
                    previous_rankings["rrf"], current_rankings["rrf"], 64
                ),
            }
            for method in ("bm25", "e5", "rrf"):
                features.update(score_geometry(method, current_scores[method]))
                features[f"{method}_score_temporal_spearman"] = safe_spearman(
                    previous_scores[method], current_scores[method]
                )
                previous_std = max(float(previous_scores[method].std()), 1.0e-8)
                features[f"{method}_top1_score_delta_previous_std"] = float(
                    (
                        current_scores[method].max()
                        - previous_scores[method].max()
                    )
                    / previous_std
                )
            if not all(math.isfinite(value) for value in features.values()):
                raise ValueError("non-finite online refresh feature")

            full_source_any = bool(full_top8 & source)
            restricted_source_any = bool(restricted_top8 & source)
            row = {
                "dataset": args.dataset_name,
                "query_id": query_id,
                "prefix_transition": f"{previous_prefix}->{current_prefix}",
                "previous_prefix_tokens": previous_prefix,
                "current_prefix_tokens": current_prefix,
                "frontier_blocks": len(candidate_ids),
                "memory_blocks": memory_blocks,
                "online_candidate_scoring_seconds": elapsed,
                "features": features,
                "labels": {
                    "full_top8_frontier_coverage": len(full_top8 & frontier) / 8.0,
                    "full_top8_frontier_miss_fraction": 1.0
                    - len(full_top8 & frontier) / 8.0,
                    "frontier_miss_above_25pct": len(full_top8 & frontier) < 6,
                    "frontier_miss_above_50pct": len(full_top8 & frontier) < 4,
                    "full_source_any_at_8": full_source_any,
                    "restricted_source_any_at_8": restricted_source_any,
                    "global_refresh_source_any_gain": float(full_source_any)
                    - float(restricted_source_any),
                    "global_refresh_strictly_needed_for_source": full_source_any
                    and not restricted_source_any,
                    "restricted_source_recall_at_8": len(restricted_top8 & source)
                    / len(source),
                    "full_source_recall_at_8": len(full_top8 & source) / len(source),
                },
                "candidate_selection_uses_current_state": False,
                "online_features_use_current_global_ranking": False,
                "online_features_use_target": False,
                "full_global_ranking_used_only_for_labels": True,
                "restricted_top8": current_rankings["rrf"][:8],
                "full_global_top8": list(current_global["top_block_ids"][:8]),
            }
            rows.append(row)

    feature_names = sorted(rows[0]["features"])
    if any(sorted(row["features"]) != feature_names for row in rows):
        raise RuntimeError("feature schema changed across rows")
    summary = {
        "source": "target-free current-state scoring over previous retrieval frontier",
        "dataset": args.dataset_name,
        "protocol": {
            "queries": len(query_ids),
            "events": len(rows),
            "memory_tokens": memory_tokens,
            "memory_blocks": memory_blocks,
            "frontier_blocks": args.frontier_blocks,
            "prefix_tokens": prefixes,
            "candidate_selection": "previous-state global RRF Top512",
            "current_online_scoring": "BM25 and E5 over previous frontier only",
            "online_features_use_current_global_ranking": False,
            "online_features_use_target": False,
            "full_global_ranking_used_only_for_labels": True,
        },
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "target_statistics": {
            "mean_full_top8_frontier_coverage": mean(
                [row["labels"]["full_top8_frontier_coverage"] for row in rows]
            ),
            "frontier_miss_above_25pct_rate": mean(
                [float(row["labels"]["frontier_miss_above_25pct"]) for row in rows]
            ),
            "frontier_miss_above_50pct_rate": mean(
                [float(row["labels"]["frontier_miss_above_50pct"]) for row in rows]
            ),
            "global_refresh_strictly_needed_for_source_rate": mean(
                [
                    float(row["labels"]["global_refresh_strictly_needed_for_source"])
                    for row in rows
                ]
            ),
            "full_source_any_at_8": mean(
                [float(row["labels"]["full_source_any_at_8"]) for row in rows]
            ),
            "restricted_source_any_at_8": mean(
                [float(row["labels"]["restricted_source_any_at_8"]) for row in rows]
            ),
        },
        "timing": {
            "offline_total_seconds": offline_seconds,
            "bm25_build_seconds": bm25_build_seconds,
            "mean_online_candidate_scoring_seconds": mean(online_seconds),
            "median_online_candidate_scoring_seconds": float(
                np.median(online_seconds)
            ),
            "online_timing_includes_two_states_for_drift_features": True,
            "online_timing_excludes_current_global_search": True,
        },
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
