from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from profile_step_state_q import step_state_text
from run_iterative_condition_retrieval import BM25Index
from run_lexical_block_retrieval import decode_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="External-embedding dense and BM25+dense RAG retrieval over 10M blocks."
    )
    parser.add_argument("--embedding_model_name_or_path", required=True)
    parser.add_argument("--qwen_tokenizer_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument(
        "--step_types", default="resolve_bridge,resolve_answer_from_bridge"
    )
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--candidate_blocks", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--query_prefix", default="query: ")
    parser.add_argument("--passage_prefix", default="passage: ")
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--rebuild_index", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_topk(scores: np.ndarray, budget: int) -> list[int]:
    block_ids = np.arange(scores.shape[0], dtype=np.int64)
    if budget >= len(scores):
        return np.lexsort((block_ids, -scores)).tolist()
    candidates = np.argpartition(scores, -budget)[-budget:]
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order].astype(np.int64).tolist()


def rank_or_zero(values: Sequence[int], target: int) -> int:
    try:
        return values.index(target) + 1
    except ValueError:
        return 0


def reciprocal_rank_fusion(
    groups: Sequence[Sequence[int]], budget: int, rrf_k: float
) -> list[int]:
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for group in groups:
        for rank, block_id_value in enumerate(group, start=1):
            block_id = int(block_id_value)
            scores[block_id] = scores.get(block_id, 0.0) + 1.0 / (rrf_k + rank)
            best_rank[block_id] = min(best_rank.get(block_id, rank), rank)
    return sorted(scores, key=lambda item: (-scores[item], best_rank[item], item))[
        :budget
    ]


def torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


@torch.inference_mode()
def encode_texts(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    prefix: str,
    pooling: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    output = []
    for start in range(0, len(texts), batch_size):
        batch_texts = [prefix + text for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model(**encoded).last_hidden_state
        if pooling == "cls":
            embeddings = hidden[:, 0]
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        embeddings = F.normalize(embeddings.float(), dim=1)
        output.append(embeddings.cpu().numpy().astype(np.float16))
        if start % (batch_size * 50) == 0:
            print(
                json.dumps(
                    {"encoded": min(start + batch_size, len(texts)), "total": len(texts)}
                ),
                flush=True,
            )
    return np.concatenate(output, axis=0)


def build_or_load_index(
    args: argparse.Namespace,
    blocks: np.ndarray,
    block_texts: list[str],
    model: Any,
    embedding_tokenizer: Any,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = index_dir / "block_embeddings.npy"
    metadata_path = index_dir / "summary.json"
    if embeddings_path.exists() and metadata_path.exists() and not args.rebuild_index:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "embedding_model": args.embedding_model_name_or_path,
            "pooling": args.pooling,
            "passage_prefix": args.passage_prefix,
            "max_length": args.max_length,
            "blocks": int(blocks.shape[0]),
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("existing embedding index metadata does not match request")
        return np.load(embeddings_path, mmap_mode="r"), 0.0

    started = time.perf_counter()
    embeddings = encode_texts(
        model,
        embedding_tokenizer,
        block_texts,
        prefix=args.passage_prefix,
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    build_seconds = time.perf_counter() - started
    np.save(embeddings_path, embeddings)
    metadata = {
        "source": "external pretrained embedding index over real 10M text blocks",
        "embedding_model": args.embedding_model_name_or_path,
        "pooling": args.pooling,
        "passage_prefix": args.passage_prefix,
        "max_length": args.max_length,
        "blocks": int(blocks.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "dtype": "float16",
        "build_seconds": build_seconds,
        "contains_synthetic_vectors": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return np.load(embeddings_path, mmap_mode="r"), build_seconds


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    methods = ("dense", "bm25", "hybrid_rrf")
    for split, step_type in sorted(
        {(str(row["split"]), str(row["step_type"])) for row in rows}
    ):
        group = [
            row
            for row in rows
            if str(row["split"]) == split and str(row["step_type"]) == step_type
        ]
        item: dict[str, Any] = {
            "split": split,
            "step_type": step_type,
            "steps": len(group),
        }
        for method in methods:
            for budget in (1, 3, 16, 512):
                item[f"{method}_recall_at_{budget}"] = statistics.fmean(
                    0 < int(row[f"{method}_rank"]) <= budget for row in group
                )
        summaries.append(item)
    return summaries


def main() -> None:
    args = parse_args()
    if args.candidate_blocks <= 0 or args.batch_size <= 0:
        raise ValueError("candidate_blocks and batch_size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = Path(args.corpus_dir)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")

    qwen_tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_tokenizer_name_or_path, use_fast=True
    )
    decode_started = time.perf_counter()
    block_texts = decode_blocks(qwen_tokenizer, blocks)
    decode_seconds = time.perf_counter() - decode_started

    embedding_tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model_name_or_path, use_fast=True
    )
    model = AutoModel.from_pretrained(
        args.embedding_model_name_or_path,
        torch_dtype=torch_dtype(args.dtype) if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    block_embeddings, index_build_seconds = build_or_load_index(
        args, blocks, block_texts, model, embedding_tokenizer, device
    )
    allowed_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    allowed_types = {
        item.strip() for item in args.step_types.split(",") if item.strip()
    }
    steps = [
        row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) in allowed_splits
        and str(row["step_type"]) in allowed_types
    ]
    steps.sort(key=lambda row: (int(row["query_id"]), int(row["step_index"])))
    if args.max_steps > 0:
        steps = steps[: args.max_steps]
    query_texts = [step_state_text(step) for step in steps]
    query_started = time.perf_counter()
    query_embeddings = encode_texts(
        model,
        embedding_tokenizer,
        query_texts,
        prefix=args.query_prefix,
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    ).astype(np.float32)
    query_encode_seconds = time.perf_counter() - query_started
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    bm25_started = time.perf_counter()
    bm25 = BM25Index(block_texts, min_df=1, max_df=1.0, k1=1.2, b=0.75)
    bm25_build_seconds = time.perf_counter() - bm25_started
    bm25_score_started = time.perf_counter()
    bm25_scores = bm25.score(query_texts)
    bm25_score_seconds = time.perf_counter() - bm25_score_started

    dense_score_started = time.perf_counter()
    dense_index = torch.from_numpy(np.asarray(block_embeddings).astype(np.float32)).to(
        device
    )
    query_tensor = torch.from_numpy(query_embeddings).to(device)
    dense_scores = (query_tensor @ dense_index.transpose(0, 1)).cpu().numpy()
    dense_score_seconds = time.perf_counter() - dense_score_started

    rows = []
    ranking_started = time.perf_counter()
    for offset, step in enumerate(steps):
        dense = stable_topk(dense_scores[offset], args.candidate_blocks)
        lexical = stable_topk(bm25_scores[offset], args.candidate_blocks)
        hybrid = reciprocal_rank_fusion(
            [dense, lexical], args.candidate_blocks, args.rrf_k
        )
        target = int(step["target_block_ids"][0])
        rows.append(
            {
                "query_id": int(step["query_id"]),
                "step_index": int(step["step_index"]),
                "split": str(step["split"]),
                "step_type": str(step["step_type"]),
                "selection_uses_gold": False,
                "query_text": query_texts[offset],
                "target_block_id": target,
                "dense_rank": rank_or_zero(dense, target),
                "bm25_rank": rank_or_zero(lexical, target),
                "hybrid_rrf_rank": rank_or_zero(hybrid, target),
                "dense_candidates": dense,
                "bm25_candidates": lexical,
                "hybrid_rrf_candidates": hybrid,
            }
        )
    ranking_seconds = time.perf_counter() - ranking_started
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "standard external-embedding RAG and BM25+dense RRF baseline",
        "embedding_model": args.embedding_model_name_or_path,
        "pooling": args.pooling,
        "query_prefix": args.query_prefix,
        "passage_prefix": args.passage_prefix,
        "selection_uses_gold": False,
        "contains_synthetic_vectors": False,
        "blocks": int(blocks.shape[0]),
        "steps": len(steps),
        "candidate_blocks": args.candidate_blocks,
        "rrf_k": args.rrf_k,
        "decode_seconds": decode_seconds,
        "index_build_seconds": index_build_seconds,
        "query_encode_seconds": query_encode_seconds,
        "bm25_build_seconds": bm25_build_seconds,
        "bm25_score_seconds": bm25_score_seconds,
        "dense_score_seconds": dense_score_seconds,
        "ranking_seconds": ranking_seconds,
        "summaries": summarize(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
