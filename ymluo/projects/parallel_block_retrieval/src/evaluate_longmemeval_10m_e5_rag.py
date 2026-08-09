from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from evaluate_longmemeval_10m_hierarchical_bm25 import (
    interval_blocks,
    selection_metrics,
    summarize,
)
from evaluate_xsum_10m_dynamic_text_retrieval import decode_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate direct and hierarchical E5 RAG on shared LongMemEval 10M."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--decode_tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--embedding_model", default="intfloat/e5-base-v2")
    parser.add_argument("--session_depths", default="1,3,8,16,32")
    parser.add_argument("--topks", default="8,32,128")
    parser.add_argument("--decode_batch_size", type=int, default=4096)
    parser.add_argument("--block_batch_size", type=int, default=512)
    parser.add_argument("--session_batch_size", type=int, default=64)
    parser.add_argument("--block_max_length", type=int, default=128)
    parser.add_argument("--session_max_length", type=int, default=512)
    parser.add_argument("--device_ids", default="0,1,2,3,4,5,6,7")
    return parser.parse_args()


def parse_ints(spec: str) -> list[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) <= 0:
        raise ValueError("integer list must contain positive values")
    return values


def parse_device_ids(spec: str) -> list[int]:
    values = [int(item.strip()) for item in spec.split(",") if item.strip()]
    if not values or min(values) < 0 or len(values) != len(set(values)):
        raise ValueError("device_ids must contain unique non-negative integers")
    return values


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


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
) -> torch.Tensor:
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
        output.append(F.normalize(pooled.float(), dim=1).half().cpu())
        if (start // batch_size + 1) % 25 == 0:
            print(f"encoded {min(start + batch_size, len(texts)):,}/{len(texts):,}", flush=True)
    return torch.cat(output, dim=0)


def cuda_ranking(
    embeddings: torch.Tensor,
    query: torch.Tensor,
    candidate_ids: np.ndarray | None,
    topk: int,
) -> tuple[list[int], float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    if candidate_ids is None:
        scores = embeddings @ query
        take = min(topk, len(scores))
        ranking = torch.topk(scores, take, sorted=True).indices
    else:
        candidate_tensor = torch.as_tensor(candidate_ids, device=embeddings.device)
        scores = embeddings.index_select(0, candidate_tensor) @ query
        take = min(topk, len(scores))
        local = torch.topk(scores, take, sorted=True).indices
        ranking = candidate_tensor.index_select(0, local)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return ranking.cpu().tolist(), seconds


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_depths = parse_ints(args.session_depths)
    topks = parse_ints(args.topks)
    max_topk = max(topks)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    queries = read_jsonl(data_dir / "queries.jsonl")
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

    decode_tokenizer = AutoTokenizer.from_pretrained(args.decode_tokenizer, use_fast=True)
    started = time.perf_counter()
    block_texts = decode_blocks(decode_tokenizer, base_blocks, args.decode_batch_size)
    decode_seconds = time.perf_counter() - started
    session_blocks = []
    session_texts = []
    owner_ids = [int(row["owner_row"]) for row in owners]
    owner_to_index = {owner_id: index for index, owner_id in enumerate(owner_ids)}
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
    sessions_by_owner = [
        np.flatnonzero(session_owner_indices == owner_index).astype(np.int64)
        for owner_index in range(len(owners))
    ]

    device_ids = parse_device_ids(args.device_ids)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{device_ids[0]}")
    embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, use_fast=True)
    base_model = AutoModel.from_pretrained(
        args.embedding_model, torch_dtype=torch.float16
    ).eval().to(device)
    encoder: Any = base_model
    if len(device_ids) > 1:
        encoder = torch.nn.DataParallel(base_model, device_ids=device_ids)

    block_cache = output_dir / "block_embeddings_fp16.pt"
    session_cache = output_dir / "session_embeddings_fp16.pt"
    if block_cache.exists():
        block_embeddings_cpu = torch.load(block_cache, map_location="cpu")
        block_index_seconds = 0.0
    else:
        started = time.perf_counter()
        block_embeddings_cpu = encode_texts(
            encoder,
            embedding_tokenizer,
            block_texts,
            prefix="passage: ",
            batch_size=args.block_batch_size,
            max_length=args.block_max_length,
            device=device,
        )
        block_index_seconds = time.perf_counter() - started
        torch.save(block_embeddings_cpu, block_cache)
    if session_cache.exists():
        session_embeddings_cpu = torch.load(session_cache, map_location="cpu")
        session_index_seconds = 0.0
    else:
        started = time.perf_counter()
        session_embeddings_cpu = encode_texts(
            encoder,
            embedding_tokenizer,
            session_texts,
            prefix="passage: ",
            batch_size=args.session_batch_size,
            max_length=args.session_max_length,
            device=device,
        )
        session_index_seconds = time.perf_counter() - started
        torch.save(session_embeddings_cpu, session_cache)
    del block_texts, session_texts, encoder
    gc.collect()

    block_embeddings = block_embeddings_cpu.to(device)
    session_embeddings = session_embeddings_cpu.to(device)
    rows = []
    query_encode_seconds = []

    @torch.inference_mode()
    def encode_query(text: str) -> torch.Tensor:
        torch.cuda.synchronize()
        started_at = time.perf_counter()
        batch = embedding_tokenizer(
            ["query: " + text],
            padding=True,
            truncation=True,
            max_length=args.block_max_length,
            return_tensors="pt",
        ).to(device)
        hidden = base_model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        embedding = F.normalize(pooled.float(), dim=1).half()[0]
        torch.cuda.synchronize()
        query_encode_seconds.append(time.perf_counter() - started_at)
        return embedding

    def append_result(
        query: dict[str, Any],
        *,
        method: str,
        ranking: list[int],
        candidate_blocks: int,
        query_seconds: float,
        selected_sessions: list[int] | None = None,
        selected_owner_indices: list[int] | None = None,
    ) -> None:
        rows.append(
            {
                "query_id": int(query["query_id"]),
                "question_id": str(query["question_id"]),
                "question_type": str(query["question_type"]),
                "is_abstention": bool(query["is_abstention"]),
                "method": method,
                "query_seconds": query_seconds,
                "candidate_blocks": candidate_blocks,
                "candidate_fraction": candidate_blocks / len(base_blocks),
                "selected_owner_indices": selected_owner_indices or [],
                "selected_session_rows": selected_sessions or [],
                "top_block_ids": ranking,
                "selection_uses_answer": False,
                **selection_metrics(
                    ranking,
                    query=query,
                    block_session_rows=block_session_rows,
                    block_owner_ids=block_owner_ids,
                    topks=topks,
                ),
            }
        )

    for query in queries:
        embedding = encode_query(str(query["question"]))
        encode_seconds = query_encode_seconds[-1]
        direct_ranking, direct_seconds = cuda_ranking(
            block_embeddings, embedding, None, max_topk
        )
        append_result(
            query,
            method="e5_global_block",
            ranking=direct_ranking,
            candidate_blocks=len(base_blocks),
            query_seconds=encode_seconds + direct_seconds,
        )

        session_ranking, session_seconds = cuda_ranking(
            session_embeddings, embedding, None, max(session_depths)
        )
        owner_index = owner_to_index[int(query["owner_row"])]
        owner_sessions = sessions_by_owner[owner_index]
        owner_session_ranking, owner_session_seconds = cuda_ranking(
            session_embeddings, embedding, owner_sessions, max(session_depths)
        )
        for depth in session_depths:
            selected_sessions = session_ranking[:depth]
            candidates = np.unique(
                np.concatenate([session_blocks[item] for item in selected_sessions])
            )
            ranking, block_seconds = cuda_ranking(
                block_embeddings, embedding, candidates, max_topk
            )
            append_result(
                query,
                method=f"e5_global_session{depth}_block",
                ranking=ranking,
                candidate_blocks=len(candidates),
                query_seconds=encode_seconds + session_seconds + block_seconds,
                selected_sessions=selected_sessions,
            )

            selected_owner_sessions = owner_session_ranking[:depth]
            owner_candidates = np.unique(
                np.concatenate(
                    [session_blocks[item] for item in selected_owner_sessions]
                )
            )
            owner_ranking, owner_block_seconds = cuda_ranking(
                block_embeddings, embedding, owner_candidates, max_topk
            )
            append_result(
                query,
                method=f"e5_owner_metadata_session{depth}_block",
                ranking=owner_ranking,
                candidate_blocks=len(owner_candidates),
                query_seconds=(
                    encode_seconds + owner_session_seconds + owner_block_seconds
                ),
                selected_sessions=selected_owner_sessions,
                selected_owner_indices=[owner_index],
            )
        print(f"finished query {int(query['query_id']) + 1}/{len(queries)}", flush=True)

    quality, by_type = summarize(rows, topks=topks, block_tokens=block_tokens)
    summary = {
        "source": "LongMemEval shared-10M E5 dense-RAG baseline",
        "protocol": data_summary,
        "embedding_model": args.embedding_model,
        "devices": device_ids,
        "offline_indexing": {
            "decode_seconds": decode_seconds,
            "block_embedding_seconds": block_index_seconds,
            "session_embedding_seconds": session_index_seconds,
            "block_embeddings": len(block_embeddings_cpu),
            "session_embeddings": len(session_embeddings_cpu),
        },
        "mean_online_query_embedding_seconds": mean(query_encode_seconds),
        "quality": quality,
        "quality_by_question_type": by_type,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
