from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from evaluate_longmemeval_10m_per_head_pca_reader import (
    fit_pca_basis,
    profile_block_keys,
    profile_query,
    read_jsonl,
    selective_head_rrf,
)
from evaluate_xsum_10m_dynamic_text_retrieval import decode_blocks
from run_iterative_condition_retrieval import BM25Index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sampled latency estimate for retrieving 100K tokens from 10M text."
    )
    parser.add_argument("--xsum_data_dir", required=True, type=Path)
    parser.add_argument("--e5_index", required=True, type=Path)
    parser.add_argument("--longmemeval_data_dir", required=True, type=Path)
    parser.add_argument("--llama_model", required=True)
    parser.add_argument("--memory_tokenizer", required=True)
    parser.add_argument("--e5_model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample_queries", type=int, default=8)
    parser.add_argument("--qk_sample_blocks", type=int, default=1024)
    parser.add_argument("--qk_timing_blocks", type=int, default=8192)
    parser.add_argument("--target_memory_tokens", type=int, default=10_000_000)
    parser.add_argument("--target_retrieved_tokens", type=int, default=100_000)
    parser.add_argument("--block_tokens", type=int, default=64)
    parser.add_argument("--fusion_depth", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def rank_numpy(scores: np.ndarray, depth: int) -> np.ndarray:
    depth = min(int(depth), len(scores))
    if depth == len(scores):
        selected = np.arange(len(scores), dtype=np.int64)
    else:
        selected = np.argpartition(-scores, depth - 1)[:depth]
    return selected[np.argsort(-scores[selected], kind="stable")]


def rrf(rankings: list[np.ndarray], depth: int, constant: float = 60.0) -> np.ndarray:
    score: dict[int, float] = {}
    best: dict[int, int] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            block_id = int(item)
            score[block_id] = score.get(block_id, 0.0) + 1.0 / (constant + position)
            best[block_id] = min(best.get(block_id, position), position)
    return np.asarray(
        sorted(score, key=lambda item: (-score[item], best[item], item))[:depth],
        dtype=np.int64,
    )


def cuda_timing_ms(
    function: Callable[[], Any], *, device: torch.device, warmup: int, repeats: int
) -> float:
    for _ in range(warmup):
        function()
    synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    synchronize(device)
    return float(start.elapsed_time(end)) / repeats


def extrapolate(sample_rows: list[dict[str, float]], target_blocks: int) -> dict[str, float]:
    retained = sample_rows[-min(3, len(sample_rows)) :]
    counts = np.asarray([row["blocks"] for row in retained], dtype=np.float64)
    times = np.asarray([row["milliseconds"] for row in retained], dtype=np.float64)
    slope = float(np.dot(counts, times) / np.dot(counts, counts))
    ratios = times * float(target_blocks) / counts
    return {
        "linear_through_origin_ms": slope * target_blocks,
        "sample_ratio_min_ms": float(ratios.min()),
        "sample_ratio_max_ms": float(ratios.max()),
    }


@torch.inference_mode()
def encode_e5_query(
    model: Any, tokenizer: Any, text: str, device: torch.device
) -> torch.Tensor:
    batch = tokenizer(
        ["query: " + text],
        padding=True,
        truncation=True,
        max_length=96,
        return_tensors="pt",
    ).to(device)
    hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return F.normalize(pooled.float(), dim=1).half()[0]


def benchmark_text_retrievers(
    args: argparse.Namespace, device: torch.device, topk: int
) -> dict[str, Any]:
    base_blocks = np.load(args.xsum_data_dir / "base_blocks.npy", mmap_mode="r")
    total_blocks = min(len(base_blocks), args.target_memory_tokens // args.block_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.memory_tokenizer, use_fast=True)
    decode_started = time.perf_counter()
    texts = decode_blocks(tokenizer, base_blocks[:total_blocks], 2048)
    decode_seconds = time.perf_counter() - decode_started
    queries = np.load(args.xsum_data_dir / "queries.npy", mmap_mode="r")
    query_texts = [
        tokenizer.decode(
            np.asarray(queries[index, :64], dtype=np.int64).tolist(),
            skip_special_tokens=True,
        )
        for index in range(min(args.sample_queries, len(queries)))
    ]

    started = time.perf_counter()
    bm25 = BM25Index(texts, min_df=1, max_df=1.0, k1=1.2, b=0.75)
    bm25_build_seconds = time.perf_counter() - started
    bm25.score_postings([query_texts[0]])
    bm25_rows: list[dict[str, Any]] = []
    for query in query_texts:
        started = time.perf_counter()
        scores = bm25.score_postings([query])[0]
        ranking = rank_numpy(scores, max(topk, args.fusion_depth))
        bm25_rows.append(
            {
                "seconds": time.perf_counter() - started,
                "ranking": ranking,
            }
        )

    e5_embeddings_np = np.load(args.e5_index, mmap_mode="r")
    if len(e5_embeddings_np) < total_blocks:
        raise RuntimeError("E5 index does not cover the requested 10M memory")
    e5_embeddings = torch.from_numpy(
        np.array(e5_embeddings_np[:total_blocks], dtype=np.float16, copy=True)
    ).to(device)
    e5_tokenizer = AutoTokenizer.from_pretrained(args.e5_model, use_fast=True)
    e5_model = AutoModel.from_pretrained(
        args.e5_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).eval().to(device)
    warm_query = encode_e5_query(e5_model, e5_tokenizer, query_texts[0], device)
    torch.topk(
        e5_embeddings @ warm_query,
        k=min(max(topk, args.fusion_depth), total_blocks),
        sorted=True,
    )
    synchronize(device)
    e5_rows: list[dict[str, Any]] = []
    for query in query_texts:
        synchronize(device)
        started = time.perf_counter()
        query_vector = encode_e5_query(e5_model, e5_tokenizer, query, device)
        synchronize(device)
        encode_seconds = time.perf_counter() - started

        synchronize(device)
        started = time.perf_counter()
        scores = e5_embeddings @ query_vector
        ranking = torch.topk(
            scores, k=min(max(topk, args.fusion_depth), total_blocks), sorted=True
        ).indices
        synchronize(device)
        scan_seconds = time.perf_counter() - started
        e5_rows.append(
            {
                "encode_seconds": encode_seconds,
                "scan_topk_seconds": scan_seconds,
                "ranking": ranking.cpu().numpy(),
            }
        )

    # Warm the Python allocator and suppress GC pauses during this short latency probe.
    rrf([bm25_rows[0]["ranking"], e5_rows[0]["ranking"]], topk)
    rrf_seconds = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for bm25_row, e5_row in zip(bm25_rows, e5_rows):
            started = time.perf_counter()
            rrf(
                [bm25_row["ranking"], e5_row["ranking"]],
                topk,
            )
            rrf_seconds.append(time.perf_counter() - started)
    finally:
        if gc_was_enabled:
            gc.enable()
    bm25_seconds = [float(row["seconds"]) for row in bm25_rows]
    e5_encode = [float(row["encode_seconds"]) for row in e5_rows]
    e5_scan = [float(row["scan_topk_seconds"]) for row in e5_rows]
    hybrid = [
        bm25_seconds[index] + e5_encode[index] + e5_scan[index] + rrf_seconds[index]
        for index in range(len(query_texts))
    ]
    result = {
        "corpus": "real XSum BBC news 10M memory",
        "blocks": total_blocks,
        "block_tokens": args.block_tokens,
        "queries": len(query_texts),
        "decode_seconds": decode_seconds,
        "bm25_index_build_seconds": bm25_build_seconds,
        "bm25_index_bytes_in_memory": int(
            bm25.weighted_documents.data.nbytes
            + bm25.weighted_documents.indices.nbytes
            + bm25.weighted_documents.indptr.nbytes
            + bm25.weighted_documents_csc.data.nbytes
            + bm25.weighted_documents_csc.indices.nbytes
            + bm25.weighted_documents_csc.indptr.nbytes
        ),
        "e5_index_build_seconds_from_existing_run": 42.25992461713031,
        "e5_index_bytes": int(e5_embeddings_np[:total_blocks].nbytes),
        "bm25_online_seconds": quantiles(bm25_seconds),
        "e5_query_encode_seconds": quantiles(e5_encode),
        "e5_scan_topk_seconds": quantiles(e5_scan),
        "e5_dense_rag_online_seconds": quantiles(
            left + right for left, right in zip(e5_encode, e5_scan)
        ),
        "hybrid_rrf_seconds": quantiles(rrf_seconds),
        "hybrid_bm25_e5_rag_online_seconds": quantiles(hybrid),
    }
    del e5_model, e5_embeddings, bm25, texts
    gc.collect()
    torch.cuda.empty_cache()
    return result


def pack_projected_int4(projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scales = projected.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    codes = torch.round(projected.float() / scales).clamp(-7, 7).to(torch.int16) + 7
    packed = codes[..., 0::2].to(torch.uint8) | (
        codes[..., 1::2].to(torch.uint8) << 4
    )
    return packed, scales.to(projected.dtype)


def benchmark_cpu_rrf(
    channels: int, blocks: int, topk: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((channels, blocks), dtype=np.float32)
    values = []
    for _ in range(3):
        started = time.perf_counter()
        selective_head_rrf(
            scores,
            output_depth=topk,
            active_fraction=16 / channels,
            per_head_depth=topk,
            rrf_constant=20.0,
        )
        values.append(time.perf_counter() - started)
    return quantiles(values)


@torch.inference_mode()
def benchmark_model_native(
    args: argparse.Namespace, device: torch.device, total_blocks: int, topk: int
) -> dict[str, Any]:
    pca_src = Path(__file__).resolve().parents[2] / "qwen3_top2_head_limit3_ppl" / "src"
    sys.path.insert(0, str(pca_src))
    import qabs_cuda_kernels as kernels

    tokenizer = AutoTokenizer.from_pretrained(args.llama_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    memory_tokenizer = AutoTokenizer.from_pretrained(args.memory_tokenizer, use_fast=True)
    blocks = np.load(args.longmemeval_data_dir / "base_blocks.npy", mmap_mode="r")
    queries = read_jsonl(args.longmemeval_data_dir / "queries.jsonl")
    sample_count = min(args.qk_sample_blocks, len(blocks))
    rng = np.random.default_rng(args.seed)
    sample_ids = rng.choice(len(blocks), size=sample_count, replace=False)
    texts = [
        memory_tokenizer.decode(blocks[int(item)], skip_special_tokens=True)
        for item in sample_ids
    ]
    synchronize(device)
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.llama_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).eval().to(device)
    synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    layers = [3, 7, 11, 15, 19, 23, 27, 31]

    synchronize(device)
    profile_started = time.perf_counter()
    raw_keys = profile_block_keys(
        model,
        tokenizer,
        texts,
        device=device,
        batch_size=8,
        max_tokens=128,
        segments=4,
        layer_indices=layers,
    )
    synchronize(device)
    block_profile_seconds = time.perf_counter() - profile_started
    basis, retained = fit_pca_basis(
        raw_keys[: min(256, sample_count)], projection_dim=64, device=device
    )
    raw_query, projected_query = profile_query(
        model,
        tokenizer,
        str(queries[0]["question"]),
        basis,
        device=device,
        tail_tokens=8,
        layer_indices=layers,
    )
    synchronize(device)

    timing_count = max(sample_count, args.qk_timing_blocks)
    repeats = math.ceil(timing_count / sample_count)
    raw_gpu = raw_keys.to(device).repeat(repeats, 1, 1, 1, 1)[:timing_count]
    pack_started = time.perf_counter()
    projected_keys = torch.einsum("nlksd,lkdr->nlksr", raw_gpu, basis)
    projected_layout = projected_keys.permute(1, 2, 0, 3, 4).reshape(
        len(layers), int(model.config.num_key_value_heads), timing_count * 4, 64
    )
    packed, scales = pack_projected_int4(projected_layout)
    synchronize(device)
    sample_pack_seconds = time.perf_counter() - pack_started

    query_heads = int(model.config.num_attention_heads)
    kv_heads = int(model.config.num_key_value_heads)
    groups = query_heads // kv_heads
    raw_grouped = raw_query.reshape(len(layers), kv_heads, groups, 8, 128)
    projected_grouped = projected_query.reshape(len(layers), kv_heads, groups, 8, 64)
    query_scales = (
        projected_grouped.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
        / 127.0
    )
    query_codes = (
        torch.round(projected_grouped.float() / query_scales)
        .clamp(-127, 127)
        .to(torch.int8)
    )
    pca_index_by_count = {
        count: (
            packed[:, :, : count * 4].contiguous(),
            scales[:, :, : count * 4].contiguous(),
        )
        for count in sorted(
            {
                value
                for value in (1024, 2048, 4096, timing_count)
                if 0 < value <= timing_count
            }
        )
    }

    def full_scores(count: int) -> torch.Tensor:
        channel_parts = []
        keys = raw_gpu[:count]
        for layer in range(len(layers)):
            score = torch.einsum(
                "kgtd,nksd->kgtns", raw_grouped[layer], keys[:, layer]
            ).amax(dim=(2, 4))
            channel_parts.append(score.reshape(query_heads, count))
        return torch.cat(channel_parts, dim=0)

    def pca_scores(count: int) -> torch.Tensor:
        key_count = count * 4
        local_packed, local_scales = pca_index_by_count[count]
        best = None
        for token in range(8):
            score = kernels.pca_int4_scores(
                query_codes[:, :, :, token].contiguous(),
                local_packed,
                local_scales,
                key_count,
            ).reshape(len(layers), kv_heads, groups, count, 4)
            score = score * query_scales[:, :, :, token, :, None]
            reduced = score.amax(dim=-1).reshape(len(layers) * query_heads, count)
            best = reduced if best is None else torch.maximum(best, reduced)
        if best is None:
            raise RuntimeError("empty query")
        return best

    sample_sizes = sorted(
        {
            value
            for value in (1024, 2048, 4096, timing_count)
            if 0 < value <= timing_count
        }
    )
    full_rows = []
    pca_rows = []
    for count in sample_sizes:
        full_ms = cuda_timing_ms(
            lambda count=count: full_scores(count),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        pca_ms = cuda_timing_ms(
            lambda count=count: pca_scores(count),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        full_rows.append({"blocks": float(count), "milliseconds": full_ms})
        pca_rows.append({"blocks": float(count), "milliseconds": pca_ms})

    full_estimate = extrapolate(full_rows, total_blocks)
    pca_estimate = extrapolate(pca_rows, total_blocks)
    cpu_rrf = benchmark_cpu_rrf(256, total_blocks, topk, args.seed)
    query_profile_seconds = 0.03480420842580497
    full_online = (
        query_profile_seconds
        + full_estimate["linear_through_origin_ms"] / 1000.0
        + cpu_rrf["median"]
    )
    pca_online = (
        query_profile_seconds
        + pca_estimate["linear_through_origin_ms"] / 1000.0
        + cpu_rrf["median"]
    )
    vectors_per_block = len(layers) * kv_heads * 4
    full_index_bytes = total_blocks * vectors_per_block * 128 * 2
    pca_index_bytes = total_blocks * vectors_per_block * (64 // 2 + 2)
    result = {
        "corpus": "real LongMemEval blocks profiled by Llama-3.1-8B",
        "sample_blocks": sample_count,
        "timing_blocks_after_repeating_real_profiles": timing_count,
        "timing_repeat_note": (
            "The real 1,024-block profile is repeated only to reach the stable GPU "
            "bandwidth regime; repeated values are not used for quality metrics."
        ),
        "layers": layers,
        "kv_heads": kv_heads,
        "query_heads": query_heads,
        "query_tail_tokens": 8,
        "segments_per_block": 4,
        "selected_layer_head_channels_after_scoring": 16,
        "model_load_seconds": model_load_seconds,
        "sample_block_profile_seconds": block_profile_seconds,
        "profile_seconds_per_block": block_profile_seconds / sample_count,
        "estimated_online_profile_all_10m_one_gpu_seconds": (
            block_profile_seconds / sample_count * total_blocks
        ),
        "estimated_online_profile_all_10m_eight_gpu_seconds": (
            block_profile_seconds / sample_count * total_blocks / 8
        ),
        "sample_pca_pack_seconds": sample_pack_seconds,
        "estimated_offline_pca_pack_all_10m_seconds": (
            sample_pack_seconds / timing_count * total_blocks
        ),
        "pca_retained_energy_mean": statistics.fmean(retained),
        "full_qk_scan_samples": full_rows,
        "pca64_int4_scan_samples": pca_rows,
        "full_qk_scan_10m_estimate": full_estimate,
        "pca64_int4_scan_10m_estimate": pca_estimate,
        "cpu_selected_head_rrf_topk_seconds": cpu_rrf,
        "query_q_profile_reference_seconds": query_profile_seconds,
        "full_qk_gpu_resident_online_estimate_seconds": full_online,
        "pca64_int4_gpu_resident_online_estimate_seconds": pca_online,
        "full_qk_index_bytes": int(full_index_bytes),
        "pca64_int4_index_bytes": int(pca_index_bytes),
        "index_note": (
            "Current L1 geometry: 8 layers x 8 KV heads x 4 segment means per block. "
            "The benchmark scores every 8-layer x 32-query-head channel before selecting 16."
        ),
    }
    del model, raw_gpu, projected_keys, projected_layout, packed, scales
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    total_blocks = math.ceil(args.target_memory_tokens / args.block_tokens)
    topk = math.ceil(args.target_retrieved_tokens / args.block_tokens)
    text_result = benchmark_text_retrievers(args, device, topk)
    model_result = benchmark_model_native(args, device, total_blocks, topk)
    report = {
        "source": "sampled real-data latency estimate for 10M to 100K-token retrieval",
        "protocol": {
            "memory_tokens": args.target_memory_tokens,
            "block_tokens": args.block_tokens,
            "memory_blocks": total_blocks,
            "retrieved_tokens_target": args.target_retrieved_tokens,
            "retrieved_blocks": topk,
            "online_excludes_offline_index_build": True,
            "reader_prefill_and_generation_excluded": True,
            "selection_uses_answer": False,
        },
        "text_retrieval": text_result,
        "model_native_retrieval": model_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
