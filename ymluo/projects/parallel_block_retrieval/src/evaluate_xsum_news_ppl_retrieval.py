from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from profile_real_qk import (
    QKCapture,
    captured_qk,
    parse_pairs,
    resolve_dtype,
    run_base_model,
)
from run_iterative_condition_retrieval import BM25Index


BASE_METHODS = (
    "query_only",
    "recent512",
    "random512",
    "bm25_512",
    "e5_512",
    "hybrid_rrf_512",
    "hybrid_recent_512",
    "qk_kmean_512",
    "qk_full128_512",
    "qk_svd32_512",
    "hybrid_qk_rrf_512",
    "full40k",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 512-token retrieval PPL on causal 40K XSum news contexts."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--embedding_model_name_or_path", default="intfloat/e5-base-v2"
    )
    parser.add_argument("--pairs", default="3:10,21:8,6:7,16:14")
    parser.add_argument("--svd_rank", type=int, default=32)
    parser.add_argument("--query_q_tokens", type=int, default=16)
    parser.add_argument("--retrieval_blocks", type=int, default=8)
    parser.add_argument("--qk_batch_blocks", type=int, default=32)
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--embedding_max_length", type=int, default=192)
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--protocols", default="natural_stream,delayed_article")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--skip_e5", action="store_true")
    parser.add_argument("--skip_qk", action="store_true")
    parser.add_argument("--skip_full40k", action="store_true")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_ranking(scores: np.ndarray) -> list[int]:
    block_ids = np.arange(len(scores), dtype=np.int64)
    return np.lexsort((block_ids, -np.asarray(scores, dtype=np.float64))).tolist()


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], *, budget: int, rrf_k: float
) -> list[int]:
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for ranking in rankings:
        for rank, block_id_value in enumerate(ranking, start=1):
            block_id = int(block_id_value)
            scores[block_id] = scores.get(block_id, 0.0) + 1.0 / (rrf_k + rank)
            best_rank[block_id] = min(best_rank.get(block_id, rank), rank)
    return sorted(scores, key=lambda item: (-scores[item], best_rank[item], item))[
        :budget
    ]


@torch.inference_mode()
def encode_e5(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    prefix: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            [prefix + text for text in texts[start : start + batch_size]],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        outputs.append(F.normalize(embeddings.float(), dim=1))
    return torch.cat(outputs, dim=0)


def pair_specs_for_model(
    model: AutoModelForCausalLM, pair_text: str
) -> list[dict[str, int]]:
    config = model.config
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    repeat_groups = query_heads // kv_heads
    specs: list[dict[str, int]] = []
    for layer, query_head in parse_pairs(pair_text):
        if not 0 <= layer < len(model.model.layers):
            raise ValueError(f"layer {layer} is outside this model")
        if not 0 <= query_head < query_heads:
            raise ValueError(f"query head {query_head} is outside this model")
        specs.append(
            {
                "layer": layer,
                "query_head": query_head,
                "kv_head": query_head // repeat_groups,
            }
        )
    return specs


def fit_svd_basis(keys: torch.Tensor, rank: int) -> tuple[torch.Tensor, list[float]]:
    profile_count = int(keys.shape[2])
    head_dim = int(keys.shape[3])
    basis_parts: list[torch.Tensor] = []
    retained: list[float] = []
    for profile in range(profile_count):
        matrix = keys[:, :, profile, :].reshape(-1, head_dim).float()
        mean = matrix.mean(dim=0)
        centered = matrix - mean
        covariance = centered.transpose(0, 1) @ centered
        values, vectors = torch.linalg.eigh(covariance)
        order = torch.argsort(values, descending=True)
        selected = order[:rank]
        basis_parts.append(vectors.index_select(1, selected).contiguous())
        retained.append(
            float(
                (
                    values.index_select(0, selected).clamp_min(0).sum()
                    / values.clamp_min(0).sum().clamp_min(1.0e-30)
                ).item()
            )
        )
    return torch.stack(basis_parts), retained


def token_qk_scores(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    similarities = torch.einsum("qpd,btpd->qbpt", query.float(), keys.float())
    return similarities.amax(dim=(0, 2, 3)) / math.sqrt(float(query.shape[-1]))


def kmean_scores(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    means = keys.float().mean(dim=1)
    centered = means - means.mean(dim=0, keepdim=True)
    query_cos = F.normalize(query.float(), dim=-1)
    key_cos = F.normalize(centered, dim=-1)
    similarities = torch.einsum("qpd,bpd->qbp", query_cos, key_cos)
    return similarities.amax(dim=(0, 2))


def build_qk_rankings(
    model: AutoModelForCausalLM,
    blocks: np.ndarray,
    query_ids: np.ndarray,
    pair_specs: list[dict[str, int]],
    *,
    svd_rank: int,
    query_q_tokens: int,
    batch_blocks: int,
    device: torch.device,
) -> tuple[dict[str, list[int]], dict[str, float], list[float]]:
    capture = QKCapture(model, sorted({int(item["layer"]) for item in pair_specs}))
    key_parts: list[torch.Tensor] = []
    synchronize(device)
    started = time.perf_counter()
    for start in range(0, len(blocks), batch_blocks):
        input_ids = torch.from_numpy(
            np.asarray(blocks[start : start + batch_blocks], dtype=np.int64)
        ).to(device)
        run_base_model(model, capture, input_ids)
        _, keys = captured_qk(model, capture, pair_specs, "pre_rope_block_qk")
        key_parts.append(keys.to(torch.float16))
    keys = torch.cat(key_parts, dim=0)
    synchronize(device)
    key_capture_seconds = time.perf_counter() - started

    synchronize(device)
    started = time.perf_counter()
    basis, retained_energy = fit_svd_basis(keys, svd_rank)
    svd_keys = torch.einsum(
        "btpd,pdr->btpr", keys.float(), basis.float()
    ).to(torch.float16)
    synchronize(device)
    svd_build_seconds = time.perf_counter() - started

    input_ids = torch.from_numpy(np.asarray(query_ids, dtype=np.int64))[None, :].to(device)
    synchronize(device)
    started = time.perf_counter()
    run_base_model(model, capture, input_ids)
    query, _ = captured_qk(model, capture, pair_specs, "pre_rope_block_qk")
    keep = min(query_q_tokens, int(query.shape[1]))
    query = query[0, -keep:].to(torch.float16)
    projected_query = torch.einsum(
        "qpd,pdr->qpr", query.float(), basis.float()
    ).to(torch.float16)
    synchronize(device)
    query_capture_seconds = time.perf_counter() - started

    method_scores: dict[str, np.ndarray] = {}
    score_seconds: dict[str, float] = {}
    for method, score_function, score_query, score_keys in (
        ("qk_kmean_512", kmean_scores, query, keys),
        ("qk_full128_512", token_qk_scores, query, keys),
        ("qk_svd32_512", token_qk_scores, projected_query, svd_keys),
    ):
        synchronize(device)
        started = time.perf_counter()
        scores = score_function(score_query, score_keys)
        synchronize(device)
        score_seconds[method] = time.perf_counter() - started
        method_scores[method] = scores.float().cpu().numpy()

    capture.close()
    del keys, svd_keys, basis, query, projected_query
    rankings = {method: stable_ranking(scores) for method, scores in method_scores.items()}
    timings = {
        "qk_key_capture_seconds": key_capture_seconds,
        "qk_svd_build_seconds": svd_build_seconds,
        "qk_query_capture_seconds": query_capture_seconds,
        **{f"{method}_score_seconds": value for method, value in score_seconds.items()},
    }
    return rankings, timings, retained_energy


@torch.inference_mode()
def target_nll(
    model: AutoModelForCausalLM,
    context_ids: np.ndarray,
    query_ids: np.ndarray,
    target_ids: np.ndarray,
    device: torch.device,
) -> tuple[float, float, int, float, int]:
    context = torch.from_numpy(np.asarray(context_ids, dtype=np.int64))
    query = torch.from_numpy(np.asarray(query_ids, dtype=np.int64))
    target = torch.from_numpy(np.asarray(target_ids, dtype=np.int64))
    prompt = torch.cat([context, query], dim=0)
    input_ids = torch.cat([prompt, target], dim=0)[None, :].to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    prompt_tokens = int(prompt.numel())
    target_tokens = int(target.numel())
    synchronize(device)
    started = time.perf_counter()
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    positions = torch.arange(
        prompt_tokens - 1,
        prompt_tokens + target_tokens - 1,
        device=device,
        dtype=torch.long,
    )
    hidden = outputs.last_hidden_state[0].index_select(0, positions)
    logits = model.lm_head(hidden).float()
    targets = input_ids[0, prompt_tokens : prompt_tokens + target_tokens]
    losses = F.cross_entropy(logits, targets, reduction="none")
    synchronize(device)
    elapsed = time.perf_counter() - started
    mean_nll = float(losses.mean().item())
    return (
        mean_nll,
        float(losses.sum().item()),
        target_tokens,
        elapsed,
        int(input_ids.shape[1]),
    )


def selected_context(blocks: np.ndarray, block_ids: Sequence[int]) -> np.ndarray:
    ordered = sorted({int(item) for item in block_ids})
    return np.asarray(blocks[ordered], dtype=np.int32).reshape(-1)


def retrieval_quality(
    selected: Sequence[int], oracle: Sequence[int]
) -> dict[str, float | bool | None]:
    if not oracle:
        return {
            "oracle_block_recall": None,
            "oracle_any_hit": None,
            "oracle_last_block_hit": None,
        }
    selected_set = set(int(item) for item in selected)
    oracle_set = set(int(item) for item in oracle)
    return {
        "oracle_block_recall": len(selected_set & oracle_set) / len(oracle_set),
        "oracle_any_hit": bool(selected_set & oracle_set),
        "oracle_last_block_hit": int(oracle[-1]) in selected_set,
    }


def bootstrap_mean_ci(values: Sequence[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(2000, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(
    rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    result_rows: list[dict[str, Any]] = []
    protocols = sorted({str(row["protocol"]) for row in rows})
    for protocol in protocols:
        protocol_rows = [row for row in rows if str(row["protocol"]) == protocol]
        by_method = {
            method: [row for row in protocol_rows if str(row["method"]) == method]
            for method in sorted({str(row["method"]) for row in protocol_rows})
        }
        full_by_sample = {
            int(row["sample_id"]): float(row["mean_nll"])
            for row in by_method.get("full40k", [])
        }
        oracle_by_sample = {
            int(row["sample_id"]): float(row["mean_nll"])
            for row in by_method.get("oracle_source512", [])
        }
        for method, group in by_method.items():
            total_nll = sum(float(row["total_nll"]) for row in group)
            total_tokens = sum(int(row["target_tokens"]) for row in group)
            micro_nll = total_nll / total_tokens
            item: dict[str, Any] = {
                "protocol": protocol,
                "method": method,
                "samples": len(group),
                "retrieved_tokens": int(group[0]["retrieved_tokens"]),
                "mean_sample_nll": statistics.fmean(float(row["mean_nll"]) for row in group),
                "micro_nll": micro_nll,
                "ppl": math.exp(min(micro_nll, 20.0)),
                "median_forward_seconds": statistics.median(
                    float(row["forward_seconds"]) for row in group
                ),
                "mean_forward_seconds": statistics.fmean(
                    float(row["forward_seconds"]) for row in group
                ),
                "mean_retrieval_seconds": statistics.fmean(
                    float(row["retrieval_seconds"]) for row in group
                ),
                "mean_online_seconds": statistics.fmean(
                    float(row["retrieval_seconds"]) + float(row["forward_seconds"])
                    for row in group
                ),
            }
            recalls = [
                float(row["oracle_block_recall"])
                for row in group
                if row["oracle_block_recall"] is not None
            ]
            if recalls:
                item["mean_oracle_block_recall"] = statistics.fmean(recalls)
                item["oracle_any_hit_rate"] = statistics.fmean(
                    bool(row["oracle_any_hit"]) for row in group
                )
                item["oracle_last_block_hit_rate"] = statistics.fmean(
                    bool(row["oracle_last_block_hit"]) for row in group
                )
            paired_full = [
                float(row["mean_nll"]) - full_by_sample[int(row["sample_id"])]
                for row in group
                if int(row["sample_id"]) in full_by_sample
            ]
            if paired_full:
                item["mean_delta_nll_vs_full40k"] = statistics.fmean(paired_full)
                item["delta_nll_vs_full40k_bootstrap95"] = bootstrap_mean_ci(
                    paired_full, seed + len(result_rows)
                )
            paired_oracle = [
                float(row["mean_nll"]) - oracle_by_sample[int(row["sample_id"])]
                for row in group
                if int(row["sample_id"]) in oracle_by_sample
            ]
            if paired_oracle:
                item["mean_delta_nll_vs_oracle512"] = statistics.fmean(paired_oracle)
            result_rows.append(item)

    index_summary: list[dict[str, Any]] = []
    for protocol in sorted({str(row["protocol"]) for row in index_rows}):
        group = [row for row in index_rows if str(row["protocol"]) == protocol]
        keys = sorted(
            {
                key
                for row in group
                for key, value in row.items()
                if key.endswith("_seconds") and isinstance(value, (int, float))
            }
        )
        item: dict[str, Any] = {"protocol": protocol, "samples": len(group)}
        for key in keys:
            values = [float(row[key]) for row in group if key in row]
            item[f"mean_{key}"] = statistics.fmean(values)
        energies = [
            float(value)
            for row in group
            for value in row.get("svd_retained_energy", [])
        ]
        if energies:
            item["mean_svd32_retained_energy"] = statistics.fmean(energies)
        index_summary.append(item)
    return {"quality_and_online_time": result_rows, "offline_index_time": index_summary}


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = setup_distributed()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)

    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    histories = np.load(data_dir / "histories.npy", mmap_mode="r")
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    targets = np.load(data_dir / "targets.npy", mmap_mode="r")
    metadata = read_jsonl(data_dir / "metadata.jsonl")
    if not (len(histories) == len(queries) == len(targets) == len(metadata)):
        raise ValueError("data arrays and metadata do not align")
    if int(histories.shape[1]) % int(data_summary["block_tokens"]):
        raise ValueError("history length is not block aligned")
    block_tokens = int(data_summary["block_tokens"])
    retrieved_tokens = args.retrieval_blocks * block_tokens
    allowed_protocols = {
        item.strip() for item in args.protocols.split(",") if item.strip()
    }
    sample_ids = [
        index
        for index, row in enumerate(metadata)
        if str(row["protocol"]) in allowed_protocols
    ]
    if args.max_samples > 0:
        sample_ids = sample_ids[: args.max_samples]
    local_sample_ids = [item for item in sample_ids if item % world_size == rank]

    dtype = resolve_dtype(args.dtype)
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    qwen_model.eval()
    qwen_model.config.use_cache = False
    pair_specs = pair_specs_for_model(qwen_model, args.pairs)

    embedding_tokenizer = None
    embedding_model = None
    if not args.skip_e5:
        embedding_tokenizer = AutoTokenizer.from_pretrained(
            args.embedding_model_name_or_path, use_fast=True
        )
        embedding_model = AutoModel.from_pretrained(
            args.embedding_model_name_or_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        embedding_model.eval()

    if local_sample_ids:
        target_nll(
            qwen_model,
            np.empty(0, dtype=np.int32),
            np.asarray(queries[local_sample_ids[0]], dtype=np.int32),
            np.asarray(targets[local_sample_ids[0]], dtype=np.int32),
            device,
        )

    local_rows: list[dict[str, Any]] = []
    local_index_rows: list[dict[str, Any]] = []
    for local_index, sample_id in enumerate(local_sample_ids, start=1):
        meta = metadata[sample_id]
        history = np.asarray(histories[sample_id], dtype=np.int32)
        query_ids = np.asarray(queries[sample_id], dtype=np.int32)
        target_ids = np.asarray(targets[sample_id], dtype=np.int32)
        blocks = history.reshape(-1, block_tokens)
        block_texts = qwen_tokenizer.batch_decode(
            blocks.tolist(), skip_special_tokens=True
        )
        query_text = qwen_tokenizer.decode(query_ids.tolist(), skip_special_tokens=True)
        index_timing: dict[str, Any] = {
            "sample_id": sample_id,
            "protocol": str(meta["protocol"]),
        }

        started = time.perf_counter()
        bm25 = BM25Index(block_texts, min_df=1, max_df=1.0, k1=1.2, b=0.75)
        index_timing["bm25_index_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        bm25_scores = bm25.score_postings([query_text])[0]
        bm25_ranking = stable_ranking(bm25_scores)
        bm25_query_seconds = time.perf_counter() - started

        e5_ranking: list[int] = []
        e5_query_seconds = 0.0
        if embedding_model is not None and embedding_tokenizer is not None:
            synchronize(device)
            started = time.perf_counter()
            block_embeddings = encode_e5(
                embedding_model,
                embedding_tokenizer,
                block_texts,
                prefix="passage: ",
                batch_size=args.embedding_batch_size,
                max_length=args.embedding_max_length,
                device=device,
            )
            synchronize(device)
            index_timing["e5_index_seconds"] = time.perf_counter() - started
            synchronize(device)
            started = time.perf_counter()
            query_embedding = encode_e5(
                embedding_model,
                embedding_tokenizer,
                [query_text],
                prefix="query: ",
                batch_size=1,
                max_length=args.embedding_max_length,
                device=device,
            )
            e5_scores = (query_embedding @ block_embeddings.transpose(0, 1))[0]
            synchronize(device)
            e5_query_seconds = time.perf_counter() - started
            e5_ranking = stable_ranking(e5_scores.float().cpu().numpy())
            del block_embeddings, query_embedding, e5_scores

        qk_rankings: dict[str, list[int]] = {}
        qk_timings: dict[str, float] = {}
        retained_energy: list[float] = []
        if not args.skip_qk:
            qk_rankings, qk_timings, retained_energy = build_qk_rankings(
                qwen_model,
                blocks,
                query_ids,
                pair_specs,
                svd_rank=args.svd_rank,
                query_q_tokens=args.query_q_tokens,
                batch_blocks=args.qk_batch_blocks,
                device=device,
            )
            index_timing.update(qk_timings)
            index_timing["svd_retained_energy"] = retained_energy

        selections: dict[str, list[int]] = {
            "query_only": [],
            "recent512": list(range(len(blocks) - args.retrieval_blocks, len(blocks))),
            "random512": sorted(
                random.Random(args.seed + sample_id).sample(
                    range(len(blocks)), args.retrieval_blocks
                )
            ),
            "bm25_512": bm25_ranking[: args.retrieval_blocks],
        }
        retrieval_times: dict[str, float] = {
            "query_only": 0.0,
            "recent512": 0.0,
            "random512": 0.0,
            "bm25_512": bm25_query_seconds,
        }
        if e5_ranking:
            fusion_started = time.perf_counter()
            hybrid_ranking = reciprocal_rank_fusion(
                [bm25_ranking, e5_ranking], budget=len(blocks), rrf_k=args.rrf_k
            )
            hybrid_fusion_seconds = time.perf_counter() - fusion_started
            selections["e5_512"] = e5_ranking[: args.retrieval_blocks]
            selections["hybrid_rrf_512"] = hybrid_ranking[: args.retrieval_blocks]
            recent_half = list(
                range(len(blocks) - args.retrieval_blocks // 2, len(blocks))
            )
            hybrid_non_recent = [item for item in hybrid_ranking if item not in recent_half]
            selections["hybrid_recent_512"] = (
                recent_half
                + hybrid_non_recent[: args.retrieval_blocks - len(recent_half)]
            )
            retrieval_times["e5_512"] = e5_query_seconds
            retrieval_times["hybrid_rrf_512"] = (
                bm25_query_seconds + e5_query_seconds + hybrid_fusion_seconds
            )
            retrieval_times["hybrid_recent_512"] = retrieval_times["hybrid_rrf_512"]

        for method, ranking in qk_rankings.items():
            selections[method] = ranking[: args.retrieval_blocks]
            retrieval_times[method] = (
                qk_timings["qk_query_capture_seconds"]
                + qk_timings[f"{method}_score_seconds"]
            )
        if e5_ranking and "qk_svd32_512" in qk_rankings:
            fusion_started = time.perf_counter()
            hybrid_qk_ranking = reciprocal_rank_fusion(
                [bm25_ranking, e5_ranking, qk_rankings["qk_svd32_512"]],
                budget=len(blocks),
                rrf_k=args.rrf_k,
            )
            hybrid_qk_fusion_seconds = time.perf_counter() - fusion_started
            selections["hybrid_qk_rrf_512"] = hybrid_qk_ranking[: args.retrieval_blocks]
            retrieval_times["hybrid_qk_rrf_512"] = (
                bm25_query_seconds
                + e5_query_seconds
                + qk_timings["qk_query_capture_seconds"]
                + qk_timings["qk_svd32_512_score_seconds"]
                + hybrid_qk_fusion_seconds
            )

        if str(meta["protocol"]) == "delayed_article":
            selections["oracle_source512"] = [
                int(item) for item in meta["oracle_block_ids"]
            ][: args.retrieval_blocks]
            retrieval_times["oracle_source512"] = 0.0

        if not args.skip_full40k:
            selections["full40k"] = list(range(len(blocks)))
            retrieval_times["full40k"] = 0.0

        oracle = [int(item) for item in meta.get("oracle_block_ids", [])]
        preferred_order = list(BASE_METHODS) + ["oracle_source512"]
        for method in preferred_order:
            if method not in selections:
                continue
            selected = selections[method]
            context = history if method == "full40k" else selected_context(blocks, selected)
            try:
                mean_nll, total_nll, target_count, forward_seconds, model_input_tokens = target_nll(
                    qwen_model, context, query_ids, target_ids, device
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"OOM while scoring {method} for sample {sample_id}"
                )
            row = {
                "sample_id": sample_id,
                "protocol": str(meta["protocol"]),
                "method": method,
                "selected_block_ids": [int(item) for item in selected],
                "retrieved_tokens": int(len(selected) * block_tokens),
                "query_tokens": int(len(query_ids)),
                "target_tokens": target_count,
                "model_input_tokens": model_input_tokens,
                "mean_nll": mean_nll,
                "total_nll": total_nll,
                "ppl": math.exp(min(mean_nll, 20.0)),
                "retrieval_seconds": float(retrieval_times[method]),
                "forward_seconds": forward_seconds,
                "selection_uses_target": False,
                **retrieval_quality(selected, oracle),
            }
            local_rows.append(row)
        local_index_rows.append(index_timing)
        print(
            json.dumps(
                {
                    "rank": rank,
                    "local_sample": local_index,
                    "local_total": len(local_sample_ids),
                    "sample_id": sample_id,
                    "protocol": meta["protocol"],
                }
            ),
            flush=True,
        )

    rows_path = output_dir / f"rows_rank{rank:03d}.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in local_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    index_path = output_dir / f"index_rows_rank{rank:03d}.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for row in local_index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    barrier(world_size)

    if rank == 0:
        all_rows = [
            row
            for shard in range(world_size)
            for row in read_jsonl(output_dir / f"rows_rank{shard:03d}.jsonl")
        ]
        all_index_rows = [
            row
            for shard in range(world_size)
            for row in read_jsonl(output_dir / f"index_rows_rank{shard:03d}.jsonl")
        ]
        all_rows.sort(key=lambda row: (int(row["sample_id"]), str(row["method"])))
        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (output_dir / "index_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in sorted(all_index_rows, key=lambda item: int(item["sample_id"])):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "source": "causal XSum 40K news PPL retrieval benchmark",
            "data_summary": data_summary,
            "model": args.model_name_or_path,
            "embedding_model": None if args.skip_e5 else args.embedding_model_name_or_path,
            "pair_specs": pair_specs,
            "svd_rank": args.svd_rank,
            "query_q_tokens": args.query_q_tokens,
            "retrieval_blocks": args.retrieval_blocks,
            "retrieval_tokens": retrieved_tokens,
            "world_size": world_size,
            "sample_ids": sample_ids,
            "causal_contract": "retrieval sees history and the 64-token query prefix, never target tokens",
            "retrieved_blocks_are_restored_to_original_order": True,
            "contains_synthetic_text": False,
            "selection_uses_target": False,
            **summarize(all_rows, all_index_rows, args.seed),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
