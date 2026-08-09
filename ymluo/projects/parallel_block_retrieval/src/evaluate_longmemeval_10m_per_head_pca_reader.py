from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


METHODS = (
    "bm25_top31",
    "pca64_selected16_top31",
    "exact_qk_selected16_top31",
    "hybrid_bm25_pca_top31",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a 10M LongMemEval external-memory adaptation of per-head "
            "hierarchical PCA64 INT4 retrieval with a strict reader-token budget."
        )
    )
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--coarse_rows", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--memory_tokenizer_name_or_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--coarse_method", default="owner_metadata_block_bm25")
    parser.add_argument("--coarse_blocks", type=int, default=128)
    parser.add_argument("--final_blocks", type=int, default=31)
    parser.add_argument("--hybrid_bm25_blocks", type=int, default=16)
    parser.add_argument("--retrieval_token_budget", type=int, default=2000)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--layers", default="3,7,11,15,19,23,27,31")
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--calibration_blocks", type=int, default=256)
    parser.add_argument("--profile_batch_size", type=int, default=8)
    parser.add_argument("--block_model_max_tokens", type=int, default=128)
    parser.add_argument("--query_tail_tokens", type=int, default=8)
    parser.add_argument("--active_head_fraction", type=float, default=0.25)
    parser.add_argument("--selected_head_channels", type=int, default=16)
    parser.add_argument("--per_head_depth", type=int, default=8)
    parser.add_argument("--rrf_constant", type=float, default=20.0)
    parser.add_argument("--include_page_dates", action="store_true")
    parser.add_argument(
        "--page_order", choices=("retrieval", "chronological", "latest_first"),
        default="chronological",
    )
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def parse_layers(spec: str, layer_count: int) -> list[int]:
    layers = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not layers or min(layers) < 0 or max(layers) >= layer_count:
        raise ValueError(f"layers must be within [0, {layer_count})")
    return layers


def normalize(text: str) -> str:
    import re

    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def answer_contains(generation: str, reference: str) -> bool:
    answer = normalize(reference)
    return len(answer) >= 2 and answer in normalize(generation)


def token_f1(prediction: str, reference: str) -> float:
    from collections import Counter

    predicted = normalize(prediction).split()
    target = normalize(reference).split()
    if not predicted or not target:
        return float(predicted == target)
    overlap = sum((Counter(predicted) & Counter(target)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(target)
    return 2.0 * precision * recall / (precision + recall)


def is_refusal(text: str) -> bool:
    value = normalize(text)
    return any(
        phrase in value
        for phrase in (
            "not enough information",
            "insufficient information",
            "cannot determine",
            "can t determine",
            "not provided",
            "not specified",
            "unknown",
        )
    )


def format_date_minutes(value: int) -> str:
    if value < 0:
        return "unknown date"
    return (datetime(1, 1, 1) + timedelta(minutes=value)).strftime("%Y-%m-%d %H:%M")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def quantize_dequantize_int4(projected: torch.Tensor) -> torch.Tensor:
    """Match the report's symmetric per-vector INT4 quantizer."""
    scales = projected.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    codes = torch.round(projected.float() / scales).clamp(-7, 7)
    return (codes * scales).to(projected.dtype)


def selective_head_rrf(
    channel_scores: np.ndarray,
    *,
    output_depth: int,
    active_fraction: float,
    per_head_depth: int,
    rrf_constant: float,
) -> tuple[list[int], dict[str, float]]:
    """Fuse independent head rankings while retaining only selective channels."""
    if channel_scores.ndim != 2:
        raise ValueError("channel_scores must have shape [channels, candidates]")
    channels, candidates = channel_scores.shape
    if channels == 0 or candidates == 0:
        return [], {"active_channels": 0.0, "mean_margin": math.nan}
    standardized = channel_scores - channel_scores.mean(axis=1, keepdims=True)
    standardized /= channel_scores.std(axis=1, keepdims=True) + 1.0e-6
    top2 = np.partition(standardized, kth=max(candidates - 2, 0), axis=1)[:, -2:]
    top2.sort(axis=1)
    margins = top2[:, -1] - top2[:, 0] if candidates > 1 else top2[:, -1]
    active_count = max(1, min(channels, int(math.ceil(active_fraction * channels))))
    active = np.argpartition(margins, -active_count)[-active_count:]
    depth = max(1, min(per_head_depth, candidates))
    fused = np.zeros(candidates, dtype=np.float64)
    for channel in active:
        order = np.argpartition(standardized[channel], -depth)[-depth:]
        order = order[np.argsort(-standardized[channel, order], kind="stable")]
        weight = float(np.clip(1.0 + margins[channel], 0.25, 4.0))
        for rank, candidate in enumerate(order, start=1):
            fused[int(candidate)] += weight / (rrf_constant + rank)
    ranking = np.argsort(-fused, kind="stable")[: min(output_depth, candidates)].tolist()
    return ranking, {
        "active_channels": float(active_count),
        "mean_margin": float(margins[active].mean()),
        "max_fused_score": float(fused[ranking[0]]) if ranking else math.nan,
    }


def quota_union(primary: Iterable[int], secondary: Iterable[int], depth: int) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for source in (primary, secondary):
        for item in source:
            value = int(item)
            if value in seen:
                continue
            seen.add(value)
            output.append(value)
            if len(output) >= depth:
                return output
    return output


def segment_means(values: torch.Tensor, mask: torch.Tensor, segments: int) -> torch.Tensor:
    """Average valid token vectors into fixed ordered segments."""
    batch, _tokens, heads, width = values.shape
    output = torch.zeros(
        (batch, heads, segments, width), dtype=values.dtype, device=values.device
    )
    for row in range(batch):
        positions = torch.nonzero(mask[row], as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        for segment, part in enumerate(torch.tensor_split(positions, segments)):
            if part.numel():
                output[row, :, segment] = values[row, part].mean(dim=0)
    return output


@torch.inference_mode()
def profile_block_keys(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    device: torch.device,
    batch_size: int,
    max_tokens: int,
    segments: int,
    layer_indices: list[int],
) -> torch.Tensor:
    layers = model.model.layers
    kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
    batches: list[torch.Tensor] = []
    tokenizer.padding_side = "right"
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        result = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        layer_values: list[torch.Tensor] = []
        for layer_index in layer_indices:
            layer = layers[layer_index]
            hidden = layer.input_layernorm(result.hidden_states[layer_index])
            keys = layer.self_attn.k_proj(hidden).view(
                hidden.shape[0], hidden.shape[1], kv_heads, head_dim
            )
            layer_values.append(segment_means(keys, attention_mask.bool(), segments).cpu())
        # [batch, layers, kv_heads, segments, head_dim]
        batches.append(torch.stack(layer_values, dim=1))
        del result, input_ids, attention_mask, layer_values
    return torch.cat(batches, dim=0)


@torch.inference_mode()
def fit_pca_basis(
    raw_keys: torch.Tensor,
    *,
    projection_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    _blocks, layers, kv_heads, _segments, head_dim = raw_keys.shape
    if projection_dim > head_dim:
        raise ValueError("projection_dim exceeds attention head dimension")
    basis = torch.empty(
        (layers, kv_heads, head_dim, projection_dim), dtype=raw_keys.dtype, device=device
    )
    retained: list[float] = []
    for layer in range(layers):
        for head in range(kv_heads):
            values = raw_keys[:, layer, head].reshape(-1, head_dim).to(device=device, dtype=torch.float32)
            second_moment = values.T @ values / max(int(values.shape[0]), 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
            basis[layer, head] = eigenvectors[:, -projection_dim:].to(raw_keys.dtype)
            retained.append(
                float(
                    eigenvalues[-projection_dim:].clamp_min(0).sum()
                    / eigenvalues.clamp_min(0).sum().clamp_min(1.0e-12)
                )
            )
    return basis, retained


@torch.inference_mode()
def profile_query(
    model: Any,
    tokenizer: Any,
    question: str,
    basis: torch.Tensor,
    *,
    device: torch.device,
    tail_tokens: int,
    layer_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    text = f"Retrieve memory evidence needed to answer this question:\n{question}"
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    result = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    query_heads = int(model.config.num_attention_heads)
    kv_heads = int(model.config.num_key_value_heads)
    groups = query_heads // kv_heads
    head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // query_heads))
    valid = torch.nonzero(attention_mask[0], as_tuple=False).flatten()[-tail_tokens:]
    raw_layers: list[torch.Tensor] = []
    projected_layers: list[torch.Tensor] = []
    for local_layer, layer_index in enumerate(layer_indices):
        layer = model.model.layers[layer_index]
        hidden = layer.input_layernorm(result.hidden_states[layer_index])
        query = layer.self_attn.q_proj(hidden).view(1, hidden.shape[1], query_heads, head_dim)[
            0, valid
        ]
        query = query.permute(1, 0, 2)
        raw_layers.append(query)
        head_basis = basis[local_layer].repeat_interleave(groups, dim=0)
        projected_layers.append(torch.einsum("htd,hdr->htr", query, head_basis))
    return torch.stack(raw_layers, dim=0), torch.stack(projected_layers, dim=0)


@torch.inference_mode()
def score_candidates(
    raw_keys: torch.Tensor,
    raw_query: torch.Tensor,
    projected_query: torch.Tensor,
    basis: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    synchronize(device)
    started = time.perf_counter()
    keys = raw_keys.to(device)
    projected = torch.einsum("clhsd,lhdr->clhsr", keys, basis)
    dequantized = quantize_dequantize_int4(projected)
    query_heads = int(projected_query.shape[1])
    kv_heads = int(dequantized.shape[2])
    groups = query_heads // kv_heads
    layer_scores: list[torch.Tensor] = []
    for layer in range(int(dequantized.shape[1])):
        layer_keys = dequantized[:, layer].repeat_interleave(groups, dim=1).permute(1, 0, 2, 3)
        scores = torch.einsum("htr,hcsr->htcs", projected_query[layer], layer_keys)
        layer_scores.append(scores.amax(dim=(1, 3)))
    pca_output = torch.cat(layer_scores, dim=0).float().cpu().numpy()
    synchronize(device)
    pca_elapsed = time.perf_counter() - started

    exact_started = time.perf_counter()
    exact_layers: list[torch.Tensor] = []
    for layer in range(int(keys.shape[1])):
        layer_keys = keys[:, layer].repeat_interleave(groups, dim=1).permute(1, 0, 2, 3)
        scores = torch.einsum("htr,hcsr->htcs", raw_query[layer], layer_keys)
        exact_layers.append(scores.amax(dim=(1, 3)))
    exact_output = torch.cat(exact_layers, dim=0).float().cpu().numpy()
    synchronize(device)
    exact_elapsed = time.perf_counter() - exact_started
    del keys, projected, dequantized, layer_scores, exact_layers
    return pca_output, exact_output, pca_elapsed, exact_elapsed


def order_blocks(
    block_ids: list[int], block_dates: np.ndarray, page_order: str
) -> list[int]:
    if page_order == "retrieval":
        return block_ids
    rank = {block_id: index for index, block_id in enumerate(block_ids)}
    reverse = page_order == "latest_first"
    known = [block_id for block_id in block_ids if int(block_dates[block_id]) >= 0]
    unknown = [block_id for block_id in block_ids if int(block_dates[block_id]) < 0]
    known.sort(key=lambda block_id: (int(block_dates[block_id]), rank[block_id]), reverse=reverse)
    return known + unknown


def pack_context(
    block_ids: list[int],
    block_texts: dict[int, str],
    block_dates: np.ndarray,
    tokenizer: Any,
    *,
    token_budget: int,
    include_dates: bool,
) -> tuple[str, list[int], int]:
    pages: list[str] = []
    selected: list[int] = []
    token_count = 0
    for block_id in block_ids:
        date = (
            f"; date {format_date_minutes(int(block_dates[block_id]))}"
            if include_dates
            else ""
        )
        page = f"[Memory page {len(pages) + 1}{date}]\n{block_texts[block_id]}"
        separator = "\n\n" if pages else ""
        page_tokens = len(tokenizer.encode(separator + page, add_special_tokens=False))
        if token_count + page_tokens > token_budget:
            continue
        pages.append(page)
        selected.append(block_id)
        token_count += page_tokens
    return "\n\n".join(pages), selected, token_count


def reader_prompt(question: str, context: str) -> str:
    return (
        "Answer the question using only the memory pages below. Resolve updates and "
        "dates from the records. If the pages are insufficient, say that there is not "
        "enough information. Return a concise final answer.\n\n"
        f"Memory pages:\n{context}\n\nQuestion: {question}"
    )


@torch.inference_mode()
def generate_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, int, int, float]:
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    attention_mask = torch.ones_like(input_ids)
    synchronize(device)
    started = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    synchronize(device)
    elapsed = time.perf_counter() - started
    generated = output[0, input_ids.shape[1] :]
    return (
        tokenizer.decode(generated, skip_special_tokens=True).strip(),
        int(input_ids.shape[1]),
        int(generated.numel()),
        elapsed,
    )


def retrieval_metrics(query: dict[str, Any], block_ids: list[int], block_sessions: np.ndarray) -> dict[str, Any]:
    if bool(query["is_abstention"]):
        hard = set(map(int, query["hard_negative_block_ids"]))
        return {
            "exact_block_any": None,
            "all_evidence_sessions": None,
            "hard_negative_block_any": bool(hard.intersection(block_ids)),
        }
    positives = set(map(int, query["positive_block_ids"]))
    sessions = set(map(int, query["positive_session_rows"]))
    selected_sessions = {int(block_sessions[item]) for item in block_ids if int(block_sessions[item]) >= 0}
    return {
        "exact_block_any": bool(positives.intersection(block_ids)),
        "all_evidence_sessions": sessions.issubset(selected_sessions),
        "hard_negative_block_any": None,
    }


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    process_started_epoch = time.time()
    if args.projection_dim <= 0 or args.projection_dim % 2:
        raise ValueError("projection_dim must be a positive even value")
    if not 0 < args.active_head_fraction <= 1:
        raise ValueError("active_head_fraction must be in (0, 1]")
    if not 0 < args.hybrid_bm25_blocks < args.final_blocks <= args.coarse_blocks:
        raise ValueError("expected hybrid_bm25_blocks < final_blocks <= coarse_blocks")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    queries = read_jsonl(args.data_dir / "queries.jsonl")
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    coarse_rows = {
        str(row["question_id"]): row
        for row in read_jsonl(args.coarse_rows)
        if str(row["method"]) == args.coarse_method
    }
    missing = [str(row["question_id"]) for row in queries if str(row["question_id"]) not in coarse_rows]
    if missing:
        raise RuntimeError(f"missing coarse rows for {len(missing)} queries")
    if any(len(coarse_rows[str(row["question_id"])]["top_block_ids"]) < args.coarse_blocks for row in queries):
        raise RuntimeError("coarse rows do not contain the requested candidate depth")

    base_blocks = np.load(args.data_dir / "base_blocks.npy", mmap_mode="r")
    block_dates = np.asarray(
        np.load(args.data_dir / "base_block_date_minutes.npy", mmap_mode="r"), dtype=np.int64
    )
    block_sessions = np.asarray(
        np.load(args.data_dir / "base_block_session_rows.npy", mmap_mode="r"), dtype=np.int64
    )
    memory_tokenizer = AutoTokenizer.from_pretrained(
        args.memory_tokenizer_name_or_path, use_fast=True
    )
    reader_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if reader_tokenizer.pad_token_id is None:
        reader_tokenizer.pad_token = reader_tokenizer.eos_token

    rng = np.random.default_rng(args.seed)
    calibration_ids = rng.choice(
        len(base_blocks), size=min(args.calibration_blocks, len(base_blocks)), replace=False
    ).tolist()
    needed_ids = set(calibration_ids)
    for query in queries:
        needed_ids.update(
            map(
                int,
                coarse_rows[str(query["question_id"])]["top_block_ids"][: args.coarse_blocks],
            )
        )
    decode_started = time.perf_counter()
    block_texts = {
        block_id: memory_tokenizer.decode(base_blocks[block_id], skip_special_tokens=True)
        for block_id in sorted(needed_ids)
    }
    decode_seconds = time.perf_counter() - decode_started

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    synchronize(device)
    model_load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    layer_indices = parse_layers(args.layers, len(model.model.layers))
    channel_count = len(layer_indices) * int(model.config.num_attention_heads)
    if not 0 < args.selected_head_channels <= channel_count:
        raise ValueError("selected_head_channels exceeds selected layer/head channels")
    active_head_fraction = args.selected_head_channels / channel_count
    synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_started

    calibration_started = time.perf_counter()
    raw_calibration = profile_block_keys(
        model,
        reader_tokenizer,
        [block_texts[item] for item in calibration_ids],
        device=device,
        batch_size=args.profile_batch_size,
        max_tokens=args.block_model_max_tokens,
        segments=args.segments,
        layer_indices=layer_indices,
    )
    basis, retained_energy = fit_pca_basis(
        raw_calibration, projection_dim=args.projection_dim, device=device
    )
    synchronize(device)
    calibration_seconds = time.perf_counter() - calibration_started
    del raw_calibration

    results_path = args.output_dir / "rows.jsonl"
    existing = read_jsonl(results_path) if results_path.exists() else []
    completed = {(str(row["question_id"]), str(row["method"])) for row in existing}
    handle = results_path.open("a", encoding="utf-8")
    eval_started = time.perf_counter()
    try:
        for query_index, query in enumerate(queries):
            question_id = str(query["question_id"])
            if all((question_id, method) in completed for method in METHODS):
                continue
            coarse = coarse_rows[question_id]
            coarse_ids = list(map(int, coarse["top_block_ids"][: args.coarse_blocks]))

            synchronize(device)
            query_started = time.perf_counter()
            raw_query, projected_query = profile_query(
                model,
                reader_tokenizer,
                str(query["question"]),
                basis,
                device=device,
                tail_tokens=args.query_tail_tokens,
                layer_indices=layer_indices,
            )
            synchronize(device)
            query_profile_seconds = time.perf_counter() - query_started

            synchronize(device)
            candidate_started = time.perf_counter()
            candidate_keys = profile_block_keys(
                model,
                reader_tokenizer,
                [block_texts[item] for item in coarse_ids],
                device=device,
                batch_size=args.profile_batch_size,
                max_tokens=args.block_model_max_tokens,
                segments=args.segments,
                layer_indices=layer_indices,
            )
            synchronize(device)
            candidate_profile_seconds = time.perf_counter() - candidate_started
            pca_scores, exact_scores, pca_score_seconds, exact_score_seconds = score_candidates(
                candidate_keys, raw_query, projected_query, basis, device=device
            )
            local_pca_ranking, pca_fusion = selective_head_rrf(
                pca_scores,
                output_depth=args.coarse_blocks,
                active_fraction=active_head_fraction,
                per_head_depth=args.per_head_depth,
                rrf_constant=args.rrf_constant,
            )
            local_exact_ranking, exact_fusion = selective_head_rrf(
                exact_scores,
                output_depth=args.coarse_blocks,
                active_fraction=active_head_fraction,
                per_head_depth=args.per_head_depth,
                rrf_constant=args.rrf_constant,
            )
            pca_ranking = [coarse_ids[item] for item in local_pca_ranking]
            exact_ranking = [coarse_ids[item] for item in local_exact_ranking]
            method_rankings = {
                "bm25_top31": coarse_ids[: args.final_blocks],
                "pca64_selected16_top31": pca_ranking[: args.final_blocks],
                "exact_qk_selected16_top31": exact_ranking[: args.final_blocks],
                "hybrid_bm25_pca_top31": quota_union(
                    coarse_ids[: args.hybrid_bm25_blocks],
                    pca_ranking,
                    args.final_blocks,
                ),
            }
            pca_retrieval_seconds = (
                query_profile_seconds + candidate_profile_seconds + pca_score_seconds
            )
            exact_retrieval_seconds = (
                query_profile_seconds + candidate_profile_seconds + exact_score_seconds
            )
            method_order = list(METHODS)
            rotation = query_index % len(method_order)
            method_order = method_order[rotation:] + method_order[:rotation]

            for method in method_order:
                if (question_id, method) in completed:
                    continue
                ranked_ids = order_blocks(method_rankings[method], block_dates, args.page_order)
                context_started = time.perf_counter()
                context, packed_ids, context_tokens = pack_context(
                    ranked_ids,
                    block_texts,
                    block_dates,
                    reader_tokenizer,
                    token_budget=args.retrieval_token_budget,
                    include_dates=args.include_page_dates,
                )
                context_seconds = time.perf_counter() - context_started
                prompt = reader_prompt(str(query["question"]), context)
                generation, prompt_tokens, generated_tokens, generation_seconds = generate_answer(
                    model,
                    reader_tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
                reference = str(query["answer"])
                contains = answer_contains(generation, reference)
                refusal = is_refusal(generation)
                strict_correct = refusal if bool(query["is_abstention"]) else contains
                metrics = retrieval_metrics(query, packed_ids, block_sessions)
                if method in {"pca64_selected16_top31", "hybrid_bm25_pca_top31"}:
                    model_native_seconds = pca_retrieval_seconds
                    fusion = pca_fusion
                elif method == "exact_qk_selected16_top31":
                    model_native_seconds = exact_retrieval_seconds
                    fusion = exact_fusion
                else:
                    model_native_seconds = 0.0
                    fusion = {"active_channels": 0.0, "mean_margin": math.nan}
                row = {
                    "query_id": int(query["query_id"]),
                    "question_id": question_id,
                    "question_type": str(query["question_type"]),
                    "is_abstention": bool(query["is_abstention"]),
                    "question": str(query["question"]),
                    "reference": reference,
                    "method": method,
                    "memory_tokens": 10_000_000,
                    "coarse_blocks": args.coarse_blocks,
                    "coarse_query_seconds": float(coarse["query_seconds"]),
                    "query_profile_seconds": query_profile_seconds if model_native_seconds else 0.0,
                    "candidate_profile_seconds": candidate_profile_seconds if model_native_seconds else 0.0,
                    "pca_score_seconds": pca_score_seconds if method in {"pca64_selected16_top31", "hybrid_bm25_pca_top31"} else 0.0,
                    "exact_score_seconds": exact_score_seconds if method == "exact_qk_selected16_top31" else 0.0,
                    "model_native_retrieval_seconds": model_native_seconds,
                    "context_build_seconds": context_seconds,
                    "generation_seconds": generation_seconds,
                    "online_total_seconds": float(coarse["query_seconds"]) + model_native_seconds + context_seconds + generation_seconds,
                    "retrieval_token_budget": args.retrieval_token_budget,
                    "retrieved_tokens": context_tokens,
                    "selected_blocks_before_packing": len(method_rankings[method]),
                    "selected_blocks_after_packing": len(packed_ids),
                    "selected_block_ids": packed_ids,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "generation": generation,
                    "answer_contains": contains,
                    "token_f1": token_f1(generation, reference) if not bool(query["is_abstention"]) else None,
                    "refusal": refusal,
                    "strict_correct": strict_correct,
                    "selection_uses_answer": False,
                    "active_head_channels": fusion["active_channels"],
                    "mean_active_head_margin": fusion["mean_margin"],
                    **metrics,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                completed.add((question_id, method))
            del candidate_keys, raw_query, projected_query, pca_scores, exact_scores
            print(
                json.dumps(
                    {
                        "query": query_index + 1,
                        "queries": len(queries),
                        "question_id": question_id,
                        "candidate_profile_seconds": candidate_profile_seconds,
                        "pca_score_seconds": pca_score_seconds,
                        "exact_score_seconds": exact_score_seconds,
                    }
                ),
                flush=True,
            )
    finally:
        handle.close()

    eval_seconds = time.perf_counter() - eval_started
    process_finished_epoch = time.time()
    rows = read_jsonl(results_path)
    summary = {
        "source": "LongMemEval 10M per-head PCA64 INT4 hierarchical retrieval and 48-token reader",
        "protocol": {
            "memory_tokens": 10_000_000,
            "coarse_retriever": args.coarse_method,
            "coarse_blocks": args.coarse_blocks,
            "fine_retriever": "pre-RoPE PCA64 INT4 per-query-head selective RRF",
            "selected_layers": layer_indices,
            "selected_head_channels_per_query": args.selected_head_channels,
            "retrieval_token_budget": args.retrieval_token_budget,
            "max_new_tokens": args.max_new_tokens,
            "selection_uses_answer": False,
            "answer_used_only_for_posthoc_metrics": True,
            "physical_cache_claim": False,
            "adaptation_note": "PCA per-head is a fine reranker after 10M BM25 scope retrieval, not a direct 10M token-level sparse-attention cache.",
        },
        "queries": len(queries),
        "rows": len(rows),
        "model_name_or_path": args.model_name_or_path,
        "decode_needed_block_text_seconds": decode_seconds,
        "model_load_seconds": model_load_seconds,
        "pca_calibration_seconds": calibration_seconds,
        "pca_retained_energy_mean": mean(retained_energy),
        "pca_retained_energy_min": min(retained_energy),
        "evaluation_seconds": eval_seconds,
        "process_started_epoch": process_started_epoch,
        "process_finished_epoch": process_finished_epoch,
        "process_elapsed_seconds": process_finished_epoch - process_started_epoch,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
