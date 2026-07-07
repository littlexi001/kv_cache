from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_real_qwen_seq_ae_search_trace import (  # noqa: E402
    Config as TraceConfig,
    TraceDataset,
    collect_trace_dataset,
    make_case,
    resolve_dtype,
    select_dataset,
    split_indices,
    train_models,
)
from run_recent_plus_kv_native_smoke import (  # noqa: E402
    cache_from_legacy,
    cache_len,
    gather_cache,
    legacy_cache,
    page_token_indices,
    prefill,
    recent_indices,
    run_query_on_cache,
    selected_text,
    synchronize,
    tensor_indices,
    unique_ordered,
)


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    prompt_tokens: int
    page_tokens: int
    recent_tokens: int
    cases: tuple[str, ...]
    layers: str
    kv_heads: str
    max_query_tokens: int
    block_size: int
    latent_dim: int
    train_fraction: float
    batch_size: int
    ae_epochs: int
    search_epochs: int
    lr: float
    rare_recon_weight: float
    rare_token_fraction: float
    block_search_weight: float
    topk_score_weight: float
    score_topk: int
    score_temperature: float
    top_pages: tuple[int, ...]
    page_halo_pages: int
    exclude_sink_pages: int
    exclude_recent_from_latent: bool
    decode_steps: int
    include_prompt_rebuild: bool
    dtype: str
    attn_implementation: str
    device_map: str
    local_files_only: bool
    seed: int


@dataclass
class EndToEndRow:
    case: str
    method: str
    context_tokens: int
    active_kv_tokens: int
    query_tokens: int
    selected_pages: str
    expanded_pages: str
    page_recall: float
    span_page_recall: float
    latent_storage_ratio_vs_kv: float
    full_prefill_seconds: float
    method_prefill_seconds: float
    gather_seconds: float
    query_seconds: float
    decode_seconds: float
    total_online_seconds: float
    answer_nll: float
    answer_exact: bool
    speedup_vs_full_online: float
    generated: str


def parse_csv_tuple(value: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return value
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_int_tuple(value: str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    out = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not out:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return out


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Train a real-Qwen latent scorer, select raw pages, gather original K/V, "
            "and evaluate answer NLL/exact."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--prompt_tokens", type=int, default=1024)
    parser.add_argument("--page_tokens", type=int, default=128)
    parser.add_argument("--recent_tokens", type=int, default=128)
    parser.add_argument("--cases", type=parse_csv_tuple, default=("old_single", "two_old", "decoy_exact"))
    parser.add_argument("--layers", default="all")
    parser.add_argument("--kv_heads", default="all")
    parser.add_argument("--max_query_tokens", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--train_fraction", type=float, default=0.75)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ae_epochs", type=int, default=8)
    parser.add_argument("--search_epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--rare_recon_weight", type=float, default=0.2)
    parser.add_argument("--rare_token_fraction", type=float, default=0.01)
    parser.add_argument("--block_search_weight", type=float, default=0.0)
    parser.add_argument("--topk_score_weight", type=float, default=0.0)
    parser.add_argument("--score_topk", type=int, default=4)
    parser.add_argument("--score_temperature", type=float, default=2.0)
    parser.add_argument("--top_pages", type=parse_int_tuple, default=(1, 2))
    parser.add_argument("--page_halo_pages", type=int, default=0)
    parser.add_argument("--exclude_sink_pages", type=int, default=1)
    parser.add_argument("--exclude_recent_from_latent", type=str2bool, default=True)
    parser.add_argument("--decode_steps", type=int, default=16)
    parser.add_argument("--include_prompt_rebuild", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--local_files_only", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=2026070702)
    args = parser.parse_args()
    return Config(**vars(args))


def to_trace_config(config: Config) -> TraceConfig:
    return TraceConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        prompt_tokens=config.prompt_tokens,
        page_tokens=config.page_tokens,
        recent_tokens=config.recent_tokens,
        cases=config.cases,
        layers=config.layers,
        kv_heads=config.kv_heads,
        max_query_tokens=config.max_query_tokens,
        block_size=config.block_size,
        latent_dim=config.latent_dim,
        train_fraction=config.train_fraction,
        batch_size=config.batch_size,
        ae_epochs=config.ae_epochs,
        search_epochs=config.search_epochs,
        lr=config.lr,
        rare_recon_weight=config.rare_recon_weight,
        rare_token_fraction=config.rare_token_fraction,
        block_search_weight=config.block_search_weight,
        topk_score_weight=config.topk_score_weight,
        score_topk=config.score_topk,
        score_temperature=config.score_temperature,
        dtype=config.dtype,
        attn_implementation=config.attn_implementation,
        device_map=config.device_map,
        local_files_only=config.local_files_only,
        seed=config.seed,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_model_and_tokenizer(config: Config) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = resolve_dtype(config.dtype)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": config.attn_implementation,
        "local_files_only": config.local_files_only,
    }
    if dtype != "auto":
        load_kwargs["torch_dtype"] = dtype
    use_device_map = config.device_map and config.device_map.strip().lower() not in {"none", "null", "empty"}
    if use_device_map:
        load_kwargs["device_map"] = config.device_map

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        local_files_only=config.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    if not use_device_map and torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer


def input_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clone_cache(cache: Any) -> Any:
    cloned = tuple((key.clone(), value.clone()) for key, value in legacy_cache(cache))
    return cache_from_legacy(cloned)


def run_one_token_with_position(
    model: Any,
    token_ids: torch.Tensor,
    past_key_values: Any,
    position: int,
) -> tuple[Any, torch.Tensor]:
    device = token_ids.device
    attention_mask = torch.ones((1, cache_len(past_key_values) + 1), dtype=torch.long, device=device)
    position_ids = torch.tensor([[position]], dtype=torch.long, device=device)
    cache_position = torch.tensor([position], dtype=torch.long, device=device)
    try:
        out = model(
            input_ids=token_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )
    except TypeError:
        out = model(
            input_ids=token_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
    return out.past_key_values, out.logits[:, -1, :]


def greedy_decode_with_positions(
    model: Any,
    tokenizer: Any,
    logits: torch.Tensor,
    past_key_values: Any,
    steps: int,
    next_position: int,
) -> tuple[str, float]:
    if steps <= 0:
        return "", 0.0
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    generated: list[int] = []
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        cache = past_key_values
        for offset in range(steps):
            generated.append(int(next_token.item()))
            cache, current_logits = run_one_token_with_position(
                model,
                next_token.to(logits.device),
                cache,
                next_position + offset,
            )
            next_token = torch.argmax(current_logits, dim=-1, keepdim=True)
    synchronize()
    return tokenizer.decode(generated, skip_special_tokens=True), time.perf_counter() - start


def answer_nll_with_positions(
    model: Any,
    answer_ids: torch.Tensor,
    logits: torch.Tensor,
    past_key_values: Any,
    answer_position_start: int,
) -> float:
    total = 0.0
    count = 0
    current_logits = logits
    cache = past_key_values
    with torch.inference_mode():
        for idx in range(int(answer_ids.shape[1])):
            target = answer_ids[:, idx]
            log_probs = torch.log_softmax(current_logits, dim=-1)
            total -= float(log_probs.gather(1, target.view(1, 1)).item())
            count += 1
            cache, current_logits = run_one_token_with_position(
                model,
                target.view(1, 1),
                cache,
                answer_position_start + idx,
            )
    return total / max(1, count)


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    return (scores - scores.mean()) / (scores.std(unbiased=False) + 1e-6)


@torch.no_grad()
def latent_page_ranking(
    dataset: TraceDataset,
    case_name: str,
    ae: torch.nn.Module,
    searcher: torch.nn.Module,
    config: Config,
    device: torch.device,
) -> dict[str, Any]:
    sample_indices = [idx for idx, meta in enumerate(dataset.meta) if meta.case == case_name]
    if not sample_indices:
        raise ValueError(f"No trace samples for case {case_name}")

    seq_len = int(dataset.k.shape[1])
    blocks = seq_len // config.block_size
    page_blocks = config.page_tokens // config.block_size
    pages = math.ceil(blocks / page_blocks)
    block_score_sum = torch.zeros(blocks, dtype=torch.float32)

    for idx in sample_indices:
        k = dataset.k[idx : idx + 1].to(device)
        v = dataset.v[idx : idx + 1].to(device)
        q = dataset.q[idx : idx + 1].to(device)
        z = ae.encode(k, v)
        scores = searcher(q, z)[0].detach().cpu()
        per_block = normalize_scores(scores.max(dim=0).values)
        block_score_sum += per_block

    block_scores = block_score_sum / float(len(sample_indices))
    page_scores: list[float] = []
    for page in range(pages):
        start = page * page_blocks
        end = min(blocks, start + page_blocks)
        page_scores.append(float(block_scores[start:end].max().item()))
    ranked_pages = sorted(range(pages), key=lambda page: (-page_scores[page], page))
    return {
        "case": case_name,
        "sample_count": len(sample_indices),
        "block_scores": [float(x) for x in block_scores.tolist()],
        "page_scores": page_scores,
        "ranked_pages": ranked_pages,
    }


def page_recall(selected_pages: list[int], evidence_pages: tuple[int, ...]) -> float:
    if not evidence_pages:
        return 1.0
    selected = set(selected_pages)
    evidence = set(evidence_pages)
    return len(selected & evidence) / float(len(evidence))


def expand_pages(selected_pages: list[int], context_len: int, page_tokens: int, halo_pages: int) -> list[int]:
    total_pages = math.ceil(context_len / page_tokens)
    expanded_pages: list[int] = []
    for page in sorted(selected_pages):
        start = max(0, page - halo_pages)
        end = min(total_pages, page + halo_pages + 1)
        expanded_pages.extend(range(start, end))
    return unique_ordered(expanded_pages)


def candidate_remote_pages(ranked_pages: list[int], context_len: int, config: Config) -> list[int]:
    pages = context_len // config.page_tokens
    excluded: set[int] = set(range(max(0, min(config.exclude_sink_pages, pages))))
    if config.exclude_recent_from_latent and config.recent_tokens > 0:
        recent_start = max(0, context_len - config.recent_tokens)
        recent_start_page = recent_start // config.page_tokens
        excluded.update(range(recent_start_page, pages))
    candidates = [page for page in ranked_pages if page not in excluded]
    return candidates or ranked_pages


def selected_indices_for_pages(
    selected_pages: list[int],
    context_len: int,
    page_tokens: int,
    recent_tokens: int,
    halo_pages: int,
) -> list[int]:
    expanded_pages = expand_pages(selected_pages, context_len, page_tokens, halo_pages)
    return unique_ordered(
        page_token_indices(expanded_pages, context_len, page_tokens)
        + recent_indices(context_len, recent_tokens)
    )


def add_cache_eval_row(
    rows: list[EndToEndRow],
    *,
    case: Any,
    method: str,
    model: Any,
    tokenizer: Any,
    query_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    cache: Any,
    active_kv_tokens: int,
    selected_pages: list[int] | str,
    expanded_pages: list[int] | str,
    recall: float,
    span_recall: float,
    latent_storage_ratio: float,
    position_start: int,
    gather_seconds: float,
    decode_steps: int,
    full_prefill_seconds: float,
    full_online_seconds: float,
) -> None:
    q_cache, logits, query_seconds = run_query_on_cache(
        model,
        query_ids,
        cache,
        position_start=position_start,
        past_len=active_kv_tokens,
    )
    answer_pos_start = position_start + int(query_ids.shape[1])
    decode_cache = clone_cache(q_cache)
    nll = answer_nll_with_positions(model, answer_ids, logits, q_cache, answer_pos_start)
    generated, decode_seconds = greedy_decode_with_positions(
        model,
        tokenizer,
        logits,
        decode_cache,
        decode_steps,
        answer_pos_start,
    )
    total = gather_seconds + query_seconds + decode_seconds
    rows.append(
        EndToEndRow(
            case=case.name,
            method=method,
            context_tokens=len(case.context_ids),
            active_kv_tokens=active_kv_tokens,
            query_tokens=int(query_ids.shape[1]),
            selected_pages=json.dumps(selected_pages, ensure_ascii=False)
            if not isinstance(selected_pages, str)
            else selected_pages,
            expanded_pages=json.dumps(expanded_pages, ensure_ascii=False)
            if not isinstance(expanded_pages, str)
            else expanded_pages,
            page_recall=recall,
            span_page_recall=span_recall,
            latent_storage_ratio_vs_kv=latent_storage_ratio,
            full_prefill_seconds=full_prefill_seconds,
            method_prefill_seconds=0.0,
            gather_seconds=gather_seconds,
            query_seconds=query_seconds,
            decode_seconds=decode_seconds,
            total_online_seconds=total,
            answer_nll=nll,
            answer_exact=case.answer in generated,
            speedup_vs_full_online=full_online_seconds / total if total > 0 else 0.0,
            generated=generated.replace("\n", " ")[:240],
        )
    )


def add_prompt_eval_row(
    rows: list[EndToEndRow],
    *,
    case: Any,
    method: str,
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    query_tokens: int,
    answer_ids: torch.Tensor,
    selected_pages: list[int],
    expanded_pages: list[int],
    recall: float,
    span_recall: float,
    latent_storage_ratio: float,
    decode_steps: int,
    full_prefill_seconds: float,
    full_online_seconds: float,
) -> None:
    prompt_cache, prompt_logits, prompt_prefill = prefill(model, prompt_ids)
    decode_cache = clone_cache(prompt_cache)
    nll = answer_nll_with_positions(
        model,
        answer_ids,
        prompt_logits,
        prompt_cache,
        int(prompt_ids.shape[1]),
    )
    generated, decode_seconds = greedy_decode_with_positions(
        model,
        tokenizer,
        prompt_logits,
        decode_cache,
        decode_steps,
        int(prompt_ids.shape[1]),
    )
    total = prompt_prefill + decode_seconds
    rows.append(
        EndToEndRow(
            case=case.name,
            method=method,
            context_tokens=len(case.context_ids),
            active_kv_tokens=int(prompt_ids.shape[1]),
            query_tokens=query_tokens,
            selected_pages=json.dumps(selected_pages, ensure_ascii=False),
            expanded_pages=json.dumps(expanded_pages, ensure_ascii=False),
            page_recall=recall,
            span_page_recall=span_recall,
            latent_storage_ratio_vs_kv=latent_storage_ratio,
            full_prefill_seconds=full_prefill_seconds,
            method_prefill_seconds=prompt_prefill,
            gather_seconds=0.0,
            query_seconds=0.0,
            decode_seconds=decode_seconds,
            total_online_seconds=total,
            answer_nll=nll,
            answer_exact=case.answer in generated,
            speedup_vs_full_online=full_online_seconds / total if total > 0 else 0.0,
            generated=generated.replace("\n", " ")[:240],
        )
    )


def aggregate_rows(rows: list[EndToEndRow]) -> list[dict[str, Any]]:
    by_method: dict[str, list[EndToEndRow]] = {}
    for row in rows:
        by_method.setdefault(row.method, []).append(row)
    out: list[dict[str, Any]] = []
    for method, method_rows in sorted(by_method.items()):
        out.append(
            {
                "method": method,
                "cases": len(method_rows),
                "mean_active_kv_tokens": sum(row.active_kv_tokens for row in method_rows) / len(method_rows),
                "mean_page_recall": sum(row.page_recall for row in method_rows) / len(method_rows),
                "mean_span_page_recall": sum(row.span_page_recall for row in method_rows) / len(method_rows),
                "mean_answer_nll": sum(row.answer_nll for row in method_rows) / len(method_rows),
                "exact_rate": sum(1.0 for row in method_rows if row.answer_exact) / len(method_rows),
                "mean_total_online_seconds": sum(row.total_online_seconds for row in method_rows) / len(method_rows),
            }
        )
    return out


def main() -> None:
    config = parse_args()
    if config.prompt_tokens % config.block_size != 0:
        raise ValueError("--prompt_tokens must be divisible by --block_size")
    if config.page_tokens % config.block_size != 0:
        raise ValueError("--page_tokens must be divisible by --block_size")
    if config.prompt_tokens % config.page_tokens != 0:
        raise ValueError("--prompt_tokens must be divisible by --page_tokens")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(config)
    device = input_device(model)

    start_wall = time.perf_counter()
    trace_config = to_trace_config(config)
    dataset = collect_trace_dataset(model, tokenizer, trace_config)
    train_indices, test_indices = split_indices(dataset.k.shape[0], config.train_fraction, config.seed)
    train = select_dataset(dataset, train_indices)
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae, searcher, train_stats = train_models(train, trace_config, train_device)

    head_dim = int(dataset.k.shape[-1])
    blocks = int(dataset.k.shape[1] // config.block_size)
    latent_storage_ratio = (blocks * config.latent_dim) / float(config.prompt_tokens * 2 * head_dim)
    rankings = {
        case_name: latent_page_ranking(dataset, case_name, ae, searcher, config, train_device)
        for case_name in config.cases
    }

    rows: list[EndToEndRow] = []
    cases_meta: list[dict[str, Any]] = []
    page_selection_meta: list[dict[str, Any]] = []

    for case_name in config.cases:
        case = make_case(tokenizer, case_name, config.prompt_tokens, config.page_tokens)
        ranking = rankings[case_name]
        context_tensor = torch.tensor(case.context_ids, dtype=torch.long, device=device).view(1, -1)
        query_ids = tokenizer(case.query, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        answer_ids = tokenizer(case.answer, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

        full_cache, _, full_prefill = prefill(model, context_tensor)
        full_q_cache, full_logits, full_query_seconds = run_query_on_cache(
            model,
            query_ids,
            full_cache,
            position_start=len(case.context_ids),
            past_len=len(case.context_ids),
        )
        full_answer_pos = len(case.context_ids) + int(query_ids.shape[1])
        full_decode_cache = clone_cache(full_q_cache)
        full_nll = answer_nll_with_positions(model, answer_ids, full_logits, full_q_cache, full_answer_pos)
        full_generated, full_decode = greedy_decode_with_positions(
            model,
            tokenizer,
            full_logits,
            full_decode_cache,
            config.decode_steps,
            full_answer_pos,
        )
        full_online = full_query_seconds + full_decode
        rows.append(
            EndToEndRow(
                case=case.name,
                method="full_kv_cache",
                context_tokens=len(case.context_ids),
                active_kv_tokens=len(case.context_ids),
                query_tokens=int(query_ids.shape[1]),
                selected_pages="all",
                expanded_pages="all",
                page_recall=1.0,
                span_page_recall=1.0,
                latent_storage_ratio_vs_kv=1.0,
                full_prefill_seconds=full_prefill,
                method_prefill_seconds=0.0,
                gather_seconds=0.0,
                query_seconds=full_query_seconds,
                decode_seconds=full_decode,
                total_online_seconds=full_online,
                answer_nll=full_nll,
                answer_exact=case.answer in full_generated,
                speedup_vs_full_online=1.0,
                generated=full_generated.replace("\n", " ")[:240],
            )
        )

        recent = recent_indices(len(case.context_ids), config.recent_tokens)
        recent_idx = tensor_indices(recent, device)
        synchronize()
        gather_start = time.perf_counter()
        recent_cache = cache_from_legacy(gather_cache(full_cache, recent_idx))
        synchronize()
        recent_gather = time.perf_counter() - gather_start
        add_cache_eval_row(
            rows,
            case=case,
            method="kv_native_recent_only_absolute_pos",
            model=model,
            tokenizer=tokenizer,
            query_ids=query_ids,
            answer_ids=answer_ids,
            cache=recent_cache,
            active_kv_tokens=int(recent_idx.numel()),
            selected_pages=[],
            expanded_pages=[],
            recall=page_recall([], case.evidence_pages),
            span_recall=page_recall([], case.evidence_pages),
            latent_storage_ratio=0.0,
            position_start=len(case.context_ids),
            gather_seconds=recent_gather,
            decode_steps=config.decode_steps,
            full_prefill_seconds=full_prefill,
            full_online_seconds=full_online,
        )

        oracle_pages = sorted(case.evidence_pages)
        oracle_indices = selected_indices_for_pages(
            oracle_pages,
            len(case.context_ids),
            config.page_tokens,
            config.recent_tokens,
            config.page_halo_pages,
        )
        oracle_expanded_pages = expand_pages(
            oracle_pages,
            len(case.context_ids),
            config.page_tokens,
            config.page_halo_pages,
        )
        oracle_idx = tensor_indices(oracle_indices, device)
        synchronize()
        gather_start = time.perf_counter()
        oracle_cache = cache_from_legacy(gather_cache(full_cache, oracle_idx))
        synchronize()
        oracle_gather = time.perf_counter() - gather_start
        add_cache_eval_row(
            rows,
            case=case,
            method="kv_native_oracle_pages_plus_recent_absolute_pos",
            model=model,
            tokenizer=tokenizer,
            query_ids=query_ids,
            answer_ids=answer_ids,
            cache=oracle_cache,
            active_kv_tokens=int(oracle_idx.numel()),
            selected_pages=oracle_pages,
            expanded_pages=oracle_expanded_pages,
            recall=page_recall(oracle_pages, case.evidence_pages),
            span_recall=page_recall(oracle_expanded_pages, case.evidence_pages),
            latent_storage_ratio=latent_storage_ratio,
            position_start=len(case.context_ids),
            gather_seconds=oracle_gather,
            decode_steps=config.decode_steps,
            full_prefill_seconds=full_prefill,
            full_online_seconds=full_online,
        )

        compact_oracle_cache = cache_from_legacy(gather_cache(full_cache, oracle_idx))
        add_cache_eval_row(
            rows,
            case=case,
            method="kv_native_oracle_pages_plus_recent_compact_pos",
            model=model,
            tokenizer=tokenizer,
            query_ids=query_ids,
            answer_ids=answer_ids,
            cache=compact_oracle_cache,
            active_kv_tokens=int(oracle_idx.numel()),
            selected_pages=oracle_pages,
            expanded_pages=oracle_expanded_pages,
            recall=page_recall(oracle_pages, case.evidence_pages),
            span_recall=page_recall(oracle_expanded_pages, case.evidence_pages),
            latent_storage_ratio=latent_storage_ratio,
            position_start=int(oracle_idx.numel()),
            gather_seconds=0.0,
            decode_steps=config.decode_steps,
            full_prefill_seconds=full_prefill,
            full_online_seconds=full_online,
        )

        if config.include_prompt_rebuild:
            oracle_prompt_text = selected_text(tokenizer, case.context_ids, oracle_indices) + "\n\n" + case.query
            oracle_prompt_ids = tokenizer(
                oracle_prompt_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(device)
            add_prompt_eval_row(
                rows,
                case=case,
                method="prompt_rebuild_oracle_pages_text",
                model=model,
                tokenizer=tokenizer,
                prompt_ids=oracle_prompt_ids,
                query_tokens=int(query_ids.shape[1]),
                answer_ids=answer_ids,
                selected_pages=oracle_pages,
                expanded_pages=oracle_expanded_pages,
                recall=page_recall(oracle_pages, case.evidence_pages),
                span_recall=page_recall(oracle_expanded_pages, case.evidence_pages),
                latent_storage_ratio=latent_storage_ratio,
                decode_steps=config.decode_steps,
                full_prefill_seconds=full_prefill,
                full_online_seconds=full_online,
            )

        candidate_pages = candidate_remote_pages(
            list(ranking["ranked_pages"]),
            len(case.context_ids),
            config,
        )
        for top_pages in config.top_pages:
            selected_pages = sorted(candidate_pages[: min(top_pages, len(candidate_pages))])
            latent_indices = selected_indices_for_pages(
                selected_pages,
                len(case.context_ids),
                config.page_tokens,
                config.recent_tokens,
                config.page_halo_pages,
            )
            latent_idx = tensor_indices(latent_indices, device)
            latent_expanded_pages = expand_pages(
                selected_pages,
                len(case.context_ids),
                config.page_tokens,
                config.page_halo_pages,
            )
            synchronize()
            gather_start = time.perf_counter()
            latent_cache = cache_from_legacy(gather_cache(full_cache, latent_idx))
            synchronize()
            latent_gather = time.perf_counter() - gather_start
            method = f"kv_native_latent_remote_top{top_pages}_pages_plus_recent_absolute_pos"
            add_cache_eval_row(
                rows,
                case=case,
                method=method,
                model=model,
                tokenizer=tokenizer,
                query_ids=query_ids,
                answer_ids=answer_ids,
                cache=latent_cache,
                active_kv_tokens=int(latent_idx.numel()),
                selected_pages=selected_pages,
                expanded_pages=latent_expanded_pages,
                recall=page_recall(selected_pages, case.evidence_pages),
                span_recall=page_recall(latent_expanded_pages, case.evidence_pages),
                latent_storage_ratio=latent_storage_ratio,
                position_start=len(case.context_ids),
                gather_seconds=latent_gather,
                decode_steps=config.decode_steps,
                full_prefill_seconds=full_prefill,
                full_online_seconds=full_online,
                )

            if config.include_prompt_rebuild:
                prompt_text = selected_text(tokenizer, case.context_ids, latent_indices) + "\n\n" + case.query
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                add_prompt_eval_row(
                    rows,
                    case=case,
                    method=f"prompt_rebuild_latent_remote_top{top_pages}_pages_text",
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    query_tokens=int(query_ids.shape[1]),
                    answer_ids=answer_ids,
                    selected_pages=selected_pages,
                    expanded_pages=latent_expanded_pages,
                    recall=page_recall(selected_pages, case.evidence_pages),
                    span_recall=page_recall(latent_expanded_pages, case.evidence_pages),
                    latent_storage_ratio=latent_storage_ratio,
                    decode_steps=config.decode_steps,
                    full_prefill_seconds=full_prefill,
                    full_online_seconds=full_online,
                )

            page_selection_meta.append(
                {
                    "case": case.name,
                    "top_pages": top_pages,
                    "selected_pages": selected_pages,
                    "expanded_pages": latent_expanded_pages,
                    "evidence_pages": list(case.evidence_pages),
                    "page_recall": page_recall(selected_pages, case.evidence_pages),
                    "span_page_recall": page_recall(latent_expanded_pages, case.evidence_pages),
                    "candidate_remote_pages": candidate_pages,
                    "ranked_pages": ranking["ranked_pages"],
                    "page_scores": ranking["page_scores"],
                }
            )

        cases_meta.append(
            {
                "case": case.name,
                "answer": case.answer,
                "evidence_pages": list(case.evidence_pages),
                "context_tokens": len(case.context_ids),
                "query": case.query,
            }
        )

    row_dicts = [asdict(row) for row in rows]
    method_summary = aggregate_rows(rows)
    wall_seconds = time.perf_counter() - start_wall

    write_csv(output_dir / "results.csv", row_dicts)
    write_csv(output_dir / "method_summary.csv", method_summary)
    (output_dir / "page_rankings.json").write_text(
        json.dumps(rankings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "trace_samples": int(dataset.k.shape[0]),
                "train_samples": int(train.k.shape[0]),
                "test_samples": int(len(test_indices)),
                "latent_storage_ratio_vs_kv": latent_storage_ratio,
                "train_stats": asdict(train_stats),
                "wall_seconds": wall_seconds,
                "cases": cases_meta,
                "page_selection": page_selection_meta,
                "method_summary": method_summary,
                "rows": row_dicts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("method,cases,mean_active_kv,mean_page_recall,mean_span_page_recall,mean_nll,exact_rate,mean_online")
    for row in method_summary:
        print(
            f"{row['method']},{row['cases']},{row['mean_active_kv_tokens']:.1f},"
            f"{row['mean_page_recall']:.4f},{row['mean_span_page_recall']:.4f},"
            f"{row['mean_answer_nll']:.4f},"
            f"{row['exact_rate']:.4f},{row['mean_total_online_seconds']:.4f}"
        )
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
