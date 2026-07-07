from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_latent_scorer_raw_span_smoke import (  # noqa: E402
    EndToEndRow,
    add_prompt_eval_row,
    aggregate_rows,
    candidate_remote_pages,
    clone_cache,
    expand_pages,
    input_device,
    load_model_and_tokenizer,
    page_recall,
    selected_indices_for_pages,
)
from run_real_qwen_seq_ae_search_trace import (  # noqa: E402
    Config as TraceConfig,
    DECOY_PAGE_PLAN,
    TraceDataset,
    collect_trace_dataset,
    distinct_pages,
    make_case,
    resolve_dtype,
    select_dataset,
    split_case_family,
    train_models,
)
from run_recent_plus_kv_native_smoke import (  # noqa: E402
    prefill,
    run_query_on_cache,
    selected_text,
)
from run_seq_autoencoder_search_smoke import LatentSearcher  # noqa: E402


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    prompt_tokens: int
    page_tokens: int
    recent_tokens: int
    cases: tuple[str, ...]
    case_families: tuple[str, ...]
    train_variant_suffixes: tuple[str, ...]
    eval_variant_suffixes: tuple[str, ...]
    train_cases: tuple[str, ...]
    eval_cases: tuple[str, ...]
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
    supervised_epochs: int
    supervised_lr: float
    hard_negative_weight: float
    hard_negative_margin: float
    page_pool_temperature: float
    init_from_attention_searcher: bool
    top_pages: tuple[int, ...]
    page_halo_pages: int
    exclude_sink_pages: int
    exclude_recent_from_latent: bool
    decode_steps: int
    include_prompt_rebuild: bool
    include_adaptive_case_top: bool
    include_adaptive_family_budget: bool
    include_evidence_composer: bool
    composer_max_tokens: int
    composer_extra_halo_pages: int
    eval_rankers: tuple[str, ...]
    dtype: str
    attn_implementation: str
    device_map: str
    local_files_only: bool
    seed: int


@dataclass
class SupervisedTrainStats:
    final_loss: float
    final_ce_loss: float
    final_hard_loss: float
    train_seconds: float


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
            "Train a supervised latent page ranker with evidence-page positives and "
            "sink/recent/decoy hard negatives."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--prompt_tokens", type=int, default=1024)
    parser.add_argument("--page_tokens", type=int, default=128)
    parser.add_argument("--recent_tokens", type=int, default=128)
    parser.add_argument("--cases", type=parse_csv_tuple, default=("old_single", "two_old", "decoy_exact"))
    parser.add_argument("--case_families", type=parse_csv_tuple, default=("old_single", "two_old", "decoy_exact"))
    parser.add_argument("--train_variant_suffixes", type=parse_csv_tuple, default=())
    parser.add_argument("--eval_variant_suffixes", type=parse_csv_tuple, default=())
    parser.add_argument("--train_cases", type=parse_csv_tuple, default=())
    parser.add_argument("--eval_cases", type=parse_csv_tuple, default=())
    parser.add_argument("--layers", default="all")
    parser.add_argument("--kv_heads", default="all")
    parser.add_argument("--max_query_tokens", type=int, default=16)
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
    parser.add_argument("--supervised_epochs", type=int, default=80)
    parser.add_argument("--supervised_lr", type=float, default=1e-3)
    parser.add_argument("--hard_negative_weight", type=float, default=0.5)
    parser.add_argument("--hard_negative_margin", type=float, default=1.0)
    parser.add_argument("--page_pool_temperature", type=float, default=0.2)
    parser.add_argument("--init_from_attention_searcher", type=str2bool, default=True)
    parser.add_argument("--top_pages", type=parse_int_tuple, default=(1, 2))
    parser.add_argument("--page_halo_pages", type=int, default=1)
    parser.add_argument("--exclude_sink_pages", type=int, default=1)
    parser.add_argument("--exclude_recent_from_latent", type=str2bool, default=True)
    parser.add_argument("--decode_steps", type=int, default=16)
    parser.add_argument("--include_prompt_rebuild", type=str2bool, default=True)
    parser.add_argument("--include_adaptive_case_top", type=str2bool, default=False)
    parser.add_argument("--include_adaptive_family_budget", type=str2bool, default=False)
    parser.add_argument("--include_evidence_composer", type=str2bool, default=False)
    parser.add_argument("--composer_max_tokens", type=int, default=96)
    parser.add_argument("--composer_extra_halo_pages", type=int, default=1)
    parser.add_argument("--eval_rankers", type=parse_csv_tuple, default=("attention", "supervised"))
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--local_files_only", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=2026070703)
    args = parser.parse_args()
    return Config(**vars(args))


def unique_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def variant_case_names(families: tuple[str, ...], suffixes: tuple[str, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for suffix in suffixes:
        suffix = suffix.strip()
        for family in families:
            names.append(family if not suffix else f"{family}_{suffix}")
    return tuple(names)


def train_case_names(config: Config) -> tuple[str, ...]:
    if config.train_cases:
        return config.train_cases
    if config.train_variant_suffixes:
        return variant_case_names(config.case_families, config.train_variant_suffixes)
    return config.cases


def eval_case_names(config: Config) -> tuple[str, ...]:
    if config.eval_cases:
        return config.eval_cases
    if config.eval_variant_suffixes:
        return variant_case_names(config.case_families, config.eval_variant_suffixes)
    return config.cases


def all_case_names(config: Config) -> tuple[str, ...]:
    return unique_tuple(train_case_names(config) + eval_case_names(config))


def to_trace_config(config: Config) -> TraceConfig:
    return TraceConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        prompt_tokens=config.prompt_tokens,
        page_tokens=config.page_tokens,
        recent_tokens=config.recent_tokens,
        cases=all_case_names(config),
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


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    return (scores - scores.mean()) / (scores.std(unbiased=False) + 1e-6)


def page_scores_from_block_scores(
    block_scores: torch.Tensor,
    page_blocks: int,
    temperature: float,
) -> torch.Tensor:
    batch, _, blocks = block_scores.shape
    pages = math.ceil(blocks / page_blocks)
    pooled: list[torch.Tensor] = []
    for page in range(pages):
        start = page * page_blocks
        end = min(blocks, start + page_blocks)
        values = block_scores[:, :, start:end].reshape(batch, -1)
        if temperature <= 0:
            pooled.append(values.max(dim=1).values)
        else:
            pooled.append(torch.logsumexp(values / temperature, dim=1) * temperature)
    return torch.stack(pooled, dim=-1)


def hard_negative_pages(case_name: str, evidence_pages: tuple[int, ...], config: Config) -> list[int]:
    pages = config.prompt_tokens // config.page_tokens
    hard: set[int] = set(range(max(0, min(config.exclude_sink_pages, pages))))
    if config.recent_tokens > 0:
        recent_start = max(0, config.prompt_tokens - config.recent_tokens)
        hard.update(range(recent_start // config.page_tokens, pages))
    family, variant = split_case_family(case_name)
    if family in {"decoy_exact", "current_conflict"}:
        page_old, _ = distinct_pages(list(DECOY_PAGE_PLAN[variant % len(DECOY_PAGE_PLAN)]), pages)
        hard.add(page_old)
    hard.difference_update(evidence_pages)
    return sorted(page for page in hard if 0 <= page < pages)


def indices_for_cases(dataset: TraceDataset, cases: tuple[str, ...]) -> list[int]:
    case_set = set(cases)
    return [idx for idx, meta in enumerate(dataset.meta) if meta.case in case_set]


def adaptive_top_pages_for_case(case_name: str) -> int:
    family, _ = split_case_family(case_name)
    return 2 if family in {"two_old", "multi_hop_bridge"} else 1


def adaptive_halo_pages_for_case(case_name: str) -> int:
    family, _ = split_case_family(case_name)
    return 1 if family in {"two_old", "multi_hop_bridge"} else 2


IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+){1,}\b")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def extract_identifiers(text: str) -> set[str]:
    return set(IDENTIFIER_RE.findall(text))


def sentence_is_useful(sentence: str, identifiers: set[str]) -> bool:
    lower = sentence.lower()
    if any(identifier in sentence for identifier in identifiers):
        return True
    keywords = (
        "verified",
        "current",
        "checksum",
        "obsolete",
        "superseded",
        "stale",
        "bridge",
        "locator",
        "intermediate",
        "marker",
        "final answer",
        "final response",
        "current revision",
        "verified update",
        "maps to",
        "points to",
        "returns",
        "resolves to",
    )
    return any(keyword in lower for keyword in keywords) and "neutral background" not in lower


def compose_evidence_text(
    tokenizer: Any,
    context_ids: list[int],
    indices: list[int],
    query: str,
    max_tokens: int,
) -> str:
    selected = selected_text(tokenizer, context_ids, indices)
    sentences = split_sentences(selected)
    identifiers = extract_identifiers(query)
    kept: list[str] = []

    for _ in range(2):
        changed = False
        for sentence in sentences:
            if sentence in kept:
                continue
            if sentence_is_useful(sentence, identifiers):
                kept.append(sentence)
                before = len(identifiers)
                identifiers.update(extract_identifiers(sentence))
                changed = changed or len(identifiers) > before
        if not changed:
            break

    if not kept:
        kept = [sentence for sentence in sentences if "neutral background" not in sentence.lower()][:4]
    if not kept:
        kept = sentences[:4]

    def priority(sentence: str) -> tuple[int, int]:
        lower = sentence.lower()
        if "obsolete" in lower or "superseded" in lower or "stale" in lower:
            return (2, len(sentence))
        if "current" in lower or "verified" in lower:
            return (3, len(sentence))
        if "final answer" in lower or "final response" in lower or "maps to" in lower:
            return (1, len(sentence))
        if "bridge" in lower or "intermediate" in lower or "marker" in lower:
            return (0, len(sentence))
        return (1, len(sentence))

    ordered = sorted(dict.fromkeys(kept), key=priority)
    text = " ".join(ordered)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if max_tokens > 0 and len(token_ids) > max_tokens:
        text = tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)
    return text.strip()


def multi_positive_ce(scores: torch.Tensor, positive_pages: tuple[int, ...]) -> torch.Tensor:
    device = scores.device
    positives = torch.tensor(list(positive_pages), dtype=torch.long, device=device)
    all_pages = torch.arange(scores.shape[-1], dtype=torch.long, device=device)
    positive_scores = scores.index_select(0, positives)
    candidate_scores = scores.index_select(0, all_pages)
    return torch.logsumexp(candidate_scores, dim=0) - torch.logsumexp(positive_scores, dim=0)


def row_hard_negative_loss(
    scores: torch.Tensor,
    positive_pages: tuple[int, ...],
    hard_pages: list[int],
    margin: float,
) -> torch.Tensor:
    if not hard_pages:
        return torch.zeros((), dtype=scores.dtype, device=scores.device)
    device = scores.device
    positives = torch.tensor(list(positive_pages), dtype=torch.long, device=device)
    hard = torch.tensor(hard_pages, dtype=torch.long, device=device)
    weakest_positive = scores.index_select(0, positives).min()
    strongest_hard = scores.index_select(0, hard).max()
    return F.relu(margin + strongest_hard - weakest_positive)


def train_supervised_page_ranker(
    train: TraceDataset,
    ae: torch.nn.Module,
    init_searcher: LatentSearcher,
    case_specs: dict[str, Any],
    config: Config,
    device: torch.device,
) -> tuple[LatentSearcher, SupervisedTrainStats]:
    if config.init_from_attention_searcher:
        searcher = copy.deepcopy(init_searcher).to(device)
    else:
        searcher = LatentSearcher(train.k.shape[-1], config.latent_dim).to(device)
    ae.eval()
    optimizer = torch.optim.AdamW(searcher.parameters(), lr=config.supervised_lr, weight_decay=1e-4)
    samples = train.k.shape[0]
    page_blocks = config.page_tokens // config.block_size
    final_loss = final_ce = final_hard = 0.0
    start_wall = time.perf_counter()

    for _ in range(config.supervised_epochs):
        order = torch.randperm(samples)
        for start in range(0, samples, config.batch_size):
            idx = order[start : start + config.batch_size]
            k = train.k.index_select(0, idx).to(device)
            v = train.v.index_select(0, idx).to(device)
            q = train.q.index_select(0, idx).to(device)
            with torch.no_grad():
                z = ae.encode(k, v)
            page_scores = page_scores_from_block_scores(
                searcher(q, z),
                page_blocks,
                config.page_pool_temperature,
            )

            ce_losses: list[torch.Tensor] = []
            hard_losses: list[torch.Tensor] = []
            for local_idx, sample_idx in enumerate(idx.tolist()):
                case_name = train.meta[sample_idx].case
                positives = case_specs[case_name].evidence_pages
                hard_pages = hard_negative_pages(case_name, positives, config)
                row_scores = page_scores[local_idx]
                ce_losses.append(multi_positive_ce(row_scores, positives))
                hard_losses.append(
                    row_hard_negative_loss(
                        row_scores,
                        positives,
                        hard_pages,
                        config.hard_negative_margin,
                    )
                )
            ce_loss = torch.stack(ce_losses).mean()
            hard_loss = torch.stack(hard_losses).mean()
            loss = ce_loss + config.hard_negative_weight * hard_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            final_ce = float(ce_loss.detach().cpu())
            final_hard = float(hard_loss.detach().cpu())

    searcher.eval()
    return searcher, SupervisedTrainStats(
        final_loss=final_loss,
        final_ce_loss=final_ce,
        final_hard_loss=final_hard,
        train_seconds=time.perf_counter() - start_wall,
    )


@torch.no_grad()
def rank_pages(
    dataset: TraceDataset,
    case_name: str,
    ae: torch.nn.Module,
    searcher: LatentSearcher,
    config: Config,
    device: torch.device,
) -> dict[str, Any]:
    sample_indices = [idx for idx, meta in enumerate(dataset.meta) if meta.case == case_name]
    if not sample_indices:
        raise ValueError(f"No trace samples for case {case_name}")
    blocks = int(dataset.k.shape[1] // config.block_size)
    page_blocks = config.page_tokens // config.block_size
    pages = math.ceil(blocks / page_blocks)
    page_score_sum = torch.zeros(pages, dtype=torch.float32)

    for idx in sample_indices:
        k = dataset.k[idx : idx + 1].to(device)
        v = dataset.v[idx : idx + 1].to(device)
        q = dataset.q[idx : idx + 1].to(device)
        z = ae.encode(k, v)
        scores = page_scores_from_block_scores(
            searcher(q, z),
            page_blocks,
            config.page_pool_temperature,
        )[0].detach().cpu()
        page_score_sum += normalize_scores(scores)

    page_scores = page_score_sum / float(len(sample_indices))
    ranked_pages = sorted(range(pages), key=lambda page: (-float(page_scores[page]), page))
    return {
        "case": case_name,
        "sample_count": len(sample_indices),
        "page_scores": [float(x) for x in page_scores.tolist()],
        "ranked_pages": ranked_pages,
    }


def ranking_summary_rows(
    rankings_by_name: dict[str, dict[str, dict[str, Any]]],
    case_specs: dict[str, Any],
    config: Config,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ranker_name, case_rankings in rankings_by_name.items():
        for top_pages in config.top_pages:
            center_recalls: list[float] = []
            span_recalls: list[float] = []
            for case_name, ranking in case_rankings.items():
                case = case_specs[case_name]
                candidates = candidate_remote_pages(ranking["ranked_pages"], len(case.context_ids), config)
                selected = sorted(candidates[: min(top_pages, len(candidates))])
                expanded = expand_pages(selected, len(case.context_ids), config.page_tokens, config.page_halo_pages)
                center_recalls.append(page_recall(selected, case.evidence_pages))
                span_recalls.append(page_recall(expanded, case.evidence_pages))
            rows.append(
                {
                    "ranker": ranker_name,
                    "top_pages": top_pages,
                    "mean_center_page_recall": sum(center_recalls) / len(center_recalls),
                    "mean_span_page_recall": sum(span_recalls) / len(span_recalls),
                }
            )
    return rows


def add_full_row(
    rows: list[EndToEndRow],
    *,
    case: Any,
    model: Any,
    tokenizer: Any,
    query_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    context_tensor: torch.Tensor,
    decode_steps: int,
) -> tuple[float, float]:
    full_cache, _, full_prefill = prefill(model, context_tensor)
    full_q_cache, full_logits, full_query_seconds = run_query_on_cache(
        model,
        query_ids,
        full_cache,
        position_start=len(case.context_ids),
        past_len=len(case.context_ids),
    )
    from run_latent_scorer_raw_span_smoke import answer_nll_with_positions, greedy_decode_with_positions

    full_answer_pos = len(case.context_ids) + int(query_ids.shape[1])
    decode_cache = clone_cache(full_q_cache)
    full_nll = answer_nll_with_positions(model, answer_ids, full_logits, full_q_cache, full_answer_pos)
    full_generated, full_decode = greedy_decode_with_positions(
        model,
        tokenizer,
        full_logits,
        decode_cache,
        decode_steps,
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
    return full_prefill, full_online


def add_composed_prompt_eval_row(
    rows: list[EndToEndRow],
    *,
    case: Any,
    method: str,
    model: Any,
    tokenizer: Any,
    indices: list[int],
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
    composer_max_tokens: int,
    device: torch.device,
) -> None:
    evidence = compose_evidence_text(
        tokenizer,
        case.context_ids,
        indices,
        case.query,
        composer_max_tokens,
    )
    prompt_text = evidence + "\n\n" + case.query
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    add_prompt_eval_row(
        rows,
        case=case,
        method=method,
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        query_tokens=query_tokens,
        answer_ids=answer_ids,
        selected_pages=selected_pages,
        expanded_pages=expanded_pages,
        recall=recall,
        span_recall=span_recall,
        latent_storage_ratio=latent_storage_ratio,
        decode_steps=decode_steps,
        full_prefill_seconds=full_prefill_seconds,
        full_online_seconds=full_online_seconds,
    )


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
    train_names = train_case_names(config)
    eval_names = eval_case_names(config)
    train_indices = indices_for_cases(dataset, train_names)
    eval_indices = indices_for_cases(dataset, eval_names)
    if not train_indices:
        raise ValueError(f"No trace samples matched train cases: {train_names}")
    if not eval_indices:
        raise ValueError(f"No trace samples matched eval cases: {eval_names}")
    train = select_dataset(dataset, train_indices)
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae, attention_searcher, attention_train_stats = train_models(train, trace_config, train_device)
    case_specs = {
        case_name: make_case(tokenizer, case_name, config.prompt_tokens, config.page_tokens)
        for case_name in all_case_names(config)
    }
    supervised_searcher, supervised_train_stats = train_supervised_page_ranker(
        train,
        ae,
        attention_searcher,
        case_specs,
        config,
        train_device,
    )

    searchers = {
        "attention": attention_searcher,
        "supervised": supervised_searcher,
    }
    rankings_by_name = {
        name: {
            case_name: rank_pages(dataset, case_name, ae, searcher, config, train_device)
            for case_name in eval_names
        }
        for name, searcher in searchers.items()
    }
    eval_case_specs = {case_name: case_specs[case_name] for case_name in eval_names}
    page_summary = ranking_summary_rows(rankings_by_name, eval_case_specs, config)

    head_dim = int(dataset.k.shape[-1])
    blocks = int(dataset.k.shape[1] // config.block_size)
    latent_storage_ratio = (blocks * config.latent_dim) / float(config.prompt_tokens * 2 * head_dim)
    rows: list[EndToEndRow] = []
    page_selection_meta: list[dict[str, Any]] = []

    if config.include_prompt_rebuild:
        for case_name in eval_names:
            case = case_specs[case_name]
            context_tensor = torch.tensor(case.context_ids, dtype=torch.long, device=device).view(1, -1)
            query_ids = tokenizer(case.query, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            answer_ids = tokenizer(case.answer, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            full_prefill, full_online = add_full_row(
                rows,
                case=case,
                model=model,
                tokenizer=tokenizer,
                query_ids=query_ids,
                answer_ids=answer_ids,
                context_tensor=context_tensor,
                decode_steps=config.decode_steps,
            )

            oracle_pages = sorted(case.evidence_pages)
            oracle_indices = selected_indices_for_pages(
                oracle_pages,
                len(case.context_ids),
                config.page_tokens,
                config.recent_tokens,
                config.page_halo_pages,
            )
            oracle_expanded = expand_pages(oracle_pages, len(case.context_ids), config.page_tokens, config.page_halo_pages)
            oracle_prompt = selected_text(tokenizer, case.context_ids, oracle_indices) + "\n\n" + case.query
            oracle_prompt_ids = tokenizer(oracle_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
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
                expanded_pages=oracle_expanded,
                recall=page_recall(oracle_pages, case.evidence_pages),
                span_recall=page_recall(oracle_expanded, case.evidence_pages),
                latent_storage_ratio=latent_storage_ratio,
                decode_steps=config.decode_steps,
                full_prefill_seconds=full_prefill,
                full_online_seconds=full_online,
            )
            if config.include_evidence_composer:
                oracle_composer_indices = selected_indices_for_pages(
                    oracle_pages,
                    len(case.context_ids),
                    config.page_tokens,
                    config.recent_tokens,
                    config.page_halo_pages + config.composer_extra_halo_pages,
                )
                add_composed_prompt_eval_row(
                    rows,
                    case=case,
                    method="prompt_rebuild_oracle_pages_composed_text",
                    model=model,
                    tokenizer=tokenizer,
                    indices=oracle_composer_indices,
                    query_tokens=int(query_ids.shape[1]),
                    answer_ids=answer_ids,
                    selected_pages=oracle_pages,
                    expanded_pages=oracle_expanded,
                    recall=page_recall(oracle_pages, case.evidence_pages),
                    span_recall=page_recall(oracle_expanded, case.evidence_pages),
                    latent_storage_ratio=latent_storage_ratio,
                    decode_steps=config.decode_steps,
                    full_prefill_seconds=full_prefill,
                    full_online_seconds=full_online,
                    composer_max_tokens=config.composer_max_tokens,
                    device=device,
                )

            for ranker_name in config.eval_rankers:
                ranking = rankings_by_name[ranker_name][case_name]
                candidates = candidate_remote_pages(ranking["ranked_pages"], len(case.context_ids), config)
                for top_pages in config.top_pages:
                    selected = sorted(candidates[: min(top_pages, len(candidates))])
                    expanded = expand_pages(selected, len(case.context_ids), config.page_tokens, config.page_halo_pages)
                    indices = selected_indices_for_pages(
                        selected,
                        len(case.context_ids),
                        config.page_tokens,
                        config.recent_tokens,
                        config.page_halo_pages,
                    )
                    prompt_text = selected_text(tokenizer, case.context_ids, indices) + "\n\n" + case.query
                    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                    add_prompt_eval_row(
                        rows,
                        case=case,
                        method=f"prompt_rebuild_{ranker_name}_top{top_pages}_pages_text",
                        model=model,
                        tokenizer=tokenizer,
                        prompt_ids=prompt_ids,
                        query_tokens=int(query_ids.shape[1]),
                        answer_ids=answer_ids,
                        selected_pages=selected,
                        expanded_pages=expanded,
                        recall=page_recall(selected, case.evidence_pages),
                        span_recall=page_recall(expanded, case.evidence_pages),
                        latent_storage_ratio=latent_storage_ratio,
                        decode_steps=config.decode_steps,
                        full_prefill_seconds=full_prefill,
                        full_online_seconds=full_online,
                    )
                    if config.include_evidence_composer:
                        composer_indices = selected_indices_for_pages(
                            selected,
                            len(case.context_ids),
                            config.page_tokens,
                            config.recent_tokens,
                            config.page_halo_pages + config.composer_extra_halo_pages,
                        )
                        add_composed_prompt_eval_row(
                            rows,
                            case=case,
                            method=f"prompt_rebuild_{ranker_name}_top{top_pages}_pages_composed_text",
                            model=model,
                            tokenizer=tokenizer,
                            indices=composer_indices,
                            query_tokens=int(query_ids.shape[1]),
                            answer_ids=answer_ids,
                            selected_pages=selected,
                            expanded_pages=expanded,
                            recall=page_recall(selected, case.evidence_pages),
                            span_recall=page_recall(expanded, case.evidence_pages),
                            latent_storage_ratio=latent_storage_ratio,
                            decode_steps=config.decode_steps,
                            full_prefill_seconds=full_prefill,
                            full_online_seconds=full_online,
                            composer_max_tokens=config.composer_max_tokens,
                            device=device,
                        )
                    page_selection_meta.append(
                        {
                            "ranker": ranker_name,
                            "case": case_name,
                            "top_pages": top_pages,
                            "selected_pages": selected,
                            "expanded_pages": expanded,
                            "evidence_pages": list(case.evidence_pages),
                            "center_page_recall": page_recall(selected, case.evidence_pages),
                            "span_page_recall": page_recall(expanded, case.evidence_pages),
                            "candidate_remote_pages": candidates,
                            "ranked_pages": ranking["ranked_pages"],
                            "page_scores": ranking["page_scores"],
                        }
                    )

                if config.include_adaptive_case_top:
                    adaptive_top = adaptive_top_pages_for_case(case_name)
                    selected = sorted(candidates[: min(adaptive_top, len(candidates))])
                    expanded = expand_pages(selected, len(case.context_ids), config.page_tokens, config.page_halo_pages)
                    indices = selected_indices_for_pages(
                        selected,
                        len(case.context_ids),
                        config.page_tokens,
                        config.recent_tokens,
                        config.page_halo_pages,
                    )
                    prompt_text = selected_text(tokenizer, case.context_ids, indices) + "\n\n" + case.query
                    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                    add_prompt_eval_row(
                        rows,
                        case=case,
                        method=f"prompt_rebuild_{ranker_name}_adaptive_pages_text",
                        model=model,
                        tokenizer=tokenizer,
                        prompt_ids=prompt_ids,
                        query_tokens=int(query_ids.shape[1]),
                        answer_ids=answer_ids,
                        selected_pages=selected,
                        expanded_pages=expanded,
                        recall=page_recall(selected, case.evidence_pages),
                        span_recall=page_recall(expanded, case.evidence_pages),
                        latent_storage_ratio=latent_storage_ratio,
                        decode_steps=config.decode_steps,
                        full_prefill_seconds=full_prefill,
                        full_online_seconds=full_online,
                    )
                    if config.include_evidence_composer:
                        composer_indices = selected_indices_for_pages(
                            selected,
                            len(case.context_ids),
                            config.page_tokens,
                            config.recent_tokens,
                            config.page_halo_pages + config.composer_extra_halo_pages,
                        )
                        add_composed_prompt_eval_row(
                            rows,
                            case=case,
                            method=f"prompt_rebuild_{ranker_name}_adaptive_pages_composed_text",
                            model=model,
                            tokenizer=tokenizer,
                            indices=composer_indices,
                            query_tokens=int(query_ids.shape[1]),
                            answer_ids=answer_ids,
                            selected_pages=selected,
                            expanded_pages=expanded,
                            recall=page_recall(selected, case.evidence_pages),
                            span_recall=page_recall(expanded, case.evidence_pages),
                            latent_storage_ratio=latent_storage_ratio,
                            decode_steps=config.decode_steps,
                            full_prefill_seconds=full_prefill,
                            full_online_seconds=full_online,
                            composer_max_tokens=config.composer_max_tokens,
                            device=device,
                        )
                    page_selection_meta.append(
                        {
                            "ranker": ranker_name,
                            "case": case_name,
                            "top_pages": adaptive_top,
                            "selection_policy": "adaptive_case_top",
                            "selected_pages": selected,
                            "expanded_pages": expanded,
                            "evidence_pages": list(case.evidence_pages),
                            "center_page_recall": page_recall(selected, case.evidence_pages),
                            "span_page_recall": page_recall(expanded, case.evidence_pages),
                            "candidate_remote_pages": candidates,
                            "ranked_pages": ranking["ranked_pages"],
                            "page_scores": ranking["page_scores"],
                        }
                    )

                if config.include_adaptive_family_budget:
                    adaptive_top = adaptive_top_pages_for_case(case_name)
                    adaptive_halo = adaptive_halo_pages_for_case(case_name)
                    selected = sorted(candidates[: min(adaptive_top, len(candidates))])
                    expanded = expand_pages(selected, len(case.context_ids), config.page_tokens, adaptive_halo)
                    indices = selected_indices_for_pages(
                        selected,
                        len(case.context_ids),
                        config.page_tokens,
                        config.recent_tokens,
                        adaptive_halo,
                    )
                    prompt_text = selected_text(tokenizer, case.context_ids, indices) + "\n\n" + case.query
                    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                    add_prompt_eval_row(
                        rows,
                        case=case,
                        method=f"prompt_rebuild_{ranker_name}_adaptive_family_budget_text",
                        model=model,
                        tokenizer=tokenizer,
                        prompt_ids=prompt_ids,
                        query_tokens=int(query_ids.shape[1]),
                        answer_ids=answer_ids,
                        selected_pages=selected,
                        expanded_pages=expanded,
                        recall=page_recall(selected, case.evidence_pages),
                        span_recall=page_recall(expanded, case.evidence_pages),
                        latent_storage_ratio=latent_storage_ratio,
                        decode_steps=config.decode_steps,
                        full_prefill_seconds=full_prefill,
                        full_online_seconds=full_online,
                    )
                    if config.include_evidence_composer:
                        composer_indices = selected_indices_for_pages(
                            selected,
                            len(case.context_ids),
                            config.page_tokens,
                            config.recent_tokens,
                            adaptive_halo + config.composer_extra_halo_pages,
                        )
                        add_composed_prompt_eval_row(
                            rows,
                            case=case,
                            method=f"prompt_rebuild_{ranker_name}_adaptive_family_budget_composed_text",
                            model=model,
                            tokenizer=tokenizer,
                            indices=composer_indices,
                            query_tokens=int(query_ids.shape[1]),
                            answer_ids=answer_ids,
                            selected_pages=selected,
                            expanded_pages=expanded,
                            recall=page_recall(selected, case.evidence_pages),
                            span_recall=page_recall(expanded, case.evidence_pages),
                            latent_storage_ratio=latent_storage_ratio,
                            decode_steps=config.decode_steps,
                            full_prefill_seconds=full_prefill,
                            full_online_seconds=full_online,
                            composer_max_tokens=config.composer_max_tokens,
                            device=device,
                        )
                    page_selection_meta.append(
                        {
                            "ranker": ranker_name,
                            "case": case_name,
                            "top_pages": adaptive_top,
                            "halo_pages": adaptive_halo,
                            "selection_policy": "adaptive_family_budget",
                            "selected_pages": selected,
                            "expanded_pages": expanded,
                            "evidence_pages": list(case.evidence_pages),
                            "center_page_recall": page_recall(selected, case.evidence_pages),
                            "span_page_recall": page_recall(expanded, case.evidence_pages),
                            "candidate_remote_pages": candidates,
                            "ranked_pages": ranking["ranked_pages"],
                            "page_scores": ranking["page_scores"],
                        }
                    )

    row_dicts = [asdict(row) for row in rows]
    method_summary = aggregate_rows(rows) if rows else []
    wall_seconds = time.perf_counter() - start_wall

    write_csv(output_dir / "page_ranking_summary.csv", page_summary)
    write_csv(output_dir / "results.csv", row_dicts)
    write_csv(output_dir / "method_summary.csv", method_summary)
    (output_dir / "page_rankings.json").write_text(
        json.dumps(rankings_by_name, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "train_cases": train_names,
                "eval_cases": eval_names,
                "all_cases": all_case_names(config),
                "trace_samples": int(dataset.k.shape[0]),
                "train_samples": int(train.k.shape[0]),
                "eval_samples": int(len(eval_indices)),
                "latent_storage_ratio_vs_kv": latent_storage_ratio,
                "attention_train_stats": asdict(attention_train_stats),
                "supervised_train_stats": asdict(supervised_train_stats),
                "wall_seconds": wall_seconds,
                "case_specs": {
                    name: {
                        "answer": case.answer,
                        "evidence_pages": list(case.evidence_pages),
                        "query": case.query,
                    }
                    for name, case in case_specs.items()
                },
                "page_ranking_summary": page_summary,
                "page_selection": page_selection_meta,
                "method_summary": method_summary,
                "rows": row_dicts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("ranker,top_pages,mean_center_page_recall,mean_span_page_recall")
    for row in page_summary:
        print(
            f"{row['ranker']},{row['top_pages']},"
            f"{row['mean_center_page_recall']:.4f},{row['mean_span_page_recall']:.4f}"
        )
    if method_summary:
        print("method,cases,mean_active_tokens,mean_span_recall,mean_nll,exact_rate")
        for row in method_summary:
            print(
                f"{row['method']},{row['cases']},{row['mean_active_kv_tokens']:.1f},"
                f"{row['mean_span_page_recall']:.4f},{row['mean_answer_nll']:.4f},"
                f"{row['exact_rate']:.4f}"
            )
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
