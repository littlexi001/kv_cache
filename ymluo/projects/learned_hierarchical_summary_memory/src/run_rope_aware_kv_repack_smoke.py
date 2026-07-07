from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    prompt_tokens: int
    page_tokens: int
    top_k: int
    cases: tuple[str, ...]
    decode_steps: int
    dtype: str
    attn_implementation: str
    seed: int


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    prompt_tokens: int
    records: tuple[tuple[int, str], ...]
    query: str
    answer: str
    evidence_pages: tuple[int, ...]


@dataclass
class ResultRow:
    case: str
    method: str
    context_tokens: int
    active_kv_tokens: int
    query_tokens: int
    selected_pages: str
    prefill_seconds: float
    gather_seconds: float
    repack_seconds: float
    query_seconds: float
    decode_seconds: float
    total_online_seconds: float
    answer_nll: float
    answer_exact: bool
    speedup_vs_full_online: float
    generated: str


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Smoke test RoPE-aware sparse KV page repacking.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--prompt_tokens", type=int, default=8192)
    parser.add_argument("--page_tokens", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--cases", type=parse_csv_tuple, default=("single_exact", "two_hop", "decoy_exact"))
    parser.add_argument("--decode_steps", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=2026070611)
    args = parser.parse_args()
    return Config(**vars(args))


def resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def legacy_cache(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return tuple(cache)


def cache_from_legacy(legacy: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> Any:
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache.from_legacy_cache(legacy)
    except Exception:
        return legacy


def cache_len(cache: Any) -> int:
    return int(legacy_cache(cache)[0][0].shape[2])


def build_base_ids(tokenizer: Any, prompt_tokens: int) -> list[int]:
    filler = (
        "This neutral background passage fills the long context. It contains ordinary "
        "facts, irrelevant names, and no requested secret answer. "
    )
    filler_ids = tokenizer(filler, add_special_tokens=False)["input_ids"]
    ids: list[int] = []
    while len(ids) < prompt_tokens:
        ids.extend(filler_ids)
    return ids[:prompt_tokens]


def replace_at(ids: list[int], start: int, replacement: list[int]) -> None:
    end = min(len(ids), start + len(replacement))
    width = end - start
    if width > 0:
        ids[start:end] = replacement[:width]


def synthetic_case(name: str, prompt_tokens: int, page_tokens: int) -> SyntheticCase:
    if name == "single_exact":
        key = "MAGIC-CODE-7319"
        answer = "ORBITAL-COPPER-284"
        page = min(5, max(0, prompt_tokens // page_tokens - 2))
        record = (
            f"Verified lookup record. The secret key {key} maps to answer {answer}. "
            f"When asked about {key}, answer exactly {answer}."
        )
        return SyntheticCase(
            name=name,
            prompt_tokens=prompt_tokens,
            records=((page, record),),
            query=f"Question: What is the answer for key {key}? Answer exactly.\nAnswer:",
            answer=answer,
            evidence_pages=(page,),
        )
    if name == "two_hop":
        prompt_tokens = max(prompt_tokens, 12 * page_tokens + 512)
        key = "BRIDGE-ALPHA-42"
        node = "NODE-TULIP-17"
        answer = "HARBOR-SILVER-902"
        record_a = f"Bridge record. The key {key} points to intermediate token {node}."
        record_b = f"Answer record. The intermediate token {node} maps to final answer {answer}."
        return SyntheticCase(
            name=name,
            prompt_tokens=prompt_tokens,
            records=((3, record_a), (11, record_b)),
            query=f"Question: For key {key}, follow the bridge and give the final answer exactly.\nAnswer:",
            answer=answer,
            evidence_pages=(3, 11),
        )
    if name == "decoy_exact":
        key = "STATUS-CODE-8801"
        answer = "VERIFIED-EMBER-447"
        decoy = "OBSOLETE-EMBER-447"
        decoy_record = f"Obsolete record. The key {key} used to map to answer {decoy}, but this record is outdated."
        true_record = f"Verified current record. The key {key} now maps to answer {answer}. Use the verified current record."
        return SyntheticCase(
            name=name,
            prompt_tokens=prompt_tokens,
            records=((4, decoy_record), (6, true_record)),
            query=f"Question: What is the verified current answer for key {key}? Answer exactly.\nAnswer:",
            answer=answer,
            evidence_pages=(4, 6),
        )
    if name == "multi_value":
        key_a = "ALPHA-ID-440"
        key_b = "BETA-ID-771"
        ans_a = "CRIMSON-LAKE-118"
        ans_b = "SILVER-VALE-903"
        rec_a = f"Verified lookup record. The key {key_a} maps to answer {ans_a}."
        rec_b = f"Verified lookup record. The key {key_b} maps to answer {ans_b}."
        return SyntheticCase(
            name=name,
            prompt_tokens=prompt_tokens,
            records=((2, rec_a), (7, rec_b)),
            query=f"Question: Give the answers for {key_a} and {key_b} in order, separated by semicolon.\nAnswer:",
            answer=f"{ans_a}; {ans_b}",
            evidence_pages=(2, 7),
        )
    raise ValueError(f"unknown case: {name}")


def make_context_ids(tokenizer: Any, case: SyntheticCase, page_tokens: int) -> list[int]:
    ids = build_base_ids(tokenizer, case.prompt_tokens)
    for page, text in case.records:
        record_ids = tokenizer("\n\n" + text + "\n\n", add_special_tokens=False)["input_ids"]
        replace_at(ids, page * page_tokens + 32, record_ids)
    return ids


def content_words(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9-]{3,}", text.lower())
    return {word for word in words if word not in {"the", "and", "for", "with", "answer", "question"}}


def lexical_pages(tokenizer: Any, context_ids: list[int], query: str, page_tokens: int, top_k: int) -> list[int]:
    query_terms = content_words(query)
    query_codes = set(re.findall(r"[A-Za-z]+-[A-Za-z0-9-]+", query))
    scored = []
    for page, start in enumerate(range(0, len(context_ids), page_tokens)):
        text = tokenizer.decode(context_ids[start : start + page_tokens], skip_special_tokens=True)
        score = len(query_terms & content_words(text))
        score += 4 * len(query_codes & set(re.findall(r"[A-Za-z]+-[A-Za-z0-9-]+", text)))
        scored.append((score, -page, page))
    scored.sort(reverse=True)
    return sorted(page for _, _, page in scored[:top_k])


def selected_pages(tokenizer: Any, context_ids: list[int], case: SyntheticCase, page_tokens: int, top_k: int) -> list[int]:
    pages = list(case.evidence_pages)
    if len(pages) >= top_k:
        return sorted(pages[:top_k])
    for page in lexical_pages(tokenizer, context_ids, case.query, page_tokens, top_k=max(top_k, len(pages) + 2)):
        if page not in pages:
            pages.append(page)
        if len(pages) >= top_k:
            break
    return sorted(pages)


def page_indices(pages: list[int], context_len: int, page_tokens: int, device: torch.device) -> torch.Tensor:
    indices: list[int] = []
    for page in pages:
        start = page * page_tokens
        end = min(context_len, start + page_tokens)
        indices.extend(range(start, end))
    if not indices:
        raise ValueError("empty page selection")
    return torch.tensor(indices, dtype=torch.long, device=device)


def gather_cache(cache: Any, indices: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    gathered = []
    for key, value in legacy_cache(cache):
        gathered.append((key.index_select(2, indices), value.index_select(2, indices)))
    return tuple(gathered)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rotary_inv_freq(model: Any, head_dim: int, device: torch.device) -> torch.Tensor:
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    inv_freq = getattr(rotary, "inv_freq", None)
    if inv_freq is not None:
        return inv_freq.to(device=device, dtype=torch.float32)
    theta = float(getattr(model.config, "rope_theta", 10000.0))
    rotary_dim = int(getattr(model.config, "head_dim", head_dim))
    return 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))


def apply_rope_delta_to_key(
    key: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    rot_dim = min(key.shape[-1], int(inv_freq.numel() * 2))
    if rot_dim <= 0:
        return key
    key_rot = key[..., :rot_dim]
    key_pass = key[..., rot_dim:]
    delta = (new_positions.to(torch.float32) - old_positions.to(torch.float32)).to(inv_freq.device)
    freqs = torch.outer(delta, inv_freq[: rot_dim // 2])
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=key.dtype).view(1, 1, -1, rot_dim)
    sin = emb.sin().to(dtype=key.dtype).view(1, 1, -1, rot_dim)
    repacked = key_rot * cos + rotate_half(key_rot) * sin
    return torch.cat((repacked, key_pass), dim=-1) if key_pass.numel() else repacked


def gather_and_rope_repack_cache(
    model: Any,
    cache: Any,
    indices: torch.Tensor,
    new_positions: torch.Tensor | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    legacy = legacy_cache(cache)
    head_dim = int(legacy[0][0].shape[-1])
    inv_freq = rotary_inv_freq(model, head_dim, indices.device)
    if new_positions is None:
        new_positions = torch.arange(int(indices.numel()), dtype=torch.long, device=indices.device)
    repacked = []
    for key, value in legacy:
        gathered_key = key.index_select(2, indices)
        gathered_value = value.index_select(2, indices)
        repacked_key = apply_rope_delta_to_key(gathered_key, indices, new_positions, inv_freq)
        repacked.append((repacked_key, gathered_value))
    return tuple(repacked)


def prefill(model: Any, input_ids: torch.Tensor) -> tuple[Any, torch.Tensor, float]:
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    synchronize()
    return out.past_key_values, out.logits[:, -1, :], time.perf_counter() - start


def run_query_on_cache(
    model: Any,
    query_ids: torch.Tensor,
    past_key_values: Any,
    position_start: int,
    past_len: int,
) -> tuple[Any, torch.Tensor, float]:
    device = query_ids.device
    q_len = int(query_ids.shape[1])
    attention_mask = torch.ones((1, past_len + q_len), dtype=torch.long, device=device)
    position_ids = torch.arange(position_start, position_start + q_len, device=device).view(1, -1)
    cache_position = torch.arange(position_start, position_start + q_len, device=device)
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        try:
            out = model(
                input_ids=query_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
            )
        except TypeError:
            out = model(
                input_ids=query_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
    synchronize()
    return out.past_key_values, out.logits[:, -1, :], time.perf_counter() - start


def greedy_decode(model: Any, tokenizer: Any, logits: torch.Tensor, past_key_values: Any, steps: int) -> tuple[str, float]:
    if steps <= 0:
        return "", 0.0
    device = logits.device
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    generated: list[int] = []
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(steps):
            generated.append(int(next_token.item()))
            out = model(input_ids=next_token.to(device), past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True).to(device)
    synchronize()
    return tokenizer.decode(generated, skip_special_tokens=True), time.perf_counter() - start


def answer_nll(model: Any, answer_ids: torch.Tensor, logits: torch.Tensor, past_key_values: Any) -> float:
    if answer_ids.numel() == 0:
        return 0.0
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
            out = model(input_ids=target.view(1, 1), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            current_logits = out.logits[:, -1, :]
    return total / max(1, count)


def selected_text(tokenizer: Any, context_ids: list[int], pages: list[int], page_tokens: int) -> str:
    chunks = []
    for page in pages:
        chunks.append(
            tokenizer.decode(
                context_ids[page * page_tokens : min(len(context_ids), (page + 1) * page_tokens)],
                skip_special_tokens=True,
            )
        )
    return "\n\n".join(chunks)


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


def add_cache_result(
    rows: list[ResultRow],
    *,
    case: SyntheticCase,
    method: str,
    model: Any,
    tokenizer: Any,
    query_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    cache: Any,
    active_kv_tokens: int,
    selected_pages_value: str,
    prefill_seconds: float,
    gather_seconds: float,
    repack_seconds: float,
    position_start: int,
    decode_steps: int,
    full_online: float,
) -> None:
    q_cache, logits, query_seconds = run_query_on_cache(
        model,
        query_ids,
        cache,
        position_start=position_start,
        past_len=active_kv_tokens,
    )
    generated, decode_seconds = greedy_decode(model, tokenizer, logits, q_cache, decode_steps)
    nll = answer_nll(model, answer_ids, logits, q_cache)
    total = gather_seconds + repack_seconds + query_seconds + decode_seconds
    rows.append(
        ResultRow(
            case=case.name,
            method=method,
            context_tokens=case.prompt_tokens,
            active_kv_tokens=active_kv_tokens,
            query_tokens=int(query_ids.shape[1]),
            selected_pages=selected_pages_value,
            prefill_seconds=prefill_seconds,
            gather_seconds=gather_seconds,
            repack_seconds=repack_seconds,
            query_seconds=query_seconds,
            decode_seconds=decode_seconds,
            total_online_seconds=total,
            answer_nll=nll,
            answer_exact=case.answer in generated,
            speedup_vs_full_online=full_online / total if total > 0 else 0.0,
            generated=generated.replace("\n", " ")[:300],
        )
    )


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = parse_args()
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(config.dtype),
        attn_implementation=config.attn_implementation,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    device = next(model.parameters()).device

    rows: list[ResultRow] = []
    case_meta: list[dict[str, Any]] = []

    for case_name in config.cases:
        case = synthetic_case(case_name, config.prompt_tokens, config.page_tokens)
        context_ids = make_context_ids(tokenizer, case, config.page_tokens)
        case = SyntheticCase(**{**asdict(case), "prompt_tokens": len(context_ids)})
        context_tensor = torch.tensor(context_ids, dtype=torch.long, device=device).view(1, -1)
        query_ids = tokenizer(case.query, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        answer_ids = tokenizer(case.answer, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        pages = selected_pages(tokenizer, context_ids, case, config.page_tokens, config.top_k)
        indices = page_indices(pages, len(context_ids), config.page_tokens, device)
        pages_json = json.dumps(pages)

        full_cache, _, full_prefill = prefill(model, context_tensor)
        full_q_cache, full_logits, full_query_seconds = run_query_on_cache(
            model,
            query_ids,
            full_cache,
            position_start=len(context_ids),
            past_len=len(context_ids),
        )
        full_generated, full_decode = greedy_decode(model, tokenizer, full_logits, full_q_cache, config.decode_steps)
        full_nll = answer_nll(model, answer_ids, full_logits, full_q_cache)
        full_online = full_query_seconds + full_decode
        rows.append(
            ResultRow(
                case=case.name,
                method="full_kv_cache",
                context_tokens=len(context_ids),
                active_kv_tokens=len(context_ids),
                query_tokens=int(query_ids.shape[1]),
                selected_pages="all",
                prefill_seconds=full_prefill,
                gather_seconds=0.0,
                repack_seconds=0.0,
                query_seconds=full_query_seconds,
                decode_seconds=full_decode,
                total_online_seconds=full_online,
                answer_nll=full_nll,
                answer_exact=case.answer in full_generated,
                speedup_vs_full_online=1.0,
                generated=full_generated.replace("\n", " ")[:300],
            )
        )

        synchronize()
        start = time.perf_counter()
        naive_legacy = gather_cache(full_cache, indices)
        naive_cache = cache_from_legacy(naive_legacy)
        synchronize()
        gather_seconds = time.perf_counter() - start
        add_cache_result(
            rows,
            case=case,
            method="naive_kv_gather_absolute_query_pos",
            model=model,
            tokenizer=tokenizer,
            query_ids=query_ids,
            answer_ids=answer_ids,
            cache=naive_cache,
            active_kv_tokens=int(indices.numel()),
            selected_pages_value=pages_json,
            prefill_seconds=0.0,
            gather_seconds=gather_seconds,
            repack_seconds=0.0,
            position_start=len(context_ids),
            decode_steps=config.decode_steps,
            full_online=full_online,
        )
        add_cache_result(
            rows,
            case=case,
            method="naive_kv_gather_compact_query_pos",
            model=model,
            tokenizer=tokenizer,
            query_ids=query_ids,
            answer_ids=answer_ids,
            cache=naive_cache,
            active_kv_tokens=int(indices.numel()),
            selected_pages_value=pages_json,
            prefill_seconds=0.0,
            gather_seconds=0.0,
            repack_seconds=0.0,
            position_start=int(indices.numel()),
            decode_steps=config.decode_steps,
            full_online=full_online,
        )

        synchronize()
        start = time.perf_counter()
        repacked_cache = cache_from_legacy(gather_and_rope_repack_cache(model, full_cache, indices))
        synchronize()
        repack_seconds = time.perf_counter() - start
        add_cache_result(
            rows,
            case=case,
            method="rope_delta_repack_compact_query_pos",
            model=model,
            tokenizer=tokenizer,
            query_ids=query_ids,
            answer_ids=answer_ids,
            cache=repacked_cache,
            active_kv_tokens=cache_len(repacked_cache),
            selected_pages_value=pages_json,
            prefill_seconds=0.0,
            gather_seconds=0.0,
            repack_seconds=repack_seconds,
            position_start=cache_len(repacked_cache),
            decode_steps=config.decode_steps,
            full_online=full_online,
        )

        shifted_positions = indices - int(indices.min().item())
        synchronize()
        start = time.perf_counter()
        shifted_cache = cache_from_legacy(
            gather_and_rope_repack_cache(model, full_cache, indices, new_positions=shifted_positions)
        )
        synchronize()
        shifted_repack_seconds = time.perf_counter() - start
        add_cache_result(
            rows,
            case=case,
            method="rope_delta_repack_shifted_query_pos",
            model=model,
            tokenizer=tokenizer,
            query_ids=query_ids,
            answer_ids=answer_ids,
            cache=shifted_cache,
            active_kv_tokens=cache_len(shifted_cache),
            selected_pages_value=pages_json,
            prefill_seconds=0.0,
            gather_seconds=0.0,
            repack_seconds=shifted_repack_seconds,
            position_start=len(context_ids) - int(indices.min().item()),
            decode_steps=config.decode_steps,
            full_online=full_online,
        )

        prompt_text = selected_text(tokenizer, context_ids, pages, config.page_tokens) + "\n\n" + case.query
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        prompt_cache, prompt_logits, prompt_prefill = prefill(model, prompt_ids)
        prompt_generated, prompt_decode = greedy_decode(model, tokenizer, prompt_logits, prompt_cache, config.decode_steps)
        prompt_nll = answer_nll(model, answer_ids, prompt_logits, prompt_cache)
        prompt_total = prompt_prefill + prompt_decode
        rows.append(
            ResultRow(
                case=case.name,
                method="prompt_rebuild_selected_pages",
                context_tokens=len(context_ids),
                active_kv_tokens=int(prompt_ids.shape[1]),
                query_tokens=int(query_ids.shape[1]),
                selected_pages=pages_json,
                prefill_seconds=prompt_prefill,
                gather_seconds=0.0,
                repack_seconds=0.0,
                query_seconds=0.0,
                decode_seconds=prompt_decode,
                total_online_seconds=prompt_total,
                answer_nll=prompt_nll,
                answer_exact=case.answer in prompt_generated,
                speedup_vs_full_online=full_online / prompt_total if prompt_total > 0 else 0.0,
                generated=prompt_generated.replace("\n", " ")[:300],
            )
        )

        case_meta.append(
            {
                "case": case.name,
                "answer": case.answer,
                "records": case.records,
                "evidence_pages": case.evidence_pages,
                "selected_pages": pages,
                "context_tokens": len(context_ids),
                "query": case.query,
            }
        )

    row_dicts = [asdict(row) for row in rows]
    write_csv(output_dir / "results.csv", row_dicts)
    (output_dir / "summary.json").write_text(
        json.dumps({"config": asdict(config), "cases": case_meta, "rows": row_dicts}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("case,method,active_kv,total_online,speedup,nll,exact,selected_pages,generated")
    for row in rows:
        print(
            f"{row.case},{row.method},{row.active_kv_tokens},{row.total_online_seconds:.4f},"
            f"{row.speedup_vs_full_online:.3f},{row.answer_nll:.4f},{row.answer_exact},"
            f"{row.selected_pages},{row.generated[:100]}"
        )
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
