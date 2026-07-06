from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_qwen8b_paper_benchmarks import content_words  # noqa: E402


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    prompt_tokens: int
    page_tokens: int
    cases: tuple[str, ...]
    decode_steps: int
    dtype: str
    attn_implementation: str
    seed: int
    query_position_mode: str
    span_before_pages: int
    span_after_pages: int


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    prompt_tokens: int
    records: tuple[tuple[int, str], ...]
    query: str
    answer: str
    evidence_pages: tuple[int, ...]
    summary_text: str


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
    parser = argparse.ArgumentParser(description="Unified smoke test for typed span-aware KV memory packaging.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--prompt_tokens", type=int, default=8192)
    parser.add_argument("--page_tokens", type=int, default=1024)
    parser.add_argument("--cases", type=parse_csv_tuple, default=("single_exact", "two_hop"))
    parser.add_argument("--decode_steps", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=2026070601)
    parser.add_argument("--query_position_mode", choices=["absolute", "compact"], default="absolute")
    parser.add_argument("--span_before_pages", type=int, default=1)
    parser.add_argument("--span_after_pages", type=int, default=0)
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


def concat_legacy_caches(
    caches: list[tuple[tuple[torch.Tensor, torch.Tensor], ...]],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if not caches:
        raise ValueError("No caches to concatenate.")
    num_layers = len(caches[0])
    merged: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in range(num_layers):
        keys = [cache[layer_idx][0] for cache in caches]
        values = [cache[layer_idx][1] for cache in caches]
        merged.append((torch.cat(keys, dim=2), torch.cat(values, dim=2)))
    return tuple(merged)


def build_base_ids(tokenizer: Any, prompt_tokens: int) -> list[int]:
    filler = (
        "This background passage fills a long context. It contains neutral facts, "
        "irrelevant names, and ordinary prose. It does not contain the requested secret answer. "
    )
    filler_ids = tokenizer(filler, add_special_tokens=False)["input_ids"]
    ids: list[int] = []
    while len(ids) < prompt_tokens:
        ids.extend(filler_ids)
    return ids[:prompt_tokens]


def replace_at(ids: list[int], start: int, replacement: list[int]) -> None:
    end = min(len(ids), start + len(replacement))
    width = end - start
    if width <= 0:
        return
    ids[start:end] = replacement[:width]


def synthetic_case(name: str, prompt_tokens: int, page_tokens: int) -> SyntheticCase:
    if name == "single_exact":
        key = "MAGIC-CODE-7319"
        answer = "ORBITAL-COPPER-284"
        record = (
            f"Verified lookup record. The secret key {key} maps to answer {answer}. "
            f"When asked about {key}, answer exactly {answer}."
        )
        query = f"Question: What is the answer for key {key}? Answer exactly.\nAnswer:"
        summary = f"Typed memory summary: key {key} -> answer {answer}."
        return SyntheticCase(
            name=name,
            prompt_tokens=prompt_tokens,
            records=((5, record),),
            query=query,
            answer=answer,
            evidence_pages=(5,),
            summary_text=summary,
        )

    if name == "two_hop":
        prompt_tokens = max(prompt_tokens, 12 * page_tokens + 512)
        key = "BRIDGE-ALPHA-42"
        node = "NODE-TULIP-17"
        answer = "HARBOR-SILVER-902"
        record_a = f"Bridge record. The key {key} points to intermediate token {node}."
        record_b = f"Answer record. The intermediate token {node} maps to final answer {answer}."
        query = f"Question: For key {key}, follow the bridge and give the final answer exactly.\nAnswer:"
        summary = f"Typed memory summary: key {key} -> intermediate {node}; {node} -> answer {answer}."
        return SyntheticCase(
            name=name,
            prompt_tokens=prompt_tokens,
            records=((3, record_a), (11, record_b)),
            query=query,
            answer=answer,
            evidence_pages=(3, 11),
            summary_text=summary,
        )

    if name == "decoy_exact":
        key = "STATUS-CODE-8801"
        answer = "VERIFIED-EMBER-447"
        decoy = "OBSOLETE-EMBER-447"
        decoy_record = f"Obsolete record. The key {key} used to map to answer {decoy}, but this record is outdated."
        true_record = f"Verified current record. The key {key} now maps to answer {answer}. Use the verified current record."
        query = f"Question: What is the verified current answer for key {key}? Answer exactly.\nAnswer:"
        summary = f"Typed memory summary: verified current key {key} -> answer {answer}; obsolete value {decoy} is not current."
        return SyntheticCase(
            name=name,
            prompt_tokens=prompt_tokens,
            records=((4, decoy_record), (6, true_record)),
            query=query,
            answer=answer,
            evidence_pages=(6,),
            summary_text=summary,
        )

    raise ValueError(f"Unknown case: {name}")


def make_context_ids(tokenizer: Any, case: SyntheticCase, page_tokens: int) -> list[int]:
    ids = build_base_ids(tokenizer, case.prompt_tokens)
    for page, text in case.records:
        record_ids = tokenizer("\n\n" + text + "\n\n", add_special_tokens=False)["input_ids"]
        replace_at(ids, page * page_tokens + 16, record_ids)
    return ids


def page_indices(pages: list[int], context_len: int, page_tokens: int, device: torch.device) -> torch.Tensor:
    indices: list[int] = []
    for page in pages:
        start = max(0, page * page_tokens)
        end = min(context_len, start + page_tokens)
        indices.extend(range(start, end))
    if not indices:
        raise ValueError("Selected page list is empty.")
    return torch.tensor(indices, dtype=torch.long, device=device)


def pages_for_evidence_span(case: SyntheticCase, context_len: int, page_tokens: int, before: int, after: int) -> list[int]:
    last_page = max(0, (context_len - 1) // page_tokens)
    pages: set[int] = set()
    for page in case.evidence_pages:
        for item in range(max(0, page - before), min(last_page, page + after) + 1):
            pages.add(item)
    return sorted(pages)


def lexical_pages(tokenizer: Any, context_ids: list[int], query: str, page_tokens: int, top_k: int) -> list[int]:
    query_terms = set(content_words(query))
    query_numbers = set(re.findall(r"[A-Za-z]*\d+[A-Za-z0-9-]*", query))
    scored = []
    for page_id, start in enumerate(range(0, len(context_ids), page_tokens)):
        text = tokenizer.decode(context_ids[start : start + page_tokens], skip_special_tokens=True)
        score = len(query_terms & set(content_words(text)))
        score += 3 * len(query_numbers & set(re.findall(r"[A-Za-z]*\d+[A-Za-z0-9-]*", text)))
        scored.append((score, -page_id, page_id))
    scored.sort(reverse=True)
    return sorted(page for score, _, page in scored[:top_k] if score > 0) or sorted(page for _, _, page in scored[:top_k])


def gather_cache(cache: Any, indices: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    gathered = []
    for key, value in legacy_cache(cache):
        gathered.append((key.index_select(2, indices), value.index_select(2, indices)))
    return tuple(gathered)


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
    context_position_start: int,
    past_len: int,
) -> tuple[Any, torch.Tensor, float]:
    device = query_ids.device
    q_len = int(query_ids.shape[1])
    attention_mask = torch.ones((1, past_len + q_len), dtype=torch.long, device=device)
    position_ids = torch.arange(context_position_start, context_position_start + q_len, device=device).view(1, -1)
    cache_position = torch.arange(context_position_start, context_position_start + q_len, device=device)
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


def cache_len(cache: Any) -> int:
    legacy = legacy_cache(cache)
    return int(legacy[0][0].shape[2])


def run_cached_method(
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
    context_position_start: int,
    gather_seconds: float,
    decode_steps: int,
    full_online: float,
) -> ResultRow:
    q_cache, logits, query_seconds = run_query_on_cache(
        model,
        query_ids,
        cache,
        context_position_start=context_position_start,
        past_len=active_kv_tokens,
    )
    generated, decode_seconds = greedy_decode(model, tokenizer, logits, q_cache, decode_steps)
    nll = answer_nll(model, answer_ids, logits, q_cache)
    total = gather_seconds + query_seconds + decode_seconds
    return ResultRow(
        case=case.name,
        method=method,
        context_tokens=case.prompt_tokens,
        active_kv_tokens=active_kv_tokens,
        query_tokens=int(query_ids.shape[1]),
        selected_pages=selected_pages_value,
        prefill_seconds=0.0,
        gather_seconds=gather_seconds,
        query_seconds=query_seconds,
        decode_seconds=decode_seconds,
        total_online_seconds=total,
        answer_nll=nll,
        answer_exact=case.answer in generated,
        speedup_vs_full_online=full_online / total if total > 0 else 0.0,
        generated=generated.replace("\n", " ")[:300],
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
    case_summaries: list[dict[str, Any]] = []

    for case_name in config.cases:
        case = synthetic_case(case_name, config.prompt_tokens, config.page_tokens)
        context_ids = make_context_ids(tokenizer, case, config.page_tokens)
        case = SyntheticCase(**{**asdict(case), "prompt_tokens": len(context_ids)})
        context_tensor = torch.tensor(context_ids, dtype=torch.long, device=device).view(1, -1)
        query_ids = tokenizer(case.query, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        answer_ids = tokenizer(case.answer, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

        full_cache, _, full_prefill = prefill(model, context_tensor)
        full_q_cache, full_logits, full_query_seconds = run_query_on_cache(
            model,
            query_ids,
            full_cache,
            context_position_start=len(context_ids),
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
                query_seconds=full_query_seconds,
                decode_seconds=full_decode,
                total_online_seconds=full_online,
                answer_nll=full_nll,
                answer_exact=case.answer in full_generated,
                speedup_vs_full_online=1.0,
                generated=full_generated.replace("\n", " ")[:300],
            )
        )

        sparse_pages = sorted({0, *case.evidence_pages})
        evidence_span_pages = pages_for_evidence_span(
            case, len(context_ids), config.page_tokens, config.span_before_pages, config.span_after_pages
        )
        prefix_pages = list(range(0, max(case.evidence_pages) + 1))
        lexical_top_pages = lexical_pages(
            tokenizer, context_ids, case.query, config.page_tokens, top_k=max(2, len(case.evidence_pages) + 1)
        )
        page_sets = {
            "arbitrary_sparse_kv_gather": sparse_pages,
            "lexical_sparse_kv_gather": lexical_top_pages,
            "evidence_span_kv_gather": evidence_span_pages,
            "prefix_span_kv_gather": prefix_pages,
        }

        for method, pages in page_sets.items():
            indices = page_indices(pages, len(context_ids), config.page_tokens, device)
            synchronize()
            start = time.perf_counter()
            gathered_legacy = gather_cache(full_cache, indices)
            gathered_cache = cache_from_legacy(gathered_legacy)
            synchronize()
            gather_seconds = time.perf_counter() - start
            position_start = len(context_ids) if config.query_position_mode == "absolute" else int(indices.numel())
            rows.append(
                run_cached_method(
                    case=case,
                    method=method,
                    model=model,
                    tokenizer=tokenizer,
                    query_ids=query_ids,
                    answer_ids=answer_ids,
                    cache=gathered_cache,
                    active_kv_tokens=int(indices.numel()),
                    selected_pages_value=json.dumps(pages),
                    context_position_start=position_start,
                    gather_seconds=gather_seconds,
                    decode_steps=config.decode_steps,
                    full_online=full_online,
                )
            )

        prompt_pages = sorted(case.evidence_pages)
        prompt_text = selected_text(tokenizer, context_ids, prompt_pages, config.page_tokens) + "\n\n" + case.query
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        prompt_cache, prompt_logits, prompt_prefill = prefill(model, prompt_ids)
        prompt_generated, prompt_decode = greedy_decode(model, tokenizer, prompt_logits, prompt_cache, config.decode_steps)
        prompt_nll = answer_nll(model, answer_ids, prompt_logits, prompt_cache)
        prompt_total = prompt_prefill + prompt_decode
        rows.append(
            ResultRow(
                case=case.name,
                method="prompt_rebuild_evidence_text",
                context_tokens=len(context_ids),
                active_kv_tokens=int(prompt_ids.shape[1]),
                query_tokens=int(query_ids.shape[1]),
                selected_pages=json.dumps(prompt_pages),
                prefill_seconds=prompt_prefill,
                gather_seconds=0.0,
                query_seconds=0.0,
                decode_seconds=prompt_decode,
                total_online_seconds=prompt_total,
                answer_nll=prompt_nll,
                answer_exact=case.answer in prompt_generated,
                speedup_vs_full_online=full_online / prompt_total if prompt_total > 0 else 0.0,
                generated=prompt_generated.replace("\n", " ")[:300],
            )
        )

        summary_ids = tokenizer(case.summary_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        summary_cache, _, summary_prefill = prefill(model, summary_ids)
        rows.append(
            run_cached_method(
                case=case,
                method="typed_summary_kv_memory",
                model=model,
                tokenizer=tokenizer,
                query_ids=query_ids,
                answer_ids=answer_ids,
                cache=summary_cache,
                active_kv_tokens=int(summary_ids.shape[1]),
                selected_pages_value="summary",
                context_position_start=int(summary_ids.shape[1]),
                gather_seconds=0.0,
                decode_steps=config.decode_steps,
                full_online=full_online,
            )
        )
        rows[-1].prefill_seconds = summary_prefill

        span_indices = page_indices(evidence_span_pages, len(context_ids), config.page_tokens, device)
        synchronize()
        start = time.perf_counter()
        span_legacy = gather_cache(full_cache, span_indices)
        hybrid_legacy = concat_legacy_caches([legacy_cache(summary_cache), span_legacy])
        hybrid_cache = cache_from_legacy(hybrid_legacy)
        synchronize()
        hybrid_gather = time.perf_counter() - start
        hybrid_len = cache_len(hybrid_cache)
        hybrid_position = len(context_ids) if config.query_position_mode == "absolute" else hybrid_len
        rows.append(
            run_cached_method(
                case=case,
                method="typed_summary_plus_span_kv",
                model=model,
                tokenizer=tokenizer,
                query_ids=query_ids,
                answer_ids=answer_ids,
                cache=hybrid_cache,
                active_kv_tokens=hybrid_len,
                selected_pages_value=json.dumps({"summary": True, "span_pages": evidence_span_pages}),
                context_position_start=hybrid_position,
                gather_seconds=hybrid_gather,
                decode_steps=config.decode_steps,
                full_online=full_online,
            )
        )

        case_summaries.append(
            {
                "case": case.name,
                "answer": case.answer,
                "records": case.records,
                "evidence_pages": case.evidence_pages,
                "sparse_pages": sparse_pages,
                "evidence_span_pages": evidence_span_pages,
                "prefix_pages": [prefix_pages[0], prefix_pages[-1]],
                "lexical_top_pages": lexical_top_pages,
            }
        )

    row_dicts = [asdict(row) for row in rows]
    write_csv(output_dir / "results.csv", row_dicts)
    (output_dir / "summary.json").write_text(
        json.dumps({"config": asdict(config), "cases": case_summaries, "rows": row_dicts}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("case,method,active_kv,total_online,speedup,answer_nll,exact,selected_pages,generated")
    for row in rows:
        print(
            f"{row.case},{row.method},{row.active_kv_tokens},{row.total_online_seconds:.4f},"
            f"{row.speedup_vs_full_online:.3f},{row.answer_nll:.4f},{row.answer_exact},"
            f"{row.selected_pages},{row.generated[:80]}"
        )


if __name__ == "__main__":
    main()
