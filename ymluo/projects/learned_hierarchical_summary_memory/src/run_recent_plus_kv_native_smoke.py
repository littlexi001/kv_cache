from __future__ import annotations

import argparse
import csv
import json
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
    recent_tokens: int
    cases: tuple[str, ...]
    decode_steps: int
    dtype: str
    attn_implementation: str
    seed: int


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    context_ids: list[int]
    query: str
    answer: str
    evidence_pages: tuple[int, ...]
    recent_record: bool


@dataclass
class ResultRow:
    case: str
    method: str
    context_tokens: int
    active_kv_tokens: int
    query_tokens: int
    selected_units: str
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


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="KV-native recent-plus smoke demo.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--prompt_tokens", type=int, default=8192)
    parser.add_argument("--page_tokens", type=int, default=1024)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--cases", type=parse_csv_tuple, default=("old_single", "recent_single", "two_old"))
    parser.add_argument("--decode_steps", type=int, default=24)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=2026070607)
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
        "This neutral background passage fills the long context. It includes ordinary "
        "narrative details, irrelevant names, and no requested secret answer. "
    )
    filler_ids = tokenizer(filler, add_special_tokens=False)["input_ids"]
    ids: list[int] = []
    while len(ids) < prompt_tokens:
        ids.extend(filler_ids)
    return ids[:prompt_tokens]


def replace_at(ids: list[int], start: int, replacement: list[int]) -> None:
    if start >= len(ids):
        return
    end = min(len(ids), start + len(replacement))
    ids[start:end] = replacement[: end - start]


def make_case(tokenizer: Any, name: str, prompt_tokens: int, page_tokens: int, recent_tokens: int) -> SyntheticCase:
    ids = build_base_ids(tokenizer, prompt_tokens)
    if name == "old_single":
        key = "OLD-MEM-KEY-7319"
        answer = "ORBITAL-COPPER-284"
        record = (
            f"Verified old memory record. The key {key} maps to answer {answer}. "
            f"If asked about {key}, answer exactly {answer}."
        )
        page = min(5, max(0, prompt_tokens // page_tokens - 2))
        replace_at(ids, page * page_tokens + 32, tokenizer("\n\n" + record + "\n\n", add_special_tokens=False)["input_ids"])
        return SyntheticCase(
            name=name,
            context_ids=ids,
            query=f"Question: What is the answer for key {key}? Answer exactly.\nAnswer:",
            answer=answer,
            evidence_pages=(page,),
            recent_record=False,
        )
    if name == "recent_single":
        key = "RECENT-MEM-KEY-8821"
        answer = "VIOLET-HARBOR-519"
        record = (
            f"Verified recent memory record. The key {key} maps to answer {answer}. "
            f"If asked about {key}, answer exactly {answer}."
        )
        start = max(0, prompt_tokens - recent_tokens + 32)
        replace_at(ids, start, tokenizer("\n\n" + record + "\n\n", add_special_tokens=False)["input_ids"])
        return SyntheticCase(
            name=name,
            context_ids=ids,
            query=f"Question: What is the answer for key {key}? Answer exactly.\nAnswer:",
            answer=answer,
            evidence_pages=(),
            recent_record=True,
        )
    if name == "two_old":
        key = "BRIDGE-ALPHA-42"
        node = "NODE-TULIP-17"
        answer = "HARBOR-SILVER-902"
        page_a = min(3, max(0, prompt_tokens // page_tokens - 3))
        page_b = min(7, max(page_a + 1, prompt_tokens // page_tokens - 2))
        rec_a = f"Bridge old memory record. The key {key} points to intermediate token {node}."
        rec_b = f"Answer old memory record. The intermediate token {node} maps to final answer {answer}."
        replace_at(ids, page_a * page_tokens + 32, tokenizer("\n\n" + rec_a + "\n\n", add_special_tokens=False)["input_ids"])
        replace_at(ids, page_b * page_tokens + 32, tokenizer("\n\n" + rec_b + "\n\n", add_special_tokens=False)["input_ids"])
        return SyntheticCase(
            name=name,
            context_ids=ids,
            query=f"Question: For key {key}, follow the bridge and give the final answer exactly.\nAnswer:",
            answer=answer,
            evidence_pages=(page_a, page_b),
            recent_record=False,
        )
    raise ValueError(name)


def page_token_indices(pages: list[int], context_len: int, page_tokens: int) -> list[int]:
    out: list[int] = []
    for page in pages:
        start = page * page_tokens
        end = min(context_len, start + page_tokens)
        out.extend(range(start, end))
    return out


def recent_indices(context_len: int, recent_tokens: int) -> list[int]:
    return list(range(max(0, context_len - recent_tokens), context_len))


def unique_ordered(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def tensor_indices(values: list[int], device: torch.device) -> torch.Tensor:
    if not values:
        raise ValueError("empty KV index list")
    return torch.tensor(values, dtype=torch.long, device=device)


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


def selected_text(tokenizer: Any, context_ids: list[int], indices: list[int]) -> str:
    if not indices:
        return ""
    # Convert selected token indices back to text in their selected order.
    return tokenizer.decode([context_ids[idx] for idx in indices], skip_special_tokens=True)


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


def add_cache_row(
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
    selected_units: Any,
    position_start: int,
    gather_seconds: float,
    decode_steps: int,
    full_prefill_seconds: float,
    full_online_seconds: float,
) -> None:
    q_cache, logits, query_seconds = run_query_on_cache(
        model, query_ids, cache, position_start=position_start, past_len=active_kv_tokens
    )
    generated, decode_seconds = greedy_decode(model, tokenizer, logits, q_cache, decode_steps)
    nll = answer_nll(model, answer_ids, logits, q_cache)
    total = gather_seconds + query_seconds + decode_seconds
    rows.append(
        ResultRow(
            case=case.name,
            method=method,
            context_tokens=len(case.context_ids),
            active_kv_tokens=active_kv_tokens,
            query_tokens=int(query_ids.shape[1]),
            selected_units=json.dumps(selected_units, ensure_ascii=False),
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
        case = make_case(tokenizer, case_name, config.prompt_tokens, config.page_tokens, config.recent_tokens)
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
        full_generated, full_decode = greedy_decode(model, tokenizer, full_logits, full_q_cache, config.decode_steps)
        full_nll = answer_nll(model, answer_ids, full_logits, full_q_cache)
        full_online = full_query_seconds + full_decode
        rows.append(
            ResultRow(
                case=case.name,
                method="full_kv_cache",
                context_tokens=len(case.context_ids),
                active_kv_tokens=len(case.context_ids),
                query_tokens=int(query_ids.shape[1]),
                selected_units="all",
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
        evidence_page_indices = page_token_indices(list(case.evidence_pages), len(case.context_ids), config.page_tokens)
        sparse_recent_plus = unique_ordered(evidence_page_indices + recent)
        prefix_end_page = max(case.evidence_pages) if case.evidence_pages else -1
        prefix_indices = list(range(0, min(len(case.context_ids), (prefix_end_page + 1) * config.page_tokens)))
        prefix_recent_plus = unique_ordered(prefix_indices + recent) if prefix_indices else recent
        recent_only = recent

        kv_methods = [
            ("kv_native_recent_only", recent_only, {"recent": True}),
            (
                "kv_native_recent_plus_sparse_pages_absolute_pos",
                sparse_recent_plus,
                {"old_pages": list(case.evidence_pages), "recent": True, "position": "absolute"},
            ),
            (
                "kv_native_recent_plus_sparse_pages_compact_pos",
                sparse_recent_plus,
                {"old_pages": list(case.evidence_pages), "recent": True, "position": "compact"},
            ),
            (
                "kv_native_prefix_to_evidence_plus_recent",
                prefix_recent_plus,
                {"prefix_to_page": prefix_end_page, "recent": True},
            ),
        ]

        for method, indices_list, selected_units in kv_methods:
            indices = tensor_indices(indices_list, device)
            synchronize()
            start = time.perf_counter()
            compact_cache = cache_from_legacy(gather_cache(full_cache, indices))
            synchronize()
            gather_seconds = time.perf_counter() - start
            compact_position = int(indices.numel())
            absolute_position = len(case.context_ids)
            position_start = compact_position if method.endswith("compact_pos") else absolute_position
            if method == "kv_native_recent_only":
                position_start = len(case.context_ids)
            add_cache_row(
                rows,
                case=case,
                method=method,
                model=model,
                tokenizer=tokenizer,
                query_ids=query_ids,
                answer_ids=answer_ids,
                cache=compact_cache,
                active_kv_tokens=int(indices.numel()),
                selected_units=selected_units,
                position_start=position_start,
                gather_seconds=gather_seconds,
                decode_steps=config.decode_steps,
                full_prefill_seconds=full_prefill,
                full_online_seconds=full_online,
            )

        prompt_indices = sparse_recent_plus
        prompt_text = selected_text(tokenizer, case.context_ids, prompt_indices) + "\n\n" + case.query
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        prompt_cache, prompt_logits, prompt_prefill = prefill(model, prompt_ids)
        prompt_generated, prompt_decode = greedy_decode(model, tokenizer, prompt_logits, prompt_cache, config.decode_steps)
        prompt_nll = answer_nll(model, answer_ids, prompt_logits, prompt_cache)
        prompt_total = prompt_prefill + prompt_decode
        rows.append(
            ResultRow(
                case=case.name,
                method="prompt_rebuild_recent_plus_sparse_text",
                context_tokens=len(case.context_ids),
                active_kv_tokens=int(prompt_ids.shape[1]),
                query_tokens=int(query_ids.shape[1]),
                selected_units=json.dumps({"old_pages": list(case.evidence_pages), "recent": True}, ensure_ascii=False),
                full_prefill_seconds=full_prefill,
                method_prefill_seconds=prompt_prefill,
                gather_seconds=0.0,
                query_seconds=0.0,
                decode_seconds=prompt_decode,
                total_online_seconds=prompt_total,
                answer_nll=prompt_nll,
                answer_exact=case.answer in prompt_generated,
                speedup_vs_full_online=full_online / prompt_total if prompt_total > 0 else 0.0,
                generated=prompt_generated.replace("\n", " ")[:240],
            )
        )

        case_meta.append(
            {
                "case": case.name,
                "answer": case.answer,
                "evidence_pages": case.evidence_pages,
                "recent_record": case.recent_record,
                "context_tokens": len(case.context_ids),
                "query": case.query,
            }
        )

    row_dicts = [asdict(row) for row in rows]
    write_csv(output_dir / "results.csv", row_dicts)
    (output_dir / "summary.json").write_text(
        json.dumps({"config": asdict(config), "cases": case_meta, "rows": row_dicts}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("case,method,active_kv,total_online,speedup,nll,exact,generated")
    for row in rows:
        print(
            f"{row.case},{row.method},{row.active_kv_tokens},{row.total_online_seconds:.4f},"
            f"{row.speedup_vs_full_online:.3f},{row.answer_nll:.4f},{row.answer_exact},"
            f"{row.generated[:100]}"
        )
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
