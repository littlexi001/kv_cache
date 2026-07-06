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
    text_path: str
    prompt_tokens: int
    page_tokens: int
    top_k: int
    query: str
    decode_steps: int
    dtype: str
    attn_implementation: str
    seed: int
    synthetic_exact: bool
    synthetic_key: str
    synthetic_answer: str
    synthetic_insert_page: int
    answer_text: str
    force_pages: str
    query_position_mode: str


@dataclass
class ResultRow:
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
    generated: str
    answer_nll: float
    answer_exact: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Smoke test KV-page gather decode without prompt re-prefill.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--prompt_tokens", type=int, default=8192)
    parser.add_argument("--page_tokens", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument(
        "--query",
        default="Question: Summarize the main events and answer briefly.\nAnswer:",
    )
    parser.add_argument("--decode_steps", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=2026070502)
    parser.add_argument("--synthetic_exact", action="store_true")
    parser.add_argument("--synthetic_key", default="MAGIC-CODE-7319")
    parser.add_argument("--synthetic_answer", default="ORBITAL-COPPER-284")
    parser.add_argument("--synthetic_insert_page", type=int, default=5)
    parser.add_argument("--answer_text", default="")
    parser.add_argument("--force_pages", default="", help="Comma-separated page ids. Overrides lexical page selection.")
    parser.add_argument("--query_position_mode", choices=["absolute", "compact"], default="absolute")
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


def select_pages(tokenizer: Any, context_ids: list[int], query: str, page_tokens: int, top_k: int) -> list[int]:
    query_terms = set(content_words(query))
    query_numbers = set(re.findall(r"[A-Za-z]*\d+[A-Za-z0-9-]*", query))
    scored = []
    for page_id, start in enumerate(range(0, len(context_ids), page_tokens)):
        ids = context_ids[start : start + page_tokens]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        score = len(query_terms & set(content_words(text)))
        score += 3 * len(query_numbers & set(re.findall(r"[A-Za-z]*\d+[A-Za-z0-9-]*", text)))
        scored.append((score, -page_id, page_id))
    scored.sort(reverse=True)
    selected = [page_id for score, _, page_id in scored[:top_k] if score > 0]
    if not selected:
        selected = [page_id for _, _, page_id in scored[:top_k]]
    return sorted(selected)


def parse_force_pages(force_pages: str) -> list[int] | None:
    if not force_pages.strip():
        return None
    return sorted({int(item.strip()) for item in force_pages.split(",") if item.strip()})


def page_indices(selected_pages: list[int], context_len: int, page_tokens: int, device: torch.device) -> torch.Tensor:
    indices: list[int] = []
    for page_id in selected_pages:
        start = page_id * page_tokens
        end = min(context_len, start + page_tokens)
        indices.extend(range(start, end))
    return torch.tensor(indices, dtype=torch.long, device=device)


def make_context_ids(tokenizer: Any, text: str, config: Config) -> list[int]:
    base_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not config.synthetic_exact:
        return base_ids[: config.prompt_tokens]

    filler = (
        "This is background prose used only to fill the long context. "
        "It contains ordinary narrative details and no special answer. "
    )
    filler_ids = tokenizer(filler, add_special_tokens=False)["input_ids"]
    record = (
        f"\n\nImportant lookup record. The secret key {config.synthetic_key} "
        f"maps to the answer {config.synthetic_answer}. "
        f"If asked about {config.synthetic_key}, answer exactly {config.synthetic_answer}.\n\n"
    )
    record_ids = tokenizer(record, add_special_tokens=False)["input_ids"]
    target = max(config.prompt_tokens, config.page_tokens)
    insert_at = min(config.synthetic_insert_page * config.page_tokens, max(0, target - len(record_ids)))
    ids: list[int] = []
    while len(ids) < target + len(record_ids) + config.page_tokens:
        ids.extend(filler_ids)
    ids = ids[:insert_at] + record_ids + ids[insert_at:]
    return ids[:target]


def gather_cache(cache: Any, indices: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    legacy = legacy_cache(cache)
    gathered = []
    for key, value in legacy:
        gathered.append((key.index_select(2, indices), value.index_select(2, indices)))
    return tuple(gathered)


def cache_from_legacy(legacy: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> Any:
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache.from_legacy_cache(legacy)
    except Exception:
        return legacy


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
            out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
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

    text = Path(config.text_path).read_text(encoding="utf-8", errors="ignore")
    if config.synthetic_exact and config.query == "Question: Summarize the main events and answer briefly.\nAnswer:":
        config = Config(
            **{
                **asdict(config),
                "query": f"Question: What is the answer for key {config.synthetic_key}? Answer exactly.\nAnswer:",
                "answer_text": config.synthetic_answer if not config.answer_text else config.answer_text,
            }
        )
    context_ids = make_context_ids(tokenizer, text, config)
    context_tensor = torch.tensor(context_ids, dtype=torch.long, device=device).view(1, -1)
    query_ids = tokenizer(config.query, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    answer_ids = tokenizer(config.answer_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    selected_pages = parse_force_pages(config.force_pages) or select_pages(
        tokenizer, context_ids, config.query, config.page_tokens, config.top_k
    )
    selected_indices = page_indices(selected_pages, len(context_ids), config.page_tokens, device)

    rows: list[ResultRow] = []

    full_cache, _, full_prefill = prefill(model, context_tensor)
    full_q_cache, full_logits, full_query_seconds = run_query_on_cache(
        model,
        query_ids,
        full_cache,
        context_position_start=len(context_ids),
        past_len=len(context_ids),
    )
    full_generated, full_decode = greedy_decode(model, tokenizer, full_logits, full_q_cache, config.decode_steps)
    full_answer_nll = answer_nll(model, answer_ids, full_logits, full_q_cache) if config.answer_text else 0.0
    rows.append(
        ResultRow(
            method="full_kv_cache",
            context_tokens=len(context_ids),
            active_kv_tokens=len(context_ids),
            query_tokens=int(query_ids.shape[1]),
            selected_pages="all",
            prefill_seconds=full_prefill,
            gather_seconds=0.0,
            query_seconds=full_query_seconds,
            decode_seconds=full_decode,
            total_online_seconds=full_query_seconds + full_decode,
            generated=full_generated.replace("\n", " ")[:300],
            answer_nll=full_answer_nll,
            answer_exact=config.answer_text in full_generated,
        )
    )

    synchronize()
    gather_start = time.perf_counter()
    compact_cache = cache_from_legacy(gather_cache(full_cache, selected_indices))
    synchronize()
    gather_seconds = time.perf_counter() - gather_start
    compact_q_cache, compact_logits, compact_query_seconds = run_query_on_cache(
        model,
        query_ids,
        compact_cache,
        context_position_start=len(context_ids) if config.query_position_mode == "absolute" else int(selected_indices.numel()),
        past_len=int(selected_indices.numel()),
    )
    compact_generated, compact_decode = greedy_decode(model, tokenizer, compact_logits, compact_q_cache, config.decode_steps)
    compact_answer_nll = answer_nll(model, answer_ids, compact_logits, compact_q_cache) if config.answer_text else 0.0
    rows.append(
        ResultRow(
            method="kv_gather_compact",
            context_tokens=len(context_ids),
            active_kv_tokens=int(selected_indices.numel()),
            query_tokens=int(query_ids.shape[1]),
            selected_pages=json.dumps(selected_pages),
            prefill_seconds=0.0,
            gather_seconds=gather_seconds,
            query_seconds=compact_query_seconds,
            decode_seconds=compact_decode,
            total_online_seconds=gather_seconds + compact_query_seconds + compact_decode,
            generated=compact_generated.replace("\n", " ")[:300],
            answer_nll=compact_answer_nll,
            answer_exact=config.answer_text in compact_generated,
        )
    )

    selected_text = "\n\n".join(
        tokenizer.decode(context_ids[page * config.page_tokens : min(len(context_ids), (page + 1) * config.page_tokens)], skip_special_tokens=True)
        for page in selected_pages
    )
    prompt_text = selected_text + "\n\n" + config.query
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    prompt_cache, prompt_logits, prompt_prefill = prefill(model, prompt_ids)
    prompt_generated, prompt_decode = greedy_decode(model, tokenizer, prompt_logits, prompt_cache, config.decode_steps)
    prompt_answer_nll = answer_nll(model, answer_ids, prompt_logits, prompt_cache) if config.answer_text else 0.0
    rows.append(
        ResultRow(
            method="prompt_rebuild_selected_text",
            context_tokens=len(context_ids),
            active_kv_tokens=int(prompt_ids.shape[1]),
            query_tokens=int(query_ids.shape[1]),
            selected_pages=json.dumps(selected_pages),
            prefill_seconds=prompt_prefill,
            gather_seconds=0.0,
            query_seconds=0.0,
            decode_seconds=prompt_decode,
            total_online_seconds=prompt_prefill + prompt_decode,
            generated=prompt_generated.replace("\n", " ")[:300],
            answer_nll=prompt_answer_nll,
            answer_exact=config.answer_text in prompt_generated,
        )
    )

    write_csv(output_dir / "results.csv", [asdict(row) for row in rows])
    (output_dir / "summary.json").write_text(
        json.dumps({"config": asdict(config), "rows": [asdict(row) for row in rows]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("method,context_tokens,active_kv_tokens,prefill,gather,query,decode,total_online,answer_nll,answer_exact,generated")
    full_online = rows[0].total_online_seconds
    for row in rows:
        speedup = full_online / row.total_online_seconds if row.total_online_seconds > 0 else 0.0
        print(
            f"{row.method},{row.context_tokens},{row.active_kv_tokens},"
            f"{row.prefill_seconds:.4f},{row.gather_seconds:.4f},{row.query_seconds:.4f},"
            f"{row.decode_seconds:.4f},{row.total_online_seconds:.4f},speedup={speedup:.3f},"
            f"{row.answer_nll:.4f},{row.answer_exact},"
            f"{row.generated[:80]}"
        )


if __name__ == "__main__":
    main()
