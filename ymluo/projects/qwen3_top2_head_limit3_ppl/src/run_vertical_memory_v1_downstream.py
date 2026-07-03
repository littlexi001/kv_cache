from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    clone_past_key_values,
    model_forward,
    pick_input_device,
    prefill_cache,
    resolve_dtype,
)
from run_qabs_downstream_kv_retrieval import LABELS  # noqa: E402
from run_qabs_downstream_task_suite import BUILDERS  # noqa: E402


STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "are",
    "as",
    "at",
    "be",
    "by",
    "class",
    "file",
    "find",
    "for",
    "from",
    "has",
    "in",
    "is",
    "its",
    "key",
    "label",
    "lookup",
    "maps",
    "of",
    "on",
    "option",
    "or",
    "read",
    "row",
    "status",
    "the",
    "to",
    "using",
    "what",
    "with",
}
CURRENT_WORDS = {"active", "approved", "current", "final", "latest", "live", "primary", "valid"}
STALE_WORDS = {"deprecated", "earlier", "expired", "non_current", "old", "obsolete", "previous", "revoked", "stale", "withdrawn"}


@dataclass(frozen=True)
class Config:
    model_name_or_path: str
    output_dir: str
    variants: str
    tasks_per_variant: int
    records_per_task: int
    seed: int
    chunk_size: int
    dtype: str
    device: str
    device_map: str
    attn_implementation: str
    max_pages: int
    page_char_budget: int
    log_every: int


@dataclass
class MemoryPage:
    page_id: int
    text: str
    entities: set[str]
    keywords: Counter[str]
    label: str
    status: str
    structural_score: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Vertical-memory v1 downstream smoke test.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--variants",
        default="structured_noisy,compact_kv,natural_kv,json_kv,needle_sentence,topic_table",
    )
    parser.add_argument("--tasks_per_variant", type=int, default=8)
    parser.add_argument("--records_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026070103)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_pages", type=int, default=1)
    parser.add_argument("--page_char_budget", type=int, default=520)
    parser.add_argument("--log_every", type=int, default=8)
    return Config(**vars(parser.parse_args()))


@torch.inference_mode()
def run_prefix(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    text: str,
) -> tuple[Any, torch.Tensor]:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device)
    for pos in range(ids.shape[-1]):
        token = ids[:, pos : pos + 1]
        outputs = model_forward(
            model,
            {
                "input_ids": token,
                "past_key_values": past_key_values,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
            },
        )
        past_key_values = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :].detach()
    return past_key_values, prev_logits


@torch.inference_mode()
def score_option(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    option: str,
) -> float:
    ids = tokenizer(" " + option, return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device)
    total = 0.0
    for pos in range(ids.shape[-1]):
        token = ids[:, pos : pos + 1]
        total += float(-F.cross_entropy(prev_logits.float(), token.reshape(-1), reduction="sum").item())
        outputs = model_forward(
            model,
            {
                "input_ids": token,
                "past_key_values": past_key_values,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
            },
        )
        past_key_values = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :].detach()
    return total


def empty_cache(input_device: torch.device) -> tuple[Any, torch.Tensor]:
    return None, torch.empty((1, 0), device=input_device)


def extract_label_from_text(text: str) -> str:
    patterns = [
        r"ANSWER_LABEL\s*=\s*([A-D])",
        r"answer_label[\"']?\s*[:=]\s*[\"']?([A-D])",
        r"\bclass\s*=\s*([A-D])",
        r"\boption\s+([A-D])\b",
        r"\blabel\s+([A-D])\b",
        r"=>\s*([A-D])\b",
        r"\|\s*([A-D])\s*(?:\||$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def extract_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for match in re.finditer(r"\b[A-Z][A-Z0-9_-]{2,}\b", text):
        entities.add(match.group(0))
    for match in re.finditer(r"\bK\d{6}-[A-Z0-9]{4,}\b", text):
        entities.add(match.group(0))
    for match in re.finditer(r"\b(?:key|id|lookup key)\s*[=:]?\s*([A-Za-z0-9_-]{5,})", text, flags=re.IGNORECASE):
        entities.add(match.group(1))
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text):
        span = match.group(0)
        if span.lower() not in STOPWORDS:
            entities.add(span)
    return entities


def extract_keywords(text: str) -> Counter[str]:
    words = [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)]
    return Counter(word for word in words if word not in STOPWORDS)


def detect_status(text: str) -> str:
    lowered = text.lower()
    stale = sum(1 for word in STALE_WORDS if word in lowered)
    current = sum(1 for word in CURRENT_WORDS if word in lowered)
    if stale > current:
        return "non_current"
    if current > 0:
        return "current"
    return "unknown"


def structural_score(text: str) -> int:
    return (
        text.count("|")
        + text.count("{")
        + text.count("}")
        + text.count(":")
        + text.count("=")
        + text.count("=>")
        + len(re.findall(r"^\s*(?:[-*]|\d+[.)])\s+", text, flags=re.MULTILINE))
    )


def split_vertical_pages(context: str, char_budget: int) -> list[str]:
    raw_units: list[str] = []
    for block in re.split(r"\n\s*\n+", context):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) > 1:
            raw_units.extend(lines)
        else:
            raw_units.extend(part.strip() for part in re.split(r"(?<=[.;!?])\s+", lines[0]) if part.strip())

    pages: list[str] = []
    current: list[str] = []
    current_chars = 0
    for unit in raw_units:
        unit_chars = len(unit)
        has_record_marker = bool(re.search(r"\b(?:key|id|answer_label|class|option|ANSWER_LABEL)\b|=>|\|", unit))
        if current and (current_chars + unit_chars > char_budget or has_record_marker):
            pages.append("\n".join(current))
            current = []
            current_chars = 0
        current.append(unit)
        current_chars += unit_chars + 1
    if current:
        pages.append("\n".join(current))
    return pages


def build_pages(context: str, char_budget: int) -> list[MemoryPage]:
    pages = []
    for idx, text in enumerate(split_vertical_pages(context, char_budget)):
        pages.append(
            MemoryPage(
                page_id=idx,
                text=text,
                entities=extract_entities(text),
                keywords=extract_keywords(text),
                label=extract_label_from_text(text),
                status=detect_status(text),
                structural_score=structural_score(text),
            )
        )
    return pages


def target_key_from_query(task: dict[str, Any]) -> str:
    query = task["query"]
    for pattern in [
        r"\bK\d{6}-[A-Z0-9]{4,}\b",
        r"\bid=([A-Za-z0-9_-]{5,})",
        r"\bkey\s+([A-Za-z0-9_-]{5,})",
        r"\blookup key\s+([A-Za-z0-9_-]{5,})",
    ]:
        match = re.search(pattern, query)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return task["target_key"]


def idf_for_pages(pages: list[MemoryPage]) -> dict[str, float]:
    df: Counter[str] = Counter()
    for page in pages:
        for word in page.keywords:
            df[word] += 1
    total = max(1, len(pages))
    return {word: math.log((1 + total) / (1 + count)) + 1.0 for word, count in df.items()}


def route_vertical_pages(task: dict[str, Any], max_pages: int, char_budget: int) -> tuple[list[MemoryPage], dict[str, Any]]:
    pages = build_pages(task["context"], char_budget)
    target_key = target_key_from_query(task)
    query_entities = extract_entities(task["query"]) | {target_key}
    query_keywords = extract_keywords(task["query"])
    idf = idf_for_pages(pages)

    scored: list[tuple[float, MemoryPage]] = []
    for page in pages:
        entity_overlap = len(page.entities & query_entities)
        keyword_score = sum(min(count, page.keywords.get(word, 0)) * idf.get(word, 1.0) for word, count in query_keywords.items())
        score = 0.0
        if target_key and target_key in page.text:
            score += 100.0
        score += 12.0 * entity_overlap
        score += 2.0 * keyword_score
        score += min(4, page.structural_score) * 0.25
        if page.status == "current":
            score += 2.0
        elif page.status == "non_current":
            score -= 8.0
        score += page.page_id / max(1, len(pages)) * 0.05
        scored.append((score, page))

    scored.sort(key=lambda item: (item[0], item[1].page_id), reverse=True)
    selected = [page for _, page in scored[:max_pages]]
    label = ""
    for page in selected:
        if page.label:
            label = page.label
            break
    meta = {
        "routed_pages": len(selected),
        "page_count": len(pages),
        "top_page_id": selected[0].page_id if selected else -1,
        "top_page_score": scored[0][0] if scored else 0.0,
        "evidence_hit": int(any(target_key and target_key in page.text for page in selected)),
        "typed_record_present": int(bool(label)),
        "typed_record_label": label,
    }
    return selected, meta


def exact_answerline_context(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    target_key = task["target_key"]
    lines = [line for line in task["context"].splitlines() if target_key in line]
    label = ""
    for line in lines:
        label = extract_label_from_text(line)
        if label:
            break
    if not label:
        return task["query"], {
            "typed_record_present": 0,
            "typed_record_label": "",
            "routed_pages": 0,
            "page_count": 0,
            "top_page_id": -1,
            "top_page_score": 0.0,
            "evidence_hit": int(bool(lines)),
        }
    evidence = " ".join(lines[:2])
    prompt = (
        "\nTyped memory summary: "
        f"ANSWER_LABEL={label}; status=current. "
        f"Lookup key {target_key} maps to option {label}. "
        "Use the current status only.\n"
    )
    if evidence:
        prompt += f"Evidence page: {evidence}\n"
    return prompt + task["query"], {
        "typed_record_present": 1,
        "typed_record_label": label,
        "routed_pages": min(2, len(lines)),
        "page_count": len(task["context"].splitlines()),
        "top_page_id": -1,
        "top_page_score": 0.0,
        "evidence_hit": 1,
    }


def vertical_memory_context(task: dict[str, Any], max_pages: int, char_budget: int) -> tuple[str, dict[str, Any]]:
    selected, meta = route_vertical_pages(task, max_pages, char_budget)
    label = meta["typed_record_label"]
    target_key = target_key_from_query(task)
    prompt = "\nTyped vertical memory summary:\n"
    if label:
        prompt += (
            f"ANSWER_LABEL={label}; status=current_or_unknown; "
            f"target_key={target_key}; primary_page={meta['top_page_id']}.\n"
        )
    else:
        prompt += "No explicit answer label was found in the routed pages.\n"
    for idx, page in enumerate(selected):
        one_line = re.sub(r"\s+", " ", page.text).strip()
        prefix = "Primary evidence page" if idx == 0 else "Auxiliary evidence page"
        prompt += (
            f"{prefix} {page.page_id}: status={page.status}; "
            f"label={page.label or 'unknown'}; evidence={one_line[:420]}\n"
        )
    prompt += "Use the primary evidence page, prefer current status, and answer with the option label.\n"
    return prompt + task["query"], meta


def evaluate_prompt(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    context_cache: Any,
    context_prev: torch.Tensor,
    prompt_suffix: str,
) -> tuple[str, dict[str, float], float]:
    started = time.perf_counter()
    query_cache, query_prev = run_prefix(
        model,
        tokenizer,
        input_device,
        clone_past_key_values(context_cache) if context_cache is not None else None,
        context_prev.detach().clone() if context_prev.numel() else context_prev,
        prompt_suffix,
    )
    scores: dict[str, float] = {}
    for label in LABELS:
        scores[label] = score_option(
            model,
            tokenizer,
            input_device,
            clone_past_key_values(query_cache),
            query_prev.detach().clone(),
            label,
        )
    elapsed = time.perf_counter() - started
    return max(scores, key=scores.get), scores, elapsed


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    keys = sorted({(row["variant"], row["mode"]) for row in rows})
    for variant, mode in keys:
        subset = [row for row in rows if row["variant"] == variant and row["mode"] == mode]
        tasks = max(1, len(subset))
        summary.append(
            {
                "variant": variant,
                "mode": mode,
                "tasks": len(subset),
                "accuracy": sum(int(row["correct"]) for row in subset) / tasks,
                "mean_eval_seconds": sum(float(row["eval_seconds"]) for row in subset) / tasks,
                "mean_total_seconds": sum(float(row["total_seconds"]) for row in subset) / tasks,
                "mean_visible_tokens": sum(int(row["visible_tokens"]) for row in subset) / tasks,
                "typed_record_coverage": sum(int(row.get("typed_record_present", 0)) for row in subset) / tasks,
                "evidence_hit_rate": sum(int(row.get("evidence_hit", 0)) for row in subset) / tasks,
                "mean_page_count": sum(int(row.get("page_count", 0)) for row in subset) / tasks,
                "mean_routed_pages": sum(int(row.get("routed_pages", 0)) for row in subset) / tasks,
            }
        )
    return summary


def append_row(
    rows: list[dict[str, Any]],
    task: dict[str, Any],
    mode: str,
    pred: str,
    scores: dict[str, float],
    eval_seconds: float,
    prefill_seconds: float,
    visible_tokens: int,
    meta: dict[str, Any],
) -> None:
    rows.append(
        {
            "variant": task["variant"],
            "task_id": task["task_id"],
            "mode": mode,
            "target_key": task["target_key"],
            "target_label": task["target_label"],
            "pred_label": pred,
            "correct": int(pred == task["target_label"]),
            "eval_seconds": eval_seconds,
            "prefill_seconds": prefill_seconds,
            "total_seconds": prefill_seconds + eval_seconds,
            "visible_tokens": visible_tokens,
            "typed_record_present": int(meta.get("typed_record_present", 0)),
            "typed_record_label": meta.get("typed_record_label", ""),
            "routed_pages": int(meta.get("routed_pages", 0)),
            "page_count": int(meta.get("page_count", 0)),
            "top_page_id": int(meta.get("top_page_id", -1)),
            "top_page_score": float(meta.get("top_page_score", 0.0)),
            "evidence_hit": int(meta.get("evidence_hit", 0)),
            **{f"score_{label}": scores[label] for label in LABELS},
        }
    )


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    variants = [name.strip() for name in config.variants.split(",") if name.strip()]
    unknown = [name for name in variants if name not in BUILDERS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available={sorted(BUILDERS)}")

    device = torch.device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if config.device_map:
        load_kwargs["device_map"] = config.device_map
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, device)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for variant_index, variant in enumerate(variants):
        rng = random.Random(config.seed + 1009 * variant_index)
        tasks = [BUILDERS[variant](rng, idx, config.records_per_task) for idx in range(config.tasks_per_variant)]
        for task_idx, task in enumerate(tasks, start=1):
            if task_idx == 1 or task_idx == len(tasks) or task_idx % config.log_every == 0:
                print(f"{variant} task {task_idx}/{len(tasks)}", flush=True)

            context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            prefill_started = time.perf_counter()
            context_cache, context_prev = prefill_cache(
                model,
                context_ids,
                context_ids.shape[-1],
                config.chunk_size,
                input_device,
            )
            prefill_seconds = time.perf_counter() - prefill_started
            pred, scores, elapsed = evaluate_prompt(model, tokenizer, input_device, context_cache, context_prev, task["query"])
            append_row(
                rows,
                task,
                "full_baseline",
                pred,
                scores,
                elapsed,
                prefill_seconds,
                int(context_ids.shape[-1]),
                {
                    "typed_record_present": 0,
                    "typed_record_label": "",
                    "routed_pages": 0,
                    "page_count": 0,
                    "top_page_id": -1,
                    "top_page_score": 0.0,
                    "evidence_hit": 0,
                },
            )

            for mode, prompt_builder in [
                ("exact_answerline_adapter", lambda item: exact_answerline_context(item)),
                ("vertical_memory_v1", lambda item: vertical_memory_context(item, config.max_pages, config.page_char_budget)),
            ]:
                prompt, meta = prompt_builder(task)
                prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
                cache, prev = empty_cache(input_device)
                pred, scores, elapsed = evaluate_prompt(model, tokenizer, input_device, cache, prev, prompt)
                append_row(
                    rows,
                    task,
                    mode,
                    pred,
                    scores,
                    elapsed,
                    0.0,
                    int(prompt_ids.shape[-1]),
                    meta,
                )

    fields = [
        "variant",
        "task_id",
        "mode",
        "target_key",
        "target_label",
        "pred_label",
        "correct",
        "eval_seconds",
        "prefill_seconds",
        "total_seconds",
        "visible_tokens",
        "typed_record_present",
        "typed_record_label",
        "routed_pages",
        "page_count",
        "top_page_id",
        "top_page_score",
        "evidence_hit",
    ] + [f"score_{label}" for label in LABELS]
    with (output_dir / "vertical_memory_v1_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize(rows)
    with (output_dir / "vertical_memory_v1_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    summary = {"seconds": time.perf_counter() - started, "summary": summary_rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
