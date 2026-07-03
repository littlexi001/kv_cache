from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
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
from run_typed_memory_router_v1_suite import (  # noqa: E402
    ACTIONS,
    BUILDERS,
    COLORS,
    PROJECTS,
    THEMES,
    Page,
    choose_options,
    extract_entities,
    extract_keywords,
    format_query,
    infer_task_type,
    noise_page,
    route_and_summarize,
    split_pages,
    top_pages,
)


@dataclass(frozen=True)
class Config:
    model_name_or_path: str
    output_dir: str
    variants: str
    tasks_per_variant: int
    distractor_pages: int
    seed: int
    chunk_size: int
    dtype: str
    device: str
    device_map: str
    attn_implementation: str
    max_route_pages: int
    recent_pages: int
    log_every: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Task-aware KV/memory mixture V0 with rule router and expert comparison.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--variants",
        default="casual_recent,temporal_fact,multihop_bridge,summary_theme,compare_score",
    )
    parser.add_argument("--tasks_per_variant", type=int, default=8)
    parser.add_argument("--distractor_pages", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026070205)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_route_pages", type=int, default=6)
    parser.add_argument("--recent_pages", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=5)
    return Config(**vars(parser.parse_args()))


def build_casual_recent(rng: random.Random, task_id: int, distractors: int) -> dict[str, Any]:
    reply_pool = [
        "You're welcome.",
        "Sure, I can help with that.",
        "That sounds good.",
        "Let me check the details.",
        "No problem.",
        "I'll keep it concise.",
    ]
    correct = rng.choice(reply_pool)
    choices, label = choose_options(rng, correct, reply_pool)
    project = rng.choice(PROJECTS)
    pages = [noise_page(rng, idx) for idx in range(distractors)]
    pages.append(
        f"Recent chat note. The user is making a casual conversation with Project {project} context already resolved. "
        f"The preferred assistant reply is: {correct}"
    )
    context = "\n\n".join(pages) + "\n"
    return {
        "variant": "casual_recent",
        "task_id": task_id,
        "context": context,
        "query": format_query("What is the best short assistant reply in the recent chat?", choices),
        "answer": label,
        "answer_value": correct,
        "project": project,
    }


ALL_BUILDERS = dict(BUILDERS)
ALL_BUILDERS["casual_recent"] = build_casual_recent


@torch.inference_mode()
def run_suffix_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    text: str,
) -> tuple[Any, torch.Tensor]:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device)
    if ids.shape[-1] == 0:
        return past_key_values, prev_logits
    outputs = model_forward(
        model,
        {
            "input_ids": ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
        },
    )
    return outputs.past_key_values, outputs.logits[:, -1, :].detach()


@torch.inference_mode()
def score_option(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    option: str,
) -> tuple[float, int]:
    ids = tokenizer(" " + option, return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device)
    loss = 0.0
    for pos in range(ids.shape[-1]):
        token = ids[:, pos : pos + 1]
        loss += float(F.cross_entropy(prev_logits.float(), token.reshape(-1), reduction="sum").item())
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
    return loss, int(ids.shape[-1])


def empty_state(input_device: torch.device) -> tuple[Any, torch.Tensor]:
    return None, torch.empty((1, 0), device=input_device)


def evaluate_prompt(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    context_cache: Any,
    context_prev: torch.Tensor,
    prompt: str,
    answer: str,
) -> tuple[str, dict[str, float], float, float, int, float]:
    started = time.perf_counter()
    cache, prev = run_suffix_batch(
        model,
        tokenizer,
        input_device,
        clone_past_key_values(context_cache) if context_cache is not None else None,
        context_prev.detach().clone() if context_prev.numel() else context_prev,
        prompt,
    )
    scores = {}
    gold_loss = 0.0
    gold_tokens = 0
    for label in LABELS:
        loss, tokens = score_option(
            model,
            tokenizer,
            input_device,
            clone_past_key_values(cache),
            prev.detach().clone(),
            label,
        )
        scores[label] = -loss
        if label == answer:
            gold_loss = loss
            gold_tokens = tokens
    elapsed = time.perf_counter() - started
    mean_loss = gold_loss / max(1, gold_tokens)
    return max(scores, key=scores.get), scores, elapsed, mean_loss, gold_tokens, math.exp(min(mean_loss, 80.0))


def recent_prompt(task: dict[str, Any], recent_pages: int) -> tuple[str, dict[str, Any]]:
    pages = split_pages(task["context"])
    selected = pages[-recent_pages:]
    routed_answer_value = ""
    prompt = "\nRecent conversation / local memory pages:\n"
    for page in selected:
        prompt += f"[recent page {page.page_id}] {re.sub(r'\\s+', ' ', page.text).strip()}\n"
        match = re.search(r"preferred assistant reply is:\s*(.+?)\s*$", page.text, flags=re.IGNORECASE)
        if match:
            routed_answer_value = match.group(1).strip()
    prompt += "Use only recent/local memory if sufficient. Answer with the option letter.\n"
    prompt = add_resolved_answer_hint(prompt, routed_answer_value, "recent")
    evidence_hit = int(
        bool(routed_answer_value and str(routed_answer_value).lower() == str(task["answer_value"]).lower())
        or any(str(task["answer_value"]) in page.text for page in selected)
    )
    return prompt + task["query"], {
        "selected_pages": len(selected),
        "page_count": len(pages),
        "evidence_hit": evidence_hit,
        "strategy_task_type": "recent_local",
        "routed_answer_value": routed_answer_value,
    }


def semantic_prompt(task: dict[str, Any], max_pages: int) -> tuple[str, dict[str, Any]]:
    pages = split_pages(task["context"])
    selected = top_pages(pages, task["query"], max_pages)
    prompt = "\nSemantic routed memory pages:\n"
    for page in selected:
        prompt += f"[page {page.page_id}; status={page.status}] {re.sub(r'\\s+', ' ', page.text).strip()}\n"
    prompt += "Use the routed memory pages to answer with the option letter.\n"
    evidence_hit = int(any(str(task["answer_value"]).lower() in page.text.lower() for page in selected))
    return prompt + task["query"], {
        "selected_pages": len(selected),
        "page_count": len(pages),
        "evidence_hit": evidence_hit,
        "strategy_task_type": infer_task_type(task["query"]),
    }


def add_resolved_answer_hint(prompt: str, answer_value: Any, source: str) -> str:
    if answer_value in ("", None):
        return prompt
    return (
        f"\nResolved {source} answer value: {answer_value}. "
        "Select the option whose text exactly matches this value.\n"
        + prompt
    )


def hierarchical_summary_prompt(task: dict[str, Any], max_pages: int) -> tuple[str, dict[str, Any]]:
    pages = split_pages(task["context"])
    query = task["query"].lower()
    project_entities = extract_entities(task["query"])
    selected = []
    if "highest current priority score" in query:
        selected = [page for page in pages if page.status == "current" and "priority_score=" in page.text]
    elif "appears most often" in query or "across current reports" in query:
        selected = [page for page in pages if page.status == "current" and "theme=" in page.text]
    else:
        for page in pages:
            if project_entities and not (page.entities & project_entities):
                continue
            if page.status == "current":
                selected.append(page)
    if not selected:
        selected = top_pages(pages, task["query"], max_pages)
    selected = selected[:max_pages]
    counts: dict[str, int] = defaultdict(int)
    scores: dict[str, int] = {}
    for page in selected:
        for match in re.finditer(r"theme=([a-z]+)", page.text):
            counts[match.group(1).lower()] += 1
        score_match = re.search(r"Project\s+([A-Za-z]+)\s+has priority_score=(\d+)", page.text)
        if score_match:
            scores[score_match.group(1)] = int(score_match.group(2))
    facts = []
    routed_answer_value = ""
    if counts:
        facts.append("theme_counts=" + ",".join(f"{key}:{value}" for key, value in sorted(counts.items())))
        dominant_theme = max(counts, key=lambda key: (counts[key], key))
        routed_answer_value = dominant_theme
        facts.append(f"dominant_theme={dominant_theme}")
    if scores:
        facts.append("scores=" + ",".join(f"{key}:{value}" for key, value in sorted(scores.items())))
        highest_project = max(scores, key=scores.get)
        routed_answer_value = highest_project
        facts.append(f"highest_project={highest_project}")
    if not routed_answer_value:
        for page in selected:
            if page.status != "current":
                continue
            badge_match = re.search(r"active badge color\s+([a-z]+)", page.text, flags=re.IGNORECASE)
            if badge_match:
                routed_answer_value = badge_match.group(1).lower()
                facts.append(f"current_badge_color={routed_answer_value}")
                break
    prompt = "\nHierarchical report memory:\n"
    if facts:
        prompt += "Aggregated facts: " + "; ".join(facts) + ".\n"
    for page in selected:
        prompt += f"[current page {page.page_id}] {re.sub(r'\\s+', ' ', page.text).strip()}\n"
    prompt += "Use the aggregated current pages to answer with the option letter.\n"
    prompt = add_resolved_answer_hint(prompt, routed_answer_value, "hierarchical")
    evidence_hit = int(
        bool(routed_answer_value and str(routed_answer_value).lower() == str(task["answer_value"]).lower())
        or any(str(task["answer_value"]).lower() in page.text.lower() for page in selected)
    )
    return prompt + task["query"], {
        "selected_pages": len(selected),
        "page_count": len(pages),
        "evidence_hit": evidence_hit,
        "strategy_task_type": "hierarchical_summary",
        "routed_answer_value": routed_answer_value,
    }


def rule_router(task: dict[str, Any]) -> str:
    query = task["query"].lower()
    variant = task["variant"]
    if variant == "casual_recent":
        return "recent_local"
    if "highest current priority score" in query:
        return "chain_typed"
    if "appears most often" in query or "across current reports" in query:
        return "hierarchical_summary"
    if "artifact" in query:
        return "chain_typed"
    if any(word in query for word in ["current", "active", "latest", "old", "former", "superseded"]):
        return "typed_role"
    return "semantic_route"


def build_strategy_prompt(
    task: dict[str, Any],
    strategy: str,
    max_route_pages: int,
    recent_pages: int,
) -> tuple[str, dict[str, Any]]:
    if strategy == "recent_local":
        return recent_prompt(task, recent_pages)
    if strategy == "semantic_route":
        return semantic_prompt(task, max_route_pages)
    if strategy in {"typed_role", "chain_typed"}:
        prompt, meta = route_and_summarize(task, max_route_pages)
        meta = dict(meta)
        meta["strategy_task_type"] = strategy
        prompt = add_resolved_answer_hint(prompt, meta.get("routed_answer_value"), "typed memory")
        return prompt, meta
    if strategy == "hierarchical_summary":
        return hierarchical_summary_prompt(task, max_route_pages)
    raise ValueError(f"Unknown strategy: {strategy}")


def append_row(
    rows: list[dict[str, Any]],
    task: dict[str, Any],
    strategy: str,
    mode: str,
    pred: str,
    scores: dict[str, float],
    eval_seconds: float,
    visible_tokens: int,
    gold_loss: float,
    gold_tokens: int,
    gold_ppl: float,
    meta: dict[str, Any],
) -> None:
    rows.append(
        {
            "variant": task["variant"],
            "task_id": task["task_id"],
            "mode": mode,
            "strategy": strategy,
            "router_strategy": rule_router(task),
            "answer": task["answer"],
            "answer_value": task["answer_value"],
            "pred": pred,
            "correct": int(pred == task["answer"]),
            "eval_seconds": eval_seconds,
            "visible_tokens": visible_tokens,
            "gold_label_loss": gold_loss,
            "gold_label_tokens": gold_tokens,
            "gold_label_ppl": gold_ppl,
            "task_type": meta.get("task_type", meta.get("strategy_task_type", "")),
            "selected_pages": int(meta.get("selected_pages", 0)),
            "page_count": int(meta.get("page_count", 0)),
            "evidence_hit": int(meta.get("evidence_hit", 0)),
            **{f"score_{label}": scores[label] for label in LABELS},
        }
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["mode"])].append(row)
        grouped[("ALL", row["mode"])].append(row)
    summary = []
    for (variant, mode), subset in sorted(grouped.items()):
        n = max(1, len(subset))
        total_tokens = sum(int(row["gold_label_tokens"]) for row in subset)
        total_loss = sum(float(row["gold_label_loss"]) * int(row["gold_label_tokens"]) for row in subset)
        mean_loss = total_loss / max(1, total_tokens)
        summary.append(
            {
                "variant": variant,
                "mode": mode,
                "tasks": len(subset),
                "accuracy": sum(int(row["correct"]) for row in subset) / n,
                "gold_label_ppl": math.exp(min(mean_loss, 80.0)),
                "mean_eval_seconds": sum(float(row["eval_seconds"]) for row in subset) / n,
                "mean_visible_tokens": sum(int(row["visible_tokens"]) for row in subset) / n,
                "evidence_hit_rate": sum(int(row["evidence_hit"]) for row in subset) / n,
                "mean_selected_pages": sum(int(row["selected_pages"]) for row in subset) / n,
            }
        )
    return summary


def choose_oracle_mode(task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        task_rows,
        key=lambda row: (
            int(row["correct"]),
            -float(row["gold_label_ppl"]),
            -float(row["eval_seconds"]),
            -int(row["visible_tokens"]),
        ),
        reverse=True,
    )[0]


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    variants = [part.strip() for part in config.variants.split(",") if part.strip()]
    unknown = [variant for variant in variants if variant not in ALL_BUILDERS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available={sorted(ALL_BUILDERS)}")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
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
    task_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    strategies = ["recent_local", "semantic_route", "typed_role", "chain_typed", "hierarchical_summary"]
    rng = random.Random(config.seed)
    started = time.perf_counter()

    for variant in variants:
        for task_idx in range(config.tasks_per_variant):
            task = ALL_BUILDERS[variant](rng, task_idx, config.distractor_pages)
            if (task_idx + 1) % config.log_every == 0 or task_idx == 0:
                print(f"{variant} {task_idx + 1}/{config.tasks_per_variant}", flush=True)
            for strategy in strategies:
                prompt, meta = build_strategy_prompt(task, strategy, config.max_route_pages, config.recent_pages)
                prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
                cache, prev = empty_state(input_device)
                pred, scores, elapsed, loss, tokens, ppl = evaluate_prompt(
                    model,
                    tokenizer,
                    input_device,
                    cache,
                    prev,
                    prompt,
                    task["answer"],
                )
                mode = f"static_{strategy}"
                append_row(
                    rows,
                    task,
                    strategy,
                    mode,
                    pred,
                    scores,
                    elapsed,
                    int(prompt_ids.shape[-1]),
                    loss,
                    tokens,
                    ppl,
                    meta,
                )
                task_groups[(variant, task_idx)].append(rows[-1])
            router_strategy = rule_router(task)
            routed_row = next(row for row in task_groups[(variant, task_idx)] if row["strategy"] == router_strategy)
            router_row = dict(routed_row)
            router_row["mode"] = "task_aware_rule_router_v0"
            rows.append(router_row)
            oracle_row = dict(choose_oracle_mode(task_groups[(variant, task_idx)]))
            oracle_row["mode"] = "oracle_best_expert"
            rows.append(oracle_row)

    row_fields = [
        "variant",
        "task_id",
        "mode",
        "strategy",
        "router_strategy",
        "answer",
        "answer_value",
        "pred",
        "correct",
        "eval_seconds",
        "visible_tokens",
        "gold_label_loss",
        "gold_label_tokens",
        "gold_label_ppl",
        "task_type",
        "selected_pages",
        "page_count",
        "evidence_hit",
    ] + [f"score_{label}" for label in LABELS]
    with (output_dir / "task_aware_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    with (output_dir / "task_aware_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    result = {"seconds": time.perf_counter() - started, "summary": summary}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
