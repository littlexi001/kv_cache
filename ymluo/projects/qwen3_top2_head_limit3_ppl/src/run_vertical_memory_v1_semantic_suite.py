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
from run_vertical_memory_v1_downstream import MemoryPage, detect_status, extract_entities, extract_keywords, idf_for_pages  # noqa: E402


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
    max_pages: int
    page_char_budget: int
    log_every: int


TOPICS = [
    ("astronomy", "telescope calibration", "spectral drift", "observatory schedule"),
    ("medicine", "trial cohort", "dosage response", "patient monitoring"),
    ("finance", "risk exposure", "liquidity buffer", "portfolio review"),
    ("law", "contract clause", "appeal deadline", "court filing"),
    ("ecology", "river survey", "habitat recovery", "species count"),
    ("robotics", "navigation stack", "sensor fusion", "field test"),
]
PROJECTS = ["Orion", "Lyra", "Vega", "Nereid", "Ibis", "Calypso", "Helix", "Marlow"]
COLORS = ["blue", "green", "silver", "amber", "violet", "white"]
OUTCOMES = [
    ("postpone the launch", "because the control test failed twice"),
    ("approve the deployment", "because the safety audit passed"),
    ("repeat the measurement", "because the sensor log was inconsistent"),
    ("archive the proposal", "because the budget owner withdrew support"),
    ("expand the pilot", "because retention improved after the change"),
    ("freeze the release", "because two critical blockers remained open"),
]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Vertical-memory v1 on non-KV semantic downstream tasks.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", default="topic_page,attribute_page,causal_page")
    parser.add_argument("--tasks_per_variant", type=int, default=10)
    parser.add_argument("--distractor_pages", type=int, default=18)
    parser.add_argument("--seed", type=int, default=2026070201)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_pages", type=int, default=1)
    parser.add_argument("--page_char_budget", type=int, default=620)
    parser.add_argument("--log_every", type=int, default=5)
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
def score_completion(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    completion: str,
) -> tuple[float, int]:
    ids = tokenizer(" " + completion, return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device)
    total_loss = 0.0
    for pos in range(ids.shape[-1]):
        token = ids[:, pos : pos + 1]
        total_loss += float(F.cross_entropy(prev_logits.float(), token.reshape(-1), reduction="sum").item())
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
    return total_loss, int(ids.shape[-1])


def choose_options(rng: random.Random, correct: str, pool: list[str]) -> tuple[dict[str, str], str]:
    negatives = [item for item in pool if item != correct]
    rng.shuffle(negatives)
    values = [correct] + negatives[:3]
    rng.shuffle(values)
    choices = {label: value for label, value in zip(LABELS, values)}
    target_label = next(label for label, value in choices.items() if value == correct)
    return choices, target_label


def format_query(question: str, choices: dict[str, str]) -> str:
    option_text = "\n".join(f"{label}. {choices[label]}" for label in LABELS)
    return f"\nQuestion: {question}\n{option_text}\nAnswer with the option letter only:"


def distractor_page(rng: random.Random, idx: int, project: str | None = None) -> str:
    topic, phrase, signal, schedule = rng.choice(TOPICS)
    name = project if project and rng.random() < 0.25 else rng.choice(PROJECTS)
    color = rng.choice(COLORS)
    outcome, reason = rng.choice(OUTCOMES)
    status = "Earlier note" if rng.random() < 0.35 else "Background note"
    return (
        f"{status} {idx}. Project {name} discussed {phrase} in the {topic} file. "
        f"The page mentioned {signal}, a {color} badge, and a tentative action to {outcome} {reason}. "
        f"This note is background material and should not answer a later requested page."
    )


def build_topic_page_task(rng: random.Random, task_idx: int, distractors: int) -> dict[str, Any]:
    project = rng.choice(PROJECTS)
    topic, phrase, signal, schedule = rng.choice(TOPICS)
    choices, target_label = choose_options(rng, topic, [item[0] for item in TOPICS])
    target_page = (
        f"Current briefing for Project {project}. The report is about {topic}. "
        f"It emphasizes {phrase}, mentions {signal}, and uses {schedule} as the organizing detail. "
        "This current briefing supersedes earlier background notes."
    )
    pages = [distractor_page(rng, idx, project) for idx in range(distractors)] + [target_page]
    rng.shuffle(pages)
    context = "\n\n".join(pages) + "\n"
    query = format_query(f"Which subject area is the current briefing for Project {project} about?", choices)
    return {
        "variant": "topic_page",
        "task_id": task_idx,
        "context": context,
        "query": query,
        "target_label": target_label,
        "gold_answer": topic,
        "target_page": target_page,
    }


def build_attribute_page_task(rng: random.Random, task_idx: int, distractors: int) -> dict[str, Any]:
    project = rng.choice(PROJECTS)
    color = rng.choice(COLORS)
    topic, phrase, signal, _ = rng.choice(TOPICS)
    choices, target_label = choose_options(rng, color, COLORS)
    stale_color = rng.choice([item for item in COLORS if item != color])
    stale_page = (
        f"Old profile for Project {project}. The badge color was once {stale_color}, "
        "but this earlier profile was retired after the registry update."
    )
    target_page = (
        f"Current profile for Project {project}. The active badge color is {color}. "
        f"The profile connects the badge to {phrase} and tracks {signal} in the {topic} notes."
    )
    pages = [distractor_page(rng, idx, project) for idx in range(distractors)] + [stale_page, target_page]
    rng.shuffle(pages)
    context = "\n\n".join(pages) + "\n"
    query = format_query(f"What is the active badge color for Project {project}?", choices)
    return {
        "variant": "attribute_page",
        "task_id": task_idx,
        "context": context,
        "query": query,
        "target_label": target_label,
        "gold_answer": color,
        "target_page": target_page,
    }


def build_causal_page_task(rng: random.Random, task_idx: int, distractors: int) -> dict[str, Any]:
    project = rng.choice(PROJECTS)
    outcome, reason = rng.choice(OUTCOMES)
    choices, target_label = choose_options(rng, outcome, [item[0] for item in OUTCOMES])
    topic, phrase, signal, _ = rng.choice(TOPICS)
    target_page = (
        f"Current decision memo for Project {project}. The team should {outcome}. "
        f"The reason is that {reason}. The memo also cites {phrase} and {signal} from the {topic} review."
    )
    pages = [distractor_page(rng, idx, project) for idx in range(distractors)] + [target_page]
    rng.shuffle(pages)
    context = "\n\n".join(pages) + "\n"
    query = format_query(f"According to the current decision memo for Project {project}, what should the team do?", choices)
    return {
        "variant": "causal_page",
        "task_id": task_idx,
        "context": context,
        "query": query,
        "target_label": target_label,
        "gold_answer": outcome,
        "target_page": target_page,
    }


BUILDERS = {
    "topic_page": build_topic_page_task,
    "attribute_page": build_attribute_page_task,
    "causal_page": build_causal_page_task,
}


def empty_cache(input_device: torch.device) -> tuple[Any, torch.Tensor]:
    return None, torch.empty((1, 0), device=input_device)


def build_paragraph_pages(context: str) -> list[MemoryPage]:
    pages: list[MemoryPage] = []
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n+", context) if block.strip()]
    for idx, text in enumerate(paragraphs):
        pages.append(
            MemoryPage(
                page_id=idx,
                text=text,
                entities=extract_entities(text),
                keywords=extract_keywords(text),
                label="",
                status=detect_status(text),
                structural_score=text.count(".") + text.count(";") + text.count(":"),
            )
        )
    return pages


def semantic_route_pages(task: dict[str, Any], max_pages: int, char_budget: int) -> tuple[list[MemoryPage], dict[str, Any]]:
    del char_budget
    pages = build_paragraph_pages(task["context"])
    query_entities = extract_entities(task["query"])
    query_keywords = extract_keywords(task["query"])
    idf = idf_for_pages(pages)
    scored: list[tuple[float, MemoryPage]] = []
    for page in pages:
        entity_overlap = len(page.entities & query_entities)
        keyword_score = sum(min(count, page.keywords.get(word, 0)) * idf.get(word, 1.0) for word, count in query_keywords.items())
        score = 14.0 * entity_overlap + 2.5 * keyword_score
        if page.status == "current":
            score += 3.0
        elif page.status == "non_current":
            score -= 7.0
        if "current" in page.text.lower() and "current" in task["query"].lower():
            score += 2.0
        scored.append((score, page))
    scored.sort(key=lambda item: (item[0], item[1].page_id), reverse=True)
    selected = [page for _, page in scored[:max_pages]]
    target_page = task["target_page"]
    meta = {
        "routed_pages": len(selected),
        "page_count": len(pages),
        "top_page_id": selected[0].page_id if selected else -1,
        "top_page_score": scored[0][0] if scored else 0.0,
        "evidence_hit": int(any(target_page in page.text for page in selected)),
    }
    return selected, meta


def vertical_prompt(task: dict[str, Any], max_pages: int, char_budget: int) -> tuple[str, dict[str, Any]]:
    selected, meta = semantic_route_pages(task, max_pages, char_budget)
    prompt = "\nSemantic vertical memory pages:\n"
    for idx, page in enumerate(selected):
        one_line = re.sub(r"\s+", " ", page.text).strip()
        prefix = "Primary evidence page" if idx == 0 else "Auxiliary evidence page"
        prompt += f"{prefix} {page.page_id}: status={detect_status(page.text)}; evidence={one_line[:520]}\n"
    prompt += "Use only the routed memory pages above and answer the question with the option letter.\n"
    return prompt + task["query"], meta


def score_task(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    context_cache: Any,
    context_prev: torch.Tensor,
    prompt_suffix: str,
    target_label: str,
) -> tuple[str, dict[str, float], float, float, int, float]:
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
    gold_loss = 0.0
    gold_tokens = 0
    for label in LABELS:
        loss, tokens = score_completion(
            model,
            tokenizer,
            input_device,
            clone_past_key_values(query_cache),
            query_prev.detach().clone(),
            label,
        )
        scores[label] = -loss
        if label == target_label:
            gold_loss = loss
            gold_tokens = tokens
    elapsed = time.perf_counter() - started
    gold_mean_loss = gold_loss / max(1, gold_tokens)
    return max(scores, key=scores.get), scores, elapsed, gold_mean_loss, gold_tokens, math.exp(min(gold_mean_loss, 80.0))


def append_row(
    rows: list[dict[str, Any]],
    task: dict[str, Any],
    mode: str,
    pred: str,
    scores: dict[str, float],
    eval_seconds: float,
    prefill_seconds: float,
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
            "target_label": task["target_label"],
            "gold_answer": task["gold_answer"],
            "pred_label": pred,
            "correct": int(pred == task["target_label"]),
            "eval_seconds": eval_seconds,
            "prefill_seconds": prefill_seconds,
            "total_seconds": eval_seconds + prefill_seconds,
            "visible_tokens": visible_tokens,
            "gold_label_loss": gold_loss,
            "gold_label_tokens": gold_tokens,
            "gold_label_ppl": gold_ppl,
            "routed_pages": int(meta.get("routed_pages", 0)),
            "page_count": int(meta.get("page_count", 0)),
            "top_page_id": int(meta.get("top_page_id", -1)),
            "top_page_score": float(meta.get("top_page_score", 0.0)),
            "evidence_hit": int(meta.get("evidence_hit", 0)),
            **{f"score_{label}": scores[label] for label in LABELS},
        }
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    keys = sorted({(row["variant"], row["mode"]) for row in rows})
    for variant, mode in keys:
        subset = [row for row in rows if row["variant"] == variant and row["mode"] == mode]
        tasks = max(1, len(subset))
        total_loss = sum(float(row["gold_label_loss"]) * int(row["gold_label_tokens"]) for row in subset)
        total_tokens = sum(int(row["gold_label_tokens"]) for row in subset)
        mean_loss = total_loss / max(1, total_tokens)
        summary.append(
            {
                "variant": variant,
                "mode": mode,
                "tasks": len(subset),
                "accuracy": sum(int(row["correct"]) for row in subset) / tasks,
                "gold_label_loss": mean_loss,
                "gold_label_ppl": math.exp(min(mean_loss, 80.0)),
                "mean_eval_seconds": sum(float(row["eval_seconds"]) for row in subset) / tasks,
                "mean_total_seconds": sum(float(row["total_seconds"]) for row in subset) / tasks,
                "mean_visible_tokens": sum(int(row["visible_tokens"]) for row in subset) / tasks,
                "evidence_hit_rate": sum(int(row.get("evidence_hit", 0)) for row in subset) / tasks,
                "mean_page_count": sum(int(row.get("page_count", 0)) for row in subset) / tasks,
                "mean_routed_pages": sum(int(row.get("routed_pages", 0)) for row in subset) / tasks,
            }
        )
    return summary


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
        tasks = [BUILDERS[variant](rng, idx, config.distractor_pages) for idx in range(config.tasks_per_variant)]
        for idx, task in enumerate(tasks, start=1):
            if idx == 1 or idx == len(tasks) or idx % config.log_every == 0:
                print(f"{variant} task {idx}/{len(tasks)}", flush=True)

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
            pred, scores, elapsed, loss, tokens, ppl = score_task(
                model,
                tokenizer,
                input_device,
                context_cache,
                context_prev,
                task["query"],
                task["target_label"],
            )
            append_row(
                rows,
                task,
                "full_baseline",
                pred,
                scores,
                elapsed,
                prefill_seconds,
                int(context_ids.shape[-1]),
                loss,
                tokens,
                ppl,
                {},
            )

            prompt, meta = vertical_prompt(task, config.max_pages, config.page_char_budget)
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
            cache, prev = empty_cache(input_device)
            pred, scores, elapsed, loss, tokens, ppl = score_task(
                model,
                tokenizer,
                input_device,
                cache,
                prev,
                prompt,
                task["target_label"],
            )
            append_row(
                rows,
                task,
                "vertical_memory_v1",
                pred,
                scores,
                elapsed,
                0.0,
                int(prompt_ids.shape[-1]),
                loss,
                tokens,
                ppl,
                meta,
            )

    fields = [
        "variant",
        "task_id",
        "mode",
        "target_label",
        "gold_answer",
        "pred_label",
        "correct",
        "eval_seconds",
        "prefill_seconds",
        "total_seconds",
        "visible_tokens",
        "gold_label_loss",
        "gold_label_tokens",
        "gold_label_ppl",
        "routed_pages",
        "page_count",
        "top_page_id",
        "top_page_score",
        "evidence_hit",
    ] + [f"score_{label}" for label in LABELS]
    with (output_dir / "semantic_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize(rows)
    with (output_dir / "semantic_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    summary = {"seconds": time.perf_counter() - started, "summary": summary_rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
