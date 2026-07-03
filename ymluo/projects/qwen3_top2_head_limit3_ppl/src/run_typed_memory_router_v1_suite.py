from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
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


PROJECTS = ["Orion", "Lyra", "Vega", "Nereid", "Ibis", "Calypso", "Helix", "Marlow"]
COLORS = ["blue", "green", "silver", "amber", "violet", "white"]
THEMES = ["safety", "latency", "budget", "quality", "coverage", "security"]
ACTIONS = ["approve deployment", "repeat measurement", "freeze release", "expand pilot", "archive proposal", "postpone launch"]
ARTIFACTS = ["RIVER-ALPHA", "RIVER-BETA", "RIVER-GAMMA", "RIVER-DELTA", "RIVER-KAPPA", "RIVER-OMEGA"]
NOISE_TOPICS = [
    "astronomy telescope calibration and spectral drift",
    "clinical dosage response and patient monitoring",
    "liquidity buffer risk exposure and portfolio review",
    "contract clause appeal deadline and court filing",
    "river survey habitat recovery and species count",
    "robotics navigation stack sensor fusion and field test",
]


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


@dataclass
class Page:
    page_id: int
    text: str
    entities: set[str]
    keywords: Counter[str]
    status: str


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Typed memory router v1 generic long-context suite.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", default="temporal_fact,multihop_bridge,summary_theme,compare_score")
    parser.add_argument("--tasks_per_variant", type=int, default=12)
    parser.add_argument("--distractor_pages", type=int, default=36)
    parser.add_argument("--seed", type=int, default=2026070203)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_route_pages", type=int, default=6)
    return Config(**vars(parser.parse_args()))


def extract_entities(text: str) -> set[str]:
    entities = set(re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}\b", text))
    for project in PROJECTS:
        if project in text:
            entities.add(project)
    for artifact in ARTIFACTS:
        if artifact in text:
            entities.add(artifact)
    return entities


def extract_keywords(text: str) -> Counter[str]:
    stop = {"the", "and", "for", "with", "from", "this", "that", "what", "which", "should", "project"}
    words = [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)]
    return Counter(word for word in words if word not in stop)


def detect_status(text: str) -> str:
    lower = text.lower()
    stale = any(word in lower for word in ["old", "former", "retired", "obsolete", "superseded", "withdrawn"])
    current = any(word in lower for word in ["current", "active", "latest", "approved", "valid"])
    if stale and not current:
        return "non_current"
    if current:
        return "current"
    return "unknown"


def split_pages(context: str) -> list[Page]:
    pages = []
    for idx, block in enumerate(part.strip() for part in re.split(r"\n\s*\n+", context) if part.strip()):
        pages.append(
            Page(
                page_id=idx,
                text=block,
                entities=extract_entities(block),
                keywords=extract_keywords(block),
                status=detect_status(block),
            )
        )
    return pages


def choose_options(rng: random.Random, correct: str, pool: list[str]) -> tuple[dict[str, str], str]:
    negatives = [item for item in pool if item != correct]
    rng.shuffle(negatives)
    values = [correct] + negatives[:3]
    rng.shuffle(values)
    choices = {label: value for label, value in zip(LABELS, values)}
    return choices, next(label for label, value in choices.items() if value == correct)


def format_query(question: str, choices: dict[str, str]) -> str:
    options = "\n".join(f"{label}. {choices[label]}" for label in LABELS)
    return f"\nQuestion: {question}\n{options}\nAnswer with the option letter only:"


def noise_page(rng: random.Random, idx: int) -> str:
    project = rng.choice(PROJECTS)
    topic = rng.choice(NOISE_TOPICS)
    color = rng.choice(COLORS)
    action = rng.choice(ACTIONS)
    status = "Background note" if rng.random() < 0.8 else "Old note"
    return (
        f"{status} {idx}. Project {project} discussed {topic}. "
        f"The note mentions color {color}, tentative action {action}, and archival comments. "
        "This page is not controlling unless the later question asks for this exact project and field."
    )


def build_temporal_fact(rng: random.Random, task_id: int, distractors: int) -> dict[str, Any]:
    project = rng.choice(PROJECTS)
    current = rng.choice(COLORS)
    old = rng.choice([color for color in COLORS if color != current])
    choices, label = choose_options(rng, current, COLORS)
    pages = [noise_page(rng, idx) for idx in range(distractors)]
    pages.extend(
        [
            f"Old profile. Project {project} formerly used badge color {old}. This retired page is obsolete.",
            f"Current profile. Project {project} has active badge color {current}. This current page supersedes old profiles.",
        ]
    )
    rng.shuffle(pages)
    return {
        "variant": "temporal_fact",
        "task_id": task_id,
        "context": "\n\n".join(pages) + "\n",
        "query": format_query(f"What is the active badge color for Project {project}?", choices),
        "answer": label,
        "answer_value": current,
        "project": project,
    }


def build_multihop_bridge(rng: random.Random, task_id: int, distractors: int) -> dict[str, Any]:
    project = rng.choice(PROJECTS)
    artifact = rng.choice(ARTIFACTS)
    action = rng.choice(ACTIONS)
    choices, label = choose_options(rng, action, ACTIONS)
    pages = [noise_page(rng, idx) for idx in range(distractors)]
    pages.extend(
        [
            f"Current routing page. Project {project} routes to controlling artifact {artifact}. Use this active bridge.",
            f"Current artifact memo. Artifact {artifact} says the team should {action}. This is the approved action.",
            f"Old artifact memo. Artifact {artifact} formerly suggested {rng.choice([x for x in ACTIONS if x != action])}. This memo is superseded.",
        ]
    )
    rng.shuffle(pages)
    return {
        "variant": "multihop_bridge",
        "task_id": task_id,
        "context": "\n\n".join(pages) + "\n",
        "query": format_query(f"According to the current artifact for Project {project}, what should the team do?", choices),
        "answer": label,
        "answer_value": action,
        "project": project,
        "artifact": artifact,
    }


def build_summary_theme(rng: random.Random, task_id: int, distractors: int) -> dict[str, Any]:
    project = rng.choice(PROJECTS)
    theme = rng.choice(THEMES)
    alternatives = [item for item in THEMES if item != theme]
    pages = [noise_page(rng, idx) for idx in range(distractors)]
    for idx in range(5):
        pages.append(f"Current report {idx}. Project {project} records theme={theme} as the dominant issue for this update.")
    for idx, alt in enumerate(rng.sample(alternatives, 3)):
        pages.append(f"Current side note {idx}. Project {project} records theme={alt} as a secondary issue.")
    pages.append(f"Old report. Project {project} once recorded theme={rng.choice(alternatives)}. This old report is obsolete.")
    rng.shuffle(pages)
    choices, label = choose_options(rng, theme, THEMES)
    return {
        "variant": "summary_theme",
        "task_id": task_id,
        "context": "\n\n".join(pages) + "\n",
        "query": format_query(f"Across current reports for Project {project}, which theme appears most often?", choices),
        "answer": label,
        "answer_value": theme,
        "project": project,
    }


def build_compare_score(rng: random.Random, task_id: int, distractors: int) -> dict[str, Any]:
    projects = rng.sample(PROJECTS, 4)
    scores = {project: rng.randint(20, 95) for project in projects}
    winner = max(scores, key=scores.get)
    choices, label = choose_options(rng, winner, projects)
    pages = [noise_page(rng, idx) for idx in range(distractors)]
    for project, score in scores.items():
        pages.append(f"Current scorecard. Project {project} has priority_score={score}. This is the active scorecard.")
    for project in projects[:2]:
        pages.append(f"Old scorecard. Project {project} had priority_score={rng.randint(5, 99)}. This old score is superseded.")
    rng.shuffle(pages)
    return {
        "variant": "compare_score",
        "task_id": task_id,
        "context": "\n\n".join(pages) + "\n",
        "query": format_query("Which project has the highest current priority score?", choices),
        "answer": label,
        "answer_value": winner,
        "projects": projects,
    }


BUILDERS = {
    "temporal_fact": build_temporal_fact,
    "multihop_bridge": build_multihop_bridge,
    "summary_theme": build_summary_theme,
    "compare_score": build_compare_score,
}


def page_score(page: Page, query: str) -> float:
    query_entities = extract_entities(query)
    query_keywords = extract_keywords(query)
    score = 12.0 * len(page.entities & query_entities)
    score += sum(min(count, page.keywords.get(word, 0)) for word, count in query_keywords.items())
    if page.status == "current":
        score += 3.0
    if page.status == "non_current":
        score -= 6.0
    return score


def top_pages(pages: list[Page], query: str, count: int, exclude: set[int] | None = None) -> list[Page]:
    exclude = exclude or set()
    scored = [(page_score(page, query), page) for page in pages if page.page_id not in exclude]
    scored.sort(key=lambda item: (item[0], item[1].page_id), reverse=True)
    return [page for _, page in scored[:count]]


def infer_task_type(query: str) -> str:
    lower = query.lower()
    if "highest current priority score" in lower:
        return "compare_score"
    if "appears most often" in lower or "across current reports" in lower:
        return "summary_theme"
    if "artifact" in lower:
        return "multihop_bridge"
    if "active badge color" in lower:
        return "temporal_fact"
    return "single_fact"


def route_and_summarize(task: dict[str, Any], max_pages: int) -> tuple[str, dict[str, Any]]:
    pages = split_pages(task["context"])
    task_type = infer_task_type(task["query"])
    selected: list[Page] = []
    facts: list[str] = [f"task_type={task_type}"]
    answer_value = ""

    if task_type == "temporal_fact":
        selected = top_pages(pages, task["query"], 2)
        for page in selected:
            match = re.search(r"active badge color\s+([a-z]+)", page.text, flags=re.IGNORECASE)
            if match and page.status == "current":
                answer_value = match.group(1).lower()
                facts.append(f"current_badge_color={answer_value}")
                break
    elif task_type == "multihop_bridge":
        bridge_pages = top_pages(pages, task["query"], 2)
        selected.extend(bridge_pages)
        artifact = ""
        for page in bridge_pages:
            match = re.search(r"artifact\s+([A-Z0-9-]+)", page.text)
            if match and page.status == "current":
                artifact = match.group(1)
                facts.append(f"bridge_artifact={artifact}")
                break
        if artifact:
            artifact_pages = top_pages(pages, artifact + " current approved action", 3, {p.page_id for p in selected})
            selected.extend(artifact_pages)
            for page in artifact_pages:
                if artifact not in page.text or page.status != "current":
                    continue
                match = re.search(r"team should ([a-z ]+?)\.", page.text, flags=re.IGNORECASE)
                if match:
                    answer_value = match.group(1).lower()
                    facts.append(f"current_action={answer_value}")
                    break
    elif task_type == "summary_theme":
        selected = top_pages(pages, task["query"], max_pages)
        counts: Counter[str] = Counter()
        for page in selected:
            if page.status != "current":
                continue
            for match in re.finditer(r"theme=([a-z]+)", page.text):
                counts[match.group(1).lower()] += 1
        if counts:
            answer_value = counts.most_common(1)[0][0]
            facts.append("theme_counts=" + ",".join(f"{key}:{value}" for key, value in counts.most_common()))
            facts.append(f"dominant_theme={answer_value}")
    elif task_type == "compare_score":
        selected = top_pages(pages, task["query"], max_pages)
        scores: dict[str, int] = {}
        for page in selected:
            if page.status != "current":
                continue
            match = re.search(r"Project\s+([A-Za-z]+)\s+has priority_score=(\d+)", page.text)
            if match:
                scores[match.group(1)] = int(match.group(2))
        if scores:
            answer_value = max(scores, key=scores.get)
            facts.append("scores=" + ",".join(f"{key}:{value}" for key, value in sorted(scores.items())))
            facts.append(f"highest_project={answer_value}")
    else:
        selected = top_pages(pages, task["query"], 1)

    selected = selected[:max_pages]
    evidence_hit = 0
    if answer_value:
        evidence_hit = int(answer_value.lower() == str(task["answer_value"]).lower())
    prompt = "\nTyped memory records:\n" + "; ".join(facts) + ".\n"
    for page in selected:
        prompt += f"[page {page.page_id}; status={page.status}] {re.sub(r'\\s+', ' ', page.text).strip()}\n"
    prompt += "Use the typed memory records and current-status evidence pages to answer.\n"
    return prompt + task["query"], {
        "task_type": task_type,
        "selected_pages": len({page.page_id for page in selected}),
        "page_count": len(pages),
        "evidence_hit": evidence_hit,
        "routed_answer_value": answer_value,
    }


@torch.inference_mode()
def run_suffix_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    text: str,
) -> tuple[Any, torch.Tensor, float, int]:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device)
    if ids.shape[-1] == 0:
        return past_key_values, prev_logits, 0.0, 0
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
    logits = outputs.logits
    loss_value = 0.0
    token_count = 0
    if prev_logits.numel() and logits.shape[1] > 0:
        shifted = torch.cat([prev_logits.unsqueeze(1), logits[:, :-1, :]], dim=1)
        loss = F.cross_entropy(shifted.float().reshape(-1, shifted.shape[-1]), ids.reshape(-1), reduction="sum")
        loss_value = float(loss.item())
        token_count = int(ids.numel())
    return outputs.past_key_values, logits[:, -1, :].detach(), loss_value, token_count


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


def evaluate(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    context_cache: Any,
    context_prev: torch.Tensor,
    prompt: str,
    answer: str,
) -> tuple[str, dict[str, float], float, float, int, float]:
    started = time.perf_counter()
    cache, prev, _, _ = run_suffix_batch(
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
            "answer": task["answer"],
            "answer_value": task["answer_value"],
            "pred": pred,
            "correct": int(pred == task["answer"]),
            "eval_seconds": eval_seconds,
            "prefill_seconds": prefill_seconds,
            "total_seconds": eval_seconds + prefill_seconds,
            "visible_tokens": visible_tokens,
            "gold_label_loss": gold_loss,
            "gold_label_tokens": gold_tokens,
            "gold_label_ppl": gold_ppl,
            "task_type": meta.get("task_type", ""),
            "selected_pages": int(meta.get("selected_pages", 0)),
            "page_count": int(meta.get("page_count", 0)),
            "evidence_hit": int(meta.get("evidence_hit", 0)),
            "routed_answer_value": meta.get("routed_answer_value", ""),
            **{f"score_{label}": scores[label] for label in LABELS},
        }
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["mode"])].append(row)
        grouped[("ALL", row["mode"])].append(row)
    out = []
    for (variant, mode), subset in sorted(grouped.items()):
        n = max(1, len(subset))
        total_tokens = sum(int(row["gold_label_tokens"]) for row in subset)
        total_loss = sum(float(row["gold_label_loss"]) * int(row["gold_label_tokens"]) for row in subset)
        mean_loss = total_loss / max(1, total_tokens)
        out.append(
            {
                "variant": variant,
                "mode": mode,
                "tasks": len(subset),
                "accuracy": sum(int(row["correct"]) for row in subset) / n,
                "gold_label_loss": mean_loss,
                "gold_label_ppl": math.exp(min(mean_loss, 80.0)),
                "mean_eval_seconds": sum(float(row["eval_seconds"]) for row in subset) / n,
                "mean_total_seconds": sum(float(row["total_seconds"]) for row in subset) / n,
                "mean_visible_tokens": sum(int(row["visible_tokens"]) for row in subset) / n,
                "evidence_hit_rate": sum(int(row["evidence_hit"]) for row in subset) / n,
                "mean_selected_pages": sum(int(row["selected_pages"]) for row in subset) / n,
            }
        )
    return out


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    variants = [part.strip() for part in config.variants.split(",") if part.strip()]
    unknown = [variant for variant in variants if variant not in BUILDERS]
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
    rng = random.Random(config.seed)
    for variant in variants:
        for task_idx in range(config.tasks_per_variant):
            task = BUILDERS[variant](rng, task_idx, config.distractor_pages)
            print(f"{variant} {task_idx + 1}/{config.tasks_per_variant}", flush=True)
            context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            started = time.perf_counter()
            context_cache, context_prev = prefill_cache(
                model,
                context_ids,
                context_ids.shape[-1],
                config.chunk_size,
                input_device,
            )
            prefill_seconds = time.perf_counter() - started
            pred, scores, elapsed, loss, tokens, ppl = evaluate(
                model,
                tokenizer,
                input_device,
                context_cache,
                context_prev,
                task["query"],
                task["answer"],
            )
            append_row(
                rows,
                task,
                "full_baseline",
                pred,
                scores,
                elapsed,
                prefill_seconds,
                int(context_ids.shape[-1]) + len(tokenizer(task["query"], add_special_tokens=False)["input_ids"]),
                loss,
                tokens,
                ppl,
                {},
            )

            prompt, meta = route_and_summarize(task, config.max_route_pages)
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
            cache, prev = empty_state(input_device)
            pred, scores, elapsed, loss, tokens, ppl = evaluate(
                model,
                tokenizer,
                input_device,
                cache,
                prev,
                prompt,
                task["answer"],
            )
            append_row(
                rows,
                task,
                "typed_memory_router_v1",
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
        "answer",
        "answer_value",
        "pred",
        "correct",
        "eval_seconds",
        "prefill_seconds",
        "total_seconds",
        "visible_tokens",
        "gold_label_loss",
        "gold_label_tokens",
        "gold_label_ppl",
        "task_type",
        "selected_pages",
        "page_count",
        "evidence_hit",
        "routed_answer_value",
    ] + [f"score_{label}" for label in LABELS]
    with (output_dir / "typed_memory_router_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    with (output_dir / "typed_memory_router_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    (output_dir / "summary.json").write_text(json.dumps({"summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
