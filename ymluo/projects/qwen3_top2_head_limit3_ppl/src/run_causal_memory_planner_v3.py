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
    pick_input_device,
    resolve_dtype,
)
from run_qabs_downstream_kv_retrieval import LABELS  # noqa: E402
from run_risk_calibrated_memory_planner_v1 import full_context_prompt, memory_need_vector  # noqa: E402
from run_task_aware_kv_mixture_v0 import ALL_BUILDERS, empty_state, evaluate_prompt  # noqa: E402
from run_typed_memory_router_v1_suite import (  # noqa: E402
    Page,
    extract_entities,
    extract_keywords,
    page_score,
    split_pages,
)


VARIANT_NAMES = ["casual_recent", "temporal_fact", "multihop_bridge", "summary_theme", "compare_score"]
NEED_FEATURES = [
    "locality_need",
    "semantic_need",
    "hop_depth",
    "temporal_conflict_need",
    "aggregation_scope",
    "risk_level",
]
FEATURE_NAMES = [
    "lexical_score",
    "lexical_rank_norm",
    "entity_overlap",
    "keyword_overlap",
    "is_current",
    "is_non_current",
    "is_unknown_status",
    "position_norm",
    "is_last_page",
    "is_recent2_page",
    "length_words_norm",
    "has_theme_marker",
    "has_priority_marker",
    "has_badge_marker",
    "has_artifact_marker",
    "has_preferred_reply_marker",
    "has_old_cue",
    "has_current_cue",
    "query_asks_recent",
    "query_asks_current",
    "query_asks_artifact",
    "query_asks_across",
    "query_asks_highest",
    *[f"need_{name}" for name in NEED_FEATURES],
    *[f"variant_{name}" for name in VARIANT_NAMES],
]


@dataclass(frozen=True)
class Config:
    model_name_or_path: str
    output_dir: str
    variants: str
    tasks_per_variant: int
    train_fraction: float
    distractor_pages: int
    seed: int
    dtype: str
    device: str
    device_map: str
    attn_implementation: str
    topk_pages: int
    max_ablate_pages: int
    positive_delta_threshold: float
    adaptive_labeling: int
    adaptive_mad_scale: float
    weak_positive_if_no_label: int
    logistic_epochs: int
    logistic_lr: float
    fallback_margin: float
    fallback_predicted_recall: float
    log_every: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Causal memory influence planner V3: label memory pages by full-context "
            "page ablation, then train a small causal page planner."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--variants",
        default="casual_recent,temporal_fact,multihop_bridge,summary_theme,compare_score",
    )
    parser.add_argument("--tasks_per_variant", type=int, default=8)
    parser.add_argument("--train_fraction", type=float, default=0.5)
    parser.add_argument("--distractor_pages", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026070208)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--topk_pages", type=int, default=3)
    parser.add_argument(
        "--max_ablate_pages",
        type=int,
        default=0,
        help="0 means ablate every page. Positive values cap ablations with lexical/recent coverage.",
    )
    parser.add_argument("--positive_delta_threshold", type=float, default=0.03)
    parser.add_argument(
        "--adaptive_labeling",
        type=int,
        default=1,
        help="Use a robust per-task threshold: max(abs_threshold, median_delta + scale * MAD).",
    )
    parser.add_argument("--adaptive_mad_scale", type=float, default=1.0)
    parser.add_argument("--weak_positive_if_no_label", type=int, default=1)
    parser.add_argument("--logistic_epochs", type=int, default=220)
    parser.add_argument("--logistic_lr", type=float, default=0.05)
    parser.add_argument("--fallback_margin", type=float, default=0.45)
    parser.add_argument("--fallback_predicted_recall", type=float, default=0.55)
    parser.add_argument("--log_every", type=int, default=5)
    return Config(**vars(parser.parse_args()))


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prompt_token_count(tokenizer: Any, prompt: str) -> int:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    return int(ids.shape[-1])


def score_margin(scores: dict[str, float]) -> float:
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) < 2:
        return 0.0
    return float(ordered[0] - ordered[1])


def selected_pages_prompt(task: dict[str, Any], pages: list[Page], selected_ids: list[int], title: str) -> str:
    by_id = {page.page_id: page for page in pages}
    prompt = f"\n{title}:\n"
    if not selected_ids:
        prompt += "[no long memory page selected]\n"
    for page_id in selected_ids:
        page = by_id[page_id]
        prompt += f"[page {page.page_id}; status={page.status}] {compact(page.text)}\n"
    prompt += (
        "Use the selected memory pages to answer the multiple-choice question. "
        "Prefer current/active pages over old/superseded pages. Output only the option letter.\n"
    )
    return prompt + task["query"]


def ablated_full_prompt(task: dict[str, Any], pages: list[Page], drop_page_id: int) -> str:
    kept = "\n\n".join(page.text for page in pages if page.page_id != drop_page_id)
    return (
        "\nFull memory context with one page removed:\n"
        + kept
        + "\nUse the remaining context to answer with the option letter.\n"
        + task["query"]
    )


def answer_text_hit(task: dict[str, Any], page: Page) -> int:
    return int(str(task["answer_value"]).lower() in page.text.lower())


def evidence_hit(task: dict[str, Any], pages: list[Page], selected_ids: list[int]) -> int:
    selected = {page_id for page_id in selected_ids}
    return int(any(page.page_id in selected and answer_text_hit(task, page) for page in pages))


def ablation_candidate_ids(task: dict[str, Any], pages: list[Page], max_ablate_pages: int) -> list[int]:
    if max_ablate_pages <= 0 or len(pages) <= max_ablate_pages:
        return [page.page_id for page in pages]

    selected: set[int] = set()
    for page in pages:
        if answer_text_hit(task, page):
            selected.add(page.page_id)
    for page in pages[-3:]:
        selected.add(page.page_id)

    ranked = sorted(pages, key=lambda page: (page_score(page, task["query"]), page.page_id), reverse=True)
    for page in ranked:
        selected.add(page.page_id)
        if len(selected) >= max_ablate_pages:
            break

    if len(selected) < max_ablate_pages:
        for page in pages:
            selected.add(page.page_id)
            if len(selected) >= max_ablate_pages:
                break
    return sorted(selected)


def page_feature_map(
    task: dict[str, Any],
    pages: list[Page],
    page: Page,
    lexical_ranks: dict[int, int],
) -> dict[str, float]:
    query = task["query"]
    query_lower = query.lower()
    query_entities = extract_entities(query)
    query_keywords = extract_keywords(query)
    keyword_overlap = sum(min(count, page.keywords.get(word, 0)) for word, count in query_keywords.items())
    need = memory_need_vector(task)
    lower = page.text.lower()
    denom = max(1, len(pages) - 1)
    features: dict[str, float] = {
        "lexical_score": float(page_score(page, query)),
        "lexical_rank_norm": float(lexical_ranks[page.page_id] / max(1, len(pages) - 1)),
        "entity_overlap": float(len(page.entities & query_entities)),
        "keyword_overlap": float(keyword_overlap),
        "is_current": float(page.status == "current"),
        "is_non_current": float(page.status == "non_current"),
        "is_unknown_status": float(page.status == "unknown"),
        "position_norm": float(page.page_id / denom),
        "is_last_page": float(page.page_id == len(pages) - 1),
        "is_recent2_page": float(page.page_id >= max(0, len(pages) - 2)),
        "length_words_norm": min(5.0, len(re.findall(r"[A-Za-z0-9-]+", page.text)) / 50.0),
        "has_theme_marker": float("theme=" in lower),
        "has_priority_marker": float("priority_score=" in lower),
        "has_badge_marker": float("badge color" in lower),
        "has_artifact_marker": float("artifact" in lower),
        "has_preferred_reply_marker": float("preferred assistant reply" in lower),
        "has_old_cue": float(any(cue in lower for cue in ["old", "former", "obsolete", "superseded", "retired"])),
        "has_current_cue": float(any(cue in lower for cue in ["current", "active", "latest", "approved", "valid"])),
        "query_asks_recent": float("recent chat" in query_lower or "assistant reply" in query_lower),
        "query_asks_current": float("current" in query_lower or "active" in query_lower),
        "query_asks_artifact": float("artifact" in query_lower),
        "query_asks_across": float("across" in query_lower or "most often" in query_lower),
        "query_asks_highest": float("highest" in query_lower or "priority score" in query_lower),
    }
    for name in NEED_FEATURES:
        features[f"need_{name}"] = float(need.get(name, 0.0))
    for name in VARIANT_NAMES:
        features[f"variant_{name}"] = float(task["variant"] == name)
    return features


def make_page_feature_rows(task: dict[str, Any], pages: list[Page]) -> dict[int, dict[str, float]]:
    ranked = sorted(pages, key=lambda page: (page_score(page, task["query"]), page.page_id), reverse=True)
    lexical_ranks = {page.page_id: rank for rank, page in enumerate(ranked)}
    return {page.page_id: page_feature_map(task, pages, page, lexical_ranks) for page in pages}


def vectorize(feature_map: dict[str, float]) -> list[float]:
    return [float(feature_map.get(name, 0.0)) for name in FEATURE_NAMES]


def evaluate_mode(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    task: dict[str, Any],
    pages: list[Page],
    mode: str,
    selected_ids: list[int],
    full_row: dict[str, Any],
    influence_by_page: dict[int, dict[str, Any]],
    estimated_causal_recall: float = 0.0,
    fallback: int = 0,
    extra_eval_seconds: float = 0.0,
    extra_visible_tokens: int = 0,
) -> dict[str, Any]:
    prompt = selected_pages_prompt(task, pages, selected_ids, f"{mode} memory pages")
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
    return make_eval_row(
        tokenizer,
        task,
        pages,
        mode,
        pred,
        scores,
        elapsed + extra_eval_seconds,
        prompt_token_count(tokenizer, prompt) + extra_visible_tokens,
        loss,
        tokens,
        ppl,
        selected_ids,
        full_row,
        influence_by_page,
        estimated_causal_recall,
        fallback,
    )


def make_eval_row(
    tokenizer: Any,
    task: dict[str, Any],
    pages: list[Page],
    mode: str,
    pred: str,
    scores: dict[str, float],
    elapsed: float,
    visible_tokens: int,
    loss: float,
    tokens: int,
    ppl: float,
    selected_ids: list[int],
    full_row: dict[str, Any],
    influence_by_page: dict[int, dict[str, Any]],
    estimated_causal_recall: float,
    fallback: int,
) -> dict[str, Any]:
    del tokenizer
    selected = set(selected_ids)
    positive_ids = {
        int(page_id)
        for page_id, row in influence_by_page.items()
        if int(row.get("causal_label", 0)) == 1
    }
    positive_mass_total = sum(max(0.0, float(row["loss_delta"])) for row in influence_by_page.values())
    positive_mass_selected = sum(
        max(0.0, float(row["loss_delta"]))
        for page_id, row in influence_by_page.items()
        if int(page_id) in selected
    )
    return {
        "variant": task["variant"],
        "task_id": task["task_id"],
        "split": full_row["split"],
        "mode": mode,
        "answer": task["answer"],
        "answer_value": task["answer_value"],
        "pred": pred,
        "correct": int(pred == task["answer"]),
        "eval_seconds": elapsed,
        "visible_tokens": visible_tokens,
        "gold_label_loss": loss,
        "gold_label_tokens": tokens,
        "gold_label_ppl": ppl,
        "margin": score_margin(scores),
        "selected_pages": len(selected_ids),
        "selected_page_ids": ",".join(str(page_id) for page_id in selected_ids),
        "page_count": len(pages),
        "evidence_hit": evidence_hit(task, pages, selected_ids),
        "teacher_pred": full_row["pred"],
        "teacher_correct": full_row["correct"],
        "teacher_gold_label_ppl": full_row["gold_label_ppl"],
        "causal_positive_pages": len(positive_ids),
        "causal_label_recall": len(selected & positive_ids) / max(1, len(positive_ids)),
        "causal_label_precision": len(selected & positive_ids) / max(1, len(selected)),
        "causal_positive_mass_recall": positive_mass_selected / max(1e-8, positive_mass_total),
        "estimated_causal_recall": estimated_causal_recall,
        "fallback": fallback,
        **{f"score_{label}": scores[label] for label in LABELS},
    }


def full_eval_row(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    task: dict[str, Any],
    pages: list[Page],
    split: str,
) -> dict[str, Any]:
    prompt, _ = full_context_prompt(task)
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
    return {
        "variant": task["variant"],
        "task_id": task["task_id"],
        "split": split,
        "mode": "full_context_teacher",
        "answer": task["answer"],
        "answer_value": task["answer_value"],
        "pred": pred,
        "correct": int(pred == task["answer"]),
        "eval_seconds": elapsed,
        "visible_tokens": prompt_token_count(tokenizer, prompt),
        "gold_label_loss": loss,
        "gold_label_tokens": tokens,
        "gold_label_ppl": ppl,
        "margin": score_margin(scores),
        "selected_pages": len(pages),
        "selected_page_ids": ",".join(str(page.page_id) for page in pages),
        "page_count": len(pages),
        "evidence_hit": int(any(answer_text_hit(task, page) for page in pages)),
        "teacher_pred": pred,
        "teacher_correct": int(pred == task["answer"]),
        "teacher_gold_label_ppl": ppl,
        "causal_positive_pages": 0,
        "causal_label_recall": 1.0,
        "causal_label_precision": 0.0,
        "causal_positive_mass_recall": 1.0,
        "estimated_causal_recall": 1.0,
        "fallback": 0,
        **{f"score_{label}": scores[label] for label in LABELS},
    }


def label_influence_rows(
    rows: list[dict[str, Any]],
    positive_delta_threshold: float,
    adaptive_labeling: int,
    adaptive_mad_scale: float,
    weak_positive_if_no_label: int,
) -> None:
    by_task: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[(str(row["variant"]), int(row["task_id"]))].append(row)

    for task_rows in by_task.values():
        deltas = [float(row["loss_delta"]) for row in task_rows]
        sorted_deltas = sorted(deltas)
        mid = len(sorted_deltas) // 2
        if len(sorted_deltas) % 2:
            median_delta = sorted_deltas[mid]
        else:
            median_delta = 0.5 * (sorted_deltas[mid - 1] + sorted_deltas[mid])
        abs_devs = sorted(abs(delta - median_delta) for delta in deltas)
        if len(abs_devs) % 2:
            mad = abs_devs[mid]
        else:
            mad = 0.5 * (abs_devs[mid - 1] + abs_devs[mid])
        robust_threshold = max(
            positive_delta_threshold,
            median_delta + adaptive_mad_scale * 1.4826 * max(mad, 1e-8),
        )
        task_threshold = robust_threshold if adaptive_labeling else positive_delta_threshold
        for row in task_rows:
            delta = float(row["loss_delta"])
            row["causal_label"] = int(
                delta >= task_threshold
                or int(row["teacher_correct"]) == 1
                and int(row["ablated_correct"]) == 0
                and delta >= positive_delta_threshold
            )
            row["negative_influence_label"] = int(delta <= -positive_delta_threshold)
            row["task_label_threshold"] = task_threshold
            row["weak_positive_label"] = 0
        if sum(int(row["causal_label"]) for row in task_rows) == 0 and weak_positive_if_no_label:
            best = max(task_rows, key=lambda row: (float(row["loss_delta"]), -int(row["page_id"])))
            if float(best["loss_delta"]) > 0:
                best["causal_label"] = 1
                best["weak_positive_label"] = 1


def train_logistic_page_model(
    page_rows: list[dict[str, Any]],
    config: Config,
) -> dict[str, Any]:
    train_rows = [row for row in page_rows if row["split"] == "train"]
    x_values = [vectorize({name: float(row[f"feature_{name}"]) for name in FEATURE_NAMES}) for row in train_rows]
    y_values = [float(row["causal_label"]) for row in train_rows]
    if not x_values or len(set(y_values)) < 2:
        return {
            "valid": 0,
            "reason": "not_enough_label_diversity",
            "feature_names": FEATURE_NAMES,
            "weights": [0.0 for _ in FEATURE_NAMES],
            "bias": 0.0,
            "mean": [0.0 for _ in FEATURE_NAMES],
            "std": [1.0 for _ in FEATURE_NAMES],
            "train_pages": len(train_rows),
            "train_positive_rate": sum(y_values) / max(1, len(y_values)),
        }

    x = torch.tensor(x_values, dtype=torch.float32)
    y = torch.tensor(y_values, dtype=torch.float32).unsqueeze(1)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    x_norm = (x - mean) / std

    model = torch.nn.Linear(x_norm.shape[1], 1)
    positives = float(y.sum().item())
    negatives = float(y.numel() - positives)
    pos_weight = torch.tensor([max(1.0, negatives / max(1.0, positives))], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.logistic_lr, weight_decay=1e-3)
    for _ in range(config.logistic_epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_norm)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(x_norm)
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()
        train_accuracy = float((pred == y).float().mean().item())

    return {
        "valid": 1,
        "reason": "",
        "feature_names": FEATURE_NAMES,
        "weights": [float(v) for v in model.weight.detach().reshape(-1).tolist()],
        "bias": float(model.bias.detach().reshape(-1)[0].item()),
        "mean": [float(v) for v in mean.tolist()],
        "std": [float(v) for v in std.tolist()],
        "train_pages": len(train_rows),
        "train_positive_rate": positives / max(1.0, float(len(train_rows))),
        "train_accuracy": train_accuracy,
    }


def learned_probability(row: dict[str, Any], model_info: dict[str, Any]) -> float:
    if not int(model_info.get("valid", 0)):
        lexical = float(row.get("feature_lexical_score", 0.0))
        current = float(row.get("feature_is_current", 0.0))
        recent = float(row.get("feature_is_recent2_page", 0.0))
        return 1.0 / (1.0 + math.exp(-(0.15 * lexical + 0.8 * current + 0.6 * recent)))
    values = [float(row[f"feature_{name}"]) for name in FEATURE_NAMES]
    mean = model_info["mean"]
    std = model_info["std"]
    weights = model_info["weights"]
    logit = float(model_info["bias"])
    for value, mu, sigma, weight in zip(values, mean, std, weights):
        logit += ((value - mu) / max(1e-6, sigma)) * weight
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))


def select_recent(pages: list[Page], topk: int) -> list[int]:
    return [page.page_id for page in pages[-topk:]]


def select_lexical(task: dict[str, Any], pages: list[Page], topk: int) -> list[int]:
    ranked = sorted(pages, key=lambda page: (page_score(page, task["query"]), page.page_id), reverse=True)
    return [page.page_id for page in ranked[:topk]]


def select_oracle_causal(influence_rows: list[dict[str, Any]], topk: int) -> list[int]:
    ranked = sorted(
        influence_rows,
        key=lambda row: (float(row["loss_delta"]), int(row["causal_label"]), -int(row["page_id"])),
        reverse=True,
    )
    return [int(row["page_id"]) for row in ranked[:topk]]


def select_learned(
    task_page_rows: list[dict[str, Any]],
    model_info: dict[str, Any],
    topk: int,
) -> tuple[list[int], float]:
    scored = [(learned_probability(row, model_info), int(row["page_id"])) for row in task_page_rows]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [page_id for _, page_id in scored[:topk]]
    total_mass = sum(prob for prob, _ in scored)
    selected_mass = sum(prob for prob, page_id in scored if page_id in set(selected))
    estimated_recall = selected_mass / max(1e-8, total_mass)
    return selected, estimated_recall


def minmax(value: float, values: list[float]) -> float:
    lo = min(values)
    hi = max(values)
    if hi <= lo + 1e-8:
        return 0.0
    return (value - lo) / (hi - lo)


def typed_role_prior(task: dict[str, Any], page: Page) -> float:
    query = task["query"].lower()
    text = page.text.lower()
    prior = 0.0
    if page.status == "current":
        prior += 0.15
    if page.status == "non_current":
        prior -= 0.20

    if "recent chat" in query or "assistant reply" in query:
        if "preferred assistant reply" in text:
            prior += 1.0
        if page.page_id >= task["context"].count("\n\n") - 1:
            prior += 0.25
    if "active badge color" in query:
        if "badge color" in text and page.status == "current":
            prior += 1.0
        elif "badge color" in text:
            prior += 0.25
    if "artifact" in query:
        if "routing page" in text and "artifact" in text and page.status == "current":
            prior += 0.75
        if "artifact memo" in text and page.status == "current":
            prior += 0.75
    if "most often" in query or "across current reports" in query:
        if "theme=" in text and page.status == "current":
            prior += 1.0
    if "highest current priority score" in query:
        if "priority_score=" in text and page.status == "current":
            prior += 1.0
    return prior


def select_hybrid_causal_lexical(
    task: dict[str, Any],
    pages: list[Page],
    task_page_rows: list[dict[str, Any]],
    model_info: dict[str, Any],
    topk: int,
) -> tuple[list[int], float]:
    learned = {int(row["page_id"]): learned_probability(row, model_info) for row in task_page_rows}
    lexical = {page.page_id: float(page_score(page, task["query"])) for page in pages}
    lexical_values = list(lexical.values())
    prior = {page.page_id: typed_role_prior(task, page) for page in pages}
    prior_values = list(prior.values())
    scored = []
    for row in task_page_rows:
        page_id = int(row["page_id"])
        score = (
            0.45 * learned[page_id]
            + 0.30 * minmax(lexical[page_id], lexical_values)
            + 0.25 * minmax(prior[page_id], prior_values)
        )
        scored.append((score, page_id))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected: list[int] = []
    for _, page_id in scored:
        if page_id not in selected:
            selected.append(page_id)
        if len(selected) >= topk:
            break

    def prioritize_role_pages(base: list[int], role_pages: list[int]) -> list[int]:
        repaired: list[int] = []
        for page_id in role_pages[:topk]:
            if page_id not in repaired:
                repaired.append(page_id)
        for page_id in base:
            if page_id not in repaired:
                repaired.append(page_id)
            if len(repaired) >= topk:
                break
        return repaired[:topk]

    # Coverage repair for aggregation queries: learned top-k can over-focus on one high-score page,
    # while summary/compare tasks need several same-role pages to preserve the aggregate.
    query = task["query"].lower()
    if ("most often" in query or "across current reports" in query) and topk >= 3:
        role_pages = [
            page.page_id
            for page in sorted(pages, key=lambda page: (page_score(page, task["query"]), page.page_id), reverse=True)
            if page.status == "current" and "theme=" in page.text.lower()
        ]
        selected = prioritize_role_pages(selected, role_pages)
    if "highest current priority score" in query and topk >= 4:
        role_pages = [
            page.page_id
            for page in sorted(pages, key=lambda page: (page_score(page, task["query"]), page.page_id), reverse=True)
            if page.status == "current" and "priority_score=" in page.text.lower()
        ]
        selected = prioritize_role_pages(selected, role_pages)

    total_mass = sum(learned.values())
    selected_mass = sum(learned.get(page_id, 0.0) for page_id in selected)
    return selected, selected_mass / max(1e-8, total_mass)


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["variant"]), str(row["mode"]))].append(row)
        grouped[(str(row["split"]), "ALL", str(row["mode"]))].append(row)
    summary = []
    for (split, variant, mode), subset in sorted(grouped.items()):
        n = max(1, len(subset))
        total_tokens = sum(int(row["gold_label_tokens"]) for row in subset)
        total_loss = sum(float(row["gold_label_loss"]) * int(row["gold_label_tokens"]) for row in subset)
        mean_loss = total_loss / max(1, total_tokens)
        summary.append(
            {
                "split": split,
                "variant": variant,
                "mode": mode,
                "tasks": len(subset),
                "accuracy": sum(int(row["correct"]) for row in subset) / n,
                "gold_label_ppl": math.exp(min(mean_loss, 80.0)),
                "mean_eval_seconds": sum(float(row["eval_seconds"]) for row in subset) / n,
                "mean_visible_tokens": sum(int(row["visible_tokens"]) for row in subset) / n,
                "mean_selected_pages": sum(int(row["selected_pages"]) for row in subset) / n,
                "evidence_hit_rate": sum(int(row["evidence_hit"]) for row in subset) / n,
                "teacher_correct_rate": sum(int(row["teacher_correct"]) for row in subset) / n,
                "mean_margin": sum(float(row["margin"]) for row in subset) / n,
                "mean_causal_label_recall": sum(float(row["causal_label_recall"]) for row in subset) / n,
                "mean_causal_label_precision": sum(float(row["causal_label_precision"]) for row in subset) / n,
                "mean_causal_positive_mass_recall": sum(float(row["causal_positive_mass_recall"]) for row in subset) / n,
                "mean_estimated_causal_recall": sum(float(row["estimated_causal_recall"]) for row in subset) / n,
                "fallback_rate": sum(int(row["fallback"]) for row in subset) / n,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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

    rng = random.Random(config.seed)
    train_cut = max(1, min(config.tasks_per_variant - 1, int(round(config.tasks_per_variant * config.train_fraction))))
    task_bundles: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for variant in variants:
        for task_idx in range(config.tasks_per_variant):
            split = "train" if task_idx < train_cut else "test"
            task = ALL_BUILDERS[variant](rng, task_idx, config.distractor_pages)
            pages = split_pages(task["context"])
            if (task_idx + 1) % config.log_every == 0 or task_idx == 0:
                print(f"{variant} {task_idx + 1}/{config.tasks_per_variant} split={split}", flush=True)

            full_row = full_eval_row(model, tokenizer, input_device, task, pages, split)
            feature_by_page = make_page_feature_rows(task, pages)
            ablate_ids = ablation_candidate_ids(task, pages, config.max_ablate_pages)
            task_page_rows: list[dict[str, Any]] = []
            for page in pages:
                features = feature_by_page[page.page_id]
                base = {
                    "variant": variant,
                    "task_id": task_idx,
                    "split": split,
                    "page_id": page.page_id,
                    "page_status": page.status,
                    "page_text": compact(page.text),
                    "answer_text_hit": answer_text_hit(task, page),
                    "was_ablated": int(page.page_id in ablate_ids),
                    "teacher_pred": full_row["pred"],
                    "teacher_correct": full_row["correct"],
                    "teacher_gold_label_loss": full_row["gold_label_loss"],
                    "teacher_gold_label_ppl": full_row["gold_label_ppl"],
                    "ablated_pred": "",
                    "ablated_correct": 0,
                    "ablated_gold_label_loss": 0.0,
                    "ablated_gold_label_ppl": 0.0,
                    "ablation_eval_seconds": 0.0,
                    "loss_delta": 0.0,
                    "ppl_delta": 0.0,
                    "causal_label": 0,
                    "negative_influence_label": 0,
                    "task_label_threshold": 0.0,
                    "weak_positive_label": 0,
                    **{f"feature_{name}": features[name] for name in FEATURE_NAMES},
                }
                task_page_rows.append(base)

            row_by_page = {int(row["page_id"]): row for row in task_page_rows}
            for page_id in ablate_ids:
                ablated_prompt = ablated_full_prompt(task, pages, page_id)
                cache, prev = empty_state(input_device)
                pred, _, elapsed, loss, _, ppl = evaluate_prompt(
                    model,
                    tokenizer,
                    input_device,
                    cache,
                    prev,
                    ablated_prompt,
                    task["answer"],
                )
                row = row_by_page[page_id]
                row["ablated_pred"] = pred
                row["ablated_correct"] = int(pred == task["answer"])
                row["ablated_gold_label_loss"] = loss
                row["ablated_gold_label_ppl"] = ppl
                row["ablation_eval_seconds"] = elapsed
                row["loss_delta"] = float(loss) - float(full_row["gold_label_loss"])
                row["ppl_delta"] = float(ppl) - float(full_row["gold_label_ppl"])

            label_influence_rows(
                task_page_rows,
                config.positive_delta_threshold,
                config.adaptive_labeling,
                config.adaptive_mad_scale,
                config.weak_positive_if_no_label,
            )
            page_rows.extend(task_page_rows)
            influence_by_page = {int(row["page_id"]): row for row in task_page_rows}
            result_rows.append(
                make_eval_row(
                    tokenizer,
                    task,
                    pages,
                    "full_context_teacher",
                    full_row["pred"],
                    {label: full_row[f"score_{label}"] for label in LABELS},
                    full_row["eval_seconds"],
                    full_row["visible_tokens"],
                    full_row["gold_label_loss"],
                    full_row["gold_label_tokens"],
                    full_row["gold_label_ppl"],
                    [page.page_id for page in pages],
                    full_row,
                    influence_by_page,
                    1.0,
                    0,
                )
            )
            task_bundles.append(
                {
                    "task": task,
                    "pages": pages,
                    "split": split,
                    "full_row": full_row,
                    "page_rows": task_page_rows,
                    "influence_by_page": influence_by_page,
                }
            )

    model_info = train_logistic_page_model(page_rows, config)
    (output_dir / "learned_page_model.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    for bundle in task_bundles:
        task = bundle["task"]
        pages = bundle["pages"]
        full_row = bundle["full_row"]
        task_page_rows = bundle["page_rows"]
        influence_by_page = bundle["influence_by_page"]

        recent_ids = select_recent(pages, config.topk_pages)
        lexical_ids = select_lexical(task, pages, config.topk_pages)
        oracle_ids = select_oracle_causal(task_page_rows, config.topk_pages)
        learned_ids, estimated_recall = select_learned(task_page_rows, model_info, config.topk_pages)
        hybrid_ids, hybrid_estimated_recall = select_hybrid_causal_lexical(
            task,
            pages,
            task_page_rows,
            model_info,
            config.topk_pages,
        )

        result_rows.append(
            evaluate_mode(
                model,
                tokenizer,
                input_device,
                task,
                pages,
                "recent_topk_pages",
                recent_ids,
                full_row,
                influence_by_page,
            )
        )
        result_rows.append(
            evaluate_mode(
                model,
                tokenizer,
                input_device,
                task,
                pages,
                "lexical_topk_pages",
                lexical_ids,
                full_row,
                influence_by_page,
            )
        )
        result_rows.append(
            evaluate_mode(
                model,
                tokenizer,
                input_device,
                task,
                pages,
                "oracle_causal_topk_pages",
                oracle_ids,
                full_row,
                influence_by_page,
                estimated_causal_recall=1.0,
            )
        )
        learned_row = evaluate_mode(
            model,
            tokenizer,
            input_device,
            task,
            pages,
            "learned_causal_topk_pages",
            learned_ids,
            full_row,
            influence_by_page,
            estimated_causal_recall=estimated_recall,
        )
        result_rows.append(learned_row)
        result_rows.append(
            evaluate_mode(
                model,
                tokenizer,
                input_device,
                task,
                pages,
                "hybrid_causal_lexical_topk_pages",
                hybrid_ids,
                full_row,
                influence_by_page,
                estimated_causal_recall=hybrid_estimated_recall,
            )
        )

        if (
            float(learned_row["margin"]) >= config.fallback_margin
            and float(learned_row["estimated_causal_recall"]) >= config.fallback_predicted_recall
        ):
            progressive = dict(learned_row)
            progressive["mode"] = "progressive_causal_v3"
            progressive["fallback"] = 0
            result_rows.append(progressive)
        else:
            progressive = dict(full_row)
            progressive.update(
                make_eval_row(
                    tokenizer,
                    task,
                    pages,
                    "progressive_causal_v3",
                    full_row["pred"],
                    {label: full_row[f"score_{label}"] for label in LABELS},
                    float(full_row["eval_seconds"]) + float(learned_row["eval_seconds"]),
                    int(full_row["visible_tokens"]) + int(learned_row["visible_tokens"]),
                    full_row["gold_label_loss"],
                    full_row["gold_label_tokens"],
                    full_row["gold_label_ppl"],
                    [page.page_id for page in pages],
                    full_row,
                    influence_by_page,
                    1.0,
                    1,
                )
            )
            result_rows.append(progressive)

    summary = summarize_results(result_rows)
    write_csv(output_dir / "page_influence.csv", page_rows)
    write_csv(output_dir / "task_results.csv", result_rows)
    write_csv(output_dir / "summary.csv", summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metadata = {
        "elapsed_seconds": time.perf_counter() - started,
        "tasks": len(task_bundles),
        "page_rows": len(page_rows),
        "result_rows": len(result_rows),
        "positive_page_rate": sum(int(row["causal_label"]) for row in page_rows) / max(1, len(page_rows)),
        "weak_positive_pages": sum(int(row["weak_positive_label"]) for row in page_rows),
        "model_valid": model_info.get("valid", 0),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
