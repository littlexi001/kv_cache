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
    model_forward,
    pick_input_device,
    resolve_dtype,
)
from run_causal_memory_planner_v3 import (  # noqa: E402
    FEATURE_NAMES,
    ablation_candidate_ids,
    answer_text_hit,
    compact,
    label_influence_rows,
    learned_probability,
    make_page_feature_rows,
    minmax,
    score_margin,
    select_learned,
    select_lexical,
    select_recent,
    train_logistic_page_model,
    typed_role_prior,
    write_csv,
)
from run_qabs_downstream_kv_retrieval import LABELS  # noqa: E402
from run_task_aware_kv_mixture_v0 import ALL_BUILDERS  # noqa: E402
from run_typed_memory_router_v1_suite import Page, page_score, split_pages  # noqa: E402


MASK_VALUE = -1.0e4


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
    log_every: int


@dataclass
class FixedPromptBundle:
    input_ids: torch.Tensor
    prompt_text: str
    page_spans: dict[int, tuple[int, int]]
    query_start: int
    prompt_tokens: int
    non_page_tokens: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-position KV visibility mask planner V4. Labels causal pages by masking "
            "page token ranges from query/answer attention while keeping token positions fixed."
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
    parser.add_argument("--seed", type=int, default=2026070209)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--topk_pages", type=int, default=5)
    parser.add_argument("--max_ablate_pages", type=int, default=0)
    parser.add_argument("--positive_delta_threshold", type=float, default=0.03)
    parser.add_argument("--adaptive_labeling", type=int, default=1)
    parser.add_argument("--adaptive_mad_scale", type=float, default=1.0)
    parser.add_argument("--weak_positive_if_no_label", type=int, default=1)
    parser.add_argument("--logistic_epochs", type=int, default=220)
    parser.add_argument("--logistic_lr", type=float, default=0.05)
    parser.add_argument("--log_every", type=int, default=5)
    return Config(**vars(parser.parse_args()))


def encode_piece(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def build_fixed_prompt(tokenizer: Any, task: dict[str, Any], pages: list[Page]) -> FixedPromptBundle:
    ids: list[int] = []
    text_parts: list[str] = []
    spans: dict[int, tuple[int, int]] = {}

    def append(text: str) -> tuple[int, int]:
        start = len(ids)
        part_ids = encode_piece(tokenizer, text)
        ids.extend(int(item) for item in part_ids)
        text_parts.append(text)
        return start, len(ids)

    append("\nFixed-position memory context. Page tokens keep their original positions.\n")
    for page in pages:
        start, end = append(f"[page {page.page_id}; status={page.status}] {compact(page.text)}\n")
        spans[page.page_id] = (start, end)
    append(
        "\nUse only memory ranges visible to the query. Prefer current/active pages over old/superseded pages. "
        "Answer with the option letter only.\n"
    )
    query_start = len(ids)
    append(task["query"])
    input_ids = torch.tensor([ids], dtype=torch.long)
    page_token_count = sum(end - start for start, end in spans.values())
    return FixedPromptBundle(
        input_ids=input_ids,
        prompt_text="".join(text_parts),
        page_spans=spans,
        query_start=query_start,
        prompt_tokens=len(ids),
        non_page_tokens=len(ids) - page_token_count,
    )


def build_additive_mask(
    batch_size: int,
    seq_len: int,
    query_start: int,
    blocked_spans: list[tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.zeros((batch_size, 1, seq_len, seq_len), device=device, dtype=dtype)
    upper = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
    mask[:, :, upper] = MASK_VALUE
    row_start = max(0, min(query_start, seq_len))
    for start, end in blocked_spans:
        start = max(0, min(start, seq_len))
        end = max(start, min(end, seq_len))
        if end > start and row_start < seq_len:
            mask[:, :, row_start:, start:end] = MASK_VALUE
    return mask


def label_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    return {label: encode_piece(tokenizer, " " + label) for label in LABELS}


@torch.inference_mode()
def evaluate_fixed_mask_prompt(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    bundle: FixedPromptBundle,
    blocked_page_ids: set[int],
    answer: str,
) -> tuple[str, dict[str, float], float, float, int, float]:
    started = time.perf_counter()
    label_ids = label_token_ids(tokenizer)
    max_option_len = max(len(ids) for ids in label_ids.values())
    prompt_ids = bundle.input_ids[0].tolist()
    prompt_len = len(prompt_ids)
    seq_len = prompt_len + max_option_len
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    input_rows = []
    option_lengths: dict[str, int] = {}
    for label in LABELS:
        option = label_ids[label]
        option_lengths[label] = len(option)
        input_rows.append(prompt_ids + option + [int(pad_id)] * (max_option_len - len(option)))
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=input_device)

    blocked_spans = [bundle.page_spans[page_id] for page_id in sorted(blocked_page_ids)]
    mask_dtype = torch.float32
    attention_mask = build_additive_mask(
        batch_size=len(LABELS),
        seq_len=seq_len,
        query_start=bundle.query_start,
        blocked_spans=blocked_spans,
        device=input_device,
        dtype=mask_dtype,
    )
    outputs = model_forward(
        model,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
        },
    )
    logits = outputs.logits
    scores: dict[str, float] = {}
    gold_loss = 0.0
    gold_tokens = 0
    for row_idx, label in enumerate(LABELS):
        option = torch.tensor(label_ids[label], dtype=torch.long, device=input_device)
        loss = 0.0
        for offset in range(option_lengths[label]):
            pred_pos = prompt_len + offset - 1
            target = option[offset : offset + 1]
            loss += float(F.cross_entropy(logits[row_idx : row_idx + 1, pred_pos, :].float(), target, reduction="sum").item())
        scores[label] = -loss
        if label == answer:
            gold_loss = loss
            gold_tokens = option_lengths[label]
    elapsed = time.perf_counter() - started
    mean_loss = gold_loss / max(1, gold_tokens)
    return max(scores, key=scores.get), scores, elapsed, mean_loss, gold_tokens, math.exp(min(mean_loss, 80.0))


def effective_visible_tokens(bundle: FixedPromptBundle, selected_ids: list[int]) -> int:
    total = bundle.non_page_tokens
    for page_id in selected_ids:
        start, end = bundle.page_spans[page_id]
        total += end - start
    return total


def span_string(bundle: FixedPromptBundle, selected_ids: list[int]) -> str:
    return ";".join(f"{page_id}:{bundle.page_spans[page_id][0]}-{bundle.page_spans[page_id][1]}" for page_id in selected_ids)


def selected_evidence_hit(task: dict[str, Any], pages: list[Page], selected_ids: list[int]) -> int:
    selected = set(selected_ids)
    return int(any(page.page_id in selected and answer_text_hit(task, page) for page in pages))


def make_result_row(
    task: dict[str, Any],
    pages: list[Page],
    bundle: FixedPromptBundle,
    mode: str,
    split: str,
    selected_ids: list[int],
    pred: str,
    scores: dict[str, float],
    elapsed: float,
    loss: float,
    tokens: int,
    ppl: float,
    full_row: dict[str, Any],
    influence_by_page: dict[int, dict[str, Any]],
    estimated_causal_recall: float,
    fallback: int = 0,
) -> dict[str, Any]:
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
        "split": split,
        "mode": mode,
        "answer": task["answer"],
        "answer_value": task["answer_value"],
        "pred": pred,
        "correct": int(pred == task["answer"]),
        "eval_seconds": elapsed,
        "visible_tokens": effective_visible_tokens(bundle, selected_ids),
        "raw_prompt_tokens": bundle.prompt_tokens,
        "selected_page_tokens": effective_visible_tokens(bundle, selected_ids) - bundle.non_page_tokens,
        "gold_label_loss": loss,
        "gold_label_tokens": tokens,
        "gold_label_ppl": ppl,
        "margin": score_margin(scores),
        "selected_pages": len(selected_ids),
        "selected_page_ids": ",".join(str(page_id) for page_id in selected_ids),
        "selected_token_ranges": span_string(bundle, selected_ids),
        "page_count": len(pages),
        "evidence_hit": selected_evidence_hit(task, pages, selected_ids),
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


def evaluate_mask_mode(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    task: dict[str, Any],
    pages: list[Page],
    bundle: FixedPromptBundle,
    mode: str,
    split: str,
    selected_ids: list[int],
    full_row: dict[str, Any],
    influence_by_page: dict[int, dict[str, Any]],
    estimated_causal_recall: float = 0.0,
) -> dict[str, Any]:
    selected = set(selected_ids)
    blocked = {page.page_id for page in pages if page.page_id not in selected}
    pred, scores, elapsed, loss, tokens, ppl = evaluate_fixed_mask_prompt(
        model,
        tokenizer,
        input_device,
        bundle,
        blocked,
        task["answer"],
    )
    return make_result_row(
        task,
        pages,
        bundle,
        mode,
        split,
        selected_ids,
        pred,
        scores,
        elapsed,
        loss,
        tokens,
        ppl,
        full_row,
        influence_by_page,
        estimated_causal_recall,
    )


def full_mask_row(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    task: dict[str, Any],
    pages: list[Page],
    bundle: FixedPromptBundle,
    split: str,
) -> dict[str, Any]:
    selected_ids = [page.page_id for page in pages]
    pred, scores, elapsed, loss, tokens, ppl = evaluate_fixed_mask_prompt(
        model,
        tokenizer,
        input_device,
        bundle,
        set(),
        task["answer"],
    )
    row = {
        "variant": task["variant"],
        "task_id": task["task_id"],
        "split": split,
        "mode": "full_fixed_position_mask_teacher",
        "answer": task["answer"],
        "answer_value": task["answer_value"],
        "pred": pred,
        "correct": int(pred == task["answer"]),
        "eval_seconds": elapsed,
        "visible_tokens": bundle.prompt_tokens,
        "raw_prompt_tokens": bundle.prompt_tokens,
        "selected_page_tokens": bundle.prompt_tokens - bundle.non_page_tokens,
        "gold_label_loss": loss,
        "gold_label_tokens": tokens,
        "gold_label_ppl": ppl,
        "margin": score_margin(scores),
        "selected_pages": len(selected_ids),
        "selected_page_ids": ",".join(str(page_id) for page_id in selected_ids),
        "selected_token_ranges": span_string(bundle, selected_ids),
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
    return row


def select_oracle_causal(influence_rows: list[dict[str, Any]], topk: int) -> list[int]:
    ranked = sorted(
        influence_rows,
        key=lambda row: (float(row["loss_delta"]), int(row["causal_label"]), -int(row["page_id"])),
        reverse=True,
    )
    return [int(row["page_id"]) for row in ranked[:topk]]


def fill_to_topk(seed_ids: list[int], candidate_ids: list[int], topk: int) -> list[int]:
    selected: list[int] = []
    for page_id in seed_ids + candidate_ids:
        if page_id not in selected:
            selected.append(page_id)
        if len(selected) >= topk:
            break
    return selected


def role_pages(task: dict[str, Any], pages: list[Page]) -> list[int]:
    query = task["query"].lower()
    ranked = sorted(pages, key=lambda page: (typed_role_prior(task, page), page_score(page, task["query"]), page.page_id), reverse=True)
    if "recent chat" in query or "assistant reply" in query:
        return [page.page_id for page in ranked if "preferred assistant reply" in page.text.lower()]
    if "active badge color" in query:
        return [page.page_id for page in ranked if page.status == "current" and "badge color" in page.text.lower()]
    if "artifact" in query:
        return [
            page.page_id
            for page in ranked
            if page.status == "current" and ("routing page" in page.text.lower() or "artifact memo" in page.text.lower())
        ]
    if "most often" in query or "across current reports" in query:
        return [page.page_id for page in ranked if page.status == "current" and "theme=" in page.text.lower()]
    if "highest current priority score" in query:
        return [page.page_id for page in ranked if page.status == "current" and "priority_score=" in page.text.lower()]
    return []


def role_coverage(task: dict[str, Any], pages: list[Page], selected_ids: list[int], topk: int) -> float:
    selected = set(selected_ids)
    roles = role_pages(task, pages)
    if not roles:
        return 0.0
    query = task["query"].lower()
    if "artifact" in query:
        selected_pages = [page for page in pages if page.page_id in selected]
        has_bridge = any("routing page" in page.text.lower() and page.status == "current" for page in selected_pages)
        has_memo = any("artifact memo" in page.text.lower() and page.status == "current" for page in selected_pages)
        return (float(has_bridge) + float(has_memo)) / 2.0
    needed = min(topk, len(roles))
    return len(selected & set(roles[:topk])) / max(1, needed)


def select_set_utility_v4(
    task: dict[str, Any],
    pages: list[Page],
    task_page_rows: list[dict[str, Any]],
    model_info: dict[str, Any],
    topk: int,
) -> tuple[list[int], float, str]:
    learned_ids, learned_recall = select_learned(task_page_rows, model_info, topk)
    lexical_ids = select_lexical(task, pages, topk)
    recent_ids = select_recent(pages, topk)
    role_ids = fill_to_topk(role_pages(task, pages), lexical_ids + learned_ids + recent_ids, topk)
    hybrid_seed = fill_to_topk(role_pages(task, pages), learned_ids + lexical_ids + recent_ids, topk)
    candidates = {
        "learned": learned_ids,
        "lexical": lexical_ids,
        "recent": recent_ids,
        "typed_role_set": role_ids,
        "hybrid_role_learned": hybrid_seed,
    }

    learned_prob = {int(row["page_id"]): learned_probability(row, model_info) for row in task_page_rows}
    total_prob = sum(learned_prob.values())
    lexical_scores = {page.page_id: max(0.0, float(page_score(page, task["query"]))) for page in pages}
    total_lexical = sum(lexical_scores.values())

    best_name = ""
    best_ids: list[int] = []
    best_score = -1.0
    best_estimated = 0.0
    for name, ids in candidates.items():
        causal_mass = sum(learned_prob.get(page_id, 0.0) for page_id in ids) / max(1e-8, total_prob)
        lexical_mass = sum(lexical_scores.get(page_id, 0.0) for page_id in ids) / max(1e-8, total_lexical)
        coverage = role_coverage(task, pages, ids, topk)
        score = 0.42 * causal_mass + 0.23 * lexical_mass + 0.35 * coverage
        if score > best_score:
            best_score = score
            best_name = name
            best_ids = ids
            best_estimated = causal_mass
    if not best_ids:
        best_ids, best_estimated, best_name = learned_ids, learned_recall, "learned_fallback"
    return best_ids, best_estimated, best_name


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
                "mean_raw_prompt_tokens": sum(int(row["raw_prompt_tokens"]) for row in subset) / n,
                "mean_selected_page_tokens": sum(int(row["selected_page_tokens"]) for row in subset) / n,
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
    model.config.use_cache = False
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
            bundle = build_fixed_prompt(tokenizer, task, pages)
            if (task_idx + 1) % config.log_every == 0 or task_idx == 0:
                print(f"{variant} {task_idx + 1}/{config.tasks_per_variant} split={split}", flush=True)

            full_row = full_mask_row(model, tokenizer, input_device, task, pages, bundle, split)
            feature_by_page = make_page_feature_rows(task, pages)
            ablate_ids = ablation_candidate_ids(task, pages, config.max_ablate_pages)
            task_page_rows: list[dict[str, Any]] = []
            for page in pages:
                features = feature_by_page[page.page_id]
                start, end = bundle.page_spans[page.page_id]
                task_page_rows.append(
                    {
                        "variant": variant,
                        "task_id": task_idx,
                        "split": split,
                        "page_id": page.page_id,
                        "page_status": page.status,
                        "page_text": compact(page.text),
                        "page_token_start": start,
                        "page_token_end": end,
                        "page_token_count": end - start,
                        "answer_text_hit": answer_text_hit(task, page),
                        "was_ablated": int(page.page_id in ablate_ids),
                        "teacher_pred": full_row["pred"],
                        "teacher_correct": full_row["correct"],
                        "teacher_gold_label_loss": full_row["gold_label_loss"],
                        "teacher_gold_label_ppl": full_row["gold_label_ppl"],
                        "masked_pred": "",
                        "masked_correct": 0,
                        "masked_gold_label_loss": 0.0,
                        "masked_gold_label_ppl": 0.0,
                        "mask_eval_seconds": 0.0,
                        "loss_delta": 0.0,
                        "ppl_delta": 0.0,
                        "causal_label": 0,
                        "negative_influence_label": 0,
                        "task_label_threshold": 0.0,
                        "weak_positive_label": 0,
                        **{f"feature_{name}": features[name] for name in FEATURE_NAMES},
                    }
                )

            row_by_page = {int(row["page_id"]): row for row in task_page_rows}
            for page_id in ablate_ids:
                pred, _, elapsed, loss, _, ppl = evaluate_fixed_mask_prompt(
                    model,
                    tokenizer,
                    input_device,
                    bundle,
                    {page_id},
                    task["answer"],
                )
                row = row_by_page[page_id]
                row["masked_pred"] = pred
                row["masked_correct"] = int(pred == task["answer"])
                row["ablated_pred"] = pred
                row["ablated_correct"] = int(pred == task["answer"])
                row["masked_gold_label_loss"] = loss
                row["masked_gold_label_ppl"] = ppl
                row["ablation_eval_seconds"] = elapsed
                row["mask_eval_seconds"] = elapsed
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
            full_result = make_result_row(
                task,
                pages,
                bundle,
                "full_fixed_position_mask_teacher",
                split,
                [page.page_id for page in pages],
                full_row["pred"],
                {label: full_row[f"score_{label}"] for label in LABELS},
                float(full_row["eval_seconds"]),
                float(full_row["gold_label_loss"]),
                int(full_row["gold_label_tokens"]),
                float(full_row["gold_label_ppl"]),
                full_row,
                influence_by_page,
                1.0,
            )
            result_rows.append(full_result)
            task_bundles.append(
                {
                    "task": task,
                    "pages": pages,
                    "bundle": bundle,
                    "split": split,
                    "full_row": full_row,
                    "page_rows": task_page_rows,
                    "influence_by_page": influence_by_page,
                }
            )

    model_info = train_logistic_page_model(page_rows, config)
    (output_dir / "learned_page_model.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    for bundle_record in task_bundles:
        task = bundle_record["task"]
        pages = bundle_record["pages"]
        bundle = bundle_record["bundle"]
        split = bundle_record["split"]
        full_row = bundle_record["full_row"]
        task_page_rows = bundle_record["page_rows"]
        influence_by_page = bundle_record["influence_by_page"]

        recent_ids = select_recent(pages, config.topk_pages)
        lexical_ids = select_lexical(task, pages, config.topk_pages)
        learned_ids, learned_estimated = select_learned(task_page_rows, model_info, config.topk_pages)
        oracle_ids = select_oracle_causal(task_page_rows, config.topk_pages)
        set_ids, set_estimated, set_source = select_set_utility_v4(task, pages, task_page_rows, model_info, config.topk_pages)

        for mode, selected_ids, estimated in [
            ("recent_kv_mask_topk_pages", recent_ids, 0.0),
            ("lexical_kv_mask_topk_pages", lexical_ids, 0.0),
            ("learned_causal_kv_mask_topk_pages", learned_ids, learned_estimated),
            ("oracle_causal_kv_mask_topk_pages", oracle_ids, 1.0),
            ("set_utility_kv_mask_v4", set_ids, set_estimated),
        ]:
            row = evaluate_mask_mode(
                model,
                tokenizer,
                input_device,
                task,
                pages,
                bundle,
                mode,
                split,
                selected_ids,
                full_row,
                influence_by_page,
                estimated,
            )
            if mode == "set_utility_kv_mask_v4":
                row["set_utility_source"] = set_source
            result_rows.append(row)

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
        "labeling": "fixed_position_query_side_kv_visibility_mask",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
