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
    resolve_dtype,
)
from run_causal_memory_planner_v3 import (  # noqa: E402
    FEATURE_NAMES,
    ablation_candidate_ids,
    answer_text_hit,
    compact,
    label_influence_rows,
    make_page_feature_rows,
    select_learned,
    select_lexical,
    select_recent,
    train_logistic_page_model,
    write_csv,
)
from run_fixed_position_kv_mask_planner_v4 import (  # noqa: E402
    FixedPromptBundle,
    build_fixed_prompt,
    select_set_utility_v4,
    span_string,
)
from run_qabs_downstream_kv_retrieval import LABELS  # noqa: E402
from run_task_aware_kv_mixture_v0 import ALL_BUILDERS  # noqa: E402
from run_typed_memory_router_v1_suite import Page, split_pages  # noqa: E402


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
    topk_budgets: str
    max_ablate_pages: int
    positive_delta_threshold: float
    adaptive_labeling: int
    adaptive_mad_scale: float
    weak_positive_if_no_label: int
    logistic_epochs: int
    logistic_lr: float
    log_every: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "KV gather memory planner V5. It prefills the full memory prefix once, gathers selected "
            "page KV ranges, and evaluates query/options with only the gathered KV cache."
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
    parser.add_argument("--seed", type=int, default=2026070301)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--topk_budgets", default="1,2,3,5,8")
    parser.add_argument("--max_ablate_pages", type=int, default=0)
    parser.add_argument("--positive_delta_threshold", type=float, default=0.03)
    parser.add_argument("--adaptive_labeling", type=int, default=1)
    parser.add_argument("--adaptive_mad_scale", type=float, default=1.0)
    parser.add_argument("--weak_positive_if_no_label", type=int, default=1)
    parser.add_argument("--logistic_epochs", type=int, default=220)
    parser.add_argument("--logistic_lr", type=float, default=0.05)
    parser.add_argument("--log_every", type=int, default=5)
    return Config(**vars(parser.parse_args()))


def parse_budgets(spec: str) -> list[int]:
    values = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    return [value for value in values if value > 0]


def encode_piece(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def label_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    return {label: encode_piece(tokenizer, " " + label) for label in LABELS}


def cache_to_legacy(past_key_values: Any) -> tuple[Any, ...]:
    if hasattr(past_key_values, "to_legacy_cache"):
        return tuple(past_key_values.to_legacy_cache())
    if isinstance(past_key_values, list):
        return tuple(past_key_values)
    if isinstance(past_key_values, tuple):
        return past_key_values
    raise TypeError(f"Unsupported cache type: {type(past_key_values)!r}")


def legacy_to_cache_like(legacy: tuple[Any, ...], template: Any) -> Any:
    from_legacy_cache = getattr(type(template), "from_legacy_cache", None)
    if callable(from_legacy_cache):
        return from_legacy_cache(legacy)
    if isinstance(template, list):
        return list(legacy)
    return legacy


def gather_past_key_values(past_key_values: Any, keep_indices: list[int]) -> Any:
    legacy = cache_to_legacy(past_key_values)
    gathered_layers = []
    for layer_cache in legacy:
        key_states, value_states = layer_cache[:2]
        idx = torch.tensor(keep_indices, dtype=torch.long, device=key_states.device)
        gathered_key = key_states.index_select(2, idx).contiguous()
        gathered_value = value_states.index_select(2, idx).contiguous()
        if len(layer_cache) > 2:
            gathered_layers.append((gathered_key, gathered_value, *layer_cache[2:]))
        else:
            gathered_layers.append((gathered_key, gathered_value))
    return legacy_to_cache_like(tuple(gathered_layers), past_key_values)


def page_token_indices(bundle: FixedPromptBundle, page_ids: list[int]) -> list[int]:
    indices: list[int] = []
    for page_id in page_ids:
        start, end = bundle.page_spans[page_id]
        indices.extend(range(start, end))
    return indices


def all_page_token_indices(bundle: FixedPromptBundle) -> set[int]:
    out: set[int] = set()
    for start, end in bundle.page_spans.values():
        out.update(range(start, end))
    return out


def keep_indices_for_pages(bundle: FixedPromptBundle, selected_page_ids: list[int]) -> list[int]:
    page_tokens = set(page_token_indices(bundle, selected_page_ids))
    all_page_tokens = all_page_token_indices(bundle)
    keep = [idx for idx in range(bundle.query_start) if idx not in all_page_tokens or idx in page_tokens]
    return keep


def prefix_token_stats(bundle: FixedPromptBundle, selected_page_ids: list[int]) -> tuple[int, int, int]:
    keep = keep_indices_for_pages(bundle, selected_page_ids)
    selected_page_tokens = len(page_token_indices(bundle, selected_page_ids))
    query_tokens = bundle.prompt_tokens - bundle.query_start
    return len(keep) + query_tokens, selected_page_tokens, query_tokens


@torch.inference_mode()
def prefill_prefix(
    model: torch.nn.Module,
    bundle: FixedPromptBundle,
    input_device: torch.device,
) -> tuple[Any, torch.Tensor, float]:
    prefix_ids = bundle.input_ids[:, : bundle.query_start].to(input_device)
    started = time.perf_counter()
    outputs = model_forward(
        model,
        {
            "input_ids": prefix_ids,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(bundle.query_start, device=input_device),
        },
    )
    elapsed = time.perf_counter() - started
    return outputs.past_key_values, outputs.logits[:, -1, :].detach(), elapsed


@torch.inference_mode()
def run_tokens_with_positions(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    position_start: int,
    input_device: torch.device,
) -> tuple[Any, torch.Tensor]:
    ids = input_ids.to(input_device)
    if ids.shape[-1] == 0:
        raise ValueError("empty suffix")
    outputs = model_forward(
        model,
        {
            "input_ids": ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(position_start, position_start + ids.shape[-1], device=input_device),
        },
    )
    return outputs.past_key_values, outputs.logits[:, -1, :].detach()


@torch.inference_mode()
def score_option_with_positions(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    option: str,
    position_start: int,
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
                "cache_position": torch.tensor([position_start + pos], device=input_device, dtype=torch.long),
            },
        )
        past_key_values = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :].detach()
    return loss, int(ids.shape[-1])


def evaluate_with_cache(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    bundle: FixedPromptBundle,
    prefix_cache: Any,
    answer: str,
) -> tuple[str, dict[str, float], float, float, int, float]:
    started = time.perf_counter()
    query_ids = bundle.input_ids[:, bundle.query_start :].to(input_device)
    query_cache, prev = run_tokens_with_positions(model, query_ids, prefix_cache, bundle.query_start, input_device)
    scores: dict[str, float] = {}
    gold_loss = 0.0
    gold_tokens = 0
    for label in LABELS:
        loss, tokens = score_option_with_positions(
            model,
            tokenizer,
            input_device,
            clone_past_key_values(query_cache),
            prev.detach().clone(),
            label,
            bundle.prompt_tokens,
        )
        scores[label] = -loss
        if label == answer:
            gold_loss = loss
            gold_tokens = tokens
    elapsed = time.perf_counter() - started
    mean_loss = gold_loss / max(1, gold_tokens)
    return max(scores, key=scores.get), scores, elapsed, mean_loss, gold_tokens, math.exp(min(mean_loss, 80.0))


def score_margin(scores: dict[str, float]) -> float:
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) < 2:
        return 0.0
    return float(ordered[0] - ordered[1])


def selected_evidence_hit(task: dict[str, Any], pages: list[Page], selected_ids: list[int]) -> int:
    selected = set(selected_ids)
    return int(any(page.page_id in selected and answer_text_hit(task, page) for page in pages))


def make_result_row(
    task: dict[str, Any],
    pages: list[Page],
    bundle: FixedPromptBundle,
    mode: str,
    split: str,
    budget: int,
    selected_ids: list[int],
    pred: str,
    scores: dict[str, float],
    query_eval_seconds: float,
    kv_gather_seconds: float,
    prefill_seconds: float,
    loss: float,
    tokens: int,
    ppl: float,
    full_row: dict[str, Any],
    influence_by_page: dict[int, dict[str, Any]],
    estimated_causal_recall: float,
) -> dict[str, Any]:
    selected = set(selected_ids)
    visible_tokens, selected_page_tokens, query_tokens = prefix_token_stats(bundle, selected_ids)
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
    online_seconds = query_eval_seconds + kv_gather_seconds
    return {
        "variant": task["variant"],
        "task_id": task["task_id"],
        "split": split,
        "mode": mode,
        "budget": budget,
        "answer": task["answer"],
        "answer_value": task["answer_value"],
        "pred": pred,
        "correct": int(pred == task["answer"]),
        "prefill_seconds": prefill_seconds,
        "kv_gather_seconds": kv_gather_seconds,
        "query_eval_seconds": query_eval_seconds,
        "online_seconds": online_seconds,
        "total_seconds": prefill_seconds + online_seconds,
        "visible_tokens": visible_tokens,
        "raw_prefix_tokens": bundle.query_start,
        "raw_prompt_tokens": bundle.prompt_tokens,
        "selected_page_tokens": selected_page_tokens,
        "query_tokens": query_tokens,
        "keep_fraction": visible_tokens / max(1, bundle.prompt_tokens),
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
        **{f"score_{label}": scores[label] for label in LABELS},
    }


def evaluate_selected_pages(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    task: dict[str, Any],
    pages: list[Page],
    bundle: FixedPromptBundle,
    split: str,
    mode: str,
    budget: int,
    selected_ids: list[int],
    full_prefix_cache: Any,
    prefill_seconds: float,
    full_row: dict[str, Any],
    influence_by_page: dict[int, dict[str, Any]],
    estimated_causal_recall: float,
) -> dict[str, Any]:
    keep_indices = keep_indices_for_pages(bundle, selected_ids)
    started = time.perf_counter()
    selected_cache = gather_past_key_values(full_prefix_cache, keep_indices)
    gather_seconds = time.perf_counter() - started
    pred, scores, query_seconds, loss, tokens, ppl = evaluate_with_cache(
        model,
        tokenizer,
        input_device,
        bundle,
        selected_cache,
        task["answer"],
    )
    return make_result_row(
        task,
        pages,
        bundle,
        mode,
        split,
        budget,
        selected_ids,
        pred,
        scores,
        query_seconds,
        gather_seconds,
        prefill_seconds,
        loss,
        tokens,
        ppl,
        full_row,
        influence_by_page,
        estimated_causal_recall,
    )


def select_oracle_causal(influence_rows: list[dict[str, Any]], topk: int) -> list[int]:
    ranked = sorted(
        influence_rows,
        key=lambda row: (float(row["loss_delta"]), int(row["causal_label"]), -int(row["page_id"])),
        reverse=True,
    )
    return [int(row["page_id"]) for row in ranked[:topk]]


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        budget = str(row.get("budget", ""))
        grouped[(str(row["split"]), str(row["variant"]), str(row["mode"]), budget)].append(row)
        grouped[(str(row["split"]), "ALL", str(row["mode"]), budget)].append(row)
    summary = []
    for (split, variant, mode, budget), subset in sorted(grouped.items()):
        n = max(1, len(subset))
        total_tokens = sum(int(row["gold_label_tokens"]) for row in subset)
        total_loss = sum(float(row["gold_label_loss"]) * int(row["gold_label_tokens"]) for row in subset)
        mean_loss = total_loss / max(1, total_tokens)
        summary.append(
            {
                "split": split,
                "variant": variant,
                "mode": mode,
                "budget": budget,
                "tasks": len(subset),
                "accuracy": sum(int(row["correct"]) for row in subset) / n,
                "gold_label_ppl": math.exp(min(mean_loss, 80.0)),
                "mean_prefill_seconds": sum(float(row["prefill_seconds"]) for row in subset) / n,
                "mean_kv_gather_seconds": sum(float(row["kv_gather_seconds"]) for row in subset) / n,
                "mean_query_eval_seconds": sum(float(row["query_eval_seconds"]) for row in subset) / n,
                "mean_online_seconds": sum(float(row["online_seconds"]) for row in subset) / n,
                "mean_total_seconds": sum(float(row["total_seconds"]) for row in subset) / n,
                "mean_visible_tokens": sum(int(row["visible_tokens"]) for row in subset) / n,
                "mean_raw_prompt_tokens": sum(int(row["raw_prompt_tokens"]) for row in subset) / n,
                "mean_keep_fraction": sum(float(row["keep_fraction"]) for row in subset) / n,
                "mean_selected_page_tokens": sum(int(row["selected_page_tokens"]) for row in subset) / n,
                "mean_selected_pages": sum(int(row["selected_pages"]) for row in subset) / n,
                "evidence_hit_rate": sum(int(row["evidence_hit"]) for row in subset) / n,
                "teacher_correct_rate": sum(int(row["teacher_correct"]) for row in subset) / n,
                "mean_margin": sum(float(row["margin"]) for row in subset) / n,
                "mean_causal_label_recall": sum(float(row["causal_label_recall"]) for row in subset) / n,
                "mean_causal_label_precision": sum(float(row["causal_label_precision"]) for row in subset) / n,
                "mean_causal_positive_mass_recall": sum(float(row["causal_positive_mass_recall"]) for row in subset) / n,
                "mean_estimated_causal_recall": sum(float(row["estimated_causal_recall"]) for row in subset) / n,
            }
        )
    return summary


def full_result_row(
    task: dict[str, Any],
    pages: list[Page],
    bundle: FixedPromptBundle,
    split: str,
    prefill_seconds: float,
    pred: str,
    scores: dict[str, float],
    query_seconds: float,
    loss: float,
    tokens: int,
    ppl: float,
) -> dict[str, Any]:
    selected_ids = [page.page_id for page in pages]
    placeholder_full = {
        "pred": pred,
        "correct": int(pred == task["answer"]),
        "gold_label_ppl": ppl,
    }
    return make_result_row(
        task,
        pages,
        bundle,
        "full_kv_cache",
        split,
        -1,
        selected_ids,
        pred,
        scores,
        query_seconds,
        0.0,
        prefill_seconds,
        loss,
        tokens,
        ppl,
        placeholder_full,
        {},
        1.0,
    )


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    budgets = parse_budgets(config.topk_budgets)

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
            bundle = build_fixed_prompt(tokenizer, task, pages)
            if (task_idx + 1) % config.log_every == 0 or task_idx == 0:
                print(f"{variant} {task_idx + 1}/{config.tasks_per_variant} split={split}", flush=True)

            full_prefix_cache, _, prefill_seconds = prefill_prefix(model, bundle, input_device)
            pred, scores, query_seconds, loss, tokens, ppl = evaluate_with_cache(
                model,
                tokenizer,
                input_device,
                bundle,
                clone_past_key_values(full_prefix_cache),
                task["answer"],
            )
            full_row = full_result_row(task, pages, bundle, split, prefill_seconds, pred, scores, query_seconds, loss, tokens, ppl)

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
                        "ablated_pred": "",
                        "ablated_correct": 0,
                        "ablated_gold_label_loss": 0.0,
                        "ablated_gold_label_ppl": 0.0,
                        "ablation_eval_seconds": 0.0,
                        "ablation_gather_seconds": 0.0,
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
            all_ids = [page.page_id for page in pages]
            for page_id in ablate_ids:
                selected_ids = [item for item in all_ids if item != page_id]
                keep = keep_indices_for_pages(bundle, selected_ids)
                gather_started = time.perf_counter()
                ablated_cache = gather_past_key_values(full_prefix_cache, keep)
                gather_seconds = time.perf_counter() - gather_started
                ablated_pred, _, eval_seconds, ablated_loss, _, ablated_ppl = evaluate_with_cache(
                    model,
                    tokenizer,
                    input_device,
                    bundle,
                    ablated_cache,
                    task["answer"],
                )
                row = row_by_page[page_id]
                row["ablated_pred"] = ablated_pred
                row["ablated_correct"] = int(ablated_pred == task["answer"])
                row["ablated_gold_label_loss"] = ablated_loss
                row["ablated_gold_label_ppl"] = ablated_ppl
                row["ablation_eval_seconds"] = eval_seconds
                row["ablation_gather_seconds"] = gather_seconds
                row["loss_delta"] = float(ablated_loss) - float(full_row["gold_label_loss"])
                row["ppl_delta"] = float(ablated_ppl) - float(full_row["gold_label_ppl"])

            label_influence_rows(
                task_page_rows,
                config.positive_delta_threshold,
                config.adaptive_labeling,
                config.adaptive_mad_scale,
                config.weak_positive_if_no_label,
            )
            page_rows.extend(task_page_rows)
            influence_by_page = {int(row["page_id"]): row for row in task_page_rows}
            full_row = dict(full_row)
            full_row["causal_positive_pages"] = sum(int(row["causal_label"]) for row in task_page_rows)
            full_row["causal_label_recall"] = 1.0
            full_row["causal_label_precision"] = 0.0
            full_row["causal_positive_mass_recall"] = 1.0
            result_rows.append(full_row)
            task_bundles.append(
                {
                    "task": task,
                    "pages": pages,
                    "bundle": bundle,
                    "split": split,
                    "prefill_seconds": prefill_seconds,
                    "full_prefix_cache": full_prefix_cache,
                    "full_row": full_row,
                    "page_rows": task_page_rows,
                    "influence_by_page": influence_by_page,
                }
            )

    model_info = train_logistic_page_model(page_rows, config)
    (output_dir / "learned_page_model.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    for record in task_bundles:
        task = record["task"]
        pages = record["pages"]
        bundle = record["bundle"]
        split = record["split"]
        full_prefix_cache = record["full_prefix_cache"]
        prefill_seconds = float(record["prefill_seconds"])
        full_row = record["full_row"]
        task_page_rows = record["page_rows"]
        influence_by_page = record["influence_by_page"]

        for budget in budgets:
            selectors: list[tuple[str, list[int], float]] = []
            selectors.append(("recent_kv_gather_topk", select_recent(pages, budget), 0.0))
            selectors.append(("lexical_kv_gather_topk", select_lexical(task, pages, budget), 0.0))
            learned_ids, learned_recall = select_learned(task_page_rows, model_info, budget)
            selectors.append(("learned_causal_kv_gather_topk", learned_ids, learned_recall))
            selectors.append(("oracle_causal_kv_gather_topk", select_oracle_causal(task_page_rows, budget), 1.0))
            set_ids, set_recall, set_source = select_set_utility_v4(task, pages, task_page_rows, model_info, budget)
            selectors.append(("set_utility_kv_gather_v5", set_ids, set_recall))
            for mode, selected_ids, estimated_recall in selectors:
                row = evaluate_selected_pages(
                    model,
                    tokenizer,
                    input_device,
                    task,
                    pages,
                    bundle,
                    split,
                    mode,
                    budget,
                    selected_ids,
                    full_prefix_cache,
                    prefill_seconds,
                    full_row,
                    influence_by_page,
                    estimated_recall,
                )
                if mode == "set_utility_kv_gather_v5":
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
        "labeling": "kv_gather_drop_page_from_full_prefix_cache",
        "budgets": budgets,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
