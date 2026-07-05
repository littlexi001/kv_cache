from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    clone_past_key_values,
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
)
from run_fixed_position_kv_mask_planner_v4 import build_fixed_prompt, select_set_utility_v4  # noqa: E402
from run_kv_gather_planner_v5 import (  # noqa: E402
    evaluate_selected_pages,
    evaluate_with_cache,
    full_result_row,
    gather_past_key_values,
    keep_indices_for_pages,
    parse_budgets,
    prefill_prefix,
    select_oracle_causal,
)
from run_task_aware_kv_mixture_v0 import ALL_BUILDERS  # noqa: E402
from run_typed_memory_router_v1_suite import split_pages  # noqa: E402


DEPLOYABLE_MODES = [
    "recent_kv_gather_topk",
    "lexical_kv_gather_topk",
    "learned_causal_kv_gather_topk",
    "set_utility_kv_gather_v7",
    "full_kv_cache",
]
UPPER_BOUND_MODES = ["oracle_causal_kv_gather_topk"]

SLA_SPECS: dict[str, dict[str, float]] = {
    "quality": {"incorrect": 1000.0, "loss": 1.0, "token": 0.0, "online_seconds": 0.0},
    "balanced": {"incorrect": 1000.0, "loss": 1.0, "token": 0.12, "online_seconds": 0.04},
    "speed": {"incorrect": 1000.0, "loss": 0.55, "token": 0.45, "online_seconds": 0.18},
}


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
    accept_margin: float
    accept_estimated_causal_recall: float
    log_every: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Oracle-regret KV memory planner V7. It emits causal page influence labels, "
            "per-strategy oracle regret labels, held-out mixed-workload planner results, "
            "and real query-side KV-gather latency."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--variants",
        default="casual_recent,temporal_fact,multihop_bridge,summary_theme,compare_score",
    )
    parser.add_argument("--tasks_per_variant", type=int, default=6)
    parser.add_argument("--train_fraction", type=float, default=0.5)
    parser.add_argument("--distractor_pages", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026070307)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--topk_budgets", default="1,3,5")
    parser.add_argument("--max_ablate_pages", type=int, default=12)
    parser.add_argument("--positive_delta_threshold", type=float, default=0.03)
    parser.add_argument("--adaptive_labeling", type=int, default=1)
    parser.add_argument("--adaptive_mad_scale", type=float, default=1.0)
    parser.add_argument("--weak_positive_if_no_label", type=int, default=1)
    parser.add_argument("--logistic_epochs", type=int, default=220)
    parser.add_argument("--logistic_lr", type=float, default=0.05)
    parser.add_argument("--accept_margin", type=float, default=0.35)
    parser.add_argument("--accept_estimated_causal_recall", type=float, default=0.30)
    parser.add_argument("--log_every", type=int, default=5)
    return Config(**vars(parser.parse_args()))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def task_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["variant"]), int(row["task_id"]), int(row.get("budget", -1))


def objective_value(row: dict[str, Any], spec: dict[str, float]) -> float:
    incorrect = 1 - int(row["correct"])
    loss = float(row["gold_label_loss"])
    keep_fraction = float(row.get("keep_fraction", 1.0))
    online_seconds = float(row.get("online_seconds", row.get("query_eval_seconds", row.get("eval_seconds", 0.0))))
    return (
        spec["incorrect"] * incorrect
        + spec["loss"] * loss
        + spec["token"] * math.log1p(max(0.0, keep_fraction))
        + spec["online_seconds"] * online_seconds
    )


def pareto_flags(rows: list[dict[str, Any]]) -> dict[str, int]:
    flags: dict[str, int] = {}
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = (
                int(other["correct"]) >= int(row["correct"])
                and float(other["gold_label_loss"]) <= float(row["gold_label_loss"]) + 1e-9
                and float(other.get("keep_fraction", 1.0)) <= float(row.get("keep_fraction", 1.0)) + 1e-9
                and float(other.get("online_seconds", 0.0)) <= float(row.get("online_seconds", 0.0)) + 1e-9
            )
            strictly_better = (
                int(other["correct"]) > int(row["correct"])
                or float(other["gold_label_loss"]) < float(row["gold_label_loss"]) - 1e-9
                or float(other.get("keep_fraction", 1.0)) < float(row.get("keep_fraction", 1.0)) - 1e-9
                or float(other.get("online_seconds", 0.0)) < float(row.get("online_seconds", 0.0)) - 1e-9
            )
            if no_worse and strictly_better:
                dominated = True
                break
        flags[str(row["mode"])] = int(not dominated)
    return flags


def add_oracle_regret_labels(strategy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in strategy_rows:
        if int(row.get("budget", -1)) < 0:
            continue
        grouped[task_key(row)].append(row)

    for key, rows in grouped.items():
        flags = pareto_flags(rows)
        deployable = [row for row in rows if row["mode"] in DEPLOYABLE_MODES]
        all_rows = [row for row in rows if row["mode"] in DEPLOYABLE_MODES + UPPER_BOUND_MODES]
        for sla, spec in SLA_SPECS.items():
            deployable_oracle = min(deployable, key=lambda row: objective_value(row, spec))
            all_oracle = min(all_rows, key=lambda row: objective_value(row, spec))
            for row in all_rows:
                objective = objective_value(row, spec)
                deployable_objective = objective_value(deployable_oracle, spec)
                all_objective = objective_value(all_oracle, spec)
                labels.append(
                    {
                        "variant": key[0],
                        "task_id": key[1],
                        "budget": key[2],
                        "split": row["split"],
                        "sla": sla,
                        "mode": row["mode"],
                        "correct": row["correct"],
                        "gold_label_loss": row["gold_label_loss"],
                        "gold_label_ppl": row["gold_label_ppl"],
                        "keep_fraction": row.get("keep_fraction", 1.0),
                        "online_seconds": row.get("online_seconds", 0.0),
                        "objective": objective,
                        "deployable_oracle_mode": deployable_oracle["mode"],
                        "all_oracle_mode": all_oracle["mode"],
                        "deployable_objective_regret": objective - deployable_objective,
                        "all_objective_regret": objective - all_objective,
                        "correct_regret": int(deployable_oracle["correct"]) - int(row["correct"]),
                        "loss_regret": float(row["gold_label_loss"]) - float(deployable_oracle["gold_label_loss"]),
                        "token_regret": float(row.get("keep_fraction", 1.0)) - float(deployable_oracle.get("keep_fraction", 1.0)),
                        "online_seconds_regret": float(row.get("online_seconds", 0.0))
                        - float(deployable_oracle.get("online_seconds", 0.0)),
                        "is_deployable_oracle": int(row["mode"] == deployable_oracle["mode"]),
                        "is_all_oracle": int(row["mode"] == all_oracle["mode"]),
                        "is_pareto": flags.get(str(row["mode"]), 0),
                    }
                )
    return labels


def train_variant_plans(regret_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, list[str]]]]:
    train = [row for row in regret_rows if row["split"] == "train" and row["mode"] in DEPLOYABLE_MODES]
    grouped: dict[tuple[str, int, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in train:
        grouped[(str(row["variant"]), int(row["budget"]), str(row["sla"]))][str(row["mode"])].append(float(row["objective"]))

    plans: dict[str, dict[str, dict[str, list[str]]]] = defaultdict(lambda: defaultdict(dict))
    for (variant, budget, sla), by_mode in grouped.items():
        scored = [(sum(values) / max(1, len(values)), mode) for mode, values in by_mode.items()]
        scored.sort(key=lambda item: (item[0], item[1]))
        plan = [mode for _, mode in scored]
        plans[sla][str(budget)][variant] = plan
    return {sla: {budget: dict(by_variant) for budget, by_variant in by_budget.items()} for sla, by_budget in plans.items()}


def row_accepts(row: dict[str, Any], config: Config, is_last: bool) -> tuple[bool, str]:
    del is_last
    margin = float(row.get("margin", 0.0))
    recall = float(row.get("estimated_causal_recall", 0.0))
    if margin >= config.accept_margin and recall >= config.accept_estimated_causal_recall:
        return True, "margin_and_causal_recall"
    if margin >= 1.5 * config.accept_margin:
        return True, "high_margin"
    return False, "low_margin_or_low_estimated_recall"


def make_planner_rows(
    strategy_rows: list[dict[str, Any]],
    regret_rows: list[dict[str, Any]],
    plans: dict[str, dict[str, dict[str, list[str]]]],
    config: Config,
) -> list[dict[str, Any]]:
    del regret_rows
    by_task_budget: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in strategy_rows:
        if int(row.get("budget", -1)) < 0:
            continue
        by_task_budget[task_key(row)][str(row["mode"])] = row

    out: list[dict[str, Any]] = []
    for key, by_mode in sorted(by_task_budget.items()):
        variant, task_id, budget = key
        split = next(iter(by_mode.values()))["split"]
        for sla, spec in SLA_SPECS.items():
            plan = plans.get(sla, {}).get(str(budget), {}).get(variant)
            if not plan:
                plan = ["set_utility_kv_gather_v7", "learned_causal_kv_gather_topk", "lexical_kv_gather_topk", "full_kv_cache"]
            plan = [mode for mode in plan if mode in by_mode]
            if "full_kv_cache" not in plan and "full_kv_cache" in by_mode:
                plan.append("full_kv_cache")

            tried: list[dict[str, Any]] = []
            selected = by_mode[plan[0]]
            accept_reason = "regret_trained_fallback"
            for idx, mode in enumerate(plan):
                row = by_mode[mode]
                tried.append(row)
                accepted, reason = row_accepts(row, config, idx == len(plan) - 1)
                if accepted:
                    selected = row
                    accept_reason = reason
                    break

            deployable_rows = [row for row in by_mode.values() if row["mode"] in DEPLOYABLE_MODES]
            oracle = min(deployable_rows, key=lambda row: objective_value(row, spec))
            planner = dict(selected)
            planner["mode"] = "risk_calibrated_progressive_planner_v7"
            planner["sla"] = sla
            planner["plan"] = ">".join(plan)
            planner["tried_modes"] = ">".join(str(row["mode"]) for row in tried)
            planner["tried_steps"] = len(tried)
            planner["accept_reason"] = accept_reason
            planner["split"] = split
            planner["variant"] = variant
            planner["task_id"] = task_id
            planner["budget"] = budget
            planner["kv_gather_seconds"] = sum(float(row.get("kv_gather_seconds", 0.0)) for row in tried)
            planner["query_eval_seconds"] = sum(float(row.get("query_eval_seconds", 0.0)) for row in tried)
            planner["online_seconds"] = sum(float(row.get("online_seconds", 0.0)) for row in tried)
            planner["total_seconds"] = float(selected.get("prefill_seconds", 0.0)) + float(planner["online_seconds"])
            planner["visible_tokens"] = sum(int(row.get("visible_tokens", 0)) for row in tried)
            planner["objective"] = objective_value(planner, spec)
            planner["deployable_oracle_mode"] = oracle["mode"]
            planner["deployable_objective_regret"] = planner["objective"] - objective_value(oracle, spec)
            planner["correct_regret"] = int(oracle["correct"]) - int(planner["correct"])
            planner["loss_regret"] = float(planner["gold_label_loss"]) - float(oracle["gold_label_loss"])
            planner["token_regret"] = float(planner.get("keep_fraction", 1.0)) - float(oracle.get("keep_fraction", 1.0))
            out.append(planner)
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sla = str(row.get("sla", ""))
        budget = str(row.get("budget", ""))
        for variant in (str(row["variant"]), "ALL"):
            grouped[(str(row["split"]), variant, str(row["mode"]), budget, sla)].append(row)

    summary: list[dict[str, Any]] = []
    for (split, variant, mode, budget, sla), subset in sorted(grouped.items()):
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
                "sla": sla,
                "tasks": len(subset),
                "accuracy": sum(int(row["correct"]) for row in subset) / n,
                "gold_label_ppl": math.exp(min(mean_loss, 80.0)),
                "mean_prefill_seconds": sum(float(row.get("prefill_seconds", 0.0)) for row in subset) / n,
                "mean_kv_gather_seconds": sum(float(row.get("kv_gather_seconds", 0.0)) for row in subset) / n,
                "mean_query_eval_seconds": sum(float(row.get("query_eval_seconds", 0.0)) for row in subset) / n,
                "mean_online_seconds": sum(float(row.get("online_seconds", 0.0)) for row in subset) / n,
                "mean_total_seconds": sum(float(row.get("total_seconds", 0.0)) for row in subset) / n,
                "mean_visible_tokens": sum(int(row.get("visible_tokens", 0)) for row in subset) / n,
                "mean_keep_fraction": sum(float(row.get("keep_fraction", 1.0)) for row in subset) / n,
                "evidence_hit_rate": sum(int(row.get("evidence_hit", 0)) for row in subset) / n,
                "mean_causal_label_recall": sum(float(row.get("causal_label_recall", 0.0)) for row in subset) / n,
                "mean_causal_positive_mass_recall": sum(float(row.get("causal_positive_mass_recall", 0.0)) for row in subset) / n,
                "mean_estimated_causal_recall": sum(float(row.get("estimated_causal_recall", 0.0)) for row in subset) / n,
                "mean_objective_regret": sum(float(row.get("deployable_objective_regret", 0.0)) for row in subset) / n,
                "mean_tried_steps": sum(float(row.get("tried_steps", 1.0)) for row in subset) / n,
            }
        )
    return summary


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (output_dir / "sla_specs.json").write_text(json.dumps(SLA_SPECS, indent=2), encoding="utf-8")

    variants = [part.strip() for part in config.variants.split(",") if part.strip()]
    unknown = [variant for variant in variants if variant not in ALL_BUILDERS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available={sorted(ALL_BUILDERS)}")
    budgets = parse_budgets(config.topk_budgets)

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
    task_records: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for variant in variants:
        for task_idx in range(config.tasks_per_variant):
            split = "train" if task_idx < train_cut else "test"
            task = ALL_BUILDERS[variant](rng, task_idx, config.distractor_pages)
            pages = split_pages(task["context"])
            bundle = build_fixed_prompt(tokenizer, task, pages)
            if task_idx == 0 or (task_idx + 1) % config.log_every == 0:
                print(f"[label] {variant} {task_idx + 1}/{config.tasks_per_variant} split={split}", flush=True)

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
                start, end = bundle.page_spans[page.page_id]
                features = feature_by_page[page.page_id]
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
            all_page_ids = [page.page_id for page in pages]
            for page_id in ablate_ids:
                selected_ids = [item for item in all_page_ids if item != page_id]
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
            task_records.append(
                {
                    "task": task,
                    "pages": pages,
                    "bundle": bundle,
                    "split": split,
                    "page_rows": task_page_rows,
                    "influence_by_page": {int(row["page_id"]): row for row in task_page_rows},
                }
            )
            del full_prefix_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    model_info = train_logistic_page_model(page_rows, config)
    (output_dir / "learned_page_model.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    strategy_rows: list[dict[str, Any]] = []
    for record_idx, record in enumerate(task_records):
        task = record["task"]
        pages = record["pages"]
        bundle = record["bundle"]
        split = record["split"]
        task_page_rows = record["page_rows"]
        influence_by_page = record["influence_by_page"]
        if record_idx == 0 or (record_idx + 1) % config.log_every == 0:
            print(f"[eval] {record_idx + 1}/{len(task_records)} {task['variant']} split={split}", flush=True)

        full_prefix_cache, _, prefill_seconds = prefill_prefix(model, bundle, input_device)
        pred, scores, query_seconds, loss, tokens, ppl = evaluate_with_cache(
            model,
            tokenizer,
            input_device,
            bundle,
            clone_past_key_values(full_prefix_cache),
            task["answer"],
        )
        full_row_base = full_result_row(task, pages, bundle, split, prefill_seconds, pred, scores, query_seconds, loss, tokens, ppl)

        for budget in budgets:
            full_row = dict(full_row_base)
            full_row["budget"] = budget
            strategy_rows.append(full_row)

            selectors: list[tuple[str, list[int], float, dict[str, Any]]] = []
            selectors.append(("recent_kv_gather_topk", select_recent(pages, budget), 0.0, {}))
            selectors.append(("lexical_kv_gather_topk", select_lexical(task, pages, budget), 0.0, {}))
            learned_ids, learned_recall = select_learned(task_page_rows, model_info, budget)
            selectors.append(("learned_causal_kv_gather_topk", learned_ids, learned_recall, {}))
            set_ids, set_recall, set_source = select_set_utility_v4(task, pages, task_page_rows, model_info, budget)
            selectors.append(("set_utility_kv_gather_v7", set_ids, set_recall, {"set_utility_source": set_source}))
            selectors.append(("oracle_causal_kv_gather_topk", select_oracle_causal(task_page_rows, budget), 1.0, {}))

            for mode, selected_ids, estimated_recall, extra in selectors:
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
                row.update(extra)
                strategy_rows.append(row)

        del full_prefix_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    regret_rows = add_oracle_regret_labels(strategy_rows)
    plans = train_variant_plans(regret_rows)
    planner_rows = make_planner_rows(strategy_rows, regret_rows, plans, config)
    all_rows = strategy_rows + planner_rows

    write_csv(output_dir / "causal_page_influence_labels.csv", page_rows)
    write_csv(output_dir / "strategy_results.csv", strategy_rows)
    write_csv(output_dir / "oracle_regret_labels.csv", regret_rows)
    write_csv(output_dir / "planner_results.csv", planner_rows)
    write_csv(output_dir / "all_results.csv", all_rows)
    write_csv(output_dir / "summary.csv", summarize(all_rows))
    (output_dir / "summary.json").write_text(json.dumps(summarize(all_rows), indent=2), encoding="utf-8")
    (output_dir / "learned_plans.json").write_text(json.dumps(plans, indent=2), encoding="utf-8")

    metadata = {
        "elapsed_seconds": time.perf_counter() - started,
        "tasks": len(task_records),
        "page_rows": len(page_rows),
        "strategy_rows": len(strategy_rows),
        "regret_rows": len(regret_rows),
        "planner_rows": len(planner_rows),
        "budgets": budgets,
        "positive_page_rate": sum(int(row["causal_label"]) for row in page_rows) / max(1, len(page_rows)),
        "weak_positive_pages": sum(int(row["weak_positive_label"]) for row in page_rows),
        "model_valid": model_info.get("valid", 0),
        "speed_path": "full_prefill_then_real_kv_gather_query_side",
        "range_sdpa_closure": "run_range_sdpa_speed_closure_v7_server.sh",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
