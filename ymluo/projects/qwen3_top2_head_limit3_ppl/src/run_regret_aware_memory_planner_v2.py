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
    pick_input_device,
    resolve_dtype,
)
from run_qabs_downstream_kv_retrieval import LABELS  # noqa: E402
from run_risk_calibrated_memory_planner_v1 import (  # noqa: E402
    STATIC_STRATEGIES,
    accept_stage,
    choose_quality_oracle,
    evaluate_strategy,
    make_planner_row,
    make_rule_router_row,
    memory_need_vector,
)
from run_task_aware_kv_mixture_v0 import ALL_BUILDERS  # noqa: E402


SLA_SPECS: dict[str, dict[str, float]] = {
    "quality": {"incorrect": 1000.0, "loss": 1.0, "token": 0.0, "seconds": 0.0},
    "balanced": {"incorrect": 1000.0, "loss": 1.0, "token": 0.10, "seconds": 0.02},
    "low_cost": {"incorrect": 1000.0, "loss": 0.35, "token": 0.45, "seconds": 0.05},
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
    max_route_pages: int
    recent_pages: int
    accept_margin: float
    risk_margin_scale: float
    mismatch_margin_penalty: float
    resolved_margin_bonus: float
    min_accept_margin: float
    log_every: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Regret-aware SLA-conditioned memory planner V2.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--variants",
        default="casual_recent,temporal_fact,multihop_bridge,summary_theme,compare_score",
    )
    parser.add_argument("--tasks_per_variant", type=int, default=10)
    parser.add_argument("--train_fraction", type=float, default=0.5)
    parser.add_argument("--distractor_pages", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026070207)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_route_pages", type=int, default=6)
    parser.add_argument("--recent_pages", type=int, default=2)
    parser.add_argument("--accept_margin", type=float, default=0.45)
    parser.add_argument("--risk_margin_scale", type=float, default=0.9)
    parser.add_argument("--mismatch_margin_penalty", type=float, default=0.45)
    parser.add_argument("--resolved_margin_bonus", type=float, default=0.25)
    parser.add_argument("--min_accept_margin", type=float, default=0.15)
    parser.add_argument("--log_every", type=int, default=5)
    return Config(**vars(parser.parse_args()))


def task_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["variant"]), int(row["task_id"])


def label_loss(row: dict[str, Any]) -> float:
    return math.log(max(float(row["gold_label_ppl"]), 1e-8))


def objective_value(row: dict[str, Any], spec: dict[str, float], full_tokens: int) -> float:
    incorrect = 1 - int(row["correct"])
    token_ratio = int(row["visible_tokens"]) / max(1, full_tokens)
    seconds = float(row["eval_seconds"])
    return (
        spec["incorrect"] * incorrect
        + spec["loss"] * label_loss(row)
        + spec["token"] * math.log1p(token_ratio)
        + spec["seconds"] * seconds
    )


def compute_pareto_flags(rows: list[dict[str, Any]]) -> dict[str, int]:
    flags = {}
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = (
                int(other["correct"]) >= int(row["correct"])
                and float(other["gold_label_ppl"]) <= float(row["gold_label_ppl"]) + 1e-9
                and int(other["visible_tokens"]) <= int(row["visible_tokens"])
                and float(other["eval_seconds"]) <= float(row["eval_seconds"]) + 1e-9
            )
            strictly_better = (
                int(other["correct"]) > int(row["correct"])
                or float(other["gold_label_ppl"]) < float(row["gold_label_ppl"]) - 1e-9
                or int(other["visible_tokens"]) < int(row["visible_tokens"])
                or float(other["eval_seconds"]) < float(row["eval_seconds"]) - 1e-9
            )
            if no_worse and strictly_better:
                dominated = True
                break
        flags[str(row["strategy"])] = int(not dominated)
    return flags


def choose_sla_oracle(
    rows: list[dict[str, Any]],
    spec: dict[str, float],
    full_tokens: int,
) -> dict[str, Any]:
    return min(rows, key=lambda row: objective_value(row, spec, full_tokens))


def copy_mode(
    source: dict[str, Any],
    mode: str,
    sla: str,
    oracle: dict[str, Any] | None,
    full_tokens: int,
    spec: dict[str, float] | None,
) -> dict[str, Any]:
    row = dict(source)
    row["mode"] = mode
    row["sla"] = sla
    if spec is not None:
        row["objective"] = objective_value(row, spec, full_tokens)
    if oracle is not None:
        row["oracle_strategy"] = oracle["strategy"]
        row["objective_regret"] = float(row.get("objective", 0.0)) - objective_value(oracle, spec or SLA_SPECS["quality"], full_tokens)
        row["correct_regret"] = int(oracle["correct"]) - int(row["correct"])
        row["ppl_regret"] = float(row["gold_label_ppl"]) - float(oracle["gold_label_ppl"])
        row["token_regret"] = int(row["visible_tokens"]) - int(oracle["visible_tokens"])
    return row


def train_sla_plans(
    task_rows: dict[tuple[str, int], dict[str, dict[str, Any]]],
    splits: dict[tuple[str, int], str],
) -> dict[str, dict[str, list[str]]]:
    plans: dict[str, dict[str, list[str]]] = defaultdict(dict)
    variants = sorted({key[0] for key in task_rows})
    for sla, spec in SLA_SPECS.items():
        for variant in variants:
            by_strategy: dict[str, list[float]] = defaultdict(list)
            for key, rows_by_strategy in task_rows.items():
                if key[0] != variant or splits[key] != "train":
                    continue
                full_tokens = int(rows_by_strategy["full_context"]["visible_tokens"])
                for strategy, row in rows_by_strategy.items():
                    by_strategy[strategy].append(objective_value(row, spec, full_tokens))
            if not by_strategy:
                plans[sla][variant] = list(STATIC_STRATEGIES)
                continue
            scored = [
                (sum(values) / max(1, len(values)), strategy)
                for strategy, values in by_strategy.items()
            ]
            scored.sort(key=lambda item: (item[0], item[1]))
            ordered = [strategy for _, strategy in scored]
            if "full_context" in ordered:
                ordered = [strategy for strategy in ordered if strategy != "full_context"] + ["full_context"]
            plans[sla][variant] = ordered
    return plans


def execute_learned_plan(
    rows_by_strategy: dict[str, dict[str, Any]],
    plan: list[str],
    task: dict[str, Any],
    config: Config,
    mode: str,
    sla: str,
    oracle: dict[str, Any],
    spec: dict[str, float],
) -> dict[str, Any]:
    tried = []
    selected = None
    reason = ""
    need = memory_need_vector(task)
    for idx, strategy in enumerate(plan, start=1):
        row = rows_by_strategy[strategy]
        tried.append(row)
        accepted, reason = accept_stage(row, need, idx == len(plan), config)
        if accepted:
            selected = row
            break
    if selected is None:
        selected = rows_by_strategy[plan[-1]]
    full_tokens = int(rows_by_strategy["full_context"]["visible_tokens"])
    result = copy_mode(selected, mode, sla, oracle, full_tokens, spec)
    result["plan"] = ">".join(plan)
    result["need_vector"] = json.dumps(need, sort_keys=True)
    result["eval_seconds"] = sum(float(row["eval_seconds"]) for row in tried)
    result["visible_tokens"] = sum(int(row["visible_tokens"]) for row in tried)
    result["tried_steps"] = len(tried)
    result["selected_step"] = plan.index(selected["strategy"]) + 1
    result["accept_reason"] = reason
    result["escalated"] = int(len(tried) > 1)
    result["objective"] = objective_value(result, spec, full_tokens)
    result["objective_regret"] = result["objective"] - objective_value(oracle, spec, full_tokens)
    result["correct_regret"] = int(oracle["correct"]) - int(result["correct"])
    result["ppl_regret"] = float(result["gold_label_ppl"]) - float(oracle["gold_label_ppl"])
    result["token_regret"] = int(result["visible_tokens"]) - int(oracle["visible_tokens"])
    return result


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for variant in (row["variant"], "ALL"):
            grouped[(str(row.get("split", "all")), variant, str(row["mode"]), str(row.get("sla", "")))].append(row)
    summary = []
    for (split, variant, mode, sla), subset in sorted(grouped.items()):
        n = max(1, len(subset))
        total_tokens = sum(int(row["gold_label_tokens"]) for row in subset)
        total_loss = sum(float(row["gold_label_loss"]) * int(row["gold_label_tokens"]) for row in subset)
        mean_loss = total_loss / max(1, total_tokens)
        summary.append(
            {
                "split": split,
                "variant": variant,
                "mode": mode,
                "sla": sla,
                "tasks": len(subset),
                "accuracy": sum(int(row["correct"]) for row in subset) / n,
                "gold_label_ppl": math.exp(min(mean_loss, 80.0)),
                "mean_eval_seconds": sum(float(row["eval_seconds"]) for row in subset) / n,
                "mean_visible_tokens": sum(int(row["visible_tokens"]) for row in subset) / n,
                "mean_objective": sum(float(row.get("objective", 0.0)) for row in subset) / n,
                "mean_objective_regret": sum(float(row.get("objective_regret", 0.0)) for row in subset) / n,
                "mean_correct_regret": sum(float(row.get("correct_regret", 0.0)) for row in subset) / n,
                "mean_ppl_regret": sum(float(row.get("ppl_regret", 0.0)) for row in subset) / n,
                "mean_token_regret": sum(float(row.get("token_regret", 0.0)) for row in subset) / n,
                "mean_is_pareto": sum(int(row.get("is_pareto", 0)) for row in subset) / n,
                "mean_tried_steps": sum(int(row.get("tried_steps", 1)) for row in subset) / n,
                "escalation_rate": sum(int(row.get("escalated", 0)) for row in subset) / n,
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
    model.config.use_cache = True
    input_device = pick_input_device(model, device)

    rng = random.Random(config.seed)
    tasks: dict[tuple[str, int], dict[str, Any]] = {}
    splits: dict[tuple[str, int], str] = {}
    task_rows: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    train_cut = max(1, min(config.tasks_per_variant - 1, int(round(config.tasks_per_variant * config.train_fraction))))
    started = time.perf_counter()

    for variant in variants:
        for task_idx in range(config.tasks_per_variant):
            task = ALL_BUILDERS[variant](rng, task_idx, config.distractor_pages)
            key = (variant, task_idx)
            split = "train" if task_idx < train_cut else "test"
            tasks[key] = task
            splits[key] = split
            if (task_idx + 1) % config.log_every == 0 or task_idx == 0:
                print(f"{variant} {task_idx + 1}/{config.tasks_per_variant} split={split}", flush=True)
            rows_by_strategy: dict[str, dict[str, Any]] = {}
            for strategy in STATIC_STRATEGIES:
                row = evaluate_strategy(model, tokenizer, input_device, task, strategy, config)
                row["split"] = split
                row["sla"] = ""
                row["objective"] = 0.0
                row["objective_regret"] = 0.0
                row["token_regret"] = 0
                rows_by_strategy[strategy] = row
            pareto_flags = compute_pareto_flags(list(rows_by_strategy.values()))
            for strategy, row in rows_by_strategy.items():
                row["is_pareto"] = pareto_flags[strategy]
                rows.append(row)
            task_rows[key] = rows_by_strategy

    learned_plans = train_sla_plans(task_rows, splits)
    with (output_dir / "learned_plans.json").open("w", encoding="utf-8") as handle:
        json.dump(learned_plans, handle, indent=2, sort_keys=True)

    for key, rows_by_strategy in task_rows.items():
        task = tasks[key]
        split = splits[key]
        full_tokens = int(rows_by_strategy["full_context"]["visible_tokens"])

        quality_oracle = dict(choose_quality_oracle(list(rows_by_strategy.values())))
        quality_oracle["split"] = split
        quality_oracle["is_pareto"] = rows_by_strategy[quality_oracle["strategy"]]["is_pareto"]
        rows.append(copy_mode(quality_oracle, "oracle_best_expert", "", None, full_tokens, None))

        rule_row = make_rule_router_row(rows_by_strategy)
        rule_row["split"] = split
        rule_row["is_pareto"] = rows_by_strategy[rule_row["strategy"]]["is_pareto"]
        rows.append(copy_mode(rule_row, "task_aware_rule_router_v0", "", None, full_tokens, None))

        v1_row = make_planner_row(task, rows_by_strategy, config, quality_oracle)
        v1_row["split"] = split
        v1_row["is_pareto"] = rows_by_strategy[v1_row["strategy"]]["is_pareto"]
        rows.append(copy_mode(v1_row, "risk_calibrated_planner_v1", "", None, full_tokens, None))

        for sla, spec in SLA_SPECS.items():
            oracle = dict(choose_sla_oracle(list(rows_by_strategy.values()), spec, full_tokens))
            oracle["split"] = split
            oracle["is_pareto"] = rows_by_strategy[oracle["strategy"]]["is_pareto"]
            rows.append(copy_mode(oracle, f"oracle_sla_{sla}", sla, oracle, full_tokens, spec))

            plan = learned_plans[sla][key[0]]
            learned = execute_learned_plan(
                rows_by_strategy,
                plan,
                task,
                config,
                f"learned_planner_v2_{sla}",
                sla,
                oracle,
                spec,
            )
            learned["split"] = split
            learned["is_pareto"] = rows_by_strategy[learned["strategy"]]["is_pareto"]
            rows.append(learned)

    row_fields = [
        "split",
        "variant",
        "task_id",
        "mode",
        "sla",
        "strategy",
        "router_strategy",
        "oracle_strategy",
        "plan",
        "need_vector",
        "answer",
        "answer_value",
        "pred",
        "raw_model_pred",
        "pred_source",
        "correct",
        "eval_seconds",
        "visible_tokens",
        "gold_label_loss",
        "gold_label_tokens",
        "gold_label_ppl",
        "margin",
        "objective",
        "objective_regret",
        "correct_regret",
        "ppl_regret",
        "token_regret",
        "task_type",
        "selected_pages",
        "page_count",
        "evidence_hit",
        "routed_answer_value",
        "is_pareto",
        "tried_steps",
        "selected_step",
        "accept_reason",
        "escalated",
        "token_cost_ratio_to_oracle",
    ] + [f"score_{label}" for label in LABELS]
    with (output_dir / "regret_planner_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    with (output_dir / "regret_planner_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    frontier_rows = [row for row in rows if row["mode"].startswith("static_") and int(row.get("is_pareto", 0))]
    with (output_dir / "pareto_frontier_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(frontier_rows)

    result = {
        "seconds": time.perf_counter() - started,
        "train_cut": train_cut,
        "learned_plans": learned_plans,
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
