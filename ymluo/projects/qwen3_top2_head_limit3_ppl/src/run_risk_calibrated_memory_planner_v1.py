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
from run_task_aware_kv_mixture_v0 import (  # noqa: E402
    ALL_BUILDERS,
    build_strategy_prompt,
    empty_state,
    evaluate_prompt,
    rule_router,
)


BASE_STRATEGIES = [
    "recent_local",
    "semantic_route",
    "typed_role",
    "chain_typed",
    "hierarchical_summary",
]
COMPOSITE_STRATEGIES: dict[str, list[str]] = {
    "recent_typed": ["recent_local", "typed_role"],
    "semantic_chain": ["semantic_route", "chain_typed"],
    "hier_chain": ["hierarchical_summary", "chain_typed"],
}
STATIC_STRATEGIES = BASE_STRATEGIES + list(COMPOSITE_STRATEGIES) + ["full_context"]


@dataclass(frozen=True)
class Config:
    model_name_or_path: str
    output_dir: str
    variants: str
    tasks_per_variant: int
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
    parser = argparse.ArgumentParser(description="Risk-calibrated progressive memory planner V1.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--variants",
        default="casual_recent,temporal_fact,multihop_bridge,summary_theme,compare_score",
    )
    parser.add_argument("--tasks_per_variant", type=int, default=8)
    parser.add_argument("--distractor_pages", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026070206)
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


def strip_query(prompt: str, task: dict[str, Any]) -> str:
    query = task["query"]
    if prompt.endswith(query):
        return prompt[: -len(query)].strip()
    return prompt.strip()


def compact_context(text: str) -> str:
    return " ".join(text.split())


def full_context_prompt(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    prompt = (
        "\nFull memory fallback context:\n"
        + task["context"]
        + "\nUse the full context to answer with the option letter.\n"
        + task["query"]
    )
    return prompt, {
        "selected_pages": -1,
        "page_count": task["context"].count("\n\n") + 1,
        "evidence_hit": int(str(task["answer_value"]).lower() in task["context"].lower()),
        "strategy_task_type": "full_context",
        "routed_answer_value": "",
    }


def composite_prompt(
    task: dict[str, Any],
    strategy: str,
    max_route_pages: int,
    recent_pages: int,
) -> tuple[str, dict[str, Any]]:
    pieces = []
    selected_pages = 0
    page_count = 0
    evidence_hit = 0
    routed_values = []
    for expert in COMPOSITE_STRATEGIES[strategy]:
        prompt, meta = build_strategy_prompt(task, expert, max_route_pages, recent_pages)
        pieces.append(f"\n--- memory expert: {expert} ---\n{strip_query(prompt, task)}")
        selected_pages += max(0, int(meta.get("selected_pages", 0)))
        page_count = max(page_count, int(meta.get("page_count", 0)))
        evidence_hit = max(evidence_hit, int(meta.get("evidence_hit", 0)))
        value = meta.get("routed_answer_value")
        if value:
            routed_values.append(str(value))
    prompt = (
        "\nComposite memory plan:\n"
        + "\n".join(pieces)
        + "\nUse the combined memory experts to answer with the option letter.\n"
        + task["query"]
    )
    return prompt, {
        "selected_pages": selected_pages,
        "page_count": page_count,
        "evidence_hit": evidence_hit,
        "strategy_task_type": strategy,
        "routed_answer_value": "|".join(routed_values),
    }


def build_any_strategy_prompt(
    task: dict[str, Any],
    strategy: str,
    max_route_pages: int,
    recent_pages: int,
) -> tuple[str, dict[str, Any]]:
    if strategy in BASE_STRATEGIES:
        return build_strategy_prompt(task, strategy, max_route_pages, recent_pages)
    if strategy in COMPOSITE_STRATEGIES:
        return composite_prompt(task, strategy, max_route_pages, recent_pages)
    if strategy == "full_context":
        return full_context_prompt(task)
    raise ValueError(f"unknown strategy: {strategy}")


def memory_need_vector(task: dict[str, Any]) -> dict[str, float]:
    variant = task["variant"]
    if variant == "casual_recent":
        return {
            "locality_need": 1.0,
            "semantic_need": 0.1,
            "hop_depth": 0.0,
            "temporal_conflict_need": 0.0,
            "aggregation_scope": 0.0,
            "risk_level": 0.20,
        }
    if variant == "temporal_fact":
        return {
            "locality_need": 0.1,
            "semantic_need": 0.5,
            "hop_depth": 0.0,
            "temporal_conflict_need": 1.0,
            "aggregation_scope": 0.0,
            "risk_level": 0.65,
        }
    if variant == "multihop_bridge":
        return {
            "locality_need": 0.1,
            "semantic_need": 0.8,
            "hop_depth": 2.0,
            "temporal_conflict_need": 0.4,
            "aggregation_scope": 0.1,
            "risk_level": 0.75,
        }
    if variant == "summary_theme":
        return {
            "locality_need": 0.0,
            "semantic_need": 0.7,
            "hop_depth": 0.0,
            "temporal_conflict_need": 0.2,
            "aggregation_scope": 1.0,
            "risk_level": 0.70,
        }
    if variant == "compare_score":
        return {
            "locality_need": 0.0,
            "semantic_need": 0.7,
            "hop_depth": 0.5,
            "temporal_conflict_need": 0.5,
            "aggregation_scope": 0.9,
            "risk_level": 0.85,
        }
    return {
        "locality_need": 0.2,
        "semantic_need": 0.5,
        "hop_depth": 0.0,
        "temporal_conflict_need": 0.0,
        "aggregation_scope": 0.0,
        "risk_level": 0.50,
    }


def make_memory_plan(task: dict[str, Any]) -> list[str]:
    variant = task["variant"]
    if variant == "casual_recent":
        return ["recent_local", "full_context"]
    if variant == "temporal_fact":
        return ["hierarchical_summary", "typed_role", "semantic_route", "full_context"]
    if variant == "multihop_bridge":
        return ["chain_typed", "semantic_chain", "full_context"]
    if variant == "summary_theme":
        return ["hierarchical_summary", "semantic_route", "full_context"]
    if variant == "compare_score":
        return ["chain_typed", "hierarchical_summary", "full_context"]
    return ["recent_local", "semantic_route", "full_context"]


def score_margin(scores: dict[str, float]) -> float:
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) < 2:
        return 0.0
    return float(ordered[0] - ordered[1])


def parse_options(query: str) -> dict[str, str]:
    options = {}
    for line in query.splitlines():
        line = line.strip()
        if len(line) >= 4 and line[1:3] == ". " and line[0] in LABELS:
            options[line[0]] = line[3:].strip()
    return options


def symbolic_label_from_value(task: dict[str, Any], routed_answer_value: Any) -> str:
    if not routed_answer_value:
        return ""
    options = parse_options(task["query"])
    values = [part.strip().lower() for part in str(routed_answer_value).split("|") if part.strip()]
    for label, option_text in options.items():
        normalized_option = option_text.strip().lower()
        for value in values:
            if value == normalized_option:
                return label
    return ""


def need_mismatch_penalty(strategy: str, need: dict[str, float], config: Config) -> float:
    long_need = max(
        need["semantic_need"],
        min(1.0, need["hop_depth"] / 2.0),
        need["temporal_conflict_need"],
        need["aggregation_scope"],
    )
    if strategy == "recent_local" and long_need > 0.5:
        return config.mismatch_margin_penalty
    if strategy == "semantic_route" and (need["hop_depth"] > 1.0 or need["aggregation_scope"] > 0.8):
        return config.mismatch_margin_penalty * 0.7
    if strategy == "hierarchical_summary" and need["hop_depth"] > 1.0:
        return config.mismatch_margin_penalty * 0.7
    return 0.0


def accept_stage(row: dict[str, Any], need: dict[str, float], is_last: bool, config: Config) -> tuple[bool, str]:
    if row.get("pred_source") == "symbolic_value":
        return True, "resolved_value_to_option"
    if is_last or row["strategy"] == "full_context":
        return True, "last_or_full_fallback"
    long_need = max(
        need["semantic_need"],
        min(1.0, need["hop_depth"] / 2.0),
        need["temporal_conflict_need"],
        need["aggregation_scope"],
    )
    if row["strategy"] == "recent_local" and long_need > 0.5 and not row.get("routed_answer_value"):
        return False, "recent_local_without_resolved_value_for_long_need"
    threshold = config.accept_margin + config.risk_margin_scale * need["risk_level"]
    threshold += need_mismatch_penalty(str(row["strategy"]), need, config)
    if row.get("routed_answer_value"):
        threshold -= config.resolved_margin_bonus
    threshold = max(config.min_accept_margin, threshold)
    margin = float(row.get("margin", 0.0))
    if margin >= threshold:
        return True, f"margin {margin:.3f} >= {threshold:.3f}"
    return False, f"margin {margin:.3f} < {threshold:.3f}"


def evaluate_strategy(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    task: dict[str, Any],
    strategy: str,
    config: Config,
) -> dict[str, Any]:
    prompt, meta = build_any_strategy_prompt(task, strategy, config.max_route_pages, config.recent_pages)
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
    routed_answer_value = meta.get("routed_answer_value", "")
    symbolic_pred = symbolic_label_from_value(task, routed_answer_value)
    pred_source = "symbolic_value" if symbolic_pred else "model_label"
    final_pred = symbolic_pred or pred
    return {
        "variant": task["variant"],
        "task_id": task["task_id"],
        "mode": f"static_{strategy}",
        "strategy": strategy,
        "router_strategy": rule_router(task),
        "plan": "",
        "need_vector": "",
        "answer": task["answer"],
        "answer_value": task["answer_value"],
        "pred": final_pred,
        "raw_model_pred": pred,
        "pred_source": pred_source,
        "correct": int(final_pred == task["answer"]),
        "eval_seconds": elapsed,
        "visible_tokens": int(prompt_ids.shape[-1]),
        "gold_label_loss": loss,
        "gold_label_tokens": tokens,
        "gold_label_ppl": ppl,
        "margin": score_margin(scores),
        "task_type": meta.get("task_type", meta.get("strategy_task_type", "")),
        "selected_pages": int(meta.get("selected_pages", 0)),
        "page_count": int(meta.get("page_count", 0)),
        "evidence_hit": int(meta.get("evidence_hit", 0)),
        "routed_answer_value": routed_answer_value,
        "tried_steps": 1,
        "selected_step": 1,
        "accept_reason": "",
        "escalated": 0,
        "correct_regret": 0.0,
        "ppl_regret": 0.0,
        "token_cost_ratio_to_oracle": 1.0,
        **{f"score_{label}": scores[label] for label in LABELS},
    }


def choose_quality_oracle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda row: (
            int(row["correct"]),
            -float(row["gold_label_ppl"]),
            -float(row["eval_seconds"]),
            -int(row["visible_tokens"]),
        ),
        reverse=True,
    )[0]


def choose_min_cost_correct_oracle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [row for row in rows if int(row["correct"])]
    if correct:
        return sorted(correct, key=lambda row: (int(row["visible_tokens"]), float(row["gold_label_ppl"])))[0]
    return choose_quality_oracle(rows)


def make_planner_row(
    task: dict[str, Any],
    static_by_strategy: dict[str, dict[str, Any]],
    config: Config,
    oracle_row: dict[str, Any],
) -> dict[str, Any]:
    need = memory_need_vector(task)
    plan = make_memory_plan(task)
    tried = []
    selected = None
    reason = ""
    for idx, strategy in enumerate(plan, start=1):
        row = static_by_strategy[strategy]
        tried.append(row)
        accepted, reason = accept_stage(row, need, idx == len(plan), config)
        if accepted:
            selected = row
            break
    if selected is None:
        selected = static_by_strategy[plan[-1]]
    cumulative_seconds = sum(float(row["eval_seconds"]) for row in tried)
    cumulative_tokens = sum(int(row["visible_tokens"]) for row in tried)
    result = dict(selected)
    result["mode"] = "risk_calibrated_planner_v1"
    result["plan"] = ">".join(plan)
    result["need_vector"] = json.dumps(need, sort_keys=True)
    result["eval_seconds"] = cumulative_seconds
    result["visible_tokens"] = cumulative_tokens
    result["tried_steps"] = len(tried)
    result["selected_step"] = plan.index(result["strategy"]) + 1
    result["accept_reason"] = reason
    result["escalated"] = int(len(tried) > 1)
    result["correct_regret"] = int(oracle_row["correct"]) - int(result["correct"])
    result["ppl_regret"] = float(result["gold_label_ppl"]) - float(oracle_row["gold_label_ppl"])
    result["token_cost_ratio_to_oracle"] = cumulative_tokens / max(1, int(oracle_row["visible_tokens"]))
    return result


def make_rule_router_row(static_by_strategy: dict[str, dict[str, Any]]) -> dict[str, Any]:
    strategy = next(iter(static_by_strategy.values()))["router_strategy"]
    row = dict(static_by_strategy[strategy])
    row["mode"] = "task_aware_rule_router_v0"
    return row


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
                "mean_margin": sum(float(row["margin"]) for row in subset) / n,
                "evidence_hit_rate": sum(int(row["evidence_hit"]) for row in subset) / n,
                "mean_selected_pages": sum(int(row["selected_pages"]) for row in subset) / n,
                "mean_tried_steps": sum(int(row["tried_steps"]) for row in subset) / n,
                "escalation_rate": sum(int(row["escalated"]) for row in subset) / n,
                "mean_correct_regret": sum(float(row["correct_regret"]) for row in subset) / n,
                "mean_ppl_regret": sum(float(row["ppl_regret"]) for row in subset) / n,
                "mean_token_cost_ratio_to_oracle": sum(float(row["token_cost_ratio_to_oracle"]) for row in subset) / n,
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

    rows: list[dict[str, Any]] = []
    rng = random.Random(config.seed)
    started = time.perf_counter()

    for variant in variants:
        for task_idx in range(config.tasks_per_variant):
            task = ALL_BUILDERS[variant](rng, task_idx, config.distractor_pages)
            if (task_idx + 1) % config.log_every == 0 or task_idx == 0:
                print(f"{variant} {task_idx + 1}/{config.tasks_per_variant}", flush=True)
            static_by_strategy = {}
            for strategy in STATIC_STRATEGIES:
                row = evaluate_strategy(model, tokenizer, input_device, task, strategy, config)
                static_by_strategy[strategy] = row
                rows.append(row)

            quality_oracle = dict(choose_quality_oracle(list(static_by_strategy.values())))
            quality_oracle["mode"] = "oracle_best_expert"
            rows.append(quality_oracle)

            min_cost_oracle = dict(choose_min_cost_correct_oracle(list(static_by_strategy.values())))
            min_cost_oracle["mode"] = "oracle_min_cost_correct"
            rows.append(min_cost_oracle)

            rows.append(make_rule_router_row(static_by_strategy))
            rows.append(make_planner_row(task, static_by_strategy, config, quality_oracle))

    row_fields = [
        "variant",
        "task_id",
        "mode",
        "strategy",
        "router_strategy",
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
        "task_type",
        "selected_pages",
        "page_count",
        "evidence_hit",
        "routed_answer_value",
        "tried_steps",
        "selected_step",
        "accept_reason",
        "escalated",
        "correct_regret",
        "ppl_regret",
        "token_cost_ratio_to_oracle",
    ] + [f"score_{label}" for label in LABELS]
    with (output_dir / "planner_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    with (output_dir / "planner_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    result = {"seconds": time.perf_counter() - started, "summary": summary}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
