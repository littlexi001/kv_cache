from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_policy_router_runtime import load_router  # noqa: E402
from run_qwen8b_paper_benchmarks import (  # noqa: E402
    Config as BenchConfig,
    SUMMARY_TASKS,
    load_longbench_cases,
    load_ruler_cases,
    resolve_action,
    safety_override_action,
)
from run_qwen8b_router_distill_from_trials import write_csv  # noqa: E402


@dataclass(frozen=True)
class Config:
    benchmark_output_dirs: tuple[str, ...]
    output_dir: str
    router_path: str
    candidate_methods: tuple[str, ...]
    budget_targets: tuple[float, ...]
    summary_rouge_slack: float


@dataclass
class PolicyRow:
    policy: str
    benchmark: str
    task: str
    case_id: str
    action: str
    score: float
    full_score: float
    token_ratio_vs_full_raw: float
    seconds: float


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Offline-evaluate router policies from precomputed benchmark trials.")
    parser.add_argument("--benchmark_output_dirs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--router_path", default="")
    parser.add_argument(
        "--candidate_methods",
        default=(
            "full_raw,recent_only,static_hier,summary1_8,summary1_4,summary1_2,"
            "retrieval_raw_k1,retrieval_raw_k2,retrieval_raw_k3,retrieval_raw_k4,retrieval_raw_k8"
        ),
    )
    parser.add_argument("--budget_targets", default="0.2,0.3,0.35,0.5")
    parser.add_argument("--summary_rouge_slack", type=float, default=0.03)
    args = parser.parse_args()
    return Config(
        benchmark_output_dirs=parse_csv_tuple(args.benchmark_output_dirs),
        output_dir=args.output_dir,
        router_path=args.router_path,
        candidate_methods=parse_csv_tuple(args.candidate_methods),
        budget_targets=parse_float_tuple(args.budget_targets),
        summary_rouge_slack=args.summary_rouge_slack,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bench_config_from_summary(path: Path) -> BenchConfig:
    payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    raw = payload["config"]
    return BenchConfig(
        output_dir=raw["output_dir"],
        model_name_or_path=raw["model_name_or_path"],
        adapter_path=raw.get("adapter_path", ""),
        longbench_data_dir=raw["longbench_data_dir"],
        ruler_data_dir=raw["ruler_data_dir"],
        longbench_tasks=tuple(raw["longbench_tasks"]),
        ruler_tasks=tuple(raw["ruler_tasks"]),
        ruler_context_lengths=tuple(int(item) for item in raw["ruler_context_lengths"]),
        methods=tuple(raw["methods"]),
        max_examples_per_task=int(raw["max_examples_per_task"]),
        block_tokens=int(raw["block_tokens"]),
        recent_tokens=int(raw["recent_tokens"]),
        max_input_tokens=int(raw["max_input_tokens"]),
        summary10_words=int(raw["summary10_words"]),
        summary100_words=int(raw["summary100_words"]),
        summary1000_words=int(raw["summary1000_words"]),
        max_new_tokens_exact=int(raw["max_new_tokens_exact"]),
        max_new_tokens_summary=int(raw["max_new_tokens_summary"]),
        dtype=raw["dtype"],
        attn_implementation=raw["attn_implementation"],
        device_map=raw["device_map"],
        cuda_visible_devices=raw.get("cuda_visible_devices", ""),
        router_path=raw.get("router_path", ""),
        seed=int(raw["seed"]),
    )


def load_trials(config: Config) -> tuple[BenchConfig, dict[tuple[str, str, str, str], dict[str, str]]]:
    dirs = [Path(item) for item in config.benchmark_output_dirs]
    bench_config = bench_config_from_summary(dirs[0])
    lookup: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for directory in dirs:
        for row in read_csv(directory / "trials.csv"):
            lookup[(row["benchmark"], row["task"], row["case_id"], row["method"])] = row
    return bench_config, lookup


def method_score(row: dict[str, str]) -> float:
    return float(row["score"])


def method_ratio(row: dict[str, str]) -> float:
    return float(row["token_ratio_vs_full_raw"])


def oracle_match_full_action(rows: list[dict[str, str]], task: str, config: Config) -> str:
    full = next(row for row in rows if row["method"] == "full_raw")
    full_score = method_score(full)
    if task in SUMMARY_TASKS:
        threshold = max(0.0, full_score - config.summary_rouge_slack)
    else:
        threshold = full_score
    successful = [row for row in rows if method_score(row) + 1e-12 >= threshold]
    if not successful:
        successful = rows
    selected = min(successful, key=lambda row: (method_ratio(row), float(row["seconds"]), row["method"]))
    return selected["method"]


def oracle_best_under_budget_action(rows: list[dict[str, str]], budget: float) -> str:
    candidates = [row for row in rows if method_ratio(row) <= budget + 1e-12]
    if not candidates:
        candidates = rows
    best_score = max(method_score(row) for row in candidates)
    selected = min(
        [row for row in candidates if abs(method_score(row) - best_score) < 1e-12],
        key=lambda row: (method_ratio(row), float(row["seconds"]), row["method"]),
    )
    return selected["method"]


def length_aware_rule_action(benchmark: str, task: str, full_tokens: int) -> str:
    if task in SUMMARY_TASKS:
        return "summary1_8"
    if benchmark == "longbench":
        if task in {"hotpotqa", "musique"} and full_tokens > 12000:
            return "retrieval_raw_k1"
        if task in {"passage_retrieval_en", "passage_count", "2wikimqa"}:
            return "retrieval_raw_k1"
        return "retrieval_raw_k1"
    if benchmark.startswith("ruler"):
        if task in {"cwe", "fwe"}:
            return "retrieval_raw_k1"
        if full_tokens <= 5000:
            return "retrieval_raw_k2"
        if task in {"niah_multiquery", "niah_multivalue", "vt"}:
            return "retrieval_raw_k2"
        return "retrieval_raw_k2"
    return "retrieval_raw_k1"


def add_row(
    rows: list[PolicyRow],
    policy: str,
    action: str,
    case_key: tuple[str, str, str],
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
) -> None:
    trial = lookup.get((*case_key, action))
    full = lookup.get((*case_key, "full_raw"))
    if trial is None or full is None:
        return
    rows.append(
        PolicyRow(
            policy=policy,
            benchmark=case_key[0],
            task=case_key[1],
            case_id=case_key[2],
            action=action,
            score=float(trial["score"]),
            full_score=float(full["score"]),
            token_ratio_vs_full_raw=float(trial["token_ratio_vs_full_raw"]),
            seconds=float(trial["seconds"]),
        )
    )


def summarize(rows: list[PolicyRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[PolicyRow]] = defaultdict(list)
    for row in rows:
        groups[(row.policy, "__overall__")].append(row)
        groups[(row.policy, "longbench" if row.benchmark == "longbench" else row.benchmark)].append(row)
        groups[(row.policy, "generation" if row.task in SUMMARY_TASKS else "exact")].append(row)
    out: list[dict[str, Any]] = []
    for (policy, group), items in sorted(groups.items()):
        score = sum(row.score for row in items) / len(items)
        full = sum(row.full_score for row in items) / len(items)
        ratio = sum(row.token_ratio_vs_full_raw for row in items) / len(items)
        seconds = sum(row.seconds for row in items) / len(items)
        payload: dict[str, Any] = {
            "policy": policy,
            "group": group,
            "samples": len(items),
            "avg_score": score,
            "avg_full_score": full,
            "relative_to_full": score / full if full else "",
            "avg_token_ratio_vs_full_raw": ratio,
            "avg_seconds": seconds,
        }
        counts = Counter(row.action for row in items)
        for action, count in sorted(counts.items()):
            payload[f"select_{action}"] = count
            payload[f"select_{action}_rate"] = count / len(items)
        out.append(payload)
    return out


def main() -> None:
    from transformers import AutoTokenizer

    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bench_config, lookup = load_trials(config)
    tokenizer = AutoTokenizer.from_pretrained(bench_config.model_name_or_path, trust_remote_code=True)
    cases = load_longbench_cases(bench_config) + load_ruler_cases(bench_config)
    router = load_router(config.router_path) if config.router_path else None

    policy_rows: list[PolicyRow] = []
    for case in cases:
        case_key = (case.benchmark, case.task, case.case_id)
        candidate_rows = [
            lookup[(*case_key, method)]
            for method in config.candidate_methods
            if (*case_key, method) in lookup
        ]
        if not candidate_rows or (*case_key, "full_raw") not in lookup:
            continue
        full_tokens = int(lookup[(*case_key, "full_raw")]["prompt_tokens"])
        add_row(policy_rows, "oracle_match_full", oracle_match_full_action(candidate_rows, case.task, config), case_key, lookup)
        for budget in config.budget_targets:
            add_row(
                policy_rows,
                f"oracle_best_under_{int(round(budget * 100))}pct",
                oracle_best_under_budget_action(candidate_rows, budget),
                case_key,
                lookup,
            )
        add_row(policy_rows, "length_aware_rule", length_aware_rule_action(case.benchmark, case.task, full_tokens), case_key, lookup)
        if router is not None:
            learned = resolve_action("router", tokenizer, case, bench_config, router)
            conservative = resolve_action("router_conservative", tokenizer, case, bench_config, router)
            add_row(policy_rows, "learned_router", learned, case_key, lookup)
            add_row(policy_rows, "learned_router_conservative", conservative, case_key, lookup)
            add_row(policy_rows, "learned_router_safety_only", safety_override_action(case, learned), case_key, lookup)

    summary = summarize(policy_rows)
    write_csv(output_dir / "policy_rows.csv", [asdict(row) for row in policy_rows])
    write_csv(output_dir / "policy_summary.csv", summary)
    (output_dir / "summary.json").write_text(
        json.dumps({"config": asdict(config), "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("policy,group,samples,avg_score,avg_full_score,relative_to_full,avg_token_ratio")
    for row in summary:
        print(
            f"{row['policy']},{row['group']},{row['samples']},"
            f"{row['avg_score']:.4f},{row['avg_full_score']:.4f},"
            f"{row['relative_to_full']:.4f},{row['avg_token_ratio_vs_full_raw']:.4f}"
        )


if __name__ == "__main__":
    main()
