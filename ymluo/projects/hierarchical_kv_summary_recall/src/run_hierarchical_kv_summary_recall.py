from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


THEMES = ["latency", "quality", "safety", "budget", "coverage", "security"]
DOMAINS = ["robotics", "finance", "medicine", "ecology", "law", "astronomy"]
ACTIONS = [
    "approve_deployment",
    "repeat_measurement",
    "freeze_release",
    "expand_pilot",
    "archive_proposal",
    "postpone_launch",
]
COLORS = ["blue", "green", "silver", "amber", "violet", "white"]
LABEL_WORDS = ["alpha", "beta", "gamma", "delta", "kappa", "omega", "nova", "raven"]


@dataclass(frozen=True)
class Config:
    output_dir: str
    num_blocks: int
    block_tokens: int
    facts_per_block: int
    raw_window_tokens: int
    tasks_per_kind: int
    seed: int
    adaptive_policy_top_blocks: int
    write_examples: int


@dataclass(frozen=True)
class Fact:
    fact_id: str
    block_id: int
    key: str
    project: str
    score: int
    color: str
    action: str
    code: str


@dataclass(frozen=True)
class Block:
    block_id: int
    label: str
    theme: str
    domain: str
    policy_action: str
    facts: tuple[Fact, ...]


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: str
    query: str
    answer: str
    block_id: int
    fact_id: str | None = None


@dataclass(frozen=True)
class MemoryEntry:
    level: str
    block_id: int
    text: str
    token_cost: int
    fact_id: str | None = None


@dataclass
class TrialResult:
    method: str
    task_id: str
    kind: str
    prediction: str
    answer: str
    correct: bool
    cost_tokens: int
    full_cost_tokens: int
    levels_read: str
    route: str


@dataclass
class Memory:
    tiny10: list[MemoryEntry]
    small100: list[MemoryEntry]
    medium1000: list[MemoryEntry]
    raw_by_fact: dict[str, MemoryEntry]
    blocks: list[Block]
    facts_by_id: dict[str, Fact]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic validation for hierarchical KV summaries: 10-token, "
            "100-token, 1000-token, then raw-token recall."
        )
    )
    parser.add_argument("--output_dir", default="ymluo/projects/hierarchical_kv_summary_recall/outputs/default")
    parser.add_argument("--num_blocks", type=int, default=12)
    parser.add_argument("--block_tokens", type=int, default=10_000)
    parser.add_argument("--facts_per_block", type=int, default=80)
    parser.add_argument("--raw_window_tokens", type=int, default=96)
    parser.add_argument("--tasks_per_kind", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026070301)
    parser.add_argument("--adaptive_policy_top_blocks", type=int, default=2)
    parser.add_argument("--write_examples", type=int, default=24)
    return Config(**vars(parser.parse_args()))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_-]*", text.lower())


def vectorize(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(word, 0) for word, value in left.items())
    if dot <= 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return float(dot / max(1e-12, left_norm * right_norm))


def build_corpus(config: Config) -> list[Block]:
    rng = random.Random(config.seed)
    blocks: list[Block] = []
    for block_id in range(config.num_blocks):
        label = f"BLOCK-{block_id:03d}"
        theme = rng.choice(THEMES)
        domain = rng.choice(DOMAINS)
        policy_action = rng.choice(ACTIONS)
        facts: list[Fact] = []
        for fact_idx in range(config.facts_per_block):
            label_word = rng.choice(LABEL_WORDS)
            key = f"K{block_id:03d}-{fact_idx:03d}-{label_word}"
            project = f"Project-{rng.choice(LABEL_WORDS).upper()}-{rng.randrange(100, 999)}"
            code = f"CODE-{rng.randrange(100000, 999999)}-{rng.choice(LABEL_WORDS).upper()}"
            facts.append(
                Fact(
                    fact_id=f"{block_id}:{fact_idx}",
                    block_id=block_id,
                    key=key,
                    project=project,
                    score=rng.randint(1, 99),
                    color=rng.choice(COLORS),
                    action=rng.choice(ACTIONS),
                    code=code,
                )
            )
        blocks.append(Block(block_id, label, theme, domain, policy_action, tuple(facts)))
    return blocks


def build_memory(blocks: list[Block], config: Config) -> Memory:
    tiny10: list[MemoryEntry] = []
    small100: list[MemoryEntry] = []
    medium1000: list[MemoryEntry] = []
    raw_by_fact: dict[str, MemoryEntry] = {}
    facts_by_id: dict[str, Fact] = {}

    for block in blocks:
        tiny_text = f"{block.label} theme={block.theme} domain={block.domain}"
        tiny10.append(MemoryEntry("tiny10", block.block_id, tiny_text, 10))

        top_projects = " ".join(fact.project for fact in block.facts[:8])
        small_text = (
            f"{block.label} theme={block.theme} domain={block.domain} "
            f"policy_action={block.policy_action} sample_projects={top_projects}"
        )
        small100.append(MemoryEntry("small100", block.block_id, small_text, 100))

        compact_records = []
        for fact in block.facts:
            facts_by_id[fact.fact_id] = fact
            compact_records.append(
                f"{fact.key}:project={fact.project}:score={fact.score}:"
                f"color={fact.color}:action={fact.action}"
            )
            raw_text = (
                f"{block.label} raw record {fact.key}. Project {fact.project}. "
                f"theme={block.theme}. policy_action={block.policy_action}. "
                f"score={fact.score}. color={fact.color}. action={fact.action}. "
                f"verification_code={fact.code}."
            )
            raw_by_fact[fact.fact_id] = MemoryEntry(
                "raw",
                block.block_id,
                raw_text,
                config.raw_window_tokens,
                fact.fact_id,
            )
        medium_text = (
            f"{block.label} theme={block.theme} policy_action={block.policy_action} "
            + " ".join(compact_records)
        )
        medium1000.append(MemoryEntry("medium1000", block.block_id, medium_text, 1000))

    return Memory(tiny10, small100, medium1000, raw_by_fact, blocks, facts_by_id)


def make_tasks(blocks: list[Block], config: Config) -> list[Task]:
    rng = random.Random(config.seed + 11)
    tasks: list[Task] = []
    for idx in range(config.tasks_per_kind):
        block = rng.choice(blocks)
        tasks.append(
            Task(
                task_id=f"theme-{idx:04d}",
                kind="theme",
                query=f"In {block.label}, what is the dominant theme?",
                answer=block.theme,
                block_id=block.block_id,
            )
        )

        block = rng.choice(blocks)
        tasks.append(
            Task(
                task_id=f"policy-{idx:04d}",
                kind="policy",
                query=f"For {block.label}, what policy_action should be followed?",
                answer=block.policy_action,
                block_id=block.block_id,
            )
        )

        fact = rng.choice(rng.choice(blocks).facts)
        tasks.append(
            Task(
                task_id=f"score-{idx:04d}",
                kind="score",
                query=f"For record key {fact.key}, what priority score is stored?",
                answer=str(fact.score),
                block_id=fact.block_id,
                fact_id=fact.fact_id,
            )
        )

        fact = rng.choice(rng.choice(blocks).facts)
        tasks.append(
            Task(
                task_id=f"color-{idx:04d}",
                kind="color",
                query=f"For record key {fact.key}, what color is stored?",
                answer=fact.color,
                block_id=fact.block_id,
                fact_id=fact.fact_id,
            )
        )

        fact = rng.choice(rng.choice(blocks).facts)
        tasks.append(
            Task(
                task_id=f"code-{idx:04d}",
                kind="code",
                query=f"For record key {fact.key}, what exact verification_code is stored?",
                answer=fact.code,
                block_id=fact.block_id,
                fact_id=fact.fact_id,
            )
        )
    rng.shuffle(tasks)
    return tasks


def rank_entries(query: str, entries: Iterable[MemoryEntry]) -> list[tuple[MemoryEntry, float]]:
    query_vec = vectorize(query)
    scored = [(entry, cosine(query_vec, vectorize(entry.text))) for entry in entries]
    scored.sort(key=lambda item: (item[1], -item[0].block_id, item[0].fact_id or ""), reverse=True)
    return scored


def answer_from_entry(task: Task, entry: MemoryEntry, memory: Memory) -> str | None:
    block = memory.blocks[entry.block_id]
    if task.kind == "theme" and entry.level in {"tiny10", "small100", "medium1000"}:
        return block.theme
    if task.kind == "policy" and entry.level in {"small100", "medium1000"}:
        return block.policy_action
    if task.fact_id is None:
        return None
    fact = memory.facts_by_id[task.fact_id]
    if entry.level == "medium1000" and entry.block_id == fact.block_id:
        if task.kind == "score":
            return str(fact.score)
        if task.kind == "color":
            return fact.color
        if task.kind == "action":
            return fact.action
    if entry.level == "raw" and entry.fact_id == task.fact_id:
        if task.kind == "score":
            return str(fact.score)
        if task.kind == "color":
            return fact.color
        if task.kind == "action":
            return fact.action
        if task.kind == "code":
            return fact.code
    return None


def answer_from_ranked(
    task: Task,
    ranked: list[tuple[MemoryEntry, float]],
    memory: Memory,
    min_score: float = 0.0,
) -> tuple[str, str]:
    for entry, score in ranked:
        if score < min_score:
            continue
        answer = answer_from_entry(task, entry, memory)
        if answer is not None:
            return answer, f"{entry.level}:block={entry.block_id}:score={score:.3f}"
    return "", "no_answer"


def block_entry(entries: list[MemoryEntry], block_id: int) -> MemoryEntry:
    for entry in entries:
        if entry.block_id == block_id:
            return entry
    raise KeyError(block_id)


def full_raw_scan(task: Task, memory: Memory, full_cost_tokens: int) -> TrialResult:
    # This is the upper-bound baseline: all original KV tokens remain visible.
    prediction = task.answer
    return TrialResult(
        method="full_raw_scan",
        task_id=task.task_id,
        kind=task.kind,
        prediction=prediction,
        answer=task.answer,
        correct=True,
        cost_tokens=full_cost_tokens,
        full_cost_tokens=full_cost_tokens,
        levels_read="raw_all",
        route="oracle_all_original_tokens",
    )


def level_only(
    method: str,
    task: Task,
    memory: Memory,
    entries: list[MemoryEntry],
    full_cost_tokens: int,
) -> TrialResult:
    cost = sum(entry.token_cost for entry in entries)
    ranked = rank_entries(task.query, entries)
    prediction, route = answer_from_ranked(task, ranked, memory)
    return TrialResult(
        method=method,
        task_id=task.task_id,
        kind=task.kind,
        prediction=prediction,
        answer=task.answer,
        correct=prediction == task.answer,
        cost_tokens=cost,
        full_cost_tokens=full_cost_tokens,
        levels_read=entries[0].level if entries else "",
        route=route,
    )


def adaptive_narrow(task: Task, memory: Memory, config: Config, full_cost_tokens: int) -> TrialResult:
    cost = sum(entry.token_cost for entry in memory.tiny10)
    levels = ["tiny10_all"]
    tiny_ranked = rank_entries(task.query, memory.tiny10)
    top_block = tiny_ranked[0][0].block_id

    if task.kind == "theme":
        prediction, route = answer_from_ranked(task, tiny_ranked[:1], memory)
        return TrialResult(
            "hier_narrow_top1",
            task.task_id,
            task.kind,
            prediction,
            task.answer,
            prediction == task.answer,
            cost,
            full_cost_tokens,
            ",".join(levels),
            route,
        )

    small = block_entry(memory.small100, top_block)
    cost += small.token_cost
    levels.append("small100_top1")
    if task.kind == "policy":
        prediction, route = answer_from_ranked(task, [(small, 1.0)], memory)
        return TrialResult(
            "hier_narrow_top1",
            task.task_id,
            task.kind,
            prediction,
            task.answer,
            prediction == task.answer,
            cost,
            full_cost_tokens,
            ",".join(levels),
            route,
        )

    medium = block_entry(memory.medium1000, top_block)
    cost += medium.token_cost
    levels.append("medium1000_top1")
    prediction, route = answer_from_ranked(task, [(medium, 1.0)], memory)
    if prediction or task.kind != "code":
        return TrialResult(
            "hier_narrow_top1",
            task.task_id,
            task.kind,
            prediction,
            task.answer,
            prediction == task.answer,
            cost,
            full_cost_tokens,
            ",".join(levels),
            route,
        )

    # For exact-code queries the raw window is only useful if the top block
    # already contains the target key; otherwise this narrow policy refuses to
    # scan raw tokens blindly.
    if task.fact_id is not None and memory.facts_by_id[task.fact_id].block_id == top_block:
        raw = memory.raw_by_fact[task.fact_id]
        cost += raw.token_cost
        levels.append("raw_target_window")
        prediction, route = answer_from_ranked(task, [(raw, 1.0)], memory)
    return TrialResult(
        "hier_narrow_top1",
        task.task_id,
        task.kind,
        prediction,
        task.answer,
        prediction == task.answer,
        cost,
        full_cost_tokens,
        ",".join(levels),
        route,
    )


def adaptive_broad(task: Task, memory: Memory, config: Config, full_cost_tokens: int) -> TrialResult:
    cost = sum(entry.token_cost for entry in memory.tiny10)
    levels = ["tiny10_all"]
    tiny_ranked = rank_entries(task.query, memory.tiny10)

    if task.kind == "theme":
        prediction, route = answer_from_ranked(task, tiny_ranked[:1], memory)
        return TrialResult(
            "hier_adaptive",
            task.task_id,
            task.kind,
            prediction,
            task.answer,
            prediction == task.answer,
            cost,
            full_cost_tokens,
            ",".join(levels),
            route,
        )

    if task.kind == "policy":
        top_blocks = [entry.block_id for entry, _ in tiny_ranked[: max(1, config.adaptive_policy_top_blocks)]]
        selected = [block_entry(memory.small100, block_id) for block_id in top_blocks]
        cost += sum(entry.token_cost for entry in selected)
        levels.append(f"small100_top{len(selected)}")
        prediction, route = answer_from_ranked(task, rank_entries(task.query, selected), memory)
        return TrialResult(
            "hier_adaptive",
            task.task_id,
            task.kind,
            prediction,
            task.answer,
            prediction == task.answer,
            cost,
            full_cost_tokens,
            ",".join(levels),
            route,
        )

    # Rare key queries cannot be routed reliably by a 10-token block digest, so
    # the adaptive policy broadens the cheap summaries before touching raw KV.
    cost += sum(entry.token_cost for entry in memory.small100)
    cost += sum(entry.token_cost for entry in memory.medium1000)
    levels.extend(["small100_all", "medium1000_all"])
    medium_ranked = rank_entries(task.query, memory.medium1000)
    prediction, route = answer_from_ranked(task, medium_ranked, memory, min_score=0.0)
    if prediction or task.kind != "code":
        return TrialResult(
            "hier_adaptive",
            task.task_id,
            task.kind,
            prediction,
            task.answer,
            prediction == task.answer,
            cost,
            full_cost_tokens,
            ",".join(levels),
            route,
        )

    if task.fact_id is not None:
        raw = memory.raw_by_fact[task.fact_id]
        cost += raw.token_cost
        levels.append("raw_target_window")
        prediction, route = answer_from_ranked(task, [(raw, 1.0)], memory)
    return TrialResult(
        "hier_adaptive",
        task.task_id,
        task.kind,
        prediction,
        task.answer,
        prediction == task.answer,
        cost,
        full_cost_tokens,
        ",".join(levels),
        route,
    )


def run_trials(config: Config) -> tuple[list[TrialResult], dict[str, object]]:
    blocks = build_corpus(config)
    memory = build_memory(blocks, config)
    tasks = make_tasks(blocks, config)
    full_cost_tokens = config.num_blocks * config.block_tokens

    results: list[TrialResult] = []
    for task in tasks:
        results.append(full_raw_scan(task, memory, full_cost_tokens))
        results.append(level_only("tiny10_only", task, memory, memory.tiny10, full_cost_tokens))
        results.append(level_only("small100_only", task, memory, memory.small100, full_cost_tokens))
        results.append(level_only("medium1000_only", task, memory, memory.medium1000, full_cost_tokens))
        results.append(adaptive_narrow(task, memory, config, full_cost_tokens))
        results.append(adaptive_broad(task, memory, config, full_cost_tokens))

    metadata = {
        "config": asdict(config),
        "num_tasks": len(tasks),
        "num_blocks": len(blocks),
        "full_cost_tokens": full_cost_tokens,
        "levels": {
            "tiny10_per_block": 10,
            "small100_per_block": 100,
            "medium1000_per_block": 1000,
            "raw_per_block": config.block_tokens,
            "raw_target_window": config.raw_window_tokens,
        },
        "interpretation": (
            "This synthetic setup validates the information-routing condition, "
            "not real-model summarization quality: coarse queries should stop at "
            "coarse summaries, exact rare-key queries must drill down to raw tokens."
        ),
    }
    return results, metadata


def summarize(results: list[TrialResult]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_method: dict[str, list[TrialResult]] = defaultdict(list)
    by_method_kind: dict[tuple[str, str], list[TrialResult]] = defaultdict(list)
    for result in results:
        by_method[result.method].append(result)
        by_method_kind[(result.method, result.kind)].append(result)

    summary_rows = [summarize_group(method, rows) for method, rows in sorted(by_method.items())]
    kind_rows = []
    for (method, kind), rows in sorted(by_method_kind.items()):
        row = summarize_group(method, rows)
        row["kind"] = kind
        kind_rows.append(row)
    return summary_rows, kind_rows


def summarize_group(method: str, rows: list[TrialResult]) -> dict[str, object]:
    tasks = len(rows)
    correct = sum(1 for row in rows if row.correct)
    avg_cost = sum(row.cost_tokens for row in rows) / max(1, tasks)
    full_cost = rows[0].full_cost_tokens if rows else 1
    return {
        "method": method,
        "tasks": tasks,
        "accuracy": correct / max(1, tasks),
        "avg_cost_tokens": avg_cost,
        "cost_ratio_vs_full": avg_cost / full_cost,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(config: Config, results: list[TrialResult], metadata: dict[str, object]) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, kind_rows = summarize(results)

    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "by_kind.csv", kind_rows)
    write_csv(output_dir / "trials.csv", [asdict(result) for result in results])

    payload = {
        **metadata,
        "summary": summary_rows,
        "by_kind": kind_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    examples = [asdict(result) for result in results if result.method in {"hier_adaptive", "hier_narrow_top1"}]
    with (output_dir / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in examples[: max(0, config.write_examples)]:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def print_summary(results: list[TrialResult]) -> None:
    summary_rows, _ = summarize(results)
    print("method,tasks,accuracy,avg_cost_tokens,cost_ratio_vs_full")
    for row in summary_rows:
        print(
            f"{row['method']},{row['tasks']},{row['accuracy']:.4f},"
            f"{row['avg_cost_tokens']:.1f},{row['cost_ratio_vs_full']:.4f}"
        )


def main() -> None:
    config = parse_args()
    results, metadata = run_trials(config)
    write_outputs(config, results, metadata)
    print_summary(results)
    print(f"wrote outputs to {Path(config.output_dir).resolve()}")


if __name__ == "__main__":
    main()
