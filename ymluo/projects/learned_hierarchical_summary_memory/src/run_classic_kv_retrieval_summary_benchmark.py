from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LABELS = ["A", "B", "C", "D"]
COLORS = ["blue", "green", "silver", "amber", "violet", "white"]
ACTIONS = ["approve", "reject", "delay", "archive", "expand", "freeze"]
MEMORY_LEVELS = ("summary10", "summary100", "summary1000", "raw")


@dataclass(frozen=True)
class Config:
    output_dir: str
    tasks_per_variant: int
    distractor_records: int
    seed: int
    raw_context_tokens: int
    summary10_tokens: int
    summary100_tokens: int
    summary1000_tokens: int


@dataclass(frozen=True)
class Case:
    case_id: str
    variant: str
    query_type: str
    answer: str
    summary10: str
    summary100: str
    summary1000: str
    raw_context: str


@dataclass
class TrialResult:
    method: str
    case_id: str
    variant: str
    query_type: str
    answer: str
    prediction: str
    correct: bool
    token_cost: int
    raw_cost: int
    memory_level: str


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Classic KV retrieval benchmark for summary memory.")
    parser.add_argument("--output_dir", default="ymluo/projects/learned_hierarchical_summary_memory/outputs/classic_kv_retrieval")
    parser.add_argument("--tasks_per_variant", type=int, default=80)
    parser.add_argument("--distractor_records", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026070303)
    parser.add_argument("--raw_context_tokens", type=int, default=10_000)
    parser.add_argument("--summary10_tokens", type=int, default=10)
    parser.add_argument("--summary100_tokens", type=int, default=100)
    parser.add_argument("--summary1000_tokens", type=int, default=1_000)
    return Config(**vars(parser.parse_args()))


def rand_key(rng: random.Random, prefix: str) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return prefix + "-" + "".join(rng.choice(alphabet) for _ in range(8))


def filler_records(rng: random.Random, count: int) -> list[str]:
    rows = []
    for idx in range(count):
        rows.append(
            f"DISTRACTOR row={idx:03d} key={rand_key(rng, 'D')} label={rng.choice(LABELS)} "
            f"color={rng.choice(COLORS)} action={rng.choice(ACTIONS)} checksum={rand_key(rng, 'CHK')}"
        )
    return rows


def build_passkey(rng: random.Random, idx: int, distractors: int) -> Case:
    key = rand_key(rng, "PASS")
    context = filler_records(rng, distractors)
    context.insert(rng.randrange(len(context) + 1), f"PASSKEY: the secret passkey is {key}.")
    return Case(
        case_id=f"passkey-{idx:04d}",
        variant="passkey",
        query_type="exact_lookup",
        answer=key,
        summary10="task=passkey exact_lookup",
        summary100=f"passkey={key}",
        summary1000=f"passkey={key}; all distractors ignored",
        raw_context="\n".join(context),
    )


def build_needle(rng: random.Random, idx: int, distractors: int) -> Case:
    label = rng.choice(LABELS)
    key = rand_key(rng, "NEEDLE")
    needle = f"Needle fact: key {key} maps to answer label {label}."
    context = filler_records(rng, distractors)
    context.insert(rng.randrange(len(context) + 1), needle)
    return Case(
        case_id=f"needle-{idx:04d}",
        variant="needle",
        query_type="label_lookup",
        answer=label,
        summary10="task=needle label_lookup",
        summary100=f"needle_key={key} answer_label={label}",
        summary1000=f"{needle}",
        raw_context="\n".join(context),
    )


def build_kv_lookup(rng: random.Random, idx: int, distractors: int) -> Case:
    records = []
    target_pos = rng.randrange(distractors)
    target_key = ""
    target_label = ""
    for pos in range(distractors):
        key = rand_key(rng, "K")
        label = rng.choice(LABELS)
        records.append(f"{key} => {label}")
        if pos == target_pos:
            target_key = key
            target_label = label
    summary_rows = records[:]
    return Case(
        case_id=f"kv-{idx:04d}",
        variant="kv_lookup",
        query_type="label_lookup",
        answer=target_label,
        summary10="task=kv_lookup table",
        summary100="\n".join(summary_rows[:8]),
        summary1000="\n".join(summary_rows),
        raw_context="\n".join(records),
    )


def build_conflict_latest(rng: random.Random, idx: int, distractors: int) -> Case:
    project = rand_key(rng, "PROJ")
    old_color = rng.choice(COLORS)
    new_color = rng.choice([item for item in COLORS if item != old_color])
    context = filler_records(rng, distractors)
    context.insert(rng.randrange(len(context) + 1), f"OLD profile: project {project} active_color={old_color}. This is obsolete.")
    context.insert(rng.randrange(len(context) + 1), f"CURRENT profile: project {project} active_color={new_color}. This supersedes old profiles.")
    return Case(
        case_id=f"conflict-{idx:04d}",
        variant="conflict_latest",
        query_type="current_value",
        answer=new_color,
        summary10="task=current_value conflict",
        summary100=f"project={project} current_color={new_color} old_color={old_color}",
        summary1000=f"CURRENT profile project={project} active_color={new_color}; OLD active_color={old_color}",
        raw_context="\n".join(context),
    )


def build_multihop(rng: random.Random, idx: int, distractors: int) -> Case:
    project = rand_key(rng, "PROJ")
    artifact = rand_key(rng, "ART")
    action = rng.choice(ACTIONS)
    context = filler_records(rng, distractors)
    context.insert(rng.randrange(len(context) + 1), f"Bridge: project {project} routes to artifact {artifact}.")
    context.insert(rng.randrange(len(context) + 1), f"Artifact memo: artifact {artifact} approved_action={action}.")
    return Case(
        case_id=f"multihop-{idx:04d}",
        variant="multihop",
        query_type="bridge_lookup",
        answer=action,
        summary10="task=multihop project artifact action",
        summary100=f"project={project} artifact={artifact}",
        summary1000=f"project={project} -> artifact={artifact}; artifact={artifact} -> action={action}",
        raw_context="\n".join(context),
    )


def build_exact_code(rng: random.Random, idx: int, distractors: int) -> Case:
    code = rand_key(rng, "CODE")
    record = rand_key(rng, "REC")
    context = filler_records(rng, distractors)
    context.insert(rng.randrange(len(context) + 1), f"Record {record} has exact verification code {code}.")
    return Case(
        case_id=f"exact-code-{idx:04d}",
        variant="exact_code",
        query_type="exact_code",
        answer=code,
        summary10="task=exact_code",
        summary100=f"record={record} has verification code stored in raw",
        summary1000=f"record={record} exact code requires raw fallback",
        raw_context="\n".join(context),
    )


BUILDERS = [
    build_passkey,
    build_needle,
    build_kv_lookup,
    build_conflict_latest,
    build_multihop,
    build_exact_code,
]


def build_cases(config: Config) -> list[Case]:
    rng = random.Random(config.seed)
    cases = []
    for builder in BUILDERS:
        for idx in range(config.tasks_per_variant):
            cases.append(builder(rng, idx, config.distractor_records))
    rng.shuffle(cases)
    return cases


def contains_answer(text: str, answer: str) -> bool:
    return answer in text


def answer_from_level(case: Case, level: str) -> str:
    if level == "summary10":
        text = case.summary10
    elif level == "summary100":
        text = case.summary100
    elif level == "summary1000":
        text = case.summary1000
    elif level == "raw":
        text = case.raw_context
    else:
        raise ValueError(level)
    return case.answer if contains_answer(text, case.answer) else ""


def adaptive_level(case: Case, allow_raw: bool) -> str:
    if case.variant in {"passkey", "needle", "conflict_latest"}:
        return "summary100"
    if case.variant in {"kv_lookup", "multihop"}:
        return "summary1000"
    if case.variant == "exact_code":
        return "raw" if allow_raw else "summary1000"
    return "summary1000"


def token_cost(level: str, config: Config) -> int:
    return {
        "summary10": config.summary10_tokens,
        "summary100": config.summary100_tokens,
        "summary1000": config.summary1000_tokens,
        "raw": config.raw_context_tokens,
    }[level]


def evaluate(
    method: str,
    cases: list[Case],
    config: Config,
    fixed_level: str | None = None,
    allow_raw: bool = False,
) -> list[TrialResult]:
    rows = []
    for case in cases:
        level = "raw" if method == "full_raw" else (fixed_level or adaptive_level(case, allow_raw))
        prediction = answer_from_level(case, level)
        rows.append(
            TrialResult(
                method=method,
                case_id=case.case_id,
                variant=case.variant,
                query_type=case.query_type,
                answer=case.answer,
                prediction=prediction,
                correct=prediction == case.answer,
                token_cost=token_cost(level, config),
                raw_cost=config.raw_context_tokens,
                memory_level=level,
            )
        )
    return rows


def summarize(rows: list[TrialResult]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    grouped_variant: dict[tuple[str, str], list[TrialResult]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)
        grouped_variant[(row.method, row.variant)].append(row)

    def one(method: str, items: list[TrialResult]) -> dict[str, Any]:
        return {
            "method": method,
            "tasks": len(items),
            "accuracy": sum(item.correct for item in items) / max(1, len(items)),
            "avg_token_cost": sum(item.token_cost for item in items) / max(1, len(items)),
            "cost_ratio_vs_raw": sum(item.token_cost for item in items) / max(1, sum(item.raw_cost for item in items)),
        }

    summary = [one(method, items) for method, items in sorted(grouped.items())]
    by_variant = []
    for (method, variant), items in sorted(grouped_variant.items()):
        row = one(method, items)
        row["variant"] = variant
        by_variant.append(row)
    return summary, by_variant


def route_mix(rows: list[TrialResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)

    mixes: list[dict[str, Any]] = []
    for method, items in sorted(grouped.items()):
        total = len(items)
        counts = Counter(row.memory_level for row in items)
        mix: dict[str, Any] = {"method": method, "tasks": total}
        for level in MEMORY_LEVELS:
            name = "full_attention" if level == "raw" else level
            mix[f"{name}_count"] = counts[level]
            mix[f"{name}_ratio"] = counts[level] / max(1, total)
        mixes.append(mix)
    return mixes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(config)
    rows: list[TrialResult] = []
    rows.extend(evaluate("full_raw", cases, config))
    rows.extend(evaluate("summary10_only", cases, config, fixed_level="summary10"))
    rows.extend(evaluate("summary100_only", cases, config, fixed_level="summary100"))
    rows.extend(evaluate("summary1000_only", cases, config, fixed_level="summary1000"))
    rows.extend(evaluate("adaptive_no_raw", cases, config, allow_raw=False))
    rows.extend(evaluate("adaptive_with_raw", cases, config, allow_raw=True))
    summary, by_variant = summarize(rows)
    mixes = route_mix(rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "by_variant.csv", by_variant)
    write_csv(output_dir / "route_mix.csv", mixes)
    write_csv(output_dir / "trials.csv", [asdict(row) for row in rows])
    payload = {
        "config": asdict(config),
        "cases": len(cases),
        "summary": summary,
        "by_variant": by_variant,
        "route_mix": mixes,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("method,tasks,accuracy,avg_token_cost,cost_ratio_vs_raw")
    for row in summary:
        print(
            f"{row['method']},{row['tasks']},{row['accuracy']:.4f},"
            f"{row['avg_token_cost']:.1f},{row['cost_ratio_vs_raw']:.4f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
