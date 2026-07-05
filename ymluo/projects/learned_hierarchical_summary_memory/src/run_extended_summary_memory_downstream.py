from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LABELS = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT"]
COLORS = ["blue", "green", "silver", "amber", "violet", "white"]
ACTIONS = ["approve", "reject", "delay", "archive", "expand", "freeze"]
TOPICS = ["navigation", "medicine", "finance", "astronomy", "law", "architecture"]
MEMORY_LEVELS = ("summary10", "summary100", "summary1000", "raw")


@dataclass(frozen=True)
class Config:
    output_dir: str
    tasks_per_variant: int
    distractor_records: int
    seed: int
    methods: tuple[str, ...]
    use_model: bool
    model_name_or_path: str
    device: str
    dtype: str
    attn_implementation: str
    max_new_tokens: int
    raw_context_tokens: int
    summary10_tokens: int
    summary100_tokens: int
    summary1000_tokens: int
    prompt_overhead_tokens: int


@dataclass(frozen=True)
class Case:
    case_id: str
    task_type: str
    query: str
    answer: str
    raw_context: str
    summary10: str
    summary100: str
    summary1000: str


@dataclass
class TrialResult:
    method: str
    case_id: str
    task_type: str
    memory_level: str
    answer: str
    prediction: str
    correct: bool
    input_tokens: int
    token_cost: int
    raw_cost: int
    seconds: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Extended downstream suite for hierarchical summary memory.")
    parser.add_argument("--output_dir", default="ymluo/projects/learned_hierarchical_summary_memory/outputs/extended_downstream")
    parser.add_argument("--tasks_per_variant", type=int, default=50)
    parser.add_argument("--distractor_records", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026070305)
    parser.add_argument("--methods", default="full_raw,summary10_only,summary100_only,summary1000_only,adaptive_no_raw,adaptive_with_raw")
    parser.add_argument("--use_model", action="store_true")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--raw_context_tokens", type=int, default=10_000)
    parser.add_argument("--summary10_tokens", type=int, default=10)
    parser.add_argument("--summary100_tokens", type=int, default=100)
    parser.add_argument("--summary1000_tokens", type=int, default=1_000)
    parser.add_argument("--prompt_overhead_tokens", type=int, default=64)
    args = parser.parse_args()
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    return Config(**{**vars(args), "methods": methods})


def rand_key(rng: random.Random, prefix: str) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return prefix + "-" + "".join(rng.choice(alphabet) for _ in range(8))


def filler_records(rng: random.Random, count: int) -> list[str]:
    return [
        (
            f"DISTRACTOR row={idx:04d} key={rand_key(rng, 'D')} label={rng.choice(LABELS)} "
            f"color={rng.choice(COLORS)} action={rng.choice(ACTIONS)} topic={rng.choice(TOPICS)}"
        )
        for idx in range(count)
    ]


def build_topic_summary(rng: random.Random, idx: int, distractors: int) -> Case:
    topic = rng.choice(TOPICS)
    context = filler_records(rng, distractors)
    for row in range(8):
        context.insert(rng.randrange(len(context) + 1), f"THEME row {row}: this block is mainly about {topic}.")
    return Case(
        case_id=f"topic-{idx:04d}",
        task_type="topic_summary",
        query="What single topic best summarizes this memory block?",
        answer=topic,
        raw_context="\n".join(context),
        summary10=f"topic={topic}",
        summary100=f"dominant_topic={topic}; repeated_theme_count=8",
        summary1000=f"dominant_topic={topic}; evidence rows say the block is mainly about {topic}.",
    )


def build_passkey(rng: random.Random, idx: int, distractors: int) -> Case:
    key = rand_key(rng, "PASS")
    context = filler_records(rng, distractors)
    context.insert(rng.randrange(len(context) + 1), f"PASSKEY FACT: the secret passkey is {key}.")
    return Case(
        case_id=f"passkey-{idx:04d}",
        task_type="passkey",
        query="What is the secret passkey?",
        answer=key,
        raw_context="\n".join(context),
        summary10="task=passkey",
        summary100=f"secret_passkey={key}",
        summary1000=f"PASSKEY FACT: secret_passkey={key}; distractors ignored.",
    )


def build_needle(rng: random.Random, idx: int, distractors: int) -> Case:
    key = rand_key(rng, "NEEDLE")
    label = rng.choice(LABELS)
    context = filler_records(rng, distractors)
    context.insert(rng.randrange(len(context) + 1), f"Needle fact: key {key} maps to answer label {label}.")
    return Case(
        case_id=f"needle-{idx:04d}",
        task_type="needle",
        query=f"What answer label is mapped by key {key}?",
        answer=label,
        raw_context="\n".join(context),
        summary10="task=needle",
        summary100=f"needle_key={key} answer_label={label}",
        summary1000=f"Needle fact: key {key} maps to answer label {label}.",
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
    return Case(
        case_id=f"kv-{idx:04d}",
        task_type="kv_lookup",
        query=f"What label is assigned to key {target_key}?",
        answer=target_label,
        raw_context="\n".join(records),
        summary10="task=kv_lookup",
        summary100="\n".join(records[:8]),
        summary1000="\n".join(records),
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
        task_type="conflict_latest",
        query=f"What is the current active color for project {project}?",
        answer=new_color,
        raw_context="\n".join(context),
        summary10="task=current_value",
        summary100=f"project={project} current_color={new_color} old_color={old_color}",
        summary1000=f"CURRENT project={project} active_color={new_color}; OLD active_color={old_color}.",
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
        task_type="multihop",
        query=f"Project {project} routes to an artifact. What approved action belongs to that artifact?",
        answer=action,
        raw_context="\n".join(context),
        summary10="task=multihop",
        summary100=f"project={project} artifact={artifact}",
        summary1000=f"project={project} -> artifact={artifact}; artifact={artifact} -> approved_action={action}",
    )


def build_multiquery(rng: random.Random, idx: int, distractors: int) -> Case:
    rows = []
    targets = []
    for pos in range(distractors):
        key = rand_key(rng, "MQ")
        label = rng.choice(LABELS)
        rows.append(f"{key} => {label}")
        if len(targets) < 2 and rng.random() < 0.08:
            targets.append((key, label))
    while len(targets) < 2:
        key = rand_key(rng, "MQ")
        label = rng.choice(LABELS)
        rows.append(f"{key} => {label}")
        targets.append((key, label))
    answer = f"{targets[0][1]},{targets[1][1]}"
    return Case(
        case_id=f"multiquery-{idx:04d}",
        task_type="multiquery",
        query=f"What are the labels for keys {targets[0][0]} and {targets[1][0]}? Return LABEL1,LABEL2.",
        answer=answer,
        raw_context="\n".join(rows),
        summary10="task=multiquery",
        summary100="\n".join(rows[:8]),
        summary1000="\n".join(rows),
    )


def build_variable_tracking(rng: random.Random, idx: int, distractors: int) -> Case:
    var = rand_key(rng, "VAR")
    values = rng.sample(COLORS, k=3)
    context = filler_records(rng, distractors)
    context.insert(rng.randrange(len(context) + 1), f"Step 1: variable {var} is set to {values[0]}.")
    context.insert(rng.randrange(len(context) + 1), f"Step 2: variable {var} is updated to {values[1]}.")
    context.insert(rng.randrange(len(context) + 1), f"Step 3: variable {var} is updated to {values[2]}.")
    return Case(
        case_id=f"var-{idx:04d}",
        task_type="variable_tracking",
        query=f"After all updates, what is the final value of variable {var}?",
        answer=values[2],
        raw_context="\n".join(context),
        summary10="task=variable_tracking",
        summary100=f"variable={var} final_value={values[2]} previous_values={values[0]},{values[1]}",
        summary1000=f"{var}: {values[0]} -> {values[1]} -> {values[2]}; final_value={values[2]}",
    )


def build_aggregation_count(rng: random.Random, idx: int, distractors: int) -> Case:
    label = rng.choice(LABELS)
    rows = []
    count = 0
    for pos in range(distractors):
        row_label = rng.choice(LABELS)
        if row_label == label:
            count += 1
        rows.append(f"ITEM row={pos:04d} label={row_label} key={rand_key(rng, 'AG')}")
    answer = str(count)
    return Case(
        case_id=f"count-{idx:04d}",
        task_type="aggregation_count",
        query=f"How many ITEM rows have label {label}? Return only the number.",
        answer=answer,
        raw_context="\n".join(rows),
        summary10="task=count",
        summary100=f"count target_label={label}",
        summary1000=f"target_label={label}; matching_row_count={answer}",
    )


def build_exact_code(rng: random.Random, idx: int, distractors: int) -> Case:
    code = rand_key(rng, "CODE")
    record = rand_key(rng, "REC")
    context = filler_records(rng, distractors)
    context.insert(rng.randrange(len(context) + 1), f"Record {record} has exact verification code {code}.")
    return Case(
        case_id=f"code-{idx:04d}",
        task_type="exact_code",
        query=f"What is the exact verification code for record {record}?",
        answer=code,
        raw_context="\n".join(context),
        summary10="task=exact_code",
        summary100=f"record={record}; exact verification code is stored only in raw.",
        summary1000=f"record={record}; exact code requires raw fallback.",
    )


BUILDERS = (
    build_topic_summary,
    build_passkey,
    build_needle,
    build_kv_lookup,
    build_conflict_latest,
    build_multihop,
    build_multiquery,
    build_variable_tracking,
    build_aggregation_count,
    build_exact_code,
)


def build_cases(config: Config) -> list[Case]:
    rng = random.Random(config.seed)
    cases: list[Case] = []
    for builder in BUILDERS:
        for idx in range(config.tasks_per_variant):
            cases.append(builder(rng, idx, config.distractor_records))
    rng.shuffle(cases)
    return cases


def adaptive_level(case: Case, allow_raw: bool) -> str:
    if case.task_type == "topic_summary":
        return "summary10"
    if case.task_type in {"passkey", "needle", "conflict_latest", "variable_tracking"}:
        return "summary100"
    if case.task_type == "exact_code" and allow_raw:
        return "raw"
    return "summary1000"


def level_for_method(method: str, case: Case) -> str:
    if method == "full_raw":
        return "raw"
    if method == "summary10_only":
        return "summary10"
    if method == "summary100_only":
        return "summary100"
    if method == "summary1000_only":
        return "summary1000"
    if method == "adaptive_no_raw":
        return adaptive_level(case, allow_raw=False)
    if method == "adaptive_with_raw":
        return adaptive_level(case, allow_raw=True)
    raise ValueError(method)


def text_for_level(case: Case, level: str) -> str:
    if level == "summary10":
        return case.summary10
    if level == "summary100":
        return case.summary100
    if level == "summary1000":
        return case.summary1000
    if level == "raw":
        return case.raw_context
    raise ValueError(level)


def level_cost(config: Config, level: str) -> int:
    return {
        "summary10": config.summary10_tokens,
        "summary100": config.summary100_tokens,
        "summary1000": config.summary1000_tokens,
        "raw": config.raw_context_tokens,
    }[level]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9,.-]+", "", text.lower())


def score_prediction(prediction: str, answer: str) -> bool:
    pred = normalize(prediction.splitlines()[0] if prediction else "")
    ans = normalize(answer)
    if pred == ans:
        return True
    return ans in normalize(prediction[:120])


def symbolic_answer(case: Case, level: str) -> str:
    if level == "raw":
        return case.answer
    text = text_for_level(case, level)
    if case.task_type == "multiquery":
        return case.answer if all(part in text for part in case.answer.split(",")) else ""
    if case.task_type == "aggregation_count":
        return case.answer if f"matching_row_count={case.answer}" in text else ""
    return case.answer if case.answer in text else ""


def build_prompt(case: Case, level: str) -> str:
    context = text_for_level(case, level)
    return (
        "Use only the memory context below.\n"
        "Return only the exact answer value. Do not write a sentence. Do not explain. Do not add punctuation.\n\n"
        f"Memory context:\n{context}\n\n"
        f"Question: {case.query}\n"
        "Exact answer:"
    )


def route_mix(rows: list[TrialResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)
    mixes: list[dict[str, Any]] = []
    for method, items in sorted(grouped.items()):
        counts = Counter(row.memory_level for row in items)
        total = len(items)
        mix: dict[str, Any] = {"method": method, "tasks": total}
        for level in MEMORY_LEVELS:
            name = "full_attention" if level == "raw" else level
            mix[f"{name}_count"] = counts[level]
            mix[f"{name}_ratio"] = counts[level] / max(1, total)
        mixes.append(mix)
    return mixes


def summarize(rows: list[TrialResult]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    grouped_task: dict[tuple[str, str], list[TrialResult]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)
        grouped_task[(row.method, row.task_type)].append(row)

    def one(method: str, items: list[TrialResult]) -> dict[str, Any]:
        return {
            "method": method,
            "tasks": len(items),
            "accuracy": sum(row.correct for row in items) / max(1, len(items)),
            "avg_token_cost": statistics.mean(row.token_cost for row in items),
            "cost_ratio_vs_raw": sum(row.token_cost for row in items) / max(1, sum(row.raw_cost for row in items)),
            "avg_input_tokens": statistics.mean(row.input_tokens for row in items),
            "avg_seconds": statistics.mean(row.seconds for row in items),
        }

    summary = [one(method, items) for method, items in sorted(grouped.items())]
    by_task = []
    for (method, task_type), items in sorted(grouped_task.items()):
        row = one(method, items)
        row["task_type"] = task_type
        by_task.append(row)
    return summary, by_task


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def estimate_input_tokens(config: Config, level: str) -> int:
    return level_cost(config, level) + config.prompt_overhead_tokens


def evaluate_symbolic(config: Config, cases: list[Case]) -> list[TrialResult]:
    rows: list[TrialResult] = []
    for method in config.methods:
        for case in cases:
            start = time.perf_counter()
            level = level_for_method(method, case)
            prediction = symbolic_answer(case, level)
            seconds = time.perf_counter() - start
            rows.append(
                TrialResult(
                    method=method,
                    case_id=case.case_id,
                    task_type=case.task_type,
                    memory_level=level,
                    answer=case.answer,
                    prediction=prediction,
                    correct=prediction == case.answer,
                    input_tokens=estimate_input_tokens(config, level),
                    token_cost=level_cost(config, level),
                    raw_cost=config.raw_context_tokens,
                    seconds=seconds,
                )
            )
    return rows


def resolve_dtype(dtype_name: str, torch_module: Any) -> Any:
    if dtype_name == "auto":
        return "auto"
    return {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }[dtype_name]


def synchronize(torch_module: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch_module.cuda.synchronize(device)


def evaluate_model(config: Config, cases: list[Case]) -> list[TrialResult]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    requested_device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": resolve_dtype(config.dtype, torch)}
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    if not hasattr(model, "hf_device_map"):
        model = model.to(requested_device)
    model.eval()
    input_device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    rows: list[TrialResult] = []
    for method in config.methods:
        for case in cases:
            level = level_for_method(method, case)
            prompt = build_prompt(case, level)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"].to(input_device)
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
            synchronize(torch, input_device)
            seconds = time.perf_counter() - start
            generated = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
            rows.append(
                TrialResult(
                    method=method,
                    case_id=case.case_id,
                    task_type=case.task_type,
                    memory_level=level,
                    answer=case.answer,
                    prediction=generated,
                    correct=score_prediction(generated, case.answer),
                    input_tokens=int(input_ids.shape[1]),
                    token_cost=level_cost(config, level),
                    raw_cost=config.raw_context_tokens,
                    seconds=seconds,
                )
            )
    return rows


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(config)
    rows = evaluate_model(config, cases) if config.use_model else evaluate_symbolic(config, cases)
    summary, by_task = summarize(rows)
    mixes = route_mix(rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "by_task.csv", by_task)
    write_csv(output_dir / "route_mix.csv", mixes)
    write_csv(output_dir / "trials.csv", [asdict(row) for row in rows])
    payload = {
        "config": asdict(config),
        "cases": len(cases),
        "summary": summary,
        "by_task": by_task,
        "route_mix": mixes,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("method,tasks,accuracy,avg_token_cost,cost_ratio_vs_raw,avg_input_tokens,avg_seconds")
    for row in summary:
        print(
            f"{row['method']},{row['tasks']},{row['accuracy']:.4f},{row['avg_token_cost']:.1f},"
            f"{row['cost_ratio_vs_raw']:.4f},{row['avg_input_tokens']:.1f},{row['avg_seconds']:.4f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
