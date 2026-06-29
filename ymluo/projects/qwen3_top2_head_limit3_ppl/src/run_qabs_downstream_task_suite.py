from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    install_qwen3_attention_patch,
    pick_input_device,
    prefill_cache,
    resolve_dtype,
)
from run_qabs_downstream_kv_retrieval import (  # noqa: E402
    LABELS,
    TOPICS,
    Config as RetrievalConfig,
    build_task as build_structured_noisy_task,
    eval_task,
    make_key,
    write_csv,
)


TaskBuilder = Callable[[random.Random, int, int], dict[str, Any]]


@dataclass(frozen=True)
class SuiteConfig:
    model_name_or_path: str
    output_dir: str
    variants: str
    tasks_per_variant: int
    records_per_task: int
    seed: int
    chunk_size: int
    dtype: str
    device: str
    device_map: str
    attn_implementation: str
    top_fraction: float
    protect_sink_tokens: int
    protect_recent_tokens: int
    modes: str
    log_every: int


def parse_args() -> SuiteConfig:
    parser = argparse.ArgumentParser(description="Multi-task downstream suite for QABS KV compression.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--variants",
        default="structured_noisy,compact_kv,natural_kv,json_kv,needle_sentence",
        help="Comma-separated task variants.",
    )
    parser.add_argument("--tasks_per_variant", type=int, default=16)
    parser.add_argument("--records_per_task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=202606295)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.05)
    parser.add_argument("--protect_sink_tokens", type=int, default=10)
    parser.add_argument("--protect_recent_tokens", type=int, default=10)
    parser.add_argument("--modes", default="baseline,qabs8cand5reuse")
    parser.add_argument("--log_every", type=int, default=8)
    return SuiteConfig(**vars(parser.parse_args()))


def _records(rng: random.Random, task_idx: int, records_per_task: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    target_index = rng.randrange(records_per_task)
    for idx in range(records_per_task):
        key = make_key(rng, f"K{task_idx:03d}{idx:03d}")
        label = rng.choice(LABELS)
        topic = TOPICS[(task_idx + idx) % len(TOPICS)]
        checksum = make_key(rng, "CHK")
        records.append({"key": key, "label": label, "topic": topic, "checksum": checksum})
    return records, records[target_index] | {"target_index": target_index}


def build_compact_kv_task(rng: random.Random, task_idx: int, records_per_task: int) -> dict[str, Any]:
    records, target = _records(rng, task_idx, records_per_task)
    context = "\n".join(f"{record['key']} => {record['label']}" for record in records) + "\n"
    query = f"\nFor key {target['key']}, the label is:"
    return _task_payload("compact_kv", task_idx, target, context, query)


def build_natural_kv_task(rng: random.Random, task_idx: int, records_per_task: int) -> dict[str, Any]:
    records, target = _records(rng, task_idx, records_per_task)
    lines = [
        f"In the {record['topic']} file, lookup key {record['key']} has answer label {record['label']}."
        for record in records
    ]
    context = "\n".join(lines) + "\n"
    query = f"\nQuestion: What is the answer label for lookup key {target['key']}? Answer label:"
    return _task_payload("natural_kv", task_idx, target, context, query)


def build_json_kv_task(rng: random.Random, task_idx: int, records_per_task: int) -> dict[str, Any]:
    records, target = _records(rng, task_idx, records_per_task)
    lines = [
        json.dumps(
            {"key": record["key"], "topic": record["topic"], "answer_label": record["label"], "checksum": record["checksum"]},
            separators=(",", ":"),
        )
        for record in records
    ]
    context = "\n".join(lines) + "\n"
    query = f"\nFind the JSON object with key {target['key']}. Its answer_label is:"
    return _task_payload("json_kv", task_idx, target, context, query)


def build_needle_sentence_task(rng: random.Random, task_idx: int, records_per_task: int) -> dict[str, Any]:
    records, target = _records(rng, task_idx, records_per_task)
    lines = []
    for idx, record in enumerate(records):
        filler = (
            f"Audit note {idx}: the {record['topic']} packet passed checksum {record['checksum']} "
            "and should be ignored unless its key is requested."
        )
        needle = f" Needle fact: key {record['key']} maps to option {record['label']}."
        lines.append(filler + needle)
    context = "\n".join(lines) + "\n"
    query = f"\nUsing the needle facts, key {target['key']} maps to option:"
    return _task_payload("needle_sentence", task_idx, target, context, query)


def build_topic_table_task(rng: random.Random, task_idx: int, records_per_task: int) -> dict[str, Any]:
    records, target = _records(rng, task_idx, records_per_task)
    lines = [
        f"row={idx:03d} | topic={record['topic']} | id={record['key']} | class={record['label']} | checksum={record['checksum']}"
        for idx, record in enumerate(records)
    ]
    context = "\n".join(lines) + "\n"
    query = f"\nRead the table row with id={target['key']}. The class is:"
    return _task_payload("topic_table", task_idx, target, context, query)


def build_structured_noisy_variant_task(rng: random.Random, task_idx: int, records_per_task: int) -> dict[str, Any]:
    task = build_structured_noisy_task(rng, task_idx, records_per_task)
    task["variant"] = "structured_noisy"
    return task


def _task_payload(
    variant: str,
    task_idx: int,
    target: dict[str, Any],
    context: str,
    query: str,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "task_id": task_idx,
        "target_key": target["key"],
        "target_label": target["label"],
        "target_index": target["target_index"],
        "context": context,
        "query": query,
    }


BUILDERS: dict[str, TaskBuilder] = {
    "structured_noisy": build_structured_noisy_variant_task,
    "compact_kv": build_compact_kv_task,
    "natural_kv": build_natural_kv_task,
    "json_kv": build_json_kv_task,
    "needle_sentence": build_needle_sentence_task,
    "topic_table": build_topic_table_task,
}


def retrieval_config(config: SuiteConfig) -> RetrievalConfig:
    return RetrievalConfig(
        model_name_or_path=config.model_name_or_path,
        output_dir=config.output_dir,
        tasks=config.tasks_per_variant,
        records_per_task=config.records_per_task,
        seed=config.seed,
        chunk_size=config.chunk_size,
        dtype=config.dtype,
        device=config.device,
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
        top_fraction=config.top_fraction,
        protect_sink_tokens=config.protect_sink_tokens,
        protect_recent_tokens=config.protect_recent_tokens,
        modes=config.modes,
        layer_budget_map_path="",
        log_every=config.log_every,
    )


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    variants = [name.strip() for name in config.variants.split(",") if name.strip()]
    unknown = [name for name in variants if name not in BUILDERS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available={sorted(BUILDERS)}")
    modes = [mode.strip() for mode in config.modes.split(",") if mode.strip()]
    if "baseline" not in modes:
        modes.insert(0, "baseline")

    device = torch.device(config.device)
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
    install_qwen3_attention_patch()
    input_device = pick_input_device(model, device)
    eval_config = retrieval_config(config)

    rows: list[dict[str, Any]] = []
    public_tasks: list[dict[str, Any]] = []
    started = time.perf_counter()
    for variant in variants:
        rng = random.Random(config.seed + 1009 * variants.index(variant))
        tasks = [BUILDERS[variant](rng, idx, config.records_per_task) for idx in range(config.tasks_per_variant)]
        for task_idx, task in enumerate(tasks, start=1):
            if task_idx == 1 or task_idx == len(tasks) or task_idx % config.log_every == 0:
                print(f"{variant} task {task_idx}/{len(tasks)}", flush=True)
            public_tasks.append({key: value for key, value in task.items() if key not in {"context", "query"}})
            context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            context_cache, context_prev = prefill_cache(
                model,
                context_ids,
                context_ids.shape[-1],
                config.chunk_size,
                input_device,
            )
            for mode in modes:
                result = eval_task(model, tokenizer, task, eval_config, input_device, mode, context_cache, context_prev)
                result["variant"] = variant
                rows.append(result)

    with (output_dir / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in public_tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")

    fields = ["variant", "task_id", "mode", "target_key", "target_index", "target_label", "pred_label", "correct"] + [
        f"score_{label}" for label in LABELS
    ]
    write_csv(output_dir / "downstream_task_suite_results.csv", rows, fields)

    summary_rows = []
    for variant in variants:
        for mode in modes:
            subset = [row for row in rows if row["variant"] == variant and row["mode"] == mode]
            correct = sum(int(row["correct"]) for row in subset)
            summary_rows.append(
                {
                    "variant": variant,
                    "mode": mode,
                    "correct": correct,
                    "total": len(subset),
                    "accuracy": correct / max(1, len(subset)),
                }
            )
    write_csv(output_dir / "summary_by_variant_mode.csv", summary_rows, ["variant", "mode", "correct", "total", "accuracy"])
    summary = {"seconds": time.perf_counter() - started, "summary": summary_rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
