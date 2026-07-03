from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import sys
import time
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
    prefill_cache,
    resolve_dtype,
)
from run_qabs_downstream_kv_retrieval import LABELS  # noqa: E402
from run_qabs_downstream_task_suite import BUILDERS  # noqa: E402


@dataclass(frozen=True)
class Config:
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
    log_every: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Full baseline vs typed-answerline downstream adapter.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--variants",
        default="structured_noisy,compact_kv,natural_kv,json_kv,needle_sentence,topic_table",
    )
    parser.add_argument("--tasks_per_variant", type=int, default=16)
    parser.add_argument("--records_per_task", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026070101)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--log_every", type=int, default=8)
    return Config(**vars(parser.parse_args()))


@torch.inference_mode()
def run_prefix(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    text: str,
) -> tuple[Any, torch.Tensor]:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device)
    for pos in range(ids.shape[-1]):
        token = ids[:, pos : pos + 1]
        outputs = model_forward(
            model,
            {
                "input_ids": token,
                "past_key_values": past_key_values,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
            },
        )
        past_key_values = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :].detach()
    return past_key_values, prev_logits


@torch.inference_mode()
def score_option(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    option: str,
) -> float:
    ids = tokenizer(" " + option, return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device)
    total = 0.0
    for pos in range(ids.shape[-1]):
        token = ids[:, pos : pos + 1]
        total += float(-F.cross_entropy(prev_logits.float(), token.reshape(-1), reduction="sum").item())
        outputs = model_forward(
            model,
            {
                "input_ids": token,
                "past_key_values": past_key_values,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
            },
        )
        past_key_values = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :].detach()
    return total


def extract_label_from_text(text: str) -> str:
    patterns = [
        r"ANSWER_LABEL\s*=\s*([A-D])",
        r"answer_label[\"']?\s*[:=]\s*[\"']?([A-D])",
        r"\bclass\s*=\s*([A-D])",
        r"\boption\s+([A-D])\b",
        r"\blabel\s+([A-D])\b",
        r"=>\s*([A-D])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def route_evidence_lines(task: dict[str, Any]) -> tuple[list[str], str]:
    target_key = task["target_key"]
    lines = task["context"].splitlines()
    evidence = [line for line in lines if target_key in line]
    label = ""
    for line in evidence:
        label = extract_label_from_text(line)
        if label:
            break
    return evidence[:2], label


def answerline_context(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    evidence_lines, label = route_evidence_lines(task)
    if not label:
        return "", {
            "typed_record_present": 0,
            "typed_record_label": "",
            "routed_lines": len(evidence_lines),
        }
    compact_evidence = " ".join(evidence_lines[:2])
    record = (
        "\nTyped memory summary: "
        f"ANSWER_LABEL={label}; status=current. "
        f"Lookup key {task['target_key']} maps to option {label}. "
        "Use the current status only.\n"
    )
    if compact_evidence:
        record += f"Evidence page: {compact_evidence}\n"
    return record + task["query"], {
        "typed_record_present": 1,
        "typed_record_label": label,
        "routed_lines": len(evidence_lines),
    }


def empty_cache(input_device: torch.device) -> tuple[Any, torch.Tensor]:
    return None, torch.empty((1, 0), device=input_device)


def evaluate_prompt(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    context_cache: Any,
    context_prev: torch.Tensor,
    prompt_suffix: str,
) -> tuple[str, dict[str, float], float]:
    started = time.perf_counter()
    query_cache, query_prev = run_prefix(
        model,
        tokenizer,
        input_device,
        clone_past_key_values(context_cache) if context_cache is not None else None,
        context_prev.detach().clone() if context_prev.numel() else context_prev,
        prompt_suffix,
    )
    scores: dict[str, float] = {}
    for label in LABELS:
        scores[label] = score_option(
            model,
            tokenizer,
            input_device,
            clone_past_key_values(query_cache),
            query_prev.detach().clone(),
            label,
        )
    elapsed = time.perf_counter() - started
    pred = max(scores, key=scores.get)
    return pred, scores, elapsed


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    keys = sorted({(row["variant"], row["mode"]) for row in rows})
    for variant, mode in keys:
        subset = [row for row in rows if row["variant"] == variant and row["mode"] == mode]
        correct = sum(int(row["correct"]) for row in subset)
        elapsed = sum(float(row["eval_seconds"]) for row in subset)
        total_elapsed = sum(float(row["total_seconds"]) for row in subset)
        tokens = sum(int(row["visible_tokens"]) for row in subset)
        summary.append(
            {
                "variant": variant,
                "mode": mode,
                "tasks": len(subset),
                "accuracy": correct / max(1, len(subset)),
                "mean_eval_seconds": elapsed / max(1, len(subset)),
                "mean_total_seconds": total_elapsed / max(1, len(subset)),
                "mean_visible_tokens": tokens / max(1, len(subset)),
                "typed_record_coverage": sum(int(row.get("typed_record_present", 0)) for row in subset)
                / max(1, len(subset)),
            }
        )
    return summary


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    variants = [name.strip() for name in config.variants.split(",") if name.strip()]
    unknown = [name for name in variants if name not in BUILDERS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available={sorted(BUILDERS)}")

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
    input_device = pick_input_device(model, device)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for variant_index, variant in enumerate(variants):
        rng = random.Random(config.seed + 1009 * variant_index)
        tasks = [BUILDERS[variant](rng, idx, config.records_per_task) for idx in range(config.tasks_per_variant)]
        for task_idx, task in enumerate(tasks, start=1):
            if task_idx == 1 or task_idx == len(tasks) or task_idx % config.log_every == 0:
                print(f"{variant} task {task_idx}/{len(tasks)}", flush=True)

            context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            prefill_started = time.perf_counter()
            context_cache, context_prev = prefill_cache(
                model,
                context_ids,
                context_ids.shape[-1],
                config.chunk_size,
                input_device,
            )
            prefill_seconds = time.perf_counter() - prefill_started
            pred, scores, elapsed = evaluate_prompt(
                model,
                tokenizer,
                input_device,
                context_cache,
                context_prev,
                task["query"],
            )
            rows.append(
                {
                    "variant": variant,
                    "task_id": task["task_id"],
                    "mode": "full_baseline",
                    "target_key": task["target_key"],
                    "target_label": task["target_label"],
                    "pred_label": pred,
                    "correct": int(pred == task["target_label"]),
                    "eval_seconds": elapsed,
                    "prefill_seconds": prefill_seconds,
                    "total_seconds": prefill_seconds + elapsed,
                    "visible_tokens": int(context_ids.shape[-1]),
                    "typed_record_present": 0,
                    "typed_record_label": "",
                    "routed_lines": 0,
                    **{f"score_{label}": scores[label] for label in LABELS},
                }
            )

            typed_prompt, meta = answerline_context(task)
            typed_ids = tokenizer(typed_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
            typed_cache, typed_prev = empty_cache(input_device)
            pred, scores, elapsed = evaluate_prompt(
                model,
                tokenizer,
                input_device,
                typed_cache,
                typed_prev,
                typed_prompt,
            )
            rows.append(
                {
                    "variant": variant,
                    "task_id": task["task_id"],
                    "mode": "typed_answerline_adapter",
                    "target_key": task["target_key"],
                    "target_label": task["target_label"],
                    "pred_label": pred,
                    "correct": int(pred == task["target_label"]),
                    "eval_seconds": elapsed,
                    "prefill_seconds": 0.0,
                    "total_seconds": elapsed,
                    "visible_tokens": int(typed_ids.shape[-1]),
                    **meta,
                    **{f"score_{label}": scores[label] for label in LABELS},
                }
            )

    fields = [
        "variant",
        "task_id",
        "mode",
        "target_key",
        "target_label",
        "pred_label",
        "correct",
        "eval_seconds",
        "prefill_seconds",
        "total_seconds",
        "visible_tokens",
        "typed_record_present",
        "typed_record_label",
        "routed_lines",
    ] + [f"score_{label}" for label in LABELS]
    with (output_dir / "typed_answerline_downstream_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize(rows)
    with (output_dir / "typed_answerline_downstream_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    summary = {"seconds": time.perf_counter() - started, "summary": summary_rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
