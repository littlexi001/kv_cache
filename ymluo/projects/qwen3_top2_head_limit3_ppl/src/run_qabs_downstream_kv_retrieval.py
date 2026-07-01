from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
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
    EvidenceSpanCoverageStats,
    QabsReuseProfileStats,
    ReuseCandidateState,
    attention_mode,
    clone_past_key_values,
    evidence_span_coverage,
    install_qwen3_attention_patch,
    model_forward,
    pick_input_device,
    prefill_cache,
    restore_layer_budget_qabs_reuse_state,
    resolve_dtype,
    snapshot_layer_budget_qabs_reuse_state,
)


LABELS = ["A", "B", "C", "D"]
TOPICS = [
    "firmware audit",
    "tax dispute",
    "satellite fault",
    "clinical review",
    "grid auction",
    "compiler report",
]


@dataclass(frozen=True)
class Config:
    model_name_or_path: str
    output_dir: str
    tasks: int
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
    layer_budget_map_path: str
    log_every: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Downstream KV retrieval eval for QABS compression.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tasks", type=int, default=48)
    parser.add_argument("--records_per_task", type=int, default=160)
    parser.add_argument("--seed", type=int, default=202606294)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.05)
    parser.add_argument("--protect_sink_tokens", type=int, default=10)
    parser.add_argument("--protect_recent_tokens", type=int, default=10)
    parser.add_argument("--modes", default="baseline,qabs8cand5reuse")
    parser.add_argument("--layer_budget_map_path", default="")
    parser.add_argument("--log_every", type=int, default=8)
    return Config(**vars(parser.parse_args()))


def make_key(rng: random.Random, prefix: str) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return prefix + "-" + "".join(rng.choice(alphabet) for _ in range(7))


def build_task(rng: random.Random, task_idx: int, records_per_task: int) -> dict[str, Any]:
    records = []
    target_index = rng.randrange(records_per_task)
    for idx in range(records_per_task):
        key = make_key(rng, f"K{task_idx:03d}{idx:03d}")
        label = rng.choice(LABELS)
        topic = TOPICS[(task_idx + idx) % len(TOPICS)]
        value = rng.uniform(-100.0, 500.0)
        checksum = make_key(rng, "CHK")
        records.append(
            {
                "key": key,
                "label": label,
                "line": (
                    f"RECORD key={key}; topic={topic}; metric={value:.5f}; "
                    f"checksum={checksum}; ANSWER_LABEL={label}; note=preserve exact label for lookup."
                ),
            }
        )
    target = records[target_index]
    context = "\n".join(record["line"] for record in records) + "\n"
    query = (
        f"\nLookup task: find the record with key={target['key']}. "
        "Return only the single ANSWER_LABEL letter. ANSWER_LABEL:"
    )
    return {
        "task_id": task_idx,
        "target_key": target["key"],
        "target_label": target["label"],
        "target_index": target_index,
        "context": context,
        "query": query,
    }


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
        chunk = ids[:, pos : pos + 1]
        kwargs = {
            "input_ids": chunk,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
        }
        outputs = model_forward(model, kwargs)
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
        kwargs = {
            "input_ids": token,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
        }
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :].detach()
    return total


def eval_task(
    model: torch.nn.Module,
    tokenizer: Any,
    task: dict[str, Any],
    config: Config,
    input_device: torch.device,
    mode: str,
    context_cache: Any,
    context_prev: torch.Tensor,
    qabs_profile_stats: QabsReuseProfileStats | None = None,
    evidence_coverage_stats: EvidenceSpanCoverageStats | None = None,
    evidence_spans: dict[str, tuple[int, int]] | None = None,
    force_evidence_spans: bool = False,
) -> dict[str, Any]:
    reuse_state = ReuseCandidateState() if mode != "baseline" else None
    with evidence_span_coverage(evidence_coverage_stats, evidence_spans or {}, force_evidence_spans), attention_mode(
            mode=mode,
            top_fraction=config.top_fraction,
            max_heads_per_token=3,
            always_keep_self=True,
            protect_sink_tokens=config.protect_sink_tokens,
            protect_recent_tokens=config.protect_recent_tokens,
            load_stats=None,
            reuse_state=reuse_state,
            qabs_fast_path=True,
            qabs_cuda_final_kernel=True,
            qabs_cuda_candidate_kernel=True,
            qabs_cuda_reuse_select_kernel=False,
            qabs_profile_stats=qabs_profile_stats,
            layer_budget_map_path=config.layer_budget_map_path,
        ):
        query_cache, query_prev = run_prefix(
            model,
            tokenizer,
            input_device,
            clone_past_key_values(context_cache),
            context_prev.detach().clone(),
            task["query"],
        )
        state_after_query = copy.deepcopy(reuse_state) if reuse_state is not None else None
        layer_budget_state_after_query = (
            snapshot_layer_budget_qabs_reuse_state() if mode == "layerbudgetattn" else None
        )
        scores: dict[str, float] = {}
        for label in LABELS:
            if reuse_state is not None:
                reuse_state.previous_layer = copy.deepcopy(state_after_query.previous_layer)
                reuse_state.refresh_head_count = state_after_query.refresh_head_count
                reuse_state.refresh_case_count = state_after_query.refresh_case_count
            if layer_budget_state_after_query is not None:
                restore_layer_budget_qabs_reuse_state(layer_budget_state_after_query)
            scores[label] = score_option(
                model,
                tokenizer,
                input_device,
                clone_past_key_values(query_cache),
                query_prev.detach().clone(),
                label,
            )
    pred = max(scores, key=scores.get)
    return {
        "task_id": task["task_id"],
        "mode": mode,
        "target_key": task["target_key"],
        "target_index": task["target_index"],
        "target_label": task["target_label"],
        "pred_label": pred,
        "correct": int(pred == task["target_label"]),
        **{f"score_{label}": scores[label] for label in LABELS},
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    rng = random.Random(config.seed)
    tasks = [build_task(rng, idx, config.records_per_task) for idx in range(config.tasks)]
    with (output_dir / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            public = {key: value for key, value in task.items() if key not in {"context", "query"}}
            handle.write(json.dumps(public, ensure_ascii=False) + "\n")

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

    modes = [mode.strip() for mode in config.modes.split(",") if mode.strip()]
    if "baseline" not in modes:
        modes.insert(0, "baseline")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for idx, task in enumerate(tasks, start=1):
        if idx == 1 or idx == len(tasks) or idx % config.log_every == 0:
            print(f"task {idx}/{len(tasks)}", flush=True)
        context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
        context_cache, context_prev = prefill_cache(
            model,
            context_ids,
            context_ids.shape[-1],
            config.chunk_size,
            input_device,
        )
        for mode in modes:
            rows.append(eval_task(model, tokenizer, task, config, input_device, mode, context_cache, context_prev))
    seconds = time.perf_counter() - started
    fields = ["task_id", "mode", "target_key", "target_index", "target_label", "pred_label", "correct"] + [
        f"score_{label}" for label in LABELS
    ]
    write_csv(output_dir / "downstream_results.csv", rows, fields)
    summary = {
        "config": asdict(config),
        "seconds": seconds,
        "accuracy": {
            mode: sum(row["correct"] for row in rows if row["mode"] == mode)
            / max(1, sum(1 for row in rows if row["mode"] == mode))
            for mode in modes
        },
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["accuracy"], indent=2), flush=True)


if __name__ == "__main__":
    main()
