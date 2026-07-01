from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import re
import sys
import time
from contextlib import contextmanager
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
    model_forward,
    pick_input_device,
    resolve_dtype,
)
from run_qabs_downstream_kv_retrieval import LABELS, write_csv  # noqa: E402
from run_qabs_downstream_task_suite import BUILDERS  # noqa: E402


_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_MODE = "baseline"
_ACTIVE_TOP_FRACTION = 0.02


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate lowrank-dot sparse attention on topic PPL and synthetic downstream tasks."
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--topic_text_dir", default="data/topic_texts")
    parser.add_argument("--topics", default="finance,history,literature,science,software,mixed_qa")
    parser.add_argument("--ppl_prefill_tokens", type=int, default=2048)
    parser.add_argument("--ppl_eval_tokens", type=int, default=128)
    parser.add_argument("--max_chars", type=int, default=2_000_000)
    parser.add_argument("--downstream_variants", default="compact_kv,json_kv,needle_sentence,topic_table")
    parser.add_argument("--downstream_tasks_per_variant", type=int, default=16)
    parser.add_argument("--records_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026063004)
    parser.add_argument("--modes", default="baseline,lrsvd32attn,lrsvd64attn")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--eval_chunk_size", type=int, default=1)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--log_every", type=int, default=4)
    return parser.parse_args()


def parse_lrsvd_rank(mode: str) -> int | None:
    match = re.fullmatch(r"lrsvd(\d+)attn", mode)
    return int(match.group(1)) if match else None


def install_qwen3_attention_patch() -> None:
    global _ORIGINAL_EAGER_ATTENTION_FORWARD
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except Exception as exc:
        raise RuntimeError("Could not import transformers.models.qwen3.modeling_qwen3.") from exc
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
        setattr(modeling_qwen3, "eager_attention_forward", _lowrank_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _lowrank_eager_attention_forward


@contextmanager
def attention_mode(mode: str, top_fraction: float):
    global _ACTIVE_MODE, _ACTIVE_TOP_FRACTION
    previous_mode = _ACTIVE_MODE
    previous_top_fraction = _ACTIVE_TOP_FRACTION
    _ACTIVE_MODE = mode
    _ACTIVE_TOP_FRACTION = top_fraction
    try:
        yield
    finally:
        _ACTIVE_MODE = previous_mode
        _ACTIVE_TOP_FRACTION = previous_top_fraction


def _dense_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=True)
    attention_output = torch.matmul(attention_weights, value_states)
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, attention_weights


def _lowrank_sparse_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    rank: int,
) -> tuple[torch.Tensor, None]:
    if query_states.shape[-2] != 1:
        raise RuntimeError("lrsvd attention requires token-by-token eval; set --eval_chunk_size 1.")
    batch_count, head_count, _, head_dim = query_states.shape
    key_count = key_states.shape[-2]
    if key_count <= 1:
        output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return output, None

    history_count = key_count - 1
    top_count = min(history_count, max(1, math.ceil(_ACTIVE_TOP_FRACTION * history_count)))
    final_count = top_count + 1
    outputs = []
    for batch in range(batch_count):
        per_head = []
        for head in range(head_count):
            history_k = key_states[batch, head, :history_count, :].float()
            query = query_states[batch, head, 0, :].float()
            usable_rank = min(rank, history_k.shape[0], history_k.shape[1])
            centered = history_k - history_k.mean(dim=0, keepdim=True)
            try:
                _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                basis = vh[:usable_rank].transpose(0, 1)
                q_proj = query @ basis
                k_proj = history_k @ basis
                lowrank_scores = (k_proj * q_proj.unsqueeze(0)).sum(dim=-1)
            except RuntimeError:
                lowrank_scores = torch.matmul(history_k, query)
            selected_history = torch.topk(lowrank_scores, k=top_count, largest=True).indices
            selected = torch.cat(
                [
                    selected_history,
                    torch.tensor([history_count], dtype=torch.long, device=selected_history.device),
                ],
                dim=0,
            )
            selected_k = key_states[batch, head, selected, :].float()
            selected_v = value_states[batch, head, selected, :].float()
            full_scores = torch.matmul(selected_k, query).mul(scaling)
            if attention_mask is not None:
                mask_row = attention_mask[batch, 0, 0, selected].float()
                full_scores = full_scores + mask_row
            weights = F.softmax(full_scores, dim=-1, dtype=torch.float32).view(1, final_count)
            per_head.append(torch.matmul(weights, selected_v).to(query_states.dtype))
        outputs.append(torch.stack(per_head, dim=0))
    attention_output = torch.stack(outputs, dim=0).transpose(1, 2).contiguous()
    return attention_output, None


def _lowrank_eager_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if scaling is None:
        scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(query_states.shape[-1])))
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states = key_states.repeat_interleave(repeat_groups, dim=1)
        value_states = value_states.repeat_interleave(repeat_groups, dim=1)
    rank = parse_lrsvd_rank(_ACTIVE_MODE)
    if rank is None:
        return _dense_attention(query_states, key_states, value_states, attention_mask, scaling, dropout, module.training)
    return _lowrank_sparse_attention(query_states, key_states, value_states, attention_mask, scaling, rank)


def clone_past_key_values(past_key_values: Any) -> Any:
    if past_key_values is None:
        return None
    if torch.is_tensor(past_key_values):
        return past_key_values.detach().clone()
    if isinstance(past_key_values, tuple):
        return tuple(clone_past_key_values(item) for item in past_key_values)
    if isinstance(past_key_values, list):
        return [clone_past_key_values(item) for item in past_key_values]
    if isinstance(past_key_values, dict):
        return {key: clone_past_key_values(value) for key, value in past_key_values.items()}
    to_legacy_cache = getattr(past_key_values, "to_legacy_cache", None)
    from_legacy_cache = getattr(type(past_key_values), "from_legacy_cache", None)
    if callable(to_legacy_cache) and callable(from_legacy_cache):
        legacy = clone_past_key_values(to_legacy_cache())
        return from_legacy_cache(legacy)
    return copy.deepcopy(past_key_values)


@torch.inference_mode()
def run_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    start: int,
    end: int,
    chunk_size: int,
    input_device: torch.device,
    past_key_values: Any | None = None,
) -> tuple[Any, torch.Tensor]:
    prev_logits = None
    for pos in range(start, end, chunk_size):
        chunk_end = min(pos + chunk_size, end)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids[:, pos:chunk_end].to(input_device),
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(pos, chunk_end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :].detach()
        del outputs
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    assert prev_logits is not None
    return past_key_values, prev_logits


@torch.inference_mode()
def eval_ppl_mode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    mode: str,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    eval_chunk_size: int,
    input_device: torch.device,
    top_fraction: float,
) -> dict[str, Any]:
    with attention_mode("baseline", top_fraction):
        past, prev_logits = run_tokens(model, input_ids, 0, prefill_tokens, chunk_size, input_device)
    nll = 0.0
    tokens = 0
    started = time.perf_counter()
    with attention_mode(mode, top_fraction):
        end = prefill_tokens + eval_tokens
        for pos in range(prefill_tokens, end, eval_chunk_size):
            chunk_end = min(pos + eval_chunk_size, end)
            targets = input_ids[:, pos:chunk_end].to(input_device)
            for local in range(targets.shape[-1]):
                target = targets[:, local].reshape(-1)
                nll += float(F.cross_entropy(prev_logits.float(), target, reduction="sum").item())
                tokens += int(target.numel())
                token_input = input_ids[:, pos + local : pos + local + 1].to(input_device)
                kwargs = {
                    "input_ids": token_input,
                    "past_key_values": past,
                    "use_cache": True,
                    "return_dict": True,
                    "output_attentions": False,
                    "output_hidden_states": False,
                }
                outputs = model_forward(model, kwargs)
                past = outputs.past_key_values
                prev_logits = outputs.logits[:, -1, :].detach()
                del outputs
    seconds = time.perf_counter() - started
    return {
        "mode": mode,
        "eval_tokens": tokens,
        "nll": nll,
        "ppl": math.exp(nll / max(1, tokens)),
        "seconds": seconds,
        "tokens_per_second": tokens / seconds if seconds > 0.0 else 0.0,
    }


def read_topic_text(path: Path, max_chars: int) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]


def resolve_topic_path(topic_text_dir: Path, topic: str) -> Path:
    candidates = [topic_text_dir / f"{topic}.txt", topic_text_dir / topic]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"topic text not found for {topic!r} under {topic_text_dir}")


def run_ppl_suite(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    output_dir: Path,
    modes: list[str],
) -> list[dict[str, Any]]:
    topic_dir = Path(args.topic_text_dir)
    topics = [topic.strip() for topic in args.topics.split(",") if topic.strip()]
    rows: list[dict[str, Any]] = []
    for topic in topics:
        path = resolve_topic_path(topic_dir, topic)
        text = read_topic_text(path, args.max_chars)
        token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
        need = args.ppl_prefill_tokens + args.ppl_eval_tokens
        if len(token_ids) < need:
            print(f"skip topic {topic}: need {need} tokens, got {len(token_ids)}", flush=True)
            continue
        input_ids = torch.tensor(token_ids[:need], dtype=torch.long).view(1, -1)
        baseline_row: dict[str, Any] | None = None
        for mode in modes:
            print(f"ppl topic={topic} mode={mode}", flush=True)
            result = eval_ppl_mode(
                model,
                input_ids,
                mode,
                args.ppl_prefill_tokens,
                args.ppl_eval_tokens,
                args.chunk_size,
                args.eval_chunk_size,
                input_device,
                args.top_fraction,
            )
            row = {
                "topic": topic,
                "mode": mode,
                "prefill_tokens": args.ppl_prefill_tokens,
                **result,
            }
            if mode == "baseline":
                baseline_row = row
                row["ppl_ratio_vs_baseline"] = 1.0
                row["time_ratio_vs_baseline"] = 1.0
            elif baseline_row is not None:
                row["ppl_ratio_vs_baseline"] = row["ppl"] / baseline_row["ppl"]
                row["time_ratio_vs_baseline"] = row["seconds"] / baseline_row["seconds"]
            else:
                row["ppl_ratio_vs_baseline"] = ""
                row["time_ratio_vs_baseline"] = ""
            rows.append(row)
    fields = [
        "topic",
        "mode",
        "prefill_tokens",
        "eval_tokens",
        "nll",
        "ppl",
        "seconds",
        "tokens_per_second",
        "ppl_ratio_vs_baseline",
        "time_ratio_vs_baseline",
    ]
    write_csv(output_dir / "topic_ppl_results.csv", rows, fields)
    return rows


@torch.inference_mode()
def run_text_prefix(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    text: str,
    mode: str,
    top_fraction: float,
) -> tuple[Any, torch.Tensor]:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device)
    with attention_mode(mode, top_fraction):
        for pos in range(ids.shape[-1]):
            kwargs = {
                "input_ids": ids[:, pos : pos + 1],
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
    mode: str,
    top_fraction: float,
) -> float:
    ids = tokenizer(" " + option, return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device)
    total = 0.0
    with attention_mode(mode, top_fraction):
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


def run_downstream_suite(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    output_dir: Path,
    modes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants = [name.strip() for name in args.downstream_variants.split(",") if name.strip()]
    unknown = [name for name in variants if name not in BUILDERS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available={sorted(BUILDERS)}")
    result_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(variants):
        rng = random.Random(args.seed + 1009 * variant_index)
        tasks = [BUILDERS[variant](rng, idx, args.records_per_task) for idx in range(args.downstream_tasks_per_variant)]
        for task_index, task in enumerate(tasks, start=1):
            if task_index == 1 or task_index == len(tasks) or task_index % args.log_every == 0:
                print(f"downstream {variant} task {task_index}/{len(tasks)}", flush=True)
            context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            with attention_mode("baseline", args.top_fraction):
                context_cache, context_prev = run_tokens(
                    model,
                    context_ids,
                    0,
                    int(context_ids.shape[-1]),
                    args.chunk_size,
                    input_device,
                )
            for mode in modes:
                query_cache, query_prev = run_text_prefix(
                    model,
                    tokenizer,
                    input_device,
                    clone_past_key_values(context_cache),
                    context_prev.detach().clone(),
                    task["query"],
                    mode,
                    args.top_fraction,
                )
                scores = {
                    label: score_option(
                        model,
                        tokenizer,
                        input_device,
                        clone_past_key_values(query_cache),
                        query_prev.detach().clone(),
                        label,
                        mode,
                        args.top_fraction,
                    )
                    for label in LABELS
                }
                pred = max(scores, key=scores.get)
                result_rows.append(
                    {
                        "variant": variant,
                        "task_id": task["task_id"],
                        "mode": mode,
                        "target_key": task["target_key"],
                        "target_index": task["target_index"],
                        "target_label": task["target_label"],
                        "pred_label": pred,
                        "correct": int(pred == task["target_label"]),
                        **{f"score_{label}": scores[label] for label in LABELS},
                    }
                )
    result_fields = [
        "variant",
        "task_id",
        "mode",
        "target_key",
        "target_index",
        "target_label",
        "pred_label",
        "correct",
    ] + [f"score_{label}" for label in LABELS]
    write_csv(output_dir / "downstream_results.csv", result_rows, result_fields)
    for variant in variants:
        for mode in modes:
            subset = [row for row in result_rows if row["variant"] == variant and row["mode"] == mode]
            correct = sum(int(row["correct"]) for row in subset)
            baseline_subset = [row for row in result_rows if row["variant"] == variant and row["mode"] == "baseline"]
            baseline_correct = sum(int(row["correct"]) for row in baseline_subset)
            summary_rows.append(
                {
                    "variant": variant,
                    "mode": mode,
                    "correct": correct,
                    "total": len(subset),
                    "accuracy": correct / max(1, len(subset)),
                    "baseline_accuracy": baseline_correct / max(1, len(baseline_subset)) if mode != "baseline" else "",
                    "delta_vs_baseline": (correct / max(1, len(subset)))
                    - (baseline_correct / max(1, len(baseline_subset)))
                    if mode != "baseline"
                    else "",
                }
            )
    write_csv(
        output_dir / "downstream_summary_by_variant_mode.csv",
        summary_rows,
        ["variant", "mode", "correct", "total", "accuracy", "baseline_accuracy", "delta_vs_baseline"],
    )
    return result_rows, summary_rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if "baseline" not in modes:
        modes.insert(0, "baseline")
    invalid = [mode for mode in modes if mode != "baseline" and parse_lrsvd_rank(mode) is None]
    if invalid:
        raise ValueError(f"invalid modes: {invalid}; expected baseline or lrsvd{{rank}}attn")

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    install_qwen3_attention_patch()
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, requested_device)

    started = time.perf_counter()
    ppl_rows = run_ppl_suite(args, model, tokenizer, input_device, output_dir, modes)
    downstream_rows, downstream_summary = run_downstream_suite(args, model, tokenizer, input_device, output_dir, modes)
    summary = {
        "seconds": time.perf_counter() - started,
        "ppl_rows": len(ppl_rows),
        "downstream_rows": len(downstream_rows),
        "downstream_summary": downstream_summary,
        "paths": {
            "topic_ppl_results": str(output_dir / "topic_ppl_results.csv"),
            "downstream_results": str(output_dir / "downstream_results.csv"),
            "downstream_summary_by_variant_mode": str(output_dir / "downstream_summary_by_variant_mode.csv"),
        },
        "notes": {
            "lrsvd_attention": (
                "Prototype sparse attention: compute SVD on historical K per layer/head/query, "
                "rank-r lowrank q-k scores select top_fraction history tokens, always keep self, "
                "then softmax over selected tokens with exact full scores."
            ),
            "speed": "Measured wall time is Python/SVD prototype eval time, not fused-kernel speed.",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
