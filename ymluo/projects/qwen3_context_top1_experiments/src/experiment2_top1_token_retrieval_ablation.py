from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "/mnt/workspace/Qwen3-0.6B"
DEFAULT_DATA_PATH = (
    "ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl"
)

_ACTIVE_PRUNE_RATIO: float | None = None
_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate full attention vs per-layer/per-head top-ratio attention pruning, "
            "and dump the selected top tokens for inspection."
        )
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", default="outputs/experiment2_attention_top1_pruning")
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--top_ratio", type=float, default=0.01)
    parser.add_argument(
        "--dump_query_scope",
        choices=["answer", "prompt_last", "all"],
        default="answer",
        help=(
            "Which query rows to dump. 'answer' dumps rows that predict/consume the gold answer tokens; "
            "'prompt_last' dumps the last N prompt rows; 'all' dumps every row and can be very large."
        ),
    )
    parser.add_argument(
        "--dump_query_last_tokens",
        type=int,
        default=16,
        help="Used only when --dump_query_scope prompt_last.",
    )
    parser.add_argument(
        "--simple_tail_tokens",
        type=int,
        default=10,
        help="Write a compact file for the last N prompt tokens and their selected top-ratio key tokens.",
    )
    parser.add_argument("--max_context_chars", type=int, default=24000)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--trust_remote_code", type=str2bool, default=True)
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def pick_input_device(model: torch.nn.Module, fallback_device: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback_device


def load_samples(path: Path, max_samples: int, max_context_chars: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            context = str(item.get("context", ""))
            if max_context_chars > 0:
                context = context[:max_context_chars]
            samples.append(
                {
                    "sample_id": item.get("sample_id", f"sample_{len(samples)}"),
                    "context": context,
                    "question": item.get("question", ""),
                    "answer": item.get("expected_answer", item.get("answer", "")),
                    "needle_text": item.get("needle_text", ""),
                }
            )
            if len(samples) >= max_samples:
                break
    return samples


def build_prompt(sample: dict[str, Any]) -> str:
    return (
        "Use the context to answer the question. Answer with the exact phrase when possible.\n\n"
        f"Context:\n{sample['context']}\n\n"
        f"Question: {sample['question']}\n"
        "Answer:"
    )


def token_offsets(tokenizer: Any, prompt: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return encoded.input_ids[0].tolist(), [(int(a), int(b)) for a, b in encoded.offset_mapping[0].tolist()]


def prompt_answer_token_offsets(
    tokenizer: Any,
    prompt: str,
    answer: str,
) -> tuple[str, list[int], list[tuple[int, int]], int]:
    answer_text = " " + answer
    eval_text = prompt + answer_text
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    token_ids, offsets = token_offsets(tokenizer, eval_text)
    return eval_text, token_ids, offsets, len(prompt_ids)


def compact_piece(text: str) -> str:
    return text.replace("\n", "\\n").replace("\t", "\\t")


def classify_token(
    token_index: int,
    offsets: list[tuple[int, int]],
    prompt: str,
    sample: dict[str, Any],
    total_tokens: int,
) -> str:
    start, end = offsets[token_index]
    if token_index < 4:
        return "sink_prefix"
    if token_index >= max(0, total_tokens - 24):
        return "query_or_recent"
    piece = prompt[start:end].lower()
    answer = str(sample["answer"]).lower()
    needle = str(sample["needle_text"]).lower()
    question = str(sample["question"]).lower()
    if piece.strip() and piece in answer:
        return "answer_evidence"
    if piece.strip() and piece in needle:
        return "needle_context"
    if piece.strip() and piece in question:
        return "query_bridge"
    if any(ch.isdigit() for ch in piece):
        return "rare_or_number"
    if any(ch in piece for ch in ["#", "{", "}", "_", "@", "/", "\\", "="]):
        return "rare_or_symbol"
    if not piece.strip():
        return "whitespace"
    return "other_context"


def _pruned_eager_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = _ACTIVE_PRUNE_RATIO
    if scaling is None:
        scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(query_states.shape[-1])))
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states = key_states.repeat_interleave(repeat_groups, dim=1)
        value_states = value_states.repeat_interleave(repeat_groups, dim=1)
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    if ratio is not None:
        keep = torch.zeros_like(scores, dtype=torch.bool)
        query_count = scores.shape[-2]
        for query_index in range(query_count):
            row = scores[:, :, query_index, :]
            valid_count = int(torch.isfinite(row[0, 0]).sum().item())
            keep_count = min(valid_count, max(1, math.ceil(ratio * valid_count))) if valid_count > 0 else 1
            _, top_indices = torch.topk(row, k=keep_count, dim=-1, largest=True)
            keep[:, :, query_index, :].scatter_(-1, top_indices, True)
        scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=True)
    attention_output = torch.matmul(attention_weights, value_states)
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, attention_weights


def install_qwen3_attention_patch() -> None:
    global _ORIGINAL_EAGER_ATTENTION_FORWARD
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except Exception as exc:
        raise RuntimeError("Could not import transformers.models.qwen3.modeling_qwen3.") from exc
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
        setattr(modeling_qwen3, "eager_attention_forward", _pruned_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _pruned_eager_attention_forward


@contextmanager
def pruning_ratio(ratio: float | None):
    global _ACTIVE_PRUNE_RATIO
    previous = _ACTIVE_PRUNE_RATIO
    _ACTIVE_PRUNE_RATIO = ratio
    try:
        yield
    finally:
        _ACTIVE_PRUNE_RATIO = previous


@torch.inference_mode()
def answer_nll(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    answer: str,
    input_device: torch.device,
    prune_ratio: float | None,
) -> dict[str, float]:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    answer_ids = tokenizer(" " + answer, return_tensors="pt", add_special_tokens=False).input_ids
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(input_device)
    with pruning_ratio(prune_ratio):
        outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    answer_start = prompt_ids.shape[1] - 1
    answer_logits = logits[:, answer_start : answer_start + answer_ids.shape[1], :]
    answer_labels = labels[:, answer_start : answer_start + answer_ids.shape[1]]
    losses = F.cross_entropy(
        answer_logits.reshape(-1, answer_logits.shape[-1]),
        answer_labels.reshape(-1),
        reduction="none",
    )
    loss = float(losses.mean())
    return {
        "loss": loss,
        "ppl": float(math.exp(min(loss, 80.0))),
        "prompt_token_count": int(prompt_ids.shape[1]),
        "answer_token_count": int(answer_ids.shape[1]),
    }


@torch.inference_mode()
def collect_full_attention_top_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    top_ratio: float,
    query_indices: list[int],
    input_device: torch.device,
) -> dict[int, dict[int, list[tuple[int, int, float]]]]:
    input_ids = input_ids.to(input_device)
    with pruning_ratio(None):
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("Model did not return attentions. Use --attn_implementation eager.")
    total_tokens = int(input_ids.shape[1])
    layer_head_rows: dict[int, dict[int, list[tuple[int, int, float]]]] = {}
    for layer_idx, attn in enumerate(attentions):
        # attn: [batch, heads, query, key], post-softmax full attention.
        attn = attn[0].float().cpu()
        head_rows: dict[int, list[tuple[int, int, float]]] = {}
        for head_idx in range(attn.shape[0]):
            rows: list[tuple[int, int, float]] = []
            for query_index in query_indices:
                if query_index < 0 or query_index >= total_tokens:
                    continue
                current = attn[head_idx, query_index, : query_index + 1]
                keep_count = max(1, math.ceil(top_ratio * current.numel()))
                values, indices = torch.topk(current, k=keep_count, largest=True)
                for key_index, value in zip(indices.tolist(), values.tolist()):
                    rows.append((query_index, int(key_index), float(value)))
            head_rows[head_idx] = rows
        layer_head_rows[layer_idx] = head_rows
    return layer_head_rows


def select_dump_query_indices(
    scope: str,
    prompt_token_count: int,
    total_token_count: int,
    dump_query_last_tokens: int,
) -> list[int]:
    if scope == "answer":
        # These are the rows for gold answer tokens in the prompt+answer forward.
        return list(range(prompt_token_count, total_token_count))
    if scope == "prompt_last":
        start = max(0, prompt_token_count - dump_query_last_tokens)
        return list(range(start, prompt_token_count))
    if scope == "all":
        return list(range(total_token_count))
    raise ValueError(f"Unsupported dump scope: {scope}")


def dump_top_token_rows(
    path: Path,
    tokenizer: Any,
    prompt: str,
    sample: dict[str, Any],
    token_ids: list[int],
    offsets: list[tuple[int, int]],
    layer_head_rows: dict[int, dict[int, list[tuple[int, int, float]]]],
) -> None:
    fields = [
        "sample_id",
        "layer",
        "head",
        "query_token_index",
        "query_token_text",
        "rank_within_query",
        "key_token_index",
        "attention_weight",
        "key_token_id",
        "key_token_text",
        "key_context_piece",
        "left_context",
        "right_context",
        "category",
    ]
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        total_tokens = len(token_ids)
        for layer, head_map in layer_head_rows.items():
            for head, rows in head_map.items():
                by_query: dict[int, list[tuple[int, float]]] = {}
                for query_index, key_index, score in rows:
                    by_query.setdefault(query_index, []).append((key_index, score))
                for query_index, pairs in by_query.items():
                    pairs = sorted(pairs, key=lambda item: item[1], reverse=True)
                    query_text = compact_piece(tokenizer.decode([token_ids[query_index]], skip_special_tokens=False))
                    for rank, (key_index, score) in enumerate(pairs, start=1):
                        start, end = offsets[key_index]
                        row = {
                            "sample_id": sample["sample_id"],
                            "layer": layer,
                            "head": head,
                            "query_token_index": query_index,
                            "query_token_text": query_text,
                            "rank_within_query": rank,
                            "key_token_index": key_index,
                            "attention_weight": score,
                            "key_token_id": token_ids[key_index],
                            "key_token_text": compact_piece(
                                tokenizer.decode([token_ids[key_index]], skip_special_tokens=False)
                            ),
                            "key_context_piece": compact_piece(prompt[start:end]),
                            "left_context": compact_piece(prompt[max(0, start - 40) : start]),
                            "right_context": compact_piece(prompt[end : min(len(prompt), end + 40)]),
                            "category": classify_token(key_index, offsets, prompt, sample, total_tokens),
                        }
                        writer.writerow(row)


def dump_simple_tail_top_tokens(
    path: Path,
    tokenizer: Any,
    prompt: str,
    sample: dict[str, Any],
    token_ids: list[int],
    offsets: list[tuple[int, int]],
    answer_token_indices: set[int],
    top_ratio: float,
    layer_head_rows: dict[int, dict[int, list[tuple[int, int, float]]]],
) -> list[dict[str, Any]]:
    fields = [
        "sample_id",
        "layer",
        "head",
        "query_token_index",
        "query_token_text",
        "selected_count",
        "selected_token_indices",
        "selected_attention_weights",
        "selected_token_texts",
        "answer_span_count",
        "front_1pct_count",
        "tail_1pct_count",
        "other_count",
        "answer_span_pct",
        "front_1pct_pct",
        "tail_1pct_pct",
        "other_pct",
    ]
    rows: list[dict[str, Any]] = []
    for layer, head_map in layer_head_rows.items():
        for head, triples in head_map.items():
            by_query: dict[int, list[tuple[int, float]]] = {}
            for query_index, key_index, score in triples:
                by_query.setdefault(query_index, []).append((key_index, score))
            for query_index, pairs in by_query.items():
                pairs = sorted(pairs, key=lambda item: item[1], reverse=True)
                visible_count = query_index + 1
                edge_count = max(1, math.ceil(top_ratio * visible_count))
                category_counts = {
                    "answer_span": 0,
                    "front_1pct": 0,
                    "tail_1pct": 0,
                    "other": 0,
                }
                key_indices: list[int] = []
                weights: list[float] = []
                token_texts: list[str] = []
                for key_index, score in pairs:
                    key_indices.append(key_index)
                    weights.append(score)
                    token_texts.append(
                        compact_piece(tokenizer.decode([token_ids[key_index]], skip_special_tokens=False))
                    )
                    if key_index in answer_token_indices:
                        category_counts["answer_span"] += 1
                    elif key_index < edge_count:
                        category_counts["front_1pct"] += 1
                    elif key_index >= visible_count - edge_count:
                        category_counts["tail_1pct"] += 1
                    else:
                        category_counts["other"] += 1
                selected_count = len(pairs)
                denom = selected_count if selected_count else 1
                rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "layer": layer,
                        "head": head,
                        "query_token_index": query_index,
                        "query_token_text": compact_piece(
                            tokenizer.decode([token_ids[query_index]], skip_special_tokens=False)
                        ),
                        "selected_count": selected_count,
                        "selected_token_indices": " ".join(str(item) for item in key_indices),
                        "selected_attention_weights": " ".join(f"{item:.8g}" for item in weights),
                        "selected_token_texts": " ".join(token_texts),
                        "answer_span_count": category_counts["answer_span"],
                        "front_1pct_count": category_counts["front_1pct"],
                        "tail_1pct_count": category_counts["tail_1pct"],
                        "other_count": category_counts["other"],
                        "answer_span_pct": category_counts["answer_span"] / denom,
                        "front_1pct_pct": category_counts["front_1pct"] / denom,
                        "tail_1pct_pct": category_counts["tail_1pct"] / denom,
                        "other_pct": category_counts["other"] / denom,
                    }
                )

    rows.sort(
        key=lambda row: (
            row["sample_id"],
            int(row["layer"]),
            int(row["head"]),
            int(row["query_token_index"]),
        )
    )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
    return rows


def answer_span_token_indices(
    prompt: str,
    answer: str,
    offsets: list[tuple[int, int]],
    prompt_token_count: int,
) -> set[int]:
    indices: set[int] = set()
    if not answer:
        return indices
    start = 0
    while True:
        char_start = prompt.find(answer, start)
        if char_start < 0:
            break
        char_end = char_start + len(answer)
        for token_index, (token_start, token_end) in enumerate(offsets[:prompt_token_count]):
            if token_start < char_end and token_end > char_start:
                indices.add(token_index)
        start = char_end
    return indices


def write_tail_summaries(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_fields = [
        "scope",
        "sample_id",
        "layer",
        "head",
        "query_row_count",
        "selected_token_count",
        "answer_span_count",
        "front_1pct_count",
        "tail_1pct_count",
        "other_count",
        "answer_span_pct",
        "front_1pct_pct",
        "tail_1pct_pct",
        "other_pct",
    ]

    def build_summary(group_rows: list[dict[str, Any]], scope: str, sample_id: str = "", layer: str = "", head: str = "") -> dict[str, Any]:
        selected = sum(int(row["selected_count"]) for row in group_rows)
        answer = sum(int(row["answer_span_count"]) for row in group_rows)
        front = sum(int(row["front_1pct_count"]) for row in group_rows)
        tail = sum(int(row["tail_1pct_count"]) for row in group_rows)
        other = sum(int(row["other_count"]) for row in group_rows)
        denom = selected if selected else 1
        return {
            "scope": scope,
            "sample_id": sample_id,
            "layer": layer,
            "head": head,
            "query_row_count": len(group_rows),
            "selected_token_count": selected,
            "answer_span_count": answer,
            "front_1pct_count": front,
            "tail_1pct_count": tail,
            "other_count": other,
            "answer_span_pct": answer / denom,
            "front_1pct_pct": front / denom,
            "tail_1pct_pct": tail / denom,
            "other_pct": other / denom,
        }

    overall_rows = [build_summary(rows, "overall")]
    with (output_dir / "tail10_position_overall_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(overall_rows)

    sample_rows: list[dict[str, Any]] = []
    for sample_id in sorted({str(row["sample_id"]) for row in rows}):
        group = [row for row in rows if str(row["sample_id"]) == sample_id]
        sample_rows.append(build_summary(group, "sample", sample_id=sample_id))
    with (output_dir / "tail10_position_sample_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(sample_rows)

    layer_head_rows: list[dict[str, Any]] = []
    keys = sorted({(int(row["layer"]), int(row["head"])) for row in rows})
    for layer, head in keys:
        group = [row for row in rows if int(row["layer"]) == layer and int(row["head"]) == head]
        layer_head_rows.append(build_summary(group, "layer_head", layer=str(layer), head=str(head)))
    with (output_dir / "tail10_position_layer_head_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(layer_head_rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "sample_count",
        "mean_loss",
        "mean_ppl",
        "mean_delta_loss_vs_full_attention",
        "mean_prompt_token_count",
    ]
    modes = sorted({row["mode"] for row in rows})
    summary_rows: list[dict[str, Any]] = []
    full_by_sample = {row["sample_id"]: float(row["loss"]) for row in rows if row["mode"] == "full_attention"}
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        mean_loss = sum(float(row["loss"]) for row in mode_rows) / len(mode_rows)
        mean_delta = sum(float(row["loss"]) - full_by_sample[row["sample_id"]] for row in mode_rows) / len(mode_rows)
        summary_rows.append(
            {
                "mode": mode,
                "sample_count": len(mode_rows),
                "mean_loss": mean_loss,
                "mean_ppl": math.exp(min(mean_loss, 80.0)),
                "mean_delta_loss_vs_full_attention": mean_delta,
                "mean_prompt_token_count": sum(int(row["prompt_token_count"]) for row in mode_rows)
                / len(mode_rows),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    if args.top_ratio <= 0.0 or args.top_ratio > 1.0:
        raise ValueError("--top_ratio must be in (0, 1].")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    install_qwen3_attention_patch()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=args.device_map if requested_device.type != "cpu" else None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    if requested_device.type == "cpu":
        model.to(requested_device)
    model.eval()
    input_device = pick_input_device(model, requested_device)

    samples = load_samples(Path(args.data_path), args.max_samples, args.max_context_chars)
    result_rows: list[dict[str, Any]] = []
    per_sample_path = output_dir / "attention_pruning_results.jsonl"
    top_token_path = output_dir / "layer_head_top1_tokens.csv"
    simple_tail_path = output_dir / "tail10_top1_tokens_by_query.csv"
    simple_tail_sample_dir = output_dir / "tail10_by_sample"
    simple_tail_sample_dir.mkdir(parents=True, exist_ok=True)
    if top_token_path.exists():
        top_token_path.unlink()
    if simple_tail_path.exists():
        simple_tail_path.unlink()
    for old_sample_file in simple_tail_sample_dir.glob("*.csv"):
        old_sample_file.unlink()
    all_simple_tail_rows: list[dict[str, Any]] = []

    with per_sample_path.open("w", encoding="utf-8") as result_handle:
        for sample in samples:
            prompt = build_prompt(sample)
            eval_text, eval_token_ids, eval_offsets, prompt_token_count = prompt_answer_token_offsets(
                tokenizer, prompt, sample["answer"]
            )
            eval_input_ids = torch.tensor([eval_token_ids], dtype=torch.long)

            full_metrics = answer_nll(model, tokenizer, prompt, sample["answer"], input_device, None)
            pruned_metrics = answer_nll(model, tokenizer, prompt, sample["answer"], input_device, args.top_ratio)
            for mode, metrics in [("full_attention", full_metrics), ("attention_top1_pruned", pruned_metrics)]:
                row = {
                    "sample_id": sample["sample_id"],
                    "mode": mode,
                    "answer": sample["answer"],
                    "loss": metrics["loss"],
                    "ppl": metrics["ppl"],
                    "delta_loss_vs_full_attention": metrics["loss"] - full_metrics["loss"],
                    "prompt_token_count": metrics["prompt_token_count"],
                    "answer_token_count": metrics["answer_token_count"],
                    "top_ratio": args.top_ratio if mode == "attention_top1_pruned" else "",
                }
                result_rows.append(row)
                result_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                result_handle.flush()

            query_indices = select_dump_query_indices(
                args.dump_query_scope,
                prompt_token_count,
                len(eval_token_ids),
                args.dump_query_last_tokens,
            )
            layer_head_rows = collect_full_attention_top_tokens(
                model, eval_input_ids, args.top_ratio, query_indices, input_device
            )
            dump_top_token_rows(
                top_token_path,
                tokenizer,
                eval_text,
                sample,
                eval_token_ids,
                eval_offsets,
                layer_head_rows,
            )
            simple_tail_query_indices = select_dump_query_indices(
                "prompt_last",
                prompt_token_count,
                len(eval_token_ids),
                args.simple_tail_tokens,
            )
            simple_tail_rows = collect_full_attention_top_tokens(
                model, eval_input_ids, args.top_ratio, simple_tail_query_indices, input_device
            )
            answer_indices = answer_span_token_indices(
                prompt,
                sample["answer"],
                eval_offsets,
                prompt_token_count,
            )
            sample_simple_tail_path = simple_tail_sample_dir / f"{sample['sample_id']}_tail10_top1_tokens_by_query.csv"
            sample_tail_rows = dump_simple_tail_top_tokens(
                sample_simple_tail_path,
                tokenizer,
                eval_text,
                sample,
                eval_token_ids,
                eval_offsets,
                answer_indices,
                args.top_ratio,
                simple_tail_rows,
            )
            dump_simple_tail_top_tokens(
                simple_tail_path,
                tokenizer,
                eval_text,
                sample,
                eval_token_ids,
                eval_offsets,
                answer_indices,
                args.top_ratio,
                simple_tail_rows,
            )
            all_simple_tail_rows.extend(sample_tail_rows)
            print(
                f"finished {sample['sample_id']}: full_loss={full_metrics['loss']:.4f}, "
                f"top{args.top_ratio:g}_loss={pruned_metrics['loss']:.4f}, "
                f"dumped_query_rows={len(query_indices)} ({args.dump_query_scope}), "
                f"simple_tail_rows={len(simple_tail_query_indices)}",
                flush=True,
            )

    write_summary(output_dir / "attention_pruning_summary.csv", result_rows)
    write_tail_summaries(output_dir, all_simple_tail_rows)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    print(
        f"wrote {per_sample_path}, {output_dir / 'attention_pruning_summary.csv'}, "
        f"{top_token_path}, {simple_tail_path}, and tail10 summary files",
        flush=True,
    )


if __name__ == "__main__":
    main()
