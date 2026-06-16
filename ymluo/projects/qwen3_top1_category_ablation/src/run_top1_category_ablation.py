from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
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


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = str(REPO_ROOT / "ymluo" / "models" / "Qwen3-0.6B")
DEFAULT_DATA_PATH = str(
    REPO_ROOT
    / "ymluo"
    / "projects"
    / "qwen3_kcache_l2_neighbor_analysis"
    / "data"
    / "needle_in_haystack"
    / "needle_in_haystack.jsonl"
)
DEFAULT_OUTPUT_DIR = str(
    REPO_ROOT / "ymluo" / "projects" / "qwen3_top1_category_ablation" / "outputs" / "top1_category_ablation"
)

_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_CONTEXT: "AblationContext | None" = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Needle-in-haystack answer PPL and final hidden-state SVD under "
            "attention top1 category ablations."
        )
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--max_context_chars", type=int, default=24000)
    parser.add_argument("--top_ratio", type=float, default=0.01)
    parser.add_argument(
        "--modes",
        default="full_attention,top1_all,answer_only,front_only,end_only,other_only",
        help=(
            "Comma-separated modes. Available: full_attention, top1_all, answer_only, front_only, "
            "end_only, other_only, drop_answer, drop_front, drop_end, drop_other."
        ),
    )
    parser.add_argument("--svd_max_vectors", type=int, default=4096)
    parser.add_argument("--svd_top_k", type=int, default=128)
    parser.add_argument("--dump_top_tokens", type=str2bool, default=True)
    parser.add_argument("--do_generate", type=str2bool, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--trust_remote_code", type=str2bool, default=True)
    return parser.parse_args()


def parse_modes(spec: str) -> list[str]:
    valid = {
        "full_attention",
        "top1_all",
        "answer_only",
        "front_only",
        "end_only",
        "other_only",
        "drop_answer",
        "drop_front",
        "drop_end",
        "drop_other",
    }
    modes = [item.strip() for item in spec.split(",") if item.strip()]
    invalid = [item for item in modes if item not in valid]
    if invalid:
        raise ValueError(f"Unsupported modes: {invalid}. Valid modes: {sorted(valid)}")
    return modes


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
    instruction = sample.get(
        "instruction",
        "Use the context to answer the question. Answer with the exact phrase when possible.",
    )
    return (
        f"{instruction}\n\n"
        f"Context:\n{sample['context']}\n\n"
        f"Question: {sample['question']}\n"
        "Answer:"
    )


def token_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    return encoded.input_ids[0].tolist(), [(int(a), int(b)) for a, b in encoded.offset_mapping[0].tolist()]


def prompt_answer_payload(tokenizer: Any, prompt: str, answer: str) -> dict[str, Any]:
    answer_text = " " + answer
    eval_text = prompt + answer_text
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    token_ids, offsets = token_offsets(tokenizer, eval_text)
    answer_ids = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    return {
        "eval_text": eval_text,
        "token_ids": token_ids,
        "offsets": offsets,
        "prompt_token_count": len(prompt_ids),
        "answer_token_count": len(answer_ids),
    }


def answer_span_token_indices(prompt: str, answer: str, offsets: list[tuple[int, int]], prompt_token_count: int) -> set[int]:
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


def evidence_span_token_indices(
    sample: dict[str, Any],
    prompt: str,
    offsets: list[tuple[int, int]],
    prompt_token_count: int,
) -> set[int]:
    evidence_texts = sample.get("answer_evidence_texts")
    if not evidence_texts:
        return answer_span_token_indices(prompt, str(sample.get("answer", "")), offsets, prompt_token_count)
    indices: set[int] = set()
    for evidence in evidence_texts:
        if not evidence:
            continue
        start = 0
        while True:
            char_start = prompt.find(str(evidence), start)
            if char_start < 0:
                break
            char_end = char_start + len(str(evidence))
            for token_index, (token_start, token_end) in enumerate(offsets[:prompt_token_count]):
                if token_start < char_end and token_end > char_start:
                    indices.add(token_index)
            start = char_end
    return indices


def category_for_key(
    key_index: int,
    visible_count: int,
    front_edge_count: int,
    end_start_index: int,
    answer_indices: set[int],
) -> str:
    if key_index in answer_indices:
        return "answer"
    if key_index < front_edge_count:
        return "front"
    if key_index >= end_start_index:
        return "end"
    return "other"


def token_indices_overlapping(offsets: list[tuple[int, int]], start_char: int, end_char: int) -> set[int]:
    return {
        token_index
        for token_index, (token_start, token_end) in enumerate(offsets)
        if token_start < end_char and token_end > start_char
    }


def first_token_overlapping(offsets: list[tuple[int, int]], start_char: int, limit: int) -> int:
    for token_index, (token_start, token_end) in enumerate(offsets[:limit]):
        if token_end > start_char:
            return token_index
    return max(0, limit - 1)


def answer_suffix_start_token(prompt: str, offsets: list[tuple[int, int]], prompt_token_count: int) -> int:
    marker_pos = prompt.find("\n\nQuestion:")
    if marker_pos < 0:
        marker_pos = prompt.find("Question:")
    if marker_pos < 0:
        tail_count = max(1, math.ceil(0.01 * prompt_token_count))
        return max(0, prompt_token_count - tail_count)
    return first_token_overlapping(offsets, marker_pos, prompt_token_count)


def end_start_for_query(visible_count: int, front_edge_count: int, suffix_start_index: int) -> int:
    tail_start = max(0, visible_count - front_edge_count)
    return min(tail_start, suffix_start_index)


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def keyword_recall(predicted: str, answer: str) -> tuple[int, int, float]:
    gold_words = normalize_words(answer)
    pred_counts = Counter(normalize_words(predicted))
    correct = 0
    for word in gold_words:
        if pred_counts[word] > 0:
            correct += 1
            pred_counts[word] -= 1
    total = len(gold_words)
    return correct, total, correct / total if total else 0.0


class AblationContext:
    def __init__(
        self,
        mode: str,
        top_ratio: float,
        answer_indices: set[int],
        query_indices: set[int],
        suffix_start_index: int,
    ) -> None:
        self.mode = mode
        self.top_ratio = top_ratio
        self.answer_indices = answer_indices
        self.query_indices = query_indices
        self.suffix_start_index = suffix_start_index

    @property
    def allowed_categories(self) -> set[str] | None:
        if self.mode == "full_attention":
            return None
        if self.mode == "top1_all":
            return {"answer", "front", "end", "other"}
        if self.mode == "answer_only":
            return {"answer"}
        if self.mode == "front_only":
            return {"front"}
        if self.mode == "end_only":
            return {"end"}
        if self.mode == "other_only":
            return {"other"}
        if self.mode == "drop_answer":
            return {"front", "end", "other"}
        if self.mode == "drop_front":
            return {"answer", "end", "other"}
        if self.mode == "drop_end":
            return {"answer", "front", "other"}
        if self.mode == "drop_other":
            return {"answer", "front", "end"}
        raise ValueError(f"Unsupported mode: {self.mode}")


def _category_keep_mask(scores: torch.Tensor, context: AblationContext) -> torch.Tensor:
    keep = torch.zeros_like(scores, dtype=torch.bool)
    query_count = int(scores.shape[-2])
    allowed = context.allowed_categories
    if allowed is None:
        return torch.ones_like(scores, dtype=torch.bool)
    for query_index in range(query_count):
        if query_index not in context.query_indices:
            keep[:, :, query_index, :] = True
            continue
        row = scores[:, :, query_index, :]
        finite = torch.isfinite(row[0, 0])
        valid_count = int(finite.sum().item())
        if valid_count <= 0:
            continue
        front_edge_count = max(1, math.ceil(context.top_ratio * valid_count))
        keep_count = min(valid_count, front_edge_count)
        values, top_indices = torch.topk(row, k=keep_count, dim=-1, largest=True)
        del values
        end_start_index = end_start_for_query(valid_count, front_edge_count, context.suffix_start_index)
        cat_mask_1d = torch.zeros(valid_count, dtype=torch.bool, device=scores.device)
        for key_index in range(valid_count):
            category = category_for_key(
                key_index,
                valid_count,
                front_edge_count,
                end_start_index,
                context.answer_indices,
            )
            if category in allowed:
                cat_mask_1d[key_index] = True
        selected_ok = cat_mask_1d.gather(0, top_indices.reshape(-1)).reshape(top_indices.shape)
        if bool(selected_ok.any().item()):
            filtered_indices = top_indices.masked_fill(~selected_ok, 0)
            filtered_keep = torch.zeros_like(row, dtype=torch.bool)
            filtered_keep.scatter_(-1, filtered_indices, selected_ok)
            keep[:, :, query_index, :] = filtered_keep
        else:
            # Avoid all-masked rows. Fallback to the strongest valid token for this row.
            fallback = torch.argmax(row, dim=-1, keepdim=True)
            keep[:, :, query_index, :].scatter_(-1, fallback, True)
    return keep


def _patched_eager_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(query_states.shape[-1])))
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states = key_states.repeat_interleave(repeat_groups, dim=1)
        value_states = value_states.repeat_interleave(repeat_groups, dim=1)
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    if _ACTIVE_CONTEXT is not None and _ACTIVE_CONTEXT.mode != "full_attention":
        keep = _category_keep_mask(scores, _ACTIVE_CONTEXT)
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
        raise RuntimeError(
            "Could not import transformers.models.qwen3.modeling_qwen3. "
            "Install a recent transformers version with Qwen3 support."
        ) from exc
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
        setattr(modeling_qwen3, "eager_attention_forward", _patched_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _patched_eager_attention_forward


@contextmanager
def active_context(context: AblationContext | None):
    global _ACTIVE_CONTEXT
    previous = _ACTIVE_CONTEXT
    _ACTIVE_CONTEXT = context
    try:
        yield
    finally:
        _ACTIVE_CONTEXT = previous


@torch.inference_mode()
def answer_nll_and_x(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_token_count: int,
    answer_token_count: int,
    input_device: torch.device,
    context: AblationContext | None,
) -> dict[str, Any]:
    input_ids = input_ids.to(input_device)
    with active_context(context):
        outputs = model(
            input_ids=input_ids,
            return_dict=True,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=False,
        )
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    answer_start = prompt_token_count - 1
    answer_logits = logits[:, answer_start : answer_start + answer_token_count, :]
    answer_labels = labels[:, answer_start : answer_start + answer_token_count]
    losses = F.cross_entropy(
        answer_logits.reshape(-1, answer_logits.shape[-1]),
        answer_labels.reshape(-1),
        reduction="none",
    )
    loss = float(losses.mean())
    predicted_answer_ids = answer_logits.argmax(dim=-1)
    token_matches = predicted_answer_ids.eq(answer_labels)
    token_accuracy = float(token_matches.float().mean())
    hidden = outputs.hidden_states[-1][0, answer_start : answer_start + answer_token_count, :].detach().float().cpu()
    del outputs, logits, labels, answer_logits, answer_labels, losses
    return {
        "loss": loss,
        "ppl": float(math.exp(min(loss, 80.0))),
        "x": hidden,
        "predicted_answer_ids": predicted_answer_ids[0].detach().cpu().tolist(),
        "answer_token_accuracy": token_accuracy,
        "answer_exact_match": bool(token_matches.all().item()),
    }


@torch.inference_mode()
def generate_answer(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    input_device: torch.device,
    context: AblationContext | None,
    max_new_tokens: int,
) -> str:
    input_ids = input_ids.to(input_device)
    with active_context(context):
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    generated = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def collect_top1_category_counts(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_token_count: int,
    answer_token_count: int,
    answer_indices: set[int],
    suffix_start_index: int,
    top_ratio: float,
    input_device: torch.device,
) -> list[dict[str, Any]]:
    input_ids = input_ids.to(input_device)
    with active_context(None):
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )
    if outputs.attentions is None:
        raise RuntimeError("Model did not return attentions. Use --attn_implementation eager.")
    query_indices = range(prompt_token_count - 1, prompt_token_count + answer_token_count - 1)
    rows: list[dict[str, Any]] = []
    for layer, attn in enumerate(outputs.attentions):
        attn = attn[0].float().cpu()
        for head in range(attn.shape[0]):
            counts = {"answer": 0, "front": 0, "end": 0, "other": 0}
            selected = 0
            mass = {"answer": 0.0, "front": 0.0, "end": 0.0, "other": 0.0}
            for query_index in query_indices:
                current = attn[head, query_index, : query_index + 1]
                visible_count = int(current.numel())
                keep_count = max(1, math.ceil(top_ratio * visible_count))
                values, indices = torch.topk(current, k=keep_count, largest=True)
                front_edge_count = max(1, math.ceil(top_ratio * visible_count))
                end_start_index = end_start_for_query(visible_count, front_edge_count, suffix_start_index)
                for key_index, value in zip(indices.tolist(), values.tolist()):
                    category = category_for_key(
                        int(key_index),
                        visible_count,
                        front_edge_count,
                        end_start_index,
                        answer_indices,
                    )
                    counts[category] += 1
                    mass[category] += float(value)
                    selected += 1
            denom = max(1, selected)
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "selected_token_count": selected,
                    "answer_count": counts["answer"],
                    "front_count": counts["front"],
                    "end_count": counts["end"],
                    "other_count": counts["other"],
                    "answer_pct": counts["answer"] / denom,
                    "front_pct": counts["front"] / denom,
                    "end_pct": counts["end"] / denom,
                    "other_pct": counts["other"] / denom,
                    "answer_attention_mass": mass["answer"],
                    "front_attention_mass": mass["front"],
                    "end_attention_mass": mass["end"],
                    "other_attention_mass": mass["other"],
                }
            )
    del outputs
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_ppl(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for mode in sorted({str(row["mode"]) for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        mean_loss = sum(float(row["loss"]) for row in mode_rows) / len(mode_rows)
        result.append(
            {
                "mode": mode,
                "sample_count": len(mode_rows),
                "mean_loss": mean_loss,
                "mean_ppl": math.exp(min(mean_loss, 80.0)),
                "mean_answer_token_count": sum(int(row["answer_token_count"]) for row in mode_rows) / len(mode_rows),
                "mean_prompt_token_count": sum(int(row["prompt_token_count"]) for row in mode_rows) / len(mode_rows),
                "mean_answer_token_accuracy": sum(float(row["answer_token_accuracy"]) for row in mode_rows)
                / len(mode_rows),
                "answer_exact_match_rate": sum(str(row["answer_exact_match"]) == "True" for row in mode_rows)
                / len(mode_rows),
                "mean_keyword_recall": sum(float(row["keyword_recall"]) for row in mode_rows) / len(mode_rows)
                if all(row.get("keyword_recall", "") != "" for row in mode_rows)
                else "",
                "contains_answer_rate": (
                    sum(str(row.get("contains_answer", "")) == "True" for row in mode_rows)
                    / sum(row.get("contains_answer", "") != "" for row in mode_rows)
                    if any(row.get("contains_answer", "") != "" for row in mode_rows)
                    else ""
                ),
            }
        )
    return result


def compute_svd_rows(x_by_mode: dict[str, list[torch.Tensor]], svd_top_k: int, centered: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, tensors in x_by_mode.items():
        if not tensors:
            continue
        matrix = torch.cat(tensors, dim=0).float()
        if centered:
            matrix = matrix - matrix.mean(dim=0, keepdim=True)
        singular_values = torch.linalg.svdvals(matrix)
        total_energy = float((singular_values * singular_values).sum())
        cumulative = 0.0
        for rank, value in enumerate(singular_values[:svd_top_k].tolist(), start=1):
            energy = (float(value) * float(value)) / max(total_energy, 1e-30)
            cumulative += energy
            rows.append(
                {
                    "mode": mode,
                    "centered": centered,
                    "rank": rank,
                    "singular_value": float(value),
                    "energy_ratio": energy,
                    "cumulative_energy_ratio": cumulative,
                    "x_rows": int(matrix.shape[0]),
                    "x_cols": int(matrix.shape[1]),
                }
            )
    return rows


def maybe_store_x(x_by_mode: dict[str, list[torch.Tensor]], mode: str, x: torch.Tensor, max_vectors: int) -> None:
    if max_vectors <= 0:
        return
    existing = sum(int(t.shape[0]) for t in x_by_mode.setdefault(mode, []))
    remaining = max_vectors - existing
    if remaining <= 0:
        return
    x_by_mode[mode].append(x[:remaining].contiguous())


def main() -> None:
    args = parse_args()
    if args.top_ratio <= 0.0 or args.top_ratio > 1.0:
        raise ValueError("--top_ratio must be in (0, 1].")
    modes = parse_modes(args.modes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    install_qwen3_attention_patch()
    requested_device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": args.attn_implementation,
    }
    if requested_device.type != "cpu" and args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if requested_device.type == "cpu" or args.device_map.lower() == "none":
        model.to(requested_device)
    model.eval()
    input_device = pick_input_device(model, requested_device)

    samples = load_samples(Path(args.data_path), args.max_samples, args.max_context_chars)
    per_sample_path = output_dir / "per_sample_results.jsonl"
    if per_sample_path.exists():
        per_sample_path.unlink()

    per_sample_rows: list[dict[str, Any]] = []
    top_token_rows: list[dict[str, Any]] = []
    x_by_mode: dict[str, list[torch.Tensor]] = {mode: [] for mode in modes}
    used_samples_path = output_dir / "used_samples.jsonl"
    if used_samples_path.exists():
        used_samples_path.unlink()

    for sample in samples:
        prompt = build_prompt(sample)
        payload = prompt_answer_payload(tokenizer, prompt, str(sample["answer"]))
        token_ids = payload["token_ids"]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        prompt_token_count = int(payload["prompt_token_count"])
        answer_token_count = int(payload["answer_token_count"])
        answer_indices = evidence_span_token_indices(sample, prompt, payload["offsets"], prompt_token_count)
        suffix_start_index = answer_suffix_start_token(prompt, payload["offsets"], prompt_token_count)
        append_jsonl(
            used_samples_path,
            {
                "sample_id": sample["sample_id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "needle_text": sample["needle_text"],
                "context": sample["context"],
                "prompt": prompt,
                "prompt_token_count": prompt_token_count,
                "answer_token_count": answer_token_count,
                "answer_evidence_token_count": len(answer_indices),
                "end_suffix_start_token": suffix_start_index,
            },
        )

        if args.dump_top_tokens:
            for row in collect_top1_category_counts(
                model,
                input_ids,
                prompt_token_count,
                answer_token_count,
                answer_indices,
                suffix_start_index,
                args.top_ratio,
                input_device,
            ):
                top_token_rows.append({"sample_id": sample["sample_id"], **row})

        for mode in modes:
            query_indices = set(range(prompt_token_count - 1, prompt_token_count + answer_token_count - 1))
            context = (
                None
                if mode == "full_attention"
                else AblationContext(mode, args.top_ratio, answer_indices, query_indices, suffix_start_index)
            )
            metrics = answer_nll_and_x(
                model,
                input_ids,
                prompt_token_count,
                answer_token_count,
                input_device,
                context,
            )
            row = {
                "sample_id": sample["sample_id"],
                "mode": mode,
                "answer": sample["answer"],
                "loss": metrics["loss"],
                "ppl": metrics["ppl"],
                "prompt_token_count": prompt_token_count,
                "answer_token_count": answer_token_count,
                "top_ratio": args.top_ratio if mode != "full_attention" else "",
                "answer_evidence_token_count": len(answer_indices),
                "answer_token_accuracy": metrics["answer_token_accuracy"],
                "answer_exact_match": str(metrics["answer_exact_match"]),
                "greedy_answer": tokenizer.decode(metrics["predicted_answer_ids"], skip_special_tokens=True).strip(),
                "keyword_correct": "",
                "keyword_total": "",
                "keyword_recall": "",
                "generated": "",
                "contains_answer": "",
            }
            keyword_correct, keyword_total, keyword_value = keyword_recall(row["greedy_answer"], str(sample["answer"]))
            row["keyword_correct"] = keyword_correct
            row["keyword_total"] = keyword_total
            row["keyword_recall"] = keyword_value
            if args.do_generate:
                prompt_input_ids = torch.tensor([token_ids[:prompt_token_count]], dtype=torch.long)
                generated = generate_answer(
                    model,
                    tokenizer,
                    prompt_input_ids,
                    input_device,
                    context,
                    args.max_new_tokens,
                )
                row["generated"] = generated
                row["contains_answer"] = str(str(sample["answer"]).lower() in generated.lower())
            per_sample_rows.append(row)
            append_jsonl(per_sample_path, row)
            maybe_store_x(x_by_mode, mode, metrics["x"], args.svd_max_vectors)
            print(
                f"finished {sample['sample_id']} {mode}: loss={metrics['loss']:.4f}, ppl={metrics['ppl']:.4f}",
                flush=True,
            )
            del metrics
        if input_device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(
        output_dir / "ppl_summary.csv",
        summarize_ppl(per_sample_rows),
        [
            "mode",
            "sample_count",
            "mean_loss",
            "mean_ppl",
            "mean_answer_token_count",
            "mean_prompt_token_count",
            "mean_answer_token_accuracy",
            "answer_exact_match_rate",
            "mean_keyword_recall",
            "contains_answer_rate",
        ],
    )
    write_csv(
        output_dir / "per_sample_results.csv",
        per_sample_rows,
        [
            "sample_id",
            "mode",
            "answer",
            "loss",
            "ppl",
            "prompt_token_count",
            "answer_token_count",
            "top_ratio",
            "answer_evidence_token_count",
            "answer_token_accuracy",
            "answer_exact_match",
            "greedy_answer",
            "keyword_correct",
            "keyword_total",
            "keyword_recall",
            "generated",
            "contains_answer",
        ],
    )
    if top_token_rows:
        write_csv(
            output_dir / "top1_category_counts_by_layer_head.csv",
            top_token_rows,
            [
                "sample_id",
                "layer",
                "head",
                "selected_token_count",
                "answer_count",
                "front_count",
                "end_count",
                "other_count",
                "answer_pct",
                "front_pct",
                "end_pct",
                "other_pct",
                "answer_attention_mass",
                "front_attention_mass",
                "end_attention_mass",
                "other_attention_mass",
            ],
        )

    svd_rows = compute_svd_rows(x_by_mode, args.svd_top_k, centered=False)
    svd_rows.extend(compute_svd_rows(x_by_mode, args.svd_top_k, centered=True))
    write_csv(
        output_dir / "x_singular_values.csv",
        svd_rows,
        [
            "mode",
            "centered",
            "rank",
            "singular_value",
            "energy_ratio",
            "cumulative_energy_ratio",
            "x_rows",
            "x_cols",
        ],
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": vars(args),
                "resolved": {
                    "modes": modes,
                    "sample_count": len(samples),
                    "input_device": str(input_device),
                    "model_path": args.model_name_or_path,
                    "data_path": args.data_path,
                },
                "outputs": {
                    "per_sample_jsonl": str(per_sample_path),
                    "per_sample_csv": str(output_dir / "per_sample_results.csv"),
                    "ppl_summary": str(output_dir / "ppl_summary.csv"),
                    "top1_category_counts": str(output_dir / "top1_category_counts_by_layer_head.csv"),
                    "x_singular_values": str(output_dir / "x_singular_values.csv"),
                    "used_samples": str(used_samples_path),
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
