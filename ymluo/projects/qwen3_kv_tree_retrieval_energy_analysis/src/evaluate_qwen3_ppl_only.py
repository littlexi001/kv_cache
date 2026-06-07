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
DEFAULT_TEXT_PATH = (
    "/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt"
)

_ACTIVE_PRUNE_RATIO: float | None = None
_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 PPL only.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/ppl_only")
    parser.add_argument("--prefill_tokens", type=int, default=5000)
    parser.add_argument("--eval_tokens", type=int, default=5000)
    parser.add_argument(
        "--eval_last_tokens_only",
        type=str2bool,
        default=False,
        help="Use the whole tokenized text as context and evaluate only the last eval_tokens tokens.",
    )
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument(
        "--ratios",
        default="",
        help="Comma-separated top attention keep ratios for optional pruned PPL. Empty means baseline only.",
    )
    return parser.parse_args()


def parse_float_list(spec: str) -> list[float]:
    if not spec.strip():
        return []
    values = [float(part.strip()) for part in spec.split(",") if part.strip()]
    if any(value <= 0.0 or value > 1.0 for value in values):
        raise ValueError("--ratios must contain floats in (0, 1].")
    return values


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


def read_text_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        if max_chars > 0:
            return handle.read(max_chars)
        return handle.read()


def pick_input_device(model: torch.nn.Module, fallback_device: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback_device


def model_forward(model: torch.nn.Module, kwargs: dict[str, Any]) -> Any:
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "cache_position" in kwargs and "cache_position" in str(exc):
            kwargs = dict(kwargs)
            kwargs.pop("cache_position")
            return model(**kwargs)
        raise


def ppl_fields() -> list[str]:
    return ["mode", "ratio", "kept_percent", "loss", "ppl", "token_count"]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def prefill_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    chunk_size: int,
    input_device: torch.device,
) -> tuple[Any, torch.Tensor | None]:
    past_key_values = None
    last_logits: torch.Tensor | None = None
    total_chunks = math.ceil(prefill_tokens / chunk_size)
    for chunk_idx, start in enumerate(range(0, prefill_tokens, chunk_size), start=1):
        end = min(start + chunk_size, prefill_tokens)
        chunk = input_ids[:, start:end].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        print(f"prefill chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :].detach()
        del outputs, chunk
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    return past_key_values, last_logits


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
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is not None:
        return
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except Exception as exc:
        raise RuntimeError("Could not import transformers.models.qwen3.modeling_qwen3 for PPL pruning.") from exc
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
def compute_eval_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    prune_ratio: float | None,
) -> tuple[float, float, int]:
    past_key_values, prev_logits = prefill_cache(model, input_ids, prefill_tokens, chunk_size, input_device)
    if prev_logits is None:
        raise RuntimeError("Prefill did not return last logits.")
    total_loss = 0.0
    total_count = 0
    eval_end = prefill_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / chunk_size)
    for chunk_idx, start in enumerate(range(prefill_tokens, eval_end, chunk_size), start=1):
        end = min(start + chunk_size, eval_end)
        chunk = input_ids[:, start:end].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        label = "baseline" if prune_ratio is None else f"ratio {prune_ratio:g}"
        print(f"ppl {label} chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        with pruning_ratio(prune_ratio):
            outputs = model_forward(model, kwargs)
        logits = outputs.logits
        shifted_logits = torch.cat([prev_logits.unsqueeze(1), logits[:, :-1, :]], dim=1)
        loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
            chunk.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss)
        total_count += int(chunk.numel())
        prev_logits = logits[:, -1, :].detach()
        past_key_values = outputs.past_key_values
        del outputs, chunk, logits, shifted_logits, loss
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    mean_loss = total_loss / max(1, total_count)
    return mean_loss, math.exp(mean_loss), total_count


def main() -> None:
    args = parse_args()
    ratios = parse_float_list(args.ratios)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)

    if args.eval_last_tokens_only:
        if len(token_ids) <= args.eval_tokens:
            raise ValueError(
                f"Tokenization produced {len(token_ids)} tokens, "
                f"not enough to evaluate the last {args.eval_tokens} tokens with a non-empty prefill."
            )
        prefill_tokens = len(token_ids) - args.eval_tokens
    else:
        prefill_tokens = args.prefill_tokens
        total_tokens_needed = prefill_tokens + args.eval_tokens
        if args.require_total_tokens and len(token_ids) < total_tokens_needed:
            raise ValueError(f"Tokenization produced {len(token_ids)} tokens, fewer than {total_tokens_needed}.")
        token_ids = token_ids[:total_tokens_needed]
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": model_dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(requested_device)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, requested_device)

    if ratios:
        install_qwen3_attention_patch()

    rows: list[dict[str, Any]] = []
    baseline_loss, baseline_ppl, baseline_count = compute_eval_loss(
        model, input_ids, prefill_tokens, args.eval_tokens, args.chunk_size, input_device, None
    )
    rows.append(
        {
            "mode": "baseline",
            "ratio": "",
            "kept_percent": 100.0,
            "loss": baseline_loss,
            "ppl": baseline_ppl,
            "token_count": baseline_count,
        }
    )
    for ratio in ratios:
        loss, ppl, count = compute_eval_loss(
            model, input_ids, prefill_tokens, args.eval_tokens, args.chunk_size, input_device, ratio
        )
        rows.append(
            {
                "mode": "top_ratio",
                "ratio": ratio,
                "kept_percent": 100.0 * ratio,
                "loss": loss,
                "ppl": ppl,
                "token_count": count,
            }
        )

    ppl_path = output_dir / "ppl_by_ratio.csv"
    write_csv(ppl_path, rows, ppl_fields())
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "resolved": {
                    "prefill_tokens": prefill_tokens,
                    "eval_tokens": args.eval_tokens,
                    "eval_last_tokens_only": args.eval_last_tokens_only,
                    "total_tokenized_tokens_used": int(input_ids.numel()),
                    "ratios": ratios,
                },
                "paths": {"ppl_by_ratio": str(ppl_path)},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote PPL outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
