from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
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
DEFAULT_TEXT_PATH = "/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt"

_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_MODULE_TO_LAYER: dict[int, int] = {}
_ACTIVE_CONTEXT: AnalysisContext | None = None


@dataclass
class AnalysisConfig:
    layers: set[int]
    heads: set[int]
    top_mass: float
    collect_vectors: bool
    attention_output_mode: str
    ppl_renormalize_selected: bool


class HeadStats:
    def __init__(self) -> None:
        self.count = 0
        self.full_norm_sum = 0.0
        self.top_norm_sum = 0.0
        self.tail_norm_sum = 0.0
        self.top_cond_norm_sum = 0.0
        self.tail_cond_norm_sum = 0.0
        self.top_mass_sum = 0.0
        self.tail_mass_sum = 0.0
        self.top_token_count_sum = 0.0
        self.tail_token_count_sum = 0.0
        self.cos_full_top_sum = 0.0
        self.cos_full_tail_sum = 0.0
        self.cos_top_tail_sum = 0.0
        self.cos_full_top_cond_sum = 0.0
        self.cos_full_tail_cond_sum = 0.0
        self.cos_top_cond_tail_cond_sum = 0.0
        self.l2_full_top_sum = 0.0
        self.l2_full_tail_sum = 0.0
        self.reconstruction_error_sum = 0.0

    def update(
        self,
        full_output: torch.Tensor,
        top_output: torch.Tensor,
        tail_output: torch.Tensor,
        top_cond_output: torch.Tensor,
        tail_cond_output: torch.Tensor,
        top_mass: torch.Tensor,
        tail_mass: torch.Tensor,
        top_count: torch.Tensor,
        tail_count: torch.Tensor,
    ) -> None:
        count = int(full_output.shape[0])
        self.count += count
        self.full_norm_sum += float(torch.linalg.vector_norm(full_output.float(), dim=-1).sum())
        self.top_norm_sum += float(torch.linalg.vector_norm(top_output.float(), dim=-1).sum())
        self.tail_norm_sum += float(torch.linalg.vector_norm(tail_output.float(), dim=-1).sum())
        self.top_cond_norm_sum += float(torch.linalg.vector_norm(top_cond_output.float(), dim=-1).sum())
        self.tail_cond_norm_sum += float(torch.linalg.vector_norm(tail_cond_output.float(), dim=-1).sum())
        self.top_mass_sum += float(top_mass.float().sum())
        self.tail_mass_sum += float(tail_mass.float().sum())
        self.top_token_count_sum += float(top_count.float().sum())
        self.tail_token_count_sum += float(tail_count.float().sum())
        self.cos_full_top_sum += float(F.cosine_similarity(full_output.float(), top_output.float(), dim=-1, eps=1e-8).sum())
        self.cos_full_tail_sum += float(F.cosine_similarity(full_output.float(), tail_output.float(), dim=-1, eps=1e-8).sum())
        self.cos_top_tail_sum += float(F.cosine_similarity(top_output.float(), tail_output.float(), dim=-1, eps=1e-8).sum())
        self.cos_full_top_cond_sum += float(
            F.cosine_similarity(full_output.float(), top_cond_output.float(), dim=-1, eps=1e-8).sum()
        )
        self.cos_full_tail_cond_sum += float(
            F.cosine_similarity(full_output.float(), tail_cond_output.float(), dim=-1, eps=1e-8).sum()
        )
        self.cos_top_cond_tail_cond_sum += float(
            F.cosine_similarity(top_cond_output.float(), tail_cond_output.float(), dim=-1, eps=1e-8).sum()
        )
        self.l2_full_top_sum += float(torch.linalg.vector_norm((full_output - top_output).float(), dim=-1).sum())
        self.l2_full_tail_sum += float(torch.linalg.vector_norm((full_output - tail_output).float(), dim=-1).sum())
        self.reconstruction_error_sum += float(
            torch.linalg.vector_norm((full_output - top_output - tail_output).float(), dim=-1).sum()
        )

    def row(self, layer: int, head: int) -> dict[str, Any]:
        denom = max(self.count, 1)
        return {
            "layer": layer,
            "head": head,
            "query_count": self.count,
            "mean_full_norm": self.full_norm_sum / denom,
            "mean_top90_norm": self.top_norm_sum / denom,
            "mean_tail10_norm": self.tail_norm_sum / denom,
            "mean_top90_cond_norm": self.top_cond_norm_sum / denom,
            "mean_tail10_cond_norm": self.tail_cond_norm_sum / denom,
            "mean_top90_mass": self.top_mass_sum / denom,
            "mean_tail10_mass": self.tail_mass_sum / denom,
            "mean_top90_token_count": self.top_token_count_sum / denom,
            "mean_tail10_token_count": self.tail_token_count_sum / denom,
            "cos_full_top90": self.cos_full_top_sum / denom,
            "cos_full_tail10": self.cos_full_tail_sum / denom,
            "cos_top90_tail10": self.cos_top_tail_sum / denom,
            "cos_full_top90_cond": self.cos_full_top_cond_sum / denom,
            "cos_full_tail10_cond": self.cos_full_tail_cond_sum / denom,
            "cos_top90_cond_tail10_cond": self.cos_top_cond_tail_cond_sum / denom,
            "mean_l2_full_minus_top90": self.l2_full_top_sum / denom,
            "mean_l2_full_minus_tail10": self.l2_full_tail_sum / denom,
            "mean_reconstruction_error": self.reconstruction_error_sum / denom,
        }


class AnalysisContext:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self.stats: dict[tuple[int, int], HeadStats] = {}

    def get_stats(self, layer: int, head: int) -> HeadStats:
        key = (layer, head)
        if key not in self.stats:
            self.stats[key] = HeadStats()
        return self.stats[key]

    def rows(self, layers: list[int], heads: list[int]) -> list[dict[str, Any]]:
        return [self.get_stats(layer, head).row(layer, head) for layer in layers for head in heads]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze top90/tail10/full attention-weighted V outputs by layer/head.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/attention_value_decomposition")
    parser.add_argument("--prefill_tokens", type=int, default=5000)
    parser.add_argument("--eval_tokens", type=int, default=5000)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--top_mass", type=float, default=0.90)
    parser.add_argument("--compute_vector_stats", type=str2bool, default=True)
    parser.add_argument("--compute_ppl", type=str2bool, default=True)
    parser.add_argument("--ppl_modes", default="full,top90,tail10")
    parser.add_argument(
        "--ppl_renormalize_selected",
        type=str2bool,
        default=False,
        help="If true, selected top/tail weights are renormalized before attn@V in PPL runs.",
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive.")
    if not (0.0 < args.top_mass < 1.0):
        raise ValueError("--top_mass must be in (0, 1).")
    return args


def parse_index_spec(spec: str, max_count: int, name: str) -> list[int]:
    normalized = spec.strip().lower()
    if normalized == "all":
        return list(range(max_count))
    selected: set[int] = set()
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            selected.update(range(int(left), int(right) + 1))
        else:
            selected.add(int(part))
    invalid = sorted(index for index in selected if index < 0 or index >= max_count)
    if invalid:
        raise ValueError(f"{name} out of range 0..{max_count - 1}: {invalid}")
    return sorted(selected)


def parse_modes(spec: str) -> list[str]:
    modes = [part.strip().lower() for part in spec.split(",") if part.strip()]
    valid = {"full", "top90", "tail10"}
    invalid = [mode for mode in modes if mode not in valid]
    if invalid:
        raise ValueError(f"Invalid --ppl_modes: {invalid}. Valid modes: {sorted(valid)}")
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


def read_text_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.read(max_chars) if max_chars > 0 else handle.read()


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def model_body(model: torch.nn.Module) -> torch.nn.Module:
    for attr_name in ("model", "transformer"):
        if hasattr(model, attr_name):
            return getattr(model, attr_name)
    return model


def register_attention_layers(model: torch.nn.Module) -> None:
    _MODULE_TO_LAYER.clear()
    layers = getattr(model_body(model), "layers", None)
    if layers is None:
        raise AttributeError("Could not find transformer layers on model.")
    for layer_idx, layer_obj in enumerate(layers):
        attn = getattr(layer_obj, "self_attn", None) or getattr(layer_obj, "attention", None)
        if attn is not None:
            _MODULE_TO_LAYER[id(attn)] = layer_idx


def repeat_kv_for_attention(value_states: torch.Tensor, attention_heads: int) -> torch.Tensor:
    if value_states.shape[1] == attention_heads:
        return value_states
    group_size = attention_heads // value_states.shape[1]
    return value_states.repeat_interleave(group_size, dim=1)


def top_tail_outputs(
    attention_weights: torch.Tensor,
    value_states_for_attention: torch.Tensor,
    top_mass_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # attention_weights: [batch, heads, query, key]
    # value_states_for_attention: [batch, heads, key, head_dim]
    sorted_weights, sorted_indices = torch.sort(attention_weights.float(), dim=-1, descending=True)
    valid_sorted = sorted_weights > 0
    cumulative = torch.cumsum(sorted_weights, dim=-1)
    rank = torch.arange(attention_weights.shape[-1], device=attention_weights.device).view(1, 1, 1, -1)
    first_crossing = (cumulative >= top_mass_threshold).float().argmax(dim=-1, keepdim=True)
    top_sorted_mask = (rank <= first_crossing) & valid_sorted
    top_mask = torch.zeros_like(top_sorted_mask, dtype=torch.bool).scatter(dim=-1, index=sorted_indices, src=top_sorted_mask)
    valid_mask = attention_weights > 0
    tail_mask = valid_mask & ~top_mask
    top_weights = attention_weights.float().masked_fill(~top_mask, 0.0)
    tail_weights = attention_weights.float().masked_fill(~tail_mask, 0.0)
    full_output = torch.matmul(attention_weights.float(), value_states_for_attention.float())
    top_output = torch.matmul(top_weights, value_states_for_attention.float())
    tail_output = torch.matmul(tail_weights, value_states_for_attention.float())
    top_mass = top_weights.sum(dim=-1)
    tail_mass = tail_weights.sum(dim=-1)
    top_cond_output = top_output / top_mass.clamp_min(1e-12).unsqueeze(-1)
    tail_cond_output = tail_output / tail_mass.clamp_min(1e-12).unsqueeze(-1)
    return full_output, top_output, tail_output, top_cond_output, tail_cond_output, top_mass, tail_mass


def selected_attention_output(
    attention_weights: torch.Tensor,
    value_states_for_attention: torch.Tensor,
    mode: str,
    top_mass_threshold: float,
    renormalize: bool,
) -> torch.Tensor:
    if mode == "full":
        return torch.matmul(attention_weights, value_states_for_attention)
    _, top_output, tail_output, _, _, top_mass, tail_mass = top_tail_outputs(
        attention_weights,
        value_states_for_attention,
        top_mass_threshold,
    )
    if mode == "top90":
        return top_output / top_mass.clamp_min(1e-12).unsqueeze(-1) if renormalize else top_output
    if mode == "tail10":
        return tail_output / tail_mass.clamp_min(1e-12).unsqueeze(-1) if renormalize else tail_output
    raise ValueError(f"Unsupported attention output mode: {mode}")


def crop_past_key_values(past_key_values: Any, max_length: int) -> Any:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        result = past_key_values.crop(max_length)
        return past_key_values if result is None else result
    if isinstance(past_key_values, tuple):
        cropped_layers = []
        for layer_cache in past_key_values:
            if isinstance(layer_cache, tuple):
                cropped_layers.append(tuple(tensor[..., :max_length, :] for tensor in layer_cache))
            else:
                cropped_layers.append(layer_cache)
        return tuple(cropped_layers)
    if isinstance(past_key_values, list):
        cropped_layers = []
        for layer_cache in past_key_values:
            if isinstance(layer_cache, tuple):
                cropped_layers.append(tuple(tensor[..., :max_length, :] for tensor in layer_cache))
            else:
                cropped_layers.append(layer_cache)
        return cropped_layers
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def update_vector_stats(
    context: AnalysisContext,
    layer: int,
    attention_weights: torch.Tensor,
    value_states_for_attention: torch.Tensor,
) -> None:
    config = context.config
    if layer not in config.layers:
        return
    full, top, tail, top_cond, tail_cond, top_mass, tail_mass = top_tail_outputs(
        attention_weights,
        value_states_for_attention,
        config.top_mass,
    )
    top_count = (top_mass > 0).new_zeros(top_mass.shape, dtype=torch.float32)
    tail_count = (tail_mass > 0).new_zeros(tail_mass.shape, dtype=torch.float32)
    sorted_weights, _ = torch.sort(attention_weights.float(), dim=-1, descending=True)
    valid_sorted = sorted_weights > 0
    cumulative = torch.cumsum(sorted_weights, dim=-1)
    rank = torch.arange(attention_weights.shape[-1], device=attention_weights.device).view(1, 1, 1, -1)
    first_crossing = (cumulative >= config.top_mass).float().argmax(dim=-1, keepdim=True)
    top_count = ((rank <= first_crossing) & valid_sorted).sum(dim=-1).float()
    tail_count = valid_sorted.sum(dim=-1).float() - top_count
    for head in sorted(config.heads):
        if head >= attention_weights.shape[1]:
            continue
        context.get_stats(layer, head).update(
            full[:, head].reshape(-1, full.shape[-1]).detach().cpu(),
            top[:, head].reshape(-1, top.shape[-1]).detach().cpu(),
            tail[:, head].reshape(-1, tail.shape[-1]).detach().cpu(),
            top_cond[:, head].reshape(-1, top_cond.shape[-1]).detach().cpu(),
            tail_cond[:, head].reshape(-1, tail_cond.shape[-1]).detach().cpu(),
            top_mass[:, head].reshape(-1).detach().cpu(),
            tail_mass[:, head].reshape(-1).detach().cpu(),
            top_count[:, head].reshape(-1).detach().cpu(),
            tail_count[:, head].reshape(-1).detach().cpu(),
        )


def patched_eager_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = _ACTIVE_CONTEXT
    if scaling is None:
        scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(query_states.shape[-1])))
    attention_heads = query_states.shape[1]
    value_states_for_attention = repeat_kv_for_attention(value_states, attention_heads)
    key_states_for_attention = repeat_kv_for_attention(key_states, attention_heads)
    scores = torch.matmul(query_states, key_states_for_attention.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=True)
    layer = _MODULE_TO_LAYER.get(id(module), -1)
    if context is not None and context.config.collect_vectors:
        update_vector_stats(context, layer, attention_weights, value_states_for_attention)
    attention_output = torch.matmul(attention_weights, value_states_for_attention)
    if context is not None and context.config.attention_output_mode != "full" and layer in context.config.layers:
        selected_output = selected_attention_output(
            attention_weights,
            value_states_for_attention,
            context.config.attention_output_mode,
            context.config.top_mass,
            context.config.ppl_renormalize_selected,
        )
        active_heads = [head for head in context.config.heads if 0 <= head < attention_output.shape[1]]
        if active_heads:
            head_index = torch.tensor(active_heads, dtype=torch.long, device=attention_output.device)
            attention_output[:, head_index] = selected_output[:, head_index]
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, attention_weights


def install_attention_patch() -> None:
    global _ORIGINAL_EAGER_ATTENTION_FORWARD
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is not None:
        return
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except Exception as exc:
        raise RuntimeError("Could not import transformers.models.qwen3.modeling_qwen3.") from exc
    _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
    setattr(modeling_qwen3, "eager_attention_forward", patched_eager_attention_forward)
    if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
        modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = patched_eager_attention_forward


@contextmanager
def active_context(context: AnalysisContext | None):
    global _ACTIVE_CONTEXT
    previous = _ACTIVE_CONTEXT
    _ACTIVE_CONTEXT = context
    try:
        yield
    finally:
        _ACTIVE_CONTEXT = previous


@torch.inference_mode()
def run_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    chunk_size: int,
    input_device: torch.device,
) -> tuple[Any, torch.Tensor]:
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
    if last_logits is None:
        raise RuntimeError("Prefill did not produce logits.")
    return past_key_values, last_logits


@torch.inference_mode()
def run_eval(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    context: AnalysisContext | None,
) -> tuple[float, float, int, Any]:
    total_loss = 0.0
    total_count = 0
    eval_end = prefill_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / chunk_size)
    with active_context(context):
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
            print(f"eval chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
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
    mean_loss = total_loss / max(1, total_count)
    return mean_loss, math.exp(mean_loss), total_count, past_key_values


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    total_tokens_needed = args.prefill_tokens + args.eval_tokens
    if args.require_total_tokens and len(token_ids) < total_tokens_needed:
        raise ValueError(f"Tokenization produced {len(token_ids)} tokens, fewer than {total_tokens_needed}.")
    input_ids = torch.tensor(token_ids[:total_tokens_needed], dtype=torch.long).view(1, -1)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": resolve_dtype(args.dtype, requested_device)}
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

    register_attention_layers(model)
    install_attention_patch()
    layer_count = int(getattr(model.config, "num_hidden_layers"))
    head_count = int(getattr(model.config, "num_attention_heads"))
    layers = parse_index_spec(args.layers, layer_count, "layers")
    heads = parse_index_spec(args.heads, head_count, "heads")

    base_config = AnalysisConfig(
        layers=set(layers),
        heads=set(heads),
        top_mass=args.top_mass,
        collect_vectors=False,
        attention_output_mode="full",
        ppl_renormalize_selected=args.ppl_renormalize_selected,
    )
    print("running shared full-attention prefill", flush=True)
    past_key_values, prev_logits = run_prefill(model, input_ids, args.prefill_tokens, args.chunk_size, input_device)

    ppl_rows: list[dict[str, Any]] = []
    if args.compute_vector_stats:
        stats_config = AnalysisConfig(**{**base_config.__dict__, "collect_vectors": True})
        stats_context = AnalysisContext(stats_config)
        print("collecting vector stats on full attention eval", flush=True)
        past_key_values = crop_past_key_values(past_key_values, args.prefill_tokens)
        loss, ppl, count, past_key_values = run_eval(
            model,
            input_ids,
            args.prefill_tokens,
            args.eval_tokens,
            args.chunk_size,
            input_device,
            past_key_values,
            prev_logits,
            stats_context,
        )
        past_key_values = crop_past_key_values(past_key_values, args.prefill_tokens)
        write_csv(
            output_dir / "value_decomposition_by_head.csv",
            stats_context.rows(layers, heads),
            [
                "layer",
                "head",
                "query_count",
                "mean_full_norm",
                "mean_top90_norm",
                "mean_tail10_norm",
                "mean_top90_cond_norm",
                "mean_tail10_cond_norm",
                "mean_top90_mass",
                "mean_tail10_mass",
                "mean_top90_token_count",
                "mean_tail10_token_count",
                "cos_full_top90",
                "cos_full_tail10",
                "cos_top90_tail10",
                "cos_full_top90_cond",
                "cos_full_tail10_cond",
                "cos_top90_cond_tail10_cond",
                "mean_l2_full_minus_top90",
                "mean_l2_full_minus_tail10",
                "mean_reconstruction_error",
            ],
        )
        if args.compute_ppl and "full" in parse_modes(args.ppl_modes):
            ppl_rows.append({"mode": "full", "loss": loss, "ppl": ppl, "token_count": count})

    if args.compute_ppl:
        existing_modes = {row["mode"] for row in ppl_rows}
        for mode in parse_modes(args.ppl_modes):
            if mode in existing_modes:
                continue
            print(f"running PPL mode={mode}", flush=True)
            mode_config = AnalysisConfig(
                layers=set(layers),
                heads=set(heads),
                top_mass=args.top_mass,
                collect_vectors=False,
                attention_output_mode=mode,
                ppl_renormalize_selected=args.ppl_renormalize_selected,
            )
            past_key_values = crop_past_key_values(past_key_values, args.prefill_tokens)
            loss, ppl, count, past_key_values = run_eval(
                model,
                input_ids,
                args.prefill_tokens,
                args.eval_tokens,
                args.chunk_size,
                input_device,
                past_key_values,
                prev_logits,
                AnalysisContext(mode_config),
            )
            past_key_values = crop_past_key_values(past_key_values, args.prefill_tokens)
            ppl_rows.append({"mode": mode, "loss": loss, "ppl": ppl, "token_count": count})
        write_csv(output_dir / "ppl_by_attention_value_mode.csv", ppl_rows, ["mode", "loss", "ppl", "token_count"])

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "text_path": args.text_path,
        "prefill_tokens": args.prefill_tokens,
        "eval_tokens": args.eval_tokens,
        "chunk_size": args.chunk_size,
        "layers": layers,
        "heads": heads,
        "top_mass": args.top_mass,
        "ppl_renormalize_selected": args.ppl_renormalize_selected,
        "outputs": {
            "value_decomposition_by_head": str(output_dir / "value_decomposition_by_head.csv"),
            "ppl_by_attention_value_mode": str(output_dir / "ppl_by_attention_value_mode.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
