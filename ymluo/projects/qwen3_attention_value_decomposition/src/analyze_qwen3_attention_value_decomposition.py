from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
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


@dataclass(frozen=True)
class VectorSpec:
    name: str
    side: str
    value: float


@dataclass
class AnalysisConfig:
    layers: set[int]
    heads: set[int]
    split_mode: str
    vector_specs: list[VectorSpec]
    collect_vectors: bool
    attention_output_mode: str
    ppl_renormalize_selected: bool
    save_pairwise_per_token: bool
    save_pairwise_hist: bool
    hist_bins: int
    pair_specs: list[tuple[str, str]]
    query_offset: int = 0


class VectorAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.norm_sum = 0.0
        self.mass_sum = 0.0
        self.token_count_sum = 0.0

    def update(self, vectors: torch.Tensor, mass: torch.Tensor, token_count: torch.Tensor) -> None:
        flat = vectors.float().reshape(-1, vectors.shape[-1])
        self.count += int(flat.shape[0])
        self.norm_sum += float(torch.linalg.vector_norm(flat, dim=-1).sum())
        self.mass_sum += float(mass.float().reshape(-1).sum())
        self.token_count_sum += float(token_count.float().reshape(-1).sum())

    def row(self, layer: int, head: int, name: str) -> dict[str, Any]:
        denom = max(self.count, 1)
        return {
            "layer": layer,
            "head": head,
            "vector": name,
            "query_count": self.count,
            "mean_norm": self.norm_sum / denom,
            "mean_attention_mass": self.mass_sum / denom,
            "mean_token_count": self.token_count_sum / denom,
        }


class PairAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.cos_sum = 0.0
        self.l2_sum = 0.0

    def update(self, left: torch.Tensor, right: torch.Tensor) -> None:
        left_flat = left.float().reshape(-1, left.shape[-1])
        right_flat = right.float().reshape(-1, right.shape[-1])
        self.count += int(left_flat.shape[0])
        self.cos_sum += float(F.cosine_similarity(left_flat, right_flat, dim=-1, eps=1e-8).sum())
        self.l2_sum += float(torch.linalg.vector_norm(left_flat - right_flat, dim=-1).sum())

    def row(self, layer: int, head: int, left: str, right: str) -> dict[str, Any]:
        denom = max(self.count, 1)
        return {
            "layer": layer,
            "head": head,
            "left": left,
            "right": right,
            "query_count": self.count,
            "mean_cosine": self.cos_sum / denom,
            "mean_l2": self.l2_sum / denom,
        }


class HistogramAccumulator:
    def __init__(self, bins: int) -> None:
        self.bins = bins
        self.counts = [0] * bins

    def update(self, values: torch.Tensor) -> None:
        hist = torch.histc(values.float().clamp(-1.0, 1.0), bins=self.bins, min=-1.0, max=1.0)
        hist_cpu = hist.detach().cpu().to(torch.long).tolist()
        for index, count in enumerate(hist_cpu):
            self.counts[index] += int(count)

    def rows(self, layer: int, head: int, left: str, right: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        width = 2.0 / self.bins
        total = sum(self.counts)
        for index, count in enumerate(self.counts):
            bin_left = -1.0 + index * width
            bin_right = bin_left + width
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "left": left,
                    "right": right,
                    "bin_index": index,
                    "bin_left": bin_left,
                    "bin_right": bin_right,
                    "bin_center": (bin_left + bin_right) / 2.0,
                    "count": count,
                    "frequency": count / total if total else 0.0,
                }
            )
        return rows


class AnalysisContext:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self.vector_stats: dict[tuple[int, int, str], VectorAccumulator] = {}
        self.pair_stats: dict[tuple[int, int, str, str], PairAccumulator] = {}
        self.hist_stats: dict[tuple[int, int, str, str], HistogramAccumulator] = {}
        self.pair_token_rows: list[dict[str, Any]] = []

    def vector_acc(self, layer: int, head: int, name: str) -> VectorAccumulator:
        key = (layer, head, name)
        if key not in self.vector_stats:
            self.vector_stats[key] = VectorAccumulator()
        return self.vector_stats[key]

    def pair_acc(self, layer: int, head: int, left: str, right: str) -> PairAccumulator:
        key = (layer, head, left, right)
        if key not in self.pair_stats:
            self.pair_stats[key] = PairAccumulator()
        return self.pair_stats[key]

    def hist_acc(self, layer: int, head: int, left: str, right: str) -> HistogramAccumulator:
        key = (layer, head, left, right)
        if key not in self.hist_stats:
            self.hist_stats[key] = HistogramAccumulator(self.config.hist_bins)
        return self.hist_stats[key]

    def vector_rows(self, layers: list[int], heads: list[int]) -> list[dict[str, Any]]:
        names = ["full"] + [spec.name for spec in self.config.vector_specs]
        return [self.vector_acc(layer, head, name).row(layer, head, name) for layer in layers for head in heads for name in names]

    def pair_rows(self, layers: list[int], heads: list[int]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in layers:
            for head in heads:
                for left, right in self.config.pair_specs:
                    rows.append(self.pair_acc(layer, head, left, right).row(layer, head, left, right))
        return rows

    def hist_rows(self, layers: list[int], heads: list[int]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in layers:
            for head in heads:
                for left, right in self.config.pair_specs:
                    rows.extend(self.hist_acc(layer, head, left, right).rows(layer, head, left, right))
        return rows


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_float_list(spec: str) -> list[float]:
    values = [float(part.strip()) for part in spec.split(",") if part.strip()]
    if any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("split values must be in (0, 1).")
    return values


def format_value(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def build_vector_specs(top_values: list[float], tail_values: list[float]) -> list[VectorSpec]:
    specs: list[VectorSpec] = []
    for value in top_values:
        specs.append(VectorSpec(f"top{format_value(value)}", "top", value))
    for value in tail_values:
        specs.append(VectorSpec(f"tail{format_value(value)}", "tail", value))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze full/top/tail attention-weighted V outputs by layer/head.")
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
    parser.add_argument("--split_mode", choices=["mass", "token_fraction"], default="mass")
    parser.add_argument("--top_values", default="0.9", help="Comma-separated top mass/fraction values, e.g. 0.5,0.9,0.99.")
    parser.add_argument("--tail_values", default="0.1", help="Comma-separated tail mass/fraction values, e.g. 0.01,0.1.")
    parser.add_argument("--compute_vector_stats", type=str2bool, default=True)
    parser.add_argument(
        "--pairwise_mode",
        choices=["full_vs_all", "top_tail_cross", "all", "custom"],
        default="full_vs_all",
        help="Which vector pairs to aggregate. top_tail_cross compares every top value with every tail value.",
    )
    parser.add_argument(
        "--pairwise_pairs",
        default="",
        help="For pairwise_mode=custom, comma-separated pairs like full|top0p9,top0p9|tail0p1.",
    )
    parser.add_argument("--save_pairwise_per_token", type=str2bool, default=False)
    parser.add_argument("--save_pairwise_hist", type=str2bool, default=False)
    parser.add_argument("--hist_bins", type=int, default=60)
    parser.add_argument("--compute_ppl", type=str2bool, default=True)
    parser.add_argument("--ppl_modes", default="full,top0p9,tail0p1")
    parser.add_argument(
        "--ppl_renormalize_selected",
        type=str2bool,
        default=False,
        help="If true, selected top/tail weights are renormalized before attn@V in PPL runs.",
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive.")
    if args.hist_bins <= 0:
        raise ValueError("--hist_bins must be positive.")
    args.top_value_list = parse_float_list(args.top_values)
    args.tail_value_list = parse_float_list(args.tail_values)
    args.vector_specs = build_vector_specs(args.top_value_list, args.tail_value_list)
    args.pair_specs = build_pair_specs(args.vector_specs, args.pairwise_mode, args.pairwise_pairs)
    return args


def build_pair_specs(vector_specs: list[VectorSpec], pairwise_mode: str, pairwise_pairs: str) -> list[tuple[str, str]]:
    names = ["full"] + [spec.name for spec in vector_specs]
    valid = set(names)
    if pairwise_mode == "full_vs_all":
        return [("full", name) for name in names if name != "full"]
    if pairwise_mode == "top_tail_cross":
        top_names = [spec.name for spec in vector_specs if spec.side == "top"]
        tail_names = [spec.name for spec in vector_specs if spec.side == "tail"]
        return [(top_name, tail_name) for top_name in top_names for tail_name in tail_names]
    if pairwise_mode == "all":
        return list(combinations(names, 2))
    pairs: list[tuple[str, str]] = []
    for raw_pair in pairwise_pairs.split(","):
        raw_pair = raw_pair.strip()
        if not raw_pair:
            continue
        if "|" not in raw_pair:
            raise ValueError(f"Invalid pair spec `{raw_pair}`. Expected left|right.")
        left, right = [part.strip() for part in raw_pair.split("|", 1)]
        left = normalize_mode_name(left)
        right = normalize_mode_name(right)
        if left not in valid or right not in valid:
            raise ValueError(f"Invalid pair `{raw_pair}`. Valid vectors: {sorted(valid)}")
        if left == right:
            raise ValueError(f"Invalid pair `{raw_pair}`: left and right are identical.")
        pairs.append((left, right))
    if not pairs:
        raise ValueError("--pairwise_mode=custom requires --pairwise_pairs.")
    return pairs


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


def normalize_mode_name(mode: str) -> str:
    mode = mode.strip().lower()
    if mode == "top90":
        return "top0p9"
    if mode == "tail10":
        return "tail0p1"
    return mode


def parse_modes(spec: str, vector_specs: list[VectorSpec]) -> list[str]:
    modes = [normalize_mode_name(part) for part in spec.split(",") if part.strip()]
    valid = {"full"} | {vector_spec.name for vector_spec in vector_specs}
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


def selection_mask(attention_weights: torch.Tensor, spec: VectorSpec, split_mode: str) -> torch.Tensor:
    weights = attention_weights.float()
    descending = spec.side == "top"
    sorted_weights, sorted_indices = torch.sort(weights, dim=-1, descending=descending)
    valid_sorted = sorted_weights > 0
    rank = torch.arange(weights.shape[-1], device=weights.device).view(1, 1, 1, -1)
    valid_counts = valid_sorted.sum(dim=-1, keepdim=True).clamp_min(1)
    if split_mode == "mass":
        cumulative = torch.cumsum(sorted_weights, dim=-1)
        first_crossing = (cumulative >= spec.value).float().argmax(dim=-1, keepdim=True)
        sorted_mask = (rank <= first_crossing) & valid_sorted
    elif split_mode == "token_fraction":
        keep = torch.ceil(valid_counts.float() * spec.value).long().clamp_min(1)
        sorted_mask = (rank < keep) & valid_sorted
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")
    return torch.zeros_like(sorted_mask, dtype=torch.bool).scatter(dim=-1, index=sorted_indices, src=sorted_mask)


def vector_outputs(
    attention_weights: torch.Tensor,
    value_states_for_attention: torch.Tensor,
    specs: list[VectorSpec],
    split_mode: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    weights = attention_weights.float()
    values = value_states_for_attention.float()
    outputs = {"full": torch.matmul(weights, values)}
    masses = {"full": weights.sum(dim=-1)}
    counts = {"full": (weights > 0).sum(dim=-1).float()}
    for spec in specs:
        mask = selection_mask(weights, spec, split_mode)
        selected_weights = weights.masked_fill(~mask, 0.0)
        outputs[spec.name] = torch.matmul(selected_weights, values)
        masses[spec.name] = selected_weights.sum(dim=-1)
        counts[spec.name] = mask.sum(dim=-1).float()
    return outputs, masses, counts


def selected_attention_output(
    attention_weights: torch.Tensor,
    value_states_for_attention: torch.Tensor,
    mode: str,
    config: AnalysisConfig,
) -> torch.Tensor:
    outputs, masses, _ = vector_outputs(attention_weights, value_states_for_attention, config.vector_specs, config.split_mode)
    if mode not in outputs:
        raise ValueError(f"Unsupported attention output mode: {mode}")
    output = outputs[mode]
    if config.ppl_renormalize_selected and mode != "full":
        output = output / masses[mode].clamp_min(1e-12).unsqueeze(-1)
    return output


def crop_past_key_values(past_key_values: Any, max_length: int) -> Any:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        result = past_key_values.crop(max_length)
        return past_key_values if result is None else result
    if isinstance(past_key_values, tuple):
        return tuple(
            tuple(tensor[..., :max_length, :] for tensor in layer_cache) if isinstance(layer_cache, tuple) else layer_cache
            for layer_cache in past_key_values
        )
    if isinstance(past_key_values, list):
        return [
            tuple(tensor[..., :max_length, :] for tensor in layer_cache) if isinstance(layer_cache, tuple) else layer_cache
            for layer_cache in past_key_values
        ]
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
    outputs, masses, counts = vector_outputs(attention_weights, value_states_for_attention, config.vector_specs, config.split_mode)
    names = list(outputs.keys())
    for head in sorted(config.heads):
        if head >= attention_weights.shape[1]:
            continue
        for name in names:
            context.vector_acc(layer, head, name).update(
                outputs[name][:, head].detach(),
                masses[name][:, head].detach(),
                counts[name][:, head].detach(),
            )
        for left, right in config.pair_specs:
            left_values = outputs[left][:, head].detach()
            right_values = outputs[right][:, head].detach()
            context.pair_acc(layer, head, left, right).update(
                left_values,
                right_values,
            )
            left_flat = None
            right_flat = None
            cos_values = None
            if config.save_pairwise_hist or config.save_pairwise_per_token:
                left_flat = left_values.float().reshape(-1, left_values.shape[-1])
                right_flat = right_values.float().reshape(-1, right_values.shape[-1])
                cos_values = F.cosine_similarity(left_flat, right_flat, dim=-1, eps=1e-8)
            if config.save_pairwise_hist and cos_values is not None:
                context.hist_acc(layer, head, left, right).update(cos_values)
            if config.save_pairwise_per_token:
                if left_flat is None or right_flat is None or cos_values is None:
                    left_flat = left_values.float().reshape(-1, left_values.shape[-1])
                    right_flat = right_values.float().reshape(-1, right_values.shape[-1])
                    cos_values = F.cosine_similarity(left_flat, right_flat, dim=-1, eps=1e-8)
                cos_values_cpu = cos_values.detach().cpu()
                l2_values = torch.linalg.vector_norm(left_flat - right_flat, dim=-1).detach().cpu()
                for local_query, (cos_value, l2_value) in enumerate(zip(cos_values_cpu.tolist(), l2_values.tolist())):
                    context.pair_token_rows.append(
                        {
                            "layer": layer,
                            "head": head,
                            "query_index": config.query_offset + local_query,
                            "left": left,
                            "right": right,
                            "cosine": cos_value,
                            "l2": l2_value,
                        }
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
        selected_output = selected_attention_output(attention_weights, value_states_for_attention, context.config.attention_output_mode, context.config)
        active_heads = [head for head in context.config.heads if 0 <= head < attention_output.shape[1]]
        if active_heads:
            head_index = torch.tensor(active_heads, dtype=torch.long, device=attention_output.device)
            attention_output[:, head_index] = selected_output[:, head_index].to(attention_output.dtype)
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
            if context is not None:
                context.config.query_offset = start - prefill_tokens
            outputs = model_forward(model, kwargs)
            logits = outputs.logits
            shifted_logits = torch.cat([prev_logits.unsqueeze(1), logits[:, :-1, :]], dim=1)
            loss = F.cross_entropy(shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(), chunk.reshape(-1), reduction="sum")
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
    ppl_modes = parse_modes(args.ppl_modes, args.vector_specs)

    base_config = AnalysisConfig(
        layers=set(layers),
        heads=set(heads),
        split_mode=args.split_mode,
        vector_specs=args.vector_specs,
        collect_vectors=False,
        attention_output_mode="full",
        ppl_renormalize_selected=args.ppl_renormalize_selected,
        save_pairwise_per_token=args.save_pairwise_per_token,
        save_pairwise_hist=args.save_pairwise_hist,
        hist_bins=args.hist_bins,
        pair_specs=args.pair_specs,
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
            model, input_ids, args.prefill_tokens, args.eval_tokens, args.chunk_size, input_device, past_key_values, prev_logits, stats_context
        )
        past_key_values = crop_past_key_values(past_key_values, args.prefill_tokens)
        write_csv(
            output_dir / "value_vectors_by_head.csv",
            stats_context.vector_rows(layers, heads),
            ["layer", "head", "vector", "query_count", "mean_norm", "mean_attention_mass", "mean_token_count"],
        )
        write_csv(
            output_dir / "value_pairwise_by_head.csv",
            stats_context.pair_rows(layers, heads),
            ["layer", "head", "left", "right", "query_count", "mean_cosine", "mean_l2"],
        )
        if args.save_pairwise_per_token:
            write_csv(
                output_dir / "value_pairwise_per_token.csv",
                stats_context.pair_token_rows,
                ["layer", "head", "query_index", "left", "right", "cosine", "l2"],
            )
        if args.save_pairwise_hist:
            write_csv(
                output_dir / "value_pairwise_hist_by_head.csv",
                stats_context.hist_rows(layers, heads),
                ["layer", "head", "left", "right", "bin_index", "bin_left", "bin_right", "bin_center", "count", "frequency"],
            )
        if args.compute_ppl and "full" in ppl_modes:
            ppl_rows.append({"mode": "full", "loss": loss, "ppl": ppl, "token_count": count})

    if args.compute_ppl:
        existing_modes = {row["mode"] for row in ppl_rows}
        for mode in ppl_modes:
            if mode in existing_modes:
                continue
            print(f"running PPL mode={mode}", flush=True)
            mode_config = AnalysisConfig(**{**base_config.__dict__, "attention_output_mode": mode})
            past_key_values = crop_past_key_values(past_key_values, args.prefill_tokens)
            loss, ppl, count, past_key_values = run_eval(
                model, input_ids, args.prefill_tokens, args.eval_tokens, args.chunk_size, input_device, past_key_values, prev_logits, AnalysisContext(mode_config)
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
        "split_mode": args.split_mode,
        "top_values": args.top_value_list,
        "tail_values": args.tail_value_list,
        "ppl_modes": ppl_modes,
        "pairwise_mode": args.pairwise_mode,
        "pairwise_pairs": args.pair_specs,
        "ppl_renormalize_selected": args.ppl_renormalize_selected,
        "save_pairwise_per_token": args.save_pairwise_per_token,
        "save_pairwise_hist": args.save_pairwise_hist,
        "hist_bins": args.hist_bins,
        "outputs": {
            "value_vectors_by_head": str(output_dir / "value_vectors_by_head.csv"),
            "value_pairwise_by_head": str(output_dir / "value_pairwise_by_head.csv"),
            "value_pairwise_per_token": str(output_dir / "value_pairwise_per_token.csv") if args.save_pairwise_per_token else None,
            "value_pairwise_hist_by_head": str(output_dir / "value_pairwise_hist_by_head.csv") if args.save_pairwise_hist else None,
            "ppl_by_attention_value_mode": str(output_dir / "ppl_by_attention_value_mode.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
