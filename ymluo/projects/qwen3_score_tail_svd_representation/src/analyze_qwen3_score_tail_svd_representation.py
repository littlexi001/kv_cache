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
GROUPS = ("score_top_1pct", "score_top_90pct", "score_tail_10pct")

_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_MODULE_TO_LAYER: dict[int, int] = {}
_ACTIVE_CONTEXT: AnalysisContext | None = None


@dataclass
class AnalysisConfig:
    layers: set[int]
    heads: set[int]
    representations: set[str]
    svd_components: int
    query_stride: int
    max_query_rows_per_layer_head: int
    max_vectors_per_group: int
    max_score_values_per_group: int


class ScoreAccumulator:
    def __init__(self, max_values: int) -> None:
        self.max_values = max_values
        self.count = 0
        self.score_sum = 0.0
        self.score_sq_sum = 0.0
        self.weight_sum = 0.0
        self.weight_sq_sum = 0.0
        self.values: list[tuple[float, float]] = []
        self.token_count_sum = 0.0
        self.query_count = 0

    def update(self, scores: torch.Tensor, weights: torch.Tensor, token_count: int) -> None:
        if scores.numel() == 0:
            return
        scores = scores.detach().float().cpu()
        weights = weights.detach().float().cpu()
        count = int(scores.numel())
        self.count += count
        self.query_count += 1
        self.token_count_sum += float(token_count)
        self.score_sum += float(scores.sum())
        self.score_sq_sum += float((scores * scores).sum())
        self.weight_sum += float(weights.sum())
        self.weight_sq_sum += float((weights * weights).sum())
        remaining = self.max_values - len(self.values)
        if remaining > 0:
            for score, weight in zip(scores[:remaining].tolist(), weights[:remaining].tolist()):
                self.values.append((float(score), float(weight)))

    def row(self, layer: int, head: int, group: str) -> dict[str, Any]:
        score_values = torch.tensor([item[0] for item in self.values], dtype=torch.float32)
        weight_values = torch.tensor([item[1] for item in self.values], dtype=torch.float32)
        score_mean = self.score_sum / max(1, self.count)
        weight_mean = self.weight_sum / max(1, self.count)
        score_var = max(0.0, self.score_sq_sum / max(1, self.count) - score_mean * score_mean)
        weight_var = max(0.0, self.weight_sq_sum / max(1, self.count) - weight_mean * weight_mean)
        row: dict[str, Any] = {
            "layer": layer,
            "head": head,
            "group": group,
            "token_count": self.count,
            "query_count": self.query_count,
            "mean_selected_tokens_per_query": self.token_count_sum / max(1, self.query_count),
            "score_mean": score_mean,
            "score_std": math.sqrt(score_var),
            "weight_mean": weight_mean,
            "weight_std": math.sqrt(weight_var),
            "sampled_value_count": len(self.values),
        }
        for prefix, values in (("score", score_values), ("weight", weight_values)):
            if values.numel():
                row[f"{prefix}_min"] = float(values.min())
                row[f"{prefix}_p10"] = float(torch.quantile(values, 0.10))
                row[f"{prefix}_p25"] = float(torch.quantile(values, 0.25))
                row[f"{prefix}_p50"] = float(torch.quantile(values, 0.50))
                row[f"{prefix}_p75"] = float(torch.quantile(values, 0.75))
                row[f"{prefix}_p90"] = float(torch.quantile(values, 0.90))
                row[f"{prefix}_p99"] = float(torch.quantile(values, 0.99))
                row[f"{prefix}_max"] = float(values.max())
            else:
                for suffix in ("min", "p10", "p25", "p50", "p75", "p90", "p99", "max"):
                    row[f"{prefix}_{suffix}"] = 0.0
        return row


class VectorStore:
    def __init__(self, max_vectors: int) -> None:
        self.max_vectors = max_vectors
        self.total_seen = 0
        self.chunks: list[torch.Tensor] = []
        self.stored = 0

    def add(self, vectors: torch.Tensor) -> None:
        if vectors.numel() == 0:
            return
        vectors = vectors.detach().float().cpu().reshape(-1, vectors.shape[-1])
        self.total_seen += int(vectors.shape[0])
        remaining = self.max_vectors - self.stored
        if remaining <= 0:
            return
        vectors = vectors[:remaining].contiguous()
        self.chunks.append(vectors)
        self.stored += int(vectors.shape[0])

    def tensor(self) -> torch.Tensor:
        if not self.chunks:
            return torch.empty(0, 0, dtype=torch.float32)
        return torch.cat(self.chunks, dim=0)


class AnalysisContext:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self.query_counts: dict[tuple[int, int], int] = {}
        self.score_stats: dict[tuple[int, int, str], ScoreAccumulator] = {}
        self.vectors: dict[tuple[int, int, str, str], VectorStore] = {}

    def should_collect_query(self, layer: int, head: int) -> bool:
        key = (layer, head)
        seen = self.query_counts.get(key, 0)
        if seen >= self.config.max_query_rows_per_layer_head:
            return False
        self.query_counts[key] = seen + 1
        return True

    def score_accumulator(self, layer: int, head: int, group: str) -> ScoreAccumulator:
        key = (layer, head, group)
        if key not in self.score_stats:
            self.score_stats[key] = ScoreAccumulator(self.config.max_score_values_per_group)
        return self.score_stats[key]

    def vector_store(self, layer: int, head: int, representation: str, group: str) -> VectorStore:
        key = (layer, head, representation, group)
        if key not in self.vectors:
            self.vectors[key] = VectorStore(self.config.max_vectors_per_group)
        return self.vectors[key]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze score-ranked top90/tail10 token representations with SVD.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/score_tail_svd_representation")
    parser.add_argument("--prefill_tokens", type=int, default=5000)
    parser.add_argument("--eval_tokens", type=int, default=1024)
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
    parser.add_argument("--representations", default="key,value,weighted_value")
    parser.add_argument("--svd_components", type=int, default=8)
    parser.add_argument("--query_stride", type=int, default=8)
    parser.add_argument("--max_query_rows_per_layer_head", type=int, default=512)
    parser.add_argument("--max_vectors_per_group", type=int, default=4096)
    parser.add_argument("--max_score_values_per_group", type=int, default=200_000)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive.")
    if args.svd_components <= 0:
        raise ValueError("--svd_components must be positive.")
    if args.query_stride <= 0:
        raise ValueError("--query_stride must be positive.")
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


def parse_representations(spec: str) -> set[str]:
    selected = {part.strip().lower() for part in spec.split(",") if part.strip()}
    valid = {"key", "value", "weighted_value"}
    invalid = sorted(selected - valid)
    if invalid:
        raise ValueError(f"Invalid representations {invalid}. Valid: {sorted(valid)}")
    return selected


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


def repeat_kv_for_attention(states: torch.Tensor, attention_heads: int) -> torch.Tensor:
    if states.shape[1] == attention_heads:
        return states
    group_size = attention_heads // states.shape[1]
    return states.repeat_interleave(group_size, dim=1)


def group_indices(scores: torch.Tensor) -> dict[str, torch.Tensor]:
    valid_count = int(scores.numel())
    if valid_count <= 0:
        empty = torch.empty(0, dtype=torch.long, device=scores.device)
        return {group: empty for group in GROUPS}
    order_desc = torch.argsort(scores, descending=True)
    top1_count = max(1, math.ceil(valid_count * 0.01))
    tail10_count = max(1, math.ceil(valid_count * 0.10))
    top90_count = max(1, valid_count - tail10_count)
    return {
        "score_top_1pct": order_desc[:top1_count],
        "score_top_90pct": order_desc[:top90_count],
        "score_tail_10pct": order_desc[-tail10_count:],
    }


def collect_score_svd_samples(
    context: AnalysisContext,
    layer: int,
    scores: torch.Tensor,
    attention_weights: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    config = context.config
    if layer not in config.layers:
        return
    batch_count, head_count, query_count, _ = scores.shape
    if batch_count != 1:
        raise ValueError("This analysis currently expects batch size 1.")
    query_indices = list(range(0, query_count, config.query_stride))
    for head in sorted(config.heads):
        if head >= head_count:
            continue
        for query_index in query_indices:
            if not context.should_collect_query(layer, head):
                break
            row_scores = scores[0, head, query_index].float()
            row_weights = attention_weights[0, head, query_index].float()
            valid_mask = row_weights > 0
            valid_positions = torch.nonzero(valid_mask, as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue
            valid_scores = row_scores[valid_positions]
            valid_weights = row_weights[valid_positions]
            selected_by_group = group_indices(valid_scores)
            for group, local_indices in selected_by_group.items():
                key_indices = valid_positions[local_indices]
                selected_scores = valid_scores[local_indices]
                selected_weights = valid_weights[local_indices]
                context.score_accumulator(layer, head, group).update(
                    selected_scores,
                    selected_weights,
                    int(local_indices.numel()),
                )
                if "key" in config.representations:
                    context.vector_store(layer, head, "key", group).add(key_states[0, head, key_indices])
                if "value" in config.representations:
                    context.vector_store(layer, head, "value", group).add(value_states[0, head, key_indices])
                if "weighted_value" in config.representations:
                    weighted = value_states[0, head, key_indices].float() * selected_weights.unsqueeze(-1)
                    context.vector_store(layer, head, "weighted_value", group).add(weighted)


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
    key_states_for_attention = repeat_kv_for_attention(key_states, attention_heads)
    value_states_for_attention = repeat_kv_for_attention(value_states, attention_heads)
    scores = torch.matmul(query_states, key_states_for_attention.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=True)
    layer = _MODULE_TO_LAYER.get(id(module), -1)
    if context is not None:
        collect_score_svd_samples(
            context,
            layer,
            scores,
            attention_weights,
            key_states_for_attention,
            value_states_for_attention,
        )
    attention_output = torch.matmul(attention_weights, value_states_for_attention)
    return attention_output.transpose(1, 2).contiguous(), attention_weights


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
def run_eval_collect(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    past_key_values: Any,
    context: AnalysisContext,
) -> None:
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
            print(f"collect chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
            outputs = model_forward(model, kwargs)
            past_key_values = outputs.past_key_values
            del outputs, chunk


def tensor_or_empty(context: AnalysisContext, layer: int, head: int, representation: str, group: str) -> torch.Tensor:
    store = context.vectors.get((layer, head, representation, group))
    if store is None:
        return torch.empty(0, 0, dtype=torch.float32)
    return store.tensor()


def projection_rows(
    context: AnalysisContext,
    layers: list[int],
    heads: list[int],
    representations: set[str],
    svd_components: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    projection: list[dict[str, Any]] = []
    centroid: list[dict[str, Any]] = []
    energy: list[dict[str, Any]] = []
    for layer in layers:
        for head in heads:
            for representation in sorted(representations):
                group_tensors = {
                    group: tensor_or_empty(context, layer, head, representation, group)
                    for group in GROUPS
                }
                tensors = [tensor for tensor in group_tensors.values() if tensor.numel()]
                if not tensors:
                    continue
                matrix = torch.cat(tensors, dim=0)
                if matrix.shape[0] < 2:
                    continue
                centered = matrix - matrix.mean(dim=0, keepdim=True)
                _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
                component_count = min(svd_components, int(vh.shape[0]))
                basis = vh[:component_count].T.contiguous()
                total_energy = float((singular_values * singular_values).sum())
                for component, singular_value in enumerate(singular_values[:component_count].tolist(), start=1):
                    value = float(singular_value)
                    energy.append(
                        {
                            "layer": layer,
                            "head": head,
                            "representation": representation,
                            "component": component,
                            "singular_value": value,
                            "energy_ratio": (value * value) / max(total_energy, 1e-12),
                        }
                    )
                centroids: dict[str, torch.Tensor] = {}
                for group, tensor in group_tensors.items():
                    if tensor.numel() == 0:
                        continue
                    group_centered = tensor - matrix.mean(dim=0, keepdim=True)
                    projected = group_centered @ basis
                    centroids[group] = tensor.mean(dim=0)
                    row: dict[str, Any] = {
                        "layer": layer,
                        "head": head,
                        "representation": representation,
                        "group": group,
                        "vector_count": int(tensor.shape[0]),
                        "total_seen": context.vectors[(layer, head, representation, group)].total_seen,
                        "mean_vector_norm": float(torch.linalg.vector_norm(tensor, dim=-1).mean()),
                        "projection_norm_mean": float(torch.linalg.vector_norm(projected, dim=-1).mean()),
                    }
                    group_energy = (projected * projected).sum(dim=-1).clamp_min(1e-12)
                    for component in range(component_count):
                        component_values = projected[:, component]
                        row[f"pc{component + 1}_mean"] = float(component_values.mean())
                        row[f"pc{component + 1}_std"] = float(component_values.std(unbiased=False))
                        row[f"pc{component + 1}_energy_ratio"] = float(
                            ((component_values * component_values) / group_energy).mean()
                        )
                    projection.append(row)
                for left in GROUPS:
                    for right in GROUPS:
                        if left >= right or left not in centroids or right not in centroids:
                            continue
                        left_c = centroids[left]
                        right_c = centroids[right]
                        centroid.append(
                            {
                                "layer": layer,
                                "head": head,
                                "representation": representation,
                                "left_group": left,
                                "right_group": right,
                                "centroid_cosine": float(F.cosine_similarity(left_c, right_c, dim=0, eps=1e-8)),
                                "centroid_l2": float(torch.linalg.vector_norm(left_c - right_c)),
                                "left_centroid_norm": float(torch.linalg.vector_norm(left_c)),
                                "right_centroid_norm": float(torch.linalg.vector_norm(right_c)),
                            }
                        )
    return projection, centroid, energy


def score_rows(context: AnalysisContext, layers: list[int], heads: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in layers:
        for head in heads:
            for group in GROUPS:
                accumulator = context.score_stats.get((layer, head, group))
                if accumulator is None:
                    accumulator = ScoreAccumulator(context.config.max_score_values_per_group)
                rows.append(accumulator.row(layer, head, group))
    return rows


def score_fields() -> list[str]:
    return [
        "layer",
        "head",
        "group",
        "token_count",
        "query_count",
        "mean_selected_tokens_per_query",
        "score_mean",
        "score_std",
        "score_min",
        "score_p10",
        "score_p25",
        "score_p50",
        "score_p75",
        "score_p90",
        "score_p99",
        "score_max",
        "weight_mean",
        "weight_std",
        "weight_min",
        "weight_p10",
        "weight_p25",
        "weight_p50",
        "weight_p75",
        "weight_p90",
        "weight_p99",
        "weight_max",
        "sampled_value_count",
    ]


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
    representations = parse_representations(args.representations)
    config = AnalysisConfig(
        layers=set(layers),
        heads=set(heads),
        representations=representations,
        svd_components=args.svd_components,
        query_stride=args.query_stride,
        max_query_rows_per_layer_head=args.max_query_rows_per_layer_head,
        max_vectors_per_group=args.max_vectors_per_group,
        max_score_values_per_group=args.max_score_values_per_group,
    )
    context = AnalysisContext(config)

    print("running shared full-attention prefill", flush=True)
    past_key_values, _ = run_prefill(model, input_ids, args.prefill_tokens, args.chunk_size, input_device)
    print("collecting score-ranked representation samples", flush=True)
    run_eval_collect(
        model,
        input_ids,
        args.prefill_tokens,
        args.eval_tokens,
        args.chunk_size,
        input_device,
        past_key_values,
        context,
    )

    all_score_rows = score_rows(context, layers, heads)
    write_csv(output_dir / "score_distribution_by_group.csv", all_score_rows, score_fields())
    write_csv(
        output_dir / "score_top_90pct_distribution.csv",
        [row for row in all_score_rows if row["group"] == "score_top_90pct"],
        score_fields(),
    )

    projection, centroid, energy = projection_rows(context, layers, heads, representations, args.svd_components)
    projection_fields = sorted({field for row in projection for field in row})
    centroid_fields = [
        "layer",
        "head",
        "representation",
        "left_group",
        "right_group",
        "centroid_cosine",
        "centroid_l2",
        "left_centroid_norm",
        "right_centroid_norm",
    ]
    energy_fields = ["layer", "head", "representation", "component", "singular_value", "energy_ratio"]
    write_csv(output_dir / "svd_projection_by_group.csv", projection, projection_fields)
    write_csv(output_dir / "centroid_similarity_by_group.csv", centroid, centroid_fields)
    write_csv(output_dir / "singular_value_energy.csv", energy, energy_fields)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "text_path": args.text_path,
        "prefill_tokens": args.prefill_tokens,
        "eval_tokens": args.eval_tokens,
        "chunk_size": args.chunk_size,
        "layers": layers,
        "heads": heads,
        "representations": sorted(representations),
        "svd_components": args.svd_components,
        "query_stride": args.query_stride,
        "max_query_rows_per_layer_head": args.max_query_rows_per_layer_head,
        "max_vectors_per_group": args.max_vectors_per_group,
        "max_score_values_per_group": args.max_score_values_per_group,
        "outputs": {
            "svd_projection_by_group": str(output_dir / "svd_projection_by_group.csv"),
            "centroid_similarity_by_group": str(output_dir / "centroid_similarity_by_group.csv"),
            "score_distribution_by_group": str(output_dir / "score_distribution_by_group.csv"),
            "score_top_90pct_distribution": str(output_dir / "score_top_90pct_distribution.csv"),
            "singular_value_energy": str(output_dir / "singular_value_energy.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
