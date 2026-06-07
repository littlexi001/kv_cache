from __future__ import annotations

import argparse
import csv
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
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

_GRAPH_CONFIG: GraphMaskConfig | None = None
_GRAPH_STATE: GraphRuntimeState | None = None
_GRAPH_ENABLED = False
_MODULE_TO_LAYER: dict[int, int] = {}
_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None


@dataclass(frozen=True)
class GraphMaskConfig:
    layers: set[int]
    kv_heads: set[int]
    attention_heads: int
    kv_head_count: int
    boundary_fraction: float
    middle_fraction: float
    seed_count: int
    graph_degree: int
    graph_update_interval: int
    graph_update_mode: str
    max_hops: int
    candidate_granularity: str
    compute_overlap_metrics: bool
    position_bins: int


@dataclass
class HeadStats:
    query_count: int = 0
    candidate_sum: float = 0.0
    candidate_p95_values: list[float] = field(default_factory=list)
    overlap_count_sum: float = 0.0
    overlap_ratio_sum: float = 0.0
    retrieval_energy_sum: float = 0.0
    oracle_energy_sum: float = 0.0
    prefix_overlap_sum: float = 0.0
    middle_overlap_sum: float = 0.0
    recent_overlap_sum: float = 0.0
    overlap_position_bins: list[int] = field(default_factory=list)


@dataclass
class GraphRuntimeState:
    built_until: dict[tuple[int, int], int] = field(default_factory=dict)
    neighbors: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    prev_indices: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    prev_weights: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    stats: dict[tuple[int, int, int], HeadStats] = field(default_factory=dict)
    timing_rows: list[dict[str, Any]] = field(default_factory=list)

    def reset_sequence(self) -> None:
        self.built_until.clear()
        self.neighbors.clear()
        self.prev_indices.clear()
        self.prev_weights.clear()

    def reset_all(self) -> None:
        self.reset_sequence()
        self.stats.clear()
        self.timing_rows.clear()


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 PPL with incremental sparse K-K L2 graph retrieval.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/kkl2_graph_ppl")
    parser.add_argument("--prefill_tokens", type=int, default=2048)
    parser.add_argument("--eval_tokens", type=int, default=2048)
    parser.add_argument("--eval_last_tokens_only", type=str2bool, default=False)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--compute_graph_ppl", type=str2bool, default=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--kv_heads", default="all")
    parser.add_argument("--boundary_fraction", type=float, default=0.005)
    parser.add_argument("--middle_fraction", type=float, default=0.01)
    parser.add_argument("--seed_count", type=int, default=16)
    parser.add_argument("--graph_degree", type=int, default=20)
    parser.add_argument("--graph_update_interval", type=int, default=100)
    parser.add_argument("--graph_update_mode", choices=["block_previous", "full_rebuild"], default="block_previous")
    parser.add_argument("--max_hops", type=int, default=2)
    parser.add_argument("--candidate_granularity", choices=["attention_head", "kv_head_union"], default="kv_head_union")
    parser.add_argument("--compute_overlap_metrics", type=str2bool, default=True)
    parser.add_argument("--position_bins", type=int, default=20)
    return parser.parse_args()


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
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid {name} range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    invalid = sorted(index for index in selected if index < 0 or index >= max_count)
    if invalid:
        raise ValueError(f"{name} out of range 0..{max_count - 1}: {invalid}")
    return sorted(selected)


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


def model_body(model: torch.nn.Module) -> torch.nn.Module:
    for attr_name in ("model", "transformer"):
        if hasattr(model, attr_name):
            return getattr(model, attr_name)
    return model


def register_attention_layers(model: torch.nn.Module) -> None:
    _MODULE_TO_LAYER.clear()
    body = model_body(model)
    layers = getattr(body, "layers", None) or getattr(body, "h", None)
    if layers is None:
        raise AttributeError("Could not find transformer layers on model.")
    for layer_idx, layer_obj in enumerate(layers):
        attn = getattr(layer_obj, "self_attn", None) or getattr(layer_obj, "attention", None)
        if attn is not None:
            _MODULE_TO_LAYER[id(attn)] = layer_idx


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def l2_topk_indices(query: torch.Tensor, keys: torch.Tensor, k: int, exclude_self_offset: int | None = None) -> torch.Tensor:
    if keys.numel() == 0 or k <= 0:
        return torch.empty((query.shape[0], 0), dtype=torch.long, device=query.device)
    query_f = query.float()
    keys_f = keys.float()
    scores = 2.0 * torch.matmul(query_f, keys_f.transpose(0, 1)) - keys_f.square().sum(dim=-1).view(1, -1)
    if exclude_self_offset is not None:
        row_count = query.shape[0]
        cols = exclude_self_offset + torch.arange(row_count, device=query.device)
        valid = cols < scores.shape[1]
        scores[torch.arange(row_count, device=query.device)[valid], cols[valid]] = -float("inf")
    topk = min(k, scores.shape[-1])
    return torch.topk(scores, k=topk, dim=-1, largest=True).indices


def pad_index_columns(indices: torch.Tensor, width: int) -> torch.Tensor:
    if indices.shape[1] == width:
        return indices
    if indices.shape[1] > width:
        return indices[:, :width]
    pad_width = width - indices.shape[1]
    pad = torch.zeros((indices.shape[0], pad_width), dtype=indices.dtype, device=indices.device)
    return torch.cat([indices, pad], dim=1)


def rebuild_full_graph(key_vectors: torch.Tensor, degree: int) -> torch.Tensor:
    token_count = int(key_vectors.shape[0])
    if token_count <= 1:
        return torch.zeros((token_count, degree), dtype=torch.long, device=key_vectors.device)
    indices = l2_topk_indices(key_vectors, key_vectors, min(degree, token_count - 1), exclude_self_offset=0)
    return pad_index_columns(indices, degree)


def update_graph_for_head(
    state: GraphRuntimeState,
    config: GraphMaskConfig,
    layer: int,
    kv_head: int,
    key_vectors: torch.Tensor,
) -> None:
    key = (layer, kv_head)
    key_count = int(key_vectors.shape[0])
    target = (key_count // config.graph_update_interval) * config.graph_update_interval
    previous = state.built_until.get(key, 0)
    if target <= previous:
        return
    start_time = time.perf_counter()
    if config.graph_update_mode == "full_rebuild":
        graph = rebuild_full_graph(key_vectors[:target], config.graph_degree)
        state.neighbors[key] = graph
        state.built_until[key] = target
        updated_rows = target
    else:
        old_graph = state.neighbors.get(key)
        rows: list[torch.Tensor] = []
        if old_graph is not None and old_graph.numel() > 0:
            rows.append(old_graph)
        for start in range(previous, target, config.graph_update_interval):
            end = min(start + config.graph_update_interval, target)
            query = key_vectors[start:end]
            history = key_vectors[:end]
            idx = l2_topk_indices(query, history, min(config.graph_degree, max(1, end - 1)), exclude_self_offset=start)
            rows.append(pad_index_columns(idx, config.graph_degree))
        state.neighbors[key] = torch.cat(rows, dim=0) if rows else torch.empty((0, config.graph_degree), dtype=torch.long, device=key_vectors.device)
        state.built_until[key] = target
        updated_rows = target - previous
    state.timing_rows.append(
        {
            "event": "graph_update",
            "layer": layer,
            "kv_head": kv_head,
            "key_count": key_count,
            "built_until": target,
            "updated_rows": updated_rows,
            "seconds": time.perf_counter() - start_time,
            "mode": config.graph_update_mode,
        }
    )


def boundary_counts(visible: int, boundary_fraction: float) -> tuple[int, int, int]:
    boundary = max(1, math.ceil(boundary_fraction * visible))
    prefix_end = min(boundary, visible)
    recent_start = max(0, visible - boundary)
    return boundary, prefix_end, recent_start


def previous_seed_indices(
    state: GraphRuntimeState,
    layer: int,
    attention_head: int,
    visible: int,
    middle_start: int,
    middle_end: int,
    seed_count: int,
) -> list[int]:
    key = (layer, attention_head)
    indices = state.prev_indices.get(key)
    weights = state.prev_weights.get(key)
    if indices is None or weights is None or indices.numel() == 0 or middle_end <= middle_start:
        return []
    valid = (indices >= middle_start) & (indices < middle_end) & (indices < visible)
    if not bool(valid.any()):
        return []
    pool_indices = indices[valid]
    pool_weights = weights[valid]
    count = min(seed_count, int(pool_indices.numel()))
    selected = torch.topk(pool_weights.float(), k=count, largest=True).indices
    return [int(value) for value in pool_indices[selected].tolist()]


def expand_graph_candidates(
    graph: torch.Tensor | None,
    seeds: list[int],
    visible: int,
    middle_start: int,
    middle_end: int,
    max_hops: int,
) -> list[int]:
    if graph is None or graph.numel() == 0 or not seeds or max_hops <= 0:
        return []
    visited: set[int] = set()
    frontier = [seed for seed in seeds if middle_start <= seed < middle_end and seed < visible]
    for seed in frontier:
        visited.add(seed)
    built_until = int(graph.shape[0])
    for _ in range(max_hops):
        next_frontier: list[int] = []
        for node in frontier:
            if node < 0 or node >= built_until:
                continue
            for neighbor in graph[node].tolist():
                idx = int(neighbor)
                if middle_start <= idx < middle_end and idx < visible and idx not in visited:
                    visited.add(idx)
                    next_frontier.append(idx)
        frontier = next_frontier
        if not frontier:
            break
    return sorted(visited)


def candidate_indices_for_head_group(
    state: GraphRuntimeState,
    config: GraphMaskConfig,
    layer: int,
    kv_head: int,
    attention_heads: list[int],
    query_token: int,
) -> tuple[list[int], set[int], set[int], set[int]]:
    visible = query_token + 1
    boundary, prefix_end, recent_start = boundary_counts(visible, config.boundary_fraction)
    prefix = set(range(prefix_end))
    recent = set(range(recent_start, visible))
    middle_start = prefix_end
    middle_end = recent_start
    seeds: list[int] = []
    for attention_head in attention_heads:
        seeds.extend(
            previous_seed_indices(
                state,
                layer,
                attention_head,
                visible,
                middle_start,
                middle_end,
                config.seed_count,
            )
        )
    graph = state.neighbors.get((layer, kv_head))
    expanded = expand_graph_candidates(graph, seeds, visible, middle_start, middle_end, config.max_hops)
    middle_budget = max(1, math.ceil(config.middle_fraction * visible))
    if len(expanded) > middle_budget:
        seed_set = set(seeds)
        expanded = sorted(expanded, key=lambda idx: (0 if idx in seed_set else 1, abs(query_token - idx)))[:middle_budget]
    candidates = sorted((prefix | recent | set(expanded)) & set(range(visible)))
    return candidates, prefix, recent, set(expanded)


def build_candidate_tensor(
    state: GraphRuntimeState,
    config: GraphMaskConfig,
    layer: int,
    key_count: int,
    query_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[tuple[int, int], tuple[set[int], set[int], set[int]]]]:
    group_size = config.attention_heads // config.kv_head_count
    per_head: list[list[list[int]]] = [[[] for _ in range(query_count)] for _ in range(config.attention_heads)]
    regions: dict[tuple[int, int], tuple[set[int], set[int], set[int]]] = {}
    for local_query in range(query_count):
        query_token = key_count - query_count + local_query
        for kv_head in range(config.kv_head_count):
            head_start = kv_head * group_size
            head_end = min((kv_head + 1) * group_size, config.attention_heads)
            if kv_head not in config.kv_heads or layer not in config.layers:
                visible = query_token + 1
                full = list(range(visible))
                for head in range(head_start, head_end):
                    per_head[head][local_query] = full
                continue
            heads = list(range(head_start, head_end))
            candidates, prefix, recent, expanded = candidate_indices_for_head_group(
                state, config, layer, kv_head, heads, query_token
            )
            if config.candidate_granularity == "attention_head":
                for head in heads:
                    per_head[head][local_query] = candidates
            else:
                for head in heads:
                    per_head[head][local_query] = candidates
            for head in heads:
                regions[(head, local_query)] = (prefix, recent, expanded)
    max_candidates = max(len(items) for by_query in per_head for items in by_query)
    max_candidates = max(1, max_candidates)
    indices = torch.zeros((config.attention_heads, query_count, max_candidates), dtype=torch.long, device=device)
    mask = torch.zeros((config.attention_heads, query_count, max_candidates), dtype=torch.bool, device=device)
    for head in range(config.attention_heads):
        for local_query in range(query_count):
            values = per_head[head][local_query]
            if not values:
                values = [max(0, key_count - query_count + local_query)]
            length = len(values)
            indices[head, local_query, :length] = torch.tensor(values, dtype=torch.long, device=device)
            mask[head, local_query, :length] = True
    return indices, mask, regions


def update_stats_and_prev(
    state: GraphRuntimeState,
    config: GraphMaskConfig,
    layer: int,
    candidate_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    attention_weights: torch.Tensor,
    full_scores: torch.Tensor | None,
    regions: dict[tuple[int, int], tuple[set[int], set[int], set[int]]],
) -> None:
    heads, query_count, _ = candidate_indices.shape
    group_size = config.attention_heads // config.kv_head_count
    for head in range(heads):
        kv_head = head // group_size
        if layer not in config.layers or kv_head not in config.kv_heads:
            continue
        for local_query in range(query_count):
            valid = candidate_mask[head, local_query]
            indices = candidate_indices[head, local_query, valid].detach()
            weights = attention_weights[head, local_query, valid].detach().float()
            state.prev_indices[(layer, head)] = indices
            state.prev_weights[(layer, head)] = weights
            stat = state.stats.setdefault((layer, kv_head, head), HeadStats(overlap_position_bins=[0] * config.position_bins))
            stat.query_count += 1
            visible = int(indices.max().item()) + 1 if indices.numel() else 1
            candidate_count = int(indices.numel())
            stat.candidate_sum += candidate_count / max(1, visible)
            stat.candidate_p95_values.append(candidate_count / max(1, visible))
            if not config.compute_overlap_metrics or full_scores is None or candidate_count <= 0:
                continue
            scores = full_scores[head, local_query, :visible].float()
            true_attention = F.softmax(scores, dim=-1)
            m = min(candidate_count, visible)
            top = torch.topk(true_attention, k=m, largest=True).indices
            candidate_set = set(int(value) for value in indices.tolist())
            top_set = set(int(value) for value in top.tolist())
            overlap = candidate_set & top_set
            stat.overlap_count_sum += len(overlap)
            stat.overlap_ratio_sum += len(overlap) / max(1, m)
            stat.retrieval_energy_sum += float(true_attention[indices].sum().detach().cpu())
            stat.oracle_energy_sum += float(true_attention[top].sum().detach().cpu())
            prefix, recent, expanded = regions.get((head, local_query), (set(), set(), set()))
            stat.prefix_overlap_sum += len(overlap & prefix)
            stat.recent_overlap_sum += len(overlap & recent)
            stat.middle_overlap_sum += len(overlap & expanded)
            for idx in overlap:
                bin_idx = min(config.position_bins - 1, int((idx / max(1, visible)) * config.position_bins))
                stat.overlap_position_bins[bin_idx] += 1


def sparse_graph_attention_forward(
    layer: int,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    assert _GRAPH_CONFIG is not None and _GRAPH_STATE is not None
    config = _GRAPH_CONFIG
    state = _GRAPH_STATE
    batch, attention_heads, query_count, head_dim = query_states.shape
    if batch != 1:
        raise RuntimeError("Sparse K-K graph attention currently supports batch size 1.")
    original_key_states = key_states
    if key_states.shape[1] != attention_heads:
        repeat_groups = attention_heads // key_states.shape[1]
        key_for_attention = key_states.repeat_interleave(repeat_groups, dim=1)
        value_for_attention = value_states.repeat_interleave(repeat_groups, dim=1)
    else:
        key_for_attention = key_states
        value_for_attention = value_states
    key_count = int(key_for_attention.shape[2])
    for kv_head in config.kv_heads:
        update_graph_for_head(state, config, layer, kv_head, original_key_states[0, kv_head].detach())
    indices, candidate_mask, regions = build_candidate_tensor(state, config, layer, key_count, query_count, query_states.device)
    head_ids = torch.arange(attention_heads, device=query_states.device).view(attention_heads, 1, 1).expand_as(indices)
    gathered_k = key_for_attention[0][head_ids, indices]
    gathered_v = value_for_attention[0][head_ids, indices]
    scores = (query_states[0].unsqueeze(2) * gathered_k).sum(dim=-1) * scaling
    scores = scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)
    full_scores = None
    if config.compute_overlap_metrics:
        full_scores = torch.matmul(query_states[0].float(), key_for_attention[0].float().transpose(1, 2)) * scaling
        token_positions = torch.arange(key_count, device=query_states.device)
        query_tokens = key_count - query_count + torch.arange(query_count, device=query_states.device)
        causal = token_positions.view(1, 1, -1) <= query_tokens.view(1, -1, 1)
        full_scores = full_scores.masked_fill(~causal, -float("inf"))
        if attention_mask is not None:
            full_scores = full_scores + attention_mask[0, :, :, :key_count].float()
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    output = (attention_weights.unsqueeze(-1) * gathered_v).sum(dim=2)
    update_stats_and_prev(state, config, layer, indices, candidate_mask, attention_weights, full_scores, regions)
    return output.transpose(0, 1).unsqueeze(0).contiguous(), None


def _graph_eager_attention_forward(
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
    layer = _MODULE_TO_LAYER.get(id(module), -1)
    if _GRAPH_ENABLED and _GRAPH_CONFIG is not None and layer in _GRAPH_CONFIG.layers:
        return sparse_graph_attention_forward(layer, query_states, key_states, value_states, attention_mask, scaling)
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states = key_states.repeat_interleave(repeat_groups, dim=1)
        value_states = value_states.repeat_interleave(repeat_groups, dim=1)
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=True)
    attention_output = torch.matmul(attention_weights, value_states)
    return attention_output.transpose(1, 2).contiguous(), attention_weights


def install_qwen3_attention_patch() -> None:
    global _ORIGINAL_EAGER_ATTENTION_FORWARD
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is not None:
        return
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except Exception as exc:
        raise RuntimeError("Could not import transformers.models.qwen3.modeling_qwen3.") from exc
    _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
    setattr(modeling_qwen3, "eager_attention_forward", _graph_eager_attention_forward)
    if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
        modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _graph_eager_attention_forward


@contextmanager
def graph_mask_enabled(enabled: bool):
    global _GRAPH_ENABLED
    previous = _GRAPH_ENABLED
    _GRAPH_ENABLED = enabled
    try:
        yield
    finally:
        _GRAPH_ENABLED = previous


@torch.inference_mode()
def prefill_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    chunk_size: int,
    input_device: torch.device,
) -> tuple[Any, torch.Tensor | None, float]:
    past_key_values = None
    last_logits: torch.Tensor | None = None
    total_chunks = math.ceil(prefill_tokens / chunk_size)
    start_time = time.perf_counter()
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
    return past_key_values, last_logits, time.perf_counter() - start_time


@torch.inference_mode()
def compute_eval_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    use_graph_mask: bool,
) -> tuple[float, float, int, float, float]:
    assert _GRAPH_STATE is not None
    _GRAPH_STATE.reset_sequence()
    with graph_mask_enabled(False):
        past_key_values, prev_logits, prefill_seconds = prefill_cache(model, input_ids, prefill_tokens, chunk_size, input_device)
    if prev_logits is None:
        raise RuntimeError("Prefill did not return last logits.")
    total_loss = 0.0
    total_count = 0
    eval_end = prefill_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / chunk_size)
    start_time = time.perf_counter()
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
        label = "kkl2_graph" if use_graph_mask else "full_attention"
        print(f"ppl {label} chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        with graph_mask_enabled(use_graph_mask):
            outputs = model_forward(model, kwargs)
        logits = outputs.logits
        shifted_logits = torch.cat([prev_logits.unsqueeze(1), logits[:, :-1, :]], dim=1)
        loss = F.cross_entropy(shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(), chunk.reshape(-1), reduction="sum")
        total_loss += float(loss)
        total_count += int(chunk.numel())
        prev_logits = logits[:, -1, :].detach()
        past_key_values = outputs.past_key_values
        del outputs, chunk, logits, shifted_logits, loss
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    decode_seconds = time.perf_counter() - start_time
    mean_loss = total_loss / max(1, total_count)
    return mean_loss, math.exp(min(mean_loss, 80.0)), total_count, prefill_seconds, decode_seconds


def ppl_fields() -> list[str]:
    return [
        "mode",
        "loss",
        "ppl",
        "token_count",
        "prefill_seconds",
        "decode_seconds",
        "tokens_per_second",
        "layers",
        "kv_heads",
        "boundary_fraction",
        "middle_fraction",
        "seed_count",
        "graph_degree",
        "graph_update_interval",
        "graph_update_mode",
        "max_hops",
        "candidate_granularity",
        "compute_overlap_metrics",
    ]


def head_summary_rows(state: GraphRuntimeState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (layer, kv_head, head), stat in sorted(state.stats.items()):
        values = torch.tensor(stat.candidate_p95_values, dtype=torch.float32) if stat.candidate_p95_values else torch.tensor([0.0])
        row = {
            "layer": layer,
            "kv_head": kv_head,
            "attention_head": head,
            "query_count": stat.query_count,
            "candidate_fraction_mean": stat.candidate_sum / max(1, stat.query_count),
            "candidate_fraction_p95": float(torch.quantile(values, 0.95)),
            "overlap_count_mean": stat.overlap_count_sum / max(1, stat.query_count),
            "overlap_ratio_mean": stat.overlap_ratio_sum / max(1, stat.query_count),
            "retrieval_attention_energy_mean": stat.retrieval_energy_sum / max(1, stat.query_count),
            "oracle_topm_attention_energy_mean": stat.oracle_energy_sum / max(1, stat.query_count),
            "prefix_overlap_mean": stat.prefix_overlap_sum / max(1, stat.query_count),
            "middle_overlap_mean": stat.middle_overlap_sum / max(1, stat.query_count),
            "recent_overlap_mean": stat.recent_overlap_sum / max(1, stat.query_count),
            "overlap_position_bins": ";".join(str(value) for value in stat.overlap_position_bins),
        }
        rows.append(row)
    return rows


def head_summary_fields() -> list[str]:
    return [
        "layer",
        "kv_head",
        "attention_head",
        "query_count",
        "candidate_fraction_mean",
        "candidate_fraction_p95",
        "overlap_count_mean",
        "overlap_ratio_mean",
        "retrieval_attention_energy_mean",
        "oracle_topm_attention_energy_mean",
        "prefix_overlap_mean",
        "middle_overlap_mean",
        "recent_overlap_mean",
        "overlap_position_bins",
    ]


def timing_fields() -> list[str]:
    return ["event", "layer", "kv_head", "key_count", "built_until", "updated_rows", "seconds", "mode"]


def main() -> None:
    global _GRAPH_CONFIG, _GRAPH_STATE
    args = parse_args()
    if args.boundary_fraction <= 0 or args.middle_fraction <= 0:
        raise ValueError("fractions must be positive.")
    if args.seed_count <= 0 or args.graph_degree <= 0 or args.graph_update_interval <= 0:
        raise ValueError("seed_count, graph_degree, and graph_update_interval must be positive.")
    if args.max_hops <= 0:
        raise ValueError("--max_hops must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    if args.eval_last_tokens_only:
        if len(token_ids) <= args.eval_tokens:
            raise ValueError(f"Tokenization produced {len(token_ids)} tokens, not enough for eval_last_tokens_only.")
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

    layer_count = int(getattr(model.config, "num_hidden_layers"))
    attention_heads = int(getattr(model.config, "num_attention_heads"))
    kv_head_count = int(getattr(model.config, "num_key_value_heads"))
    layers = parse_index_spec(args.layers, layer_count, "layers")
    kv_heads = parse_index_spec(args.kv_heads, kv_head_count, "kv_heads")
    _GRAPH_CONFIG = GraphMaskConfig(
        layers=set(layers),
        kv_heads=set(kv_heads),
        attention_heads=attention_heads,
        kv_head_count=kv_head_count,
        boundary_fraction=args.boundary_fraction,
        middle_fraction=args.middle_fraction,
        seed_count=args.seed_count,
        graph_degree=args.graph_degree,
        graph_update_interval=args.graph_update_interval,
        graph_update_mode=args.graph_update_mode,
        max_hops=args.max_hops,
        candidate_granularity=args.candidate_granularity,
        compute_overlap_metrics=args.compute_overlap_metrics,
        position_bins=args.position_bins,
    )
    _GRAPH_STATE = GraphRuntimeState()
    register_attention_layers(model)
    install_qwen3_attention_patch()
    input_device = pick_input_device(model, requested_device)

    metadata = {
        "layers": ",".join(str(index) for index in layers),
        "kv_heads": ",".join(str(index) for index in kv_heads),
        "boundary_fraction": args.boundary_fraction,
        "middle_fraction": args.middle_fraction,
        "seed_count": args.seed_count,
        "graph_degree": args.graph_degree,
        "graph_update_interval": args.graph_update_interval,
        "graph_update_mode": args.graph_update_mode,
        "max_hops": args.max_hops,
        "candidate_granularity": args.candidate_granularity,
        "compute_overlap_metrics": args.compute_overlap_metrics,
    }
    rows: list[dict[str, Any]] = []
    baseline_loss, baseline_ppl, baseline_count, baseline_prefill, baseline_decode = compute_eval_loss(
        model, input_ids, prefill_tokens, args.eval_tokens, args.chunk_size, input_device, False
    )
    rows.append(
        {
            "mode": "full_attention",
            "loss": baseline_loss,
            "ppl": baseline_ppl,
            "token_count": baseline_count,
            "prefill_seconds": baseline_prefill,
            "decode_seconds": baseline_decode,
            "tokens_per_second": baseline_count / max(baseline_decode, 1e-9),
            **metadata,
        }
    )
    graph_summary_path = None
    timing_path = None
    if args.compute_graph_ppl:
        _GRAPH_STATE.reset_all()
        graph_loss, graph_ppl, graph_count, graph_prefill, graph_decode = compute_eval_loss(
            model, input_ids, prefill_tokens, args.eval_tokens, args.chunk_size, input_device, True
        )
        rows.append(
            {
                "mode": "kkl2_graph",
                "loss": graph_loss,
                "ppl": graph_ppl,
                "token_count": graph_count,
                "prefill_seconds": graph_prefill,
                "decode_seconds": graph_decode,
                "tokens_per_second": graph_count / max(graph_decode, 1e-9),
                **metadata,
            }
        )
        graph_summary_path = output_dir / "summary_by_layer_head.csv"
        write_csv(graph_summary_path, head_summary_rows(_GRAPH_STATE), head_summary_fields())
        timing_path = output_dir / "graph_timing.csv"
        write_csv(timing_path, _GRAPH_STATE.timing_rows, timing_fields())

    ppl_path = output_dir / "ppl_by_mode.csv"
    write_csv(ppl_path, rows, ppl_fields())
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "resolved": {
                    "prefill_tokens": prefill_tokens,
                    "eval_tokens": args.eval_tokens,
                    "total_tokenized_tokens_used": int(input_ids.numel()),
                    "layers": layers,
                    "kv_heads": kv_heads,
                    "attention_heads": attention_heads,
                    "kv_head_count": kv_head_count,
                },
                "paths": {
                    "ppl_by_mode": str(ppl_path),
                    "summary_by_layer_head": str(graph_summary_path) if graph_summary_path is not None else None,
                    "graph_timing": str(timing_path) if timing_path is not None else None,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote K-K L2 graph PPL outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
