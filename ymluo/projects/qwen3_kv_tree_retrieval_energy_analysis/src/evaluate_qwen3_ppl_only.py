from __future__ import annotations

import argparse
import csv
import json
import math
import time
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
DEFAULT_TEXT_PATH = (
    "/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt"
)

_TREE_CONFIG: TreeMaskConfig | None = None
_TREE_ENABLED = False
_MODULE_TO_LAYER: dict[int, int] = {}
_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_TREE_CANDIDATE_TOKEN_SUM: torch.Tensor | None = None
_TREE_CANDIDATE_OBS_COUNT: torch.Tensor | None = None


@dataclass(frozen=True)
class TreeLayout:
    leaf_size: int
    leaf_ranges: list[tuple[int, int]]
    mid_ranges: list[tuple[int, int]]
    mid_children: list[list[int]]
    big_ranges: list[tuple[int, int]]
    big_children: list[list[int]]


@dataclass(frozen=True)
class TreeMaskConfig:
    layers: set[int]
    kv_heads: set[int]
    boundary_fraction: float
    leaf_fraction: float
    leaf_size: int
    tree_fanout: int
    branch_counts: tuple[int, int, int]
    candidate_granularity: str
    attention_impl: str


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and tree-retrieval-masked Qwen3 PPL.")
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
    parser.add_argument("--compute_baseline_ppl", type=str2bool, default=True)
    parser.add_argument("--compute_tree_ppl", type=str2bool, default=True)
    parser.add_argument(
        "--tree_prefill",
        type=str2bool,
        default=True,
        help="Use tree attention during the tree PPL prefill. False means baseline prefill + tree eval.",
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument("--kv_heads", default="all")
    parser.add_argument("--boundary_fraction", type=float, default=0.01)
    parser.add_argument("--leaf_fraction", type=float, default=0.001)
    parser.add_argument("--leaf_size", type=int, default=0)
    parser.add_argument("--tree_fanout", type=int, default=10)
    parser.add_argument("--tree_branch_counts", default="5,5,5")
    parser.add_argument("--candidate_granularity", choices=["attention_head", "kv_head_union"], default="attention_head")
    parser.add_argument(
        "--tree_attention_impl",
        choices=["mask", "sparse_gather", "shared_matmul"],
        default="sparse_gather",
    )
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


def parse_branch_counts(spec: str) -> tuple[int, int, int]:
    values = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("--tree_branch_counts must be three positive integers, e.g. 5,5,5")
    return values[0], values[1], values[2]


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


def ppl_fields() -> list[str]:
    return [
        "mode",
        "loss",
        "ppl",
        "token_count",
        "layers",
        "kv_heads",
        "boundary_fraction",
        "boundary_token_rule",
        "leaf_fraction",
        "leaf_size",
        "tree_fanout",
        "tree_branch_counts",
        "candidate_granularity",
        "tree_attention_impl",
        "tree_prefill",
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed_section(name: str, device: torch.device, timings: dict[str, float]):
    synchronize_if_cuda(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        synchronize_if_cuda(device)
        elapsed = time.perf_counter() - start
        timings[name] = timings.get(name, 0.0) + elapsed
        print(f"timer {name}: {elapsed:.3f}s", flush=True)


def timing_fields() -> list[str]:
    return ["mode", "prefill_seconds", "eval_seconds", "total_seconds", "tokens_per_second", "avg_candidate_tokens"]


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


def build_tree_layout(total_tokens: int, leaf_size: int, fanout: int) -> TreeLayout:
    leaf_ranges = [(start, min(start + leaf_size, total_tokens)) for start in range(0, total_tokens, leaf_size)]
    mid_ranges: list[tuple[int, int]] = []
    mid_children: list[list[int]] = []
    for start in range(0, len(leaf_ranges), fanout):
        children = list(range(start, min(start + fanout, len(leaf_ranges))))
        mid_children.append(children)
        mid_ranges.append((leaf_ranges[children[0]][0], leaf_ranges[children[-1]][1]))
    big_ranges: list[tuple[int, int]] = []
    big_children: list[list[int]] = []
    for start in range(0, len(mid_ranges), fanout):
        children = list(range(start, min(start + fanout, len(mid_ranges))))
        big_children.append(children)
        big_ranges.append((mid_ranges[children[0]][0], mid_ranges[children[-1]][1]))
    return TreeLayout(leaf_size, leaf_ranges, mid_ranges, mid_children, big_ranges, big_children)


def ranges_tensor(ranges: list[tuple[int, int]], device: torch.device) -> torch.Tensor:
    return torch.tensor(ranges, dtype=torch.long, device=device)


def padded_children_tensor(children: list[list[int]], width: int, device: torch.device) -> torch.Tensor:
    result = torch.full((len(children), width), -1, dtype=torch.long, device=device)
    for row, values in enumerate(children):
        if values:
            result[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
    return result


def fixed_boundary_token_count(key_count: int) -> int:
    return 500 if key_count > 10_000 else 50


def boundary_count_for_visible(visible: torch.Tensor, key_count: int) -> torch.Tensor:
    raw = torch.full_like(visible, fixed_boundary_token_count(key_count))
    max_nonoverlap = torch.div(visible, 2, rounding_mode="floor").clamp_min(1)
    return torch.minimum(raw, max_nonoverlap)


def query_clipped_centers(
    prefix_sum: torch.Tensor,
    ranges: torch.Tensor,
    middle_start: torch.Tensor,
    middle_end: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Returns centers with shape [query_count, node_count, head_dim].
    starts = ranges[:, 0].unsqueeze(0)
    ends = ranges[:, 1].unsqueeze(0)
    left = torch.maximum(starts, middle_start.unsqueeze(1))
    right = torch.minimum(ends, middle_end.unsqueeze(1))
    counts = (right - left).clamp_min(0)
    centers = (prefix_sum[right] - prefix_sum[left]) / counts.clamp_min(1).unsqueeze(-1)
    return centers, counts > 0


def score_candidate_nodes(
    queries: torch.Tensor,
    centers: torch.Tensor,
    valid_nodes: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    # queries: [heads, query_count, head_dim]
    # centers: [query_count, node_count, head_dim]
    safe_ids = candidate_ids.clamp_min(0)
    query_count = queries.shape[1]
    candidate_rank = candidate_ids.ndim - 2
    query_index_shape = (1, query_count) + (1,) * candidate_rank
    query_index = torch.arange(query_count, device=queries.device).view(query_index_shape)
    candidate_centers = centers[query_index, safe_ids]
    candidate_valid = valid_nodes[query_index, safe_ids] & (candidate_ids >= 0)
    query_view = queries.view(queries.shape[0], query_count, *([1] * candidate_rank), queries.shape[-1])
    scores = (query_view.float() * candidate_centers).sum(dim=-1)
    return scores.masked_fill(~candidate_valid, -torch.inf)


def select_tree_leaf_ids_for_kv_head(
    queries: torch.Tensor,
    key_vectors: torch.Tensor,
    layout: TreeLayout,
    config: TreeMaskConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = queries.device
    query_count = queries.shape[1]
    key_count = key_vectors.shape[0]
    query_tokens = key_count - query_count + torch.arange(query_count, dtype=torch.long, device=device)
    visible = query_tokens + 1
    boundary_count = boundary_count_for_visible(visible, key_count)
    middle_start = torch.minimum(boundary_count, visible)
    recent_start = (visible - boundary_count).clamp_min(0)
    middle_end = torch.maximum(middle_start, recent_start)

    zeros = torch.zeros((1, key_vectors.shape[-1]), dtype=torch.float32, device=device)
    prefix_sum = torch.cat([zeros, torch.cumsum(key_vectors.detach().float(), dim=0)], dim=0)
    big_ranges = ranges_tensor(layout.big_ranges, device)
    mid_ranges = ranges_tensor(layout.mid_ranges, device)
    leaf_ranges = ranges_tensor(layout.leaf_ranges, device)
    big_to_mid = padded_children_tensor(layout.big_children, config.tree_fanout, device)
    mid_to_leaf = padded_children_tensor(layout.mid_children, config.tree_fanout, device)

    big_centers, big_valid = query_clipped_centers(prefix_sum, big_ranges, middle_start, middle_end)
    big_ids = torch.arange(big_ranges.shape[0], dtype=torch.long, device=device).view(1, 1, -1)
    big_scores = score_candidate_nodes(queries, big_centers, big_valid, big_ids.expand(queries.shape[0], query_count, -1))
    big_k = min(config.branch_counts[0], big_scores.shape[-1])
    top_big_scores, top_big_ids = torch.topk(big_scores, k=big_k, dim=-1)
    top_big_ids = top_big_ids.masked_fill(~torch.isfinite(top_big_scores), -1)

    mid_centers, mid_valid = query_clipped_centers(prefix_sum, mid_ranges, middle_start, middle_end)
    mid_candidates = big_to_mid[top_big_ids.clamp_min(0)]
    mid_candidates = mid_candidates.masked_fill(top_big_ids.unsqueeze(-1) < 0, -1)
    mid_scores = score_candidate_nodes(queries, mid_centers, mid_valid, mid_candidates)
    mid_k = min(config.branch_counts[1], mid_scores.shape[-1])
    top_mid_scores, top_mid_pos = torch.topk(mid_scores, k=mid_k, dim=-1)
    top_mid_ids = torch.gather(mid_candidates, dim=-1, index=top_mid_pos)
    top_mid_ids = top_mid_ids.masked_fill(~torch.isfinite(top_mid_scores), -1).flatten(start_dim=2)

    leaf_centers, leaf_valid = query_clipped_centers(prefix_sum, leaf_ranges, middle_start, middle_end)
    leaf_candidates = mid_to_leaf[top_mid_ids.clamp_min(0)]
    leaf_candidates = leaf_candidates.masked_fill(top_mid_ids.unsqueeze(-1) < 0, -1)
    leaf_scores = score_candidate_nodes(queries, leaf_centers, leaf_valid, leaf_candidates)
    leaf_k = min(config.branch_counts[2], leaf_scores.shape[-1])
    top_leaf_scores, top_leaf_pos = torch.topk(leaf_scores, k=leaf_k, dim=-1)
    top_leaf_ids = torch.gather(leaf_candidates, dim=-1, index=top_leaf_pos)
    top_leaf_ids = top_leaf_ids.masked_fill(~torch.isfinite(top_leaf_scores), -1).flatten(start_dim=2)
    return top_leaf_ids, leaf_ranges, middle_start, middle_end


def scatter_leaf_tokens(
    keep: torch.Tensor,
    selected_leaf_ids: torch.Tensor,
    leaf_ranges: torch.Tensor,
    middle_start: torch.Tensor,
    middle_end: torch.Tensor,
    leaf_size: int,
) -> None:
    safe_leaf_ids = selected_leaf_ids.clamp_min(0)
    starts = leaf_ranges[safe_leaf_ids, 0]
    ends = leaf_ranges[safe_leaf_ids, 1]
    starts = torch.maximum(starts, middle_start.view(1, -1, 1))
    ends = torch.minimum(ends, middle_end.view(1, -1, 1))
    offsets = torch.arange(leaf_size, dtype=torch.long, device=keep.device).view(1, 1, 1, -1)
    token_ids = starts.unsqueeze(-1) + offsets
    valid = (selected_leaf_ids.unsqueeze(-1) >= 0) & (token_ids < ends.unsqueeze(-1))
    token_ids = token_ids.masked_fill(~valid, 0).flatten(start_dim=2)
    keep.scatter_(dim=-1, index=token_ids, src=torch.ones_like(token_ids, dtype=torch.bool))


def leaf_token_ids_from_selection(
    selected_leaf_ids: torch.Tensor,
    leaf_ranges: torch.Tensor,
    middle_start: torch.Tensor,
    middle_end: torch.Tensor,
    leaf_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_leaf_ids = selected_leaf_ids.clamp_min(0)
    starts = leaf_ranges[safe_leaf_ids, 0]
    ends = leaf_ranges[safe_leaf_ids, 1]
    starts = torch.maximum(starts, middle_start.view(1, -1, 1))
    ends = torch.minimum(ends, middle_end.view(1, -1, 1))
    offsets = torch.arange(leaf_size, dtype=torch.long, device=selected_leaf_ids.device).view(1, 1, 1, -1)
    token_ids = starts.unsqueeze(-1) + offsets
    valid = (selected_leaf_ids.unsqueeze(-1) >= 0) & (token_ids < ends.unsqueeze(-1))
    return token_ids.flatten(start_dim=2), valid.flatten(start_dim=2)


def tree_candidate_ids(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    layer: int,
    config: TreeMaskConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, attention_heads, query_count, _ = query_states.shape
    _, kv_heads, key_count, _ = key_states.shape
    if batch != 1:
        raise ValueError("Tree sparse attention currently supports batch size 1.")
    if layer not in config.layers or set(range(kv_heads)) - config.kv_heads:
        raise RuntimeError("sparse_gather requires the current layer and all KV heads to be tree-masked.")

    group_size = attention_heads // kv_heads
    leaf_size = config.leaf_size if config.leaf_size > 0 else max(1, math.ceil(config.leaf_fraction * key_count))
    layout = build_tree_layout(key_count, leaf_size, config.tree_fanout)

    device = query_states.device
    query_tokens = key_count - query_count + torch.arange(query_count, dtype=torch.long, device=device)
    visible = query_tokens + 1
    boundary_count = boundary_count_for_visible(visible, key_count)
    max_boundary = int(boundary_count.max().item())
    boundary_offsets = torch.arange(max_boundary, dtype=torch.long, device=device).view(1, -1)

    prefix_ids = boundary_offsets.expand(query_count, -1)
    prefix_valid = boundary_offsets < boundary_count.view(-1, 1)
    recent_start = (visible - boundary_count).clamp_min(0)
    recent_ids = recent_start.view(-1, 1) + boundary_offsets
    recent_valid = (boundary_offsets < boundary_count.view(-1, 1)) & (recent_ids < visible.view(-1, 1))

    per_head_ids: list[torch.Tensor] = []
    per_head_valid: list[torch.Tensor] = []
    for kv_head in range(kv_heads):
        head_start = kv_head * group_size
        head_end = min((kv_head + 1) * group_size, attention_heads)
        if head_start >= head_end:
            continue

        queries = query_states[0, head_start:head_end].detach()
        key_vectors = key_states[0, kv_head].detach()
        selected_leaf_ids, leaf_ranges, middle_start, middle_end = select_tree_leaf_ids_for_kv_head(
            queries,
            key_vectors,
            layout,
            config,
        )
        leaf_ids, leaf_valid = leaf_token_ids_from_selection(
            selected_leaf_ids,
            leaf_ranges,
            middle_start,
            middle_end,
            leaf_size,
        )
        head_count = head_end - head_start
        if config.candidate_granularity == "kv_head_union":
            leaf_ids = leaf_ids.transpose(0, 1).reshape(query_count, -1).unsqueeze(0).expand(head_count, -1, -1)
            leaf_valid = leaf_valid.transpose(0, 1).reshape(query_count, -1).unsqueeze(0).expand(head_count, -1, -1)

        head_prefix_ids = prefix_ids.unsqueeze(0).expand(head_count, -1, -1)
        head_prefix_valid = prefix_valid.unsqueeze(0).expand(head_count, -1, -1)
        head_recent_ids = recent_ids.unsqueeze(0).expand(head_count, -1, -1)
        head_recent_valid = recent_valid.unsqueeze(0).expand(head_count, -1, -1)
        ids = torch.cat([head_prefix_ids, head_recent_ids, leaf_ids], dim=-1)
        valid = torch.cat([head_prefix_valid, head_recent_valid, leaf_valid], dim=-1)
        per_head_ids.append(ids.masked_fill(~valid, 0))
        per_head_valid.append(valid)

    candidate_ids = torch.cat(per_head_ids, dim=0).unsqueeze(0)
    candidate_valid = torch.cat(per_head_valid, dim=0).unsqueeze(0)
    return candidate_ids, candidate_valid


def shared_candidate_ids_for_chunk(
    candidate_ids: torch.Tensor,
    candidate_valid: torch.Tensor,
    key_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sentinel = torch.full_like(candidate_ids, key_count)
    flat = torch.where(candidate_valid, candidate_ids, sentinel).flatten(start_dim=2)
    sorted_ids, _ = torch.sort(flat, dim=-1)
    sorted_valid = sorted_ids < key_count
    duplicate = torch.zeros_like(sorted_valid)
    duplicate[..., 1:] = sorted_ids[..., 1:] == sorted_ids[..., :-1]
    sorted_valid = sorted_valid & ~duplicate
    return sorted_ids.masked_fill(~sorted_valid, 0), sorted_valid


def reset_tree_candidate_stats(device: torch.device) -> None:
    global _TREE_CANDIDATE_TOKEN_SUM, _TREE_CANDIDATE_OBS_COUNT
    _TREE_CANDIDATE_TOKEN_SUM = torch.zeros((), dtype=torch.float64, device=device)
    _TREE_CANDIDATE_OBS_COUNT = torch.zeros((), dtype=torch.float64, device=device)


def record_tree_candidate_stats(candidate_valid: torch.Tensor) -> None:
    if _TREE_CANDIDATE_TOKEN_SUM is None or _TREE_CANDIDATE_OBS_COUNT is None:
        return
    per_query_head = candidate_valid.sum(dim=-1, dtype=torch.float64)
    _TREE_CANDIDATE_TOKEN_SUM.add_(per_query_head.sum())
    _TREE_CANDIDATE_OBS_COUNT.add_(
        torch.tensor(per_query_head.numel(), dtype=torch.float64, device=per_query_head.device)
    )


def read_tree_candidate_stats() -> dict[str, float]:
    if _TREE_CANDIDATE_TOKEN_SUM is None or _TREE_CANDIDATE_OBS_COUNT is None:
        return {}
    count = float(_TREE_CANDIDATE_OBS_COUNT.item())
    total = float(_TREE_CANDIDATE_TOKEN_SUM.item())
    return {"avg_candidate_tokens": total / max(count, 1.0)}


def clear_tree_candidate_stats() -> None:
    global _TREE_CANDIDATE_TOKEN_SUM, _TREE_CANDIDATE_OBS_COUNT
    _TREE_CANDIDATE_TOKEN_SUM = None
    _TREE_CANDIDATE_OBS_COUNT = None


def tree_keep_mask(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    layer: int,
    scores: torch.Tensor,
    config: TreeMaskConfig,
) -> torch.Tensor:
    batch, attention_heads, query_count, _ = query_states.shape
    _, kv_heads, key_count, _ = key_states.shape
    if batch != 1:
        raise ValueError("Tree PPL currently supports batch size 1.")
    valid_scores = torch.isfinite(scores)
    if layer not in config.layers:
        return valid_scores

    group_size = attention_heads // kv_heads
    active_kv_heads = {kv_head for kv_head in range(kv_heads) if kv_head in config.kv_heads}
    if not active_kv_heads:
        return valid_scores

    leaf_size = config.leaf_size if config.leaf_size > 0 else max(1, math.ceil(config.leaf_fraction * key_count))
    layout = build_tree_layout(key_count, leaf_size, config.tree_fanout)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    token_positions = torch.arange(key_count, dtype=torch.long, device=scores.device)
    query_tokens = key_count - query_count + torch.arange(query_count, dtype=torch.long, device=scores.device)
    visible = query_tokens + 1
    causal_keep = token_positions.view(1, -1) < visible.view(-1, 1)
    boundary_count = boundary_count_for_visible(visible, key_count)
    prefix_keep = token_positions.view(1, -1) < boundary_count.view(-1, 1)
    recent_start = (visible - boundary_count).clamp_min(0)
    recent_keep = (token_positions.view(1, -1) >= recent_start.view(-1, 1)) & causal_keep
    boundary_keep = (prefix_keep | recent_keep) & causal_keep

    for kv_head in range(kv_heads):
        head_start = kv_head * group_size
        head_end = min((kv_head + 1) * group_size, attention_heads)
        if head_start >= head_end:
            continue
        if kv_head not in active_kv_heads:
            keep[:, head_start:head_end] = causal_keep.view(1, 1, query_count, key_count)
            continue

        queries = query_states[0, head_start:head_end].detach()
        key_vectors = key_states[0, kv_head].detach()
        selected_leaf_ids, leaf_ranges, middle_start, middle_end = select_tree_leaf_ids_for_kv_head(
            queries,
            key_vectors,
            layout,
            config,
        )
        kv_keep = boundary_keep.view(1, query_count, key_count).expand(head_end - head_start, -1, -1).clone()
        scatter_leaf_tokens(kv_keep, selected_leaf_ids, leaf_ranges, middle_start, middle_end, leaf_size)
        if config.candidate_granularity == "kv_head_union":
            kv_keep = kv_keep.any(dim=0, keepdim=True).expand(head_end - head_start, -1, -1)
        keep[:, head_start:head_end] = kv_keep.unsqueeze(0)
    return keep & valid_scores


def max_candidate_count(key_count: int, leaf_size: int, group_size: int, config: TreeMaskConfig) -> int:
    boundary = fixed_boundary_token_count(key_count)
    leaf_tokens = config.branch_counts[0] * config.branch_counts[1] * config.branch_counts[2] * leaf_size
    if config.candidate_granularity == "kv_head_union":
        leaf_tokens *= group_size
    return min(key_count, 2 * boundary + leaf_tokens)


def sparse_tree_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    layer: int,
    config: TreeMaskConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch, attention_heads, query_count, head_dim = query_states.shape
    _, kv_heads, key_count, _ = key_states.shape
    if batch != 1:
        raise ValueError("Tree sparse attention currently supports batch size 1.")
    if layer not in config.layers or set(range(kv_heads)) - config.kv_heads:
        raise RuntimeError("sparse_gather requires the current layer and all KV heads to be tree-masked.")

    group_size = attention_heads // kv_heads
    candidate_ids, candidate_valid = tree_candidate_ids(query_states, key_states, layer, config)
    candidate_slots = candidate_ids.shape[-1]
    record_tree_candidate_stats(candidate_valid)

    if key_states.shape[1] != attention_heads:
        key_states_for_attention = key_states.repeat_interleave(group_size, dim=1)
        value_states_for_attention = value_states.repeat_interleave(group_size, dim=1)
    else:
        key_states_for_attention = key_states
        value_states_for_attention = value_states

    batch_index = torch.arange(batch, device=query_states.device).view(batch, 1, 1, 1)
    head_index = torch.arange(attention_heads, device=query_states.device).view(1, attention_heads, 1, 1)
    selected_keys = key_states_for_attention[batch_index, head_index, candidate_ids]
    selected_values = value_states_for_attention[batch_index, head_index, candidate_ids]

    scores = (query_states.unsqueeze(3) * selected_keys).sum(dim=-1) * scaling
    if attention_mask is not None:
        mask = attention_mask[:, :, :, :key_count].expand(batch, attention_heads, query_count, key_count)
        gathered_mask = torch.gather(mask, dim=-1, index=candidate_ids)
        scores = scores + gathered_mask
    scores = scores.masked_fill(~candidate_valid, torch.finfo(scores.dtype).min)
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.sum(attention_weights.unsqueeze(-1) * selected_values, dim=3)
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, None


def shared_matmul_tree_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    layer: int,
    config: TreeMaskConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch, attention_heads, query_count, head_dim = query_states.shape
    _, kv_heads, key_count, _ = key_states.shape
    if batch != 1:
        raise ValueError("Tree shared-matmul attention currently supports batch size 1.")
    if layer not in config.layers or set(range(kv_heads)) - config.kv_heads:
        raise RuntimeError("shared_matmul requires the current layer and all KV heads to be tree-masked.")

    group_size = attention_heads // kv_heads
    per_query_ids, per_query_valid = tree_candidate_ids(query_states, key_states, layer, config)
    candidate_ids, candidate_valid = shared_candidate_ids_for_chunk(per_query_ids, per_query_valid, key_count)
    record_tree_candidate_stats(candidate_valid.unsqueeze(2).expand(-1, -1, query_count, -1))

    if key_states.shape[1] != attention_heads:
        key_states_for_attention = key_states.repeat_interleave(group_size, dim=1)
        value_states_for_attention = value_states.repeat_interleave(group_size, dim=1)
    else:
        key_states_for_attention = key_states
        value_states_for_attention = value_states

    batch_index = torch.arange(batch, device=query_states.device).view(batch, 1, 1)
    head_index = torch.arange(attention_heads, device=query_states.device).view(1, attention_heads, 1)
    selected_keys = key_states_for_attention[batch_index, head_index, candidate_ids]
    selected_values = value_states_for_attention[batch_index, head_index, candidate_ids]

    scores = torch.matmul(query_states, selected_keys.transpose(-2, -1)) * scaling
    query_tokens = key_count - query_count + torch.arange(query_count, dtype=torch.long, device=query_states.device)
    visible = query_tokens + 1
    visible_keep = candidate_ids.unsqueeze(2) < visible.view(1, 1, query_count, 1)
    valid_keep = candidate_valid.unsqueeze(2) & visible_keep
    if attention_mask is not None:
        mask = attention_mask[:, :, :, :key_count].expand(batch, attention_heads, query_count, key_count)
        gathered_mask = torch.gather(mask, dim=-1, index=candidate_ids.unsqueeze(2).expand(-1, -1, query_count, -1))
        scores = scores + gathered_mask
    scores = scores.masked_fill(~valid_keep, torch.finfo(scores.dtype).min)
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.matmul(attention_weights, selected_values)
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, None


def _tree_eager_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = _TREE_CONFIG
    if scaling is None:
        scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(query_states.shape[-1])))
    original_key_states = key_states
    layer = _MODULE_TO_LAYER.get(id(module), -1)
    if _TREE_ENABLED and config is not None and config.attention_impl == "shared_matmul":
        try:
            return shared_matmul_tree_attention(query_states, key_states, value_states, attention_mask, scaling, layer, config)
        except RuntimeError:
            pass
    if _TREE_ENABLED and config is not None and config.attention_impl == "sparse_gather":
        try:
            return sparse_tree_attention(query_states, key_states, value_states, attention_mask, scaling, layer, config)
        except RuntimeError:
            pass
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states_for_attention = key_states.repeat_interleave(repeat_groups, dim=1)
        value_states = value_states.repeat_interleave(repeat_groups, dim=1)
    else:
        key_states_for_attention = key_states
    scores = torch.matmul(query_states, key_states_for_attention.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    if _TREE_ENABLED and config is not None:
        keep = tree_keep_mask(query_states, original_key_states, layer, scores, config)
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
        raise RuntimeError("Could not import transformers.models.qwen3.modeling_qwen3 for tree PPL.") from exc
    _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
    setattr(modeling_qwen3, "eager_attention_forward", _tree_eager_attention_forward)
    if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
        modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _tree_eager_attention_forward


@contextmanager
def tree_mask_enabled(enabled: bool):
    global _TREE_ENABLED
    previous = _TREE_ENABLED
    _TREE_ENABLED = enabled
    try:
        yield
    finally:
        _TREE_ENABLED = previous


@torch.inference_mode()
def compute_eval_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    use_tree_mask: bool,
    tree_prefill: bool,
) -> tuple[float, float, int, dict[str, float]]:
    label = "tree" if use_tree_mask else "baseline"
    use_tree_prefill = use_tree_mask and tree_prefill
    timings: dict[str, float] = {}
    clear_tree_candidate_stats()
    with timed_section(f"{label}_prefill", input_device, timings):
        with tree_mask_enabled(use_tree_prefill):
            past_key_values, prev_logits = prefill_cache(model, input_ids, prefill_tokens, chunk_size, input_device)
    if prev_logits is None:
        raise RuntimeError("Prefill did not return last logits.")
    total_loss = 0.0
    total_count = 0
    eval_end = prefill_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / chunk_size)
    if use_tree_mask:
        reset_tree_candidate_stats(input_device)
    with timed_section(f"{label}_eval", input_device, timings):
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
            print(f"ppl {label} chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
            with tree_mask_enabled(use_tree_mask):
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
    if use_tree_mask:
        timings.update(read_tree_candidate_stats())
    mean_loss = total_loss / max(1, total_count)
    timings[f"{label}_total"] = timings.get(f"{label}_prefill", 0.0) + timings.get(f"{label}_eval", 0.0)
    print(
        f"timer {label}_total: {timings[f'{label}_total']:.3f}s, "
        f"eval throughput: {total_count / max(timings.get(f'{label}_eval', 0.0), 1e-9):.2f} tokens/s",
        flush=True,
    )
    return mean_loss, math.exp(mean_loss), total_count, timings


def tree_metadata_row(args: argparse.Namespace, layer_indices: list[int], kv_head_indices: list[int]) -> dict[str, Any]:
    return {
        "layers": ",".join(str(index) for index in layer_indices),
        "kv_heads": ",".join(str(index) for index in kv_head_indices),
        "boundary_fraction": args.boundary_fraction,
        "boundary_token_rule": "50 per side for <=10k key tokens; 500 per side for >10k key tokens",
        "leaf_fraction": args.leaf_fraction,
        "leaf_size": args.leaf_size,
        "tree_fanout": args.tree_fanout,
        "tree_branch_counts": args.tree_branch_counts,
        "candidate_granularity": args.candidate_granularity,
        "tree_attention_impl": args.tree_attention_impl,
        "tree_prefill": args.tree_prefill,
    }


def main() -> None:
    global _TREE_CONFIG
    args = parse_args()
    if args.boundary_fraction <= 0 or args.leaf_fraction <= 0:
        raise ValueError("fractions must be positive.")
    if args.tree_fanout <= 1:
        raise ValueError("--tree_fanout must be greater than 1.")
    branch_counts = parse_branch_counts(args.tree_branch_counts)

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

    layer_count = int(getattr(model.config, "num_hidden_layers"))
    kv_head_count = int(getattr(model.config, "num_key_value_heads"))
    layer_indices = parse_index_spec(args.layers, layer_count, "layers")
    kv_head_indices = parse_index_spec(args.kv_heads, kv_head_count, "kv_heads")
    _TREE_CONFIG = TreeMaskConfig(
        layers=set(layer_indices),
        kv_heads=set(kv_head_indices),
        boundary_fraction=args.boundary_fraction,
        leaf_fraction=args.leaf_fraction,
        leaf_size=args.leaf_size,
        tree_fanout=args.tree_fanout,
        branch_counts=branch_counts,
        candidate_granularity=args.candidate_granularity,
        attention_impl=args.tree_attention_impl,
    )
    register_attention_layers(model)
    install_qwen3_attention_patch()
    input_device = pick_input_device(model, requested_device)

    metadata = tree_metadata_row(args, layer_indices, kv_head_indices)
    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    if args.compute_baseline_ppl:
        baseline_loss, baseline_ppl, baseline_count, baseline_timings = compute_eval_loss(
            model, input_ids, prefill_tokens, args.eval_tokens, args.chunk_size, input_device, False, args.tree_prefill
        )
        rows.append({"mode": "baseline", "loss": baseline_loss, "ppl": baseline_ppl, "token_count": baseline_count, **metadata})
        timing_rows.append(
            {
                "mode": "baseline",
                "prefill_seconds": baseline_timings.get("baseline_prefill", 0.0),
                "eval_seconds": baseline_timings.get("baseline_eval", 0.0),
                "total_seconds": baseline_timings.get("baseline_total", 0.0),
                "tokens_per_second": baseline_count / max(baseline_timings.get("baseline_eval", 0.0), 1e-9),
                "avg_candidate_tokens": "",
            }
        )
    if args.compute_tree_ppl:
        tree_loss, tree_ppl, tree_count, tree_timings = compute_eval_loss(
            model, input_ids, prefill_tokens, args.eval_tokens, args.chunk_size, input_device, True, args.tree_prefill
        )
        rows.append({"mode": "tree", "loss": tree_loss, "ppl": tree_ppl, "token_count": tree_count, **metadata})
        timing_rows.append(
            {
                "mode": "tree",
                "prefill_seconds": tree_timings.get("tree_prefill", 0.0),
                "eval_seconds": tree_timings.get("tree_eval", 0.0),
                "total_seconds": tree_timings.get("tree_total", 0.0),
                "tokens_per_second": tree_count / max(tree_timings.get("tree_eval", 0.0), 1e-9),
                "avg_candidate_tokens": tree_timings.get("avg_candidate_tokens", ""),
            }
        )

    ppl_path = output_dir / "ppl_by_tree.csv"
    timing_path = output_dir / "timing_by_mode.csv"
    write_csv(ppl_path, rows, ppl_fields())
    write_csv(timing_path, timing_rows, timing_fields())
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "resolved": {
                    "prefill_tokens": prefill_tokens,
                    "eval_tokens": args.eval_tokens,
                    "eval_last_tokens_only": args.eval_last_tokens_only,
                    "total_tokenized_tokens_used": int(input_ids.numel()),
                    "layers": layer_indices,
                    "kv_heads": kv_head_indices,
                    "tree_branch_counts": branch_counts,
                },
                "timings": timing_rows,
                "paths": {"ppl_by_tree": str(ppl_path), "timing_by_mode": str(timing_path)},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote tree PPL outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
