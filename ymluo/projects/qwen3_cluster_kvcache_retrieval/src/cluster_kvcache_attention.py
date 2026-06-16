from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class ClusterKVConfig:
    mode: str = "baseline"
    cluster_size: int = 50
    keep_ratio: float = 0.02
    force_endpoints: bool = True
    endpoints_count_in_budget: bool = True
    min_keep_clusters: int = 1
    profile: bool = False


@dataclass
class ProfileBucket:
    calls: int = 0
    elapsed_ms: float = 0.0


@dataclass
class CudaEventProfiler:
    enabled: bool = False
    pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    buckets: dict[str, ProfileBucket] = field(default_factory=dict)

    def record_segment(self, name: str, start: torch.cuda.Event, end: torch.cuda.Event) -> None:
        if self.enabled:
            self.pending.append((name, start, end))

    def synchronize_and_flush(self) -> None:
        if not self.enabled or not self.pending:
            return
        torch.cuda.synchronize()
        for name, start, end in self.pending:
            bucket = self.buckets.setdefault(name, ProfileBucket())
            bucket.calls += 1
            bucket.elapsed_ms += float(start.elapsed_time(end))
        self.pending.clear()

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "calls": bucket.calls,
                "elapsed_ms": bucket.elapsed_ms,
                "mean_ms": bucket.elapsed_ms / bucket.calls if bucket.calls else 0.0,
            }
            for name, bucket in sorted(self.buckets.items())
        }

    def reset(self) -> None:
        self.pending.clear()
        self.buckets.clear()


PROFILER = CudaEventProfiler()
_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, kv_heads * n_rep, seq_len, head_dim)


def _event() -> torch.cuda.Event | None:
    if PROFILER.enabled and torch.cuda.is_available():
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
    return None


def _record(name: str, start: torch.cuda.Event | None) -> torch.cuda.Event | None:
    if start is None:
        return None
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    PROFILER.record_segment(name, start, end)
    next_start = torch.cuda.Event(enable_timing=True)
    next_start.record()
    return next_start


def _cluster_centers(key_states: torch.Tensor, cluster_size: int) -> torch.Tensor:
    batch, heads, key_len, head_dim = key_states.shape
    cluster_count = math.ceil(key_len / cluster_size)
    padded_len = cluster_count * cluster_size
    if padded_len != key_len:
        pad_tokens = padded_len - key_len
        key_states = F.pad(key_states, (0, 0, 0, pad_tokens))
    grouped = key_states.view(batch, heads, cluster_count, cluster_size, head_dim)
    centers = grouped.sum(dim=-2)
    if padded_len != key_len:
        counts = torch.full((cluster_count,), cluster_size, device=key_states.device, dtype=key_states.dtype)
        counts[-1] = key_len - (cluster_count - 1) * cluster_size
        centers = centers / counts.view(1, 1, cluster_count, 1).clamp_min(1)
    else:
        centers = centers / float(cluster_size)
    return centers


def _cluster_counts(key_len: int, cluster_size: int, device: torch.device) -> torch.Tensor:
    cluster_count = math.ceil(key_len / cluster_size)
    counts = torch.full((cluster_count,), cluster_size, device=device, dtype=torch.float32)
    counts[-1] = key_len - (cluster_count - 1) * cluster_size
    return counts.clamp_min_(1.0)


def _cache_is_valid(cache: dict[str, Any] | None, key_states: torch.Tensor, cfg: ClusterKVConfig) -> bool:
    if cache is None:
        return False
    return (
        cache.get("cluster_size") == cfg.cluster_size
        and cache.get("batch") == key_states.shape[0]
        and cache.get("heads") == key_states.shape[1]
        and cache.get("head_dim") == key_states.shape[-1]
        and cache.get("device") == key_states.device
    )


def _build_center_cache(module: torch.nn.Module, key_states: torch.Tensor, cfg: ClusterKVConfig) -> torch.Tensor:
    centers = _cluster_centers(key_states, cfg.cluster_size)
    cache = {
        "cluster_size": cfg.cluster_size,
        "batch": key_states.shape[0],
        "heads": key_states.shape[1],
        "head_dim": key_states.shape[-1],
        "device": key_states.device,
        "key_len": key_states.shape[-2],
        "centers": centers,
        "counts": _cluster_counts(key_states.shape[-2], cfg.cluster_size, key_states.device),
    }
    module._cluster_kv_center_cache = cache
    return centers


def _cached_cluster_centers(
    module: torch.nn.Module,
    key_states: torch.Tensor,
    cfg: ClusterKVConfig,
) -> torch.Tensor:
    key_len = key_states.shape[-2]
    cache = getattr(module, "_cluster_kv_center_cache", None)
    if not _cache_is_valid(cache, key_states, cfg):
        return _build_center_cache(module, key_states, cfg)

    cached_len = int(cache["key_len"])
    if cached_len == key_len:
        return cache["centers"]
    if cached_len + 1 != key_len:
        return _build_center_cache(module, key_states, cfg)

    token_index = key_len - 1
    cluster_index = token_index // cfg.cluster_size
    position_in_cluster = token_index % cfg.cluster_size
    new_key = key_states[:, :, -1, :]
    centers = cache["centers"]
    counts = cache["counts"]

    if position_in_cluster == 0:
        centers = torch.cat([centers, new_key.unsqueeze(-2)], dim=-2)
        counts = torch.cat([counts, torch.ones(1, device=key_states.device, dtype=torch.float32)])
    else:
        old_count = counts[cluster_index].to(dtype=centers.dtype)
        new_count = old_count + 1.0
        centers[..., cluster_index, :] = (
            centers[..., cluster_index, :] * old_count + new_key
        ) / new_count
        counts[cluster_index] = counts[cluster_index] + 1.0

    cache["key_len"] = key_len
    cache["centers"] = centers
    cache["counts"] = counts
    return centers


def _select_cluster_ids(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cfg: ClusterKVConfig,
) -> torch.Tensor:
    key_len = key_states.shape[-2]
    cluster_count = math.ceil(key_len / cfg.cluster_size)
    keep_count = max(cfg.min_keep_clusters, math.ceil(cfg.keep_ratio * cluster_count))
    keep_count = min(keep_count, cluster_count)

    centers = _cached_cluster_centers(module, key_states, cfg)
    q = F.normalize(query_states.float(), p=2, dim=-1)
    c = F.normalize(centers.float(), p=2, dim=-1)
    scores = torch.einsum("bhqd,bhcd->bhqc", q, c)

    select_count = keep_count
    if cfg.force_endpoints and not cfg.endpoints_count_in_budget:
        select_count = min(cluster_count, keep_count + min(cluster_count, 2))
    if cfg.force_endpoints and cfg.endpoints_count_in_budget:
        selectable_scores = scores.clone()
        selectable_scores[..., 0] = float("inf")
        selectable_scores[..., -1] = float("inf")
    elif cfg.force_endpoints:
        selectable_scores = scores.clone()
        selectable_scores[..., 0] = float("inf")
        selectable_scores[..., -1] = float("inf")
    else:
        selectable_scores = scores

    return torch.topk(selectable_scores, k=select_count, dim=-1).indices.sort(dim=-1).values


def _selected_cluster_mask(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cfg: ClusterKVConfig,
) -> torch.Tensor:
    key_len = key_states.shape[-2]
    cluster_ids = _select_cluster_ids(module, query_states, key_states, cfg)
    token_cluster_ids = torch.arange(key_len, device=key_states.device) // cfg.cluster_size
    return (cluster_ids.unsqueeze(-1) == token_cluster_ids.view(1, 1, 1, 1, key_len)).any(dim=-2)


def _gather_selected_kv(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cluster_ids: torch.Tensor,
    cluster_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Optimized decode path for q_len=1. selected_positions is per batch/head.
    batch, heads, key_len, head_dim = key_states.shape
    offsets = torch.arange(cluster_size, device=key_states.device)
    positions = cluster_ids.squeeze(-2).unsqueeze(-1) * cluster_size + offsets
    positions = positions.reshape(batch, heads, -1)
    valid = positions < key_len
    positions = positions.clamp_max(key_len - 1)
    gather_index = positions.unsqueeze(-1).expand(batch, heads, positions.shape[-1], head_dim)
    selected_key = key_states.gather(dim=2, index=gather_index)
    selected_value = value_states.gather(dim=2, index=gather_index)
    return selected_key, selected_value, valid


def cluster_kvcache_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg: ClusterKVConfig = module.cluster_kv_config
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states = repeat_kv(key_states, repeat_groups)
        value_states = repeat_kv(value_states, repeat_groups)

    if scaling is None:
        scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(query_states.shape[-1])))

    if cfg.mode == "cluster" and query_states.shape[-2] == 1:
        start = _event()
        cluster_ids = _select_cluster_ids(module, query_states, key_states, cfg)
        start = _record("cluster_center_score_topk_ms", start)
        selected_key, selected_value, valid = _gather_selected_kv(
            key_states, value_states, cluster_ids, cfg.cluster_size
        )
        start = _record("gather_selected_kv_ms", start)
        scores = torch.matmul(query_states, selected_key.transpose(2, 3)) * scaling
        scores = scores.masked_fill(~valid.unsqueeze(-2), torch.finfo(scores.dtype).min)
        attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_output = torch.matmul(attention_weights, selected_value)
        attention_output = attention_output.transpose(1, 2).contiguous()
        _record("sparse_qk_softmax_value_ms", start)
        return attention_output, attention_weights

    start = _event()
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    start = _record("qk_scores_ms", start)

    start = _record("cluster_select_and_mask_ms", start)

    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=True)
    attention_output = torch.matmul(attention_weights, value_states)
    attention_output = attention_output.transpose(1, 2).contiguous()
    _record("softmax_value_ms", start)
    return attention_output, attention_weights


def install_qwen3_cluster_attention_patch(model: torch.nn.Module, cfg: ClusterKVConfig) -> None:
    global _ORIGINAL_EAGER_ATTENTION_FORWARD
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except Exception as exc:
        raise RuntimeError("Could not import transformers.models.qwen3.modeling_qwen3.") from exc

    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")

    if cfg.mode == "baseline":
        setattr(modeling_qwen3, "eager_attention_forward", _ORIGINAL_EAGER_ATTENTION_FORWARD)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _ORIGINAL_EAGER_ATTENTION_FORWARD
        PROFILER.enabled = cfg.profile
        return

    setattr(modeling_qwen3, "eager_attention_forward", cluster_kvcache_attention_forward)
    if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
        modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = cluster_kvcache_attention_forward

    patched = 0
    for module in model.modules():
        if module.__class__.__name__ == "Qwen3Attention":
            module.cluster_kv_config = cfg
            if hasattr(module, "_cluster_kv_center_cache"):
                delattr(module, "_cluster_kv_center_cache")
            patched += 1
    if patched == 0:
        raise ValueError("No Qwen3Attention modules found. Check model class and transformers version.")

    model.config._attn_implementation = "eager"
    PROFILER.enabled = cfg.profile
