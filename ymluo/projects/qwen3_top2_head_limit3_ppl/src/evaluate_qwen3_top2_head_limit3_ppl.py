from __future__ import annotations

import argparse
import csv
import json
import math
import re
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


DEFAULT_MODEL_PATH = "ymluo/models/Qwen3-0.6B"
DEFAULT_TEXT_PATH = "external/needle-in-a-haystack/needlehaystack/PaulGrahamEssays/worked.txt"

_ACTIVE_MODE: str = "baseline"
_ACTIVE_TOP_FRACTION: float = 0.02
_ACTIVE_MAX_HEADS_PER_TOKEN: int = 3
_ACTIVE_ALWAYS_KEEP_SELF: bool = True
_ACTIVE_LOAD_STATS: "LoadStats | None" = None
_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline, top2 historical attention, and top2 with max 3 heads per historical token."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/top2_head_limit3_ppl")
    parser.add_argument("--prefill_tokens", type=int, default=1024)
    parser.add_argument("--eval_tokens", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--max_heads_per_token", type=int, default=3)
    parser.add_argument(
        "--always_keep_self",
        type=str2bool,
        default=True,
        help="Keep each query token's own key/value even when pruning historical tokens.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--modes", default="baseline,top2,top2limit3score")
    parser.add_argument("--make_plots", type=str2bool, default=True)
    parser.add_argument("--plot_dpi", type=int, default=180)
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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_modes(spec: str) -> list[str]:
    modes = [part.strip().lower() for part in spec.split(",") if part.strip()]
    invalid = [
        mode
        for mode in modes
        if parse_mode_config(mode)[0]
        not in {
            "baseline",
            "top2",
            "limit_random",
            "limit_score",
            "limit_score_fill",
            "limit_score_gap",
            "limit_score_protect",
        }
    ]
    if invalid:
        raise ValueError(
            "Invalid modes: "
            f"{invalid}. Valid examples: baseline, top2, top2limit3, top2limit3score, "
            "top2limit3gap1p0, top2limit3protects16r1p0."
        )
    if not modes:
        raise ValueError("--modes cannot be empty.")
    return modes


def parse_mode_config(mode: str) -> tuple[str, int | None]:
    if mode == "baseline":
        return "baseline", None
    if mode == "top2":
        return "top2", None
    match = re.fullmatch(r"top2limit(\d+)protects(\d+)r([0-9]+(?:p[0-9]+)?)", mode)
    if match:
        return "limit_score_protect", int(match.group(1))
    match = re.fullmatch(r"top2limit(\d+)gap([0-9]+(?:p[0-9]+)?)", mode)
    if match:
        return "limit_score_gap", int(match.group(1))
    match = re.fullmatch(r"top2limit(\d+)(scorefill|score)?", mode)
    if match:
        max_heads = int(match.group(1))
        strategy = (
            "limit_score_fill"
            if match.group(2) == "scorefill"
            else "limit_score"
            if match.group(2) == "score"
            else "limit_random"
        )
        return strategy, max_heads
    return "invalid", None


def parse_gap_margin(mode: str) -> float | None:
    match = re.fullmatch(r"top2limit(\d+)gap([0-9]+(?:p[0-9]+)?)", mode)
    if not match:
        return None
    return float(match.group(2).replace("p", "."))


def parse_protect_params(mode: str) -> tuple[int, float] | None:
    match = re.fullmatch(r"top2limit(\d+)protects(\d+)r([0-9]+(?:p[0-9]+)?)", mode)
    if not match:
        return None
    sink_tokens = int(match.group(2))
    recent_percent = float(match.group(3).replace("p", "."))
    return sink_tokens, recent_percent / 100.0


@dataclass
class LoadStats:
    layer_count: int
    head_count: int

    def __post_init__(self) -> None:
        shape = (self.layer_count, self.head_count)
        self.query_counts = torch.zeros(shape, dtype=torch.long)
        self.history_token_counts = torch.zeros(shape, dtype=torch.long)
        self.original_kept = torch.zeros(shape, dtype=torch.long)
        self.final_kept = torch.zeros(shape, dtype=torch.long)
        self.removed = torch.zeros(shape, dtype=torch.long)
        self.max_final_kept_per_query = torch.zeros(shape, dtype=torch.long)
        self.max_removed_per_query = torch.zeros(shape, dtype=torch.long)

    def update(
        self,
        layer: int,
        original_keep: torch.Tensor,
        final_keep: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> None:
        # Tensors have shape [batch, heads, key].
        original_counts = original_keep.sum(dim=(0, 2)).cpu().to(torch.long)
        final_counts = final_keep.sum(dim=(0, 2)).cpu().to(torch.long)
        removed_counts = (original_keep & ~final_keep).sum(dim=(0, 2)).cpu().to(torch.long)
        history_counts = history_valid.sum(dim=(0, 2)).cpu().to(torch.long)
        final_per_query = final_keep.sum(dim=2).cpu().to(torch.long)
        removed_per_query = (original_keep & ~final_keep).sum(dim=2).cpu().to(torch.long)
        batch_count = int(original_keep.shape[0])
        self.query_counts[layer] += batch_count
        self.history_token_counts[layer] += history_counts
        self.original_kept[layer] += original_counts
        self.final_kept[layer] += final_counts
        self.removed[layer] += removed_counts
        self.max_final_kept_per_query[layer] = torch.maximum(
            self.max_final_kept_per_query[layer],
            final_per_query.max(dim=0).values,
        )
        self.max_removed_per_query[layer] = torch.maximum(
            self.max_removed_per_query[layer],
            removed_per_query.max(dim=0).values,
        )

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in range(self.layer_count):
            for head in range(self.head_count):
                query_count = int(self.query_counts[layer, head])
                history_count = int(self.history_token_counts[layer, head])
                original = int(self.original_kept[layer, head])
                final = int(self.final_kept[layer, head])
                removed = int(self.removed[layer, head])
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "query_count": query_count,
                        "history_token_cases": history_count,
                        "original_top2_kept": original,
                        "final_kept_after_limit3": final,
                        "removed_by_limit3": removed,
                        "final_kept_per_query_mean": final / query_count if query_count else 0.0,
                        "original_kept_per_query_mean": original / query_count if query_count else 0.0,
                        "removed_per_query_mean": removed / query_count if query_count else 0.0,
                        "kept_fraction_of_original_top2": final / original if original else 0.0,
                        "kept_fraction_of_history_cases": final / history_count if history_count else 0.0,
                        "max_final_kept_per_query": int(self.max_final_kept_per_query[layer, head]),
                        "max_removed_per_query": int(self.max_removed_per_query[layer, head]),
                    }
                )
        return rows


def _top2_history_keep_for_query(row_scores: torch.Tensor, finite: torch.Tensor, top_fraction: float) -> torch.Tensor:
    # row_scores and finite: [batch, heads, key].
    history_valid = finite.clone()
    valid_count = int(finite[0, 0].sum().item())
    if valid_count <= 1:
        return torch.zeros_like(finite, dtype=torch.bool)
    self_index = valid_count - 1
    history_valid[:, :, self_index] = False
    history_count = valid_count - 1
    keep_count = min(history_count, max(1, math.ceil(top_fraction * history_count)))
    history_scores = row_scores.masked_fill(~history_valid, torch.finfo(row_scores.dtype).min)
    _, top_indices = torch.topk(history_scores, k=keep_count, dim=-1, largest=True)
    keep = torch.zeros_like(finite, dtype=torch.bool)
    keep.scatter_(-1, top_indices, True)
    keep &= history_valid
    return keep


def _limit_heads_per_token_random(original_keep: torch.Tensor, max_heads: int) -> torch.Tensor:
    # original_keep: [batch, heads, key], only historical keys are true.
    if max_heads <= 0:
        return torch.zeros_like(original_keep)
    batch, _, key_count = original_keep.shape
    final_keep = original_keep.clone()
    counts = original_keep.sum(dim=1)
    over = torch.nonzero(counts > max_heads, as_tuple=False)
    for batch_index, key_index in over.tolist():
        selected_heads = torch.nonzero(original_keep[batch_index, :, key_index], as_tuple=False).flatten()
        if selected_heads.numel() <= max_heads:
            continue
        perm = torch.randperm(selected_heads.numel(), device=selected_heads.device)
        keep_heads = selected_heads[perm[:max_heads]]
        final_keep[batch_index, :, key_index] = False
        final_keep[batch_index, keep_heads, key_index] = True
    return final_keep


def _limit_heads_per_token_by_score(
    original_keep: torch.Tensor,
    row_scores: torch.Tensor,
    max_heads: int,
) -> torch.Tensor:
    # original_keep and row_scores: [batch, heads, key].
    if max_heads <= 0:
        return torch.zeros_like(original_keep)
    final_keep = original_keep.clone()
    counts = original_keep.sum(dim=1)
    over = torch.nonzero(counts > max_heads, as_tuple=False)
    for batch_index, key_index in over.tolist():
        selected_heads = torch.nonzero(original_keep[batch_index, :, key_index], as_tuple=False).flatten()
        if selected_heads.numel() <= max_heads:
            continue
        selected_scores = row_scores[batch_index, selected_heads, key_index]
        _, order = torch.topk(selected_scores, k=max_heads, largest=True)
        keep_heads = selected_heads[order]
        final_keep[batch_index, :, key_index] = False
        final_keep[batch_index, keep_heads, key_index] = True
    return final_keep


def _limit_heads_per_token_by_score_fill(
    original_keep: torch.Tensor,
    row_scores: torch.Tensor,
    max_heads: int,
    history_valid: torch.Tensor,
) -> torch.Tensor:
    # First enforce the token cap by score, then fill each head back to its original top2 load.
    final_keep = _limit_heads_per_token_by_score(original_keep, row_scores, max_heads)
    batch, head_count, _ = original_keep.shape
    target_per_head = original_keep.sum(dim=2)
    token_counts = final_keep.sum(dim=1)
    fill_scores = row_scores.masked_fill(~history_valid, torch.finfo(row_scores.dtype).min)
    sorted_indices = torch.argsort(fill_scores, dim=-1, descending=True)
    for batch_index in range(batch):
        for head in range(head_count):
            target = int(target_per_head[batch_index, head].item())
            current = int(final_keep[batch_index, head].sum().item())
            if current >= target:
                continue
            for key_index in sorted_indices[batch_index, head].tolist():
                if current >= target:
                    break
                if not bool(history_valid[batch_index, head, key_index]):
                    continue
                if bool(final_keep[batch_index, head, key_index]):
                    continue
                if int(token_counts[batch_index, key_index].item()) >= max_heads:
                    continue
                final_keep[batch_index, head, key_index] = True
                token_counts[batch_index, key_index] += 1
                current += 1
    return final_keep


def _limit_heads_per_token_by_score_gap(
    original_keep: torch.Tensor,
    row_scores: torch.Tensor,
    min_heads: int,
    margin: float,
) -> torch.Tensor:
    # Keep at least min_heads by score. Also keep extra selected heads if their
    # score is within margin of the min_heads-th kept score for that token.
    if min_heads <= 0:
        return torch.zeros_like(original_keep)
    final_keep = original_keep.clone()
    counts = original_keep.sum(dim=1)
    over = torch.nonzero(counts > min_heads, as_tuple=False)
    for batch_index, key_index in over.tolist():
        selected_heads = torch.nonzero(original_keep[batch_index, :, key_index], as_tuple=False).flatten()
        selected_scores = row_scores[batch_index, selected_heads, key_index]
        sorted_scores, order = torch.sort(selected_scores, descending=True)
        threshold = sorted_scores[min_heads - 1] - margin
        keep_heads = selected_heads[order[sorted_scores >= threshold]]
        final_keep[batch_index, :, key_index] = False
        final_keep[batch_index, keep_heads, key_index] = True
    return final_keep


def _protected_history_keys(
    finite: torch.Tensor,
    sink_tokens: int,
    recent_fraction: float,
) -> torch.Tensor:
    # Output shape: [batch, key]. True means the historical token should not be
    # limited across heads for this query.
    batch, _, key_count = finite.shape
    protected = torch.zeros((batch, key_count), dtype=torch.bool, device=finite.device)
    valid_count = int(finite[0, 0].sum().item())
    history_count = max(0, valid_count - 1)
    if history_count <= 0:
        return protected
    if sink_tokens > 0:
        protected[:, : min(sink_tokens, history_count)] = True
    if recent_fraction > 0:
        recent_count = min(history_count, max(1, math.ceil(recent_fraction * history_count)))
        protected[:, history_count - recent_count : history_count] = True
    return protected


def _limit_heads_per_token_by_score_protected(
    original_keep: torch.Tensor,
    row_scores: torch.Tensor,
    max_heads: int,
    protected_keys: torch.Tensor,
) -> torch.Tensor:
    # Protected historical tokens keep all original top2-selected heads. Other
    # historical tokens keep only the max_heads selected heads with largest score.
    if max_heads <= 0:
        return torch.zeros_like(original_keep)
    final_keep = original_keep.clone()
    counts = original_keep.sum(dim=1)
    over = torch.nonzero((counts > max_heads) & ~protected_keys, as_tuple=False)
    for batch_index, key_index in over.tolist():
        selected_heads = torch.nonzero(original_keep[batch_index, :, key_index], as_tuple=False).flatten()
        if selected_heads.numel() <= max_heads:
            continue
        selected_scores = row_scores[batch_index, selected_heads, key_index]
        _, order = torch.topk(selected_scores, k=max_heads, largest=True)
        keep_heads = selected_heads[order]
        final_keep[batch_index, :, key_index] = False
        final_keep[batch_index, keep_heads, key_index] = True
    return final_keep


def _limited_eager_attention_forward(
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

    mode_kind, mode_max_heads = parse_mode_config(_ACTIVE_MODE)
    if mode_kind in {
        "top2",
        "limit_random",
        "limit_score",
        "limit_score_fill",
        "limit_score_gap",
        "limit_score_protect",
    }:
        final_keep = torch.zeros_like(scores, dtype=torch.bool)
        query_count = scores.shape[-2]
        for query_index in range(query_count):
            row = scores[:, :, query_index, :]
            finite = torch.isfinite(row)
            history_original = _top2_history_keep_for_query(row, finite, _ACTIVE_TOP_FRACTION)
            if mode_kind == "limit_random":
                history_final = _limit_heads_per_token_random(
                    history_original,
                    mode_max_heads if mode_max_heads is not None else _ACTIVE_MAX_HEADS_PER_TOKEN,
                )
                if _ACTIVE_LOAD_STATS is not None:
                    layer_idx = int(getattr(module, "layer_idx", 0))
                    history_valid = finite.clone()
                    valid_count = int(finite[0, 0].sum().item())
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_original, history_final, history_valid)
            elif mode_kind == "limit_score":
                history_final = _limit_heads_per_token_by_score(
                    history_original,
                    row,
                    mode_max_heads if mode_max_heads is not None else _ACTIVE_MAX_HEADS_PER_TOKEN,
                )
                if _ACTIVE_LOAD_STATS is not None:
                    layer_idx = int(getattr(module, "layer_idx", 0))
                    history_valid = finite.clone()
                    valid_count = int(finite[0, 0].sum().item())
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_original, history_final, history_valid)
            elif mode_kind == "limit_score_fill":
                layer_idx = int(getattr(module, "layer_idx", 0))
                history_valid = finite.clone()
                valid_count = int(finite[0, 0].sum().item())
                if valid_count > 0:
                    history_valid[:, :, valid_count - 1] = False
                history_final = _limit_heads_per_token_by_score_fill(
                    history_original,
                    row,
                    mode_max_heads if mode_max_heads is not None else _ACTIVE_MAX_HEADS_PER_TOKEN,
                    history_valid,
                )
                if _ACTIVE_LOAD_STATS is not None:
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_original, history_final, history_valid)
            elif mode_kind == "limit_score_gap":
                history_final = _limit_heads_per_token_by_score_gap(
                    history_original,
                    row,
                    mode_max_heads if mode_max_heads is not None else _ACTIVE_MAX_HEADS_PER_TOKEN,
                    parse_gap_margin(_ACTIVE_MODE) or 0.0,
                )
                if _ACTIVE_LOAD_STATS is not None:
                    layer_idx = int(getattr(module, "layer_idx", 0))
                    history_valid = finite.clone()
                    valid_count = int(finite[0, 0].sum().item())
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_original, history_final, history_valid)
            elif mode_kind == "limit_score_protect":
                sink_tokens, recent_fraction = parse_protect_params(_ACTIVE_MODE) or (0, 0.0)
                protected_keys = _protected_history_keys(finite, sink_tokens, recent_fraction)
                history_final = _limit_heads_per_token_by_score_protected(
                    history_original,
                    row,
                    mode_max_heads if mode_max_heads is not None else _ACTIVE_MAX_HEADS_PER_TOKEN,
                    protected_keys,
                )
                if _ACTIVE_LOAD_STATS is not None:
                    layer_idx = int(getattr(module, "layer_idx", 0))
                    history_valid = finite.clone()
                    valid_count = int(finite[0, 0].sum().item())
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_original, history_final, history_valid)
            else:
                history_final = history_original
            keep = history_final
            if _ACTIVE_ALWAYS_KEEP_SELF:
                valid_count = int(finite[0, 0].sum().item())
                if valid_count > 0:
                    keep[:, :, valid_count - 1] = True
            final_keep[:, :, query_index, :] = keep
        scores = scores.masked_fill(~final_keep, torch.finfo(scores.dtype).min)

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
        setattr(modeling_qwen3, "eager_attention_forward", _limited_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _limited_eager_attention_forward


@contextmanager
def attention_mode(
    mode: str,
    top_fraction: float,
    max_heads_per_token: int,
    always_keep_self: bool,
    load_stats: LoadStats | None,
):
    global _ACTIVE_MODE, _ACTIVE_TOP_FRACTION, _ACTIVE_MAX_HEADS_PER_TOKEN, _ACTIVE_ALWAYS_KEEP_SELF, _ACTIVE_LOAD_STATS
    previous = (
        _ACTIVE_MODE,
        _ACTIVE_TOP_FRACTION,
        _ACTIVE_MAX_HEADS_PER_TOKEN,
        _ACTIVE_ALWAYS_KEEP_SELF,
        _ACTIVE_LOAD_STATS,
    )
    _ACTIVE_MODE = mode
    _ACTIVE_TOP_FRACTION = top_fraction
    _ACTIVE_MAX_HEADS_PER_TOKEN = max_heads_per_token
    _ACTIVE_ALWAYS_KEEP_SELF = always_keep_self
    _ACTIVE_LOAD_STATS = load_stats
    try:
        yield
    finally:
        (
            _ACTIVE_MODE,
            _ACTIVE_TOP_FRACTION,
            _ACTIVE_MAX_HEADS_PER_TOKEN,
            _ACTIVE_ALWAYS_KEEP_SELF,
            _ACTIVE_LOAD_STATS,
        ) = previous


@torch.inference_mode()
def prefill_cache(
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
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    if last_logits is None:
        raise RuntimeError("Prefill produced no logits.")
    return past_key_values, last_logits


@torch.inference_mode()
def compute_eval_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    mode: str,
    top_fraction: float,
    max_heads_per_token: int,
    always_keep_self: bool,
    load_stats: LoadStats | None,
) -> tuple[float, float, int, float]:
    print(f"starting mode: {mode}", flush=True)
    started = time.perf_counter()
    past_key_values, prev_logits = prefill_cache(model, input_ids, prefill_tokens, chunk_size, input_device)
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
        print(f"ppl {mode} chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        with attention_mode(mode, top_fraction, max_heads_per_token, always_keep_self, load_stats):
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
    return mean_loss, math.exp(min(mean_loss, 80.0)), total_count, time.perf_counter() - started


def plot_outputs(output_dir: Path, ppl_rows: list[dict[str, Any]], load_rows: list[dict[str, Any]], dpi: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if ppl_rows:
        labels = [str(row["mode"]) for row in ppl_rows]
        ppls = [float(row["ppl"]) for row in ppl_rows]
        losses = [float(row["loss"]) for row in ppl_rows]
        fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
        colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#ff9da6", "#9d755d"]
        bars = ax.bar(labels, ppls, color=[colors[index % len(colors)] for index in range(len(labels))])
        ax.set_title("PPL by attention selection mode")
        ax.set_xlabel("Attention selection mode")
        ax.set_ylabel("Perplexity on evaluation tokens")
        ax.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, ppls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
        fig.tight_layout()
        path = plot_dir / "ppl_by_mode.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

        fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
        bars = ax.bar(labels, ppls, color=[colors[index % len(colors)] for index in range(len(labels))])
        ax.set_yscale("log")
        ax.set_title("PPL by attention selection mode (log scale)")
        ax.set_xlabel("Attention selection mode")
        ax.set_ylabel("Perplexity on evaluation tokens, log scale")
        ax.grid(True, axis="y", alpha=0.25, which="both")
        for bar, value in zip(bars, ppls):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}", ha="center", va="bottom")
        fig.tight_layout()
        path = plot_dir / "ppl_by_mode_logy.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

        fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
        bars = ax.bar(labels, losses, color=[colors[index % len(colors)] for index in range(len(labels))])
        ax.set_title("Mean next-token loss by attention selection mode")
        ax.set_xlabel("Attention selection mode")
        ax.set_ylabel("Mean cross entropy loss")
        ax.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, losses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
        fig.tight_layout()
        path = plot_dir / "loss_by_mode.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

    if load_rows:
        focus_mode = "top2limit3score" if any(row.get("mode") == "top2limit3score" for row in load_rows) else str(load_rows[0]["mode"])
        focus_rows = [row for row in load_rows if row.get("mode") == focus_mode]
        layers = sorted({int(row["layer"]) for row in focus_rows})
        heads = sorted({int(row["head"]) for row in focus_rows})
        matrix = torch.zeros((len(layers), len(heads)), dtype=torch.float32)
        kept_fraction = torch.zeros((len(layers), len(heads)), dtype=torch.float32)
        layer_to_pos = {layer: pos for pos, layer in enumerate(layers)}
        head_to_pos = {head: pos for pos, head in enumerate(heads)}
        for row in focus_rows:
            layer = int(row["layer"])
            head = int(row["head"])
            matrix[layer_to_pos[layer], head_to_pos[head]] = float(row["final_kept_per_query_mean"])
            kept_fraction[layer_to_pos[layer], head_to_pos[head]] = float(row["kept_fraction_of_original_top2"])

        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
        image = ax.imshow(matrix.numpy(), aspect="auto", cmap="viridis")
        ax.set_title(f"{focus_mode} historical-token load per layer and head")
        ax.set_xlabel("Attention head index")
        ax.set_ylabel("Layer index")
        ax.set_xticks(heads)
        ax.set_yticks(layers)
        fig.colorbar(image, ax=ax, label="Mean kept historical tokens per query")
        fig.tight_layout()
        path = plot_dir / f"{focus_mode}_head_load_heatmap.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

        layer_means = matrix.mean(dim=1).tolist()
        layer_mins = matrix.min(dim=1).values.tolist()
        layer_maxs = matrix.max(dim=1).values.tolist()
        fig, ax = plt.subplots(figsize=(11, 4), dpi=dpi)
        ax.plot(layers, layer_means, marker="o", label="mean over heads")
        ax.fill_between(layers, layer_mins, layer_maxs, alpha=0.2, label="min-max over heads")
        ax.set_title(f"{focus_mode} head load by layer")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Mean kept historical tokens per query")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = plot_dir / f"{focus_mode}_head_load_by_layer.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
        image = ax.imshow(kept_fraction.numpy(), aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"Fraction of original top2 selections kept after {focus_mode}")
        ax.set_xlabel("Attention head index")
        ax.set_ylabel("Layer index")
        ax.set_xticks(heads)
        ax.set_yticks(layers)
        fig.colorbar(image, ax=ax, label="Final kept / original top2 kept")
        fig.tight_layout()
        path = plot_dir / f"{focus_mode}_kept_fraction_heatmap.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    if args.prefill_tokens <= 0 or args.eval_tokens <= 0:
        raise ValueError("--prefill_tokens and --eval_tokens must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive.")
    if not (0.0 < args.top_fraction <= 1.0):
        raise ValueError("--top_fraction must be in (0, 1].")
    if args.max_heads_per_token <= 0:
        raise ValueError("--max_heads_per_token must be positive.")
    modes = parse_modes(args.modes)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    total_needed = args.prefill_tokens + args.eval_tokens
    if args.require_total_tokens and len(token_ids) < total_needed:
        raise ValueError(f"Tokenization produced {len(token_ids)} tokens, fewer than {total_needed}.")
    token_ids = token_ids[:total_needed]
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
    install_qwen3_attention_patch()

    layer_count = int(getattr(model.config, "num_hidden_layers"))
    head_count = int(getattr(model.config, "num_attention_heads"))
    input_device = pick_input_device(model, requested_device)

    ppl_rows: list[dict[str, Any]] = []
    load_rows: list[dict[str, Any]] = []
    for mode in modes:
        mode_kind, mode_max_heads = parse_mode_config(mode)
        protect_params = parse_protect_params(mode)
        stats = (
            LoadStats(layer_count, head_count)
            if mode_kind in {"limit_random", "limit_score", "limit_score_fill", "limit_score_gap", "limit_score_protect"}
            else None
        )
        loss, ppl, token_count, seconds = compute_eval_loss(
            model,
            input_ids,
            args.prefill_tokens,
            args.eval_tokens,
            args.chunk_size,
            input_device,
            mode,
            args.top_fraction,
            mode_max_heads if mode_max_heads is not None else args.max_heads_per_token,
            args.always_keep_self,
            stats,
        )
        ppl_rows.append(
            {
                "mode": mode,
                "loss": loss,
                "ppl": ppl,
                "token_count": token_count,
                "seconds": seconds,
                "top_fraction": args.top_fraction if mode != "baseline" else "",
                "max_heads_per_token": (
                    mode_max_heads
                    if mode_kind
                    in {"limit_random", "limit_score", "limit_score_fill", "limit_score_gap", "limit_score_protect"}
                    else ""
                ),
                "limit_strategy": (
                    "random"
                    if mode_kind == "limit_random"
                    else "score"
                    if mode_kind == "limit_score"
                    else "score_fill"
                    if mode_kind == "limit_score_fill"
                    else "score_gap"
                    if mode_kind == "limit_score_gap"
                    else "score_protect"
                    if mode_kind == "limit_score_protect"
                    else ""
                ),
                "protected_sink_tokens": protect_params[0] if protect_params else "",
                "protected_recent_fraction": protect_params[1] if protect_params else "",
                "always_keep_self": args.always_keep_self if mode != "baseline" else "",
            }
        )
        if stats is not None:
            for row in stats.rows():
                row = dict(row)
                row["mode"] = mode
                row["limit_strategy"] = (
                    "random"
                    if mode_kind == "limit_random"
                    else "score"
                    if mode_kind == "limit_score"
                    else "score_fill"
                    if mode_kind == "limit_score_fill"
                    else "score_gap"
                    if mode_kind == "limit_score_gap"
                    else "score_protect"
                    if mode_kind == "limit_score_protect"
                    else ""
                )
                row["max_heads_per_token"] = mode_max_heads
                row["protected_sink_tokens"] = protect_params[0] if protect_params else ""
                row["protected_recent_fraction"] = protect_params[1] if protect_params else ""
                load_rows.append(row)
    write_csv(
        output_dir / "ppl_by_mode.csv",
        ppl_rows,
        [
            "mode",
            "loss",
            "ppl",
            "token_count",
            "seconds",
            "top_fraction",
            "max_heads_per_token",
            "limit_strategy",
            "protected_sink_tokens",
            "protected_recent_fraction",
            "always_keep_self",
        ],
    )

    if load_rows:
        write_csv(
            output_dir / "limit_load_by_head.csv",
            load_rows,
            [
                "mode",
                "limit_strategy",
                "max_heads_per_token",
                "protected_sink_tokens",
                "protected_recent_fraction",
                "layer",
                "head",
                "query_count",
                "history_token_cases",
                "original_top2_kept",
                "final_kept_after_limit3",
                "removed_by_limit3",
                "final_kept_per_query_mean",
                "original_kept_per_query_mean",
                "removed_per_query_mean",
                "kept_fraction_of_original_top2",
                "kept_fraction_of_history_cases",
                "max_final_kept_per_query",
                "max_removed_per_query",
            ],
        )
        score3_rows = [row for row in load_rows if row["mode"] == "top2limit3score"]
        if score3_rows:
            write_csv(
                output_dir / "top2limit3score_load_by_head.csv",
                score3_rows,
                [
                    "mode",
                    "limit_strategy",
                    "max_heads_per_token",
                    "protected_sink_tokens",
                    "protected_recent_fraction",
                    "layer",
                    "head",
                    "query_count",
                    "history_token_cases",
                    "original_top2_kept",
                    "final_kept_after_limit3",
                    "removed_by_limit3",
                    "final_kept_per_query_mean",
                    "original_kept_per_query_mean",
                    "removed_per_query_mean",
                    "kept_fraction_of_original_top2",
                    "kept_fraction_of_history_cases",
                    "max_final_kept_per_query",
                    "max_removed_per_query",
                ],
            )

    plot_paths = plot_outputs(output_dir, ppl_rows, load_rows, args.plot_dpi) if args.make_plots else []
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "resolved": {
                    "total_tokens_used": int(input_ids.numel()),
                    "prefill_tokens": args.prefill_tokens,
                    "eval_tokens": args.eval_tokens,
                    "layer_count": layer_count,
                    "head_count": head_count,
                    "modes": modes,
                    "history_selection_rule": "top ceil(top_fraction * history_token_count) historical keys per head",
                    "limit_rule": (
                        "if a historical key is selected by more than max_heads_per_token heads, "
                        "random modes keep a random subset and score modes keep the highest-score heads"
                    ),
                    "self_token_rule": "kept unconditionally when always_keep_self=true",
                },
                "paths": {
                    "ppl_by_mode": str(output_dir / "ppl_by_mode.csv"),
                    "limit_load_by_head": str(output_dir / "limit_load_by_head.csv") if load_rows else None,
                    "top2limit3score_load_by_head": (
                        str(output_dir / "top2limit3score_load_by_head.csv")
                        if any(row["mode"] == "top2limit3score" for row in load_rows)
                        else None
                    ),
                    "plots": plot_paths,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
