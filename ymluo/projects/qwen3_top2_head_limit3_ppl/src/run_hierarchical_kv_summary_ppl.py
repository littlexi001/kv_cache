from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    clone_past_key_values,
    model_forward,
    pick_input_device,
    prefill_cache,
    read_text_prefix,
    resolve_dtype,
)


_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_MODE = "baseline"
_ACTIVE_HIER_STATE: "HierKVState | None" = None
_ACTIVE_SINK_TOKENS = 64
_ACTIVE_RECENT_TOKENS = 512


@dataclass(frozen=True)
class RangeUnit:
    unit_id: int
    level: str
    start: int
    end: int
    parent_id: int | None = None

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class HierKVConfig:
    prefill_tokens: int
    sink_tokens: int
    recent_tokens: int
    block_tokens: int
    mid_tokens: int
    leaf_tokens: int
    top_blocks: int
    mids_per_block: int
    leafs_per_mid: int
    seed_leafs: int
    route_refresh_tokens: int


class HierKVState:
    def __init__(self, config: HierKVConfig) -> None:
        self.config = config
        self.blocks, self.mids, self.leafs = build_fixed_hierarchy(
            config.prefill_tokens,
            config.block_tokens,
            config.mid_tokens,
            config.leaf_tokens,
        )
        self.block_to_mids: dict[int, list[int]] = defaultdict(list)
        self.mid_to_leafs: dict[int, list[int]] = defaultdict(list)
        for mid in self.mids:
            if mid.parent_id is not None:
                self.block_to_mids[int(mid.parent_id)].append(mid.unit_id)
        for leaf in self.leafs:
            if leaf.parent_id is not None:
                self.mid_to_leafs[int(leaf.parent_id)].append(leaf.unit_id)

        self._summaries: dict[tuple[int, str, torch.device], torch.Tensor] = {}
        self._last_selected: dict[tuple[int, int, int], tuple[list[int], list[int], list[int]]] = {}
        self.selected_block_counts: list[int] = []
        self.selected_mid_counts: list[int] = []
        self.selected_leaf_counts: list[int] = []
        self.selected_leaf_tokens: list[int] = []
        self.kept_attention_tokens: list[int] = []
        self.summary_score_counts: list[int] = []
        self.route_compute_calls = 0
        self.route_apply_calls = 0

    def _range_mean(self, key_states: torch.Tensor, ranges: list[RangeUnit], history_count: int) -> torch.Tensor:
        parts = []
        for item in ranges:
            start = max(0, min(item.start, history_count))
            end = max(start + 1, min(item.end, history_count))
            parts.append(key_states[:, :, start:end, :].float().mean(dim=2))
        if not parts:
            batch, heads, _, dim = key_states.shape
            return torch.empty((batch, heads, 0, dim), dtype=torch.float32, device=key_states.device)
        return torch.stack(parts, dim=2)

    def summaries(
        self,
        layer_idx: int,
        level: str,
        key_states: torch.Tensor,
        history_count: int,
    ) -> torch.Tensor:
        cache_key = (layer_idx, level, key_states.device)
        cached = self._summaries.get(cache_key)
        if cached is not None:
            return cached
        if level == "block":
            ranges = self.blocks
        elif level == "mid":
            ranges = self.mids
        elif level == "leaf":
            ranges = self.leafs
        else:
            raise ValueError(level)
        out = self._range_mean(key_states, ranges, history_count)
        self._summaries[cache_key] = out
        return out

    def _eligible(self, units: list[RangeUnit], remote_end: int) -> list[int]:
        return [
            unit.unit_id
            for unit in units
            if unit.end > self.config.sink_tokens and unit.start < remote_end
        ]

    @staticmethod
    def _topk_ids(scores: torch.Tensor, ids: list[int], k: int) -> list[int]:
        if k <= 0 or not ids:
            return []
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=scores.device)
        selected_scores = scores.index_select(0, ids_tensor)
        count = min(k, ids_tensor.numel())
        pos = torch.topk(selected_scores, k=count, largest=True).indices
        return [int(ids_tensor[item].item()) for item in pos]

    def selected_ranges_for_head(
        self,
        layer_idx: int,
        head_idx: int,
        query_token: int,
        block_scores: torch.Tensor,
        mid_scores: torch.Tensor,
        leaf_scores: torch.Tensor,
        remote_end: int,
    ) -> tuple[list[int], list[int], list[int], int]:
        refresh = max(1, self.config.route_refresh_tokens)
        cache_key = (layer_idx, head_idx, query_token // refresh)
        cached = self._last_selected.get(cache_key)
        if cached is not None:
            blocks, mids, leafs = cached
            return blocks, mids, leafs, 0
        self.route_compute_calls += 1

        eligible_blocks = self._eligible(self.blocks, remote_end)
        blocks = self._topk_ids(block_scores, eligible_blocks, self.config.top_blocks)

        mids: list[int] = []
        summary_scores = len(eligible_blocks)
        eligible_mids = set(self._eligible(self.mids, remote_end))
        for block_id in blocks:
            child_mids = [mid_id for mid_id in self.block_to_mids.get(block_id, []) if mid_id in eligible_mids]
            summary_scores += len(child_mids)
            mids.extend(self._topk_ids(mid_scores, child_mids, self.config.mids_per_block))

        leafs: list[int] = []
        eligible_leafs = set(self._eligible(self.leafs, remote_end))
        for mid_id in mids:
            child_leafs = [leaf_id for leaf_id in self.mid_to_leafs.get(mid_id, []) if leaf_id in eligible_leafs]
            summary_scores += len(child_leafs)
            leafs.extend(self._topk_ids(leaf_scores, child_leafs, self.config.leafs_per_mid))

        if self.config.seed_leafs > 0:
            all_leafs = sorted(eligible_leafs)
            summary_scores += len(all_leafs)
            leafs.extend(self._topk_ids(leaf_scores, all_leafs, self.config.seed_leafs))

        blocks = sorted(set(blocks), key=lambda idx: self.blocks[idx].start)
        mids = sorted(set(mids), key=lambda idx: self.mids[idx].start)
        leafs = sorted(set(leafs), key=lambda idx: self.leafs[idx].start)
        self._last_selected[cache_key] = (blocks, mids, leafs)
        return blocks, mids, leafs, summary_scores

    def record_selection(
        self,
        selected_blocks: list[int],
        selected_mids: list[int],
        selected_leafs: list[int],
        kept_tokens: int,
        summary_scores: int,
    ) -> None:
        self.selected_block_counts.append(len(selected_blocks))
        self.selected_mid_counts.append(len(selected_mids))
        self.selected_leaf_counts.append(len(selected_leafs))
        self.selected_leaf_tokens.append(sum(self.leafs[leaf_id].length for leaf_id in selected_leafs))
        self.kept_attention_tokens.append(kept_tokens)
        if summary_scores > 0:
            self.summary_score_counts.append(summary_scores)
        self.route_apply_calls += 1

    def stats(self) -> dict[str, float]:
        def avg(values: list[int]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        return {
            "block_count": float(len(self.blocks)),
            "mid_count": float(len(self.mids)),
            "leaf_count": float(len(self.leafs)),
            "avg_selected_blocks": avg(self.selected_block_counts),
            "avg_selected_mids": avg(self.selected_mid_counts),
            "avg_selected_leafs": avg(self.selected_leaf_counts),
            "avg_selected_leaf_tokens": avg(self.selected_leaf_tokens),
            "avg_kept_attention_tokens": avg(self.kept_attention_tokens),
            "avg_summary_scores_when_computed": avg(self.summary_score_counts),
            "route_compute_calls": float(self.route_compute_calls),
            "route_apply_calls": float(self.route_apply_calls),
        }


def build_fixed_hierarchy(
    prefill_tokens: int,
    block_tokens: int,
    mid_tokens: int,
    leaf_tokens: int,
) -> tuple[list[RangeUnit], list[RangeUnit], list[RangeUnit]]:
    blocks: list[RangeUnit] = []
    mids: list[RangeUnit] = []
    leafs: list[RangeUnit] = []
    block_tokens = max(1, block_tokens)
    mid_tokens = max(1, mid_tokens)
    leaf_tokens = max(1, leaf_tokens)

    for block_start in range(0, prefill_tokens, block_tokens):
        block_end = min(prefill_tokens, block_start + block_tokens)
        block_id = len(blocks)
        blocks.append(RangeUnit(block_id, "block10k", block_start, block_end))
        for mid_start in range(block_start, block_end, mid_tokens):
            mid_end = min(block_end, mid_start + mid_tokens)
            mid_id = len(mids)
            mids.append(RangeUnit(mid_id, "mid1k", mid_start, mid_end, block_id))
            for leaf_start in range(mid_start, mid_end, leaf_tokens):
                leaf_end = min(mid_end, leaf_start + leaf_tokens)
                leafs.append(RangeUnit(len(leafs), "leaf100", leaf_start, leaf_end, mid_id))
    return blocks, mids, leafs


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected bool, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed 10k/1k/100-token hierarchical KV-summary PPL evaluator."
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--text_path", default="data/war_and_peace_pg2600.txt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefill_tokens", type=int, default=20_000)
    parser.add_argument("--eval_tokens", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--eval_chunk_size", type=int, default=1)
    parser.add_argument("--modes", default="baseline,recent,hierkv")
    parser.add_argument("--sink_tokens", type=int, default=64)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--block_tokens", type=int, default=10_000)
    parser.add_argument("--mid_tokens", type=int, default=1_000)
    parser.add_argument("--leaf_tokens", type=int, default=100)
    parser.add_argument("--top_blocks", type=int, default=1)
    parser.add_argument("--mids_per_block", type=int, default=2)
    parser.add_argument("--leafs_per_mid", type=int, default=2)
    parser.add_argument("--seed_leafs", type=int, default=0)
    parser.add_argument("--route_refresh_tokens", type=int, default=16)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--reuse_prefill_cache", type=str2bool, default=True)
    parser.add_argument("--log_every", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026070301)
    return parser.parse_args()


def _padded_indices_from_lists(index_lists: list[list[list[int]]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    batch_count = len(index_lists)
    head_count = len(index_lists[0]) if batch_count else 0
    max_len = max((len(values) for batch in index_lists for values in batch), default=1)
    final_indices = torch.zeros((batch_count, head_count, max_len), dtype=torch.long, device=device)
    final_valid = torch.zeros((batch_count, head_count, max_len), dtype=torch.bool, device=device)
    for batch_idx, batch in enumerate(index_lists):
        for head_idx, values in enumerate(batch):
            if not values:
                continue
            tensor = torch.tensor(values, dtype=torch.long, device=device)
            final_indices[batch_idx, head_idx, : tensor.numel()] = tensor
            final_valid[batch_idx, head_idx, : tensor.numel()] = True
    return final_indices, final_valid


def _unique_sorted_indices(parts: list[range], self_index: int) -> list[int]:
    seen: set[int] = set()
    values: list[int] = []
    for part in parts:
        for idx in part:
            if idx not in seen:
                seen.add(idx)
                values.append(idx)
    if self_index not in seen:
        values.append(self_index)
    values.sort()
    return values


def _selected_attention_from_indices(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    final_indices: torch.Tensor,
    final_valid: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    head_dim = query_states.shape[-1]
    head_count = query_states.shape[1]
    gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states, dim=2, index=gather)
    selected_values = torch.gather(value_states, dim=2, index=gather)
    selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        key_count = key_states.shape[-2]
        mask_row = attention_mask[:, :, 0, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
    selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
    weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    output = torch.sum(weights[:, :, :, None] * selected_values, dim=2)
    return output[:, None, :, :].contiguous(), None


def _recent_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    if query_states.shape[-2] != 1:
        raise RuntimeError("recent mode requires --eval_chunk_size 1.")
    batch_count, head_count, _, _ = query_states.shape
    history_count = key_states.shape[-2] - 1
    parts = []
    if _ACTIVE_SINK_TOKENS > 0:
        parts.append(range(0, min(_ACTIVE_SINK_TOKENS, history_count)))
    if _ACTIVE_RECENT_TOKENS > 0:
        parts.append(range(max(0, history_count - _ACTIVE_RECENT_TOKENS), history_count))
    values = _unique_sorted_indices(parts, history_count)
    final_indices = torch.tensor(values, dtype=torch.long, device=query_states.device).view(1, 1, -1)
    final_indices = final_indices.expand(batch_count, head_count, -1).contiguous()
    final_valid = torch.ones_like(final_indices, dtype=torch.bool)
    return _selected_attention_from_indices(
        query_states,
        key_states,
        value_states,
        attention_mask,
        final_indices,
        final_valid,
        scaling,
    )


def _hierkv_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    if _ACTIVE_HIER_STATE is None:
        raise RuntimeError("hierkv mode requires active HierKVState.")
    if query_states.shape[-2] != 1:
        raise RuntimeError("hierkv mode requires --eval_chunk_size 1.")
    batch_count, head_count, _, _ = query_states.shape
    history_count = key_states.shape[-2] - 1
    query_token = history_count
    cfg = _ACTIVE_HIER_STATE.config
    remote_end = max(cfg.sink_tokens, history_count - cfg.recent_tokens)
    layer_idx = int(getattr(module, "layer_idx", 0))

    block_summaries = _ACTIVE_HIER_STATE.summaries(layer_idx, "block", key_states, history_count)
    mid_summaries = _ACTIVE_HIER_STATE.summaries(layer_idx, "mid", key_states, history_count)
    leaf_summaries = _ACTIVE_HIER_STATE.summaries(layer_idx, "leaf", key_states, history_count)
    q = query_states[:, :, 0, :].float()
    block_scores = torch.einsum("bhd,bhpd->bhp", q, block_summaries)
    mid_scores = torch.einsum("bhd,bhpd->bhp", q, mid_summaries)
    leaf_scores = torch.einsum("bhd,bhpd->bhp", q, leaf_summaries)

    index_lists: list[list[list[int]]] = [[[] for _ in range(head_count)] for _ in range(batch_count)]
    for batch_idx in range(batch_count):
        for head_idx in range(head_count):
            selected_blocks, selected_mids, selected_leafs, summary_scores = _ACTIVE_HIER_STATE.selected_ranges_for_head(
                layer_idx,
                head_idx,
                query_token,
                block_scores[batch_idx, head_idx],
                mid_scores[batch_idx, head_idx],
                leaf_scores[batch_idx, head_idx],
                remote_end,
            )
            parts: list[range] = []
            if cfg.sink_tokens > 0:
                parts.append(range(0, min(cfg.sink_tokens, history_count)))
            for leaf_id in selected_leafs:
                leaf = _ACTIVE_HIER_STATE.leafs[leaf_id]
                start = max(0, min(leaf.start, remote_end))
                end = max(start, min(leaf.end, remote_end))
                if start < end:
                    parts.append(range(start, end))
            if cfg.recent_tokens > 0:
                parts.append(range(max(0, history_count - cfg.recent_tokens), history_count))
            values = _unique_sorted_indices(parts, history_count)
            index_lists[batch_idx][head_idx] = values
            _ACTIVE_HIER_STATE.record_selection(
                selected_blocks,
                selected_mids,
                selected_leafs,
                len(values),
                summary_scores,
            )
    final_indices, final_valid = _padded_indices_from_lists(index_lists, query_states.device)
    return _selected_attention_from_indices(
        query_states,
        key_states,
        value_states,
        attention_mask,
        final_indices,
        final_valid,
        scaling,
    )


def _patched_attention_forward(
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
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states = key_states.repeat_interleave(repeat_groups, dim=1)
        value_states = value_states.repeat_interleave(repeat_groups, dim=1)
    if _ACTIVE_MODE == "hierkv":
        return _hierkv_attention_forward(module, query_states, key_states, value_states, attention_mask, scaling)
    if _ACTIVE_MODE == "recent":
        return _recent_attention_forward(query_states, key_states, value_states, attention_mask, scaling)

    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        weights = F.dropout(weights, p=dropout, training=True)
    output = torch.matmul(weights, value_states)
    return output.transpose(1, 2).contiguous(), weights


def install_attention_patch() -> None:
    global _ORIGINAL_EAGER_ATTENTION_FORWARD
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
        setattr(modeling_qwen3, "eager_attention_forward", _patched_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _patched_attention_forward


class active_mode:
    def __init__(self, mode: str, hier_state: HierKVState | None, sink_tokens: int, recent_tokens: int) -> None:
        self.mode = mode
        self.hier_state = hier_state
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.previous: tuple[str, HierKVState | None, int, int] | None = None

    def __enter__(self) -> None:
        global _ACTIVE_MODE, _ACTIVE_HIER_STATE, _ACTIVE_SINK_TOKENS, _ACTIVE_RECENT_TOKENS
        self.previous = (_ACTIVE_MODE, _ACTIVE_HIER_STATE, _ACTIVE_SINK_TOKENS, _ACTIVE_RECENT_TOKENS)
        _ACTIVE_MODE = self.mode
        _ACTIVE_HIER_STATE = self.hier_state
        _ACTIVE_SINK_TOKENS = self.sink_tokens
        _ACTIVE_RECENT_TOKENS = self.recent_tokens

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        global _ACTIVE_MODE, _ACTIVE_HIER_STATE, _ACTIVE_SINK_TOKENS, _ACTIVE_RECENT_TOKENS
        if self.previous is not None:
            _ACTIVE_MODE, _ACTIVE_HIER_STATE, _ACTIVE_SINK_TOKENS, _ACTIVE_RECENT_TOKENS = self.previous


@torch.inference_mode()
def compute_eval_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    eval_chunk_size: int,
    input_device: torch.device,
    mode: str,
    hier_state: HierKVState | None,
    sink_tokens: int,
    recent_tokens: int,
    initial_past_key_values: Any,
    initial_prev_logits: torch.Tensor,
    clone_initial_cache: bool,
    log_every: int,
) -> tuple[float, float, int, float]:
    if clone_initial_cache:
        print(f"cloning shared prefill cache for mode: {mode}", flush=True)
        past_key_values = clone_past_key_values(initial_past_key_values)
        prev_logits = initial_prev_logits.detach().clone()
    else:
        past_key_values = initial_past_key_values
        prev_logits = initial_prev_logits
    total_loss = 0.0
    total_count = 0
    eval_end = prefill_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / eval_chunk_size)
    started = time.perf_counter()
    with active_mode(mode, hier_state, sink_tokens, recent_tokens):
        for chunk_idx, start in enumerate(range(prefill_tokens, eval_end, eval_chunk_size), start=1):
            end = min(start + eval_chunk_size, eval_end)
            chunk = input_ids[:, start:end].to(input_device)
            outputs = model_forward(
                model,
                {
                    "input_ids": chunk,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "return_dict": True,
                    "output_attentions": False,
                    "cache_position": torch.arange(start, end, device=input_device),
                },
            )
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
            if log_every > 0 and (chunk_idx % log_every == 0 or chunk_idx == total_chunks):
                ppl_so_far = math.exp(total_loss / max(1, total_count))
                print(
                    f"{mode} chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1} "
                    f"ppl_so_far={ppl_so_far:.4f}",
                    flush=True,
                )
            del outputs, chunk, logits, shifted_logits, loss
            if input_device.type == "cuda":
                torch.cuda.empty_cache()
    seconds = time.perf_counter() - started
    loss_value = total_loss / max(1, total_count)
    return loss_value, math.exp(loss_value), total_count, seconds


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.eval_chunk_size != 1 and any(mode in args.modes for mode in ["recent", "hierkv"]):
        raise RuntimeError("recent/hierkv require --eval_chunk_size 1.")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))
    total_needed = args.prefill_tokens + args.eval_tokens
    if args.require_total_tokens and len(token_ids) < total_needed:
        raise RuntimeError(f"Need {total_needed} tokens, got {len(token_ids)}")
    token_ids = token_ids[:total_needed]
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    print(f"loading model: {args.model_name_or_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(requested_device)
    model.eval()
    model.config.use_cache = True
    install_attention_patch()
    input_device = pick_input_device(model, requested_device)

    print("starting shared full prefill cache", flush=True)
    with active_mode("baseline", None, args.sink_tokens, args.recent_tokens):
        prefill_start = time.perf_counter()
        shared_past, shared_prev_logits = prefill_cache(
            model,
            input_ids,
            args.prefill_tokens,
            args.chunk_size,
            input_device,
        )
        shared_prefill_seconds = time.perf_counter() - prefill_start
    print(f"shared prefill cache ready: {shared_prefill_seconds:.3f}s", flush=True)

    rows: list[dict[str, Any]] = []
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for mode in modes:
        if mode not in {"baseline", "recent", "hierkv"}:
            raise ValueError(f"Unknown mode: {mode}")
        hier_state = None
        if mode == "hierkv":
            hier_state = HierKVState(
                HierKVConfig(
                    prefill_tokens=args.prefill_tokens,
                    sink_tokens=args.sink_tokens,
                    recent_tokens=args.recent_tokens,
                    block_tokens=args.block_tokens,
                    mid_tokens=args.mid_tokens,
                    leaf_tokens=args.leaf_tokens,
                    top_blocks=args.top_blocks,
                    mids_per_block=args.mids_per_block,
                    leafs_per_mid=args.leafs_per_mid,
                    seed_leafs=args.seed_leafs,
                    route_refresh_tokens=args.route_refresh_tokens,
                )
            )
        loss, ppl, token_count, seconds = compute_eval_loss(
            model,
            input_ids,
            args.prefill_tokens,
            args.eval_tokens,
            args.eval_chunk_size,
            input_device,
            mode,
            hier_state,
            args.sink_tokens,
            args.recent_tokens,
            shared_past,
            shared_prev_logits,
            clone_initial_cache=len(modes) > 1,
            log_every=args.log_every,
        )
        stats = hier_state.stats() if hier_state is not None else {}
        rows.append(
            {
                "mode": mode,
                "loss": loss,
                "ppl": ppl,
                "token_count": token_count,
                "seconds": seconds,
                "shared_prefill_seconds": shared_prefill_seconds,
                "sink_tokens": args.sink_tokens,
                "recent_tokens": args.recent_tokens,
                "block_tokens": args.block_tokens if mode == "hierkv" else "",
                "mid_tokens": args.mid_tokens if mode == "hierkv" else "",
                "leaf_tokens": args.leaf_tokens if mode == "hierkv" else "",
                "top_blocks": args.top_blocks if mode == "hierkv" else "",
                "mids_per_block": args.mids_per_block if mode == "hierkv" else "",
                "leafs_per_mid": args.leafs_per_mid if mode == "hierkv" else "",
                "seed_leafs": args.seed_leafs if mode == "hierkv" else "",
                "route_refresh_tokens": args.route_refresh_tokens if mode == "hierkv" else "",
                **stats,
            }
        )
    write_csv(output_dir / "ppl_by_mode.csv", rows)
    summary = {
        "args": vars(args),
        "resolved": {
            "total_tokens_used": len(token_ids),
            "method": (
                "Fixed hierarchical KV summaries: 10k-token block K means, 1k-token mid K means, "
                "100-token leaf K means. Decode attention keeps sink + recent + selected raw leaf KV ranges."
            ),
        },
        "paths": {"ppl_by_mode": str(output_dir / "ppl_by_mode.csv")},
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
