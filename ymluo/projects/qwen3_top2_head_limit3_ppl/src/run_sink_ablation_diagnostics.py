from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
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
    resolve_dtype,
)
from run_qabs_downstream_kv_retrieval import LABELS, score_option, write_csv  # noqa: E402
from run_qabs_downstream_task_suite import BUILDERS  # noqa: E402
from run_qabs_evidence_span_coverage import evidence_spans  # noqa: E402


_ORIGINAL_QWEN3_ATTENTION_FORWARD: Any | None = None
_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_COLLECTOR: "SinkAblationCollector | None" = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sink content/position/KV ablation diagnostics.")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", default="compact_kv,json_kv,needle_sentence,topic_table")
    parser.add_argument("--tasks_per_variant", type=int, default=4)
    parser.add_argument("--records_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026063011)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--layers", default="0,4,8,13,20,27")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--sink_tokens", type=int, default=16)
    parser.add_argument("--keep_prefix_tokens", type=int, default=2)
    parser.add_argument("--recent_tokens", type=int, default=16)
    parser.add_argument("--replacement_text", default=" X")
    parser.add_argument(
        "--conditions",
        default=(
            "baseline,replace_sink_content,move_sink_middle,move_sink_end,"
            "keep_prefix2_text,zero_sink_kv,drop_sink_kv,keep_prefix2_drop_sink_kv"
        ),
    )
    parser.add_argument("--collect_attention", type=str2bool, default=True)
    parser.add_argument("--max_eval_query_tokens", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1)
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
    if not selected:
        raise ValueError(f"No {name} selected from spec {spec!r}")
    return sorted(selected)


@dataclass
class GroupPositions:
    sink_content: set[int]
    front_positions: set[int]
    sink_keep_prefix: set[int]
    sink_rest: set[int]
    recent: set[int]
    evidence_key: set[int]
    evidence_label: set[int]
    evidence_record: set[int]
    evidence_any: set[int]

    def as_dict(self) -> dict[str, set[int]]:
        return {
            "sink_content": self.sink_content,
            "front_positions": self.front_positions,
            "sink_keep_prefix": self.sink_keep_prefix,
            "sink_rest": self.sink_rest,
            "recent": self.recent,
            "evidence_key": self.evidence_key,
            "evidence_label": self.evidence_label,
            "evidence_record": self.evidence_record,
            "evidence_any": self.evidence_any,
        }


@dataclass
class ConditionSpec:
    name: str
    text_ablation: str
    kv_ablation: str


@dataclass
class MeanAcc:
    count: int = 0
    sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, values: dict[str, float]) -> None:
        self.count += 1
        for key, value in values.items():
            if math.isfinite(value):
                self.sums[key] += float(value)

    def mean(self, key: str) -> float:
        return self.sums.get(key, 0.0) / self.count if self.count else 0.0

    def row(self, extra: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        return {**extra, "rows": self.count, **{field_name: self.mean(field_name) for field_name in fields}}


class SinkAblationCollector:
    def __init__(
        self,
        *,
        selected_layers: list[int],
        selected_heads: list[int],
        top_fraction: float,
        collect_attention: bool,
    ) -> None:
        self.selected_layers = set(selected_layers)
        self.selected_heads = set(selected_heads)
        self.top_fraction = top_fraction
        self.collect_attention = collect_attention
        self.phase = "idle"
        self.condition = ""
        self.kv_ablation = "none"
        self.prefill_tokens = 0
        self.group_positions = GroupPositions(set(), set(), set(), set(), set(), set(), set(), set(), set())
        self.pre_k_history: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.current_pre_q: dict[int, torch.Tensor] = {}
        self.acc: dict[tuple[str, str, int | str, int | str], MeanAcc] = defaultdict(MeanAcc)

    def begin_condition(
        self,
        *,
        condition: str,
        kv_ablation: str,
        prefill_tokens: int,
        group_positions: GroupPositions,
    ) -> None:
        self.phase = "prefill"
        self.condition = condition
        self.kv_ablation = kv_ablation
        self.prefill_tokens = prefill_tokens
        self.group_positions = group_positions
        self.pre_k_history.clear()
        self.current_pre_q.clear()

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def store_pre_states(self, layer: int, q_pre: torch.Tensor, k_pre: torch.Tensor) -> None:
        if layer not in self.selected_layers:
            return
        if k_pre.shape[1] != q_pre.shape[1]:
            repeat_groups = q_pre.shape[1] // k_pre.shape[1]
            k_pre = k_pre.repeat_interleave(repeat_groups, dim=1)
        self.pre_k_history[layer].append(k_pre.detach())
        self.current_pre_q[layer] = q_pre.detach()

    def ablation_positions(self, key_count: int) -> list[int]:
        if self.phase not in {"query", "label"}:
            return []
        if self.kv_ablation == "zero_sink_kv" or self.kv_ablation == "drop_sink_kv":
            positions = self.group_positions.sink_content
        elif self.kv_ablation == "keep_prefix2_drop_sink_kv":
            positions = self.group_positions.sink_rest
        else:
            positions = set()
        return sorted(pos for pos in positions if 0 <= pos < key_count)

    def observe_attention(
        self,
        *,
        layer: int,
        query_states_post: torch.Tensor,
        key_states_post: torch.Tensor,
        raw_scores_post: torch.Tensor,
        masked_scores: torch.Tensor,
        query_index: int,
    ) -> None:
        if not self.collect_attention or self.phase != "query" or layer not in self.selected_layers:
            return
        finite = torch.isfinite(masked_scores[:, :, query_index, :])
        valid_count = int(finite[0, 0].sum().item())
        if valid_count <= 2:
            return
        history_count = valid_count - 1
        top_count = min(history_count, max(1, math.ceil(self.top_fraction * history_count)))
        scores = masked_scores[0, :, query_index, :history_count].detach().float()
        raw_post = raw_scores_post[0, :, query_index, :history_count].detach().float()
        attn = F.softmax(masked_scores[0, :, query_index, :valid_count].detach().float(), dim=-1)[:, :history_count]
        top_indices = torch.topk(scores, k=top_count, dim=-1).indices

        pre_q = self.current_pre_q.get(layer)
        pre_k = torch.cat(self.pre_k_history[layer], dim=2) if self.pre_k_history.get(layer) else None
        if pre_q is not None and pre_k is not None and pre_k.shape[2] >= history_count:
            pre_scores = torch.matmul(
                pre_q[:, :, query_index : query_index + 1, :].float(),
                pre_k[:, :, :history_count, :].float().transpose(2, 3),
            )
            scaling = 1.0 / math.sqrt(pre_q.shape[-1])
            pre_scores = (pre_scores * scaling)[0, :, 0, :].detach().float()
            pre_k_norm = torch.linalg.vector_norm(pre_k[0, :, :history_count, :].float(), dim=-1)
        else:
            pre_scores = None
            pre_k_norm = None
        post_k_norm = torch.linalg.vector_norm(key_states_post[0, :, :history_count, :].detach().float(), dim=-1)

        groups = self.group_positions.as_dict()
        for head in range(scores.shape[0]):
            if head not in self.selected_heads:
                continue
            top_set = set(int(x) for x in top_indices[head].detach().cpu().tolist())
            top_mass = float(attn[head, list(top_set)].sum().item()) if top_set else 0.0
            base_values = {"top2_mass": top_mass}
            for group_name, positions in groups.items():
                valid_positions = sorted(pos for pos in positions if 0 <= pos < history_count)
                if not valid_positions:
                    values = {
                        **base_values,
                        "group_token_fraction": 0.0,
                        "group_attention_mass": 0.0,
                        "group_top2_recall": 0.0,
                        "group_top2_mass": 0.0,
                        "post_k_norm": 0.0,
                        "post_logit_mean": 0.0,
                        "post_logit_max": 0.0,
                        "pre_k_norm": 0.0,
                        "pre_logit_mean": 0.0,
                        "pre_logit_max": 0.0,
                    }
                else:
                    idx = torch.tensor(valid_positions, device=attn.device, dtype=torch.long)
                    overlap = sorted(top_set.intersection(valid_positions))
                    overlap_idx = torch.tensor(overlap, device=attn.device, dtype=torch.long) if overlap else None
                    values = {
                        **base_values,
                        "group_token_fraction": len(valid_positions) / history_count,
                        "group_attention_mass": float(attn[head, idx].sum().item()),
                        "group_top2_recall": len(overlap) / max(1, top_count),
                        "group_top2_mass": float(attn[head, overlap_idx].sum().item()) if overlap_idx is not None else 0.0,
                        "post_k_norm": float(post_k_norm[head, idx].mean().item()),
                        "post_logit_mean": float(raw_post[head, idx].mean().item()),
                        "post_logit_max": float(raw_post[head, idx].max().item()),
                        "pre_k_norm": 0.0,
                        "pre_logit_mean": 0.0,
                        "pre_logit_max": 0.0,
                    }
                    if pre_scores is not None and pre_k_norm is not None:
                        values["pre_k_norm"] = float(pre_k_norm[head, idx].mean().item())
                        values["pre_logit_mean"] = float(pre_scores[head, idx].mean().item())
                        values["pre_logit_max"] = float(pre_scores[head, idx].max().item())
                for key in [
                    ("overall", group_name, "all", "all"),
                    ("layer", group_name, layer, "all"),
                    ("layer_head", group_name, layer, head),
                ]:
                    self.acc[key].add(values)

    def rows(self) -> list[dict[str, Any]]:
        fields = [
            "top2_mass",
            "group_token_fraction",
            "group_attention_mass",
            "group_top2_recall",
            "group_top2_mass",
            "post_k_norm",
            "post_logit_mean",
            "post_logit_max",
            "pre_k_norm",
            "pre_logit_mean",
            "pre_logit_max",
        ]
        rows = []
        for (scope, group, layer, head), acc in sorted(self.acc.items(), key=lambda item: str(item[0])):
            rows.append(
                acc.row(
                    {
                        "condition": self.condition,
                        "scope": scope,
                        "group": group,
                        "layer": "" if layer == "all" else layer,
                        "head": "" if head == "all" else head,
                    },
                    fields,
                )
            )
        return rows


def condition_specs(names: list[str]) -> list[ConditionSpec]:
    mapping = {
        "baseline": ConditionSpec("baseline", "baseline", "none"),
        "replace_sink_content": ConditionSpec("replace_sink_content", "replace_sink_content", "none"),
        "move_sink_middle": ConditionSpec("move_sink_middle", "move_sink_middle", "none"),
        "move_sink_end": ConditionSpec("move_sink_end", "move_sink_end", "none"),
        "keep_prefix2_text": ConditionSpec("keep_prefix2_text", "keep_prefix_text", "none"),
        "zero_sink_kv": ConditionSpec("zero_sink_kv", "baseline", "zero_sink_kv"),
        "drop_sink_kv": ConditionSpec("drop_sink_kv", "baseline", "drop_sink_kv"),
        "keep_prefix2_drop_sink_kv": ConditionSpec(
            "keep_prefix2_drop_sink_kv", "baseline", "keep_prefix2_drop_sink_kv"
        ),
    }
    unknown = [name for name in names if name not in mapping]
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}; available={sorted(mapping)}")
    return [mapping[name] for name in names]


def token_set_from_span(span: tuple[int, int] | None, mapper: dict[int, int | None]) -> set[int]:
    if span is None:
        return set()
    out = set()
    for old in range(span[0], span[1]):
        new = mapper.get(old)
        if new is not None:
            out.add(new)
    return out


def build_context_variant(
    *,
    context_ids: torch.Tensor,
    spans: dict[str, tuple[int, int]],
    spec: ConditionSpec,
    replacement_id: int,
    sink_tokens: int,
    keep_prefix_tokens: int,
    recent_tokens: int,
) -> tuple[torch.Tensor, GroupPositions]:
    ids = context_ids[0].tolist()
    n = len(ids)
    s = min(sink_tokens, n)
    keep = min(keep_prefix_tokens, s)
    mapper: dict[int, int | None] = {}

    if spec.text_ablation == "replace_sink_content":
        new_ids = ids[:]
        for i in range(s):
            new_ids[i] = replacement_id
        mapper = {i: i for i in range(n)}
        sink_positions = set(range(s))
    elif spec.text_ablation == "move_sink_middle":
        sink = ids[:s]
        rest = ids[s:]
        insert = len(rest) // 2
        new_ids = rest[:insert] + sink + rest[insert:]
        for i in range(n):
            if i < s:
                mapper[i] = insert + i
            else:
                j = i - s
                mapper[i] = j if j < insert else j + s
        sink_positions = set(range(insert, insert + s))
    elif spec.text_ablation == "move_sink_end":
        sink = ids[:s]
        rest = ids[s:]
        new_ids = rest + sink
        for i in range(n):
            mapper[i] = len(rest) + i if i < s else i - s
        sink_positions = set(range(len(rest), len(rest) + s))
    elif spec.text_ablation == "keep_prefix_text":
        new_ids = ids[:keep] + ids[s:]
        for i in range(n):
            if i < keep:
                mapper[i] = i
            elif i < s:
                mapper[i] = None
            else:
                mapper[i] = i - (s - keep)
        sink_positions = set(range(keep))
    else:
        new_ids = ids[:]
        mapper = {i: i for i in range(n)}
        sink_positions = set(range(s))

    new_n = len(new_ids)
    front_positions = set(range(min(s, new_n)))
    sink_keep_prefix = {pos for old in range(keep) if (pos := mapper.get(old)) is not None}
    sink_rest = {pos for old in range(keep, s) if (pos := mapper.get(old)) is not None}
    recent = set(range(max(0, new_n - recent_tokens), new_n))
    evidence_key = token_set_from_span(spans.get("key"), mapper)
    evidence_label = token_set_from_span(spans.get("label"), mapper)
    evidence_record = token_set_from_span(spans.get("record"), mapper)
    evidence_any = set().union(evidence_key, evidence_label, evidence_record)
    groups = GroupPositions(
        sink_content=sink_positions,
        front_positions=front_positions,
        sink_keep_prefix=sink_keep_prefix,
        sink_rest=sink_rest,
        recent=recent,
        evidence_key=evidence_key,
        evidence_label=evidence_label,
        evidence_record=evidence_record,
        evidence_any=evidence_any,
    )
    return torch.tensor([new_ids], dtype=context_ids.dtype), groups


def _sink_eager_attention_forward(
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

    positions: list[int] = []
    kv_ablation = "none"
    if _ACTIVE_COLLECTOR is not None:
        positions = _ACTIVE_COLLECTOR.ablation_positions(key_states.shape[-2])
        kv_ablation = _ACTIVE_COLLECTOR.kv_ablation
    if positions and kv_ablation == "zero_sink_kv":
        idx = torch.tensor(positions, device=key_states.device, dtype=torch.long)
        key_states = key_states.clone()
        value_states = value_states.clone()
        key_states.index_fill_(2, idx, 0.0)
        value_states.index_fill_(2, idx, 0.0)

    raw_scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    scores = raw_scores
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    if positions and kv_ablation in {"drop_sink_kv", "keep_prefix2_drop_sink_kv"}:
        idx = torch.tensor(positions, device=scores.device, dtype=torch.long)
        scores = scores.clone()
        scores.index_fill_(3, idx, torch.finfo(scores.dtype).min)

    if _ACTIVE_COLLECTOR is not None:
        layer = int(getattr(module, "layer_idx", 0))
        query_count = scores.shape[-2]
        for query_index in range(query_count):
            _ACTIVE_COLLECTOR.observe_attention(
                layer=layer,
                query_states_post=query_states,
                key_states_post=key_states,
                raw_scores_post=raw_scores,
                masked_scores=scores,
                query_index=query_index,
            )

    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=True)
    attention_output = torch.matmul(attention_weights, value_states)
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, attention_weights


def install_sink_attention_patch() -> None:
    global _ORIGINAL_QWEN3_ATTENTION_FORWARD, _ORIGINAL_EAGER_ATTENTION_FORWARD
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except Exception as exc:
        raise RuntimeError("Could not import transformers.models.qwen3.modeling_qwen3.") from exc
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
        setattr(modeling_qwen3, "eager_attention_forward", _sink_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _sink_eager_attention_forward
    if _ORIGINAL_QWEN3_ATTENTION_FORWARD is None:
        _ORIGINAL_QWEN3_ATTENTION_FORWARD = modeling_qwen3.Qwen3Attention.forward

        def patched_forward(
            self: torch.nn.Module,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_value: Any = None,
            cache_position: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, Any]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            if _ACTIVE_COLLECTOR is not None:
                _ACTIVE_COLLECTOR.store_pre_states(int(getattr(self, "layer_idx", 0)), query_states, key_states)
            cos, sin = position_embeddings
            query_states, key_states = modeling_qwen3.apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            attention_interface = modeling_qwen3.eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = modeling_qwen3.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        modeling_qwen3.Qwen3Attention.forward = patched_forward


@contextmanager
def active_collector(collector: SinkAblationCollector):
    global _ACTIVE_COLLECTOR
    previous = _ACTIVE_COLLECTOR
    _ACTIVE_COLLECTOR = collector
    try:
        yield
    finally:
        _ACTIVE_COLLECTOR = previous


@torch.inference_mode()
def prefill_context(
    model: torch.nn.Module,
    context_ids: torch.Tensor,
    chunk_size: int,
    input_device: torch.device,
) -> tuple[Any, torch.Tensor]:
    past_key_values = None
    last_logits = None
    total = context_ids.shape[-1]
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        kwargs: dict[str, Any] = {
            "input_ids": context_ids[:, start:end].to(input_device),
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :].detach()
        del outputs
    if last_logits is None:
        raise RuntimeError("empty context")
    return past_key_values, last_logits


@torch.inference_mode()
def run_query_loss(
    model: torch.nn.Module,
    query_ids: torch.Tensor,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    input_device: torch.device,
    max_eval_query_tokens: int,
) -> tuple[Any, torch.Tensor, float, int]:
    total_loss = 0.0
    total_tokens = 0
    eval_len = query_ids.shape[-1] if max_eval_query_tokens <= 0 else min(query_ids.shape[-1], max_eval_query_tokens)
    for pos in range(eval_len):
        token = query_ids[:, pos : pos + 1].to(input_device)
        total_loss += float(F.cross_entropy(prev_logits.float(), token.reshape(-1), reduction="sum").item())
        total_tokens += int(token.numel())
        kwargs = {
            "input_ids": token,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
        }
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :].detach()
        del outputs
    return past_key_values, prev_logits, total_loss, total_tokens


@torch.inference_mode()
def score_labels(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
) -> tuple[str, dict[str, float]]:
    scores = {}
    for label in LABELS:
        scores[label] = score_option(
            model,
            tokenizer,
            input_device,
            clone_past_key_values(past_key_values),
            prev_logits.detach().clone(),
            label,
        )
    return max(scores, key=scores.get), scores


def summarize_task_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(row["variant"], row["condition"]) for row in rows})
    for variant, condition in keys:
        subset = [row for row in rows if row["variant"] == variant and row["condition"] == condition]
        total_loss = sum(float(row["query_loss"]) for row in subset)
        total_tokens = sum(int(row["query_tokens_scored"]) for row in subset)
        correct = sum(int(row["correct"]) for row in subset)
        out.append(
            {
                "variant": variant,
                "condition": condition,
                "tasks": len(subset),
                "accuracy": correct / max(1, len(subset)),
                "query_loss": total_loss / max(1, total_tokens),
                "query_ppl": math.exp(total_loss / max(1, total_tokens)) if total_tokens else 0.0,
            }
        )
    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        total_loss = sum(float(row["query_loss"]) for row in subset)
        total_tokens = sum(int(row["query_tokens_scored"]) for row in subset)
        correct = sum(int(row["correct"]) for row in subset)
        out.append(
            {
                "variant": "overall",
                "condition": condition,
                "tasks": len(subset),
                "accuracy": correct / max(1, len(subset)),
                "query_loss": total_loss / max(1, total_tokens),
                "query_ppl": math.exp(total_loss / max(1, total_tokens)) if total_tokens else 0.0,
            }
        )
    return out


def summarize_attention_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "top2_mass",
        "group_token_fraction",
        "group_attention_mass",
        "group_top2_recall",
        "group_top2_mass",
        "post_k_norm",
        "post_logit_mean",
        "post_logit_max",
        "pre_k_norm",
        "pre_logit_mean",
        "pre_logit_max",
    ]
    acc: dict[tuple[str, str, str], MeanAcc] = defaultdict(MeanAcc)
    for row in rows:
        if row["scope"] != "overall":
            continue
        key = (row["condition"], row["group"], row["scope"])
        values = {field: float(row[field]) for field in fields}
        acc[key].add(values)
    return [accum.row({"condition": k[0], "group": k[1], "scope": k[2]}, fields) for k, accum in sorted(acc.items())]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in variants if name not in BUILDERS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available={sorted(BUILDERS)}")
    specs = condition_specs([name.strip() for name in args.conditions.split(",") if name.strip()])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    replacement_ids = tokenizer(args.replacement_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    replacement_id = int(replacement_ids[0, 0].item())
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    install_sink_attention_patch()
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, device)

    layer_count = int(model.config.num_hidden_layers)
    head_count = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
    selected_layers = parse_index_spec(args.layers, layer_count, "layers")
    selected_heads = parse_index_spec(args.heads, head_count, "heads")

    task_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for variant_index, variant in enumerate(variants):
        rng = random.Random(args.seed + 1009 * variant_index)
        tasks = [BUILDERS[variant](rng, idx, args.records_per_task) for idx in range(args.tasks_per_variant)]
        for task_number, task in enumerate(tasks, start=1):
            if task_number == 1 or task_number == len(tasks) or task_number % args.log_every == 0:
                print(f"{variant} task {task_number}/{len(tasks)}", flush=True)
            context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            query_ids = tokenizer(task["query"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            spans = evidence_spans(tokenizer, task)
            for spec in specs:
                context_variant, groups = build_context_variant(
                    context_ids=context_ids,
                    spans=spans,
                    spec=spec,
                    replacement_id=replacement_id,
                    sink_tokens=args.sink_tokens,
                    keep_prefix_tokens=args.keep_prefix_tokens,
                    recent_tokens=args.recent_tokens,
                )
                collector = SinkAblationCollector(
                    selected_layers=selected_layers,
                    selected_heads=selected_heads,
                    top_fraction=args.top_fraction,
                    collect_attention=args.collect_attention,
                )
                collector.begin_condition(
                    condition=spec.name,
                    kv_ablation=spec.kv_ablation,
                    prefill_tokens=int(context_variant.shape[-1]),
                    group_positions=groups,
                )
                with active_collector(collector):
                    context_cache, context_prev = prefill_context(
                        model,
                        context_variant,
                        args.chunk_size,
                        input_device,
                    )
                    collector.set_phase("query")
                    query_cache, query_prev, query_loss, query_tokens_scored = run_query_loss(
                        model,
                        query_ids,
                        clone_past_key_values(context_cache),
                        context_prev.detach().clone(),
                        input_device,
                        args.max_eval_query_tokens,
                    )
                    collector.set_phase("label")
                    pred, scores = score_labels(
                        model,
                        tokenizer,
                        input_device,
                        clone_past_key_values(query_cache),
                        query_prev.detach().clone(),
                    )
                task_rows.append(
                    {
                        "variant": variant,
                        "task_id": task["task_id"],
                        "condition": spec.name,
                        "text_ablation": spec.text_ablation,
                        "kv_ablation": spec.kv_ablation,
                        "context_tokens": int(context_variant.shape[-1]),
                        "query_tokens": int(query_ids.shape[-1]),
                        "query_tokens_scored": query_tokens_scored,
                        "query_loss": query_loss,
                        "query_ppl": math.exp(query_loss / max(1, query_tokens_scored)),
                        "target_key": task["target_key"],
                        "target_label": task["target_label"],
                        "pred_label": pred,
                        "correct": int(pred == task["target_label"]),
                        **{f"score_{label}": scores[label] for label in LABELS},
                    }
                )
                attention_rows.extend(collector.rows())
                del context_cache, query_cache
                if input_device.type == "cuda":
                    torch.cuda.empty_cache()

    task_fields = [
        "variant",
        "task_id",
        "condition",
        "text_ablation",
        "kv_ablation",
        "context_tokens",
        "query_tokens",
        "query_tokens_scored",
        "query_loss",
        "query_ppl",
        "target_key",
        "target_label",
        "pred_label",
        "correct",
        *[f"score_{label}" for label in LABELS],
    ]
    attn_fields = [
        "condition",
        "scope",
        "group",
        "layer",
        "head",
        "rows",
        "top2_mass",
        "group_token_fraction",
        "group_attention_mass",
        "group_top2_recall",
        "group_top2_mass",
        "post_k_norm",
        "post_logit_mean",
        "post_logit_max",
        "pre_k_norm",
        "pre_logit_mean",
        "pre_logit_max",
    ]
    write_csv(output_dir / "sink_ablation_task_results.csv", task_rows, task_fields)
    write_csv(output_dir / "sink_ablation_summary.csv", summarize_task_rows(task_rows), [
        "variant",
        "condition",
        "tasks",
        "accuracy",
        "query_loss",
        "query_ppl",
    ])
    write_csv(output_dir / "sink_ablation_attention_stats.csv", attention_rows, attn_fields)
    write_csv(output_dir / "sink_ablation_attention_overall.csv", summarize_attention_rows(attention_rows), [
        "condition",
        "group",
        "scope",
        "rows",
        "top2_mass",
        "group_token_fraction",
        "group_attention_mass",
        "group_top2_recall",
        "group_top2_mass",
        "post_k_norm",
        "post_logit_mean",
        "post_logit_max",
        "pre_k_norm",
        "pre_logit_mean",
        "pre_logit_max",
    ])

    summary = {
        "seconds": time.perf_counter() - started,
        "resolved": {
            "layer_count": layer_count,
            "head_count": head_count,
            "head_dim": head_dim,
            "selected_layers": selected_layers,
            "selected_heads": selected_heads,
            "conditions": [spec.name for spec in specs],
            "tasks": len(task_rows),
            "replacement_id": replacement_id,
            "replacement_text": args.replacement_text,
        },
        "outputs": {
            "task_results": str(output_dir / "sink_ablation_task_results.csv"),
            "summary": str(output_dir / "sink_ablation_summary.csv"),
            "attention_stats": str(output_dir / "sink_ablation_attention_stats.csv"),
            "attention_overall": str(output_dir / "sink_ablation_attention_overall.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
