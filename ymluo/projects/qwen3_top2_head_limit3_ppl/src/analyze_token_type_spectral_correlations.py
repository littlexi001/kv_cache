from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
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
    model_forward,
    pick_input_device,
    resolve_dtype,
)
from run_qabs_downstream_task_suite import BUILDERS  # noqa: E402
from run_qabs_evidence_span_coverage import evidence_spans  # noqa: E402


_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_COLLECTOR: "TokenTypeSpectralCollector | None" = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze token-type distributions and correlations in K-SVD spectral directions."
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", default="compact_kv,json_kv,needle_sentence,topic_table")
    parser.add_argument("--tasks_per_variant", type=int, default=4)
    parser.add_argument("--records_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026063007)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--layers", default="0,4,8,13,20,27")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--rank_cutoffs", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--direction_count", type=int, default=16)
    parser.add_argument("--sink_tokens", type=int, default=16)
    parser.add_argument("--recent_tokens", type=int, default=64)
    parser.add_argument("--max_query_tokens_per_task", type=int, default=2)
    parser.add_argument("--max_tokens_per_group_per_row", type=int, default=48)
    parser.add_argument("--include_other_sample", type=str2bool, default=True)
    parser.add_argument("--center_k", type=str2bool, default=True)
    parser.add_argument("--svd_device", default="cuda")
    parser.add_argument("--svd_dtype", choices=["float32", "float64"], default="float32")
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


def parse_rank_cutoffs(value: str, max_rank: int) -> list[int]:
    cutoffs = sorted({int(part) for part in value.split(",") if part.strip()})
    cutoffs = [rank for rank in cutoffs if 1 <= rank <= max_rank]
    if max_rank not in cutoffs:
        cutoffs.append(max_rank)
    return sorted(cutoffs)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def choose_query_tokens(prefill_tokens: int, query_tokens: int, max_query_tokens: int) -> list[int]:
    positions = list(range(prefill_tokens, prefill_tokens + query_tokens))
    if max_query_tokens <= 0 or len(positions) <= max_query_tokens:
        return positions
    return positions[-max_query_tokens:]


def classify_piece(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "space"
    if stripped.isdigit():
        return "number"
    if stripped.isalpha():
        if stripped.isupper():
            return "upper_alpha"
        if stripped.islower():
            return "lower_alpha"
        return "mixed_alpha"
    if any(ch.isdigit() for ch in stripped) and any(ch.isalpha() for ch in stripped):
        return "alnum"
    if all(not ch.isalnum() for ch in stripped):
        return "punct"
    return "mixed"


@dataclass
class WeightedAccumulator:
    rows: int = 0
    weight: float = 0.0
    sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, values: dict[str, float], weight: float = 1.0) -> None:
        self.rows += 1
        self.weight += weight
        for key, value in values.items():
            if math.isfinite(value):
                self.sums[key] += float(value) * weight

    def row(self, extra: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        row = {**extra, "rows": self.rows, "weight": self.weight}
        denom = self.weight if self.weight > 0.0 else 1.0
        for field_name in fields:
            row[field_name] = self.sums.get(field_name, 0.0) / denom
        return row


class TokenTypeSpectralCollector:
    def __init__(
        self,
        *,
        selected_layers: list[int],
        selected_heads: list[int],
        rank_cutoffs: list[int],
        direction_count: int,
        top_fraction: float,
        sink_tokens: int,
        recent_tokens: int,
        max_tokens_per_group_per_row: int,
        include_other_sample: bool,
        center_k: bool,
        svd_device: torch.device,
        svd_dtype: torch.dtype,
    ) -> None:
        self.selected_layers = set(selected_layers)
        self.selected_heads = set(selected_heads)
        self.rank_cutoffs = rank_cutoffs
        self.direction_count = direction_count
        self.top_fraction = top_fraction
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.max_tokens_per_group_per_row = max_tokens_per_group_per_row
        self.include_other_sample = include_other_sample
        self.center_k = center_k
        self.svd_device = svd_device
        self.svd_dtype = svd_dtype

        self.query_tokens: set[int] = set()
        self.base_groups: dict[str, set[int]] = {}
        self.token_ids: list[int] = []
        self.token_texts: list[str] = []
        self.task_meta: dict[str, Any] = {}

        self.observed_rows = 0
        self.skipped_svd_rows = 0
        self.type_acc: dict[tuple[str, str, int | str], WeightedAccumulator] = defaultdict(WeightedAccumulator)
        self.direction_acc: dict[tuple[str, int], WeightedAccumulator] = defaultdict(WeightedAccumulator)
        self.pair_acc: dict[tuple[str, str, int], WeightedAccumulator] = defaultdict(WeightedAccumulator)
        self.position_acc: dict[tuple[str, str, int | str], WeightedAccumulator] = defaultdict(WeightedAccumulator)
        self.lexical_counts: Counter[tuple[str, str]] = Counter()
        self.token_examples: dict[str, Counter[str]] = defaultdict(Counter)

    def set_task(
        self,
        *,
        task_meta: dict[str, Any],
        query_tokens: set[int],
        base_groups: dict[str, set[int]],
        token_ids: list[int],
        token_texts: list[str],
    ) -> None:
        self.task_meta = task_meta
        self.query_tokens = query_tokens
        self.base_groups = base_groups
        self.token_ids = token_ids
        self.token_texts = token_texts

    def observe(
        self,
        *,
        layer: int,
        query_token: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        scores: torch.Tensor,
        query_index: int,
    ) -> None:
        if layer not in self.selected_layers or query_token not in self.query_tokens:
            return
        finite = torch.isfinite(scores[:, :, query_index, :])
        valid_count = int(finite[0, 0].sum().item())
        if valid_count <= 2:
            return
        history_count = valid_count - 1
        top_count = min(history_count, max(1, math.ceil(self.top_fraction * history_count)))
        row_scores = scores[0, :, query_index, :history_count].detach().float()
        attention_weights = F.softmax(scores[0, :, query_index, :valid_count].detach().float(), dim=-1)[
            :, :history_count
        ]
        top_indices = torch.topk(row_scores, k=top_count, dim=-1, largest=True).indices.detach()

        for head in range(key_states.shape[1]):
            if head not in self.selected_heads:
                continue
            self._observe_head(
                layer=layer,
                head=head,
                query_token=query_token,
                query_vector=query_states[0, head, query_index, :].detach(),
                key_matrix=key_states[0, head, :history_count, :].detach(),
                score_row=row_scores[head],
                attention_row=attention_weights[head],
                top_tokens=top_indices[head],
            )

    def _observe_head(
        self,
        *,
        layer: int,
        head: int,
        query_token: int,
        query_vector: torch.Tensor,
        key_matrix: torch.Tensor,
        score_row: torch.Tensor,
        attention_row: torch.Tensor,
        top_tokens: torch.Tensor,
    ) -> None:
        working = key_matrix.to(device=self.svd_device, dtype=self.svd_dtype)
        mean = working.mean(dim=0, keepdim=True) if self.center_k else torch.zeros_like(working[:1])
        working = working - mean
        try:
            _, singular_values, vh = torch.linalg.svd(working, full_matrices=False)
        except RuntimeError:
            self.skipped_svd_rows += 1
            return
        self.observed_rows += 1

        rank = int(singular_values.numel())
        cutoffs = sorted({r for r in self.rank_cutoffs if r <= rank} | {rank})
        basis = vh.transpose(0, 1)
        q = query_vector.to(device=self.svd_device, dtype=self.svd_dtype).view(1, -1) - mean
        q_coeff = (q @ basis).flatten()
        q_norm = float(torch.linalg.vector_norm(q_coeff).item())

        group_tokens = self._tokens_for_row(int(key_matrix.shape[0]), top_tokens.detach().cpu().tolist())
        token_to_is_top2 = {int(token): True for token in top_tokens.detach().cpu().tolist()}
        group_coeffs: dict[str, torch.Tensor] = {}

        for group_name, token_indices in group_tokens.items():
            coeff_rows = []
            for token_index in token_indices:
                if token_index < 0 or token_index >= key_matrix.shape[0]:
                    continue
                k = key_matrix[token_index].to(device=self.svd_device, dtype=self.svd_dtype).view(1, -1) - mean
                k_coeff = (k @ basis).flatten()
                coeff_rows.append(k_coeff)
                self._add_token_event(
                    group=group_name,
                    layer=layer,
                    head=head,
                    token_index=token_index,
                    history_count=int(key_matrix.shape[0]),
                    is_top2=float(1.0 if token_to_is_top2.get(token_index, False) else 0.0),
                    q_coeff=q_coeff,
                    k_coeff=k_coeff,
                    q_norm=q_norm,
                    full_score=float(score_row[token_index].item()),
                    attention_mass=float(attention_row[token_index].item()),
                    cutoffs=cutoffs,
                )
            if coeff_rows:
                group_coeffs[group_name] = torch.stack(coeff_rows, dim=0).mean(dim=0)

        self._add_pair_events(group_coeffs, cutoffs)

    def _tokens_for_row(self, history_count: int, top_tokens: list[int]) -> dict[str, list[int]]:
        groups: dict[str, set[int]] = {}
        for name, tokens in self.base_groups.items():
            groups[name] = {token for token in tokens if 0 <= token < history_count}
        groups["top2_selected"] = {int(token) for token in top_tokens if 0 <= int(token) < history_count}
        if self.include_other_sample:
            covered = set().union(*groups.values()) if groups else set()
            other = [idx for idx in range(history_count) if idx not in covered]
            groups["other_sample"] = set(other[: self.max_tokens_per_group_per_row])
        capped: dict[str, list[int]] = {}
        cap = max(1, self.max_tokens_per_group_per_row)
        for name, tokens in groups.items():
            ordered = sorted(tokens)
            if name != "top2_selected":
                ordered = ordered[:cap]
            if ordered:
                capped[name] = ordered
        return capped

    def _add_token_event(
        self,
        *,
        group: str,
        layer: int,
        head: int,
        token_index: int,
        history_count: int,
        is_top2: float,
        q_coeff: torch.Tensor,
        k_coeff: torch.Tensor,
        q_norm: float,
        full_score: float,
        attention_mass: float,
        cutoffs: list[int],
    ) -> None:
        k_norm = float(torch.linalg.vector_norm(k_coeff).item())
        dot = q_coeff * k_coeff
        full_dot = float(dot.sum().item())
        abs_dot = dot.abs()
        abs_dot_total = float(abs_dot.sum().item())
        k_energy_cdf = torch.cumsum(k_coeff.square(), dim=0)
        k_energy_total = float(k_energy_cdf[-1].item()) if k_energy_cdf.numel() else 0.0
        abs_dot_cdf = torch.cumsum(abs_dot, dim=0)
        values = {
            "selected_rate": is_top2,
            "attention_mass": attention_mass,
            "full_score": full_score,
            "full_cosine": full_dot / (q_norm * k_norm) if q_norm > 0.0 and k_norm > 0.0 else 0.0,
            "relative_position": token_index / max(1.0, float(history_count)),
            "distance_to_query": float(history_count - token_index),
        }
        for cutoff in cutoffs:
            values[f"token_energy_top{cutoff}"] = (
                float(k_energy_cdf[cutoff - 1].item()) / k_energy_total if k_energy_total > 0.0 else 0.0
            )
            values[f"abs_qk_contrib_top{cutoff}"] = (
                float(abs_dot_cdf[cutoff - 1].item()) / abs_dot_total if abs_dot_total > 0.0 else 0.0
            )

        for key in [("overall", group, "all"), ("layer", group, layer), ("layer_head", group, (layer, head))]:
            self.type_acc[key].add(values)

        max_dir = min(self.direction_count, int(k_coeff.numel()))
        for direction in range(max_dir):
            dir_values = {
                "mean_coeff": float(k_coeff[direction].item()),
                "mean_abs_coeff": float(k_coeff[direction].abs().item()),
                "mean_coeff_sq": float(k_coeff[direction].square().item()),
                "mean_qk_prod": float(dot[direction].item()),
                "mean_abs_qk_prod": float(abs_dot[direction].item()),
            }
            self.direction_acc[(group, direction + 1)].add(dir_values)

        self._add_position_event(group, layer, history_count, token_index, is_top2, attention_mass)
        self._add_lexical_event(group, token_index)

    def _add_position_event(
        self,
        group: str,
        layer: int,
        history_count: int,
        token_index: int,
        is_top2: float,
        attention_mass: float,
    ) -> None:
        distance = history_count - token_index
        if token_index < self.sink_tokens:
            bucket = "sink"
        elif distance <= 8:
            bucket = "recent_1_8"
        elif distance <= 16:
            bucket = "recent_9_16"
        elif distance <= 32:
            bucket = "recent_17_32"
        elif distance <= 64:
            bucket = "recent_33_64"
        elif distance <= 128:
            bucket = "middle_65_128"
        else:
            bucket = "remote_129_plus"
        values = {
            "selected_rate": is_top2,
            "attention_mass": attention_mass,
            "distance_to_query": float(distance),
        }
        for key in [("overall", bucket, "all"), ("group", f"{group}:{bucket}", "all"), ("layer", bucket, layer)]:
            self.position_acc[key].add(values)

    def _add_lexical_event(self, group: str, token_index: int) -> None:
        if 0 <= token_index < len(self.token_texts):
            text = self.token_texts[token_index]
            self.lexical_counts[(group, classify_piece(text))] += 1
            normalized = text.replace("\n", "\\n")
            if normalized:
                self.token_examples[group][normalized] += 1

    def _add_pair_events(self, group_coeffs: dict[str, torch.Tensor], cutoffs: list[int]) -> None:
        names = sorted(group_coeffs)
        for left_i, left in enumerate(names):
            a = group_coeffs[left]
            for right in names[left_i + 1 :]:
                b = group_coeffs[right]
                rank = min(int(a.numel()), int(b.numel()))
                for cutoff in cutoffs:
                    if cutoff > rank:
                        continue
                    av = a[:cutoff]
                    bv = b[:cutoff]
                    an = float(torch.linalg.vector_norm(av).item())
                    bn = float(torch.linalg.vector_norm(bv).item())
                    cosine = float((av * bv).sum().item()) / (an * bn) if an > 0.0 and bn > 0.0 else 0.0
                    self.pair_acc[(left, right, cutoff)].add({"centroid_cosine": cosine})

    def type_rows(self, metric_fields: list[str]) -> list[dict[str, Any]]:
        rows = []
        for (scope, group, index), acc in sorted(self.type_acc.items(), key=lambda item: str(item[0])):
            row: dict[str, Any] = {"scope": scope, "group": group, "layer": "", "head": ""}
            if scope == "layer":
                row["layer"] = index
            elif scope == "layer_head":
                layer, head = index  # type: ignore[misc]
                row["layer"] = layer
                row["head"] = head
            rows.append(acc.row(row, metric_fields))
        return rows

    def direction_rows(self) -> list[dict[str, Any]]:
        fields = ["mean_coeff", "mean_abs_coeff", "mean_coeff_sq", "mean_qk_prod", "mean_abs_qk_prod"]
        rows = []
        for (group, direction), acc in sorted(self.direction_acc.items(), key=lambda item: str(item[0])):
            rows.append(acc.row({"group": group, "direction": direction}, fields))
        return rows

    def pair_rows(self) -> list[dict[str, Any]]:
        rows = []
        for (left, right, rank), acc in sorted(self.pair_acc.items(), key=lambda item: str(item[0])):
            rows.append(acc.row({"group_left": left, "group_right": right, "rank": rank}, ["centroid_cosine"]))
        return rows

    def position_rows(self) -> list[dict[str, Any]]:
        rows = []
        for (scope, bucket, layer), acc in sorted(self.position_acc.items(), key=lambda item: str(item[0])):
            rows.append(acc.row({"scope": scope, "bucket": bucket, "layer": "" if layer == "all" else layer}, [
                "selected_rate",
                "attention_mass",
                "distance_to_query",
            ]))
        return rows

    def lexical_rows(self) -> list[dict[str, Any]]:
        rows = []
        totals: Counter[str] = Counter()
        for (group, _category), count in self.lexical_counts.items():
            totals[group] += count
        for (group, category), count in sorted(self.lexical_counts.items()):
            rows.append(
                {
                    "group": group,
                    "category": category,
                    "count": count,
                    "fraction": count / totals[group] if totals[group] else 0.0,
                }
            )
        return rows

    def example_rows(self, top_k: int = 20) -> list[dict[str, Any]]:
        rows = []
        for group, counter in sorted(self.token_examples.items()):
            for text, count in counter.most_common(top_k):
                rows.append({"group": group, "token_text": text, "count": count})
        return rows


def build_base_groups(
    *,
    prefill_tokens: int,
    spans: dict[str, tuple[int, int]],
    sink_tokens: int,
    recent_tokens: int,
) -> dict[str, set[int]]:
    groups: dict[str, set[int]] = {
        "sink": set(range(0, min(sink_tokens, prefill_tokens))),
        "recent": set(range(max(0, prefill_tokens - recent_tokens), prefill_tokens)),
        "recent_1_16": set(range(max(0, prefill_tokens - 16), prefill_tokens)),
        "recent_17_64": set(range(max(0, prefill_tokens - recent_tokens), max(0, prefill_tokens - 16))),
    }
    for name, (start, end) in spans.items():
        groups[f"evidence_{name}"] = set(range(max(0, start), min(prefill_tokens, end)))
    evidence_all = set()
    for name, tokens in groups.items():
        if name.startswith("evidence_"):
            evidence_all.update(tokens)
    if evidence_all:
        groups["evidence_any"] = evidence_all
    return {name: tokens for name, tokens in groups.items() if tokens}


def _token_type_eager_attention_forward(
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

    if _ACTIVE_COLLECTOR is not None:
        layer = int(getattr(module, "layer_idx", 0))
        query_count = scores.shape[-2]
        key_count = scores.shape[-1]
        chunk_query_start = key_count - query_count
        for query_index in range(query_count):
            query_token = chunk_query_start + query_index
            _ACTIVE_COLLECTOR.observe(
                layer=layer,
                query_token=query_token,
                query_states=query_states,
                key_states=key_states,
                scores=scores,
                query_index=query_index,
            )

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
        setattr(modeling_qwen3, "eager_attention_forward", _token_type_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _token_type_eager_attention_forward


@contextmanager
def active_collector(collector: TokenTypeSpectralCollector):
    global _ACTIVE_COLLECTOR
    previous = _ACTIVE_COLLECTOR
    _ACTIVE_COLLECTOR = collector
    try:
        yield
    finally:
        _ACTIVE_COLLECTOR = previous


@torch.inference_mode()
def prefill_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    chunk_size: int,
    input_device: torch.device,
) -> Any:
    past_key_values = None
    total_chunks = math.ceil(prefill_tokens / chunk_size)
    for chunk_idx, start in enumerate(range(0, prefill_tokens, chunk_size), start=1):
        end = min(start + chunk_size, prefill_tokens)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids[:, start:end].to(input_device),
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
        del outputs
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"prefill chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
    return past_key_values


@torch.inference_mode()
def run_query_eval(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    collector: TokenTypeSpectralCollector,
) -> None:
    eval_end = prefill_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / chunk_size)
    with active_collector(collector):
        for chunk_idx, start in enumerate(range(prefill_tokens, eval_end, chunk_size), start=1):
            end = min(start + chunk_size, eval_end)
            kwargs: dict[str, Any] = {
                "input_ids": input_ids[:, start:end].to(input_device),
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
            del outputs
            if input_device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"eval chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in variants if name not in BUILDERS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available={sorted(BUILDERS)}")

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    install_qwen3_attention_patch()
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, requested_device)

    layer_count = int(model.config.num_hidden_layers)
    head_count = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
    selected_layers = parse_index_spec(args.layers, layer_count, "layers")
    selected_heads = parse_index_spec(args.heads, head_count, "heads")
    rank_cutoffs = parse_rank_cutoffs(args.rank_cutoffs, head_dim)

    collector = TokenTypeSpectralCollector(
        selected_layers=selected_layers,
        selected_heads=selected_heads,
        rank_cutoffs=rank_cutoffs,
        direction_count=min(args.direction_count, head_dim),
        top_fraction=args.top_fraction,
        sink_tokens=args.sink_tokens,
        recent_tokens=args.recent_tokens,
        max_tokens_per_group_per_row=args.max_tokens_per_group_per_row,
        include_other_sample=args.include_other_sample,
        center_k=args.center_k,
        svd_device=torch.device(args.svd_device if torch.cuda.is_available() else "cpu"),
        svd_dtype=torch.float64 if args.svd_dtype == "float64" else torch.float32,
    )

    rng = random.Random(args.seed)
    start_time = time.perf_counter()
    task_rows: list[dict[str, Any]] = []
    for variant in variants:
        tasks = [BUILDERS[variant](rng, idx, args.records_per_task) for idx in range(args.tasks_per_variant)]
        for task_number, task in enumerate(tasks, start=1):
            if task_number == 1 or task_number == len(tasks) or task_number % args.log_every == 0:
                print(f"{variant} task {task_number}/{len(tasks)}", flush=True)
            context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            query_ids = tokenizer(task["query"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            if query_ids.numel() == 0:
                continue
            input_ids = torch.cat([context_ids, query_ids], dim=1)
            prefill_tokens = int(context_ids.shape[1])
            eval_tokens = int(query_ids.shape[1])
            spans = evidence_spans(tokenizer, task)
            base_groups = build_base_groups(
                prefill_tokens=prefill_tokens,
                spans=spans,
                sink_tokens=args.sink_tokens,
                recent_tokens=args.recent_tokens,
            )
            context_token_ids = context_ids[0].tolist()
            token_texts = [tokenizer.decode([token_id], clean_up_tokenization_spaces=False) for token_id in context_token_ids]
            query_tokens = set(choose_query_tokens(prefill_tokens, eval_tokens, args.max_query_tokens_per_task))
            collector.set_task(
                task_meta={"variant": variant, "task_id": task["task_id"]},
                query_tokens=query_tokens,
                base_groups=base_groups,
                token_ids=context_token_ids,
                token_texts=token_texts,
            )
            task_rows.append(
                {
                    "variant": variant,
                    "task_id": task["task_id"],
                    "prefill_tokens": prefill_tokens,
                    "query_tokens": eval_tokens,
                    "sampled_query_tokens": " ".join(str(token) for token in sorted(query_tokens)),
                    "target_key": task["target_key"],
                    "target_label": task["target_label"],
                    "key_span": f"{spans.get('key', ('', ''))[0]}:{spans.get('key', ('', ''))[1]}" if "key" in spans else "",
                    "label_span": f"{spans.get('label', ('', ''))[0]}:{spans.get('label', ('', ''))[1]}" if "label" in spans else "",
                    "record_span": f"{spans.get('record', ('', ''))[0]}:{spans.get('record', ('', ''))[1]}" if "record" in spans else "",
                }
            )
            past_key_values = prefill_cache(model, input_ids, prefill_tokens, args.chunk_size, input_device)
            run_query_eval(model, input_ids, past_key_values, prefill_tokens, eval_tokens, args.chunk_size, input_device, collector)
            del past_key_values
            if input_device.type == "cuda":
                torch.cuda.empty_cache()

    metric_fields = [
        "selected_rate",
        "attention_mass",
        "full_score",
        "full_cosine",
        "relative_position",
        "distance_to_query",
    ]
    for cutoff in rank_cutoffs:
        metric_fields.append(f"token_energy_top{cutoff}")
        metric_fields.append(f"abs_qk_contrib_top{cutoff}")

    write_csv(
        output_dir / "token_type_stats.csv",
        collector.type_rows(metric_fields),
        ["scope", "group", "layer", "head", "rows", "weight", *metric_fields],
    )
    write_csv(
        output_dir / "direction_stats.csv",
        collector.direction_rows(),
        [
            "group",
            "direction",
            "rows",
            "weight",
            "mean_coeff",
            "mean_abs_coeff",
            "mean_coeff_sq",
            "mean_qk_prod",
            "mean_abs_qk_prod",
        ],
    )
    write_csv(
        output_dir / "pair_direction_correlations.csv",
        collector.pair_rows(),
        ["group_left", "group_right", "rank", "rows", "weight", "centroid_cosine"],
    )
    write_csv(
        output_dir / "recent_position_bins.csv",
        collector.position_rows(),
        ["scope", "bucket", "layer", "rows", "weight", "selected_rate", "attention_mass", "distance_to_query"],
    )
    write_csv(
        output_dir / "lexical_type_stats.csv",
        collector.lexical_rows(),
        ["group", "category", "count", "fraction"],
    )
    write_csv(
        output_dir / "token_text_examples.csv",
        collector.example_rows(),
        ["group", "token_text", "count"],
    )
    write_csv(
        output_dir / "tasks.csv",
        task_rows,
        [
            "variant",
            "task_id",
            "prefill_tokens",
            "query_tokens",
            "sampled_query_tokens",
            "target_key",
            "target_label",
            "key_span",
            "label_span",
            "record_span",
        ],
    )

    summary = {
        "output_dir": str(output_dir),
        "seconds": time.perf_counter() - start_time,
        "resolved": {
            "layer_count": layer_count,
            "head_count": head_count,
            "head_dim": head_dim,
            "selected_layers": selected_layers,
            "selected_heads": selected_heads,
            "rank_cutoffs": rank_cutoffs,
            "direction_count": min(args.direction_count, head_dim),
            "tasks": len(task_rows),
            "observed_rows": collector.observed_rows,
            "skipped_svd_rows": collector.skipped_svd_rows,
            "token_groups": sorted({key[1] for key in collector.type_acc}),
        },
        "outputs": {
            "token_type_stats": str(output_dir / "token_type_stats.csv"),
            "direction_stats": str(output_dir / "direction_stats.csv"),
            "pair_direction_correlations": str(output_dir / "pair_direction_correlations.csv"),
            "recent_position_bins": str(output_dir / "recent_position_bins.csv"),
            "lexical_type_stats": str(output_dir / "lexical_type_stats.csv"),
            "token_text_examples": str(output_dir / "token_text_examples.csv"),
            "tasks": str(output_dir / "tasks.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
