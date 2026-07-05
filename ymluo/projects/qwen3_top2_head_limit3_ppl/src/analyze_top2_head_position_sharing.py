from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _install_torchvision_fake_registration_guard() -> None:
    register_fake = getattr(torch.library, "register_fake", None)
    if register_fake is None or getattr(register_fake, "_top2_share_guarded", False):
        return

    def guarded_register_fake(op_name: str, *args: Any, **kwargs: Any):
        decorator = register_fake(op_name, *args, **kwargs)

        def guarded_decorator(fn: Any) -> Any:
            try:
                return decorator(fn)
            except RuntimeError as exc:
                if "operator torchvision::" in str(exc) and "does not exist" in str(exc):
                    return fn
                raise

        return guarded_decorator

    guarded_register_fake._top2_share_guarded = True
    torch.library.register_fake = guarded_register_fake


_install_torchvision_fake_registration_guard()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "ymluo/models/Qwen3-0.6B"
DEFAULT_TEXT_PATH = "external/needle-in-a-haystack/needlehaystack/PaulGrahamEssays/worked.txt"

_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_COLLECTOR: "HeadPositionSharingCollector | None" = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether Qwen3 attention heads can share true full-QK top-fraction "
            "historical token position selections."
        )
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/top2_head_position_sharing")
    parser.add_argument("--total_tokens", type=int, default=2048)
    parser.add_argument("--prefill_tokens", type=int, default=1536)
    parser.add_argument("--eval_tokens", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument(
        "--remote_only",
        type=str2bool,
        default=False,
        help="If true, select true top-fraction over all history, then keep only non-sink/non-recent positions.",
    )
    parser.add_argument("--exclude_sink_tokens", type=int, default=0)
    parser.add_argument("--exclude_recent_tokens", type=int, default=0)
    parser.add_argument("--max_query_samples", type=int, default=0, help="Use <=0 to analyze all eval queries.")
    parser.add_argument("--query_stride", type=int, default=0)
    parser.add_argument("--group_recall_thresholds", default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--stability_thresholds", default="0.50,0.70,0.80,0.90")
    parser.add_argument("--include_token_text", type=str2bool, default=True)
    parser.add_argument("--write_query_layer_group_metrics", type=str2bool, default=False)
    parser.add_argument("--write_query_layer_pair_metrics", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def parse_float_list(value: str, name: str) -> list[float]:
    numbers = sorted({float(part) for part in value.split(",") if part.strip()})
    if not numbers:
        raise ValueError(f"--{name} must contain at least one number.")
    invalid = [number for number in numbers if not (0.0 <= number <= 1.0)]
    if invalid:
        raise ValueError(f"--{name} values must be in [0, 1], got {invalid}.")
    return numbers


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


def threshold_field(prefix: str, threshold: float) -> str:
    return f"{prefix}_{threshold:.2f}".replace(".", "p")


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]


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


def safe_token_text(tokenizer: Any, token_id: int) -> tuple[str, str]:
    piece = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    return str(piece).replace("\n", "\\n").replace("\r", "\\r"), text.replace("\n", "\\n").replace("\r", "\\r")


def token_prefix(
    tokenizer: Any,
    token_ids: list[int],
    query_token: int,
    include_token_text: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "query_token_index": query_token,
        "query_token_id": int(token_ids[query_token]),
    }
    if include_token_text:
        piece, text = safe_token_text(tokenizer, token_ids[query_token])
        row["query_token_piece"] = piece
        row["query_token_text"] = text
    return row


def build_query_samples(prefill_tokens: int, eval_tokens: int, query_stride: int, max_query_samples: int) -> list[int]:
    queries = list(range(prefill_tokens, prefill_tokens + eval_tokens))
    if query_stride > 0:
        queries = queries[::query_stride]
    if max_query_samples > 0 and len(queries) > max_query_samples:
        if max_query_samples == 1:
            return [queries[len(queries) // 2]]
        step = (len(queries) - 1) / (max_query_samples - 1)
        indices = sorted({round(i * step) for i in range(max_query_samples)})
        queries = [queries[index] for index in indices]
    return queries


@dataclass
class MeanAccumulator:
    cases: int = 0
    sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, values: dict[str, float], weight: int = 1) -> None:
        if weight <= 0:
            return
        self.cases += int(weight)
        for key, value in values.items():
            if math.isfinite(float(value)):
                self.sums[key] += float(value) * weight

    def row(self, extra: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        row = {**extra, "cases": self.cases}
        for field_name in fields:
            row[field_name] = self.sums.get(field_name, 0.0) / self.cases if self.cases else 0.0
        return row


class DirectedSharingAccumulator:
    def __init__(self, thresholds: list[float]) -> None:
        self.thresholds = thresholds
        self.cases = 0
        self.intersection_sum = 0.0
        self.donor_size_sum = 0.0
        self.target_size_sum = 0.0
        self.union_size_sum = 0.0
        self.top2_recall_sum = 0.0
        self.top2_recall_sumsq = 0.0
        self.top2_jaccard_sum = 0.0
        self.target_top2_attention_mass_sum = 0.0
        self.donor_position_attention_mass_sum = 0.0
        self.attention_mass_recall_sum = 0.0
        self.recall_min = float("inf")
        self.recall_max = float("-inf")
        self.recall_ge_counts = {threshold: 0 for threshold in thresholds}

    def add(
        self,
        intersection: float,
        donor_size: float,
        target_size: float,
        union_size: float,
        top2_recall: float,
        top2_jaccard: float,
        target_top2_attention_mass: float,
        donor_position_attention_mass: float,
        attention_mass_recall: float,
    ) -> None:
        self.cases += 1
        self.intersection_sum += intersection
        self.donor_size_sum += donor_size
        self.target_size_sum += target_size
        self.union_size_sum += union_size
        self.top2_recall_sum += top2_recall
        self.top2_recall_sumsq += top2_recall * top2_recall
        self.top2_jaccard_sum += top2_jaccard
        self.target_top2_attention_mass_sum += target_top2_attention_mass
        self.donor_position_attention_mass_sum += donor_position_attention_mass
        self.attention_mass_recall_sum += attention_mass_recall
        self.recall_min = min(self.recall_min, top2_recall)
        self.recall_max = max(self.recall_max, top2_recall)
        for threshold in self.thresholds:
            if top2_recall >= threshold:
                self.recall_ge_counts[threshold] += 1

    def mean_top2_recall(self) -> float:
        return self.top2_recall_sum / self.cases if self.cases else 0.0

    def mean_attention_mass_recall(self) -> float:
        return self.attention_mass_recall_sum / self.cases if self.cases else 0.0

    def recall_ge_fraction(self, threshold: float) -> float:
        return self.recall_ge_counts.get(threshold, 0) / self.cases if self.cases else 0.0

    def row(self, extra: dict[str, Any]) -> dict[str, Any]:
        mean_recall = self.mean_top2_recall()
        variance = self.top2_recall_sumsq / self.cases - mean_recall * mean_recall if self.cases else 0.0
        row = {
            **extra,
            "cases": self.cases,
            "mean_intersection": self.intersection_sum / self.cases if self.cases else 0.0,
            "mean_donor_size": self.donor_size_sum / self.cases if self.cases else 0.0,
            "mean_target_size": self.target_size_sum / self.cases if self.cases else 0.0,
            "mean_union_size": self.union_size_sum / self.cases if self.cases else 0.0,
            "top2_recall_mean": mean_recall,
            "top2_recall_std": math.sqrt(max(0.0, variance)),
            "top2_recall_min": self.recall_min if self.cases else 0.0,
            "top2_recall_max": self.recall_max if self.cases else 0.0,
            "top2_jaccard_mean": self.top2_jaccard_sum / self.cases if self.cases else 0.0,
            "target_top2_attention_mass_mean": self.target_top2_attention_mass_sum / self.cases
            if self.cases
            else 0.0,
            "donor_position_attention_mass_mean": self.donor_position_attention_mass_sum / self.cases
            if self.cases
            else 0.0,
            "attention_mass_recall_mean": self.mean_attention_mass_recall(),
        }
        for threshold in self.thresholds:
            row[f"{threshold_field('top2_recall_ge', threshold)}_fraction"] = self.recall_ge_fraction(threshold)
        return row


class HeadPositionSharingCollector:
    def __init__(
        self,
        selected_layers: list[int],
        selected_heads: list[int],
        query_tokens: set[int],
        top_fraction: float,
        remote_only: bool,
        exclude_sink_tokens: int,
        exclude_recent_tokens: int,
        stability_thresholds: list[float],
        write_query_layer_pair_metrics: bool,
    ) -> None:
        self.selected_layers = selected_layers
        self.selected_layers_set = set(selected_layers)
        self.selected_heads = selected_heads
        self.query_tokens = query_tokens
        self.top_fraction = top_fraction
        self.remote_only = remote_only
        self.exclude_sink_tokens = exclude_sink_tokens
        self.exclude_recent_tokens = exclude_recent_tokens
        self.stability_thresholds = stability_thresholds
        self.write_query_layer_pair_metrics = write_query_layer_pair_metrics

        self.pair_stats: dict[tuple[int, int, int], DirectedSharingAccumulator] = {}
        self.query_stats: dict[int, MeanAccumulator] = defaultdict(MeanAccumulator)
        self.query_layer_stats: dict[tuple[int, int], MeanAccumulator] = defaultdict(MeanAccumulator)
        self.query_layer_matrices: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self.query_layer_pair_rows: list[dict[str, Any]] = []
        self.observed_query_tokens: set[int] = set()
        self.topk_by_query: dict[int, int] = {}

    def _pair_accumulator(self, layer: int, donor_head: int, target_head: int) -> DirectedSharingAccumulator:
        key = (layer, donor_head, target_head)
        if key not in self.pair_stats:
            self.pair_stats[key] = DirectedSharingAccumulator(self.stability_thresholds)
        return self.pair_stats[key]

    def observe(self, layer: int, query_token: int, scores: torch.Tensor, query_index: int) -> None:
        if layer not in self.selected_layers_set or query_token not in self.query_tokens:
            return
        finite = torch.isfinite(scores[:, :, query_index, :])
        valid_count = min(int(finite[0, 0].sum().item()), query_token + 1)
        if valid_count <= 1:
            return

        history_count = valid_count - 1
        top_count = min(history_count, max(1, math.ceil(self.top_fraction * history_count)))
        head_index = torch.tensor(self.selected_heads, dtype=torch.long, device=scores.device)
        row_scores = scores[0, head_index, query_index, :history_count].detach().float()
        top_indices = torch.topk(row_scores, k=top_count, dim=-1, largest=True).indices
        if self.remote_only:
            remote_end = max(0, history_count - self.exclude_recent_tokens)
            selected_top_mask = (top_indices >= self.exclude_sink_tokens) & (top_indices < remote_end)
        else:
            selected_top_mask = torch.ones_like(top_indices, dtype=torch.bool)

        selected = torch.zeros((len(self.selected_heads), history_count), dtype=torch.bool, device=row_scores.device)
        selected.scatter_(1, top_indices, selected_top_mask)
        selected_f = selected.float()
        target_sizes = selected_f.sum(dim=1)
        if int((target_sizes > 0).sum().item()) == 0:
            return

        attention_weights = F.softmax(scores[0, head_index, query_index, :valid_count].detach().float(), dim=-1)[
            :, :history_count
        ]
        intersections = selected_f @ selected_f.T
        unions = target_sizes[:, None] + target_sizes[None, :] - intersections
        top2_recall = intersections / target_sizes[None, :].clamp_min(1.0)
        top2_jaccard = intersections / unions.clamp_min(1.0)
        target_top2_mass = (selected_f * attention_weights).sum(dim=1)
        donor_position_mass = selected_f @ attention_weights.T
        attention_mass_recall = donor_position_mass / target_top2_mass[None, :].clamp_min(1.0e-30)
        valid_target = target_sizes > 0

        self.observed_query_tokens.add(query_token)
        self.topk_by_query[query_token] = top_count
        matrix_record = {
            "top2_recall": top2_recall.detach().cpu(),
            "top2_jaccard": top2_jaccard.detach().cpu(),
            "attention_mass_recall": attention_mass_recall.detach().cpu(),
            "target_top2_mass": target_top2_mass.detach().cpu(),
            "target_sizes": target_sizes.detach().cpu(),
        }
        self.query_layer_matrices[(query_token, layer)] = matrix_record

        directed_cases = 0
        query_sums = defaultdict(float)
        n_heads = len(self.selected_heads)
        for donor_idx, donor_head in enumerate(self.selected_heads):
            donor_size = float(target_sizes[donor_idx].item())
            for target_idx, target_head in enumerate(self.selected_heads):
                if donor_idx == target_idx or not bool(valid_target[target_idx].item()):
                    continue
                directed_cases += 1
                intersection = float(intersections[donor_idx, target_idx].item())
                target_size = float(target_sizes[target_idx].item())
                union_size = float(unions[donor_idx, target_idx].item())
                recall_value = float(top2_recall[donor_idx, target_idx].item())
                jaccard_value = float(top2_jaccard[donor_idx, target_idx].item())
                target_mass_value = float(target_top2_mass[target_idx].item())
                donor_mass_value = float(donor_position_mass[donor_idx, target_idx].item())
                mass_recall_value = float(attention_mass_recall[donor_idx, target_idx].item())

                self._pair_accumulator(layer, donor_head, target_head).add(
                    intersection=intersection,
                    donor_size=donor_size,
                    target_size=target_size,
                    union_size=union_size,
                    top2_recall=recall_value,
                    top2_jaccard=jaccard_value,
                    target_top2_attention_mass=target_mass_value,
                    donor_position_attention_mass=donor_mass_value,
                    attention_mass_recall=mass_recall_value,
                )
                query_sums["top2_recall_mean"] += recall_value
                query_sums["top2_jaccard_mean"] += jaccard_value
                query_sums["attention_mass_recall_mean"] += mass_recall_value
                query_sums["target_top2_attention_mass_mean"] += target_mass_value
                query_sums["donor_position_attention_mass_mean"] += donor_mass_value

                if self.write_query_layer_pair_metrics:
                    self.query_layer_pair_rows.append(
                        {
                            "query_token_index": query_token,
                            "layer": layer,
                            "donor_head": donor_head,
                            "target_head": target_head,
                            "donor_size": donor_size,
                            "target_size": target_size,
                            "intersection": intersection,
                            "union_size": union_size,
                            "top2_recall": recall_value,
                            "top2_jaccard": jaccard_value,
                            "target_top2_attention_mass": target_mass_value,
                            "donor_position_attention_mass": donor_mass_value,
                            "attention_mass_recall": mass_recall_value,
                        }
                    )

        if directed_cases:
            values = {key: value / directed_cases for key, value in query_sums.items()}
            self.query_stats[query_token].add(values, weight=directed_cases)
            self.query_layer_stats[(query_token, layer)].add(values, weight=directed_cases)

    def directed_pair_rows(self) -> list[dict[str, Any]]:
        return [
            stats.row({"layer": layer, "donor_head": donor_head, "target_head": target_head})
            for (layer, donor_head, target_head), stats in sorted(self.pair_stats.items())
        ]

    def mean_matrix(self, layer: int, metric: str) -> torch.Tensor:
        n_heads = len(self.selected_heads)
        matrix = torch.zeros((n_heads, n_heads), dtype=torch.float64)
        for idx in range(n_heads):
            matrix[idx, idx] = 1.0
        head_to_index = {head: idx for idx, head in enumerate(self.selected_heads)}
        for (stats_layer, donor_head, target_head), stats in self.pair_stats.items():
            if stats_layer != layer or stats.cases <= 0:
                continue
            donor_idx = head_to_index[donor_head]
            target_idx = head_to_index[target_head]
            if metric == "top2_recall":
                matrix[donor_idx, target_idx] = stats.mean_top2_recall()
            elif metric == "attention_mass_recall":
                matrix[donor_idx, target_idx] = stats.mean_attention_mass_recall()
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        return matrix

    def stability_matrix(self, layer: int, threshold: float) -> torch.Tensor:
        n_heads = len(self.selected_heads)
        matrix = torch.ones((n_heads, n_heads), dtype=torch.float64)
        head_to_index = {head: idx for idx, head in enumerate(self.selected_heads)}
        for (stats_layer, donor_head, target_head), stats in self.pair_stats.items():
            if stats_layer != layer:
                continue
            donor_idx = head_to_index[donor_head]
            target_idx = head_to_index[target_head]
            matrix[donor_idx, target_idx] = stats.recall_ge_fraction(threshold)
        return matrix


def _patched_eager_attention_forward(
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
            _ACTIVE_COLLECTOR.observe(layer, query_token, scores, query_index)

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
        setattr(modeling_qwen3, "eager_attention_forward", _patched_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _patched_eager_attention_forward


@contextmanager
def active_collector(collector: HeadPositionSharingCollector):
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
        print(f"prefill chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        del outputs
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    return past_key_values


@torch.inference_mode()
def run_eval(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    collector: HeadPositionSharingCollector,
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
            print(f"eval chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
            outputs = model_forward(model, kwargs)
            past_key_values = outputs.past_key_values
            del outputs
            if input_device.type == "cuda":
                torch.cuda.empty_cache()


def make_best_donor_rows(
    collector: HeadPositionSharingCollector,
    selected_layers: list[int],
    selected_heads: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in selected_layers:
        for target_head in selected_heads:
            candidates: list[tuple[int, DirectedSharingAccumulator]] = []
            for donor_head in selected_heads:
                if donor_head == target_head:
                    continue
                stats = collector.pair_stats.get((layer, donor_head, target_head))
                if stats is not None and stats.cases:
                    candidates.append((donor_head, stats))
            if not candidates:
                continue
            best_by_recall = max(candidates, key=lambda item: item[1].mean_top2_recall())
            best_by_mass = max(candidates, key=lambda item: item[1].mean_attention_mass_recall())
            recall_row = best_by_recall[1].row({})
            mass_row = best_by_mass[1].row({})
            row = {
                "layer": layer,
                "target_head": target_head,
                "best_donor_by_top2_recall": best_by_recall[0],
                "best_top2_recall_mean": recall_row["top2_recall_mean"],
                "best_top2_recall_std": recall_row["top2_recall_std"],
                "best_top2_jaccard_mean": recall_row["top2_jaccard_mean"],
                "best_top2_attention_mass_recall_mean": recall_row["attention_mass_recall_mean"],
                "best_donor_by_attention_mass": best_by_mass[0],
                "best_attention_mass_recall_mean": mass_row["attention_mass_recall_mean"],
                "best_attention_mass_top2_recall_mean": mass_row["top2_recall_mean"],
                "best_attention_mass_jaccard_mean": mass_row["top2_jaccard_mean"],
            }
            for threshold in collector.stability_thresholds:
                row[f"best_top2_{threshold_field('recall_ge', threshold)}_fraction"] = recall_row[
                    f"{threshold_field('top2_recall_ge', threshold)}_fraction"
                ]
                row[f"best_mass_{threshold_field('recall_ge', threshold)}_fraction"] = mass_row[
                    f"{threshold_field('top2_recall_ge', threshold)}_fraction"
                ]
            rows.append(row)
    return rows


def greedy_star_groups(
    heads: list[int],
    recall_matrix: torch.Tensor,
    mass_matrix: torch.Tensor,
    stability_matrix: torch.Tensor,
    threshold: float,
) -> list[dict[str, Any]]:
    n_heads = len(heads)
    remaining = set(range(n_heads))
    groups: list[dict[str, Any]] = []
    while remaining:
        best_rep = min(remaining)
        best_members: list[int] = [best_rep]
        best_score: tuple[int, float, float] = (1, 1.0, 1.0)
        for rep_idx in sorted(remaining):
            members = [
                member_idx
                for member_idx in sorted(remaining)
                if member_idx == rep_idx or float(recall_matrix[rep_idx, member_idx].item()) >= threshold
            ]
            non_self = [member_idx for member_idx in members if member_idx != rep_idx]
            if non_self:
                mean_recall = sum(float(recall_matrix[rep_idx, member_idx].item()) for member_idx in non_self) / len(
                    non_self
                )
                mean_mass = sum(float(mass_matrix[rep_idx, member_idx].item()) for member_idx in non_self) / len(
                    non_self
                )
            else:
                mean_recall = 1.0
                mean_mass = 1.0
            score = (len(members), mean_recall, mean_mass)
            if score > best_score:
                best_rep = rep_idx
                best_members = members
                best_score = score
        non_self_members = [member_idx for member_idx in best_members if member_idx != best_rep]
        if non_self_members:
            recall_values = [float(recall_matrix[best_rep, member_idx].item()) for member_idx in non_self_members]
            mass_values = [float(mass_matrix[best_rep, member_idx].item()) for member_idx in non_self_members]
            stability_values = [float(stability_matrix[best_rep, member_idx].item()) for member_idx in non_self_members]
        else:
            recall_values = [1.0]
            mass_values = [1.0]
            stability_values = [1.0]
        groups.append(
            {
                "representative_head": heads[best_rep],
                "member_heads": [heads[member_idx] for member_idx in best_members],
                "member_count": len(best_members),
                "selectors_saved": max(0, len(best_members) - 1),
                "shared_target_count": len(non_self_members),
                "rep_to_member_top2_recall_mean": sum(recall_values) / len(recall_values),
                "rep_to_member_top2_recall_min": min(recall_values),
                "rep_to_member_attention_mass_recall_mean": sum(mass_values) / len(mass_values),
                "rep_to_member_attention_mass_recall_min": min(mass_values),
                "rep_to_member_stable_fraction_mean": sum(stability_values) / len(stability_values),
                "rep_to_member_stable_fraction_min": min(stability_values),
            }
        )
        for member_idx in best_members:
            remaining.remove(member_idx)
    return groups


def make_group_rows(
    collector: HeadPositionSharingCollector,
    selected_layers: list[int],
    selected_heads: list[int],
    thresholds: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[float, int], list[dict[str, Any]]]]:
    group_rows: list[dict[str, Any]] = []
    summary_acc: dict[float, dict[str, float]] = {}
    groups_by_threshold_layer: dict[tuple[float, int], list[dict[str, Any]]] = {}
    for threshold in thresholds:
        summary_acc[threshold] = defaultdict(float)
        for layer in selected_layers:
            recall_matrix = collector.mean_matrix(layer, "top2_recall")
            mass_matrix = collector.mean_matrix(layer, "attention_mass_recall")
            stable_matrix = collector.stability_matrix(layer, threshold)
            groups = greedy_star_groups(selected_heads, recall_matrix, mass_matrix, stable_matrix, threshold)
            groups_by_threshold_layer[(threshold, layer)] = groups
            for group_id, group in enumerate(groups):
                row = {
                    "threshold": threshold,
                    "layer": layer,
                    "group_id": group_id,
                    "representative_head": group["representative_head"],
                    "member_heads": ",".join(str(head) for head in group["member_heads"]),
                    "member_count": group["member_count"],
                    "selectors_saved": group["selectors_saved"],
                    "shared_target_count": group["shared_target_count"],
                    "rep_to_member_top2_recall_mean": group["rep_to_member_top2_recall_mean"],
                    "rep_to_member_top2_recall_min": group["rep_to_member_top2_recall_min"],
                    "rep_to_member_attention_mass_recall_mean": group[
                        "rep_to_member_attention_mass_recall_mean"
                    ],
                    "rep_to_member_attention_mass_recall_min": group["rep_to_member_attention_mass_recall_min"],
                    "rep_to_member_stable_fraction_mean": group["rep_to_member_stable_fraction_mean"],
                    "rep_to_member_stable_fraction_min": group["rep_to_member_stable_fraction_min"],
                }
                group_rows.append(row)
                summary_acc[threshold]["group_count"] += 1
                summary_acc[threshold]["member_count_sum"] += group["member_count"]
                summary_acc[threshold]["selectors_saved"] += group["selectors_saved"]
                summary_acc[threshold]["groups_size_gt1"] += 1 if group["member_count"] > 1 else 0
                shared_count = max(0, int(group["shared_target_count"]))
                if shared_count:
                    summary_acc[threshold]["shared_target_count"] += shared_count
                    summary_acc[threshold]["shared_target_recall_sum"] += (
                        group["rep_to_member_top2_recall_mean"] * shared_count
                    )
                    summary_acc[threshold]["shared_target_mass_recall_sum"] += (
                        group["rep_to_member_attention_mass_recall_mean"] * shared_count
                    )
                    summary_acc[threshold]["shared_target_stable_fraction_sum"] += (
                        group["rep_to_member_stable_fraction_mean"] * shared_count
                    )

    total_selectors = len(selected_layers) * len(selected_heads)
    summary_rows = []
    for threshold in thresholds:
        acc = summary_acc[threshold]
        shared_targets = acc.get("shared_target_count", 0.0)
        group_count = acc.get("group_count", 0.0)
        summary_rows.append(
            {
                "threshold": threshold,
                "selected_layers": len(selected_layers),
                "heads_per_layer": len(selected_heads),
                "selectors_without_sharing": total_selectors,
                "selectors_with_sharing": int(group_count),
                "selectors_saved": int(acc.get("selectors_saved", 0.0)),
                "selector_reduction_fraction": acc.get("selectors_saved", 0.0) / total_selectors
                if total_selectors
                else 0.0,
                "groups_size_gt1": int(acc.get("groups_size_gt1", 0.0)),
                "mean_group_size": acc.get("member_count_sum", 0.0) / group_count if group_count else 0.0,
                "shared_target_count": int(shared_targets),
                "shared_target_top2_recall_mean": acc.get("shared_target_recall_sum", 0.0) / shared_targets
                if shared_targets
                else 1.0,
                "shared_target_attention_mass_recall_mean": acc.get("shared_target_mass_recall_sum", 0.0)
                / shared_targets
                if shared_targets
                else 1.0,
                "shared_target_stable_fraction_mean": acc.get("shared_target_stable_fraction_sum", 0.0)
                / shared_targets
                if shared_targets
                else 1.0,
            }
        )
    return group_rows, summary_rows, groups_by_threshold_layer


def make_query_rows(
    collector: HeadPositionSharingCollector,
    tokenizer: Any,
    token_ids: list[int],
    include_token_text: bool,
) -> list[dict[str, Any]]:
    metric_fields = [
        "top2_recall_mean",
        "top2_jaccard_mean",
        "attention_mass_recall_mean",
        "target_top2_attention_mass_mean",
        "donor_position_attention_mass_mean",
    ]
    rows = []
    for query_token, acc in sorted(collector.query_stats.items()):
        row = token_prefix(tokenizer, token_ids, query_token, include_token_text)
        row.update(acc.row({}, metric_fields))
        rows.append(row)
    return rows


def make_query_layer_rows(
    collector: HeadPositionSharingCollector,
    tokenizer: Any,
    token_ids: list[int],
    include_token_text: bool,
) -> list[dict[str, Any]]:
    metric_fields = [
        "top2_recall_mean",
        "top2_jaccard_mean",
        "attention_mass_recall_mean",
        "target_top2_attention_mass_mean",
        "donor_position_attention_mass_mean",
    ]
    rows = []
    for (query_token, layer), acc in sorted(collector.query_layer_stats.items()):
        row = token_prefix(tokenizer, token_ids, query_token, include_token_text)
        row["layer"] = layer
        row.update(acc.row({}, metric_fields))
        rows.append(row)
    return rows


def make_query_group_rows(
    collector: HeadPositionSharingCollector,
    selected_layers: list[int],
    selected_heads: list[int],
    thresholds: list[float],
    groups_by_threshold_layer: dict[tuple[float, int], list[dict[str, Any]]],
    tokenizer: Any,
    token_ids: list[int],
    include_token_text: bool,
    write_layer_rows: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    head_to_index = {head: idx for idx, head in enumerate(selected_heads)}
    query_rows: list[dict[str, Any]] = []
    query_layer_rows: list[dict[str, Any]] = []
    query_tokens = sorted(collector.observed_query_tokens)
    for threshold in thresholds:
        selector_count = sum(len(groups_by_threshold_layer.get((threshold, layer), [])) for layer in selected_layers)
        selectors_without_sharing = len(selected_layers) * len(selected_heads)
        for query_token in query_tokens:
            all_values: list[float] = []
            all_mass_values: list[float] = []
            shared_values: list[float] = []
            shared_mass_values: list[float] = []
            for layer in selected_layers:
                layer_record = collector.query_layer_matrices.get((query_token, layer))
                if layer_record is None:
                    continue
                layer_all_values: list[float] = []
                layer_all_mass_values: list[float] = []
                layer_shared_values: list[float] = []
                layer_shared_mass_values: list[float] = []
                recall_matrix = layer_record["top2_recall"]
                mass_matrix = layer_record["attention_mass_recall"]
                target_sizes = layer_record["target_sizes"]
                for group in groups_by_threshold_layer.get((threshold, layer), []):
                    rep_idx = head_to_index[int(group["representative_head"])]
                    for member_head in group["member_heads"]:
                        member_idx = head_to_index[int(member_head)]
                        if float(target_sizes[member_idx].item()) <= 0.0:
                            continue
                        if member_idx == rep_idx:
                            recall_value = 1.0
                            mass_value = 1.0
                        else:
                            recall_value = float(recall_matrix[rep_idx, member_idx].item())
                            mass_value = float(mass_matrix[rep_idx, member_idx].item())
                            shared_values.append(recall_value)
                            shared_mass_values.append(mass_value)
                            layer_shared_values.append(recall_value)
                            layer_shared_mass_values.append(mass_value)
                        all_values.append(recall_value)
                        all_mass_values.append(mass_value)
                        layer_all_values.append(recall_value)
                        layer_all_mass_values.append(mass_value)
                if write_layer_rows:
                    row = token_prefix(tokenizer, token_ids, query_token, include_token_text)
                    row.update(
                        {
                            "threshold": threshold,
                            "layer": layer,
                            "selectors_without_sharing": len(selected_heads),
                            "selectors_with_sharing": len(groups_by_threshold_layer.get((threshold, layer), [])),
                            "selectors_saved": len(selected_heads)
                            - len(groups_by_threshold_layer.get((threshold, layer), [])),
                            "all_head_cases": len(layer_all_values),
                            "all_head_top2_recall_mean": sum(layer_all_values) / len(layer_all_values)
                            if layer_all_values
                            else 0.0,
                            "all_head_attention_mass_recall_mean": sum(layer_all_mass_values)
                            / len(layer_all_mass_values)
                            if layer_all_mass_values
                            else 0.0,
                            "shared_target_cases": len(layer_shared_values),
                            "shared_target_top2_recall_mean": sum(layer_shared_values) / len(layer_shared_values)
                            if layer_shared_values
                            else 1.0,
                            "shared_target_attention_mass_recall_mean": sum(layer_shared_mass_values)
                            / len(layer_shared_mass_values)
                            if layer_shared_mass_values
                            else 1.0,
                        }
                    )
                    query_layer_rows.append(row)

            row = token_prefix(tokenizer, token_ids, query_token, include_token_text)
            row.update(
                {
                    "threshold": threshold,
                    "selectors_without_sharing": selectors_without_sharing,
                    "selectors_with_sharing": selector_count,
                    "selectors_saved": selectors_without_sharing - selector_count,
                    "selector_reduction_fraction": (selectors_without_sharing - selector_count)
                    / selectors_without_sharing
                    if selectors_without_sharing
                    else 0.0,
                    "all_head_cases": len(all_values),
                    "all_head_top2_recall_mean": sum(all_values) / len(all_values) if all_values else 0.0,
                    "all_head_attention_mass_recall_mean": sum(all_mass_values) / len(all_mass_values)
                    if all_mass_values
                    else 0.0,
                    "shared_target_cases": len(shared_values),
                    "shared_target_top2_recall_mean": sum(shared_values) / len(shared_values)
                    if shared_values
                    else 1.0,
                    "shared_target_attention_mass_recall_mean": sum(shared_mass_values) / len(shared_mass_values)
                    if shared_mass_values
                    else 1.0,
                }
            )
            query_rows.append(row)
    return query_rows, query_layer_rows


def main() -> None:
    args = parse_args()
    if not (0.0 < args.top_fraction <= 1.0):
        raise ValueError("--top_fraction must be in (0, 1].")
    if args.exclude_sink_tokens < 0 or args.exclude_recent_tokens < 0:
        raise ValueError("--exclude_sink_tokens and --exclude_recent_tokens must be non-negative.")
    if args.prefill_tokens + args.eval_tokens > args.total_tokens:
        raise ValueError("--prefill_tokens + --eval_tokens must be <= --total_tokens.")

    group_thresholds = parse_float_list(args.group_recall_thresholds, "group_recall_thresholds")
    stability_thresholds = sorted(
        set(parse_float_list(args.stability_thresholds, "stability_thresholds") + group_thresholds)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))
    if args.require_total_tokens and len(token_ids) < args.total_tokens:
        raise ValueError(f"Need {args.total_tokens} tokens, got {len(token_ids)}.")
    token_ids = token_ids[: args.total_tokens]
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": model_dtype}
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
    selected_layers = parse_index_spec(args.layers, layer_count, "layers")
    selected_heads = parse_index_spec(args.heads, head_count, "heads")
    query_samples = build_query_samples(args.prefill_tokens, args.eval_tokens, args.query_stride, args.max_query_samples)

    collector = HeadPositionSharingCollector(
        selected_layers=selected_layers,
        selected_heads=selected_heads,
        query_tokens=set(query_samples),
        top_fraction=args.top_fraction,
        remote_only=args.remote_only,
        exclude_sink_tokens=args.exclude_sink_tokens,
        exclude_recent_tokens=args.exclude_recent_tokens,
        stability_thresholds=stability_thresholds,
        write_query_layer_pair_metrics=args.write_query_layer_pair_metrics,
    )

    started = time.perf_counter()
    past = prefill_cache(model, input_ids, args.prefill_tokens, args.chunk_size, input_device)
    run_eval(model, input_ids, past, args.prefill_tokens, args.eval_tokens, args.chunk_size, input_device, collector)
    seconds = time.perf_counter() - started

    pair_rows = collector.directed_pair_rows()
    group_rows, threshold_summary_rows, groups_by_threshold_layer = make_group_rows(
        collector,
        selected_layers,
        selected_heads,
        group_thresholds,
    )
    best_donor_rows = make_best_donor_rows(collector, selected_layers, selected_heads)
    query_rows = make_query_rows(collector, tokenizer, token_ids, args.include_token_text)
    query_layer_rows = make_query_layer_rows(collector, tokenizer, token_ids, args.include_token_text)
    query_group_rows, query_layer_group_rows = make_query_group_rows(
        collector,
        selected_layers,
        selected_heads,
        group_thresholds,
        groups_by_threshold_layer,
        tokenizer,
        token_ids,
        args.include_token_text,
        args.write_query_layer_group_metrics,
    )

    pair_fields = [
        "layer",
        "donor_head",
        "target_head",
        "cases",
        "mean_intersection",
        "mean_donor_size",
        "mean_target_size",
        "mean_union_size",
        "top2_recall_mean",
        "top2_recall_std",
        "top2_recall_min",
        "top2_recall_max",
        "top2_jaccard_mean",
        "target_top2_attention_mass_mean",
        "donor_position_attention_mass_mean",
        "attention_mass_recall_mean",
    ]
    pair_fields += [f"{threshold_field('top2_recall_ge', threshold)}_fraction" for threshold in stability_thresholds]
    write_csv(output_dir / "head_sharing_donor_target.csv", pair_rows, pair_fields)

    best_fields = [
        "layer",
        "target_head",
        "best_donor_by_top2_recall",
        "best_top2_recall_mean",
        "best_top2_recall_std",
        "best_top2_jaccard_mean",
        "best_top2_attention_mass_recall_mean",
        "best_donor_by_attention_mass",
        "best_attention_mass_recall_mean",
        "best_attention_mass_top2_recall_mean",
        "best_attention_mass_jaccard_mean",
    ]
    for threshold in stability_thresholds:
        best_fields.append(f"best_top2_{threshold_field('recall_ge', threshold)}_fraction")
        best_fields.append(f"best_mass_{threshold_field('recall_ge', threshold)}_fraction")
    write_csv(output_dir / "head_best_donors.csv", best_donor_rows, best_fields)

    group_fields = [
        "threshold",
        "layer",
        "group_id",
        "representative_head",
        "member_heads",
        "member_count",
        "selectors_saved",
        "shared_target_count",
        "rep_to_member_top2_recall_mean",
        "rep_to_member_top2_recall_min",
        "rep_to_member_attention_mass_recall_mean",
        "rep_to_member_attention_mass_recall_min",
        "rep_to_member_stable_fraction_mean",
        "rep_to_member_stable_fraction_min",
    ]
    write_csv(output_dir / "shared_head_groups.csv", group_rows, group_fields)

    threshold_summary_fields = [
        "threshold",
        "selected_layers",
        "heads_per_layer",
        "selectors_without_sharing",
        "selectors_with_sharing",
        "selectors_saved",
        "selector_reduction_fraction",
        "groups_size_gt1",
        "mean_group_size",
        "shared_target_count",
        "shared_target_top2_recall_mean",
        "shared_target_attention_mass_recall_mean",
        "shared_target_stable_fraction_mean",
    ]
    write_csv(output_dir / "sharing_threshold_summary.csv", threshold_summary_rows, threshold_summary_fields)

    query_fields = ["query_token_index", "query_token_id"]
    if args.include_token_text:
        query_fields += ["query_token_piece", "query_token_text"]
    query_fields += [
        "cases",
        "top2_recall_mean",
        "top2_jaccard_mean",
        "attention_mass_recall_mean",
        "target_top2_attention_mass_mean",
        "donor_position_attention_mass_mean",
    ]
    write_csv(output_dir / "query_sharing_stability.csv", query_rows, query_fields)
    write_csv(output_dir / "query_layer_sharing_stability.csv", query_layer_rows, query_fields + ["layer"])

    query_group_fields = ["query_token_index", "query_token_id"]
    if args.include_token_text:
        query_group_fields += ["query_token_piece", "query_token_text"]
    query_group_fields += [
        "threshold",
        "selectors_without_sharing",
        "selectors_with_sharing",
        "selectors_saved",
        "selector_reduction_fraction",
        "all_head_cases",
        "all_head_top2_recall_mean",
        "all_head_attention_mass_recall_mean",
        "shared_target_cases",
        "shared_target_top2_recall_mean",
        "shared_target_attention_mass_recall_mean",
    ]
    write_csv(output_dir / "query_group_recall_by_threshold.csv", query_group_rows, query_group_fields)

    if args.write_query_layer_group_metrics:
        query_layer_group_fields = list(query_group_fields)
        query_layer_group_fields.insert(query_layer_group_fields.index("selectors_without_sharing"), "layer")
        write_csv(
            output_dir / "query_layer_group_recall_by_threshold.csv",
            query_layer_group_rows,
            query_layer_group_fields,
        )

    if args.write_query_layer_pair_metrics:
        pair_query_fields = ["query_token_index"] + pair_fields[:3] + [
            "donor_size",
            "target_size",
            "intersection",
            "union_size",
            "top2_recall",
            "top2_jaccard",
            "target_top2_attention_mass",
            "donor_position_attention_mass",
            "attention_mass_recall",
        ]
        write_csv(output_dir / "query_layer_head_pair_metrics.csv", collector.query_layer_pair_rows, pair_query_fields)

    summary = {
        "args": vars(args),
        "resolved": {
            "total_tokens_loaded": int(input_ids.numel()),
            "layer_count": layer_count,
            "head_count": head_count,
            "selected_layers": selected_layers,
            "selected_heads": selected_heads,
            "sampled_query_tokens_requested": query_samples,
            "sampled_query_tokens_observed": sorted(collector.observed_query_tokens),
            "topk_by_query": {str(key): value for key, value in sorted(collector.topk_by_query.items())},
            "seconds": seconds,
            "directed_pair_rows": len(pair_rows),
            "metric_definitions": {
                "top2_recall_mean": (
                    "For donor head d and target head t: mean |S_d(q) intersect S_t(q)| / |S_t(q)|. "
                    "This is the target head's true top-fraction position recall if it reuses the donor's positions."
                ),
                "attention_mass_recall_mean": (
                    "Target head full-attention mass on donor positions divided by target head full-attention mass "
                    "on its own true top-fraction positions."
                ),
                "shared_head_groups": (
                    "Greedy per-layer star groups. One representative head computes top-fraction positions; "
                    "member heads reuse the representative positions when rep_to_member top2 recall exceeds threshold."
                ),
                "query_group_recall_by_threshold": (
                    "Per query-token evaluation of the static groups learned from aggregate pair statistics. "
                    "Use this to detect query-token-dependent instability."
                ),
            },
        },
        "paths": {
            "head_sharing_donor_target": str(output_dir / "head_sharing_donor_target.csv"),
            "head_best_donors": str(output_dir / "head_best_donors.csv"),
            "shared_head_groups": str(output_dir / "shared_head_groups.csv"),
            "sharing_threshold_summary": str(output_dir / "sharing_threshold_summary.csv"),
            "query_sharing_stability": str(output_dir / "query_sharing_stability.csv"),
            "query_layer_sharing_stability": str(output_dir / "query_layer_sharing_stability.csv"),
            "query_group_recall_by_threshold": str(output_dir / "query_group_recall_by_threshold.csv"),
            "query_layer_group_recall_by_threshold": str(output_dir / "query_layer_group_recall_by_threshold.csv")
            if args.write_query_layer_group_metrics
            else None,
            "query_layer_head_pair_metrics": str(output_dir / "query_layer_head_pair_metrics.csv")
            if args.write_query_layer_pair_metrics
            else None,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "seconds": seconds,
                "observed_queries": len(collector.observed_query_tokens),
                "directed_pair_rows": len(pair_rows),
                "threshold_summary": threshold_summary_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
