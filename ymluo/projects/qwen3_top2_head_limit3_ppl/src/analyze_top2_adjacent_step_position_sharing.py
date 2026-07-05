from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from analyze_top2_head_position_sharing import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEXT_PATH,
    active_collector,
    build_query_samples,
    install_qwen3_attention_patch,
    parse_float_list,
    parse_index_spec,
    pick_input_device,
    prefill_cache,
    read_text_prefix,
    resolve_dtype,
    run_eval,
    safe_token_text,
    str2bool,
    threshold_field,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether the same attention head can reuse top-fraction historical token "
            "positions across adjacent decode steps."
        )
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/top2_adjacent_step_position_sharing")
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
    parser.add_argument("--write_query_layer_head_metrics", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


@dataclass
class AdjacentStepAccumulator:
    thresholds: list[float]
    cases: int = 0
    recall_sum: float = 0.0
    recall_sumsq: float = 0.0
    recall_min: float = float("inf")
    recall_max: float = float("-inf")
    old_history_recall_sum: float = 0.0
    attention_mass_recall_sum: float = 0.0
    ge_counts: dict[float, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ge_counts = {threshold: 0 for threshold in self.thresholds}

    def add(self, recall: float, old_history_recall: float, attention_mass_recall: float) -> None:
        self.cases += 1
        self.recall_sum += recall
        self.recall_sumsq += recall * recall
        self.recall_min = min(self.recall_min, recall)
        self.recall_max = max(self.recall_max, recall)
        self.old_history_recall_sum += old_history_recall
        self.attention_mass_recall_sum += attention_mass_recall
        for threshold in self.thresholds:
            if recall >= threshold:
                self.ge_counts[threshold] += 1

    def mean_recall(self) -> float:
        return self.recall_sum / self.cases if self.cases else 0.0

    def row(self, extra: dict[str, Any]) -> dict[str, Any]:
        mean = self.mean_recall()
        variance = self.recall_sumsq / self.cases - mean * mean if self.cases else 0.0
        row = {
            **extra,
            "cases": self.cases,
            "prev_to_current_top2_recall_mean": mean,
            "prev_to_current_top2_recall_std": math.sqrt(max(0.0, variance)),
            "prev_to_current_top2_recall_min": self.recall_min if self.cases else 0.0,
            "prev_to_current_top2_recall_max": self.recall_max if self.cases else 0.0,
            "old_history_top2_recall_mean": self.old_history_recall_sum / self.cases if self.cases else 0.0,
            "attention_mass_recall_mean": self.attention_mass_recall_sum / self.cases if self.cases else 0.0,
        }
        for threshold in self.thresholds:
            row[f"{threshold_field('top2_recall_ge', threshold)}_fraction"] = (
                self.ge_counts[threshold] / self.cases if self.cases else 0.0
            )
        return row


@dataclass
class QueryAdjacentAccumulator:
    thresholds: list[float]
    cases: int = 0
    recall_sum: float = 0.0
    old_history_recall_sum: float = 0.0
    attention_mass_recall_sum: float = 0.0
    ge_counts: dict[float, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ge_counts = {threshold: 0 for threshold in self.thresholds}

    def add(self, recall: float, old_history_recall: float, attention_mass_recall: float) -> None:
        self.cases += 1
        self.recall_sum += recall
        self.old_history_recall_sum += old_history_recall
        self.attention_mass_recall_sum += attention_mass_recall
        for threshold in self.thresholds:
            if recall >= threshold:
                self.ge_counts[threshold] += 1

    def row(self, extra: dict[str, Any]) -> dict[str, Any]:
        row = {
            **extra,
            "cases": self.cases,
            "prev_to_current_top2_recall_mean": self.recall_sum / self.cases if self.cases else 0.0,
            "old_history_top2_recall_mean": self.old_history_recall_sum / self.cases if self.cases else 0.0,
            "attention_mass_recall_mean": self.attention_mass_recall_sum / self.cases if self.cases else 0.0,
        }
        for threshold in self.thresholds:
            row[f"{threshold_field('top2_recall_ge', threshold)}_fraction"] = (
                self.ge_counts[threshold] / self.cases if self.cases else 0.0
            )
        return row


@dataclass
class QueryGroupAccumulator:
    cases: int = 0
    selector_starts: int = 0
    shared_steps: int = 0
    recall_sum: float = 0.0
    old_history_recall_sum: float = 0.0
    attention_mass_recall_sum: float = 0.0

    def add_selector_start(self) -> None:
        self.cases += 1
        self.selector_starts += 1

    def add_shared(self, recall: float, old_history_recall: float, attention_mass_recall: float) -> None:
        self.cases += 1
        self.shared_steps += 1
        self.recall_sum += recall
        self.old_history_recall_sum += old_history_recall
        self.attention_mass_recall_sum += attention_mass_recall

    def row(self, extra: dict[str, Any]) -> dict[str, Any]:
        selectors_saved = self.shared_steps
        return {
            **extra,
            "selector_cases_without_sharing": self.cases,
            "selectors_with_sharing": self.selector_starts,
            "selectors_saved": selectors_saved,
            "selector_reduction_fraction": selectors_saved / self.cases if self.cases else 0.0,
            "shared_step_cases": self.shared_steps,
            "shared_step_top2_recall_mean": self.recall_sum / self.shared_steps if self.shared_steps else 1.0,
            "shared_step_old_history_recall_mean": self.old_history_recall_sum / self.shared_steps
            if self.shared_steps
            else 1.0,
            "shared_step_attention_mass_recall_mean": self.attention_mass_recall_sum / self.shared_steps
            if self.shared_steps
            else 1.0,
        }


@dataclass
class StepGroupState:
    threshold: float
    layer: int
    head: int
    group_id: int
    representative_query_token: int
    start_query_token: int
    last_query_token: int
    representative_mask: torch.Tensor
    step_count: int = 1
    shared_step_count: int = 0
    recall_sum: float = 0.0
    recall_min: float = float("inf")
    old_history_recall_sum: float = 0.0
    old_history_recall_min: float = float("inf")
    attention_mass_recall_sum: float = 0.0
    attention_mass_recall_min: float = float("inf")

    def add_shared(self, query_token: int, recall: float, old_history_recall: float, mass_recall: float) -> None:
        self.last_query_token = query_token
        self.step_count += 1
        self.shared_step_count += 1
        self.recall_sum += recall
        self.recall_min = min(self.recall_min, recall)
        self.old_history_recall_sum += old_history_recall
        self.old_history_recall_min = min(self.old_history_recall_min, old_history_recall)
        self.attention_mass_recall_sum += mass_recall
        self.attention_mass_recall_min = min(self.attention_mass_recall_min, mass_recall)

    def row(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "layer": self.layer,
            "head": self.head,
            "group_id": self.group_id,
            "representative_query_token": self.representative_query_token,
            "start_query_token": self.start_query_token,
            "end_query_token": self.last_query_token,
            "step_count": self.step_count,
            "selectors_saved": max(0, self.step_count - 1),
            "shared_step_count": self.shared_step_count,
            "shared_step_top2_recall_mean": self.recall_sum / self.shared_step_count
            if self.shared_step_count
            else 1.0,
            "shared_step_top2_recall_min": self.recall_min if self.shared_step_count else 1.0,
            "shared_step_old_history_recall_mean": self.old_history_recall_sum / self.shared_step_count
            if self.shared_step_count
            else 1.0,
            "shared_step_old_history_recall_min": self.old_history_recall_min if self.shared_step_count else 1.0,
            "shared_step_attention_mass_recall_mean": self.attention_mass_recall_sum / self.shared_step_count
            if self.shared_step_count
            else 1.0,
            "shared_step_attention_mass_recall_min": self.attention_mass_recall_min
            if self.shared_step_count
            else 1.0,
        }


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


class AdjacentStepPositionSharingCollector:
    def __init__(
        self,
        selected_layers: list[int],
        selected_heads: list[int],
        query_tokens: set[int],
        top_fraction: float,
        remote_only: bool,
        exclude_sink_tokens: int,
        exclude_recent_tokens: int,
        group_thresholds: list[float],
        stability_thresholds: list[float],
        write_query_layer_head_metrics: bool,
    ) -> None:
        self.selected_layers = selected_layers
        self.selected_layers_set = set(selected_layers)
        self.selected_heads = selected_heads
        self.query_tokens = query_tokens
        self.top_fraction = top_fraction
        self.remote_only = remote_only
        self.exclude_sink_tokens = exclude_sink_tokens
        self.exclude_recent_tokens = exclude_recent_tokens
        self.group_thresholds = group_thresholds
        self.stability_thresholds = sorted(set(group_thresholds + stability_thresholds))
        self.write_query_layer_head_metrics = write_query_layer_head_metrics

        self.previous_by_layer: dict[int, dict[str, Any]] = {}
        self.adjacent_stats: dict[tuple[int, int], AdjacentStepAccumulator] = {}
        self.query_adjacent_stats: dict[int, QueryAdjacentAccumulator] = {}
        self.query_group_stats: dict[tuple[int, float], QueryGroupAccumulator] = {}
        self.group_states: dict[tuple[float, int, int], StepGroupState] = {}
        self.next_group_id: dict[tuple[float, int, int], int] = defaultdict(int)
        self.group_rows: list[dict[str, Any]] = []
        self.query_layer_head_rows: list[dict[str, Any]] = []
        self.observed_query_tokens: set[int] = set()
        self.topk_by_query: dict[int, int] = {}

    def _adjacent_accumulator(self, layer: int, head: int) -> AdjacentStepAccumulator:
        key = (layer, head)
        if key not in self.adjacent_stats:
            self.adjacent_stats[key] = AdjacentStepAccumulator(self.stability_thresholds)
        return self.adjacent_stats[key]

    def _query_adjacent_accumulator(self, query_token: int) -> QueryAdjacentAccumulator:
        if query_token not in self.query_adjacent_stats:
            self.query_adjacent_stats[query_token] = QueryAdjacentAccumulator(self.stability_thresholds)
        return self.query_adjacent_stats[query_token]

    def _query_group_accumulator(self, query_token: int, threshold: float) -> QueryGroupAccumulator:
        key = (query_token, threshold)
        if key not in self.query_group_stats:
            self.query_group_stats[key] = QueryGroupAccumulator()
        return self.query_group_stats[key]

    def _close_group(self, key: tuple[float, int, int]) -> None:
        state = self.group_states.pop(key, None)
        if state is not None:
            self.group_rows.append(state.row())

    def _start_group(
        self,
        threshold: float,
        layer: int,
        head: int,
        query_token: int,
        representative_mask: torch.Tensor,
    ) -> None:
        key = (threshold, layer, head)
        group_id = self.next_group_id[key]
        self.next_group_id[key] += 1
        self.group_states[key] = StepGroupState(
            threshold=threshold,
            layer=layer,
            head=head,
            group_id=group_id,
            representative_query_token=query_token,
            start_query_token=query_token,
            last_query_token=query_token,
            representative_mask=representative_mask.clone(),
        )

    @staticmethod
    def _reuse_metrics(
        donor_mask: torch.Tensor,
        target_mask: torch.Tensor,
        target_attention: torch.Tensor,
        target_top2_mass: float,
    ) -> tuple[float, float, float]:
        shared_len = min(int(donor_mask.numel()), int(target_mask.numel()))
        if shared_len <= 0:
            return 0.0, 0.0, 0.0
        donor_aligned = donor_mask[:shared_len]
        target_aligned = target_mask[:shared_len]
        target_size = float(target_mask.sum().item())
        old_target_size = float(target_aligned.sum().item())
        intersection = float((donor_aligned & target_aligned).sum().item())
        recall = intersection / max(1.0, target_size)
        old_history_recall = intersection / max(1.0, old_target_size)
        mass = float((donor_aligned.float() * target_attention[:shared_len].float()).sum().item())
        mass_recall = mass / max(1.0e-30, target_top2_mass)
        return recall, old_history_recall, mass_recall

    def _update_adjacent_stats(
        self,
        layer: int,
        query_token: int,
        selected: torch.Tensor,
        attention_weights: torch.Tensor,
        target_top2_mass: torch.Tensor,
    ) -> None:
        previous = self.previous_by_layer.get(layer)
        if previous is None or int(previous["query_token"]) != query_token - 1:
            return
        previous_selected = previous["selected"]
        query_acc = self._query_adjacent_accumulator(query_token)
        for head_idx, head in enumerate(self.selected_heads):
            if float(selected[head_idx].sum().item()) <= 0.0:
                continue
            recall, old_recall, mass_recall = self._reuse_metrics(
                previous_selected[head_idx],
                selected[head_idx],
                attention_weights[head_idx],
                float(target_top2_mass[head_idx].item()),
            )
            self._adjacent_accumulator(layer, head).add(recall, old_recall, mass_recall)
            query_acc.add(recall, old_recall, mass_recall)
            if self.write_query_layer_head_metrics:
                self.query_layer_head_rows.append(
                    {
                        "query_token_index": query_token,
                        "layer": layer,
                        "head": head,
                        "prev_query_token_index": query_token - 1,
                        "prev_to_current_top2_recall": recall,
                        "old_history_top2_recall": old_recall,
                        "attention_mass_recall": mass_recall,
                    }
                )

    def _update_group_states(
        self,
        layer: int,
        query_token: int,
        selected: torch.Tensor,
        attention_weights: torch.Tensor,
        target_top2_mass: torch.Tensor,
    ) -> None:
        for threshold in self.group_thresholds:
            for head_idx, head in enumerate(self.selected_heads):
                if float(selected[head_idx].sum().item()) <= 0.0:
                    continue
                key = (threshold, layer, head)
                state = self.group_states.get(key)
                query_group_acc = self._query_group_accumulator(query_token, threshold)
                if state is None or state.last_query_token != query_token - 1:
                    self._close_group(key)
                    self._start_group(threshold, layer, head, query_token, selected[head_idx])
                    query_group_acc.add_selector_start()
                    continue

                recall, old_recall, mass_recall = self._reuse_metrics(
                    state.representative_mask,
                    selected[head_idx],
                    attention_weights[head_idx],
                    float(target_top2_mass[head_idx].item()),
                )
                if recall >= threshold:
                    state.add_shared(query_token, recall, old_recall, mass_recall)
                    query_group_acc.add_shared(recall, old_recall, mass_recall)
                else:
                    self._close_group(key)
                    self._start_group(threshold, layer, head, query_token, selected[head_idx])
                    query_group_acc.add_selector_start()

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
        if int(selected.sum().item()) == 0:
            return

        attention_weights = F.softmax(scores[0, head_index, query_index, :valid_count].detach().float(), dim=-1)[
            :, :history_count
        ]
        target_top2_mass = (selected.float() * attention_weights).sum(dim=1)

        selected_cpu = selected.detach().cpu()
        attention_cpu = attention_weights.detach().cpu()
        target_mass_cpu = target_top2_mass.detach().cpu()

        self._update_adjacent_stats(layer, query_token, selected_cpu, attention_cpu, target_mass_cpu)
        self._update_group_states(layer, query_token, selected_cpu, attention_cpu, target_mass_cpu)

        self.previous_by_layer[layer] = {"query_token": query_token, "selected": selected_cpu}
        self.observed_query_tokens.add(query_token)
        self.topk_by_query[query_token] = top_count

    def finalize(self) -> None:
        for key in list(self.group_states):
            self._close_group(key)

    def layer_head_rows(self) -> list[dict[str, Any]]:
        return [
            stats.row({"layer": layer, "head": head})
            for (layer, head), stats in sorted(self.adjacent_stats.items())
        ]

    def query_adjacent_rows(self, tokenizer: Any, token_ids: list[int], include_token_text: bool) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for query_token, stats in sorted(self.query_adjacent_stats.items()):
            rows.append(stats.row(token_prefix(tokenizer, token_ids, query_token, include_token_text)))
        return rows

    def query_group_rows(self, tokenizer: Any, token_ids: list[int], include_token_text: bool) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (query_token, threshold), stats in sorted(self.query_group_stats.items()):
            row = stats.row(token_prefix(tokenizer, token_ids, query_token, include_token_text))
            row["threshold"] = threshold
            rows.append(row)
        return rows

    def threshold_summary_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        groups_by_threshold: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in self.group_rows:
            groups_by_threshold[float(row["threshold"])].append(row)
        for threshold in self.group_thresholds:
            groups = groups_by_threshold.get(threshold, [])
            selector_cases = sum(int(row["step_count"]) for row in groups)
            group_count = len(groups)
            selectors_saved = sum(int(row["selectors_saved"]) for row in groups)
            shared_step_count = sum(int(row["shared_step_count"]) for row in groups)
            recall_sum = sum(float(row["shared_step_top2_recall_mean"]) * int(row["shared_step_count"]) for row in groups)
            old_recall_sum = sum(
                float(row["shared_step_old_history_recall_mean"]) * int(row["shared_step_count"]) for row in groups
            )
            mass_sum = sum(
                float(row["shared_step_attention_mass_recall_mean"]) * int(row["shared_step_count"]) for row in groups
            )
            rows.append(
                {
                    "threshold": threshold,
                    "selected_layers": len(self.selected_layers),
                    "heads_per_layer": len(self.selected_heads),
                    "selector_cases_without_sharing": selector_cases,
                    "selectors_with_sharing": group_count,
                    "selectors_saved": selectors_saved,
                    "selector_reduction_fraction": selectors_saved / selector_cases if selector_cases else 0.0,
                    "group_count": group_count,
                    "groups_size_gt1": sum(1 for row in groups if int(row["step_count"]) > 1),
                    "mean_group_steps": selector_cases / group_count if group_count else 0.0,
                    "max_group_steps": max((int(row["step_count"]) for row in groups), default=0),
                    "shared_step_count": shared_step_count,
                    "shared_step_top2_recall_mean": recall_sum / shared_step_count if shared_step_count else 1.0,
                    "shared_step_old_history_recall_mean": old_recall_sum / shared_step_count
                    if shared_step_count
                    else 1.0,
                    "shared_step_attention_mass_recall_mean": mass_sum / shared_step_count
                    if shared_step_count
                    else 1.0,
                }
            )
        return rows


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

    collector = AdjacentStepPositionSharingCollector(
        selected_layers=selected_layers,
        selected_heads=selected_heads,
        query_tokens=set(query_samples),
        top_fraction=args.top_fraction,
        remote_only=args.remote_only,
        exclude_sink_tokens=args.exclude_sink_tokens,
        exclude_recent_tokens=args.exclude_recent_tokens,
        group_thresholds=group_thresholds,
        stability_thresholds=stability_thresholds,
        write_query_layer_head_metrics=args.write_query_layer_head_metrics,
    )

    started = time.perf_counter()
    past = prefill_cache(model, input_ids, args.prefill_tokens, args.chunk_size, input_device)
    run_eval(model, input_ids, past, args.prefill_tokens, args.eval_tokens, args.chunk_size, input_device, collector)
    collector.finalize()
    seconds = time.perf_counter() - started

    threshold_summary_rows = collector.threshold_summary_rows()
    layer_head_rows = collector.layer_head_rows()
    query_adjacent_rows = collector.query_adjacent_rows(tokenizer, token_ids, args.include_token_text)
    query_group_rows = collector.query_group_rows(tokenizer, token_ids, args.include_token_text)

    layer_head_fields = [
        "layer",
        "head",
        "cases",
        "prev_to_current_top2_recall_mean",
        "prev_to_current_top2_recall_std",
        "prev_to_current_top2_recall_min",
        "prev_to_current_top2_recall_max",
        "old_history_top2_recall_mean",
        "attention_mass_recall_mean",
    ]
    layer_head_fields += [
        f"{threshold_field('top2_recall_ge', threshold)}_fraction" for threshold in stability_thresholds
    ]
    write_csv(output_dir / "layer_head_adjacent_step_stats.csv", layer_head_rows, layer_head_fields)

    query_adjacent_fields = ["query_token_index", "query_token_id"]
    if args.include_token_text:
        query_adjacent_fields += ["query_token_piece", "query_token_text"]
    query_adjacent_fields += [
        "cases",
        "prev_to_current_top2_recall_mean",
        "old_history_top2_recall_mean",
        "attention_mass_recall_mean",
    ]
    query_adjacent_fields += [
        f"{threshold_field('top2_recall_ge', threshold)}_fraction" for threshold in stability_thresholds
    ]
    write_csv(output_dir / "query_adjacent_step_stability.csv", query_adjacent_rows, query_adjacent_fields)

    group_fields = [
        "threshold",
        "layer",
        "head",
        "group_id",
        "representative_query_token",
        "start_query_token",
        "end_query_token",
        "step_count",
        "selectors_saved",
        "shared_step_count",
        "shared_step_top2_recall_mean",
        "shared_step_top2_recall_min",
        "shared_step_old_history_recall_mean",
        "shared_step_old_history_recall_min",
        "shared_step_attention_mass_recall_mean",
        "shared_step_attention_mass_recall_min",
    ]
    write_csv(output_dir / "shared_step_groups.csv", collector.group_rows, group_fields)

    threshold_summary_fields = [
        "threshold",
        "selected_layers",
        "heads_per_layer",
        "selector_cases_without_sharing",
        "selectors_with_sharing",
        "selectors_saved",
        "selector_reduction_fraction",
        "group_count",
        "groups_size_gt1",
        "mean_group_steps",
        "max_group_steps",
        "shared_step_count",
        "shared_step_top2_recall_mean",
        "shared_step_old_history_recall_mean",
        "shared_step_attention_mass_recall_mean",
    ]
    write_csv(output_dir / "step_sharing_threshold_summary.csv", threshold_summary_rows, threshold_summary_fields)

    query_group_fields = ["query_token_index", "query_token_id"]
    if args.include_token_text:
        query_group_fields += ["query_token_piece", "query_token_text"]
    query_group_fields += [
        "threshold",
        "selector_cases_without_sharing",
        "selectors_with_sharing",
        "selectors_saved",
        "selector_reduction_fraction",
        "shared_step_cases",
        "shared_step_top2_recall_mean",
        "shared_step_old_history_recall_mean",
        "shared_step_attention_mass_recall_mean",
    ]
    write_csv(output_dir / "query_group_recall_by_threshold.csv", query_group_rows, query_group_fields)

    if args.write_query_layer_head_metrics:
        per_case_fields = [
            "query_token_index",
            "layer",
            "head",
            "prev_query_token_index",
            "prev_to_current_top2_recall",
            "old_history_top2_recall",
            "attention_mass_recall",
        ]
        write_csv(output_dir / "query_layer_head_adjacent_metrics.csv", collector.query_layer_head_rows, per_case_fields)

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
            "metric_definitions": {
                "prev_to_current_top2_recall": (
                    "For the same layer/head, |S(q-1) intersect S(q)| / |S(q)|. "
                    "This is the current step's true top-fraction recall if it reuses the previous step's positions."
                ),
                "old_history_top2_recall": (
                    "|S(q-1) intersect S(q)| divided by the current top-fraction positions that are old enough "
                    "to have existed at q-1. This separates new-token effects from old-history drift."
                ),
                "shared_step_groups": (
                    "Contiguous per-layer/head query runs. One representative query computes top-fraction positions; "
                    "later steps reuse the representative positions while representative-to-current recall stays "
                    "above the group threshold."
                ),
            },
        },
        "paths": {
            "layer_head_adjacent_step_stats": str(output_dir / "layer_head_adjacent_step_stats.csv"),
            "query_adjacent_step_stability": str(output_dir / "query_adjacent_step_stability.csv"),
            "shared_step_groups": str(output_dir / "shared_step_groups.csv"),
            "step_sharing_threshold_summary": str(output_dir / "step_sharing_threshold_summary.csv"),
            "query_group_recall_by_threshold": str(output_dir / "query_group_recall_by_threshold.csv"),
            "query_layer_head_adjacent_metrics": str(output_dir / "query_layer_head_adjacent_metrics.csv")
            if args.write_query_layer_head_metrics
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
                "threshold_summary": threshold_summary_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
