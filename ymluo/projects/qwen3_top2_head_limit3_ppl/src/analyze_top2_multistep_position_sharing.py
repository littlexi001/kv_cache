from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict, deque
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
            "Measure how far one top-fraction historical token selection can be reused "
            "for the same layer/head across multiple decode steps."
        )
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/top2_multistep_position_sharing")
    parser.add_argument("--total_tokens", type=int, default=4608)
    parser.add_argument("--prefill_tokens", type=int, default=4096)
    parser.add_argument("--eval_tokens", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument(
        "--remote_only",
        type=str2bool,
        default=True,
        help="If true, select true top-fraction over all history, then evaluate only non-sink/non-recent positions.",
    )
    parser.add_argument("--exclude_sink_tokens", type=int, default=64)
    parser.add_argument("--exclude_recent_tokens", type=int, default=512)
    parser.add_argument("--max_query_samples", type=int, default=0, help="Use <=0 to analyze all eval queries.")
    parser.add_argument("--query_stride", type=int, default=0)
    parser.add_argument("--lags", default="1,2,4,8,16,32,64")
    parser.add_argument("--horizons", default="2,3,4,8,16,32,64")
    parser.add_argument("--thresholds", default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--include_token_text", type=str2bool, default=False)
    parser.add_argument("--write_block_rows", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def parse_int_list(value: str, name: str) -> list[int]:
    numbers = sorted({int(part) for part in value.split(",") if part.strip()})
    if not numbers:
        raise ValueError(f"--{name} must contain at least one integer.")
    invalid = [number for number in numbers if number <= 0]
    if invalid:
        raise ValueError(f"--{name} values must be positive, got {invalid}.")
    return numbers


@dataclass
class MetricAccumulator:
    thresholds: list[float]
    cases: int = 0
    recall_sum: float = 0.0
    recall_sumsq: float = 0.0
    recall_min: float = float("inf")
    recall_max: float = float("-inf")
    old_history_recall_sum: float = 0.0
    attention_mass_recall_sum: float = 0.0
    attention_mass_recall_sumsq: float = 0.0
    attention_mass_recall_min: float = float("inf")
    attention_mass_recall_max: float = float("-inf")
    ge_counts: dict[float, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ge_counts = {threshold: 0 for threshold in self.thresholds}

    def add(self, recall: float, old_history_recall: float, attention_mass_recall: float) -> None:
        if not math.isfinite(recall) or not math.isfinite(attention_mass_recall):
            return
        self.cases += 1
        self.recall_sum += recall
        self.recall_sumsq += recall * recall
        self.recall_min = min(self.recall_min, recall)
        self.recall_max = max(self.recall_max, recall)
        self.old_history_recall_sum += old_history_recall
        self.attention_mass_recall_sum += attention_mass_recall
        self.attention_mass_recall_sumsq += attention_mass_recall * attention_mass_recall
        self.attention_mass_recall_min = min(self.attention_mass_recall_min, attention_mass_recall)
        self.attention_mass_recall_max = max(self.attention_mass_recall_max, attention_mass_recall)
        for threshold in self.thresholds:
            if recall >= threshold:
                self.ge_counts[threshold] += 1

    def row(self, extra: dict[str, Any]) -> dict[str, Any]:
        recall_mean = self.recall_sum / self.cases if self.cases else 0.0
        recall_var = self.recall_sumsq / self.cases - recall_mean * recall_mean if self.cases else 0.0
        mass_mean = self.attention_mass_recall_sum / self.cases if self.cases else 0.0
        mass_var = self.attention_mass_recall_sumsq / self.cases - mass_mean * mass_mean if self.cases else 0.0
        row = {
            **extra,
            "metric_cases": self.cases,
            "top2_recall_mean": recall_mean,
            "top2_recall_std": math.sqrt(max(0.0, recall_var)),
            "top2_recall_min": self.recall_min if self.cases else 0.0,
            "top2_recall_max": self.recall_max if self.cases else 0.0,
            "old_history_top2_recall_mean": self.old_history_recall_sum / self.cases if self.cases else 0.0,
            "attention_mass_recall_mean": mass_mean,
            "attention_mass_recall_std": math.sqrt(max(0.0, mass_var)),
            "attention_mass_recall_min": self.attention_mass_recall_min if self.cases else 0.0,
            "attention_mass_recall_max": self.attention_mass_recall_max if self.cases else 0.0,
        }
        for threshold in self.thresholds:
            row[f"{threshold_field('top2_recall_ge', threshold)}_fraction"] = (
                self.ge_counts[threshold] / self.cases if self.cases else 0.0
            )
        return row


@dataclass
class HorizonAccumulator:
    thresholds: list[float]
    selector_cases_without_sharing: int = 0
    selector_start_cases: int = 0
    selector_shared_cases: int = 0
    metrics: MetricAccumulator | None = None

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = MetricAccumulator(self.thresholds)

    def add_start(self, count: int) -> None:
        self.selector_cases_without_sharing += count
        self.selector_start_cases += count

    def add_shared(self, count: int) -> None:
        self.selector_cases_without_sharing += count
        self.selector_shared_cases += count

    def add_metric(self, recall: float, old_history_recall: float, attention_mass_recall: float) -> None:
        if self.metrics is None:
            self.metrics = MetricAccumulator(self.thresholds)
        self.metrics.add(recall, old_history_recall, attention_mass_recall)

    def row(self, extra: dict[str, Any]) -> dict[str, Any]:
        row = {
            **extra,
            "selector_cases_without_sharing": self.selector_cases_without_sharing,
            "selectors_with_sharing": self.selector_start_cases,
            "selectors_saved": self.selector_shared_cases,
            "selector_reduction_fraction": self.selector_shared_cases / self.selector_cases_without_sharing
            if self.selector_cases_without_sharing
            else 0.0,
        }
        if self.metrics is not None:
            row.update(self.metrics.row({}))
        return row


@dataclass
class SelectionRecord:
    query_token: int
    selected: torch.Tensor
    attention_weights: torch.Tensor
    target_top2_mass: torch.Tensor


@dataclass
class BlockState:
    horizon: int
    layer: int
    head: int
    block_id: int
    anchor_query_token: int
    start_query_token: int
    last_query_token: int
    shared_steps: int = 0
    recall_sum: float = 0.0
    recall_min: float = float("inf")
    mass_sum: float = 0.0
    mass_min: float = float("inf")

    def add_shared(self, query_token: int, recall: float, mass_recall: float) -> None:
        self.last_query_token = query_token
        self.shared_steps += 1
        self.recall_sum += recall
        self.recall_min = min(self.recall_min, recall)
        self.mass_sum += mass_recall
        self.mass_min = min(self.mass_min, mass_recall)

    def row(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "layer": self.layer,
            "head": self.head,
            "block_id": self.block_id,
            "anchor_query_token": self.anchor_query_token,
            "start_query_token": self.start_query_token,
            "end_query_token": self.last_query_token,
            "block_step_count": self.last_query_token - self.start_query_token + 1,
            "shared_steps": self.shared_steps,
            "shared_top2_recall_mean": self.recall_sum / self.shared_steps if self.shared_steps else 1.0,
            "shared_top2_recall_min": self.recall_min if self.shared_steps else 1.0,
            "shared_attention_mass_recall_mean": self.mass_sum / self.shared_steps if self.shared_steps else 1.0,
            "shared_attention_mass_recall_min": self.mass_min if self.shared_steps else 1.0,
        }


def token_prefix(tokenizer: Any, token_ids: list[int], query_token: int, include_token_text: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "query_token_index": query_token,
        "query_token_id": int(token_ids[query_token]),
    }
    if include_token_text:
        piece, text = safe_token_text(tokenizer, token_ids[query_token])
        row["query_token_piece"] = piece
        row["query_token_text"] = text
    return row


class MultiStepPositionSharingCollector:
    def __init__(
        self,
        selected_layers: list[int],
        selected_heads: list[int],
        query_tokens: set[int],
        first_query_token: int,
        top_fraction: float,
        remote_only: bool,
        exclude_sink_tokens: int,
        exclude_recent_tokens: int,
        lags: list[int],
        horizons: list[int],
        thresholds: list[float],
        write_block_rows: bool,
    ) -> None:
        self.selected_layers = selected_layers
        self.selected_layers_set = set(selected_layers)
        self.selected_heads = selected_heads
        self.query_tokens = query_tokens
        self.first_query_token = first_query_token
        self.top_fraction = top_fraction
        self.remote_only = remote_only
        self.exclude_sink_tokens = exclude_sink_tokens
        self.exclude_recent_tokens = exclude_recent_tokens
        self.lags = lags
        self.lag_set = set(lags)
        self.max_lag = max(lags)
        self.horizons = horizons
        self.thresholds = thresholds
        self.write_block_rows = write_block_rows

        self.history_by_layer: dict[int, deque[SelectionRecord]] = defaultdict(deque)
        self.offset_stats: dict[int, MetricAccumulator] = {
            lag: MetricAccumulator(thresholds) for lag in lags
        }
        self.offset_layer_head_stats: dict[tuple[int, int, int], MetricAccumulator] = {}
        self.horizon_stats: dict[int, HorizonAccumulator] = {
            horizon: HorizonAccumulator(thresholds) for horizon in horizons
        }
        self.horizon_layer_head_stats: dict[tuple[int, int, int], HorizonAccumulator] = {}
        self.horizon_anchors: dict[tuple[int, int], SelectionRecord] = {}
        self.block_states: dict[tuple[int, int, int], BlockState] = {}
        self.next_block_id: dict[tuple[int, int, int], int] = defaultdict(int)
        self.block_rows: list[dict[str, Any]] = []
        self.observed_query_tokens: set[int] = set()
        self.topk_by_query: dict[int, int] = {}

    def _offset_layer_head_accumulator(self, lag: int, layer: int, head: int) -> MetricAccumulator:
        key = (lag, layer, head)
        if key not in self.offset_layer_head_stats:
            self.offset_layer_head_stats[key] = MetricAccumulator(self.thresholds)
        return self.offset_layer_head_stats[key]

    def _horizon_layer_head_accumulator(self, horizon: int, layer: int, head: int) -> HorizonAccumulator:
        key = (horizon, layer, head)
        if key not in self.horizon_layer_head_stats:
            self.horizon_layer_head_stats[key] = HorizonAccumulator(self.thresholds)
        return self.horizon_layer_head_stats[key]

    def _close_block(self, key: tuple[int, int, int]) -> None:
        state = self.block_states.pop(key, None)
        if state is not None and self.write_block_rows:
            self.block_rows.append(state.row())

    def _start_block(self, horizon: int, layer: int, head: int, query_token: int) -> None:
        key = (horizon, layer, head)
        self._close_block(key)
        block_id = self.next_block_id[key]
        self.next_block_id[key] += 1
        self.block_states[key] = BlockState(
            horizon=horizon,
            layer=layer,
            head=head,
            block_id=block_id,
            anchor_query_token=query_token,
            start_query_token=query_token,
            last_query_token=query_token,
        )

    @staticmethod
    def _reuse_metrics(
        donor_mask: torch.Tensor,
        target_mask: torch.Tensor,
        target_attention: torch.Tensor,
        target_top2_mass: float,
    ) -> tuple[float, float, float] | None:
        if float(target_mask.sum().item()) <= 0.0:
            return None
        shared_len = min(int(donor_mask.numel()), int(target_mask.numel()))
        if shared_len <= 0:
            return None
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

    def _make_selection(self, scores: torch.Tensor, query_token: int, query_index: int) -> SelectionRecord | None:
        finite = torch.isfinite(scores[:, :, query_index, :])
        valid_count = min(int(finite[0, 0].sum().item()), query_token + 1)
        if valid_count <= 1:
            return None

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
        attention_weights = F.softmax(scores[0, head_index, query_index, :valid_count].detach().float(), dim=-1)[
            :, :history_count
        ]
        target_top2_mass = (selected.float() * attention_weights).sum(dim=1)
        self.topk_by_query[query_token] = top_count
        return SelectionRecord(
            query_token=query_token,
            selected=selected.detach().cpu(),
            attention_weights=attention_weights.detach().cpu(),
            target_top2_mass=target_top2_mass.detach().cpu(),
        )

    def _update_offset_stats(self, layer: int, current: SelectionRecord) -> None:
        history = self.history_by_layer[layer]
        by_query = {record.query_token: record for record in history}
        for lag in self.lags:
            donor = by_query.get(current.query_token - lag)
            if donor is None:
                continue
            for head_idx, head in enumerate(self.selected_heads):
                metrics = self._reuse_metrics(
                    donor.selected[head_idx],
                    current.selected[head_idx],
                    current.attention_weights[head_idx],
                    float(current.target_top2_mass[head_idx].item()),
                )
                if metrics is None:
                    continue
                recall, old_recall, mass_recall = metrics
                self.offset_stats[lag].add(recall, old_recall, mass_recall)
                self._offset_layer_head_accumulator(lag, layer, head).add(recall, old_recall, mass_recall)

    def _update_horizon_stats(self, layer: int, current: SelectionRecord) -> None:
        head_count = len(self.selected_heads)
        relative_query = current.query_token - self.first_query_token
        for horizon in self.horizons:
            horizon_acc = self.horizon_stats[horizon]
            anchor_key = (horizon, layer)
            anchor = self.horizon_anchors.get(anchor_key)
            is_anchor_step = relative_query % horizon == 0 or anchor is None
            if is_anchor_step:
                horizon_acc.add_start(head_count)
                for head in self.selected_heads:
                    self._horizon_layer_head_accumulator(horizon, layer, head).add_start(1)
                    self._start_block(horizon, layer, head, current.query_token)
                self.horizon_anchors[anchor_key] = current
                continue

            horizon_acc.add_shared(head_count)
            for head_idx, head in enumerate(self.selected_heads):
                layer_head_acc = self._horizon_layer_head_accumulator(horizon, layer, head)
                layer_head_acc.add_shared(1)
                metrics = self._reuse_metrics(
                    anchor.selected[head_idx],
                    current.selected[head_idx],
                    current.attention_weights[head_idx],
                    float(current.target_top2_mass[head_idx].item()),
                )
                if metrics is None:
                    continue
                recall, old_recall, mass_recall = metrics
                horizon_acc.add_metric(recall, old_recall, mass_recall)
                layer_head_acc.add_metric(recall, old_recall, mass_recall)
                block_key = (horizon, layer, head)
                state = self.block_states.get(block_key)
                if state is not None:
                    state.add_shared(current.query_token, recall, mass_recall)

    def observe(self, layer: int, query_token: int, scores: torch.Tensor, query_index: int) -> None:
        if layer not in self.selected_layers_set or query_token not in self.query_tokens:
            return
        current = self._make_selection(scores, query_token, query_index)
        if current is None:
            return
        self._update_offset_stats(layer, current)
        self._update_horizon_stats(layer, current)

        history = self.history_by_layer[layer]
        history.append(current)
        while history and history[0].query_token < query_token - self.max_lag:
            history.popleft()
        self.observed_query_tokens.add(query_token)

    def finalize(self) -> None:
        for key in list(self.block_states):
            self._close_block(key)

    def offset_rows(self) -> list[dict[str, Any]]:
        return [self.offset_stats[lag].row({"lag": lag}) for lag in self.lags]

    def offset_layer_head_rows(self) -> list[dict[str, Any]]:
        return [
            stats.row({"lag": lag, "layer": layer, "head": head})
            for (lag, layer, head), stats in sorted(self.offset_layer_head_stats.items())
        ]

    def horizon_rows(self) -> list[dict[str, Any]]:
        return [self.horizon_stats[horizon].row({"horizon": horizon}) for horizon in self.horizons]

    def horizon_layer_head_rows(self) -> list[dict[str, Any]]:
        return [
            stats.row({"horizon": horizon, "layer": layer, "head": head})
            for (horizon, layer, head), stats in sorted(self.horizon_layer_head_stats.items())
        ]

    def block_threshold_summary_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        rows_by_horizon: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.block_rows:
            if int(row["shared_steps"]) > 0:
                rows_by_horizon[int(row["horizon"])].append(row)
        for horizon in self.horizons:
            blocks = rows_by_horizon.get(horizon, [])
            for threshold in self.thresholds:
                passed = [
                    row
                    for row in blocks
                    if float(row["shared_top2_recall_min"]) >= threshold
                ]
                rows.append(
                    {
                        "horizon": horizon,
                        "threshold": threshold,
                        "blocks_with_shared_steps": len(blocks),
                        "blocks_min_recall_ge_threshold": len(passed),
                        "block_pass_fraction": len(passed) / len(blocks) if blocks else 0.0,
                        "mean_shared_steps_if_pass": sum(int(row["shared_steps"]) for row in passed) / len(passed)
                        if passed
                        else 0.0,
                        "mean_min_recall_all_blocks": sum(float(row["shared_top2_recall_min"]) for row in blocks)
                        / len(blocks)
                        if blocks
                        else 0.0,
                        "mean_min_mass_recall_all_blocks": sum(
                            float(row["shared_attention_mass_recall_min"]) for row in blocks
                        )
                        / len(blocks)
                        if blocks
                        else 0.0,
                    }
                )
        return rows


def metric_fields(thresholds: list[float]) -> list[str]:
    fields = [
        "metric_cases",
        "top2_recall_mean",
        "top2_recall_std",
        "top2_recall_min",
        "top2_recall_max",
        "old_history_top2_recall_mean",
        "attention_mass_recall_mean",
        "attention_mass_recall_std",
        "attention_mass_recall_min",
        "attention_mass_recall_max",
    ]
    fields += [f"{threshold_field('top2_recall_ge', threshold)}_fraction" for threshold in thresholds]
    return fields


def main() -> None:
    args = parse_args()
    if not (0.0 < args.top_fraction <= 1.0):
        raise ValueError("--top_fraction must be in (0, 1].")
    if args.exclude_sink_tokens < 0 or args.exclude_recent_tokens < 0:
        raise ValueError("--exclude_sink_tokens and --exclude_recent_tokens must be non-negative.")
    if args.prefill_tokens + args.eval_tokens > args.total_tokens:
        raise ValueError("--prefill_tokens + --eval_tokens must be <= --total_tokens.")

    lags = parse_int_list(args.lags, "lags")
    horizons = parse_int_list(args.horizons, "horizons")
    thresholds = parse_float_list(args.thresholds, "thresholds")

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

    collector = MultiStepPositionSharingCollector(
        selected_layers=selected_layers,
        selected_heads=selected_heads,
        query_tokens=set(query_samples),
        first_query_token=min(query_samples),
        top_fraction=args.top_fraction,
        remote_only=args.remote_only,
        exclude_sink_tokens=args.exclude_sink_tokens,
        exclude_recent_tokens=args.exclude_recent_tokens,
        lags=lags,
        horizons=horizons,
        thresholds=thresholds,
        write_block_rows=args.write_block_rows,
    )

    started = time.perf_counter()
    past = prefill_cache(model, input_ids, args.prefill_tokens, args.chunk_size, input_device)
    run_eval(model, input_ids, past, args.prefill_tokens, args.eval_tokens, args.chunk_size, input_device, collector)
    collector.finalize()
    seconds = time.perf_counter() - started

    metric_field_names = metric_fields(thresholds)
    offset_rows = collector.offset_rows()
    offset_layer_head_rows = collector.offset_layer_head_rows()
    horizon_rows = collector.horizon_rows()
    horizon_layer_head_rows = collector.horizon_layer_head_rows()
    block_threshold_rows = collector.block_threshold_summary_rows()

    write_csv(output_dir / "offset_recall_by_lag.csv", offset_rows, ["lag"] + metric_field_names)
    write_csv(
        output_dir / "offset_recall_by_lag_layer_head.csv",
        offset_layer_head_rows,
        ["lag", "layer", "head"] + metric_field_names,
    )
    horizon_fields = [
        "horizon",
        "selector_cases_without_sharing",
        "selectors_with_sharing",
        "selectors_saved",
        "selector_reduction_fraction",
    ] + metric_field_names
    write_csv(output_dir / "fixed_horizon_summary.csv", horizon_rows, horizon_fields)
    write_csv(
        output_dir / "fixed_horizon_layer_head.csv",
        horizon_layer_head_rows,
        ["horizon", "layer", "head"] + horizon_fields[1:],
    )
    block_threshold_fields = [
        "horizon",
        "threshold",
        "blocks_with_shared_steps",
        "blocks_min_recall_ge_threshold",
        "block_pass_fraction",
        "mean_shared_steps_if_pass",
        "mean_min_recall_all_blocks",
        "mean_min_mass_recall_all_blocks",
    ]
    write_csv(output_dir / "fixed_horizon_block_threshold_summary.csv", block_threshold_rows, block_threshold_fields)
    if args.write_block_rows:
        block_fields = [
            "horizon",
            "layer",
            "head",
            "block_id",
            "anchor_query_token",
            "start_query_token",
            "end_query_token",
            "block_step_count",
            "shared_steps",
            "shared_top2_recall_mean",
            "shared_top2_recall_min",
            "shared_attention_mass_recall_mean",
            "shared_attention_mass_recall_min",
        ]
        write_csv(output_dir / "fixed_horizon_blocks.csv", collector.block_rows, block_fields)

    query_rows = [
        token_prefix(tokenizer, token_ids, query_token, args.include_token_text)
        for query_token in sorted(collector.observed_query_tokens)
    ]
    query_fields = ["query_token_index", "query_token_id"]
    if args.include_token_text:
        query_fields += ["query_token_piece", "query_token_text"]
    write_csv(output_dir / "observed_queries.csv", query_rows, query_fields)

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
                "lag": "Reuse the same layer/head top-fraction positions selected lag decode steps earlier.",
                "horizon": (
                    "Fixed block size. One selector is computed at the first query in each block and reused for "
                    "the remaining horizon-1 queries in that block."
                ),
                "top2_recall": "|S(anchor) intersect S(target)| / |S(target)| for the same layer/head.",
                "old_history_top2_recall": (
                    "Intersection divided by target top-fraction positions that existed at the anchor step."
                ),
                "attention_mass_recall": (
                    "Target-step attention mass captured by anchor positions divided by target true top-fraction mass."
                ),
                "block_pass_fraction": (
                    "Fraction of layer/head fixed-horizon blocks whose minimum shared-step recall is above threshold."
                ),
            },
        },
        "paths": {
            "offset_recall_by_lag": str(output_dir / "offset_recall_by_lag.csv"),
            "offset_recall_by_lag_layer_head": str(output_dir / "offset_recall_by_lag_layer_head.csv"),
            "fixed_horizon_summary": str(output_dir / "fixed_horizon_summary.csv"),
            "fixed_horizon_layer_head": str(output_dir / "fixed_horizon_layer_head.csv"),
            "fixed_horizon_block_threshold_summary": str(output_dir / "fixed_horizon_block_threshold_summary.csv"),
            "fixed_horizon_blocks": str(output_dir / "fixed_horizon_blocks.csv") if args.write_block_rows else None,
            "observed_queries": str(output_dir / "observed_queries.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "seconds": seconds,
                "observed_queries": len(collector.observed_query_tokens),
                "offset_recall_by_lag": offset_rows,
                "fixed_horizon_summary": horizon_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
