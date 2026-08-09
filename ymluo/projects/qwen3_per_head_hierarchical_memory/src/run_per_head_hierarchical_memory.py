from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F


PROJECTS_DIR = Path(__file__).resolve().parents[2]
BASE_SRC = PROJECTS_DIR / "qwen3_head_routed_retrieval" / "src"
if str(BASE_SRC) not in sys.path:
    sys.path.insert(0, str(BASE_SRC))

from run_head_retriever_imitation import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    RetrieverBank,
    activate_probe,
    evaluate_phase,
    install_attention_patch,
    model_forward,
    pick_input_device,
    prefill_cache,
    resolve_dtype,
    safe_div,
    str2bool,
    write_csv,
)


SCORE_METHODS = ("position", "lexical", "semantic", "format", "repeat")
POLICIES = ("sink_recent_500", "flat_function_500", "hier_function_500")
KNOWN_HEAD_FUNCTIONS = {
    "mixed_or_common",
    "self",
    "previous_token",
    "local_recent",
    "sink",
    "punctuation",
    "lexical_copy",
    "syntactic_dependency",
    "structural_anchor",
    "semantic_evidence",
}
STAT_NAMES = (
    "oracle_events",
    "hits",
    "oracle_mass",
    "intersection_mass",
    "selected_mass",
    "history_mass",
    "remote_oracle_events",
    "remote_hits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a separate L0/L1/L2 memory hierarchy for every attention head."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--atlas_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefill_tokens", type=int, default=16_384)
    parser.add_argument("--train_queries", type=int, default=64)
    parser.add_argument("--test_queries", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--prefill_chunk_size", type=int, default=128)
    parser.add_argument("--oracle_ratio", type=float, default=0.02)
    parser.add_argument("--l0_capacity", type=int, default=500)
    parser.add_argument("--l0_recent_tokens", type=int, default=256)
    parser.add_argument(
        "--promotion_policy",
        choices=("uniform", "confidence_gated"),
        default="uniform",
    )
    parser.add_argument("--medium_promotion_slots", type=int, default=20)
    parser.add_argument(
        "--promotion_categories",
        default="semantic_evidence,lexical_copy,structural_anchor",
        help="Comma-separated conservative head functions allowed to promote old tokens.",
    )
    parser.add_argument(
        "--remote_cutoff_tokens",
        type=int,
        default=0,
        help="Positions older than this are remote; 0 uses the full L0 capacity.",
    )
    parser.add_argument("--l1_capacity", type=int, default=4096)
    parser.add_argument("--l2_block_size", type=int, default=64)
    parser.add_argument("--l2_block_budget", type=int, default=64)
    parser.add_argument("--sink_tokens", type=int, default=4)
    parser.add_argument("--query_window", type=int, default=32)
    parser.add_argument("--repeat_max_n", type=int, default=4)
    parser.add_argument("--l1_retention_bonus", type=float, default=0.05)
    parser.add_argument("--l0_retention_bonus", type=float, default=0.05)
    parser.add_argument("--function_temperature", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=20260716)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def historical_budget(history_tokens: int, ratio: float) -> int:
    if history_tokens <= 0:
        return 0
    return min(history_tokens, max(1, int(math.ceil(history_tokens * ratio))))


def _positive(value: str | float) -> float:
    return max(0.0, float(value))


def functional_method_weights(
    atlas_rows: Sequence[dict[str, str]], *, temperature: float = 1.0
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ordered = sorted(atlas_rows, key=lambda row: (int(row["layer"]), int(row["head"])))
    raw_rows: list[list[float]] = []
    route_rows: list[dict[str, Any]] = []
    for row in ordered:
        position = max(
            _positive(row["attention_z_self"]),
            _positive(row["attention_z_previous_token"]),
            _positive(row["attention_z_local_recent"]),
            _positive(row["attention_z_sink"]),
        )
        lexical = _positive(row["attention_z_lexical_copy"])
        semantic = max(
            _positive(row["attention_z_semantic_evidence"]),
            0.75 * _positive(row["attention_z_syntactic_dependency"]),
        )
        format_score = max(
            _positive(row["attention_z_punctuation"]),
            _positive(row["attention_z_structural_anchor"]),
        )
        repeat = 0.8 * lexical
        raw = [position, lexical, semantic, format_score, repeat]
        raw_rows.append(raw)
        route_rows.append(
            {
                "head_id": row.get("head_id", f"L{int(row['layer']):02d}H{int(row['head']):02d}"),
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "dominant_signature": row["dominant_signature"],
                "confidence": row["confidence"],
                "raw_position": position,
                "raw_lexical": lexical,
                "raw_semantic": semantic,
                "raw_format": format_score,
                "raw_repeat": repeat,
            }
        )
    weights = torch.softmax(torch.tensor(raw_rows, dtype=torch.float32) / temperature, dim=-1)
    for index, row in enumerate(route_rows):
        for method_index, method in enumerate(SCORE_METHODS):
            row[f"weight_{method}"] = float(weights[index, method_index])
        row["mixture_primary_method"] = SCORE_METHODS[int(weights[index].argmax())]
    return weights, route_rows


def read_atlas(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 448:
        raise ValueError(f"expected 448 head rows in atlas, got {len(rows)}")
    return rows


def promotion_slots_from_atlas(
    atlas_rows: Sequence[dict[str, str]],
    *,
    l0_capacity: int,
    l0_recent_tokens: int,
    sink_tokens: int,
    policy: str,
    medium_slots: int,
    active_categories: Sequence[str] | None = None,
) -> torch.Tensor:
    max_slots = max(0, l0_capacity - sink_tokens - l0_recent_tokens)
    if policy == "uniform":
        return torch.full((len(atlas_rows),), max_slots, dtype=torch.long)
    if policy != "confidence_gated":
        raise ValueError(f"unknown promotion policy: {policy}")
    if active_categories is None:
        active_categories = ("semantic_evidence", "lexical_copy", "structural_anchor")
    active_categories = {item.strip() for item in active_categories if item.strip()}
    unknown_categories = active_categories - KNOWN_HEAD_FUNCTIONS
    if unknown_categories:
        raise ValueError(f"unknown promotion categories: {sorted(unknown_categories)}")
    slots: list[int] = []
    for row in sorted(atlas_rows, key=lambda item: (int(item["layer"]), int(item["head"]))):
        conservative = row["conservative_function"]
        confidence = row["confidence"]
        if conservative not in active_categories or confidence == "低":
            slots.append(0)
        elif confidence == "高":
            slots.append(max_slots)
        else:
            slots.append(min(max_slots, medium_slots))
    return torch.tensor(slots, dtype=torch.long)


class PerHeadHierarchicalMemory:
    """Independent persistent L1/L0 state for every query head.

    L2 is the immutable full-history external block index exposed by ``RetrieverBank``.
    This class never consumes attention scores.
    """

    def __init__(
        self,
        retriever: RetrieverBank,
        head_weights: torch.Tensor,
        *,
        l0_capacity: int,
        l0_recent_tokens: int,
        l1_capacity: int,
        l2_block_size: int,
        l2_block_budget: int,
        sink_tokens: int,
        l1_retention_bonus: float,
        l0_retention_bonus: float,
        promotion_slots: torch.Tensor | None = None,
    ) -> None:
        if head_weights.ndim != 2 or head_weights.shape[1] != len(SCORE_METHODS):
            raise ValueError("head_weights must have shape [heads, 5]")
        if l0_capacity <= 0 or l0_capacity > 500:
            raise ValueError("L0 capacity must be in [1, 500]")
        if l1_capacity < l0_capacity:
            raise ValueError("L1 capacity must be at least L0 capacity")
        if l0_recent_tokens + sink_tokens > l0_capacity:
            raise ValueError("sink plus recent reservation cannot exceed L0 capacity")
        if l2_block_size <= 0 or l2_block_budget <= 0:
            raise ValueError("L2 block settings must be positive")
        self.retriever = retriever
        self.weights = head_weights.to("cpu", dtype=torch.float32).contiguous()
        self.head_count = int(head_weights.shape[0])
        self.l0_capacity = l0_capacity
        self.l0_recent_tokens = l0_recent_tokens
        self.l1_capacity = l1_capacity
        self.l2_block_size = l2_block_size
        self.l2_block_budget = l2_block_budget
        self.sink_tokens = sink_tokens
        self.l1_retention_bonus = l1_retention_bonus
        self.l0_retention_bonus = l0_retention_bonus
        uniform_slots = max(0, l0_capacity - sink_tokens - l0_recent_tokens)
        if promotion_slots is None:
            promotion_slots = torch.full(
                (self.head_count,), uniform_slots, dtype=torch.long
            )
        if promotion_slots.shape != (self.head_count,):
            raise ValueError("promotion_slots must have one value per head")
        if int(promotion_slots.min()) < 0 or int(promotion_slots.max()) > l0_capacity:
            raise ValueError("invalid per-head promotion slot count")
        self.promotion_slots = promotion_slots.to("cpu", dtype=torch.long).contiguous()
        self.warm_positions = torch.empty((self.head_count, 0), dtype=torch.long)
        self.hot_positions = torch.empty((self.head_count, 0), dtype=torch.long)
        self._selection_cache: dict[int, dict[str, torch.Tensor]] = {}
        self._last_updated = -1
        self.diagnostic_rows: list[dict[str, Any]] = []

    def _normalized_score_matrix(self, current: int) -> torch.Tensor:
        score_map = self.retriever.scores(current)
        normalized: list[torch.Tensor] = []
        for method in SCORE_METHODS:
            score = score_map[method].float()
            centered = score - score.mean()
            scale = centered.square().mean().sqrt().clamp_min(1e-6)
            normalized.append(centered / scale)
        base = torch.stack(normalized, dim=0)
        composite = self.weights @ base
        # A deterministic per-head perturbation only breaks exact score ties.
        positions = torch.arange(current, dtype=torch.float32).unsqueeze(0)
        heads = torch.arange(self.head_count, dtype=torch.float32).unsqueeze(1)
        tie = ((positions + 1) * (heads + 17)).remainder(104729.0) * 1e-10
        return composite + tie

    def _mandatory_positions(self, current: int, recent_tokens: int) -> torch.Tensor:
        sink = torch.arange(min(self.sink_tokens, current), dtype=torch.long)
        recent_start = max(0, current - recent_tokens)
        recent = torch.arange(recent_start, current, dtype=torch.long)
        return torch.unique(torch.cat([sink, recent]), sorted=True)

    def _assemble_hot(
        self,
        composite: torch.Tensor,
        eligible_mask: torch.Tensor,
        *,
        use_retention: bool,
    ) -> torch.Tensor:
        current = int(composite.shape[1])
        k = min(self.l0_capacity, current)
        sink_count = min(self.sink_tokens, current, k)
        slots = self.promotion_slots.clamp(max=max(0, k - sink_count))
        recent_counts = k - sink_count - slots
        positions = torch.arange(current, dtype=torch.long).unsqueeze(0)
        mandatory_mask = positions < sink_count
        recent_threshold = current - recent_counts.unsqueeze(1)
        mandatory_mask = mandatory_mask | (positions >= recent_threshold)
        candidate_mask = eligible_mask | mandatory_mask
        ranks = composite.clone()
        if use_retention and self.hot_positions.numel():
            valid = (self.hot_positions >= 0) & (self.hot_positions < current)
            if valid.any():
                rows = torch.arange(self.head_count).unsqueeze(1).expand_as(self.hot_positions)[valid]
                cols = self.hot_positions[valid]
                ranks[rows, cols] += self.l0_retention_bonus
        ranks += mandatory_mask.float() * 1_000_000.0
        ranks.masked_fill_(~candidate_mask, -torch.inf)
        return torch.topk(ranks, k=k, dim=-1).indices.sort(dim=-1).values

    def _update_hierarchy(self, composite: torch.Tensor) -> torch.Tensor:
        current = int(composite.shape[1])
        blocks = math.ceil(current / self.l2_block_size)
        padded_tokens = blocks * self.l2_block_size
        padded = F.pad(composite, (0, padded_tokens - current), value=0.0)
        block_sum = padded.view(self.head_count, blocks, self.l2_block_size).sum(dim=-1)
        counts = torch.full((blocks,), self.l2_block_size, dtype=torch.float32)
        counts[-1] = current - (blocks - 1) * self.l2_block_size
        block_scores = block_sum / counts.unsqueeze(0)
        block_k = min(self.l2_block_budget, blocks)
        top_blocks = torch.topk(block_scores, k=block_k, dim=-1).indices
        offsets = torch.arange(self.l2_block_size, dtype=torch.long)
        cold_candidates = (
            top_blocks.unsqueeze(-1) * self.l2_block_size + offsets.view(1, 1, -1)
        ).reshape(self.head_count, -1)
        cold_valid = cold_candidates < current

        warm_mask = torch.zeros((self.head_count, current), dtype=torch.bool)
        safe_candidates = cold_candidates.clamp(max=current - 1)
        warm_mask.scatter_(1, safe_candidates, cold_valid)
        ranks = composite.clone()
        if self.warm_positions.numel():
            valid = (self.warm_positions >= 0) & (self.warm_positions < current)
            safe_warm = self.warm_positions.clamp(min=0, max=current - 1)
            warm_mask.scatter_(1, safe_warm, valid)
            rows = torch.arange(self.head_count).unsqueeze(1).expand_as(self.warm_positions)[valid]
            cols = self.warm_positions[valid]
            ranks[rows, cols] += self.l1_retention_bonus
        ranks.masked_fill_(~warm_mask, -torch.inf)
        warm_k = min(self.l1_capacity, current)
        self.warm_positions = torch.topk(ranks, k=warm_k, dim=-1).indices.sort(dim=-1).values

        eligible = torch.zeros((self.head_count, current), dtype=torch.bool)
        eligible.scatter_(1, self.warm_positions, True)
        self.hot_positions = self._assemble_hot(
            composite, eligible, use_retention=True
        )
        return self.hot_positions

    def _compute(self, current: int) -> dict[str, torch.Tensor]:
        if current <= 0:
            raise ValueError("current must be positive")
        composite = self._normalized_score_matrix(current)
        capacity = min(self.l0_capacity, current)

        sink_count = min(self.sink_tokens, current, capacity)
        recent_slots = max(0, capacity - sink_count)
        recent = self._mandatory_positions(current, recent_slots)
        if len(recent) < capacity:
            fill = torch.arange(max(0, current - capacity), current, dtype=torch.long)
            recent = torch.unique(torch.cat([recent, fill]), sorted=True)
        if len(recent) > capacity:
            sink = torch.arange(sink_count, dtype=torch.long)
            non_sink = recent[recent >= sink_count]
            recent = torch.cat([sink, non_sink[-(capacity - sink_count) :]])
        recent_policy = recent.unsqueeze(0).expand(self.head_count, -1).clone()

        all_eligible = torch.ones((self.head_count, current), dtype=torch.bool)
        flat_policy = self._assemble_hot(composite, all_eligible, use_retention=False)
        hierarchical_policy = self._update_hierarchy(composite)
        result = {
            "sink_recent_500": recent_policy,
            "flat_function_500": flat_policy,
            "hier_function_500": hierarchical_policy.clone(),
        }
        for name, selection in result.items():
            if selection.shape != (self.head_count, capacity):
                raise AssertionError(f"{name} returned shape {tuple(selection.shape)}")
            if int(selection.max()) >= current or int(selection.min()) < 0:
                raise AssertionError(f"{name} selected a non-historical token")
            if any(len(torch.unique(row)) != capacity for row in selection):
                raise AssertionError(f"{name} contains duplicate token positions")
        self.diagnostic_rows.append(
            {
                "current": current,
                "cold_history_tokens": current,
                "cold_block_count": math.ceil(current / self.l2_block_size),
                "l2_shortlisted_blocks_per_head": min(
                    self.l2_block_budget, math.ceil(current / self.l2_block_size)
                ),
                "l1_min_tokens": int(self.warm_positions.shape[1]),
                "l1_max_tokens": int(self.warm_positions.shape[1]),
                "l0_min_tokens": int(hierarchical_policy.shape[1]),
                "l0_max_tokens": int(hierarchical_policy.shape[1]),
                "unique_hierarchical_hot_sets": int(
                    torch.unique(hierarchical_policy, dim=0).shape[0]
                ),
            }
        )
        return result

    def selections(self, current: int) -> dict[str, torch.Tensor]:
        if current in self._selection_cache:
            return self._selection_cache[current]
        if self._last_updated >= 0 and current < self._last_updated:
            raise KeyError(f"selection for historical query {current} was discarded")
        result = self._compute(current)
        self._selection_cache[current] = result
        self._last_updated = current
        return result

    def discard_before(self, current: int) -> None:
        for key in [key for key in self._selection_cache if key < current]:
            del self._selection_cache[key]


class HierarchicalMemoryProbe:
    def __init__(
        self,
        bank: PerHeadHierarchicalMemory,
        *,
        layer_count: int,
        head_count: int,
        kv_head_count: int,
        train_start: int,
        train_end: int,
        test_end: int,
        oracle_ratio: float,
        sink_tokens: int,
        remote_cutoff_tokens: int,
        device: torch.device,
    ) -> None:
        if bank.head_count != layer_count * head_count:
            raise ValueError("bank head count must equal layer_count * head_count")
        self.bank = bank
        self.layer_count = layer_count
        self.head_count = head_count
        self.kv_head_count = kv_head_count
        self.train_start = train_start
        self.train_end = train_end
        self.test_end = test_end
        self.oracle_ratio = oracle_ratio
        self.sink_tokens = sink_tokens
        self.remote_cutoff_tokens = remote_cutoff_tokens
        shape = (2, len(POLICIES), layer_count, head_count)
        self.stats = {
            name: torch.zeros(shape, dtype=torch.float64, device=device) for name in STAT_NAMES
        }
        self.query_rows = torch.zeros((2, layer_count), dtype=torch.int64, device=device)
        self.gqa_union_tokens = torch.zeros(
            (len(POLICIES), layer_count, kv_head_count), dtype=torch.float64, device=device
        )
        self.gqa_single_tokens = torch.zeros_like(self.gqa_union_tokens)
        self.gqa_query_rows = torch.zeros_like(self.gqa_union_tokens)

    def split_index(self, current: int) -> int | None:
        if self.train_start <= current < self.train_end:
            return 0
        if self.train_end <= current < self.test_end:
            return 1
        return None

    @torch.inference_mode()
    def update(
        self,
        module: torch.nn.Module,
        scores: torch.Tensor,
        full_weights: torch.Tensor,
    ) -> None:
        if scores.shape[0] != 1:
            raise ValueError("hierarchical-memory probe requires batch size 1")
        layer = int(getattr(module, "layer_idx", 0))
        _, heads, query_count, key_count = scores.shape
        if heads != self.head_count:
            raise ValueError(f"expected {self.head_count} query heads, got {heads}")
        past_tokens = key_count - query_count
        flat_start = layer * self.head_count
        flat_end = flat_start + self.head_count
        for query_index in range(query_count):
            current = past_tokens + query_index
            split = self.split_index(current)
            if split is None or current <= 0:
                continue
            selections = self.bank.selections(current)
            budget = historical_budget(current, self.oracle_ratio)
            history_scores = scores[0, :, query_index, :current]
            oracle_indices = torch.topk(history_scores, k=budget, dim=-1).indices
            history_prob = full_weights[0, :, query_index, :current].float()
            oracle_prob = history_prob.gather(1, oracle_indices)
            oracle_mass = oracle_prob.sum(dim=-1)
            history_mass = history_prob.sum(dim=-1)
            remote_cutoff = max(self.sink_tokens, current - self.remote_cutoff_tokens)
            oracle_remote = (oracle_indices >= self.sink_tokens) & (
                oracle_indices < remote_cutoff
            )
            self.query_rows[split, layer] += 1
            for policy_index, policy in enumerate(POLICIES):
                selected = selections[policy][flat_start:flat_end].to(scores.device)
                selected_mask = torch.zeros(
                    (self.head_count, current), dtype=torch.bool, device=scores.device
                )
                selected_mask.scatter_(1, selected, True)
                oracle_hit = selected_mask.gather(1, oracle_indices)
                self.stats["oracle_events"][split, policy_index, layer] += budget
                self.stats["hits"][split, policy_index, layer] += oracle_hit.sum(dim=-1)
                self.stats["oracle_mass"][split, policy_index, layer] += oracle_mass
                self.stats["intersection_mass"][split, policy_index, layer] += (
                    oracle_prob * oracle_hit
                ).sum(dim=-1)
                self.stats["selected_mass"][split, policy_index, layer] += history_prob.gather(
                    1, selected
                ).sum(dim=-1)
                self.stats["history_mass"][split, policy_index, layer] += history_mass
                self.stats["remote_oracle_events"][split, policy_index, layer] += (
                    oracle_remote.sum(dim=-1)
                )
                self.stats["remote_hits"][split, policy_index, layer] += (
                    oracle_hit & oracle_remote
                ).sum(dim=-1)
                if split == 1:
                    self._update_gqa(policy_index, layer, selected)
        if layer == self.layer_count - 1 and query_count:
            self.bank.discard_before(past_tokens + query_count)

    def _update_gqa(
        self, policy_index: int, layer: int, selected: torch.Tensor
    ) -> None:
        heads_per_group = self.head_count // self.kv_head_count
        for group in range(self.kv_head_count):
            start = group * heads_per_group
            end = start + heads_per_group
            union = torch.unique(selected[start:end].reshape(-1)).numel()
            self.gqa_union_tokens[policy_index, layer, group] += union
            self.gqa_single_tokens[policy_index, layer, group] += selected.shape[1]
            self.gqa_query_rows[policy_index, layer, group] += 1

    def metric_rows(self) -> list[dict[str, Any]]:
        stats = {name: tensor.detach().cpu() for name, tensor in self.stats.items()}
        rows: list[dict[str, Any]] = []
        for split_index, split in enumerate(("train", "test")):
            for policy_index, policy in enumerate(POLICIES):
                for layer in range(self.layer_count):
                    for head in range(self.head_count):
                        events = float(stats["oracle_events"][split_index, policy_index, layer, head])
                        hits = float(stats["hits"][split_index, policy_index, layer, head])
                        oracle_mass = float(
                            stats["oracle_mass"][split_index, policy_index, layer, head]
                        )
                        intersection = float(
                            stats["intersection_mass"][split_index, policy_index, layer, head]
                        )
                        history_mass = float(
                            stats["history_mass"][split_index, policy_index, layer, head]
                        )
                        remote_events = float(
                            stats["remote_oracle_events"][split_index, policy_index, layer, head]
                        )
                        remote_hits = float(
                            stats["remote_hits"][split_index, policy_index, layer, head]
                        )
                        rows.append(
                            {
                                "split": split,
                                "policy": policy,
                                "layer": layer,
                                "head": head,
                                "oracle_events": int(events),
                                "hits": int(hits),
                                "oracle_position_recall": safe_div(hits, events),
                                "oracle_mass_recall": safe_div(intersection, oracle_mass),
                                "selected_history_attention_mass_fraction": safe_div(
                                    float(
                                        stats["selected_mass"][
                                            split_index, policy_index, layer, head
                                        ]
                                    ),
                                    history_mass,
                                ),
                                "remote_oracle_events": int(remote_events),
                                "remote_hits": int(remote_hits),
                                "remote_oracle_position_recall": safe_div(
                                    remote_hits, remote_events
                                ),
                            }
                        )
        return rows

    def aggregate_rows(self) -> list[dict[str, Any]]:
        stats = {name: tensor.detach().cpu() for name, tensor in self.stats.items()}
        rows: list[dict[str, Any]] = []
        for split_index, split in enumerate(("train", "test")):
            for policy_index, policy in enumerate(POLICIES):
                events = float(stats["oracle_events"][split_index, policy_index].sum())
                hits = float(stats["hits"][split_index, policy_index].sum())
                oracle_mass = float(stats["oracle_mass"][split_index, policy_index].sum())
                intersection = float(
                    stats["intersection_mass"][split_index, policy_index].sum()
                )
                selected_mass = float(stats["selected_mass"][split_index, policy_index].sum())
                history_mass = float(stats["history_mass"][split_index, policy_index].sum())
                remote_events = float(
                    stats["remote_oracle_events"][split_index, policy_index].sum()
                )
                remote_hits = float(stats["remote_hits"][split_index, policy_index].sum())
                rows.append(
                    {
                        "split": split,
                        "policy": policy,
                        "oracle_events": int(events),
                        "hits": int(hits),
                        "oracle_position_recall": safe_div(hits, events),
                        "oracle_mass_recall": safe_div(intersection, oracle_mass),
                        "selected_history_attention_mass_fraction": safe_div(
                            selected_mass, history_mass
                        ),
                        "remote_oracle_events": int(remote_events),
                        "remote_hits": int(remote_hits),
                        "remote_oracle_position_recall": safe_div(
                            remote_hits, remote_events
                        ),
                    }
                )
        return rows

    def gqa_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        union = self.gqa_union_tokens.detach().cpu()
        single = self.gqa_single_tokens.detach().cpu()
        queries = self.gqa_query_rows.detach().cpu()
        for policy_index, policy in enumerate(POLICIES):
            for layer in range(self.layer_count):
                for group in range(self.kv_head_count):
                    rows.append(
                        {
                            "policy": policy,
                            "layer": layer,
                            "kv_group": group,
                            "query_rows": int(queries[policy_index, layer, group]),
                            "mean_union_tokens": safe_div(
                                float(union[policy_index, layer, group]),
                                float(queries[policy_index, layer, group]),
                            ),
                            "union_vs_single_head_l0": safe_div(
                                float(union[policy_index, layer, group]),
                                float(single[policy_index, layer, group]),
                            ),
                        }
                    )
        return rows


def main() -> None:
    args = parse_args()
    if not (0 < args.oracle_ratio <= 1):
        raise ValueError("oracle_ratio must be in (0, 1]")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.random_seed)
    started = time.perf_counter()

    text_path = Path(args.text_path)
    with text_path.open("r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read(args.max_chars) if args.max_chars > 0 else handle.read()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    total_tokens = args.prefill_tokens + args.train_queries + args.test_queries
    token_ids_list = tokenizer(
        text,
        add_special_tokens=args.add_special_tokens,
        truncation=True,
        max_length=total_tokens,
    )["input_ids"]
    if len(token_ids_list) < total_tokens:
        raise ValueError(f"tokenized text has {len(token_ids_list)} tokens; need {total_tokens}")
    token_ids_list = token_ids_list[:total_tokens]
    input_ids = torch.tensor(token_ids_list, dtype=torch.long).view(1, -1)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": resolve_dtype(args.dtype, requested_device),
    }
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(requested_device)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, requested_device)
    install_attention_patch()

    with torch.inference_mode():
        token_embeddings = model.get_input_embeddings()(input_ids.to(input_device))[0].float().cpu()
    decoded_tokens = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in token_ids_list
    ]
    external_retriever = RetrieverBank(
        input_ids[0],
        token_embeddings,
        decoded_tokens,
        ratio=args.oracle_ratio,
        query_window=args.query_window,
        block_size=args.l2_block_size,
        repeat_max_n=args.repeat_max_n,
        sink_tokens=args.sink_tokens,
        hybrid_position_fraction=0.5,
        random_seed=args.random_seed,
    )
    atlas_rows = read_atlas(Path(args.atlas_csv))
    head_weights, route_rows = functional_method_weights(
        atlas_rows, temperature=args.function_temperature
    )
    layer_count = int(model.config.num_hidden_layers)
    head_count = int(model.config.num_attention_heads)
    kv_head_count = int(getattr(model.config, "num_key_value_heads", head_count))
    if len(head_weights) != layer_count * head_count:
        raise ValueError("atlas/model head count mismatch")
    memory = PerHeadHierarchicalMemory(
        external_retriever,
        head_weights,
        l0_capacity=args.l0_capacity,
        l0_recent_tokens=args.l0_recent_tokens,
        l1_capacity=args.l1_capacity,
        l2_block_size=args.l2_block_size,
        l2_block_budget=args.l2_block_budget,
        sink_tokens=args.sink_tokens,
        l1_retention_bonus=args.l1_retention_bonus,
        l0_retention_bonus=args.l0_retention_bonus,
        promotion_slots=promotion_slots_from_atlas(
            atlas_rows,
            l0_capacity=args.l0_capacity,
            l0_recent_tokens=args.l0_recent_tokens,
            sink_tokens=args.sink_tokens,
            policy=args.promotion_policy,
            medium_slots=args.medium_promotion_slots,
            active_categories=args.promotion_categories.split(","),
        ),
    )
    for row, slots in zip(route_rows, memory.promotion_slots.tolist()):
        row["promotion_slots"] = int(slots)
        row["resident_recent_tokens"] = args.l0_capacity - args.sink_tokens - int(slots)
    train_start = args.prefill_tokens
    train_end = train_start + args.train_queries
    test_end = train_end + args.test_queries
    probe = HierarchicalMemoryProbe(
        memory,
        layer_count=layer_count,
        head_count=head_count,
        kv_head_count=kv_head_count,
        train_start=train_start,
        train_end=train_end,
        test_end=test_end,
        oracle_ratio=args.oracle_ratio,
        sink_tokens=args.sink_tokens,
        remote_cutoff_tokens=(
            args.remote_cutoff_tokens
            if args.remote_cutoff_tokens > 0
            else args.l0_capacity
        ),
        device=input_device,
    )

    print("prefill starting", flush=True)
    past, previous_logits = prefill_cache(
        model, input_ids, args.prefill_tokens, args.prefill_chunk_size, input_device
    )
    print("train-query probe starting", flush=True)
    past, previous_logits, train_nll, train_count = evaluate_phase(
        model,
        input_ids,
        past,
        previous_logits,
        start=train_start,
        end=train_end,
        chunk_size=args.chunk_size,
        input_device=input_device,
        probe=probe,
        phase="train",
        log_every=args.log_every,
    )
    print("test-query probe starting", flush=True)
    _, _, test_nll, test_count = evaluate_phase(
        model,
        input_ids,
        past,
        previous_logits,
        start=train_end,
        end=test_end,
        chunk_size=args.chunk_size,
        input_device=input_device,
        probe=probe,
        phase="test",
        log_every=args.log_every,
    )

    metric_rows = probe.metric_rows()
    aggregate_rows = probe.aggregate_rows()
    gqa_rows = probe.gqa_rows()
    write_csv(output_dir / "per_head_memory_metrics.csv", metric_rows)
    write_csv(output_dir / "aggregate_memory_metrics.csv", aggregate_rows)
    write_csv(output_dir / "gqa_memory_union.csv", gqa_rows)
    write_csv(output_dir / "head_function_mixture.csv", route_rows)
    write_csv(output_dir / "memory_diagnostics.csv", memory.diagnostic_rows)

    test_rows = {row["policy"]: row for row in aggregate_rows if row["split"] == "test"}
    gqa_test = {
        policy: safe_div(
            float(probe.gqa_union_tokens[index].sum()),
            float(probe.gqa_single_tokens[index].sum()),
        )
        for index, policy in enumerate(POLICIES)
    }
    route_counts = Counter(row["mixture_primary_method"] for row in route_rows)
    max_l0 = max(int(row["l0_max_tokens"]) for row in memory.diagnostic_rows)
    min_unique = min(int(row["unique_hierarchical_hot_sets"]) for row in memory.diagnostic_rows)
    summary = {
        "args": vars(args),
        "resolved": {
            "total_tokens": total_tokens,
            "layers": layer_count,
            "query_heads": head_count,
            "total_head_memories": layer_count * head_count,
            "kv_heads": kv_head_count,
            "policies": POLICIES,
            "primary_function_mixture_counts": dict(route_counts),
            "promotion_slot_counts": dict(
                Counter(int(value) for value in memory.promotion_slots.tolist())
            ),
        },
        "full_attention_ppl": {
            "train": math.exp(train_nll / max(1, train_count)),
            "test": math.exp(test_nll / max(1, test_count)),
        },
        "test_metrics": test_rows,
        "test_mean_gqa_union_vs_single_l0": gqa_test,
        "invariants": {
            "configured_l0_capacity": args.l0_capacity,
            "observed_max_l0_tokens": max_l0,
            "l0_capacity_respected": max_l0 <= 500 and max_l0 <= args.l0_capacity,
            "minimum_distinct_hierarchical_hot_sets_across_queries": min_unique,
            "all_448_heads_have_independent_state_rows": len(route_rows) == 448,
        },
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": (
            "Offline oracle-imitation evaluation. Retrieval does not read attention QK; "
            "full attention is used only to score whether each head-specific L0 contains "
            "the oracle Top-2% positions. Sparse PPL injection remains a separate stage."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["test_metrics"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(summary["invariants"], ensure_ascii=False, indent=2), flush=True)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
