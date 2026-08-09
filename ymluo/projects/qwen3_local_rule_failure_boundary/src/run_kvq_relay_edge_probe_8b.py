from __future__ import annotations

"""Frozen-KV K->V->Q->K relay edge diagnostic for Qwen3-8B.

The probe never changes attention or the KV cache and never consumes a gold
label while proposing blocks or scoring directed edges.  It asks a narrower
question: if the value content of source block B were added at a fixed layer
boundary, does the next layer's pre-RoPE query move toward destination block
C's pre-RoPE keys?

This file is intentionally independent from the phase, value-attribution,
safety, and query-span runners.  The current implementation uses a central
finite difference through only

    next-layer input RMSNorm -> Q projection -> Q normalization.

It is therefore a local layer-boundary diagnostic, not a full-block causal
intervention and not a sparse-attention consumer.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

import run_length_causal_mechanism_20260717 as causal
import run_local_global_rope_probe_8b as model_runner
import run_local_rule_failure_boundary as base


SCORE_NAMES = (
    "kvq_relay",
    "shuffled_v_relay",
    "norm_matched_random_v_relay",
    "reverse_edge_relay",
    "kk_similarity",
    "pre_score_pair",
)

PROTOCOL_VERSION = "kvq-relay-edge-v2"
CASE_KEY_FIELDS = ("target_context_tokens", "condition", "seed")


def parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text("", encoding="utf-8")
        temporary.replace(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def clear_allocator() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class CandidateBlock:
    index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def center(self) -> float:
        return 0.5 * (self.start + self.end - 1)


def resolve_relay_layers(
    layer_count: int, explicit_layers: Sequence[int] | None = None
) -> tuple[int, ...]:
    """Resolve exactly four valid source layers unless explicitly overridden."""

    if layer_count < 5:
        raise ValueError("KVQ-R needs at least five decoder layers")
    if explicit_layers:
        values = tuple(dict.fromkeys(int(value) for value in explicit_layers))
    else:
        values = (
            layer_count // 4,
            layer_count // 2,
            (3 * layer_count) // 4,
            layer_count - 2,
        )
        values = tuple(dict.fromkeys(values))
    if len(values) != 4:
        raise ValueError(f"relay source layers must contain four unique values: {values}")
    if any(value < 0 or value + 1 >= layer_count for value in values):
        raise ValueError(f"relay layer outside [0, {layer_count - 2}]: {values}")
    return values


def segment_label_free_blocks(
    token_ids: Sequence[int],
    start: int,
    end: int,
    max_block_tokens: int,
    boundary_token_ids: set[int] | frozenset[int],
) -> list[CandidateBlock]:
    """Split visible tokens using only newline-like IDs and a maximum length."""

    if max_block_tokens <= 0:
        raise ValueError("max_block_tokens must be positive")
    if not 0 <= start < end <= len(token_ids):
        raise ValueError(f"invalid block region [{start}, {end}) for {len(token_ids)} tokens")
    blocks: list[CandidateBlock] = []
    cursor = int(start)
    for position in range(start, end):
        length = position + 1 - cursor
        boundary = int(token_ids[position]) in boundary_token_ids
        if boundary or length >= max_block_tokens:
            blocks.append(CandidateBlock(len(blocks), cursor, position + 1))
            cursor = position + 1
    if cursor < end:
        blocks.append(CandidateBlock(len(blocks), cursor, end))
    if not blocks or any(block.length <= 0 for block in blocks):
        raise RuntimeError("label-free block segmentation produced an empty block")
    return blocks


def newline_like_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> set[int]:
    """Find visible newline tokens without reading any record/evidence metadata."""

    output: set[int] = set()
    for token_id in sorted(set(map(int, token_ids))):
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if "\n" in text or "\r" in text:
            output.add(token_id)
    return output


def gqa_query_key_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Pre-RoPE QK scores without materializing repeated GQA keys.

    Args:
        query: ``[query_heads, head_dim]``.
        key: ``[kv_heads, sequence, head_dim]``.
    Returns:
        ``[query_heads, sequence]`` scaled by ``1/sqrt(head_dim)``.
    """

    if query.dim() != 2 or key.dim() != 3:
        raise ValueError("query/key must be [Hq,D] and [Hkv,T,D]")
    query_heads, dim = query.shape
    kv_heads, _, key_dim = key.shape
    if dim != key_dim or query_heads % kv_heads != 0:
        raise ValueError("incompatible GQA query/key shapes")
    groups = query_heads // kv_heads
    grouped = query.float().reshape(kv_heads, groups, dim)
    scores = torch.einsum("kgd,ktd->kgt", grouped, key.float())
    return scores.reshape(query_heads, key.shape[1]) / math.sqrt(dim)


def block_logmeanexp_scores(
    token_scores: torch.Tensor,
    blocks: Sequence[CandidateBlock],
    temperature: float,
) -> torch.Tensor:
    """Length-corrected block scores, returned as ``[heads, blocks]``."""

    if token_scores.dim() != 2 or temperature <= 0:
        raise ValueError("token_scores must be [heads,tokens] and temperature > 0")
    output: list[torch.Tensor] = []
    for block in blocks:
        local = token_scores[:, block.start : block.end].float()
        if local.shape[-1] != block.length:
            raise ValueError("block lies outside token_scores")
        score = temperature * torch.logsumexp(local / temperature, dim=-1)
        score = score - temperature * math.log(block.length)
        output.append(score)
    return torch.stack(output, dim=-1)


def robust_standardize(values: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    flat = values.float().reshape(-1)
    median = flat.median()
    mad = (flat - median).abs().median()
    scale = 1.4826 * mad
    if float(scale.item()) < epsilon:
        scale = flat.std(unbiased=False).clamp_min(epsilon)
    return (values.float() - median) / scale


def select_candidate_blocks(
    layer_block_scores: Mapping[int, torch.Tensor],
    blocks: Sequence[CandidateBlock],
    maximum_blocks: int,
) -> tuple[list[CandidateBlock], torch.Tensor]:
    """Label-free Top-M over robustly aggregated final-query pre-scores."""

    if not layer_block_scores or maximum_blocks <= 0:
        raise ValueError("layer scores and a positive maximum_blocks are required")
    layer_values: list[torch.Tensor] = []
    for scores in layer_block_scores.values():
        if scores.dim() != 2 or int(scores.shape[1]) != len(blocks):
            raise ValueError("each layer score must be [heads, all_blocks]")
        layer_values.append(robust_standardize(scores.float().mean(dim=0)))
    aggregate = torch.stack(layer_values, dim=0).mean(dim=0)
    keep = min(len(blocks), int(maximum_blocks))
    ranked = torch.argsort(aggregate, descending=True, stable=True)[:keep].tolist()
    selected = sorted((blocks[int(index)] for index in ranked), key=lambda item: item.start)
    return selected, aggregate


def aggregate_block_head_values(
    token_scores: torch.Tensor,
    value: torch.Tensor,
    blocks: Sequence[CandidateBlock],
    temperature: float,
) -> torch.Tensor:
    """Query-weight original V inside each block, with exact GQA mapping.

    Args:
        token_scores: ``[query_heads, sequence]``.
        value: ``[kv_heads, sequence, head_dim]``.
    Returns:
        ``[blocks, query_heads, head_dim]``.
    """

    if token_scores.dim() != 2 or value.dim() != 3 or temperature <= 0:
        raise ValueError("invalid token_scores/value/temperature")
    query_heads = int(token_scores.shape[0])
    kv_heads = int(value.shape[0])
    if query_heads % kv_heads != 0 or token_scores.shape[1] != value.shape[1]:
        raise ValueError("incompatible GQA token scores and values")
    groups = query_heads // kv_heads
    kv_index = torch.arange(query_heads, device=value.device) // groups
    output: list[torch.Tensor] = []
    for block in blocks:
        weights = torch.softmax(
            token_scores[:, block.start : block.end].float() / temperature,
            dim=-1,
        ).to(device=value.device, dtype=value.dtype)
        local_value = value[:, block.start : block.end, :].index_select(0, kv_index)
        output.append(torch.einsum("ht,htd->hd", weights, local_value))
    return torch.stack(output, dim=0)


def aggregate_block_head_keys(
    token_scores: torch.Tensor,
    key: torch.Tensor,
    blocks: Sequence[CandidateBlock],
    temperature: float,
) -> torch.Tensor:
    """Return query-head-mapped, query-weighted pre-RoPE K representatives."""

    return aggregate_block_head_values(token_scores, key, blocks, temperature)


def norm_matched_random_values(values: torch.Tensor, seed: int) -> torch.Tensor:
    """Randomize each block/head direction while preserving its L2 norm."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_cpu = torch.randn(values.shape, generator=generator, dtype=torch.float32)
    random_values = random_cpu.to(device=values.device)
    source_norm = values.float().norm(dim=-1, keepdim=True)
    random_norm = random_values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    matched = random_values * (source_norm / random_norm)
    return matched.to(dtype=values.dtype)


def shuffled_block_values(values: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(values.shape[0], generator=generator)
    if values.shape[0] > 1:
        identity = torch.arange(values.shape[0])
        fixed = torch.nonzero(permutation == identity, as_tuple=False).flatten()
        if fixed.numel() == 1:
            index = int(fixed[0])
            other = (index + 1) % values.shape[0]
            temporary = int(permutation[index])
            permutation[index] = permutation[other]
            permutation[other] = temporary
        elif fixed.numel() > 1:
            permutation[fixed] = permutation[fixed].roll(1)
        if bool((permutation == identity).any()):
            raise RuntimeError("failed to construct a deranged shuffled-V control")
    return values.index_select(0, permutation.to(values.device)), permutation


def project_head_values(attention: torch.nn.Module, head_values: torch.Tensor) -> torch.Tensor:
    """Apply W_O to concatenated head writes while cancelling any module bias."""

    if head_values.dim() != 3:
        raise ValueError("head_values must be [blocks,query_heads,head_dim]")
    flattened = head_values.reshape(head_values.shape[0], -1)
    with torch.inference_mode():
        projected = attention.o_proj(flattened)
        zero = attention.o_proj(torch.zeros_like(flattened[:1]))
    return projected - zero


def match_projected_write_norms(
    candidate_delta: torch.Tensor, reference_delta: torch.Tensor
) -> torch.Tensor:
    """Match each block's post-W_O residual-write norm to the native write."""

    if candidate_delta.shape != reference_delta.shape or candidate_delta.dim() != 2:
        raise ValueError("projected writes must have matching [blocks,hidden_dim] shapes")
    candidate_norm_raw = candidate_delta.float().norm(dim=-1, keepdim=True)
    reference_norm = reference_delta.float().norm(dim=-1, keepdim=True)
    if bool(((candidate_norm_raw <= 1e-12) & (reference_norm > 1e-12)).any()):
        raise RuntimeError("random control projected to zero under W_O")
    candidate_norm = candidate_norm_raw.clamp_min(1e-12)
    scale = (reference_norm / candidate_norm).to(candidate_delta.dtype)
    return candidate_delta * scale


def next_layer_query_projection(layer: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Evaluate only input RMSNorm -> Q_proj -> q_norm at one layer."""

    if hidden.dim() != 2:
        raise ValueError("hidden must be [batch,hidden_dim]")
    attention = layer.self_attn
    normalized = layer.input_layernorm(hidden)
    projected = attention.q_proj(normalized)
    head_dim = int(attention.head_dim)
    shaped = projected.view(projected.shape[0], 1, -1, head_dim)
    return attention.q_norm(shaped)[:, 0]


def finite_difference_next_queries(
    layer: torch.nn.Module,
    baseline_hidden: torch.Tensor,
    delta_hidden: torch.Tensor,
    epsilon: float,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Central finite difference of next-layer pre-RoPE Q for batched writes."""

    if baseline_hidden.dim() != 2 or baseline_hidden.shape[0] != 1:
        raise ValueError("baseline_hidden must be [1,hidden_dim]")
    if delta_hidden.dim() != 2 or delta_hidden.shape[1] != baseline_hidden.shape[1]:
        raise ValueError("delta_hidden must be [blocks,hidden_dim]")
    if epsilon <= 0 or batch_size <= 0:
        raise ValueError("epsilon and batch_size must be positive")
    derivatives: list[torch.Tensor] = []
    with torch.inference_mode():
        baseline = next_layer_query_projection(layer, baseline_hidden)
        for start in range(0, delta_hidden.shape[0], batch_size):
            delta = delta_hidden[start : start + batch_size]
            hidden = baseline_hidden.expand(delta.shape[0], -1)
            plus = next_layer_query_projection(layer, hidden + epsilon * delta)
            minus = next_layer_query_projection(layer, hidden - epsilon * delta)
            derivatives.append((plus.float() - minus.float()) / (2.0 * epsilon))
    return torch.cat(derivatives, dim=0), baseline


def finite_difference_audit(
    layer: torch.nn.Module,
    baseline_hidden: torch.Tensor,
    delta_hidden: torch.Tensor,
    captured_baseline_query: torch.Tensor,
    epsilon: float,
    batch_size: int,
    epsilon_derivative: torch.Tensor | None = None,
) -> dict[str, Any]:
    count = int(delta_hidden.shape[0])
    if count <= 0:
        raise ValueError("finite-difference audit needs at least one block")
    if epsilon_derivative is None:
        first, recomputed = finite_difference_next_queries(
            layer, baseline_hidden, delta_hidden, epsilon, batch_size
        )
    else:
        if int(epsilon_derivative.shape[0]) != count:
            raise ValueError("epsilon_derivative must cover every candidate block")
        first = epsilon_derivative
        with torch.inference_mode():
            recomputed = next_layer_query_projection(layer, baseline_hidden)
    second, _ = finite_difference_next_queries(
        layer, baseline_hidden, delta_hidden, epsilon / 2.0, batch_size
    )
    difference = (first.float() - second.float()).reshape(count, -1)
    denominator = second.float().reshape(count, -1).norm(dim=-1).clamp_min(1e-12)
    relative = difference.norm(dim=-1) / denominator
    first_flat = first.float().reshape(count, -1)
    second_flat = second.float().reshape(count, -1)
    cosine = torch.nn.functional.cosine_similarity(first_flat, second_flat, dim=-1)
    captured = captured_baseline_query.to(recomputed.device).float()
    baseline_error = (recomputed.float() - captured.float()).abs().max()
    return {
        "audit_block_count": count,
        "baseline_q_reconstruction_max_abs": float(baseline_error.item()),
        "fd_halving_relative_error_max": float(relative.max().item()),
        "fd_halving_relative_error_mean": float(relative.mean().item()),
        "fd_halving_cosine_min": float(cosine.min().item()),
        "fd_halving_cosine_mean": float(cosine.mean().item()),
        "fd_halving_relative_error_by_source": relative.cpu().tolist(),
        "fd_halving_cosine_by_source": cosine.cpu().tolist(),
    }


def directed_relay_scores(
    delta_query: torch.Tensor,
    destination_key: torch.Tensor,
    destination_blocks: Sequence[CandidateBlock],
    temperature: float,
) -> torch.Tensor:
    """Score all directed source->destination block pairs."""

    if delta_query.dim() != 3 or destination_key.dim() != 3 or temperature <= 0:
        raise ValueError("invalid relay tensors or temperature")
    source_count, query_heads, dim = delta_query.shape
    kv_heads = int(destination_key.shape[0])
    if query_heads % kv_heads != 0 or destination_key.shape[-1] != dim:
        raise ValueError("incompatible GQA relay tensors")
    groups = query_heads // kv_heads
    kv_index = torch.arange(query_heads, device=delta_query.device) // groups
    columns: list[torch.Tensor] = []
    for block in destination_blocks:
        local_key = destination_key[:, block.start : block.end, :]
        local_key = local_key.to(device=delta_query.device, dtype=delta_query.dtype)
        local_key = local_key.index_select(0, kv_index)
        logits = torch.einsum("bhd,htd->bht", delta_query, local_key)
        logits = logits.float() / math.sqrt(dim)
        score = temperature * torch.logsumexp(logits / temperature, dim=-1)
        score = score - temperature * math.log(block.length)
        columns.append(score.mean(dim=-1))
    output = torch.stack(columns, dim=-1)
    if output.shape != (source_count, len(destination_blocks)):
        raise RuntimeError("directed relay matrix shape mismatch")
    return output


def key_key_similarity(
    source_keys: torch.Tensor, destination_keys: torch.Tensor
) -> torch.Tensor:
    """Mean per-query-head cosine similarity between block K representatives."""

    if source_keys.dim() != 3 or destination_keys.dim() != 3:
        raise ValueError("key representatives must be [blocks,heads,dim]")
    source = torch.nn.functional.normalize(source_keys.float(), dim=-1)
    destination = torch.nn.functional.normalize(destination_keys.float(), dim=-1)
    return torch.einsum("bhd,chd->bch", source, destination).mean(dim=-1)


def robust_matrix_for_aggregation(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("edge matrix must be square")
    count = matrix.shape[0]
    mask = ~torch.eye(count, dtype=torch.bool, device=matrix.device)
    reference = matrix[mask] if bool(mask.any()) else matrix.reshape(-1)
    median = reference.float().median()
    mad = (reference.float() - median).abs().median()
    scale = (1.4826 * mad).clamp_min(1e-6)
    return (matrix.float() - median) / scale


def aggregate_layer_matrices(
    per_layer: Mapping[int, Mapping[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    if not per_layer:
        raise ValueError("no layer matrices")
    output: dict[str, torch.Tensor] = {}
    for name in SCORE_NAMES:
        matrices = [robust_matrix_for_aggregation(values[name]) for values in per_layer.values()]
        output[name] = torch.stack(matrices, dim=0).mean(dim=0)
    return output


def binary_auroc(
    positive_scores: Sequence[float], negative_scores: Sequence[float]
) -> float | None:
    if not positive_scores or not negative_scores:
        return None
    wins = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positive_scores) * len(negative_scores))


def spans_overlap(block: CandidateBlock, span: Sequence[int]) -> bool:
    return block.start < int(span[1]) and block.end > int(span[0])


def _event_group(event: Mapping[str, Any]) -> str:
    kind = str(event["kind"])
    label = str(event["label"])
    if kind == "relevant":
        return "relevant"
    if kind == "competitor" and "_" in label:
        return f"competitor:{label.rsplit('_', 1)[0]}"
    if kind == "conflict":
        return "conflict"
    return f"{kind}:{label}"


def event_pair_set(
    blocks: Sequence[CandidateBlock], events: Sequence[Mapping[str, Any]], kind: str
) -> set[tuple[int, int]]:
    grouped: dict[str, dict[int, list[int]]] = {}
    for event in events:
        if str(event["kind"]) != kind:
            continue
        group = _event_group(event)
        step = int(event["step"])
        span = (int(event["start_token"]), int(event["end_token"]))
        hits = [index for index, block in enumerate(blocks) if spans_overlap(block, span)]
        grouped.setdefault(group, {}).setdefault(step, []).extend(hits)
    output: set[tuple[int, int]] = set()
    for steps in grouped.values():
        for source in steps.get(0, []):
            for destination in steps.get(1, []):
                if source != destination:
                    output.add((source, destination))
    return output


def choose_evaluation_pairs(
    blocks: Sequence[CandidateBlock],
    candidate_scores: torch.Tensor,
    events: Sequence[Mapping[str, Any]],
    maximum_negatives: int,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], str], dict[str, Any]]:
    """Apply labels only after edge scoring to form a matched AUROC slice."""

    positives = event_pair_set(blocks, events, "relevant")
    structured = event_pair_set(blocks, events, "conflict")
    structured |= event_pair_set(blocks, events, "competitor")
    structured -= positives
    count = len(blocks)
    gold_block_indices: set[int] = set()
    gold_hits_by_step: dict[int, set[int]] = {0: set(), 1: set()}
    for event in events:
        if str(event["kind"]) != "relevant":
            continue
        span = (int(event["start_token"]), int(event["end_token"]))
        hits = {
            index for index, block in enumerate(blocks) if spans_overlap(block, span)
        }
        gold_block_indices.update(hits)
        gold_hits_by_step.setdefault(int(event["step"]), set()).update(hits)
    pool = [
        (source, destination)
        for source in range(count)
        for destination in range(count)
        if source != destination
        and (source, destination) not in positives
        and source not in gold_block_indices
        and destination not in gold_block_indices
    ]
    selected: dict[tuple[int, int], str] = {
        pair: "structured_record" for pair in sorted(structured) if pair in pool
    }
    target_count = min(max(1, int(maximum_negatives)), len(pool)) if pool else 0
    positive_features: list[torch.Tensor] = []
    scale = max(1.0, float(blocks[-1].end - blocks[0].start)) if blocks else 1.0
    for source, destination in positives:
        positive_features.append(
            torch.tensor(
                [
                    float(candidate_scores[source]),
                    float(candidate_scores[destination]),
                    math.log1p(abs(blocks[destination].center - blocks[source].center) / scale),
                ]
            )
        )
    if positive_features and len(selected) < target_count:
        target = torch.stack(positive_features).mean(dim=0)
        ranked: list[tuple[float, tuple[int, int]]] = []
        for pair in pool:
            if pair in selected:
                continue
            source, destination = pair
            feature = torch.tensor(
                [
                    float(candidate_scores[source]),
                    float(candidate_scores[destination]),
                    math.log1p(abs(blocks[destination].center - blocks[source].center) / scale),
                ]
            )
            ranked.append((float((feature - target).square().sum().item()), pair))
        ranked.sort(key=lambda item: (item[0], item[1]))
        for _, pair in ranked[: target_count - len(selected)]:
            selected[pair] = "prescore_distance_match"
    metadata = {
        "positive_pair_count": len(positives),
        "structured_negative_available": len(structured),
        "matched_negative_count": len(selected),
        "gold_candidate_block_count": len(gold_block_indices),
        "gold_hop0_candidate_covered": int(bool(gold_hits_by_step.get(0))),
        "gold_hop1_candidate_covered": int(bool(gold_hits_by_step.get(1))),
        "gold_both_candidate_covered": int(
            bool(gold_hits_by_step.get(0)) and bool(gold_hits_by_step.get(1))
        ),
        "gold_edge_resolved": int(bool(positives)),
    }
    return positives, selected, metadata


def _tensor_signature(tensor: torch.Tensor, sequence_chunk: int = 4096) -> dict[str, Any]:
    """Full-byte, chunked signature used for prefix immutability auditing."""

    total = 0.0
    absolute = 0.0
    squared = 0.0
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    if tensor.dim() >= 3:
        sequence_dim = tensor.dim() - 2
        sequence = int(tensor.shape[sequence_dim])
        for start in range(0, sequence, sequence_chunk):
            selectors = [slice(None)] * tensor.dim()
            selectors[sequence_dim] = slice(start, min(sequence, start + sequence_chunk))
            # ``DynamicCache`` truncation returns a strided view.  Force the
            # same contiguous layout before reduction so an append+truncate
            # does not create a false immutability failure from a different
            # floating-point reduction order.
            chunk = tensor[tuple(selectors)].detach().contiguous().cpu()
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
            numeric = chunk.float()
            total += float(numeric.sum().item())
            absolute += float(numeric.abs().sum().item())
            squared += float(numeric.square().sum().item())
    else:
        chunk = tensor.detach().contiguous().cpu()
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
        numeric = chunk.float()
        total = float(numeric.sum().item())
        absolute = float(numeric.abs().sum().item())
        squared = float(numeric.square().sum().item())
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sum": total,
        "abs_sum": absolute,
        "square_sum": squared,
        "content_sha256": digest.hexdigest(),
    }


def cache_prefix_fingerprint(
    cache: Any, layers: Sequence[int], prefix_length: int
) -> dict[str, Any]:
    if not hasattr(cache, "key_cache") or not hasattr(cache, "value_cache"):
        raise TypeError("KVQ relay probe requires DynamicCache key_cache/value_cache")
    output: dict[str, Any] = {"prefix_length": int(prefix_length), "layers": {}}
    for layer in layers:
        key = cache.key_cache[int(layer)][:, :, :prefix_length, :]
        value = cache.value_cache[int(layer)][:, :, :prefix_length, :]
        output["layers"][str(layer)] = {
            "key": _tensor_signature(key),
            "value": _tensor_signature(value),
        }
    return output


def reset_dynamic_cache(cache: Any, prefix_length: int) -> None:
    if not hasattr(cache, "key_cache") or not hasattr(cache, "value_cache"):
        raise TypeError("KVQ relay probe requires DynamicCache")
    for index in range(len(cache.key_cache)):
        cache.key_cache[index] = cache.key_cache[index][:, :, :prefix_length, :]
        cache.value_cache[index] = cache.value_cache[index][:, :, :prefix_length, :]
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = int(prefix_length)


@dataclass
class CaptureState:
    source_layers: tuple[int, ...]
    capture_layers: tuple[int, ...]
    storage_device: str
    mode: str = "idle"
    pre_key_chunks: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    pre_keys: dict[int, torch.Tensor] = field(default_factory=dict)
    query_pre: dict[int, torch.Tensor] = field(default_factory=dict)
    layer_inputs: dict[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pre_key_chunks = {layer: [] for layer in self.capture_layers}


def _make_key_capture_hook(state: CaptureState, layer_index: int):
    def hook(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if state.mode != "prefix":
            return
        state.pre_key_chunks[layer_index].append(
            output.detach().transpose(1, 2).contiguous().to(state.storage_device)
        )

    return hook


def _make_query_capture_hook(state: CaptureState, layer_index: int):
    def hook(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if state.mode != "query":
            return
        state.query_pre[layer_index] = output.detach()[:, -1].contiguous().clone()

    return hook


def _make_layer_input_hook(state: CaptureState, layer_index: int):
    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if state.mode != "query":
            return
        hidden = inputs[0]
        state.layer_inputs[layer_index] = hidden.detach()[:, -1].contiguous().clone()

    return hook


def decoder_layers(model: Any) -> Sequence[torch.nn.Module]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("expected Qwen3ForCausalLM.model.layers")
    return layers


@contextmanager
def install_capture_hooks(model: Any, state: CaptureState) -> Iterator[None]:
    layers = decoder_layers(model)
    handles: list[Any] = []
    for layer_index in state.capture_layers:
        layer = layers[int(layer_index)]
        attention = layer.self_attn
        handles.append(
            attention.k_norm.register_forward_hook(
                _make_key_capture_hook(state, int(layer_index))
            )
        )
        handles.append(
            attention.q_norm.register_forward_hook(
                _make_query_capture_hook(state, int(layer_index))
            )
        )
        handles.append(
            layer.register_forward_pre_hook(
                _make_layer_input_hook(state, int(layer_index))
            )
        )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def finalize_prefix_keys(state: CaptureState, expected_length: int) -> None:
    for layer in state.capture_layers:
        chunks = state.pre_key_chunks[layer]
        if not chunks:
            raise RuntimeError(f"layer {layer} captured no prefix pre-RoPE keys")
        keys = torch.cat(chunks, dim=2)
        if int(keys.shape[2]) != int(expected_length):
            raise RuntimeError(
                f"layer {layer} captured {keys.shape[2]} keys, expected {expected_length}"
            )
        state.pre_keys[layer] = keys
        state.pre_key_chunks[layer] = []


def build_case(
    tokenizer: Any,
    *,
    target_context_tokens: int,
    seed: int,
    condition: str,
    placement: str,
    code_mode: str,
) -> dict[str, Any]:
    """Reuse the existing controlled two-hop record generator unchanged."""

    body = causal.build_body(
        tokenizer,
        seed=int(seed),
        target_context_tokens=int(target_context_tokens),
        condition=str(condition),
        placement=str(placement),
        code_mode=str(code_mode),
    )
    start_code, steps, gold_answer = causal.query_parameters(body, "full2")
    suffix = causal.build_suffix("chat_concise", start_code, steps, "full2")
    wrapper_prefix, wrapper_suffix = causal.chat_wrapper_ids(tokenizer)
    body_ids = body["body_ids"][0].tolist()
    suffix_ids = base.token_ids(tokenizer, suffix)
    prompt_ids = wrapper_prefix + body_ids + suffix_ids + wrapper_suffix
    offset = len(wrapper_prefix)
    events: list[dict[str, Any]] = []
    for event in body["events"]:
        item = asdict(event)
        item["start_token"] = offset + int(event.start_token)
        item["end_token"] = offset + int(event.end_token)
        events.append(item)
    return {
        "prompt_ids": prompt_ids,
        "body_start": offset,
        "body_end": offset + len(body_ids),
        "target_context_tokens": int(target_context_tokens),
        "seed": int(seed),
        "condition": str(condition),
        "placement": str(placement),
        "code_mode": str(code_mode),
        "gold_answer": gold_answer,
        "gold_codes": list(body["gold_codes"]),
        "events": events,
    }


def public_case(case: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in case.items() if key != "prompt_ids"}


def _cache_value(cache: Any, layer: int, prefix_length: int) -> torch.Tensor:
    return cache.value_cache[int(layer)][0, :, :prefix_length, :]


def _candidate_score_vector(
    aggregate_all_blocks: torch.Tensor,
    all_blocks: Sequence[CandidateBlock],
    selected_blocks: Sequence[CandidateBlock],
) -> torch.Tensor:
    index_by_start = {block.start: block.index for block in all_blocks}
    return torch.tensor(
        [float(aggregate_all_blocks[index_by_start[block.start]]) for block in selected_blocks],
        dtype=torch.float32,
    )


def _layer_edge_matrices(
    *,
    layer_index: int,
    layers: Sequence[torch.nn.Module],
    cache: Any,
    state: CaptureState,
    selected_blocks: Sequence[CandidateBlock],
    token_scores: Mapping[int, torch.Tensor],
    block_scores: Mapping[int, torch.Tensor],
    prefix_length: int,
    block_temperature: float,
    value_temperature: float,
    fd_epsilon: float,
    fd_batch_size: int,
    random_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, torch.Tensor]]:
    source_attention = layers[layer_index].self_attn
    next_layer = layers[layer_index + 1]
    source_value = _cache_value(cache, layer_index, prefix_length)
    head_values = aggregate_block_head_values(
        token_scores[layer_index], source_value, selected_blocks, value_temperature
    )
    shuffled, permutation = shuffled_block_values(head_values, random_seed + 17)
    randomized = norm_matched_random_values(head_values, random_seed + 31)

    delta_native = project_head_values(source_attention, head_values)
    delta_shuffled = project_head_values(source_attention, shuffled)
    delta_random = project_head_values(source_attention, randomized)
    delta_random = match_projected_write_norms(delta_random, delta_native)
    hidden = state.layer_inputs[layer_index + 1]
    destination_key = state.pre_keys[layer_index + 1][0]
    captured_next_query = state.query_pre[layer_index + 1]

    dq_native, recomputed = finite_difference_next_queries(
        next_layer, hidden, delta_native.to(hidden.device), fd_epsilon, fd_batch_size
    )
    dq_shuffled, _ = finite_difference_next_queries(
        next_layer, hidden, delta_shuffled.to(hidden.device), fd_epsilon, fd_batch_size
    )
    dq_random, _ = finite_difference_next_queries(
        next_layer, hidden, delta_random.to(hidden.device), fd_epsilon, fd_batch_size
    )
    native_edge = directed_relay_scores(
        dq_native, destination_key, selected_blocks, block_temperature
    ).cpu()
    shuffled_edge = directed_relay_scores(
        dq_shuffled, destination_key, selected_blocks, block_temperature
    ).cpu()
    random_edge = directed_relay_scores(
        dq_random, destination_key, selected_blocks, block_temperature
    ).cpu()

    source_key_representatives = aggregate_block_head_keys(
        token_scores[layer_index],
        state.pre_keys[layer_index][0],
        selected_blocks,
        value_temperature,
    )
    destination_key_representatives = aggregate_block_head_keys(
        token_scores[layer_index + 1],
        state.pre_keys[layer_index + 1][0],
        selected_blocks,
        value_temperature,
    )
    kk = key_key_similarity(source_key_representatives, destination_key_representatives)
    source_pre = block_scores[layer_index].mean(dim=0)
    destination_pre = block_scores[layer_index + 1].mean(dim=0)
    pre_pair = source_pre[:, None] + destination_pre[None, :]
    matrices = {
        "kvq_relay": native_edge,
        "shuffled_v_relay": shuffled_edge,
        "norm_matched_random_v_relay": random_edge,
        "reverse_edge_relay": native_edge.transpose(0, 1).contiguous(),
        "kk_similarity": kk.cpu(),
        "pre_score_pair": pre_pair.cpu(),
    }
    audit = finite_difference_audit(
        next_layer,
        hidden,
        delta_native.to(hidden.device),
        captured_next_query,
        fd_epsilon,
        fd_batch_size,
        epsilon_derivative=dq_native,
    )
    baseline_error = (
        recomputed.float() - captured_next_query.to(recomputed.device).float()
    ).abs().max()
    audit.update(
        {
            "layer": int(layer_index),
            "source_value_norm_mean": float(head_values.float().norm(dim=-1).mean().item()),
            "delta_hidden_norm_mean": float(delta_native.float().norm(dim=-1).mean().item()),
            "random_value_norm_max_abs_error": float(
                (randomized.float().norm(dim=-1) - head_values.float().norm(dim=-1))
                .abs()
                .max()
                .item()
            ),
            "random_delta_hidden_norm_max_abs_error": float(
                (
                    delta_random.float().norm(dim=-1)
                    - delta_native.float().norm(dim=-1)
                )
                .abs()
                .max()
                .item()
            ),
            "shuffle_permutation": permutation.tolist(),
            "baseline_q_reconstruction_max_abs": float(baseline_error.item()),
        }
    )
    relay_tensors = {
        "source_block_head_values": head_values.detach().cpu(),
        "source_delta_hidden": delta_native.detach().cpu(),
        "delta_query_next": dq_native.detach().cpu(),
        "source_key_representatives": source_key_representatives.detach().cpu(),
        "destination_key_representatives": destination_key_representatives.detach().cpu(),
        "baseline_hidden_next": hidden.detach().cpu(),
        "baseline_query_next": captured_next_query.detach().cpu(),
    }
    return matrices, audit, relay_tensors


def score_case_label_free(
    *,
    model: Any,
    cache: Any,
    state: CaptureState,
    case_seed: int,
    target_context_tokens: int,
    all_blocks: Sequence[CandidateBlock],
    maximum_blocks: int,
    block_temperature: float,
    value_temperature: float,
    fd_epsilon: float,
    fd_batch_size: int,
) -> dict[str, Any]:
    """Produce all candidate and edge scores without accepting label arguments."""

    token_scores: dict[int, torch.Tensor] = {}
    all_block_scores: dict[int, torch.Tensor] = {}
    for layer in state.capture_layers:
        query = state.query_pre[layer][0].detach().cpu()
        key = state.pre_keys[layer][0].detach().cpu()
        token_scores[layer] = gqa_query_key_scores(query, key)
        all_block_scores[layer] = block_logmeanexp_scores(
            token_scores[layer], all_blocks, block_temperature
        )
    selected, aggregate_all = select_candidate_blocks(
        {layer: all_block_scores[layer] for layer in state.source_layers},
        all_blocks,
        maximum_blocks,
    )
    candidate_scores = _candidate_score_vector(aggregate_all, all_blocks, selected)
    selected_block_scores = {
        layer: block_logmeanexp_scores(token_scores[layer], selected, block_temperature)
        for layer in state.capture_layers
    }
    layers = decoder_layers(model)
    per_layer: dict[int, dict[str, torch.Tensor]] = {}
    relay_tensors: dict[int, dict[str, torch.Tensor]] = {}
    audits: list[dict[str, Any]] = []
    random_base = 8_103_001 + int(case_seed) * 1009 + int(target_context_tokens)
    for layer in state.source_layers:
        matrices, audit, tensors = _layer_edge_matrices(
            layer_index=layer,
            layers=layers,
            cache=cache,
            state=state,
            selected_blocks=selected,
            token_scores=token_scores,
            block_scores=selected_block_scores,
            prefix_length=int(state.pre_keys[layer].shape[2]),
            block_temperature=block_temperature,
            value_temperature=value_temperature,
            fd_epsilon=fd_epsilon,
            fd_batch_size=fd_batch_size,
            random_seed=random_base + layer * 97,
        )
        per_layer[layer] = matrices
        relay_tensors[layer] = tensors
        audits.append(audit)
    aggregate = aggregate_layer_matrices(per_layer)
    return {
        "all_block_count": len(all_blocks),
        "candidate_blocks": selected,
        "candidate_scores": candidate_scores,
        "per_layer": per_layer,
        "relay_tensors": relay_tensors,
        "aggregate": aggregate,
        "finite_difference_audits": audits,
    }


def _edge_rows_for_case(
    *,
    case: Mapping[str, Any],
    scored: Mapping[str, Any],
    maximum_negatives: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blocks: list[CandidateBlock] = list(scored["candidate_blocks"])
    positives, negatives, evaluation = choose_evaluation_pairs(
        blocks,
        scored["candidate_scores"],
        case["events"],
        maximum_negatives,
    )
    selected_pairs = sorted(positives | set(negatives))
    rows: list[dict[str, Any]] = []
    matrices_by_layer = {
        **{int(layer): matrices for layer, matrices in scored["per_layer"].items()},
        -1: scored["aggregate"],
    }
    for layer, matrices in matrices_by_layer.items():
        for source, destination in selected_pairs:
            row: dict[str, Any] = {
                "target_context_tokens": int(case["target_context_tokens"]),
                "seed": int(case["seed"]),
                "condition": str(case["condition"]),
                "layer": int(layer),
                "source_candidate_index": int(source),
                "destination_candidate_index": int(destination),
                "source_start": blocks[source].start,
                "source_end": blocks[source].end,
                "destination_start": blocks[destination].start,
                "destination_end": blocks[destination].end,
                "label": 1 if (source, destination) in positives else 0,
                "negative_match_type": negatives.get((source, destination), ""),
                "source_candidate_score": float(scored["candidate_scores"][source]),
                "destination_candidate_score": float(scored["candidate_scores"][destination]),
            }
            for name in SCORE_NAMES:
                row[name] = float(matrices[name][source, destination])
            rows.append(row)
    aggregate_rows = [row for row in rows if int(row["layer"]) == -1]
    positive_count = sum(int(row["label"]) == 1 for row in aggregate_rows)
    negative_count = sum(int(row["label"]) == 0 for row in aggregate_rows)
    evaluation["auroc_valid"] = int(positive_count > 0 and negative_count > 0)
    if positive_count == 0:
        evaluation["auroc_invalid_reason"] = "gold_edge_not_resolved"
    elif negative_count == 0:
        evaluation["auroc_invalid_reason"] = "no_matched_negative"
    else:
        evaluation["auroc_invalid_reason"] = None
    for name in SCORE_NAMES:
        positive_values = [float(row[name]) for row in aggregate_rows if int(row["label"]) == 1]
        negative_values = [float(row[name]) for row in aggregate_rows if int(row["label"]) == 0]
        evaluation[f"{name}_auroc"] = binary_auroc(positive_values, negative_values)
    return rows, evaluation


def _raw_artifact(
    case: Mapping[str, Any],
    scored: Mapping[str, Any],
    state: CaptureState,
    prefix_audit: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case": public_case(case),
        "candidate_blocks": [
            {
                "candidate_local_index": local_index,
                "label_free_block_index": block.index,
                "start": block.start,
                "end": block.end,
            }
            for local_index, block in enumerate(scored["candidate_blocks"])
        ],
        "candidate_scores": scored["candidate_scores"].cpu(),
        "per_layer": {
            int(layer): {name: matrix.cpu() for name, matrix in matrices.items()}
            for layer, matrices in scored["per_layer"].items()
        },
        "relay_tensors": {
            int(layer): {name: tensor.cpu() for name, tensor in tensors.items()}
            for layer, tensors in scored["relay_tensors"].items()
        },
        "final_query_pre_q": {
            int(layer): tensor.detach().cpu()
            for layer, tensor in state.query_pre.items()
        },
        "baseline_layer_inputs": {
            int(layer): tensor.detach().cpu()
            for layer, tensor in state.layer_inputs.items()
        },
        "aggregate": {name: matrix.cpu() for name, matrix in scored["aggregate"].items()},
        "finite_difference_audits": scored["finite_difference_audits"],
        "prefix_immutable_audit": dict(prefix_audit),
        "evaluation": dict(evaluation),
        "label_use_boundary": (
            "case.events were passed only to _edge_rows_for_case after all candidate, "
            "control, and aggregate score matrices were frozen"
        ),
    }


def _seed_cluster_bootstrap_ci(
    seed_values: Sequence[tuple[int, float]], bootstrap_samples: int = 2000
) -> tuple[float | None, float | None]:
    """Bootstrap seeds as clusters so repeated lengths never become pseudo-seeds."""

    grouped: dict[int, list[float]] = {}
    for seed, value in seed_values:
        grouped.setdefault(int(seed), []).append(float(value))
    seeds = sorted(grouped)
    if not seeds:
        return None, None
    if len(seeds) == 1:
        value = statistics.fmean(grouped[seeds[0]])
        return value, value
    seed_material = json.dumps(seed_values, separators=(",", ":")).encode("utf-8")
    generator = random.Random(int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big"))
    estimates: list[float] = []
    for _ in range(int(bootstrap_samples)):
        sampled = [seeds[generator.randrange(len(seeds))] for _ in seeds]
        values = [value for seed in sampled for value in grouped[seed]]
        estimates.append(statistics.fmean(values))
    estimates.sort()
    low = estimates[int(0.025 * (len(estimates) - 1))]
    high = estimates[int(0.975 * (len(estimates) - 1))]
    return low, high


def summarize_edge_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Macro-average per-case AUROC; never pool incomparable scores across cases."""

    aggregate = [row for row in rows if int(row["layer"]) == -1]
    grouped_rows: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    for row in aggregate:
        key = (
            int(row["target_context_tokens"]),
            str(row["condition"]),
            int(row["seed"]),
        )
        grouped_rows.setdefault(key, []).append(row)

    case_metrics: list[dict[str, Any]] = []
    for (length, condition, seed), case_rows in sorted(grouped_rows.items()):
        positives = [row for row in case_rows if int(row["label"]) == 1]
        negatives = [row for row in case_rows if int(row["label"]) == 0]
        fd_pass = all(
            int(row.get("finite_difference_audit_pass", 1)) == 1 for row in case_rows
        )
        valid = bool(fd_pass and positives and negatives)
        item: dict[str, Any] = {
            "target_context_tokens": length,
            "condition": condition,
            "seed": seed,
            "valid": int(valid),
            "positive_edge_count": len(positives),
            "matched_negative_edge_count": len(negatives),
        }
        for name in SCORE_NAMES:
            item[name] = (
                binary_auroc(
                    [float(row[name]) for row in positives],
                    [float(row[name]) for row in negatives],
                )
                if valid
                else None
            )
        case_metrics.append(item)

    keys = sorted(
        {(row["target_context_tokens"], row["condition"]) for row in case_metrics}
    )
    output: list[dict[str, Any]] = []
    for length, condition in [*keys, (-1, "all")]:
        selected = [
            row
            for row in case_metrics
            if length == -1
            or (
                int(row["target_context_tokens"]) == length
                and str(row["condition"]) == condition
            )
        ]
        if not selected:
            continue
        valid_cases = [row for row in selected if int(row["valid"]) == 1]
        item: dict[str, Any] = {
            "target_context_tokens": length,
            "condition": condition,
            "case_with_edge_rows_count": len(selected),
            "valid_case_auroc_count": len(valid_cases),
            "invalid_case_auroc_count": len(selected) - len(valid_cases),
            "positive_edge_count": sum(int(row["positive_edge_count"]) for row in valid_cases),
            "matched_negative_edge_count": sum(
                int(row["matched_negative_edge_count"]) for row in valid_cases
            ),
            "aggregation": "per-case AUROC macro mean; seed-cluster bootstrap",
        }
        for name in SCORE_NAMES:
            seed_values = [
                (int(row["seed"]), float(row[name]))
                for row in valid_cases
                if row[name] is not None
            ]
            values = [value for _, value in seed_values]
            mean = statistics.fmean(values) if values else None
            low, high = _seed_cluster_bootstrap_ci(seed_values)
            item[f"{name}_auroc"] = mean
            item[f"{name}_auroc_macro_mean"] = mean
            item[f"{name}_auroc_seed_bootstrap_ci_low"] = low
            item[f"{name}_auroc_seed_bootstrap_ci_high"] = high
        output.append(item)
    return output


def summarize_case_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (int(row["target_context_tokens"]), str(row["condition"]))
            for row in rows
        }
    )
    output: list[dict[str, Any]] = []
    for length, condition in [*keys, (-1, "all")]:
        selected = [
            row
            for row in rows
            if length == -1
            or (
                int(row["target_context_tokens"]) == length
                and str(row["condition"]) == condition
            )
        ]
        if not selected:
            continue
        output.append(
            {
                "target_context_tokens": length,
                "condition": condition,
                "case_count": len(selected),
                "gold_both_candidate_coverage": statistics.fmean(
                    int(row["gold_both_candidate_covered"]) for row in selected
                ),
                "gold_both_candidate_missing_rate": statistics.fmean(
                    1 - int(row["gold_both_candidate_covered"]) for row in selected
                ),
                "gold_edge_resolution_rate": statistics.fmean(
                    int(row["gold_edge_resolved"]) for row in selected
                ),
                "gold_edge_missing_rate": statistics.fmean(
                    1 - int(row["gold_edge_resolved"]) for row in selected
                ),
                "case_auroc_valid_rate": statistics.fmean(
                    int(row.get("auroc_valid", 0)) for row in selected
                ),
                "finite_difference_audit_pass_rate": statistics.fmean(
                    int(row["finite_difference_audit_pass"]) for row in selected
                ),
                "candidate_block_count_mean": statistics.fmean(
                    int(row["candidate_block_count"]) for row in selected
                ),
                "prefill_seconds_mean": statistics.fmean(
                    float(row["prefill_seconds"]) for row in selected
                ),
                "edge_score_seconds_mean": statistics.fmean(
                    float(row["edge_score_seconds"]) for row in selected
                ),
            }
        )
    return output


def case_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return (
        int(row["target_context_tokens"]),
        str(row["condition"]),
        int(row["seed"]),
    )


def case_stem(key: tuple[int, str, int]) -> str:
    length, condition, seed = key
    safe_condition = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in condition
    )
    return f"length_{length}_condition_{safe_condition}_seed_{seed}"


def expected_case_keys(config: Mapping[str, Any]) -> set[tuple[int, str, int]]:
    return {
        (int(length), str(condition), int(seed))
        for length in config["resolved_lengths"]
        for condition in config["resolved_conditions"]
        for seed in range(
            int(config["seed_start"]),
            int(config["seed_start"]) + int(config["num_seeds"]),
        )
    }


def collect_committed_cases(
    output_dir: Path,
    full_config_hash: str,
    expected: set[tuple[int, str, int]] | None = None,
    require_complete: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load only case transactions whose commit marker was written last."""

    cases_root = output_dir / "cases"
    case_rows_by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    edge_rows: list[dict[str, Any]] = []
    for done_path in sorted(cases_root.glob("*/done.json")) if cases_root.exists() else []:
        directory = done_path.parent
        done = read_json(done_path)
        if str(done.get("full_config_hash")) != str(full_config_hash):
            raise RuntimeError(f"incompatible committed case config: {done_path}")
        required = (
            directory / "case_row.json",
            directory / "edge_rows.json",
            directory / "audit.json",
            directory / "raw.pt",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"committed case is incomplete: {missing}")
        row = read_json(directory / "case_row.json")
        key = case_key(row)
        marker_key = (
            int(done["target_context_tokens"]),
            str(done["condition"]),
            int(done["seed"]),
        )
        if key != marker_key or directory.name != case_stem(key):
            raise RuntimeError(f"case transaction identity mismatch: {directory}")
        if key in case_rows_by_key:
            raise RuntimeError(f"duplicate committed case: {key}")
        local_edges = read_json(directory / "edge_rows.json")
        if not isinstance(local_edges, list) or any(case_key(edge) != key for edge in local_edges):
            raise RuntimeError(f"edge rows do not match committed case: {directory}")
        case_rows_by_key[key] = row
        edge_rows.extend(local_edges)

    observed = set(case_rows_by_key)
    if expected is not None:
        unexpected = observed - expected
        missing = expected - observed
        if unexpected:
            raise RuntimeError(f"unexpected committed cases: {sorted(unexpected)}")
        if require_complete and missing:
            raise RuntimeError(f"missing committed cases: {sorted(missing)}")
    case_rows = [case_rows_by_key[key] for key in sorted(case_rows_by_key)]
    edge_rows.sort(
        key=lambda row: (
            *case_key(row),
            int(row["layer"]),
            int(row["source_candidate_index"]),
            int(row["destination_candidate_index"]),
        )
    )
    return case_rows, edge_rows


def commit_case_transaction(
    output_dir: Path,
    full_config_hash: str,
    case_row: Mapping[str, Any],
    edge_rows: Sequence[dict[str, Any]],
    audit: Mapping[str, Any],
    raw_artifact: Mapping[str, Any],
) -> None:
    key = case_key(case_row)
    directory = output_dir / "cases" / case_stem(key)
    directory.mkdir(parents=True, exist_ok=True)
    done_path = directory / "done.json"
    if done_path.exists():
        done_path.unlink()
    atomic_torch_save(directory / "raw.pt", raw_artifact)
    write_json(directory / "audit.json", audit)
    write_json(directory / "edge_rows.json", list(edge_rows))
    write_json(directory / "case_row.json", dict(case_row))
    write_json(
        done_path,
        {
            "target_context_tokens": key[0],
            "condition": key[1],
            "seed": key[2],
            "full_config_hash": full_config_hash,
        },
    )


def write_aggregate_outputs(
    output_dir: Path,
    case_rows: Sequence[dict[str, Any]],
    edge_rows: Sequence[dict[str, Any]],
) -> None:
    write_jsonl(output_dir / "case_rows.jsonl", case_rows)
    write_jsonl(output_dir / "edge_rows.jsonl", edge_rows)
    write_csv(output_dir / "case_rows.csv", case_rows)
    write_csv(output_dir / "edge_rows.csv", edge_rows)
    summary = summarize_edge_rows(edge_rows)
    case_summary = summarize_case_rows(case_rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "case_summary.csv", case_summary)
    write_json(output_dir / "case_summary.json", case_summary)


def merge_shards(output_dir: Path, shard_paths: Sequence[str]) -> None:
    if not shard_paths:
        raise ValueError("--merge-shards requires at least one shard")
    case_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    method_hash: str | None = None
    observed_keys: set[tuple[int, str, int]] = set()
    shard_manifest: list[dict[str, Any]] = []
    for raw in shard_paths:
        shard = Path(raw).resolve()
        if not (shard / "done.txt").exists() or not (shard / "config.json").exists():
            raise RuntimeError(f"shard is not complete: {shard}")
        config = read_json(shard / "config.json")
        validate_run_config_hashes(config)
        current_method_hash = str(config.get("method_config_hash", ""))
        full_hash = str(config.get("full_config_hash", ""))
        if not current_method_hash or not full_hash:
            raise RuntimeError(f"shard lacks versioned config hashes: {shard}")
        if method_hash is None:
            method_hash = current_method_hash
        elif current_method_hash != method_hash:
            raise RuntimeError("cannot merge shards with different method configs")
        expected = expected_case_keys(config)
        local_cases, local_edges = collect_committed_cases(
            shard, full_hash, expected=expected, require_complete=True
        )
        local_keys = {case_key(row) for row in local_cases}
        overlap = observed_keys & local_keys
        if overlap:
            raise RuntimeError(f"duplicate case keys across shards: {sorted(overlap)}")
        observed_keys.update(local_keys)
        case_rows.extend(local_cases)
        edge_rows.extend(local_edges)
        shard_manifest.append(
            {
                "path": str(shard),
                "full_config_hash": full_hash,
                "case_count": len(local_cases),
                "raw_case_directories": [
                    str(shard / "cases" / case_stem(case_key(row))) for row in local_cases
                ],
            }
        )
    seed_slices: dict[tuple[int, str], set[int]] = {}
    for length, condition, seed in observed_keys:
        seed_slices.setdefault((length, condition), set()).add(seed)
    for slice_key, seeds in seed_slices.items():
        contiguous = set(range(min(seeds), max(seeds) + 1))
        if seeds != contiguous:
            raise RuntimeError(
                f"non-contiguous merged seed coverage for {slice_key}: {sorted(seeds)}"
            )
    case_rows.sort(key=case_key)
    edge_rows.sort(
        key=lambda row: (
            *case_key(row),
            int(row["layer"]),
            int(row["source_candidate_index"]),
            int(row["destination_candidate_index"]),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    done_path = output_dir / "done.txt"
    if done_path.exists():
        done_path.unlink()
    write_aggregate_outputs(output_dir, case_rows, edge_rows)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "method_config_hash": method_hash,
        "case_count": len(case_rows),
        "unique_case_key_count": len(observed_keys),
        "shards": shard_manifest,
    }
    write_json(output_dir / "config.json", manifest)
    write_json(output_dir / "manifest.json", manifest)
    temporary_done = done_path.with_suffix(".txt.tmp")
    temporary_done.write_text("ok\n", encoding="utf-8")
    temporary_done.replace(done_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify controlled two-hop candidate-block edges with a frozen-KV "
            "K->V->next-Q->K relay diagnostic. No sparse consumer is run."
        )
    )
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merge-shards", default="")
    parser.add_argument("--lengths", default="8192,16384,32768")
    parser.add_argument("--conditions", default="mixed")
    parser.add_argument("--placement", choices=causal.PLACEMENTS, default="prefix")
    parser.add_argument(
        "--code-mode",
        choices=("legacy", "single_token", "english_single_token"),
        default="english_single_token",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--relay-layers", default="")
    parser.add_argument("--max-block-tokens", type=int, default=64)
    parser.add_argument("--max-candidate-blocks", type=int, default=64)
    parser.add_argument("--maximum-matched-negatives", type=int, default=32)
    parser.add_argument("--block-temperature", type=float, default=1.0)
    parser.add_argument("--value-temperature", type=float, default=1.0)
    parser.add_argument("--fd-epsilon", type=float, default=0.05)
    parser.add_argument("--fd-batch-size", type=int, default=8)
    parser.add_argument("--fd-audit-relative-tolerance", type=float, default=0.35)
    parser.add_argument("--fd-audit-cosine-tolerance", type=float, default=0.90)
    parser.add_argument("--baseline-q-max-abs-tolerance", type=float, default=1e-4)
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--global-max-position", type=int, default=70000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[int], list[str], list[int]]:
    lengths = sorted(set(parse_int_csv(args.lengths)))
    conditions = parse_str_csv(args.conditions)
    unknown = sorted(set(conditions) - set(causal.CONDITIONS))
    explicit_layers = parse_int_csv(args.relay_layers)
    if not lengths or min(lengths) <= 0:
        raise ValueError("positive --lengths are required")
    if not conditions or unknown:
        raise ValueError(f"invalid conditions; unknown={unknown}")
    if explicit_layers and len(set(explicit_layers)) != 4:
        raise ValueError("--relay-layers must specify exactly four unique source layers")
    if args.seed_start < 0 or args.num_seeds <= 0:
        raise ValueError("--seed-start must be non-negative and --num-seeds positive")
    positive_names = (
        "max_block_tokens",
        "max_candidate_blocks",
        "maximum_matched_negatives",
        "block_temperature",
        "value_temperature",
        "fd_epsilon",
        "fd_batch_size",
        "fd_audit_relative_tolerance",
        "baseline_q_max_abs_tolerance",
        "prefill_chunk_size",
    )
    if any(float(getattr(args, name)) <= 0 for name in positive_names):
        raise ValueError("block, finite-difference, and chunk settings must be positive")
    if args.max_candidate_blocks > 64:
        raise ValueError("the first-stage protocol hard-caps candidates at 64 blocks")
    if not -1.0 <= args.fd_audit_cosine_tolerance <= 1.0:
        raise ValueError("fd_audit_cosine_tolerance must be in [-1, 1]")
    return lengths, conditions, explicit_layers


def _model_config(args: argparse.Namespace) -> argparse.Namespace:
    return args


def build_run_config(
    args: argparse.Namespace,
    lengths: Sequence[int],
    conditions: Sequence[str],
    pre_key_storage: str,
) -> dict[str, Any]:
    excluded = {
        "output_dir",
        "merge_shards",
        "seed_start",
        "num_seeds",
        "dry_run",
    }
    method_config = {
        "protocol_version": PROTOCOL_VERSION,
        **{key: value for key, value in vars(args).items() if key not in excluded},
        "resolved_lengths": list(lengths),
        "resolved_conditions": list(conditions),
        "pre_key_storage": pre_key_storage,
        "candidate_and_score_use_labels": False,
        "consumer": "none; edge diagnostic only",
        "finite_difference_scope": "next-layer input RMSNorm -> Q_proj -> q_norm",
        "finite_difference_audit_scope": "all selected candidate blocks",
        "candidate_cap": 64,
    }
    method_hash = stable_json_hash(method_config)
    full_payload = {
        "method_config_hash": method_hash,
        "seed_start": int(args.seed_start),
        "num_seeds": int(args.num_seeds),
    }
    return {
        **method_config,
        "method_config": method_config,
        "method_config_hash": method_hash,
        **full_payload,
        "full_config_hash": stable_json_hash(full_payload),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def validate_run_config_hashes(config: Mapping[str, Any]) -> None:
    method_config = config.get("method_config")
    if not isinstance(method_config, Mapping):
        raise RuntimeError("config.json lacks the canonical method_config payload")
    method_hash = stable_json_hash(method_config)
    if method_hash != str(config.get("method_config_hash", "")):
        raise RuntimeError("method_config_hash does not match config contents")
    full_payload = {
        "method_config_hash": method_hash,
        "seed_start": int(config["seed_start"]),
        "num_seeds": int(config["num_seeds"]),
    }
    if stable_json_hash(full_payload) != str(config.get("full_config_hash", "")):
        raise RuntimeError("full_config_hash does not match shard assignment")


def initialize_or_validate_run_config(output_dir: Path, config: Mapping[str, Any]) -> None:
    validate_run_config_hashes(config)
    path = output_dir / "config.json"
    if path.exists():
        existing = read_json(path)
        validate_run_config_hashes(existing)
        if str(existing.get("full_config_hash", "")) != str(config["full_config_hash"]):
            raise RuntimeError(
                "output directory already contains a different or unversioned run config"
            )
        return
    if (output_dir / "cases").exists() or (output_dir / "done.txt").exists():
        raise RuntimeError("output directory has artifacts but no compatible config.json")
    write_json(path, dict(config))


def write_done_marker(path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("ok\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_shards:
        merge_shards(output_dir, parse_str_csv(args.merge_shards))
        return
    lengths, conditions, explicit_layers = validate_args(args)
    storage = os.environ.get("KVQ_PREKEY_STORAGE", "cpu").lower()
    if storage not in {"cpu", "cuda"}:
        raise ValueError("KVQ_PREKEY_STORAGE must be cpu or cuda")
    config = build_run_config(args, lengths, conditions, storage)
    initialize_or_validate_run_config(output_dir, config)
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    if not args.model_name_or_path:
        raise ValueError("--model-name-or-path is required outside dry-run/merge mode")

    expected = expected_case_keys(config)
    existing_case_rows, existing_edge_rows = collect_committed_cases(
        output_dir,
        str(config["full_config_hash"]),
        expected=expected,
        require_complete=False,
    )
    completed = {case_key(row) for row in existing_case_rows}
    if completed == expected:
        write_aggregate_outputs(output_dir, existing_case_rows, existing_edge_rows)
        write_done_marker(output_dir / "done.txt")
        return
    done_path = output_dir / "done.txt"
    if done_path.exists():
        done_path.unlink()

    model, tokenizer = model_runner.load_model(_model_config(args))
    layers = decoder_layers(model)
    source_layers = resolve_relay_layers(len(layers), explicit_layers)
    capture_layers = tuple(sorted(set(source_layers) | {layer + 1 for layer in source_layers}))
    resolved_design = {
        "source_layers": list(source_layers),
        "capture_layers": list(capture_layers),
        "pre_key_storage": storage,
        "forbidden_scoring_inputs": [
            "gold span",
            "event kind/label/step",
            "answer token",
            "answer gradient",
            "loss gradient",
        ],
        "label_entry_point": "_edge_rows_for_case after score_case_label_free returns",
    }
    design_path = output_dir / "resolved_design.json"
    if design_path.exists() and read_json(design_path) != resolved_design:
        raise RuntimeError("resolved model design differs from the existing shard")
    write_json(design_path, resolved_design)

    for length in lengths:
        for condition in conditions:
            for seed in range(args.seed_start, args.seed_start + args.num_seeds):
                key = (length, condition, seed)
                if key in completed:
                    continue
                case = build_case(
                    tokenizer,
                    target_context_tokens=length,
                    seed=seed,
                    condition=condition,
                    placement=args.placement,
                    code_mode=args.code_mode,
                )
                prompt_ids = list(map(int, case["prompt_ids"]))
                boundary_ids = newline_like_token_ids(
                    tokenizer, prompt_ids[case["body_start"] : case["body_end"]]
                )
                all_blocks = segment_label_free_blocks(
                    prompt_ids,
                    int(case["body_start"]),
                    int(case["body_end"]),
                    args.max_block_tokens,
                    boundary_ids,
                )
                prompt = torch.tensor(prompt_ids, dtype=torch.long).view(1, -1)
                prefix_length = int(prompt.shape[1]) - 1
                state = CaptureState(source_layers, capture_layers, storage)
                with install_capture_hooks(model, state):
                    state.mode = "prefix"
                    legacy, prefill_seconds = base.prefill_sequence(
                        model, prompt[:, :-1], args.prefill_chunk_size
                    )
                    state.mode = "idle"
                    finalize_prefix_keys(state, prefix_length)
                    cache = base.cache_from_legacy(legacy)
                    del legacy
                    prefix_before = cache_prefix_fingerprint(
                        cache, capture_layers, prefix_length
                    )
                    state.mode = "query"
                    base.synchronize()
                    query_started = time.perf_counter()
                    with torch.inference_mode():
                        output = base.forward_with_cache(
                            model,
                            prompt[:, -1:].to(base.input_device(model)),
                            cache,
                            prefix_length,
                        )
                    base.synchronize()
                    query_seconds = time.perf_counter() - query_started
                    state.mode = "idle"
                    del output
                    reset_dynamic_cache(cache, prefix_length)
                    prefix_after_query = cache_prefix_fingerprint(
                        cache, capture_layers, prefix_length
                    )
                    if prefix_before != prefix_after_query:
                        raise RuntimeError("prefix KV changed after query pass and reset")
                    score_started = time.perf_counter()
                    scored = score_case_label_free(
                        model=model,
                        cache=cache,
                        state=state,
                        case_seed=seed,
                        target_context_tokens=length,
                        all_blocks=all_blocks,
                        maximum_blocks=args.max_candidate_blocks,
                        block_temperature=args.block_temperature,
                        value_temperature=args.value_temperature,
                        fd_epsilon=args.fd_epsilon,
                        fd_batch_size=args.fd_batch_size,
                    )
                    base.synchronize()
                    score_seconds = time.perf_counter() - score_started
                    prefix_after_score = cache_prefix_fingerprint(
                        cache, capture_layers, prefix_length
                    )
                    if prefix_before != prefix_after_score:
                        raise RuntimeError("offline relay scoring mutated prefix KV")

                edge_rows, evaluation = _edge_rows_for_case(
                    case=case,
                    scored=scored,
                    maximum_negatives=args.maximum_matched_negatives,
                )
                audits = list(scored["finite_difference_audits"])
                fd_pass = all(
                    float(row["fd_halving_relative_error_max"])
                    <= args.fd_audit_relative_tolerance
                    and float(row["fd_halving_cosine_min"])
                    >= args.fd_audit_cosine_tolerance
                    and float(row["baseline_q_reconstruction_max_abs"])
                    <= args.baseline_q_max_abs_tolerance
                    for row in audits
                )
                for edge_row in edge_rows:
                    edge_row["finite_difference_audit_pass"] = int(fd_pass)
                prefix_audit = {
                    "pass": True,
                    "before": prefix_before,
                    "after_query_reset": prefix_after_query,
                    "after_edge_scoring": prefix_after_score,
                }
                case_row = {
                    "target_context_tokens": length,
                    "prompt_tokens": int(prompt.shape[1]),
                    "condition": condition,
                    "seed": seed,
                    "all_label_free_block_count": int(scored["all_block_count"]),
                    "candidate_block_count": len(scored["candidate_blocks"]),
                    "relay_layer_count": len(source_layers),
                    "relay_layers": ",".join(map(str, source_layers)),
                    "prefix_immutable_audit_pass": 1,
                    "finite_difference_audit_pass": int(fd_pass),
                    "fd_audited_source_block_count_min": min(
                        int(row["audit_block_count"]) for row in audits
                    ),
                    "fd_halving_relative_error_max": max(
                        float(row["fd_halving_relative_error_max"]) for row in audits
                    ),
                    "fd_halving_cosine_min": min(
                        float(row["fd_halving_cosine_min"]) for row in audits
                    ),
                    "baseline_q_reconstruction_max_abs": max(
                        float(row["baseline_q_reconstruction_max_abs"]) for row in audits
                    ),
                    "prefill_seconds": prefill_seconds,
                    "query_capture_seconds": query_seconds,
                    "edge_score_seconds": score_seconds,
                    **evaluation,
                }
                audit_payload = {
                    "case": public_case(case),
                    "finite_difference_audits": audits,
                    "finite_difference_audit_pass": fd_pass,
                    "prefix_immutable_audit": prefix_audit,
                    "evaluation": evaluation,
                }
                commit_case_transaction(
                    output_dir=output_dir,
                    full_config_hash=str(config["full_config_hash"]),
                    case_row=case_row,
                    edge_rows=edge_rows,
                    audit=audit_payload,
                    raw_artifact=_raw_artifact(
                        case, scored, state, prefix_audit, evaluation
                    ),
                )
                completed.add(key)
                print(
                    f"length={length} condition={condition} seed={seed} "
                    f"candidates={len(scored['candidate_blocks'])} "
                    f"positive={evaluation['positive_pair_count']} "
                    f"kvq_auc={evaluation['kvq_relay_auroc']} fd_pass={fd_pass}",
                    flush=True,
                )
                del cache, prompt, scored, state
                clear_allocator()

    case_rows, edge_rows = collect_committed_cases(
        output_dir,
        str(config["full_config_hash"]),
        expected=expected,
        require_complete=True,
    )
    write_aggregate_outputs(output_dir, case_rows, edge_rows)
    write_done_marker(output_dir / "done.txt")


if __name__ == "__main__":
    main()
