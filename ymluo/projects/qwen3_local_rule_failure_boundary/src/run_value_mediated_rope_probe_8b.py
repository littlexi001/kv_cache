from __future__ import annotations

"""Oracle audit of value-mediated RoPE suppression in frozen Qwen3-8B.

The prefix cache is produced once, without gradients, and is never mutated.
Only the final query token is replayed.  The instrumented baseline keeps the
native RoPE logits and opens an autograd graph from the final-token embedding
through the answer logits.  For every sampled token it records

    d margin / d score_j = a_j * g^T (v_j - o),

where ``g`` is the downstream gradient at the per-head attention output.  The
gold and conflict answer IDs are used only to define this diagnostic margin;
they are never exposed to the model input or used as a deployable selector.

The primary causal validation freezes the top-N baseline events per class and
replays a *singleton* intervention for each event: one score in one layer, one
head, and one token receives a small lift.  A deterministic token from the same
layer/head/class is replayed as a matched control.  This avoids conflating a
local derivative with a joint 36x32 intervention.  The earlier joint arm is
retained only as an opt-in nonlinear stress test.
"""

import argparse
import csv
import gc
import json
import math
import os
import statistics
import time
import types
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

import run_local_rule_failure_boundary as base
import run_suppression_certificate_safety_probe_8b as safety


CLASS_ORDER = safety.CLASS_ORDER
PLAN_KINDS = ("target", "random")
CUSTOM_NOOP_BASELINE = "custom_noop_baseline"
SCORE_LIFT_CAP = 0.25
ORACLE_GRADIENT_TARGET = "gold_digit_vs_conflict_digit_margin"
SINGLETON_RANKING_METRICS = (
    "abs_positive_suppression_x_dm_dscore",
    "abs_suppression_x_dm_dscore",
    "abs_dm_dscore",
    "positive_suppression_gap",
)
DEFAULT_SINGLETON_RANKING_METRIC = (
    "abs_positive_suppression_x_dm_dscore"
)

_ACTIVE_CONTROLLER: "ValueMediatedController | None" = None


def rounded(value: float, digits: int = 10) -> float:
    return float(f"{float(value):.{digits}g}")


def parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
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


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def value_mediated_derivative(
    attention_probability: torch.Tensor,
    selected_values: torch.Tensor,
    head_output: torch.Tensor,
    output_gradient: torch.Tensor,
) -> torch.Tensor:
    """Return ``a_j * g^T(v_j-o)`` for selected tokens.

    Shapes are ``attention=[H,N]``, ``values=[H,N,D]``, and
    ``head_output=gradient=[H,D]``.  Float32 accumulation makes the recorded
    diagnostic independent of the surrounding BF16 attention kernel.
    """

    if attention_probability.dim() != 2 or selected_values.dim() != 3:
        raise ValueError("attention must be [H,N] and values must be [H,N,D]")
    if head_output.shape != output_gradient.shape:
        raise ValueError("head output and output gradient must have equal shapes")
    if selected_values.shape[:2] != attention_probability.shape:
        raise ValueError("selected value and attention token axes do not match")
    if selected_values.shape[0] != head_output.shape[0]:
        raise ValueError("head axis does not match")
    centered = selected_values.float() - head_output.float().unsqueeze(1)
    directional = (
        centered * output_gradient.float().unsqueeze(1)
    ).sum(dim=-1)
    return attention_probability.float() * directional


def grouped_query_scores(
    query: torch.Tensor, key: torch.Tensor, groups: int
) -> torch.Tensor:
    """GQA QK without materializing a full repeated KV tensor."""

    batch, query_heads, query_tokens, dimension = query.shape
    if int(key.shape[0]) != int(batch) or int(key.shape[-1]) != int(dimension):
        raise ValueError("query and key batch/head dimensions are incompatible")
    kv_heads = int(key.shape[1])
    if int(groups) < 1 or kv_heads * int(groups) != int(query_heads):
        raise ValueError("invalid grouped-query head ratio")
    grouped_query = query.reshape(
        batch, kv_heads, int(groups), query_tokens, dimension
    )
    scores = torch.einsum("bhgqd,bhkd->bhgqk", grouped_query, key)
    return scores.reshape(batch, query_heads, query_tokens, key.shape[-2])


def grouped_attention_output(
    weights: torch.Tensor, value: torch.Tensor, groups: int
) -> torch.Tensor:
    """GQA weighted Value sum without repeating the long Value cache."""

    batch, query_heads, query_tokens, key_count = weights.shape
    kv_heads = int(value.shape[1])
    if (
        int(groups) < 1
        or kv_heads * int(groups) != int(query_heads)
        or int(value.shape[-2]) != int(key_count)
    ):
        raise ValueError("weights and grouped Value cache are incompatible")
    grouped_weights = weights.reshape(
        batch, kv_heads, int(groups), query_tokens, key_count
    )
    output = torch.einsum("bhgqk,bhkd->bhgqd", grouped_weights, value)
    return output.reshape(
        batch, query_heads, query_tokens, int(value.shape[-1])
    )


def sampled_certificate_bundle(
    query_pre: torch.Tensor,
    query_post: torch.Tensor,
    selected_key_post: torch.Tensor,
    native_scores: torch.Tensor,
    positions: torch.Tensor,
    *,
    query_position: int,
    inv_freq: torch.Tensor,
    rope_scale: float,
    score_scale: float,
    anchor_distances: Sequence[int],
    fixed_anchor_distance: int,
) -> dict[str, torch.Tensor]:
    """Safety certificate math on already-gathered keys.

    This is algebraically the same diagnostic as
    ``safety.certificate_bundle`` but avoids expanding the complete 32K GQA
    key cache to all query heads inside an autograd-enabled pass.
    """

    positions = positions.to(device=selected_key_post.device, dtype=torch.long)
    if int(selected_key_post.shape[-2]) != int(positions.numel()):
        raise ValueError("selected keys and sampled positions do not match")
    key_pre = safety.invert_selected_rope(
        selected_key_post, positions, inv_freq, rope_scale
    )
    pre_score = (
        query_pre.float() * key_pre.float()
    ).sum(dim=-1)[0] * float(score_scale) * float(rope_scale) ** 2
    post_score = native_scores[0, :, 0, :].index_select(1, positions).float()

    anchor_values = torch.tensor(
        list(anchor_distances),
        dtype=torch.long,
        device=selected_key_post.device,
    ).clamp_max(int(query_position))
    anchor_scores = []
    old = positions.view(1, -1).expand(query_post.shape[1], -1)
    for distance in anchor_values.tolist():
        distances = torch.full_like(old, int(distance))
        anchor_scores.append(
            safety._score_at_moved_positions(
                query_post,
                selected_key_post,
                old,
                distances,
                query_position,
                inv_freq,
                score_scale,
            )
        )
    anchor_stack = torch.stack(anchor_scores, dim=-1)
    grid_envelope_score, best_index = anchor_stack.max(dim=-1)
    best_anchor_distance = anchor_values[best_index]
    fixed_index = min(
        range(len(anchor_distances)),
        key=lambda index: abs(
            int(anchor_distances[index]) - int(fixed_anchor_distance)
        ),
    )
    anchor_score = anchor_stack[..., fixed_index]
    phase_upper_score = safety.phase_upper_scores(
        query_pre.expand(-1, -1, positions.numel(), -1),
        key_pre,
        rope_scale,
        score_scale,
    )[0]
    reconstructed = safety.relative_score_from_pre(
        query_pre.expand(-1, -1, positions.numel(), -1),
        key_pre,
        int(query_position) - positions,
        inv_freq,
        rope_scale,
        score_scale,
    )[0]
    return {
        "post_score": post_score,
        "pre_score": pre_score,
        "anchor_score": anchor_score,
        "grid_envelope_score": grid_envelope_score,
        "phase_upper_score": phase_upper_score,
        "pre_suppression": pre_score - post_score,
        "anchor_suppression": anchor_score - post_score,
        "grid_envelope_suppression": grid_envelope_score - post_score,
        "phase_upper_suppression": phase_upper_score - post_score,
        "best_anchor_distance": best_anchor_distance,
        "reconstruction_error": reconstructed - post_score,
    }


def direct_ov_proxies(
    output_projection: torch.nn.Module,
    unembedding: torch.nn.Module,
    selected_values: torch.Tensor,
    head_output: torch.Tensor,
    attention_probability: torch.Tensor,
    *,
    gold_token_id: int,
    conflict_token_id: int,
) -> dict[str, torch.Tensor]:
    """Compute LOCOS-style direct OV controls on sampled values.

    ``locos_direct_ov_gold`` implements the generic per-position form
    ``a * u_gold^T W_O V``.  The margin variant replaces ``u_gold`` with
    ``u_gold-u_conflict``.  The centered variant additionally uses ``V-o`` so
    it is a direct-OV approximation to the softmax score derivative.  None of
    these controls includes downstream residual/MLP/normalization Jacobians.
    """

    if selected_values.dim() != 3 or attention_probability.dim() != 2:
        raise ValueError("selected values must be [H,N,D] and attention [H,N]")
    heads, tokens, head_dim = selected_values.shape
    if head_output.shape != (heads, head_dim):
        raise ValueError("head output must be [H,D]")
    if attention_probability.shape != (heads, tokens):
        raise ValueError("attention and selected values do not match")
    hidden_size = heads * head_dim
    flat = torch.zeros(
        (heads * tokens, hidden_size),
        dtype=selected_values.dtype,
        device=selected_values.device,
    )
    centered_flat = torch.zeros_like(flat)
    for head in range(heads):
        start = head * head_dim
        end = start + head_dim
        row_start = head * tokens
        row_end = row_start + tokens
        flat[row_start:row_end, start:end] = selected_values[head]
        centered_flat[row_start:row_end, start:end] = (
            selected_values[head] - head_output[head].unsqueeze(0)
        )
    projected = output_projection(torch.cat((flat, centered_flat), dim=0))
    bias = getattr(output_projection, "bias", None)
    if bias is not None:
        projected = projected - bias
    projected_value, projected_centered = projected.chunk(2, dim=0)
    weight = getattr(unembedding, "weight", None)
    if weight is None:
        raise TypeError("unembedding module has no weight")
    gold_vector = weight[int(gold_token_id)].float()
    conflict_vector = weight[int(conflict_token_id)].float()
    margin_vector = gold_vector - conflict_vector
    gold_direct = torch.matmul(projected_value.float(), gold_vector)
    margin_direct = torch.matmul(projected_value.float(), margin_vector)
    centered_margin = torch.matmul(projected_centered.float(), margin_vector)
    attention_flat = attention_probability.reshape(-1).float()
    return {
        "locos_direct_ov_gold": (
            attention_flat * gold_direct
        ).reshape(heads, tokens),
        "locos_direct_ov_margin": (
            attention_flat * margin_direct
        ).reshape(heads, tokens),
        "direct_ov_centered_margin_derivative": (
            attention_flat * centered_margin
        ).reshape(heads, tokens),
    }


def select_target_and_random_positions(
    positions: torch.Tensor,
    suppression_gap: torch.Tensor,
    *,
    seed: int,
    layer_index: int,
    class_index: int,
) -> dict[str, torch.Tensor]:
    """Pick one max-gap and one deterministic alternative per head."""

    if positions.dim() != 1 or suppression_gap.dim() != 2:
        raise ValueError("positions must be [N] and suppression gap [H,N]")
    if int(suppression_gap.shape[1]) != int(positions.numel()):
        raise ValueError("candidate dimensions do not match")
    candidate_count = int(positions.numel())
    if candidate_count < 2:
        raise ValueError("at least two sampled tokens are required for random control")
    target_local = suppression_gap.float().argmax(dim=-1)
    heads = int(suppression_gap.shape[0])
    head_ids = torch.arange(heads, device=target_local.device, dtype=torch.long)
    # The offset is in [1,N-1], so the control can never equal the target.  It
    # is deterministic across machines and does not consume global RNG state.
    offset = 1 + (
        int(seed) * 1009
        + int(layer_index) * 9176
        + int(class_index) * 131
        + head_ids * 37
    ) % (candidate_count - 1)
    random_local = (target_local + offset) % candidate_count
    shared = positions.to(target_local.device).view(1, -1).expand(heads, -1)
    return {
        "target_local_indices": target_local.detach().cpu(),
        "random_local_indices": random_local.detach().cpu(),
        "target_positions": shared.gather(1, target_local.view(-1, 1))
        .squeeze(1)
        .detach()
        .cpu(),
        "random_positions": shared.gather(1, random_local.view(-1, 1))
        .squeeze(1)
        .detach()
        .cpu(),
        "target_suppression_gap": suppression_gap.gather(
            1, target_local.view(-1, 1)
        )
        .squeeze(1)
        .detach()
        .cpu(),
        "random_suppression_gap": suppression_gap.gather(
            1, random_local.view(-1, 1)
        )
        .squeeze(1)
        .detach()
        .cpu(),
    }


def apply_uniform_score_lift(
    native_scores: torch.Tensor,
    positions: torch.Tensor,
    score_lift: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Add the same bounded logit lift to one frozen position per head."""

    if not (0.0 < float(score_lift) <= SCORE_LIFT_CAP + 1e-12):
        raise ValueError(f"score_lift must be in (0,{SCORE_LIFT_CAP}]")
    if native_scores.dim() != 4 or int(native_scores.shape[0]) != 1:
        raise ValueError("native scores must have shape [1,H,Q,K]")
    if int(native_scores.shape[2]) != 1:
        raise ValueError("this probe only supports one final query token")
    positions = positions.to(device=native_scores.device, dtype=torch.long)
    if positions.dim() != 1 or int(positions.numel()) != int(native_scores.shape[1]):
        raise ValueError("positions must contain exactly one entry per head")
    if bool((positions < 0).any()) or bool((positions >= native_scores.shape[-1]).any()):
        raise IndexError("planned score-lift position is outside the KV cache")
    modified = native_scores.clone()
    head_ids = torch.arange(native_scores.shape[1], device=native_scores.device)
    modified[0, head_ids, 0, positions] = (
        modified[0, head_ids, 0, positions] + float(score_lift)
    )
    return modified, {
        "applied_count": int(positions.numel()),
        "score_lift": rounded(score_lift),
        "score_delta_sum": rounded(float(score_lift) * positions.numel()),
    }


def apply_single_score_lift(
    native_scores: torch.Tensor,
    *,
    head: int,
    position: int,
    score_lift: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Replay one frozen ``(head, token)`` score delta for one query.

    Zero is accepted only so the causal baseline can traverse this exact clone,
    indexing, and assignment path without changing the mathematical score.
    """

    if not (0.0 <= float(score_lift) <= SCORE_LIFT_CAP + 1e-12):
        raise ValueError(f"score_lift must be in [0,{SCORE_LIFT_CAP}]")
    if native_scores.dim() != 4 or tuple(native_scores.shape[:1]) != (1,):
        raise ValueError("native scores must have shape [1,H,Q,K]")
    if int(native_scores.shape[2]) != 1:
        raise ValueError("this probe only supports one final query token")
    if not 0 <= int(head) < int(native_scores.shape[1]):
        raise IndexError("planned head is outside the attention score tensor")
    if not 0 <= int(position) < int(native_scores.shape[-1]):
        raise IndexError("planned token position is outside the KV cache")
    modified = native_scores.clone()
    modified[0, int(head), 0, int(position)] = (
        modified[0, int(head), 0, int(position)] + float(score_lift)
    )
    return modified, {
        "applied_count": int(float(score_lift) != 0.0),
        "replayed_coordinate_count": 1,
        "score_lift": rounded(score_lift),
        "score_delta_sum": rounded(score_lift),
    }


def singleton_ranking_value(row: dict[str, Any], metric: str) -> float:
    """Compute a frozen baseline-only ranking value for one sampled event."""

    if metric == "abs_positive_suppression_x_dm_dscore":
        return abs(float(row["positive_suppression_x_dm_dscore"]))
    if metric == "abs_suppression_x_dm_dscore":
        return abs(float(row["suppression_x_dm_dscore"]))
    if metric == "abs_dm_dscore":
        return abs(float(row["dm_dscore"]))
    if metric == "positive_suppression_gap":
        return max(0.0, float(row["suppression_gap"]))
    raise ValueError(f"unknown singleton ranking metric: {metric}")


def _sample_identity(row: dict[str, Any]) -> tuple[int, int, str, int, int]:
    return (
        int(row["layer"]),
        int(row["head"]),
        str(row["class"]),
        int(row["sample_index"]),
        int(row["token_position"]),
    )


def _frozen_sample_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    """Copy only baseline quantities needed after candidate selection."""

    fields = (
        "layer",
        "head",
        "class",
        "sample_index",
        "token_position",
        "relative_distance",
        "is_decisive_token",
        "post_score",
        "grid_envelope_score",
        "suppression_gap",
        "attention_probability",
        "best_anchor_distance",
        "dm_dscore",
        "suppression_x_dm_dscore",
        "positive_suppression_x_dm_dscore",
        "locos_direct_ov_gold",
        "locos_direct_ov_margin",
        "direct_ov_centered_margin_derivative",
        "suppression_x_direct_ov_centered_margin",
    )
    return {name: row[name] for name in fields}


def freeze_singleton_candidates(
    sample_rows: Sequence[dict[str, Any]],
    *,
    top_n: int,
    ranking_metric: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Freeze global top-N events per class and matched random controls.

    Ranking and control selection use only immutable baseline rows.  Each
    control is drawn from the same layer, head, and class and cannot equal its
    target.  Integer arithmetic keeps the choice deterministic across hosts and
    avoids Python's process-randomized hash.
    """

    if int(top_n) < 0:
        raise ValueError("singleton top-N must be non-negative")
    if ranking_metric not in SINGLETON_RANKING_METRICS:
        raise ValueError(f"unknown singleton ranking metric: {ranking_metric}")
    if int(top_n) == 0:
        return []

    frozen: list[dict[str, Any]] = []
    for class_index, category in enumerate(CLASS_ORDER):
        class_rows = [row for row in sample_rows if row["class"] == category]
        ranked = sorted(
            class_rows,
            key=lambda row: (
                -singleton_ranking_value(row, ranking_metric),
                *_sample_identity(row),
            ),
        )[: int(top_n)]
        for rank, target_row in enumerate(ranked, start=1):
            peers = sorted(
                (
                    row
                    for row in class_rows
                    if int(row["layer"]) == int(target_row["layer"])
                    and int(row["head"]) == int(target_row["head"])
                ),
                key=_sample_identity,
            )
            if len(peers) < 2:
                raise RuntimeError(
                    "singleton matched control needs two samples in each "
                    "layer/head/class cell"
                )
            target_identity = _sample_identity(target_row)
            target_index = next(
                index
                for index, peer in enumerate(peers)
                if _sample_identity(peer) == target_identity
            )
            offset = 1 + (
                int(seed) * 1009
                + int(class_index) * 131
                + int(rank) * 53
                + int(target_row["layer"]) * 9176
                + int(target_row["head"]) * 37
                + int(target_row["token_position"]) * 7
            ) % (len(peers) - 1)
            random_row = peers[(target_index + offset) % len(peers)]
            if _sample_identity(random_row) == target_identity:
                raise AssertionError("deterministic singleton control equals target")
            frozen.append(
                {
                    "pair_id": f"{category}_{rank:03d}",
                    "class": category,
                    "candidate_rank": int(rank),
                    "ranking_metric": ranking_metric,
                    "ranking_value": rounded(
                        singleton_ranking_value(target_row, ranking_metric)
                    ),
                    "target": _frozen_sample_snapshot(target_row),
                    "random": _frozen_sample_snapshot(random_row),
                    "candidate_frozen_before_intervention": 1,
                    "matched_layer_head_class": 1,
                }
            )
    return frozen


def singleton_predictors(
    sample: dict[str, Any], score_lift: float
) -> dict[str, float]:
    """Return exact first-order and proxy predictors for one frozen event."""

    return {
        "predicted_first_order_delta_gold_conflict_margin": float(score_lift)
        * float(sample["dm_dscore"]),
        "predicted_direct_ov_delta_gold_conflict_margin": float(score_lift)
        * float(sample["direct_ov_centered_margin_derivative"]),
        "selected_suppression_gap_sum": float(sample["suppression_gap"]),
        "selected_attention_probability_sum": float(
            sample["attention_probability"]
        ),
        "selected_suppression_x_exact_sensitivity_sum": float(
            sample["suppression_x_dm_dscore"]
        ),
        "selected_suppression_x_direct_ov_sum": float(
            sample["suppression_x_direct_ov_centered_margin"]
        ),
        "selected_locos_direct_ov_gold_sum": float(
            sample["locos_direct_ov_gold"]
        ),
        "selected_locos_direct_ov_margin_sum": float(
            sample["locos_direct_ov_margin"]
        ),
    }


def sign_with_tolerance(value: float, tolerance: float = 1e-8) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def closure_metrics(predicted: float, actual: float) -> dict[str, Any]:
    absolute = abs(float(predicted) - float(actual))
    symmetric = absolute / max(
        abs(float(predicted)) + abs(float(actual)), 1e-8
    )
    relative_actual = absolute / max(abs(float(actual)), 1e-8)
    predicted_sign = sign_with_tolerance(float(predicted))
    actual_sign = sign_with_tolerance(float(actual))
    return {
        "predicted_first_order_delta_gold_conflict_margin": rounded(predicted),
        "actual_delta_gold_conflict_margin": rounded(actual),
        "first_order_absolute_closure_error": rounded(absolute),
        "first_order_symmetric_closure_error": rounded(symmetric),
        "first_order_relative_to_actual_error": rounded(relative_actual),
        "predicted_margin_change_sign": predicted_sign,
        "actual_margin_change_sign": actual_sign,
        "first_order_sign_match": int(predicted_sign == actual_sign),
        "first_order_sign_informative": int(
            predicted_sign != 0 or actual_sign != 0
        ),
    }


def audit_noop_referenced_case(
    case_rows: Sequence[dict[str, Any]],
    frozen_candidates: Sequence[dict[str, Any]],
    noop_answer: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless every causal replay is referenced to epsilon=0.

    The diagnostic/autograd pass is intentionally excluded: it is used to
    freeze candidates and obtain derivatives, but it is not a numerically
    matched baseline for inference-mode singleton replays.
    """

    noop_rows = [
        row
        for row in case_rows
        if row.get("intervention_class") == CUSTOM_NOOP_BASELINE
    ]
    if len(noop_rows) != 1:
        raise RuntimeError(
            f"expected exactly one custom no-op row, found {len(noop_rows)}"
        )
    noop_row = noop_rows[0]
    if (
        float(noop_row.get("uniform_score_lift", float("nan"))) != 0.0
        or int(noop_row.get("applied_count", -1)) != 0
        or int(noop_row.get("replayed_coordinate_count", -1)) != 1
        or int(noop_row.get("epsilon_zero_noop_control", 0)) != 1
    ):
        raise RuntimeError("custom no-op row is not an audited epsilon=0 replay")

    interventions = [
        row
        for row in case_rows
        if row.get("intervention_scope") in {"singleton", "joint"}
    ]
    singleton_rows = [
        row for row in interventions if row["intervention_scope"] == "singleton"
    ]
    expected_singletons = 2 * len(frozen_candidates)
    if len(singleton_rows) != expected_singletons:
        raise RuntimeError(
            f"expected {expected_singletons} singleton rows, "
            f"found {len(singleton_rows)}"
        )

    noop_margin = float(noop_answer["gold_conflict_margin"])
    noop_nll = float(noop_answer["gold_nll"])
    for row in interventions:
        if row.get("comparison_baseline") != CUSTOM_NOOP_BASELINE:
            raise RuntimeError("an intervention does not reference the custom no-op")
        if row.get("causal_delta_reference") != CUSTOM_NOOP_BASELINE:
            raise RuntimeError("causal delta reference metadata is inconsistent")
        expected_margin_delta = rounded(
            float(row["gold_conflict_margin"]) - noop_margin
        )
        expected_nll_delta = rounded(float(row["gold_nll"]) - noop_nll)
        if abs(
            float(row["delta_gold_conflict_margin"])
            - float(expected_margin_delta)
        ) > 1e-8:
            raise RuntimeError("margin delta was not computed against custom no-op")
        if abs(float(row["delta_gold_nll"]) - float(expected_nll_delta)) > 1e-8:
            raise RuntimeError("NLL delta was not computed against custom no-op")

    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in singleton_rows:
        pairs.setdefault(str(row["pair_id"]), []).append(row)
    if len(pairs) != len(frozen_candidates):
        raise RuntimeError("singleton pair IDs are not one-to-one with candidates")
    for pair_id, rows in pairs.items():
        if {str(row["plan_kind"]) for row in rows} != set(PLAN_KINDS):
            raise RuntimeError(f"pair {pair_id} lacks target/random controls")
        if len(rows) != 2:
            raise RuntimeError(f"pair {pair_id} contains duplicate replay rows")
        target_row = next(row for row in rows if row["plan_kind"] == "target")
        random_row = next(row for row in rows if row["plan_kind"] == "random")
        for field_name in (
            "intervention_class",
            "selected_baseline_layer",
            "selected_baseline_head",
        ):
            if target_row[field_name] != random_row[field_name]:
                raise RuntimeError(f"pair {pair_id} is not matched on {field_name}")
        if (
            int(target_row["selected_baseline_token_position"])
            == int(random_row["selected_baseline_token_position"])
        ):
            raise RuntimeError(f"pair {pair_id} uses the target as its control")
        if any(int(row.get("applied_count", 0)) != 1 for row in rows):
            raise RuntimeError(f"pair {pair_id} is not a singleton intervention")

    return {
        "passed": True,
        "causal_delta_reference": CUSTOM_NOOP_BASELINE,
        "epsilon_zero_noop_count": 1,
        "epsilon_zero_noop_applied_count": 0,
        "epsilon_zero_replayed_coordinate_count": 1,
        "singleton_candidate_count": len(frozen_candidates),
        "singleton_replay_count": len(singleton_rows),
        "matched_target_random_pair_count": len(pairs),
        "all_intervention_deltas_recomputed_from_noop": True,
        "all_singleton_pairs_matched_and_distinct": True,
    }


@dataclass
class ValueMediatedController:
    mode: str
    case: dict[str, Any]
    anchor_distances: tuple[int, ...]
    fixed_anchor_distance: int
    score_lift: float
    gold_token_id: int | None = None
    conflict_token_id: int | None = None
    target_class: str | None = None
    plan_kind: str | None = None
    intervention_scope: str = "joint"
    singleton_layer: int | None = None
    singleton_head: int | None = None
    singleton_position: int | None = None
    plan: dict[str, dict[int, dict[str, torch.Tensor]]] = field(
        default_factory=lambda: {name: {} for name in CLASS_ORDER}
    )
    sample_rows: list[dict[str, Any]] = field(default_factory=list)
    reconstruction_error_max: float = 0.0
    gradient_layers_completed: set[int] = field(default_factory=set)
    planned_layer_count: int = 0
    applied_count: int = 0

    def plan_layer_and_attach_gradient(
        self,
        layer_index: int,
        query_pre: torch.Tensor,
        query_post: torch.Tensor,
        key_post: torch.Tensor,
        value: torch.Tensor,
        groups: int,
        native_scores: torch.Tensor,
        native_weights: torch.Tensor,
        head_output: torch.Tensor,
        query_position: int,
        inv_freq: torch.Tensor,
        rope_scale: float,
        score_scale: float,
        output_projection: torch.nn.Module,
        unembedding: torch.nn.Module,
    ) -> None:
        if int(layer_index) in self.plan[CLASS_ORDER[0]]:
            raise RuntimeError(f"layer {layer_index} was planned more than once")
        class_positions: list[torch.Tensor] = []
        row_indices_by_head: list[list[int]] = [
            [] for _ in range(int(query_post.shape[1]))
        ]

        for class_index, category in enumerate(CLASS_ORDER):
            raw_positions = self.case["sample_positions"][category]
            if len(raw_positions) < 2:
                raise RuntimeError(
                    f"class {category} needs at least two sampled tokens"
                )
            positions = torch.tensor(
                raw_positions, dtype=torch.long, device=key_post.device
            )
            with torch.no_grad():
                selected_kv_keys = safety.gather_shared_positions(
                    key_post.detach(), positions
                )
                selected_query_keys = repeat_kv(selected_kv_keys, groups)
                bundle = sampled_certificate_bundle(
                    query_pre.detach(),
                    query_post.detach(),
                    selected_query_keys,
                    native_scores.detach(),
                    positions,
                    query_position=query_position,
                    inv_freq=inv_freq,
                    rope_scale=rope_scale,
                    score_scale=score_scale,
                    anchor_distances=self.anchor_distances,
                    fixed_anchor_distance=self.fixed_anchor_distance,
                )
            gap = bundle["grid_envelope_suppression"]
            selection = select_target_and_random_positions(
                positions,
                gap,
                seed=int(self.case["seed"]),
                layer_index=int(layer_index),
                class_index=int(class_index),
            )
            self.plan[category][int(layer_index)] = selection
            self.reconstruction_error_max = max(
                self.reconstruction_error_max,
                float(bundle["reconstruction_error"].abs().max().item()),
            )
            decisive = set(map(int, self.case["decisive_positions"][category]))
            target_local = selection["target_local_indices"].tolist()
            random_local = selection["random_local_indices"].tolist()
            class_attention = native_weights[0, :, 0, :].index_select(1, positions)
            for head in range(int(query_post.shape[1])):
                for local_index, position in enumerate(raw_positions):
                    row_index = len(self.sample_rows)
                    row_indices_by_head[head].append(row_index)
                    suppression = float(gap[head, local_index].item())
                    row = {
                        "layer": int(layer_index),
                        "head": int(head),
                        "class": category,
                        "sample_index": int(local_index),
                        "token_position": int(position),
                        "relative_distance": int(query_position - int(position)),
                        "is_decisive_token": int(int(position) in decisive),
                        "selected_for_target": int(local_index == target_local[head]),
                        "selected_for_random": int(local_index == random_local[head]),
                        "post_score": rounded(
                            float(bundle["post_score"][head, local_index].item())
                        ),
                        "grid_envelope_score": rounded(
                            float(
                                bundle["grid_envelope_score"][
                                    head, local_index
                                ].item()
                            )
                        ),
                        "suppression_gap": rounded(suppression),
                        "attention_probability": rounded(
                            float(class_attention[head, local_index].item())
                        ),
                        "best_anchor_distance": int(
                            bundle["best_anchor_distance"][
                                head, local_index
                            ].item()
                        ),
                        "reconstruction_error": rounded(
                            float(
                                bundle["reconstruction_error"][
                                    head, local_index
                                ].item()
                            )
                        ),
                        "dm_dscore": None,
                        "suppression_x_dm_dscore": None,
                        "positive_suppression_x_dm_dscore": None,
                        "locos_direct_ov_gold": None,
                        "locos_direct_ov_margin": None,
                        "direct_ov_centered_margin_derivative": None,
                        "suppression_x_direct_ov_centered_margin": None,
                        "oracle_gradient_target": ORACLE_GRADIENT_TARGET,
                        "oracle_diagnostic_only": 1,
                    }
                    self.sample_rows.append(row)
            class_positions.append(positions)

        all_positions = torch.cat(class_positions, dim=0)
        selected_kv_values = safety.gather_shared_positions(value, all_positions)
        selected_values = repeat_kv(selected_kv_values, groups)[0].detach()
        selected_attention = native_weights[0, :, 0, :].index_select(
            1, all_positions
        ).detach()
        detached_output = head_output[0, :, 0, :].detach()
        if self.gold_token_id is None or self.conflict_token_id is None:
            raise RuntimeError("oracle token IDs are missing from baseline controller")
        with torch.no_grad():
            ov_controls = direct_ov_proxies(
                output_projection,
                unembedding,
                selected_values,
                detached_output,
                selected_attention,
                gold_token_id=int(self.gold_token_id),
                conflict_token_id=int(self.conflict_token_id),
            )
        expected_per_head = int(all_positions.numel())
        if any(len(items) != expected_per_head for items in row_indices_by_head):
            raise AssertionError("sample-row indexing is incomplete")
        for head, row_indices in enumerate(row_indices_by_head):
            for sample_index, row_index in enumerate(row_indices):
                row = self.sample_rows[row_index]
                for field_name, values in ov_controls.items():
                    row[field_name] = rounded(
                        float(values[head, sample_index].item())
                    )
                row["suppression_x_direct_ov_centered_margin"] = rounded(
                    float(row["suppression_gap"])
                    * float(row["direct_ov_centered_margin_derivative"])
                )

        def record_gradient(gradient: torch.Tensor) -> torch.Tensor:
            derivative = value_mediated_derivative(
                selected_attention,
                selected_values,
                detached_output,
                gradient[0, :, 0, :].detach(),
            ).cpu()
            for head, row_indices in enumerate(row_indices_by_head):
                for sample_index, row_index in enumerate(row_indices):
                    value = float(derivative[head, sample_index].item())
                    row = self.sample_rows[row_index]
                    gap_value = float(row["suppression_gap"])
                    row["dm_dscore"] = rounded(value)
                    row["suppression_x_dm_dscore"] = rounded(
                        gap_value * value
                    )
                    row["positive_suppression_x_dm_dscore"] = rounded(
                        max(0.0, gap_value) * value
                    )
            self.gradient_layers_completed.add(int(layer_index))
            return gradient

        if not head_output.requires_grad:
            raise RuntimeError(
                "instrumented baseline has no autograd graph; final-token "
                "embedding must require gradients"
            )
        head_output.register_hook(record_gradient)
        self.planned_layer_count += 1

    def intervene_layer(
        self, layer_index: int, native_scores: torch.Tensor
    ) -> torch.Tensor:
        if self.target_class not in CLASS_ORDER:
            raise RuntimeError("intervention target class is missing")
        if self.plan_kind not in PLAN_KINDS:
            raise RuntimeError("intervention plan kind is missing")
        if self.intervention_scope == "singleton":
            if (
                self.singleton_layer is None
                or self.singleton_head is None
                or self.singleton_position is None
            ):
                raise RuntimeError("singleton intervention coordinates are missing")
            if int(layer_index) != int(self.singleton_layer):
                return native_scores
            modified, summary = apply_single_score_lift(
                native_scores,
                head=int(self.singleton_head),
                position=int(self.singleton_position),
                score_lift=self.score_lift,
            )
            self.applied_count += int(summary["applied_count"])
            return modified
        if self.intervention_scope != "joint":
            raise ValueError(
                f"unknown intervention scope: {self.intervention_scope}"
            )
        entry = self.plan[self.target_class][int(layer_index)]
        modified, summary = apply_uniform_score_lift(
            native_scores,
            entry[f"{self.plan_kind}_positions"],
            self.score_lift,
        )
        self.applied_count += int(summary["applied_count"])
        return modified

    def finalize_gradients(self) -> None:
        expected = set(self.plan[CLASS_ORDER[0]])
        if self.gradient_layers_completed != expected:
            missing = sorted(expected - self.gradient_layers_completed)
            extra = sorted(self.gradient_layers_completed - expected)
            raise RuntimeError(
                f"autograd hooks incomplete: missing={missing}, extra={extra}"
            )
        missing_rows = [
            index
            for index, row in enumerate(self.sample_rows)
            if row["dm_dscore"] is None
        ]
        if missing_rows:
            raise RuntimeError(
                f"value-mediated derivatives missing for {len(missing_rows)} rows"
            )

    def predicted_margin_change(self, category: str, plan_kind: str) -> float:
        if plan_kind not in PLAN_KINDS:
            raise ValueError(f"unknown plan kind: {plan_kind}")
        field_name = f"selected_for_{plan_kind}"
        values = [
            float(row["dm_dscore"])
            for row in self.sample_rows
            if row["class"] == category and int(row[field_name]) == 1
        ]
        expected = self.planned_layer_count * len(
            self.plan[category][next(iter(self.plan[category]))][
                f"{plan_kind}_positions"
            ]
        )
        if len(values) != expected:
            raise RuntimeError(
                f"plan {category}/{plan_kind} has {len(values)} derivatives; "
                f"expected {expected}"
            )
        return float(self.score_lift) * math.fsum(values)

    def plan_predictors(self, category: str, plan_kind: str) -> dict[str, float]:
        """Aggregate exact-gradient and cheaper controls on one frozen plan."""

        if plan_kind not in PLAN_KINDS:
            raise ValueError(f"unknown plan kind: {plan_kind}")
        selected_field = f"selected_for_{plan_kind}"
        selected = [
            row
            for row in self.sample_rows
            if row["class"] == category and int(row[selected_field]) == 1
        ]
        if not selected:
            raise RuntimeError(f"empty plan for {category}/{plan_kind}")
        return {
            "predicted_first_order_delta_gold_conflict_margin": float(
                self.score_lift
            )
            * math.fsum(float(row["dm_dscore"]) for row in selected),
            "predicted_direct_ov_delta_gold_conflict_margin": float(
                self.score_lift
            )
            * math.fsum(
                float(row["direct_ov_centered_margin_derivative"])
                for row in selected
            ),
            "selected_suppression_gap_sum": math.fsum(
                float(row["suppression_gap"]) for row in selected
            ),
            "selected_attention_probability_sum": math.fsum(
                float(row["attention_probability"]) for row in selected
            ),
            "selected_suppression_x_exact_sensitivity_sum": math.fsum(
                float(row["suppression_x_dm_dscore"]) for row in selected
            ),
            "selected_suppression_x_direct_ov_sum": math.fsum(
                float(row["suppression_x_direct_ov_centered_margin"])
                for row in selected
            ),
            "selected_locos_direct_ov_gold_sum": math.fsum(
                float(row["locos_direct_ov_gold"]) for row in selected
            ),
            "selected_locos_direct_ov_margin_sum": math.fsum(
                float(row["locos_direct_ov_margin"]) for row in selected
            ),
        }

    def intervention_summary(self) -> dict[str, Any]:
        return {
            "applied_count": int(self.applied_count),
            "uniform_score_lift": rounded(self.score_lift),
            "plan_kind": self.plan_kind,
            "target_class": self.target_class,
            "intervention_scope": self.intervention_scope,
            "singleton_layer": self.singleton_layer,
            "singleton_head": self.singleton_head,
            "singleton_position": self.singleton_position,
        }


@contextmanager
def activate(controller: ValueMediatedController | None):
    global _ACTIVE_CONTROLLER
    previous = _ACTIVE_CONTROLLER
    _ACTIVE_CONTROLLER = controller
    try:
        yield
    finally:
        _ACTIVE_CONTROLLER = previous


def repeat_kv(values: torch.Tensor, groups: int) -> torch.Tensor:
    return values if groups == 1 else values.repeat_interleave(groups, dim=1)


def add_attention_mask(
    scores: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    if attention_mask is None:
        return scores
    return scores + attention_mask[
        :, :, -scores.shape[-2] :, : scores.shape[-1]
    ]


def value_mediated_attention_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_value: Any | None = None,
    cache_position: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    controller = _ACTIVE_CONTROLLER
    if controller is None or int(hidden_states.shape[-2]) != 1:
        return self._value_mediated_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    modeling_qwen3 = self._value_mediated_modeling_qwen3
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_pre = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    current_key_pre = self.k_norm(
        self.k_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    current_value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    cos = cos.to(query_pre.device)
    sin = sin.to(query_pre.device)
    query_post, current_key_post = modeling_qwen3.apply_rotary_pos_emb(
        query_pre, current_key_pre, cos, sin
    )
    key_post, value = safety.read_only_final_query_kv(
        past_key_value,
        int(self.layer_idx),
        current_key_post,
        current_value,
    )

    groups = int(query_post.shape[1] // key_post.shape[1])
    key_count = int(key_post.shape[-2])
    query_position = key_count - 1
    score_scale = float(
        getattr(self, "scaling", 1.0 / math.sqrt(query_post.shape[-1]))
    )
    native_scores = grouped_query_scores(query_post, key_post, groups) * score_scale
    native_scores = add_attention_mask(native_scores, attention_mask)
    scores = native_scores
    if controller.mode == "intervene":
        scores = controller.intervene_layer(int(self.layer_idx), native_scores)
    elif controller.mode != "baseline":
        raise ValueError(f"unknown controller mode: {controller.mode}")

    weights = F.softmax(scores.float(), dim=-1).to(query_post.dtype)
    head_output = grouped_attention_output(weights, value, groups)

    if controller.mode == "baseline":
        rotary = self._value_mediated_rotary_ref()
        if rotary is None:
            raise RuntimeError("model rotary embedding was released unexpectedly")
        unembedding = self._value_mediated_lm_head_ref()
        if unembedding is None:
            raise RuntimeError("model unembedding was released unexpectedly")
        controller.plan_layer_and_attach_gradient(
            int(self.layer_idx),
            query_pre,
            query_post,
            key_post,
            value,
            groups,
            native_scores,
            weights,
            head_output,
            query_position,
            rotary.inv_freq.detach().float().to(query_post.device),
            safety.attention_scaling((cos, sin)),
            score_scale,
            self.o_proj,
            unembedding,
        )

    attention_output = head_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attention_output), weights


def patch_model(model: Any) -> None:
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    found = 0
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3Attention":
            continue
        if not hasattr(module, "_value_mediated_original_forward"):
            module._value_mediated_original_forward = module.forward
            module._value_mediated_modeling_qwen3 = modeling_qwen3
            module._value_mediated_rotary_ref = weakref.ref(model.model.rotary_emb)
            module._value_mediated_lm_head_ref = weakref.ref(model.lm_head)
            module.forward = types.MethodType(value_mediated_attention_forward, module)
        found += 1
    if found == 0:
        raise RuntimeError("no Qwen3Attention modules found")


def freeze_model_parameters(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def prefill_sequence_no_grad(
    model: Any, prompt_prefix: torch.Tensor, chunk_size: int
) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], float]:
    """Prefill a normal (non-inference-tensor) immutable prefix cache."""

    device = base.input_device(model)
    ids = prompt_prefix.to(device)
    past = None
    past_len = 0
    base.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, int(ids.shape[1]), int(chunk_size)):
            chunk = ids[:, start : start + int(chunk_size)]
            output = base.forward_with_cache(model, chunk, past, past_len)
            past = output.past_key_values
            past_len += int(chunk.shape[1])
            del output
    base.synchronize()
    return base.legacy_cache(past), time.perf_counter() - started


def prefix_cache_signature(
    legacy: tuple[tuple[torch.Tensor, torch.Tensor], ...]
) -> tuple[tuple[Any, ...], ...]:
    """Cheap identity/version signature detecting in-place prefix mutation."""

    signature: list[tuple[Any, ...]] = []
    for key, value in legacy:
        signature.append(
            (
                tuple(key.shape),
                int(key.data_ptr()),
                int(key._version),
                tuple(value.shape),
                int(value.data_ptr()),
                int(value._version),
            )
        )
    return tuple(signature)


def forward_final_with_autograd(
    model: Any,
    final_token_ids: torch.Tensor,
    past_key_values: Any,
    past_len: int,
) -> tuple[Any, torch.Tensor]:
    device = base.input_device(model)
    ids = final_token_ids.to(device)
    inputs_embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    q_len = int(ids.shape[1])
    kwargs: dict[str, Any] = {
        "inputs_embeds": inputs_embeds,
        "past_key_values": past_key_values,
        "attention_mask": torch.ones(
            (1, int(past_len) + q_len), dtype=torch.long, device=device
        ),
        "position_ids": torch.arange(
            int(past_len), int(past_len) + q_len, device=device
        ).view(1, -1),
        "cache_position": torch.arange(
            int(past_len), int(past_len) + q_len, device=device
        ),
        "use_cache": True,
        "return_dict": True,
        "logits_to_keep": 1,
        "output_hidden_states": True,
    }
    return model(**kwargs), inputs_embeds


def forward_final_with_hidden_states(
    model: Any,
    final_token_ids: torch.Tensor,
    past_key_values: Any,
    past_len: int,
) -> Any:
    """Replay one final token and retain only the small per-layer query states."""

    device = base.input_device(model)
    ids = final_token_ids.to(device)
    q_len = int(ids.shape[1])
    return model(
        input_ids=ids,
        past_key_values=past_key_values,
        attention_mask=torch.ones(
            (1, int(past_len) + q_len), dtype=torch.long, device=device
        ),
        position_ids=torch.arange(
            int(past_len), int(past_len) + q_len, device=device
        ).view(1, -1),
        cache_position=torch.arange(
            int(past_len), int(past_len) + q_len, device=device
        ),
        use_cache=True,
        return_dict=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )


def fp32_pair_logit_tensors(
    model: Any,
    output: Any,
    gold_token_id: int,
    conflict_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute two unembedding rows in FP32 while preserving autograd.

    The final hidden state may still be BF16/FP16.  This removes the coarse
    low-precision LM-head dot product from the causal margin, but cannot recover
    information already rounded inside the transformer.
    """

    hidden_states = getattr(output, "hidden_states", None)
    if not hidden_states:
        raise RuntimeError("final hidden states are required for FP32 pair margin")
    final_hidden_source = hidden_states[-1][0, -1]
    weight = getattr(model.lm_head, "weight", None)
    if weight is None:
        raise TypeError("model lm_head has no weight")
    final_hidden = final_hidden_source.to(device=weight.device, dtype=torch.float32)
    gold_weight = weight[int(gold_token_id)].float()
    conflict_weight = weight[int(conflict_token_id)].float()
    gold_logit = torch.dot(final_hidden, gold_weight)
    conflict_logit = torch.dot(final_hidden, conflict_weight)
    bias = getattr(model.lm_head, "bias", None)
    if bias is not None:
        gold_logit = gold_logit + bias[int(gold_token_id)].float()
        conflict_logit = conflict_logit + bias[int(conflict_token_id)].float()
    return gold_logit, conflict_logit, final_hidden_source


def answer_metrics_with_fp32_pair(
    model: Any,
    tokenizer: Any,
    output: Any,
    answer_ids: dict[str, int],
    conflict_answer: str,
) -> dict[str, Any]:
    """Use native full-vocabulary metrics plus an FP32 two-token margin."""

    metrics = safety.answer_metrics(
        tokenizer, output.logits.detach(), answer_ids, conflict_answer
    )
    native_margin = float(metrics["gold_conflict_margin"])
    gold_id = int(answer_ids["nine"])
    conflict_id = int(answer_ids[conflict_answer])
    gold_logit, conflict_logit, final_hidden = fp32_pair_logit_tensors(
        model, output, gold_id, conflict_id
    )
    fp32_margin = float((gold_logit - conflict_logit).detach().item())
    metrics.update(
        {
            "gold_logit_fp32_pair": rounded(float(gold_logit.detach().item())),
            "conflict_logit_fp32_pair": rounded(
                float(conflict_logit.detach().item())
            ),
            "gold_conflict_margin_model_logits": rounded(native_margin),
            "gold_conflict_margin_fp32_pair": rounded(fp32_margin),
            # Make the higher-resolution pair readout the primary causal margin
            # consumed by safety.delta_metrics and closure_metrics.
            "gold_conflict_margin": rounded(fp32_margin),
            "pair_margin_compute_dtype": "float32",
            "final_hidden_source_dtype": str(final_hidden.dtype).replace(
                "torch.", ""
            ),
            "pair_margin_hidden_precision_limited": int(
                final_hidden.dtype != torch.float32
            ),
            "full_vocab_metrics_source": "model_logits_hidden_precision_limited",
        }
    )
    return metrics


def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    dx = [float(value) - mean_x for value in x]
    dy = [float(value) - mean_y for value in y]
    denominator = math.sqrt(math.fsum(v * v for v in dx) * math.fsum(v * v for v in dy))
    if denominator == 0.0:
        return float("nan")
    return math.fsum(a * b for a, b in zip(dx, dy)) / denominator


def average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(map(float, values)), key=lambda item: item[1])
    ranks = [0.0] * len(ordered)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = ((index + 1) + end) / 2.0
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = average
        index = end
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    return correlation(average_ranks(x), average_ranks(y))


def _finite_or_na(value: float) -> float | str:
    return rounded(value) if math.isfinite(value) else "NA"


def sample_summary(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lengths = sorted({int(row["target_context_tokens"]) for row in samples})
    for length_label, subset in [
        *((str(length), [row for row in samples if int(row["target_context_tokens"]) == length]) for length in lengths),
        ("all", list(samples)),
    ]:
        for category in CLASS_ORDER:
            selected = [row for row in subset if row["class"] == category]
            if not selected:
                continue
            output: dict[str, Any] = {
                "target_context_tokens": length_label,
                "class": category,
                "n": len(selected),
            }
            for field_name in (
                "suppression_gap",
                "attention_probability",
                "dm_dscore",
                "suppression_x_dm_dscore",
                "positive_suppression_x_dm_dscore",
                "locos_direct_ov_gold",
                "locos_direct_ov_margin",
                "direct_ov_centered_margin_derivative",
                "suppression_x_direct_ov_centered_margin",
            ):
                values = [float(row[field_name]) for row in selected]
                output[f"mean_{field_name}"] = rounded(statistics.fmean(values))
                output[f"mean_abs_{field_name}"] = rounded(
                    statistics.fmean(abs(value) for value in values)
                )
                output[f"positive_{field_name}_fraction"] = rounded(
                    sum(value > 0.0 for value in values) / len(values)
                )
            rows.append(output)
    return rows


def prediction_summary(case_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    interventions = [
        row
        for row in case_rows
        if row.get("plan_kind") in PLAN_KINDS
        and "predicted_first_order_delta_gold_conflict_margin" in row
    ]
    lengths = sorted({int(row["target_context_tokens"]) for row in interventions})
    scopes = sorted(
        {str(row.get("intervention_scope", "joint")) for row in interventions}
    )
    groups: list[tuple[str, str, str, str, list[dict[str, Any]]]] = []
    for scope in scopes:
        scope_rows = [
            row
            for row in interventions
            if str(row.get("intervention_scope", "joint")) == scope
        ]
        length_groups: list[tuple[str, list[dict[str, Any]]]] = [
            *(
                (
                    str(length),
                    [
                        row
                        for row in scope_rows
                        if int(row["target_context_tokens"]) == length
                    ],
                )
                for length in lengths
            ),
        ]
        # An all-length aggregate is useful only when it combines at least two
        # distinct lengths.  Emitting it for a one-length smoke produces ten
        # byte-for-byte redundant rows and looks like duplicated interventions.
        if len(lengths) > 1:
            length_groups.append(("all", scope_rows))
        for length_label, subset in length_groups:
            for plan_kind in PLAN_KINDS:
                plan_rows = [
                    row for row in subset if row["plan_kind"] == plan_kind
                ]
                groups.append(
                    (scope, length_label, plan_kind, "all", plan_rows)
                )
                for category in CLASS_ORDER:
                    groups.append(
                        (
                            scope,
                            length_label,
                            plan_kind,
                            category,
                            [
                                row
                                for row in plan_rows
                                if row["intervention_class"] == category
                            ],
                        )
                    )
    output: list[dict[str, Any]] = []
    for scope, length_label, plan_kind, category, rows in groups:
        if not rows:
            continue
        predicted = [
            float(row["predicted_first_order_delta_gold_conflict_margin"])
            for row in rows
        ]
        actual = [float(row["actual_delta_gold_conflict_margin"]) for row in rows]
        summary_row: dict[str, Any] = {
            "intervention_scope": scope,
            "target_context_tokens": length_label,
            "plan_kind": plan_kind,
            "intervention_class": category,
            "n": len(rows),
            "mean_predicted_delta_margin": rounded(
                statistics.fmean(predicted)
            ),
            "mean_actual_delta_margin": rounded(statistics.fmean(actual)),
            "pearson_predicted_vs_actual": _finite_or_na(
                correlation(predicted, actual)
            ),
            "spearman_predicted_vs_actual": _finite_or_na(
                spearman_correlation(predicted, actual)
            ),
            "sign_accuracy": rounded(
                statistics.fmean(
                    float(row["first_order_sign_match"]) for row in rows
                )
            ),
            "informative_sign_accuracy": rounded(
                statistics.fmean(
                    float(row["first_order_sign_match"])
                    for row in rows
                    if int(row["first_order_sign_informative"]) == 1
                )
            )
            if any(
                int(row["first_order_sign_informative"]) == 1 for row in rows
            )
            else "NA",
            "mean_absolute_closure_error": rounded(
                statistics.fmean(
                    float(row["first_order_absolute_closure_error"])
                    for row in rows
                )
            ),
            "mean_symmetric_closure_error": rounded(
                statistics.fmean(
                    float(row["first_order_symmetric_closure_error"])
                    for row in rows
                )
            ),
            "mean_delta_gold_nll": rounded(
                statistics.fmean(float(row["delta_gold_nll"]) for row in rows)
            ),
            "gold_ppl_exp_mean_nll": rounded(
                math.exp(
                    min(
                        statistics.fmean(
                            float(row["gold_nll"]) for row in rows
                        ),
                        700.0,
                    )
                )
            ),
            "next_token_accuracy": rounded(
                statistics.fmean(
                    float(row["next_token_correct"]) for row in rows
                )
            ),
        }
        proxy_fields = (
            "predicted_direct_ov_delta_gold_conflict_margin",
            "selected_suppression_gap_sum",
            "selected_attention_probability_sum",
            "selected_suppression_x_exact_sensitivity_sum",
            "selected_suppression_x_direct_ov_sum",
            "selected_locos_direct_ov_gold_sum",
            "selected_locos_direct_ov_margin_sum",
        )
        for field_name in proxy_fields:
            if not all(field_name in row for row in rows):
                continue
            proxy = [float(row[field_name]) for row in rows]
            summary_row[f"pearson_{field_name}_vs_actual"] = _finite_or_na(
                correlation(proxy, actual)
            )
            summary_row[f"spearman_{field_name}_vs_actual"] = _finite_or_na(
                spearman_correlation(proxy, actual)
            )
        output.append(summary_row)
    return output


def collect_raw(
    source_dirs: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for source in source_dirs:
        raw = source / "raw"
        for result_path in sorted(raw.glob("*_result.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            case_rows.extend(result["case_rows"])
        for sample_path in sorted(raw.glob("*_value_samples.jsonl")):
            samples.extend(read_jsonl(sample_path))
    return case_rows, samples


MERGE_PARTITION_CONFIG_FIELDS = frozenset(
    {
        "output_dir",
        "lengths",
        "resolved_lengths",
        "seed_start",
        "num_seeds",
        "cuda_visible_devices",
        "merge_shards",
    }
)


def build_merge_config(
    output_dir: Path,
    source_dirs: Sequence[Path],
    merge_shards_argument: str,
) -> dict[str, Any]:
    """Build merge provenance from the configs that produced each shard.

    Merge-mode CLI defaults are intentionally excluded: they did not produce
    the shard data and therefore are not experiment provenance.
    """

    if not source_dirs:
        raise ValueError("at least one shard is required for merge provenance")

    shard_entries: list[dict[str, Any]] = []
    shard_configs: list[dict[str, Any]] = []
    for source in source_dirs:
        resolved_source = source.resolve()
        config_path = resolved_source / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"shard config is missing: {config_path}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid shard config JSON: {config_path}") from error
        if not isinstance(config, dict):
            raise ValueError(f"shard config must be a JSON object: {config_path}")
        shard_configs.append(config)
        shard_entries.append(
            {
                "source_dir": str(resolved_source),
                "config_path": str(config_path),
                "config": config,
            }
        )

    shared_config: dict[str, Any] = {}
    all_fields = set().union(*(config.keys() for config in shard_configs))
    missing = object()
    for field in sorted(all_fields - MERGE_PARTITION_CONFIG_FIELDS):
        reference = shard_configs[0].get(field, missing)
        for shard_index, config in enumerate(shard_configs[1:], start=1):
            current = config.get(field, missing)
            if current != reference:
                reference_value = "<missing>" if reference is missing else reference
                current_value = "<missing>" if current is missing else current
                raise ValueError(
                    f"inconsistent shard config field {field!r}: "
                    f"{shard_entries[0]['source_dir']}={reference_value!r}, "
                    f"{shard_entries[shard_index]['source_dir']}={current_value!r}"
                )
        if reference is not missing:
            shared_config[field] = reference

    return {
        "merge_schema_version": 2,
        "resolved_sources": [entry["source_dir"] for entry in shard_entries],
        "merge_invocation": {
            "output_dir": str(output_dir.resolve()),
            "merge_shards": merge_shards_argument,
        },
        "partition_config_fields": sorted(MERGE_PARTITION_CONFIG_FIELDS),
        "shared_config": shared_config,
        "shards": shard_entries,
    }


def write_aggregate_outputs(output_dir: Path, source_dirs: Sequence[Path]) -> None:
    case_rows, samples = collect_raw(source_dirs)
    sample_rows = sample_summary(samples)
    prediction_rows = prediction_summary(case_rows)
    write_csv(output_dir / "case_rows.csv", case_rows)
    write_csv(output_dir / "value_samples.csv", samples)
    write_csv(output_dir / "value_sample_summary.csv", sample_rows)
    write_csv(output_dir / "first_order_prediction_summary.csv", prediction_rows)
    singleton_prediction_rows = [
        row
        for row in prediction_rows
        if row.get("intervention_scope") == "singleton"
    ]
    write_csv(
        output_dir / "singleton_prediction_summary.csv",
        singleton_prediction_rows,
    )
    write_json(
        output_dir / "summary.json",
        {
            "source_dirs": [str(path) for path in source_dirs],
            "case_row_count": len(case_rows),
            "value_sample_count": len(samples),
            "oracle_diagnostic_only": True,
            "oracle_gradient_target": ORACLE_GRADIENT_TARGET,
            "value_sample_summary": sample_rows,
            "first_order_prediction_summary": prediction_rows,
            "singleton_prediction_summary": singleton_prediction_rows,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oracle value-mediated RoPE suppression probe"
    )
    parser.add_argument("--model-name-or-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lengths", default="8192,32768")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--class-sample-count", type=int, default=8)
    parser.add_argument("--packet-gap-tokens", type=int, default=16)
    parser.add_argument("--anchor-distances", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--fixed-anchor-distance", type=int, default=128)
    parser.add_argument("--score-lift", type=float, default=SCORE_LIFT_CAP)
    parser.add_argument("--singleton-top-n", type=int, default=16)
    parser.add_argument(
        "--singleton-ranking-metric",
        choices=SINGLETON_RANKING_METRICS,
        default=DEFAULT_SINGLETON_RANKING_METRIC,
    )
    parser.add_argument(
        "--run-joint-interventions",
        action="store_true",
        help="also run the old all-layer/all-head nonlinear stress arm",
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help=(
            "use NF4 weights; omit this flag for an unquantized model in "
            "--dtype (BF16 by default)"
        ),
    )
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--merge-shards", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[int], tuple[int, ...]]:
    lengths = sorted(set(parse_int_csv(args.lengths)))
    anchors = tuple(sorted(set(parse_int_csv(args.anchor_distances))))
    if not lengths or min(lengths) <= 1:
        raise ValueError("lengths must be greater than one token")
    if not anchors or min(anchors) < 0:
        raise ValueError("anchor distances must be non-negative")
    if int(args.fixed_anchor_distance) not in anchors:
        raise ValueError("fixed anchor distance must occur in --anchor-distances")
    if int(args.class_sample_count) < 2:
        raise ValueError("at least two class samples are required for random control")
    if int(args.num_seeds) < 1 or int(args.prefill_chunk_size) < 1:
        raise ValueError("seed count and prefill chunk size must be positive")
    if not (0.0 < float(args.score_lift) <= SCORE_LIFT_CAP + 1e-12):
        raise ValueError(f"score lift must be in (0,{SCORE_LIFT_CAP}]")
    if int(args.singleton_top_n) < 0:
        raise ValueError("singleton top-N must be non-negative")
    if (
        int(args.singleton_top_n) == 0
        and not bool(args.run_joint_interventions)
    ):
        raise ValueError(
            "enable singleton candidates or --run-joint-interventions"
        )
    if str(args.singleton_ranking_metric) not in SINGLETON_RANKING_METRICS:
        raise ValueError("unknown singleton ranking metric")
    if str(args.attn_implementation) != "eager":
        raise ValueError("this causal probe requires --attn-implementation eager")
    return lengths, anchors


def _case_file_stem(length: int, seed: int) -> str:
    return f"length_{int(length)}_seed_{int(seed)}"


def run_case(
    model: Any,
    tokenizer: Any,
    answer_ids: dict[str, int],
    case: dict[str, Any],
    args: argparse.Namespace,
    anchor_distances: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt = torch.tensor(case["prompt_ids"], dtype=torch.long).view(1, -1)
    prefix_length = int(prompt.shape[1]) - 1
    base.synchronize()
    started = time.perf_counter()
    legacy, prefill_seconds = prefill_sequence_no_grad(
        model, prompt[:, :-1], args.prefill_chunk_size
    )
    immutable_signature = prefix_cache_signature(legacy)

    base.synchronize()
    native_started = time.perf_counter()
    cache = base.cache_from_legacy(legacy)
    with torch.inference_mode():
        native_output = forward_final_with_hidden_states(
            model,
            prompt[:, -1:].to(base.input_device(model)),
            cache,
            prefix_length,
        )
    base.synchronize()
    native_seconds = time.perf_counter() - native_started
    native_answer = answer_metrics_with_fp32_pair(
        model, tokenizer, native_output, answer_ids, case["conflict_answer"]
    )
    del native_output, cache

    baseline_controller = ValueMediatedController(
        mode="baseline",
        case=case,
        anchor_distances=anchor_distances,
        fixed_anchor_distance=int(args.fixed_anchor_distance),
        score_lift=float(args.score_lift),
        gold_token_id=int(answer_ids["nine"]),
        conflict_token_id=int(answer_ids[case["conflict_answer"]]),
    )
    base.synchronize()
    baseline_started = time.perf_counter()
    cache = base.cache_from_legacy(legacy)
    model.zero_grad(set_to_none=True)
    with activate(baseline_controller), torch.enable_grad():
        baseline_output, final_embedding = forward_final_with_autograd(
            model, prompt[:, -1:], cache, prefix_length
        )
        gold_id = int(answer_ids["nine"])
        conflict_id = int(answer_ids[case["conflict_answer"]])
        gold_logit, conflict_logit, _ = fp32_pair_logit_tensors(
            model, baseline_output, gold_id, conflict_id
        )
        oracle_margin = gold_logit - conflict_logit
        instrumented_answer = answer_metrics_with_fp32_pair(
            model,
            tokenizer,
            baseline_output,
            answer_ids,
            case["conflict_answer"],
        )
        oracle_margin.backward()
    baseline_controller.finalize_gradients()
    frozen_candidates = freeze_singleton_candidates(
        baseline_controller.sample_rows,
        top_n=int(args.singleton_top_n),
        ranking_metric=str(args.singleton_ranking_metric),
        seed=int(case["seed"]),
    )
    base.synchronize()
    baseline_seconds = time.perf_counter() - baseline_started
    input_gradient_norm = float(final_embedding.grad.float().norm().item())
    del oracle_margin, gold_logit, conflict_logit
    del baseline_output, final_embedding, cache
    model.zero_grad(set_to_none=True)

    instrumentation_delta = safety.delta_metrics(
        instrumented_answer, native_answer
    )

    # Run one inference-mode epsilon=0 replay through the exact same patched
    # attention path used by every intervention.  This removes the common
    # autograd-vs-inference and input-embedding-vs-input-ID numerical offset
    # from all reported causal deltas.
    noop_sample = baseline_controller.sample_rows[0]
    noop_controller = ValueMediatedController(
        mode="intervene",
        case=case,
        anchor_distances=anchor_distances,
        fixed_anchor_distance=int(args.fixed_anchor_distance),
        score_lift=0.0,
        target_class=str(noop_sample["class"]),
        plan_kind="target",
        intervention_scope="singleton",
        singleton_layer=int(noop_sample["layer"]),
        singleton_head=int(noop_sample["head"]),
        singleton_position=int(noop_sample["token_position"]),
    )
    cache = base.cache_from_legacy(legacy)
    base.synchronize()
    noop_started = time.perf_counter()
    with activate(noop_controller), torch.inference_mode():
        noop_output = forward_final_with_hidden_states(
            model,
            prompt[:, -1:].to(base.input_device(model)),
            cache,
            prefix_length,
        )
    base.synchronize()
    noop_seconds = time.perf_counter() - noop_started
    if noop_controller.applied_count != 0:
        raise RuntimeError("epsilon=0 no-op unexpectedly changed an attention score")
    noop_answer = answer_metrics_with_fp32_pair(
        model, tokenizer, noop_output, answer_ids, case["conflict_answer"]
    )
    noop_delta_from_native = safety.delta_metrics(noop_answer, native_answer)
    noop_delta_from_instrumented = safety.delta_metrics(
        noop_answer, instrumented_answer
    )
    del noop_output, cache

    common = {
        "target_context_tokens": int(case["total_tokens"]),
        "seed": int(case["seed"]),
        "gold_answer": "nine",
        "conflict_answer": case["conflict_answer"],
        "gold_output": case["gold_output"],
        "conflict_output": case["conflict_output"],
        "oracle_gradient_target": ORACLE_GRADIENT_TARGET,
        "oracle_diagnostic_only": 1,
        "score_lift_cap": rounded(args.score_lift),
        "pair_margin_compute_dtype": "float32",
        "pair_margin_hidden_precision_limited": int(
            str(args.dtype) != "float32"
        ),
    }
    zero_delta = {
        "delta_gold_nll": 0.0,
        "gold_ppl_ratio": 1.0,
        "delta_gold_ppl": 0.0,
        "delta_gold_full_vocab_margin": 0.0,
        "delta_gold_conflict_margin": 0.0,
    }
    case_rows: list[dict[str, Any]] = [
        {
            **common,
            "intervention_class": "native_baseline",
            "plan_kind": "none",
            "intervention_scope": "none",
            "comparison_baseline": "native_baseline",
            **native_answer,
            **zero_delta,
            "prefill_seconds": rounded(prefill_seconds),
            "query_seconds": rounded(native_seconds),
        },
        {
            **common,
            "intervention_class": "instrumented_baseline",
            "plan_kind": "none",
            "intervention_scope": "none",
            "comparison_baseline": "native_baseline",
            **instrumented_answer,
            **instrumentation_delta,
            "input_embedding_gradient_norm": rounded(input_gradient_norm),
            "prefill_seconds": rounded(prefill_seconds),
            "query_seconds": rounded(baseline_seconds),
        },
        {
            **common,
            "intervention_class": CUSTOM_NOOP_BASELINE,
            "plan_kind": "none",
            "intervention_scope": "noop",
            "comparison_baseline": CUSTOM_NOOP_BASELINE,
            "causal_delta_reference": CUSTOM_NOOP_BASELINE,
            "epsilon_zero_noop_control": 1,
            "uniform_score_lift": 0.0,
            "applied_count": 0,
            "replayed_coordinate_count": 1,
            "singleton_layer": int(noop_sample["layer"]),
            "singleton_head": int(noop_sample["head"]),
            "singleton_position": int(noop_sample["token_position"]),
            **noop_answer,
            **zero_delta,
            "delta_from_native_gold_conflict_margin": noop_delta_from_native[
                "delta_gold_conflict_margin"
            ],
            "delta_from_instrumented_gold_conflict_margin": (
                noop_delta_from_instrumented["delta_gold_conflict_margin"]
            ),
            "delta_from_native_gold_nll": noop_delta_from_native[
                "delta_gold_nll"
            ],
            "delta_from_instrumented_gold_nll": (
                noop_delta_from_instrumented["delta_gold_nll"]
            ),
            "prefill_seconds": rounded(prefill_seconds),
            "query_seconds": rounded(noop_seconds),
        },
    ]
    singleton_interventions: list[dict[str, Any]] = []
    for candidate in frozen_candidates:
        category = str(candidate["class"])
        for plan_kind in PLAN_KINDS:
            selected = candidate[plan_kind]
            controller = ValueMediatedController(
                mode="intervene",
                case=case,
                anchor_distances=anchor_distances,
                fixed_anchor_distance=int(args.fixed_anchor_distance),
                score_lift=float(args.score_lift),
                target_class=category,
                plan_kind=plan_kind,
                intervention_scope="singleton",
                singleton_layer=int(selected["layer"]),
                singleton_head=int(selected["head"]),
                singleton_position=int(selected["token_position"]),
            )
            predictors = singleton_predictors(selected, float(args.score_lift))
            predicted = predictors[
                "predicted_first_order_delta_gold_conflict_margin"
            ]
            cache = base.cache_from_legacy(legacy)
            base.synchronize()
            query_started = time.perf_counter()
            with activate(controller), torch.inference_mode():
                output = forward_final_with_hidden_states(
                    model,
                    prompt[:, -1:].to(base.input_device(model)),
                    cache,
                    prefix_length,
                )
            base.synchronize()
            query_seconds = time.perf_counter() - query_started
            if controller.applied_count != 1:
                raise RuntimeError(
                    "singleton replay must modify exactly one attention score"
                )
            answer = answer_metrics_with_fp32_pair(
                model, tokenizer, output, answer_ids, case["conflict_answer"]
            )
            delta = safety.delta_metrics(answer, noop_answer)
            actual = float(delta["delta_gold_conflict_margin"])
            first_order = closure_metrics(predicted, actual)
            intervention_metadata = controller.intervention_summary()
            row = {
                **common,
                "intervention_class": category,
                "plan_kind": plan_kind,
                "comparison_baseline": CUSTOM_NOOP_BASELINE,
                "causal_delta_reference": CUSTOM_NOOP_BASELINE,
                "epsilon_zero_noop_control": 0,
                "reference_gold_conflict_margin": noop_answer[
                    "gold_conflict_margin"
                ],
                "reference_gold_nll": noop_answer["gold_nll"],
                "pair_id": candidate["pair_id"],
                "candidate_rank": int(candidate["candidate_rank"]),
                "candidate_ranking_metric": candidate["ranking_metric"],
                "candidate_ranking_value": candidate["ranking_value"],
                "candidate_frozen_before_intervention": int(
                    candidate["candidate_frozen_before_intervention"]
                ),
                "matched_layer_head_class": int(
                    candidate["matched_layer_head_class"]
                ),
                "selected_baseline_layer": int(selected["layer"]),
                "selected_baseline_head": int(selected["head"]),
                "selected_baseline_token_position": int(
                    selected["token_position"]
                ),
                "selected_baseline_sample_index": int(
                    selected["sample_index"]
                ),
                "selected_baseline_dm_dscore": selected["dm_dscore"],
                "selected_baseline_suppression_gap": selected[
                    "suppression_gap"
                ],
                "selected_baseline_attention_probability": selected[
                    "attention_probability"
                ],
                **answer,
                **delta,
                **first_order,
                **{name: rounded(value) for name, value in predictors.items()},
                **intervention_metadata,
                "prefill_seconds": rounded(prefill_seconds),
                "query_seconds": rounded(query_seconds),
            }
            case_rows.append(row)
            singleton_interventions.append(row)
            del output, cache

    joint_interventions: dict[str, Any] = {}
    if bool(args.run_joint_interventions):
        for category in CLASS_ORDER:
            joint_interventions[category] = {}
            for plan_kind in PLAN_KINDS:
                controller = ValueMediatedController(
                    mode="intervene",
                    case=case,
                    anchor_distances=anchor_distances,
                    fixed_anchor_distance=int(args.fixed_anchor_distance),
                    score_lift=float(args.score_lift),
                    target_class=category,
                    plan_kind=plan_kind,
                    intervention_scope="joint",
                    plan=baseline_controller.plan,
                )
                predictors = baseline_controller.plan_predictors(
                    category, plan_kind
                )
                predicted = predictors[
                    "predicted_first_order_delta_gold_conflict_margin"
                ]
                cache = base.cache_from_legacy(legacy)
                base.synchronize()
                query_started = time.perf_counter()
                with activate(controller), torch.inference_mode():
                    output = forward_final_with_hidden_states(
                        model,
                        prompt[:, -1:].to(base.input_device(model)),
                        cache,
                        prefix_length,
                    )
                base.synchronize()
                query_seconds = time.perf_counter() - query_started
                answer = answer_metrics_with_fp32_pair(
                    model,
                    tokenizer,
                    output,
                    answer_ids,
                    case["conflict_answer"],
                )
                delta = safety.delta_metrics(answer, noop_answer)
                actual = float(delta["delta_gold_conflict_margin"])
                first_order = closure_metrics(predicted, actual)
                intervention_metadata = controller.intervention_summary()
                row = {
                    **common,
                    "intervention_class": category,
                    "plan_kind": plan_kind,
                    "comparison_baseline": CUSTOM_NOOP_BASELINE,
                    "causal_delta_reference": CUSTOM_NOOP_BASELINE,
                    "epsilon_zero_noop_control": 0,
                    "reference_gold_conflict_margin": noop_answer[
                        "gold_conflict_margin"
                    ],
                    "reference_gold_nll": noop_answer["gold_nll"],
                    **answer,
                    **delta,
                    **first_order,
                    **{
                        name: rounded(value)
                        for name, value in predictors.items()
                    },
                    **intervention_metadata,
                    "prefill_seconds": rounded(prefill_seconds),
                    "query_seconds": rounded(query_seconds),
                }
                case_rows.append(row)
                joint_interventions[category][plan_kind] = row
                del output, cache

    if prefix_cache_signature(legacy) != immutable_signature:
        raise RuntimeError("the shared prefix cache was mutated by a final-query replay")

    case_replay_audit = audit_noop_referenced_case(
        case_rows, frozen_candidates, noop_answer
    )

    sample_rows = [{**common, **row} for row in baseline_controller.sample_rows]
    result = {
        "schema_version": 3,
        "experiment": "value_mediated_rope_probe_qwen3_8b",
        "case": safety.public_case(case),
        "native_baseline_answer": native_answer,
        "instrumented_baseline_answer": instrumented_answer,
        "instrumentation_delta_from_native": instrumentation_delta,
        "custom_noop_baseline_answer": noop_answer,
        "custom_noop_delta_from_native": noop_delta_from_native,
        "custom_noop_delta_from_instrumented": noop_delta_from_instrumented,
        "causal_delta_reference": CUSTOM_NOOP_BASELINE,
        "case_replay_audit": case_replay_audit,
        "oracle_gradient_target": ORACLE_GRADIENT_TARGET,
        "oracle_diagnostic_only": True,
        "not_a_deployment_selector": True,
        "input_embedding_gradient_norm": rounded(input_gradient_norm),
        "singleton_candidate_ranking_metric": str(
            args.singleton_ranking_metric
        ),
        "singleton_top_n_per_class": int(args.singleton_top_n),
        "frozen_singleton_candidates": frozen_candidates,
        "singleton_interventions": singleton_interventions,
        "joint_interventions_enabled": bool(args.run_joint_interventions),
        "joint_interventions": joint_interventions,
        "interventions": {
            "singleton": singleton_interventions,
            "joint": joint_interventions,
        },
        "case_rows": case_rows,
        "certificate_reconstruction_error_max": rounded(
            baseline_controller.reconstruction_error_max
        ),
        "prefix_cache_immutable": True,
        "timing": {
            "prefill_seconds": rounded(prefill_seconds),
            "baseline_query_seconds": rounded(baseline_seconds),
            "custom_noop_query_seconds": rounded(noop_seconds),
            "total_seconds": rounded(time.perf_counter() - started),
        },
    }
    del legacy, prompt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, sample_rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_shards:
        sources = [
            Path(item.strip())
            for item in args.merge_shards.split(",")
            if item.strip()
        ]
        if not sources:
            raise ValueError("--merge-shards supplied no source directories")
        merge_config = build_merge_config(
            output_dir, sources, args.merge_shards
        )
        write_aggregate_outputs(output_dir, sources)
        write_json(output_dir / "merge_config.json", merge_config)
        print(f"merged {len(sources)} shards into {output_dir}", flush=True)
        return

    lengths, anchors = validate_args(args)
    if not args.model_name_or_path:
        raise ValueError("--model-name-or-path is required unless --merge-shards is used")
    config = {
        **vars(args),
        "resolved_lengths": lengths,
        "resolved_anchor_distances": anchors,
        "class_order": CLASS_ORDER,
        "plan_kinds": PLAN_KINDS,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "frozen_final_query": True,
        "immutable_prefix_cache": True,
        "oracle_gradient_target": ORACLE_GRADIENT_TARGET,
        "oracle_diagnostic_only": True,
        "not_a_deployment_method": True,
        "answer_labels_used_for_selection": bool(
            int(args.singleton_top_n) > 0
            and str(args.singleton_ranking_metric)
            != "positive_suppression_gap"
        ),
        "answer_labels_used_for_gradient_audit_only": False,
        "answer_labels_used_for_gradient_audit": True,
        "singleton_candidates_frozen_before_intervention": True,
        "singleton_primary_causal_validation": int(args.singleton_top_n) > 0,
        "joint_interventions_default_off": True,
        "causal_final_query_cache_path": "read_only_layer_local_concat",
        "causal_delta_reference": CUSTOM_NOOP_BASELINE,
        "epsilon_zero_custom_attention_noop": True,
        "model_weight_mode": (
            "nf4_quantized" if bool(args.load_in_4bit) else f"unquantized_{args.dtype}"
        ),
        "pair_margin_compute_dtype": "float32",
        "pair_margin_hidden_precision_limited": True,
        "full_vocab_metrics_source": "native_model_logits",
        "uniform_score_lift": float(args.score_lift),
    }
    write_json(output_dir / "config.json", config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    previews = [
        safety.public_case(
            safety.build_case(
                tokenizer,
                total_tokens=length,
                seed=args.seed_start,
                packet_gap_tokens=args.packet_gap_tokens,
                class_sample_count=args.class_sample_count,
            )
        )
        for length in lengths
    ]
    write_json(output_dir / "design.json", {"config": config, "cases": previews})
    if args.dry_run:
        print(
            json.dumps({"config": config, "cases": previews}, ensure_ascii=False, indent=2)
        )
        return

    model, tokenizer = safety.load_model(args, max(lengths))
    freeze_model_parameters(model)
    patch_model(model)
    answer_ids = safety.answer_token_ids(tokenizer)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for length in lengths:
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            stem = _case_file_stem(length, seed)
            result_path = raw_dir / f"{stem}_result.json"
            sample_path = raw_dir / f"{stem}_value_samples.jsonl"
            if result_path.exists() and sample_path.exists():
                print(f"length={length} seed={seed} already complete", flush=True)
                continue
            case = safety.build_case(
                tokenizer,
                total_tokens=length,
                seed=seed,
                packet_gap_tokens=args.packet_gap_tokens,
                class_sample_count=args.class_sample_count,
            )
            result, samples = run_case(
                model, tokenizer, answer_ids, case, args, anchors
            )
            temporary_samples = sample_path.with_suffix(sample_path.suffix + ".tmp")
            if temporary_samples.exists():
                temporary_samples.unlink()
            append_jsonl(temporary_samples, samples)
            temporary_samples.replace(sample_path)
            write_json(result_path, result)
            print(
                f"length={length} seed={seed} ppl="
                f"{result['custom_noop_baseline_answer']['gold_ppl']:.4f} "
                f"margin="
                f"{result['custom_noop_baseline_answer']['gold_conflict_margin']:.4f} "
                f"noop_minus_instrumented_margin="
                f"{result['custom_noop_delta_from_instrumented']['delta_gold_conflict_margin']:.4f} "
                f"input_grad_norm={result['input_embedding_gradient_norm']:.4f} "
                f"reconstruction_error="
                f"{result['certificate_reconstruction_error_max']:.6g}",
                flush=True,
            )

    write_aggregate_outputs(output_dir, [output_dir])
    (output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
