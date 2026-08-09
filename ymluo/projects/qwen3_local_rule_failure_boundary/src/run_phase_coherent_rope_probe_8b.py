from __future__ import annotations

"""Screen local-order/global-semantic rotary kernels on frozen Qwen3-8B.

This module deliberately reuses the controlled two-hop data, cache handling,
metrics, and baselines from ``run_local_global_rope_probe_8b``.  It adds
full-attention diagnostic variants that change only the final-query attention
kernel.  Prefix representations remain those of the pretrained model; the
experiment is therefore an inference-time causal screen, not yet a claim about
training a model end to end with the new kernel.
"""

import math
import os
import statistics
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from phase_rescue_solver import phase_lift, solve_phase_rescue
import run_local_global_rope_probe_8b as runner
import run_rope_retrieval_repair_8b as rope_repair


COUNTERFACTUAL_SELECTION_VARIANTS = (
    "cfs_w128_lift25_postscore",
    "cfs_w128_lift50_postscore",
    "cfs_w128_lift100_postscore",
    "cfs_w128_lift50_gap1_postscore",
    "cfs_dual_w128_lift25_postscore",
    "cfs_dual_w128_lift50_postscore",
    "cfs_w4k_lift25_postscore",
    "cfs_w4k_lift50_postscore",
    "cfs_dual_w4k_lift25_postscore",
)


LEGACY_STRICT_MPR_VARIANTS = (
    "strict_mpr_pre_w128_lift25_gap1_f8_cap0p25",
    "strict_mpr_pre_w128_lift25_gap1_f8_cap0p25_random",
    "strict_mpr_pre_w128_lift25_gap1_f8_cap0p25_masspreserve",
    "strict_mpr_pre_w128_lift25_gap1_f8_cap0p25_random_masspreserve",
)


TOKEN_SPARSE_STRICT_MPR_VARIANTS = (
    "strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25",
    "strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25_random",
    "strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25_masspreserve",
    "strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25_random_masspreserve",
    "strict_mpr_pre_w128_lift25_gap1_t4_f8_cap0p25",
    "strict_mpr_pre_w128_lift25_gap1_t4_f8_cap0p25_random",
    "strict_mpr_pre_w128_lift25_gap1_t4_f8_cap0p25_masspreserve",
    "strict_mpr_pre_w128_lift25_gap1_t4_f8_cap0p25_random_masspreserve",
)


# Keep the earlier uncapped variants importable for existing analysis scripts,
# but use the explicitly named t1/t4 family in all new screens.
STRICT_MPR_VARIANTS = (
    *LEGACY_STRICT_MPR_VARIANTS,
    *TOKEN_SPARSE_STRICT_MPR_VARIANTS,
)


MINIMAL_RESCUE_VARIANTS = (
    "exact_pre_top2_postscore",
    "exact_pre_top2_blend25",
    "exact_dual_top2_postscore",
    "clipped128_top2_clippedscore",
    "clipped128_top2_postscore",
    "pre_top2_clipped128score",
    "mpr_pre_w128_lift25",
    "mpr_pre_w128_lift50",
    "mpr_dual_w128_lift25",
    "mpr_dual_w128_lift50",
    "mpr_dual_w128_lift25_masspreserve",
    "mpr_pre_w128_lift50_gap1",
    "mpr_dual_w128_lift25_gap1",
    "mpr_dual_w128_lift50_gap1",
    "mpr_dual_w128_lift25_gap1_masspreserve",
    "mpr_pre_w128_lift25_gap1_f8",
    "mpr_pre_w128_lift25_gap1_f16",
    "mpr_dual_w128_lift25_gap1_f8",
    "mpr_dual_w128_lift25_gap1_f16",
    "mpr_pre_w4k_lift25",
    "mpr_pre_w4k_lift25_masspreserve",
    "mpr_pre_w4k_lift25_gap1",
    "mpr_dual_w4k_lift25",
    "mpr_dual_w4k_lift25_gap1",
    "mpr_dual_w4k_lift25_gap1_f8",
    *STRICT_MPR_VARIANTS,
    *COUNTERFACTUAL_SELECTION_VARIANTS,
)


NEW_VARIANTS = (
    "relative_rope_reconstructed_full",
    "remote_nope_raw_full",
    "remote_nope_cal_full",
    "distance_fade_4k_full",
    "distance_fade_8k_full",
    "distance_fade_16k_full",
    "phase_coherent_c1_full",
    "phase_coherent_c2_full",
    "phase_coherent_c4_full",
    "phase_coherent_c2_cal_full",
    "phase_coherent_w4k_c1_full",
    "phase_coherent_w4k_c4_full",
    "phase_coherent_w4k_c16_full",
    "phase_coherent_w4k_c4_cal_full",
    "phase_coherent_w8k_c4_cal_full",
    "phase_coherent_norm_w4k_c4_full",
    "phase_coherent_norm_w4k_c4_cal_full",
    "phase_return_c1_full",
    "phase_return_c2_full",
    "phase_return_c4_full",
    "phase_clamp_c2_full",
    "distance_saturate_w4k_t4k_full",
    "distance_saturate_w4k_t16k_full",
    "distance_log_w4k_t4k_full",
    *MINIMAL_RESCUE_VARIANTS,
)
runner.VARIANTS = tuple(dict.fromkeys((*runner.VARIANTS, *NEW_VARIANTS)))
_BASE_FORWARD = runner.local_global_attention_forward
_BASE_PATCH_MODEL = runner.patch_model
_BASE_PREFILL_SEQUENCE = runner.base.prefill_sequence
_CAPTURE_PREFIX_KEYS = False
_STRICT_REFERENCE_EPOCH = 0
_PREFIX_KEY_STORAGE = os.environ.get("PHASE_PREKEY_STORAGE", "cuda").lower()
if _PREFIX_KEY_STORAGE not in {"cuda", "cpu"}:
    raise ValueError("PHASE_PREKEY_STORAGE must be 'cuda' or 'cpu'")
_BASE_METRIC_SUMMARY = runner.MetricAccumulator.summary
_BASE_SUMMARIZE = runner.summarize


@dataclass(frozen=True)
class FrozenStrictReferencePlan:
    """Exact-baseline decisions replayed by every strict-MPR variant.

    The plan deliberately freezes token positions, remote/trigger masks, and
    numeric target lifts.  It does not freeze later-layer hidden states: each
    intervention still acts on the variant's current Q/K geometry, but Q drift
    can no longer cause candidate reselection or change which pairs are treated.
    """

    epoch: int
    layer_idx: int
    key_count: int
    keep_count: int
    head_count: int
    token_cap: int
    signature: tuple[float, float, float, int, int, float]
    selected: torch.Tensor
    selected_remote: torch.Tensor
    selected_delta: torch.Tensor
    rescue_eligible: torch.Tensor
    raw_trigger: torch.Tensor
    trigger: torch.Tensor
    desired_lift: torch.Tensor
    counterfactual_gap: torch.Tensor


@dataclass(frozen=True)
class FrozenStrictInterventionReference:
    """Per-pair treatment strength captured from the non-random strict arm."""

    epoch: int
    layer_idx: int
    key_count: int
    keep_count: int
    head_count: int
    token_cap: int
    signature: tuple[float, float, float, int, int, float]
    shift_norm: torch.Tensor
    support_count: torch.Tensor
    achieved_lift: torch.Tensor


def _phase_metric_summary(self: runner.MetricAccumulator) -> dict[str, float]:
    summary = _BASE_METRIC_SUMMARY(self)
    remote = int(getattr(self, "phase_remote_count", 0))
    triggers = int(getattr(self, "phase_trigger_count", 0))
    gold = int(getattr(self, "phase_gold_count", 0))
    gold_triggers = int(getattr(self, "phase_gold_trigger_count", 0))
    strict_calls = int(getattr(self, "strict_phase_solver_calls", 0))
    strict_triggers = int(getattr(self, "strict_phase_trigger_count", 0))
    strict_eligible = int(getattr(self, "strict_phase_remote_eligible_count", 0))
    strict_raw = int(getattr(self, "strict_phase_raw_trigger_count", 0))
    strict_capped = int(getattr(self, "strict_phase_capped_trigger_count", 0))
    summary.update(
        {
            "phase_rescue_trigger_fraction": triggers / max(1, remote),
            "phase_rescue_score_lift_mean": float(
                getattr(self, "phase_score_lift_sum", 0.0)
            )
            / max(1, triggers),
            "phase_rescue_shift_rms_mean": float(
                getattr(self, "phase_shift_rms_sum", 0.0)
            )
            / max(1, triggers),
            "phase_rescue_active_planes_mean": float(
                getattr(self, "phase_active_plane_sum", 0.0)
            )
            / max(1, triggers),
            "gold_phase_trigger_fraction": gold_triggers / max(1, gold),
            "selected_gold_phase_trigger_fraction": gold_triggers / max(1, gold),
            "gold_phase_score_lift_mean": float(
                getattr(self, "phase_gold_score_lift_sum", 0.0)
            )
            / max(1, gold_triggers),
            "phase_rescue_realized_lift_ratio_mean": float(
                getattr(self, "phase_realized_lift_ratio_sum", 0.0)
            )
            / max(1, triggers),
            "phase_rescue_negative_lift_fraction": float(
                getattr(self, "phase_negative_lift_count", 0)
            )
            / max(1, triggers),
            "strict_phase_solver_calls": strict_calls,
            "strict_phase_remote_eligible_count": strict_eligible,
            "strict_phase_raw_trigger_count": strict_raw,
            "strict_phase_capped_trigger_count": strict_capped,
            "strict_phase_raw_trigger_fraction": strict_raw
            / max(1, strict_eligible),
            "strict_phase_capped_trigger_fraction": strict_capped
            / max(1, strict_eligible),
            "strict_phase_max_triggers_per_head": int(
                getattr(self, "strict_phase_max_triggers_per_head", 0)
            ),
            "strict_phase_token_cap": int(
                getattr(self, "strict_phase_token_cap", 0)
            ),
            "strict_phase_token_cap_noop_count": int(
                getattr(self, "strict_phase_token_cap_noop_count", 0)
            ),
            "strict_phase_token_cap_noop_max": float(
                getattr(self, "strict_phase_token_cap_noop_max", 0.0)
            ),
            "strict_phase_frozen_token_cap_mismatch_max": int(
                getattr(self, "strict_phase_frozen_token_cap_mismatch_max", 0)
            ),
            "strict_phase_feasible_fraction": float(
                getattr(self, "strict_phase_feasible_count", 0)
            )
            / max(1, strict_calls),
            "strict_phase_target_lift_mean": float(
                getattr(self, "strict_phase_target_lift_sum", 0.0)
            )
            / max(1, strict_calls),
            "strict_phase_solver_lift_mean": float(
                getattr(self, "strict_phase_solver_lift_sum", 0.0)
            )
            / max(1, strict_calls),
            "strict_phase_applied_lift_mean": float(
                getattr(self, "strict_phase_applied_lift_sum", 0.0)
            )
            / max(1, strict_triggers),
            "strict_phase_support_mean": float(
                getattr(self, "strict_phase_support_sum", 0)
            )
            / max(1, strict_calls),
            "strict_phase_support_max": int(
                getattr(self, "strict_phase_support_max", 0)
            ),
            "strict_phase_budget_mean": float(
                getattr(self, "strict_phase_budget_sum", 0)
            )
            / max(1, strict_calls),
            "strict_phase_cap_mean": float(
                getattr(self, "strict_phase_cap_sum", 0.0)
            )
            / max(1, strict_calls),
            "strict_phase_shift_abs_max": float(
                getattr(self, "strict_phase_shift_abs_max", 0.0)
            ),
            "strict_phase_shift_l2_mean": float(
                getattr(self, "strict_phase_shift_l2_sum", 0.0)
            )
            / max(1, strict_calls),
            "strict_phase_random_reference_l2_mean": float(
                getattr(self, "strict_phase_random_reference_l2_sum", 0.0)
            )
            / max(1, strict_calls),
            "strict_phase_random_reference_lift_mean": float(
                getattr(self, "strict_phase_random_reference_lift_sum", 0.0)
            )
            / max(1, strict_calls),
            "strict_phase_random_norm_match_max": float(
                getattr(self, "strict_phase_random_norm_match_max", 0.0)
            ),
            "strict_phase_random_support_delta_max": int(
                getattr(self, "strict_phase_random_support_delta_max", 0)
            ),
            "strict_phase_nontrigger_noop_max": float(
                getattr(self, "strict_phase_nontrigger_noop_max", 0.0)
            ),
            "strict_phase_partition_error_mean": float(
                getattr(self, "strict_phase_partition_error_sum", 0.0)
            )
            / max(1, int(getattr(self, "strict_phase_partition_head_count", 0))),
            "strict_phase_partition_error_max": float(
                getattr(self, "strict_phase_partition_error_max", 0.0)
            ),
            "strict_phase_partition_head_count": int(
                getattr(self, "strict_phase_partition_head_count", 0)
            ),
            "strict_phase_partition_preserve": float(
                getattr(self, "strict_phase_partition_preserve", 0)
            ),
            "strict_phase_random_support": float(
                getattr(self, "strict_phase_random_support", 0)
            ),
            "strict_phase_exact_pre_selector": float(
                getattr(self, "strict_phase_exact_pre_selector", 0)
            ),
            "strict_phase_frozen_reference": float(
                getattr(self, "strict_phase_frozen_reference", 0)
            ),
            "strict_phase_frozen_support_mismatch_max": int(
                getattr(self, "strict_phase_frozen_support_mismatch_max", 0)
            ),
        }
    )
    return summary


runner.MetricAccumulator.summary = _phase_metric_summary


_STRICT_WEIGHTED_SUMMARY_FIELDS = (
    "strict_phase_feasible_fraction",
    "strict_phase_target_lift_mean",
    "strict_phase_solver_lift_mean",
    "strict_phase_applied_lift_mean",
    "strict_phase_support_mean",
    "strict_phase_budget_mean",
    "strict_phase_cap_mean",
    "strict_phase_shift_l2_mean",
    "strict_phase_random_reference_l2_mean",
    "strict_phase_random_reference_lift_mean",
)
_STRICT_MAX_SUMMARY_FIELDS = (
    "strict_phase_support_max",
    "strict_phase_shift_abs_max",
    "strict_phase_random_norm_match_max",
    "strict_phase_random_support_delta_max",
    "strict_phase_nontrigger_noop_max",
    "strict_phase_partition_error_max",
    "strict_phase_frozen_support_mismatch_max",
    "strict_phase_max_triggers_per_head",
    "strict_phase_token_cap_noop_max",
    "strict_phase_frozen_token_cap_mismatch_max",
)
_STRICT_FLAG_SUMMARY_FIELDS = (
    "strict_phase_partition_preserve",
    "strict_phase_random_support",
    "strict_phase_exact_pre_selector",
    "strict_phase_frozen_reference",
    "strict_phase_token_cap",
)
_STRICT_PHASE_DIAGNOSTIC_FIELDS = (
    "phase_rescue_trigger_fraction",
    "phase_rescue_score_lift_mean",
    "phase_rescue_shift_rms_mean",
    "phase_rescue_active_planes_mean",
    "gold_phase_trigger_fraction",
    "selected_gold_phase_trigger_fraction",
    "gold_phase_score_lift_mean",
    "phase_rescue_realized_lift_ratio_mean",
    "phase_rescue_negative_lift_fraction",
)


def _phase_summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain strict-MPR audit diagnostics in aggregate artifacts."""

    output = _BASE_SUMMARIZE(rows)
    grouped = {
        (int(item["target_context_tokens"]), str(item["variant"])): item
        for item in output
    }
    for key, aggregate in grouped.items():
        if key[1] not in STRICT_MPR_VARIANTS:
            continue
        selected = [
            row
            for row in rows
            if (int(row["target_context_tokens"]), str(row["variant"])) == key
        ]
        if not selected:
            continue
        weights = [max(0.0, float(row["strict_phase_solver_calls"])) for row in selected]
        total_weight = sum(weights)
        aggregate["strict_phase_solver_calls"] = int(
            sum(int(row["strict_phase_solver_calls"]) for row in selected)
        )
        for count_field in (
            "strict_phase_remote_eligible_count",
            "strict_phase_raw_trigger_count",
            "strict_phase_capped_trigger_count",
            "strict_phase_token_cap_noop_count",
        ):
            aggregate[count_field] = int(
                sum(int(row[count_field]) for row in selected)
            )
        strict_eligible = int(aggregate["strict_phase_remote_eligible_count"])
        aggregate["strict_phase_raw_trigger_fraction"] = float(
            aggregate["strict_phase_raw_trigger_count"]
        ) / max(1, strict_eligible)
        aggregate["strict_phase_capped_trigger_fraction"] = float(
            aggregate["strict_phase_capped_trigger_count"]
        ) / max(1, strict_eligible)
        for field in _STRICT_WEIGHTED_SUMMARY_FIELDS:
            if total_weight > 0.0:
                aggregate[field] = sum(
                    float(row[field]) * weight
                    for row, weight in zip(selected, weights)
                ) / total_weight
            else:
                aggregate[field] = statistics.fmean(float(row[field]) for row in selected)
        for field in _STRICT_MAX_SUMMARY_FIELDS:
            aggregate[field] = max(float(row[field]) for row in selected)
        for field in _STRICT_FLAG_SUMMARY_FIELDS:
            aggregate[field] = max(float(row[field]) for row in selected)
        for field in _STRICT_PHASE_DIAGNOSTIC_FIELDS:
            aggregate[field] = statistics.fmean(
                float(row[field]) for row in selected
            )

        partition_weight = sum(
            int(row["strict_phase_partition_head_count"]) for row in selected
        )
        aggregate["strict_phase_partition_head_count"] = partition_weight
        if partition_weight:
            aggregate["strict_phase_partition_error_mean"] = sum(
                float(row["strict_phase_partition_error_mean"])
                * int(row["strict_phase_partition_head_count"])
                for row in selected
            ) / partition_weight
        else:
            aggregate["strict_phase_partition_error_mean"] = 0.0
    return output


runner.summarize = _phase_summarize


def _phase_cutoff(variant: str) -> float:
    if "_c1_" in variant:
        return 1.0
    if "_c2_" in variant:
        return 2.0
    if "_c4_" in variant:
        return 4.0
    if "_c16_" in variant:
        return 16.0
    raise ValueError(f"variant has no phase cutoff: {variant}")


def _distance_scale(variant: str) -> float:
    if "_4k_" in variant or "_t4k_" in variant:
        return 4096.0
    if "_8k_" in variant or "_t8k_" in variant:
        return 8192.0
    if "_16k_" in variant or "_t16k_" in variant:
        return 16384.0
    raise ValueError(f"variant has no distance scale: {variant}")


def _onset(variant: str) -> float:
    if "_w4k_" in variant:
        return 4096.0
    if "_w8k_" in variant:
        return 8192.0
    return 0.0


def _expand_pair_values(values: torch.Tensor, head_dim: int) -> torch.Tensor:
    if values.shape[-1] * 2 != head_dim:
        raise ValueError(
            f"pair count {values.shape[-1]} is incompatible with head_dim={head_dim}"
        )
    return torch.cat((values, values), dim=-1)


def relative_rotary_scores(
    query_pre: torch.Tensor,
    key_pre: torch.Tensor,
    relative_distance: torch.Tensor,
    inv_freq: torch.Tensor,
    rotate_half: Callable[[torch.Tensor], torch.Tensor],
    attention_scale: float,
    score_scale: float,
    phase_map: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute q^T R(-phase_map(delta*omega)) k for every cached key."""

    if query_pre.shape[-2] != 1:
        raise ValueError("relative_rotary_scores currently expects one query")
    head_dim = int(query_pre.shape[-1])
    phase = relative_distance.float().unsqueeze(-1) * inv_freq.float().unsqueeze(0)
    effective = phase_map(phase)
    cos = _expand_pair_values(torch.cos(effective), head_dim).to(key_pre.dtype)
    # The cached key precedes the query, so p - t = -delta.
    sin = _expand_pair_values(-torch.sin(effective), head_dim).to(key_pre.dtype)
    cos = cos.view(1, 1, key_pre.shape[-2], head_dim)
    sin = sin.view(1, 1, key_pre.shape[-2], head_dim)
    rotated_key = key_pre * cos + rotate_half(key_pre) * sin
    scores = torch.matmul(query_pre, rotated_key.transpose(2, 3)) * score_scale
    return scores * float(attention_scale) ** 2


def phase_coherent_scores(
    query_pre: torch.Tensor,
    key_pre: torch.Tensor,
    relative_distance: torch.Tensor,
    inv_freq: torch.Tensor,
    rotate_half: Callable[[torch.Tensor], torch.Tensor],
    attention_scale: float,
    score_scale: float,
    cutoff: float,
    onset: float = 0.0,
    normalize_multiplier: bool = False,
) -> torch.Tensor:
    """Frequency-wise transition from exact RoPE locally to NoPE remotely.

    For x=delta*omega, kappa=exp(-(abs(x)/cutoff)^2) and

      K_i(delta) = kappa R_i(-x) + (1-kappa) I.

    High-frequency pairs become position-free first; low-frequency pairs retain
    positional structure over longer distances.  At delta=0 the kernel is
    exactly standard RoPE, while every pair converges to the pre-RoPE dot
    product as delta grows.
    """

    if query_pre.shape[-2] != 1:
        raise ValueError("phase_coherent_scores currently expects one query")
    head_dim = int(query_pre.shape[-1])
    distance = relative_distance.float().unsqueeze(-1)
    phase = distance * inv_freq.float().unsqueeze(0)
    transition_phase = (
        distance - float(onset)
    ).clamp_min(0.0) * inv_freq.float().unsqueeze(0)
    kappa_pair = torch.exp(
        -torch.square(torch.abs(transition_phase) / float(cutoff))
    )
    cos_pair = kappa_pair * torch.cos(phase) + (1.0 - kappa_pair)
    sin_pair = -kappa_pair * torch.sin(phase)
    if normalize_multiplier:
        magnitude = torch.sqrt(cos_pair.square() + sin_pair.square()).clamp_min(1e-4)
        cos_pair = cos_pair / magnitude
        sin_pair = sin_pair / magnitude
    cos = _expand_pair_values(cos_pair, head_dim).to(key_pre.dtype)
    sin = _expand_pair_values(sin_pair, head_dim).to(key_pre.dtype)
    cos = cos.view(1, 1, key_pre.shape[-2], head_dim)
    sin = sin.view(1, 1, key_pre.shape[-2], head_dim)
    mixed_key = key_pre * cos + rotate_half(key_pre) * sin
    scores = torch.matmul(query_pre, mixed_key.transpose(2, 3)) * score_scale
    return scores * float(attention_scale) ** 2


def _rescue_boundary(variant: str) -> float:
    return 4096.0 if "_w4k_" in variant else 128.0


def _rescue_fraction(variant: str) -> float:
    if "_lift100" in variant:
        return 1.0
    return 0.50 if "_lift50" in variant else 0.25


def _rescue_gap_threshold(variant: str) -> float:
    if "_gap0p5" in variant:
        return 0.5
    if "_gap1" in variant:
        return 1.0
    if "_gap2" in variant:
        return 2.0
    return 0.0


def _rescue_frequency_budget(variant: str, pair_count: int) -> int:
    fields = set(variant.split("_"))
    if "f8" in fields:
        return min(8, pair_count)
    if "f16" in fields:
        return min(16, pair_count)
    return pair_count


def _strict_phase_cap(variant: str) -> float:
    fields = set(variant.split("_"))
    if "cap0p1" in fields:
        return 0.10
    if "cap0p5" in fields:
        return 0.50
    if "cap0p25" in fields:
        return 0.25
    raise ValueError(f"strict MPR variant has no explicit phase cap: {variant}")


def _strict_random_support(variant: str) -> bool:
    return "random" in set(variant.split("_"))


def _strict_token_cap(variant: str) -> int:
    """Maximum treated tokens per layer/query-head; zero means legacy uncapped."""

    fields = set(variant.split("_"))
    if "t1" in fields:
        return 1
    if "t4" in fields:
        return 4
    if variant in LEGACY_STRICT_MPR_VARIANTS:
        return 0
    raise ValueError(f"strict MPR variant has no explicit token cap: {variant}")


def _strict_is_reference_arm(variant: str) -> bool:
    """Plain treatment arm whose realized strength seeds its random controls."""

    fields = set(variant.split("_"))
    return variant in STRICT_MPR_VARIANTS and not {
        "random",
        "masspreserve",
    }.intersection(fields)


def _uses_exact_pre_selector(variant: str) -> bool:
    """Whether a variant belongs to the frozen exact-pre support family."""

    return variant == "exact_pre_top2_postscore" or variant in STRICT_MPR_VARIANTS


def _strict_frequency_budget_marker(variant: str) -> int:
    """Variant-level plane budget before clamping to the model head width."""

    fields = set(variant.split("_"))
    if "f8" in fields:
        return 8
    if "f16" in fields:
        return 16
    return 0


def _strict_reference_signature(
    variant: str,
) -> tuple[float, float, float, int, int, float]:
    """Parameters whose baseline-derived trigger/target plan must match."""

    return (
        _rescue_boundary(variant),
        _rescue_fraction(variant),
        _rescue_gap_threshold(variant),
        _strict_token_cap(variant),
        _strict_frequency_budget_marker(variant),
        _strict_phase_cap(variant),
    )


def _normalize_strict_signature(
    signature: tuple[float, float, float, int, int, float],
) -> tuple[float, float, float, int, int, float]:
    if len(signature) != 6:
        raise ValueError(
            "strict signature must contain boundary/lift/gap/token-cap/"
            "frequency-budget/phase-cap"
        )
    normalized = (
        float(signature[0]),
        float(signature[1]),
        float(signature[2]),
        int(signature[3]),
        int(signature[4]),
        float(signature[5]),
    )
    if normalized[0] < 0.0 or not 0.0 <= normalized[1] <= 1.0:
        raise ValueError(f"invalid strict signature: {normalized}")
    if normalized[3] < 0 or normalized[4] < 0 or normalized[5] <= 0.0:
        raise ValueError(f"invalid strict signature: {normalized}")
    return normalized


def cap_strict_token_triggers(
    raw_trigger: torch.Tensor,
    counterfactual_gap: torch.Tensor,
    selected: torch.Tensor,
    token_cap: int,
) -> torch.Tensor:
    """Keep a deterministic per-head top-gap subset of the raw trigger mask.

    Ties are broken by the original token position (smaller position first),
    then by frozen support slot.  The two stable sorts make this independent of
    GPU top-k tie behavior after the exact baseline support has been frozen.
    ``token_cap=0`` is retained only for backwards-compatible uncapped arms.
    """

    if raw_trigger.ndim != 2:
        raise ValueError(f"raw_trigger must be 2-D, got {raw_trigger.shape}")
    if counterfactual_gap.shape != raw_trigger.shape:
        raise ValueError("counterfactual_gap must match raw_trigger")
    if selected.shape != raw_trigger.shape:
        raise ValueError("selected must match raw_trigger")
    if token_cap < 0:
        raise ValueError("token_cap must be non-negative")
    if token_cap == 0 or token_cap >= raw_trigger.shape[-1]:
        return raw_trigger.clone()

    # First establish the deterministic secondary order.  Stable descending
    # gap sort then preserves that position order for exactly tied gaps.
    position_order = torch.argsort(selected, dim=-1, stable=True)
    ordered_gap = counterfactual_gap.gather(-1, position_order)
    ordered_raw = raw_trigger.gather(-1, position_order)
    ordered_gap = ordered_gap.masked_fill(~ordered_raw, -torch.inf)
    gap_order = torch.argsort(
        ordered_gap,
        dim=-1,
        descending=True,
        stable=True,
    )
    ranked_slots = position_order.gather(-1, gap_order)
    capped = torch.zeros_like(raw_trigger)
    capped.scatter_(-1, ranked_slots[:, :token_cap], True)
    return capped & raw_trigger


def _gather_vectors(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    index = positions.view(1, positions.shape[0], -1, 1).expand(
        1,
        positions.shape[0],
        positions.shape[1],
        values.shape[-1],
    )
    return values.gather(2, index)


def clipped_relative_scores(
    query_pre: torch.Tensor,
    key_pre: torch.Tensor,
    relative_distance: torch.Tensor,
    inv_freq: torch.Tensor,
    rotate_half: Callable[[torch.Tensor], torch.Tensor],
    attention_scale: float,
    score_scale: float,
    boundary: float,
) -> torch.Tensor:
    distance = relative_distance.float()
    effective = distance.clamp_max(float(boundary))
    ratio = effective / distance.clamp_min(1.0)
    return relative_rotary_scores(
        query_pre,
        key_pre,
        relative_distance,
        inv_freq,
        rotate_half,
        attention_scale,
        score_scale,
        lambda phase: phase * ratio.unsqueeze(-1),
    )


def minimal_phase_rescue_scores(
    query_pre: torch.Tensor,
    selected_key_pre: torch.Tensor,
    selected_delta: torch.Tensor,
    post_selected: torch.Tensor,
    remote_mask: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scale: float,
    score_scale: float,
    boundary: float,
    lift_fraction: float,
    minimum_counterfactual_gap: float = 0.0,
    frequency_budget: int | None = None,
    max_phase_shift: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the minimum first-order phase change needed for a partial rescue.

    The counterfactual asks how each selected remote pair would score if its
    relative distance were clipped to the local boundary.  Only pairs harmed by
    the actual remote phase are changed.  Among all linearized phase changes
    that recover ``lift_fraction`` of that suppression, the L2-minimum solution
    is parallel to the per-frequency score gradient.  We then evaluate the
    nonlinear trigonometric score exactly and add only its delta to the native
    post-RoPE logit, which guarantees an exact no-op for untriggered pairs.
    """

    half = int(query_pre.shape[-1] // 2)
    qx = query_pre[0, :, :, :half].expand(-1, selected_key_pre.shape[2], -1)
    qy = query_pre[0, :, :, half:].expand(-1, selected_key_pre.shape[2], -1)
    kx = selected_key_pre[0, ..., :half]
    ky = selected_key_pre[0, ..., half:]
    a = qx.float() * kx.float() + qy.float() * ky.float()
    b = qx.float() * ky.float() - qy.float() * kx.float()
    phase = selected_delta.float().unsqueeze(-1) * inv_freq.float().view(1, 1, -1)
    base_pair = a * torch.cos(phase) + b * torch.sin(phase)
    clipped_phase = selected_delta.float().clamp_max(float(boundary)).unsqueeze(
        -1
    ) * inv_freq.float().view(1, 1, -1)
    counterfactual_pair = a * torch.cos(clipped_phase) + b * torch.sin(
        clipped_phase
    )
    total_scale = float(score_scale) * float(attention_scale) ** 2
    counterfactual_gap = (
        counterfactual_pair - base_pair
    ).sum(dim=-1) * total_scale
    desired_lift = float(lift_fraction) * torch.relu(counterfactual_gap)
    trigger = remote_mask & (
        counterfactual_gap > float(minimum_counterfactual_gap)
    )

    # d/d(delta_i) [A cos(x-delta_i) + B sin(x-delta_i)] at delta_i=0.
    gradient = (a * torch.sin(phase) - b * torch.cos(phase)) * total_scale
    pair_count = int(gradient.shape[-1])
    active_budget = pair_count if frequency_budget is None else max(
        1, min(int(frequency_budget), pair_count)
    )
    if active_budget < pair_count:
        suppressing_plane = (counterfactual_pair - base_pair) > 0.0
        eligible_gradient = gradient.masked_fill(~suppressing_plane, 0.0)
        active = torch.topk(
            eligible_gradient.abs(),
            k=active_budget,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        active_mask = torch.zeros_like(gradient, dtype=torch.bool)
        active_mask.scatter_(-1, active, True)
        gradient = gradient * active_mask
    denominator = gradient.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    phase_shift = desired_lift.unsqueeze(-1) * gradient / denominator
    phase_shift = phase_shift.clamp(-max_phase_shift, max_phase_shift)
    phase_shift = phase_shift * trigger.unsqueeze(-1)
    corrected_pair = a * torch.cos(phase - phase_shift) + b * torch.sin(
        phase - phase_shift
    )
    exact_lift = (corrected_pair - base_pair).sum(dim=-1) * total_scale
    corrected = torch.where(
        trigger,
        post_selected.float() + exact_lift,
        post_selected.float(),
    )
    return corrected.to(post_selected.dtype), {
        "trigger": trigger,
        "remote_mask": remote_mask,
        "exact_lift": exact_lift,
        "desired_lift": desired_lift,
        "phase_shift_rms": torch.sqrt(phase_shift.square().mean(dim=-1)),
        "counterfactual_gap": counterfactual_gap,
        "active_plane_count": (phase_shift != 0.0).sum(dim=-1),
    }


def strict_phase_reference_quantities(
    query_pre: torch.Tensor,
    selected_key_pre: torch.Tensor,
    selected_delta: torch.Tensor,
    selected: torch.Tensor,
    rescue_eligible: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scale: float,
    score_scale: float,
    boundary: float,
    lift_fraction: float,
    minimum_counterfactual_gap: float,
    token_cap: int,
) -> dict[str, torch.Tensor]:
    """Compute raw and token-capped plans from the exact baseline only."""

    half = int(query_pre.shape[-1] // 2)
    qx = query_pre[0, :, :, :half].expand(-1, selected_key_pre.shape[2], -1)
    qy = query_pre[0, :, :, half:].expand(-1, selected_key_pre.shape[2], -1)
    kx = selected_key_pre[0, ..., :half]
    ky = selected_key_pre[0, ..., half:]
    a = qx.float() * kx.float() + qy.float() * ky.float()
    b = qx.float() * ky.float() - qy.float() * kx.float()
    phase = selected_delta.float().unsqueeze(-1) * inv_freq.float().view(1, 1, -1)
    base_pair = a * torch.cos(phase) + b * torch.sin(phase)
    clipped_phase = selected_delta.float().clamp_max(float(boundary)).unsqueeze(
        -1
    ) * inv_freq.float().view(1, 1, -1)
    counterfactual_pair = a * torch.cos(clipped_phase) + b * torch.sin(
        clipped_phase
    )
    total_scale = float(score_scale) * float(attention_scale) ** 2
    counterfactual_gap = (
        counterfactual_pair - base_pair
    ).sum(dim=-1) * total_scale
    desired_lift = float(lift_fraction) * torch.relu(counterfactual_gap)
    raw_trigger = rescue_eligible & (
        counterfactual_gap > float(minimum_counterfactual_gap)
    )
    trigger = cap_strict_token_triggers(
        raw_trigger,
        counterfactual_gap,
        selected,
        token_cap,
    )
    return {
        "raw_trigger": raw_trigger,
        "trigger": trigger,
        "desired_lift": desired_lift,
        "counterfactual_gap": counterfactual_gap,
    }


def make_frozen_strict_reference_plan(
    *,
    epoch: int,
    layer_idx: int,
    key_count: int,
    signature: tuple[float, float, float, int, int, float],
    selected: torch.Tensor,
    selected_remote: torch.Tensor,
    selected_delta: torch.Tensor,
    rescue_eligible: torch.Tensor,
    quantities: dict[str, torch.Tensor],
) -> FrozenStrictReferencePlan:
    """Detach immutable-by-convention CPU snapshots from the exact baseline."""

    def snapshot(value: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        result = value.detach().to(device="cpu", dtype=dtype).clone().contiguous()
        result.requires_grad_(False)
        return result

    normalized_signature = _normalize_strict_signature(signature)
    token_cap = normalized_signature[3]
    return FrozenStrictReferencePlan(
        epoch=int(epoch),
        layer_idx=int(layer_idx),
        key_count=int(key_count),
        keep_count=int(selected.shape[-1]),
        head_count=int(selected.shape[0]),
        token_cap=token_cap,
        signature=normalized_signature,
        selected=snapshot(selected, torch.long),
        selected_remote=snapshot(selected_remote, torch.bool),
        selected_delta=snapshot(selected_delta, torch.long),
        rescue_eligible=snapshot(rescue_eligible, torch.bool),
        raw_trigger=snapshot(quantities["raw_trigger"], torch.bool),
        trigger=snapshot(quantities["trigger"], torch.bool),
        desired_lift=snapshot(quantities["desired_lift"], torch.float32),
        counterfactual_gap=snapshot(
            quantities["counterfactual_gap"], torch.float32
        ),
    )


def replay_frozen_strict_reference_plan(
    plan: FrozenStrictReferencePlan,
    *,
    epoch: int,
    layer_idx: int,
    key_count: int,
    keep_count: int,
    head_count: int,
    signature: tuple[float, float, float, int, int, float],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Validate a non-stale plan and copy it without invoking a selector."""

    normalized_signature = _normalize_strict_signature(signature)
    expected = {
        "epoch": (plan.epoch, int(epoch)),
        "layer_idx": (plan.layer_idx, int(layer_idx)),
        "key_count": (plan.key_count, int(key_count)),
        "keep_count": (plan.keep_count, int(keep_count)),
        "head_count": (plan.head_count, int(head_count)),
        "token_cap": (plan.token_cap, normalized_signature[3]),
        "signature": (plan.signature, normalized_signature),
    }
    mismatches = [
        f"{name}: frozen={frozen!r}, current={current!r}"
        for name, (frozen, current) in expected.items()
        if frozen != current
    ]
    if mismatches:
        raise RuntimeError("stale/incompatible strict reference: " + "; ".join(mismatches))

    shape = (head_count, keep_count)
    tensors = {
        "selected": plan.selected,
        "selected_remote": plan.selected_remote,
        "selected_delta": plan.selected_delta,
        "rescue_eligible": plan.rescue_eligible,
        "raw_trigger": plan.raw_trigger,
        "trigger": plan.trigger,
        "desired_lift": plan.desired_lift,
        "counterfactual_gap": plan.counterfactual_gap,
    }
    for name, value in tensors.items():
        if tuple(value.shape) != shape:
            raise RuntimeError(
                f"frozen {name} has shape {tuple(value.shape)}, expected {shape}"
            )
    if plan.selected.numel():
        if int(plan.selected.min()) < 0 or int(plan.selected.max()) >= key_count:
            raise RuntimeError("frozen support contains an out-of-range position")
        for row in plan.selected:
            if int(torch.unique(row).numel()) != keep_count:
                raise RuntimeError("frozen support contains duplicate positions")
    expected_delta = (key_count - 1 - plan.selected).clamp_min(0)
    if not torch.equal(plan.selected_delta, expected_delta):
        raise RuntimeError("frozen support has inconsistent relative distances")
    if bool((plan.rescue_eligible & ~plan.selected_remote).any()):
        raise RuntimeError("frozen rescue eligibility is not a remote-mask subset")
    expected_eligible = plan.selected_remote & (
        plan.selected_delta > float(plan.signature[0])
    )
    if not torch.equal(plan.rescue_eligible, expected_eligible):
        raise RuntimeError("frozen rescue eligibility is inconsistent with boundary")
    expected_raw_trigger = expected_eligible & (
        plan.counterfactual_gap > float(plan.signature[2])
    )
    if not torch.equal(plan.raw_trigger, expected_raw_trigger):
        raise RuntimeError("frozen raw trigger is inconsistent with gap threshold")
    expected_trigger = cap_strict_token_triggers(
        expected_raw_trigger,
        plan.counterfactual_gap,
        plan.selected,
        plan.token_cap,
    )
    if not torch.equal(plan.trigger, expected_trigger):
        raise RuntimeError("frozen trigger is inconsistent with token cap")
    if plan.token_cap > 0 and bool(
        (plan.trigger.sum(dim=-1) > plan.token_cap).any()
    ):
        raise RuntimeError("frozen trigger exceeds token cap")
    expected_target = float(plan.signature[1]) * torch.relu(
        plan.counterfactual_gap
    )
    if not torch.allclose(plan.desired_lift, expected_target, atol=1e-6, rtol=1e-6):
        raise RuntimeError("frozen target lift is inconsistent with reference gap")

    return {
        name: value.to(device=device).clone()
        for name, value in tensors.items()
    }


def make_frozen_strict_intervention_reference(
    *,
    epoch: int,
    layer_idx: int,
    key_count: int,
    signature: tuple[float, float, float, int, int, float],
    stats: dict[str, torch.Tensor],
) -> FrozenStrictInterventionReference:
    """Snapshot the actual normal-arm L2 used for cross-arm random matching."""

    def snapshot(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return value.detach().to(device="cpu", dtype=dtype).clone().contiguous()

    normalized_signature = _normalize_strict_signature(signature)
    shape = tuple(int(value) for value in stats["shift_norm"].shape)
    if len(shape) != 2:
        raise ValueError(f"strict intervention tensors must be 2-D, got {shape}")
    return FrozenStrictInterventionReference(
        epoch=int(epoch),
        layer_idx=int(layer_idx),
        key_count=int(key_count),
        keep_count=shape[1],
        head_count=shape[0],
        token_cap=normalized_signature[3],
        signature=normalized_signature,
        shift_norm=snapshot(stats["shift_norm"], torch.float32),
        support_count=snapshot(stats["active_plane_count"], torch.long),
        achieved_lift=snapshot(stats["solver_lift"], torch.float32),
    )


def replay_frozen_strict_intervention_reference(
    plan: FrozenStrictInterventionReference,
    *,
    epoch: int,
    layer_idx: int,
    key_count: int,
    keep_count: int,
    head_count: int,
    signature: tuple[float, float, float, int, int, float],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load the preceding non-random arm's strength or fail closed."""

    normalized_signature = _normalize_strict_signature(signature)
    expected = {
        "epoch": (plan.epoch, int(epoch)),
        "layer_idx": (plan.layer_idx, int(layer_idx)),
        "key_count": (plan.key_count, int(key_count)),
        "keep_count": (plan.keep_count, int(keep_count)),
        "head_count": (plan.head_count, int(head_count)),
        "token_cap": (plan.token_cap, normalized_signature[3]),
        "signature": (plan.signature, normalized_signature),
    }
    mismatches = [
        f"{name}: frozen={frozen!r}, current={current!r}"
        for name, (frozen, current) in expected.items()
        if frozen != current
    ]
    if mismatches:
        raise RuntimeError(
            "stale/incompatible strict intervention reference: "
            + "; ".join(mismatches)
        )
    shape = (int(head_count), int(keep_count))
    for name, value in (
        ("shift_norm", plan.shift_norm),
        ("support_count", plan.support_count),
        ("achieved_lift", plan.achieved_lift),
    ):
        if tuple(value.shape) != shape:
            raise RuntimeError(
                f"frozen intervention {name} has shape {tuple(value.shape)}, "
                f"expected {shape}"
            )
    return {
        "shift_norm": plan.shift_norm.to(device=device).clone(),
        "support_count": plan.support_count.to(device=device).clone(),
        "achieved_lift": plan.achieved_lift.to(device=device).clone(),
    }


def select_or_replay_strict_support(
    *,
    variant: str,
    selection_scores: torch.Tensor,
    keep_count: int,
    local_window: int,
    sink_tokens: int,
    plan: FrozenStrictReferencePlan | None,
    epoch: int,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    """Select once for the baseline; replay without consulting drifted scores."""

    if variant not in STRICT_MPR_VARIANTS:
        selected, selected_remote = runner.local_global_selection(
            selection_scores,
            keep_count,
            local_window,
            sink_tokens,
        )
        return selected, selected_remote, None
    if plan is None:
        raise RuntimeError(
            "strict-MPR requires exact_pre_top2_postscore to run first "
            f"for layer {layer_idx} in the current prefill epoch"
        )
    replay = replay_frozen_strict_reference_plan(
        plan,
        epoch=epoch,
        layer_idx=layer_idx,
        key_count=int(selection_scores.shape[-1]),
        keep_count=keep_count,
        head_count=int(selection_scores.shape[0]),
        signature=_strict_reference_signature(variant),
        device=selection_scores.device,
    )
    return replay["selected"], replay["selected_remote"], replay


def strict_phase_rescue_scores(
    query_pre: torch.Tensor,
    selected_key_pre: torch.Tensor,
    selected_delta: torch.Tensor,
    post_selected: torch.Tensor,
    remote_mask: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scale: float,
    score_scale: float,
    boundary: float,
    lift_fraction: float,
    minimum_counterfactual_gap: float,
    frequency_budget: int,
    max_phase_shift: float,
    *,
    random_frequency_support: bool = False,
    random_seed_base: int = 20260801,
    frozen_raw_trigger: torch.Tensor | None = None,
    frozen_trigger: torch.Tensor | None = None,
    frozen_desired_lift: torch.Tensor | None = None,
    frozen_counterfactual_gap: torch.Tensor | None = None,
    matched_random_norm: torch.Tensor | None = None,
    matched_random_support: torch.Tensor | None = None,
    matched_random_lift: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply a strict-constrained nonlinear phase rescue to selected tokens.

    The large 64-plane sparse-support problem uses the solver's guarded
    lift/risk heuristic; only the continuous nonlinear solve on the chosen
    support is exact.  Token candidates are deliberately *not* selected here.
    The caller replays the exact baseline's frozen token, trigger, and target
    plan.  The t1/t4 variants are sparse twice: the baseline admits at most one
    or four tokens per query head, and the solver changes at most the declared
    number of frequency planes inside each admitted token.

    The random ablation uses the same frozen pair plan and replays the actual
    per-pair phase L2 captured from the preceding non-random strict arm.  It
    distributes that norm over the same actual number of non-zero random
    planes.  Thus support count and L2 are both matched strictly across arms;
    achieved lift is deliberately not matched and is recorded as an outcome.

    This diagnostic path runs the constrained solve on CPU in float64 and
    copies the exactly re-evaluated score lift back to the attention device.
    Non-triggered pairs are copied directly from ``post_selected`` and are
    therefore bitwise no-ops.
    """

    if max_phase_shift <= 0.0:
        raise ValueError("max_phase_shift must be positive")
    half = int(query_pre.shape[-1] // 2)
    active_budget = max(1, min(int(frequency_budget), half))
    qx = query_pre[0, :, :, :half].expand(-1, selected_key_pre.shape[2], -1)
    qy = query_pre[0, :, :, half:].expand(-1, selected_key_pre.shape[2], -1)
    kx = selected_key_pre[0, ..., :half]
    ky = selected_key_pre[0, ..., half:]
    a = qx.float() * kx.float() + qy.float() * ky.float()
    b = qx.float() * ky.float() - qy.float() * kx.float()
    phase = selected_delta.float().unsqueeze(-1) * inv_freq.float().view(1, 1, -1)
    base_pair = a * torch.cos(phase) + b * torch.sin(phase)
    clipped_phase = selected_delta.float().clamp_max(float(boundary)).unsqueeze(
        -1
    ) * inv_freq.float().view(1, 1, -1)
    counterfactual_pair = a * torch.cos(clipped_phase) + b * torch.sin(
        clipped_phase
    )
    total_scale = float(score_scale) * float(attention_scale) ** 2
    current_counterfactual_gap = (
        counterfactual_pair - base_pair
    ).sum(dim=-1) * total_scale
    if any(
        item is not None
        for item in (
            frozen_raw_trigger,
            frozen_trigger,
            frozen_desired_lift,
            frozen_counterfactual_gap,
        )
    ):
        if any(
            item is None
            for item in (
                frozen_raw_trigger,
                frozen_trigger,
                frozen_desired_lift,
                frozen_counterfactual_gap,
            )
        ):
            raise ValueError("all frozen trigger-plan tensors must be supplied")
        assert frozen_raw_trigger is not None
        assert frozen_trigger is not None
        assert frozen_desired_lift is not None
        assert frozen_counterfactual_gap is not None
        for name, value in (
            ("frozen_raw_trigger", frozen_raw_trigger),
            ("frozen_trigger", frozen_trigger),
            ("frozen_desired_lift", frozen_desired_lift),
            ("frozen_counterfactual_gap", frozen_counterfactual_gap),
        ):
            if value.shape != post_selected.shape:
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} does not match "
                    f"selected scores {tuple(post_selected.shape)}"
                )
        raw_trigger = frozen_raw_trigger.to(
            device=post_selected.device, dtype=torch.bool
        )
        trigger = frozen_trigger.to(device=post_selected.device, dtype=torch.bool)
        if bool((trigger & ~raw_trigger).any()):
            raise ValueError("frozen capped trigger is not a raw-trigger subset")
        if bool((raw_trigger & ~remote_mask).any()):
            raise ValueError("frozen raw trigger is not a rescue-eligible subset")
        desired_lift = frozen_desired_lift.to(
            device=post_selected.device, dtype=torch.float32
        )
        counterfactual_gap = frozen_counterfactual_gap.to(
            device=post_selected.device, dtype=torch.float32
        )
    else:
        counterfactual_gap = current_counterfactual_gap
        desired_lift = float(lift_fraction) * torch.relu(counterfactual_gap)
        raw_trigger = remote_mask & (
            counterfactual_gap > float(minimum_counterfactual_gap)
        )
        trigger = raw_trigger

    matched_inputs = (matched_random_norm, matched_random_support, matched_random_lift)
    if random_frequency_support:
        if any(value is None for value in matched_inputs):
            raise ValueError(
                "random strict control requires the frozen non-random "
                "norm/support/lift reference"
            )
        for name, value in (
            ("matched_random_norm", matched_random_norm),
            ("matched_random_support", matched_random_support),
            ("matched_random_lift", matched_random_lift),
        ):
            assert value is not None
            if value.shape != post_selected.shape:
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} does not match "
                    f"selected scores {tuple(post_selected.shape)}"
                )
    elif any(value is not None for value in matched_inputs):
        raise ValueError("matched random references were supplied to a normal arm")

    corrected = post_selected.clone()
    exact_lift = torch.zeros_like(post_selected, dtype=torch.float32)
    solver_lift = torch.zeros_like(post_selected, dtype=torch.float32)
    phase_shift_rms = torch.zeros_like(post_selected, dtype=torch.float32)
    shift_norm = torch.zeros_like(post_selected, dtype=torch.float32)
    shift_abs_max = torch.zeros_like(post_selected, dtype=torch.float32)
    active_plane_count = torch.zeros_like(post_selected, dtype=torch.long)
    solver_feasible = torch.zeros_like(trigger, dtype=torch.bool)
    matched_reference_norm = torch.zeros_like(post_selected, dtype=torch.float32)
    norm_match_error = torch.zeros_like(post_selected, dtype=torch.float32)
    matched_reference_support = torch.zeros_like(post_selected, dtype=torch.long)
    support_match_error = torch.zeros_like(post_selected, dtype=torch.long)
    matched_reference_lift = torch.zeros_like(post_selected, dtype=torch.float32)

    trigger_indices = trigger.nonzero(as_tuple=False)
    if int(trigger_indices.shape[0]) > 0:
        # The scale is folded into A/B so target_lift and achieved_lift are in
        # the same QK-logit units that are added to post_selected.
        a_cpu = (a * total_scale).detach().cpu().numpy().astype(np.float64)
        b_cpu = (b * total_scale).detach().cpu().numpy().astype(np.float64)
        phase_cpu = phase.detach().cpu().numpy().astype(np.float64)
        target_cpu = desired_lift.detach().cpu().numpy().astype(np.float64)
        delta_cpu = selected_delta.detach().cpu().numpy()
        matched_norm_cpu = (
            None
            if matched_random_norm is None
            else matched_random_norm.detach().cpu().numpy().astype(np.float64)
        )
        matched_support_cpu = (
            None
            if matched_random_support is None
            else matched_random_support.detach().cpu().numpy().astype(np.int64)
        )
        matched_lift_cpu = (
            None
            if matched_random_lift is None
            else matched_random_lift.detach().cpu().numpy().astype(np.float64)
        )
        for head_tensor, slot_tensor in trigger_indices:
            head = int(head_tensor.item())
            slot = int(slot_tensor.item())
            if random_frequency_support:
                assert matched_norm_cpu is not None
                assert matched_support_cpu is not None
                assert matched_lift_cpu is not None
                reference_norm = float(matched_norm_cpu[head, slot])
                reference_support_count = int(matched_support_cpu[head, slot])
                reference_lift = float(matched_lift_cpu[head, slot])
                matched_reference_norm[head, slot] = reference_norm
                matched_reference_support[head, slot] = reference_support_count
                matched_reference_lift[head, slot] = reference_lift

                # Stable across devices and process scheduling.  Random planes
                # receive equal magnitudes whose count and total L2 exactly
                # replay the normal arm.  Achieved lift remains an outcome.
                seed = (
                    int(random_seed_base)
                    ^ ((head + 1) * 0x9E3779B1)
                    ^ ((slot + 1) * 0x85EBCA77)
                    ^ (int(delta_cpu[head, slot]) * 0xC2B2AE3D)
                ) & 0xFFFFFFFF
                generator = np.random.default_rng(seed)
                random_shifts = np.zeros(half, dtype=np.float64)
                if reference_norm > 1e-9:
                    if not 0 < reference_support_count <= active_budget:
                        raise RuntimeError(
                            "invalid frozen non-random support count: "
                            f"{reference_support_count}"
                        )
                    random_support = generator.choice(
                        half, size=reference_support_count, replace=False
                    )
                    magnitude = reference_norm / math.sqrt(reference_support_count)
                    if magnitude > float(max_phase_shift) + 1e-6:
                        raise RuntimeError(
                            "cross-arm L2 cannot be represented under the "
                            f"phase cap: magnitude={magnitude}, "
                            f"cap={max_phase_shift}"
                        )
                    preferred_phase = np.arctan2(
                        b_cpu[head, slot], a_cpu[head, slot]
                    )
                    wrapped_error = (
                        phase_cpu[head, slot] - preferred_phase + math.pi
                    ) % (2.0 * math.pi) - math.pi
                    beneficial_direction = np.sign(wrapped_error[random_support])
                    zero_direction = beneficial_direction == 0.0
                    if np.any(zero_direction):
                        beneficial_direction[zero_direction] = generator.choice(
                            np.asarray([-1.0, 1.0], dtype=np.float64),
                            size=int(np.sum(zero_direction)),
                            replace=True,
                        )
                    random_shifts[random_support] = (
                        beneficial_direction * magnitude
                    )
                lift_value = phase_lift(
                    a_cpu[head, slot],
                    b_cpu[head, slot],
                    phase_cpu[head, slot],
                    random_shifts,
                )
                result_norm = float(np.linalg.norm(random_shifts))
                result_support = tuple(
                    np.flatnonzero(random_shifts != 0.0).tolist()
                )
                result_shift_abs_max = float(
                    np.max(np.abs(random_shifts), initial=0.0)
                )
                result_feasible = bool(
                    lift_value + 1e-9 >= float(target_cpu[head, slot])
                )
            else:
                # This is heuristic only in its sparse support choice for the
                # 64-plane problem; the fixed-support trigonometric solve and
                # lift re-evaluation are exact to solver tolerance.
                reference_result = solve_phase_rescue(
                    a_cpu[head, slot],
                    b_cpu[head, slot],
                    phase_cpu[head, slot],
                    target_lift=float(target_cpu[head, slot]),
                    budget=active_budget,
                    phase_cap=float(max_phase_shift),
                    max_support_combinations=4096,
                )
                lift_value = float(reference_result.achieved_lift)
                result_norm = float(reference_result.norm)
                result_support = reference_result.support
                result_shift_abs_max = float(
                    np.max(np.abs(reference_result.shifts), initial=0.0)
                )
                result_feasible = bool(reference_result.feasible)
                matched_reference_norm[head, slot] = result_norm
                matched_reference_support[head, slot] = len(result_support)
                matched_reference_lift[head, slot] = lift_value

            exact_lift[head, slot] = lift_value
            solver_lift[head, slot] = lift_value
            corrected[head, slot] = (
                post_selected[head, slot].float() + lift_value
            ).to(post_selected.dtype)
            shift_norm[head, slot] = result_norm
            phase_shift_rms[head, slot] = result_norm / math.sqrt(half)
            shift_abs_max[head, slot] = result_shift_abs_max
            active_plane_count[head, slot] = len(result_support)
            solver_feasible[head, slot] = result_feasible
            norm_match_error[head, slot] = abs(
                result_norm - float(matched_reference_norm[head, slot].item())
            )
            support_match_error[head, slot] = abs(
                len(result_support)
                - int(matched_reference_support[head, slot].item())
            )

    nontrigger = ~trigger
    if bool(nontrigger.any()):
        nontrigger_noop_max = (
            corrected.float() - post_selected.float()
        ).masked_select(nontrigger).abs().amax()
    else:
        nontrigger_noop_max = torch.zeros(
            (), dtype=torch.float32, device=post_selected.device
        )
    return corrected, {
        "raw_trigger": raw_trigger,
        "trigger": trigger,
        "remote_mask": remote_mask,
        "exact_lift": exact_lift,
        "solver_lift": solver_lift,
        "desired_lift": desired_lift,
        "phase_shift_rms": phase_shift_rms,
        "shift_norm": shift_norm,
        "shift_abs_max": shift_abs_max,
        "counterfactual_gap": counterfactual_gap,
        "active_plane_count": active_plane_count,
        "solver_feasible": solver_feasible,
        "nontrigger_noop_max": nontrigger_noop_max,
        "matched_reference_norm": matched_reference_norm,
        "norm_match_error": norm_match_error,
        "matched_reference_support": matched_reference_support,
        "support_match_error": support_match_error,
        "matched_reference_lift": matched_reference_lift,
        "current_counterfactual_gap": current_counterfactual_gap,
    }


def preserve_remote_partition(
    corrected: torch.Tensor,
    native: torch.Tensor,
    remote_mask: torch.Tensor,
) -> torch.Tensor:
    negative_infinity = torch.full_like(corrected.float(), -torch.inf)
    corrected_remote = torch.where(remote_mask, corrected.float(), negative_infinity)
    native_remote = torch.where(remote_mask, native.float(), negative_infinity)
    correction = torch.logsumexp(corrected_remote, dim=-1, keepdim=True) - torch.logsumexp(
        native_remote,
        dim=-1,
        keepdim=True,
    )
    correction = torch.where(
        torch.isfinite(correction), correction, torch.zeros_like(correction)
    )
    return torch.where(
        remote_mask,
        corrected.float() - correction,
        corrected.float(),
    ).to(corrected.dtype)


def sparse_log_partition_error(
    corrected: torch.Tensor,
    native: torch.Tensor,
) -> torch.Tensor:
    """Per-head absolute log-partition error on the logits actually consumed."""

    return (
        torch.logsumexp(corrected.float(), dim=-1)
        - torch.logsumexp(native.float(), dim=-1)
    ).abs()


def record_phase_rescue_metrics(
    controller: runner.Controller,
    selected: torch.Tensor,
    stats: dict[str, torch.Tensor],
    key_count: int,
) -> None:
    metrics = controller.metrics
    trigger = stats["trigger"]
    remote_mask = stats["remote_mask"]
    remote_count = int(remote_mask.sum().item())
    trigger_count = int(trigger.sum().item())
    metrics.phase_remote_count = int(
        getattr(metrics, "phase_remote_count", 0)
    ) + remote_count
    metrics.phase_trigger_count = int(
        getattr(metrics, "phase_trigger_count", 0)
    ) + trigger_count
    metrics.phase_score_lift_sum = float(
        getattr(metrics, "phase_score_lift_sum", 0.0)
    ) + float(stats["exact_lift"].masked_select(trigger).sum().item())
    metrics.phase_shift_rms_sum = float(
        getattr(metrics, "phase_shift_rms_sum", 0.0)
    ) + float(stats["phase_shift_rms"].masked_select(trigger).sum().item())
    metrics.phase_active_plane_sum = float(
        getattr(metrics, "phase_active_plane_sum", 0.0)
    ) + float(stats["active_plane_count"].masked_select(trigger).sum().item())
    realized = stats["exact_lift"].masked_select(trigger)
    desired = stats["desired_lift"].masked_select(trigger).clamp_min(1e-8)
    metrics.phase_realized_lift_ratio_sum = float(
        getattr(metrics, "phase_realized_lift_ratio_sum", 0.0)
    ) + float((realized / desired).sum().item())
    metrics.phase_negative_lift_count = int(
        getattr(metrics, "phase_negative_lift_count", 0)
    ) + int((realized < 0.0).sum().item())

    gold = controller.evidence_mask(key_count, selected.device)[selected] & remote_mask
    gold_trigger = gold & trigger
    metrics.phase_gold_count = int(getattr(metrics, "phase_gold_count", 0)) + int(
        gold.sum().item()
    )
    metrics.phase_gold_trigger_count = int(
        getattr(metrics, "phase_gold_trigger_count", 0)
    ) + int(gold_trigger.sum().item())
    metrics.phase_gold_score_lift_sum = float(
        getattr(metrics, "phase_gold_score_lift_sum", 0.0)
    ) + float(stats["exact_lift"].masked_select(gold_trigger).sum().item())


def record_strict_phase_metrics(
    controller: runner.Controller,
    stats: dict[str, torch.Tensor],
    applied_lift: torch.Tensor,
    frequency_budget: int,
    phase_cap: float,
    token_cap: int,
    random_frequency_support: bool,
    partition_preserve: bool,
) -> None:
    """Record enough invariants to audit every hard strict-MPR constraint."""

    metrics = controller.metrics
    trigger = stats["trigger"]
    raw_trigger = stats["raw_trigger"]
    rescue_eligible = stats["remote_mask"]
    calls = int(trigger.sum().item())
    raw_calls = int(raw_trigger.sum().item())
    eligible_count = int(rescue_eligible.sum().item())
    if bool((trigger & ~raw_trigger).any()):
        raise RuntimeError("token-capped trigger is not a raw-trigger subset")
    per_head_trigger_count = trigger.sum(dim=-1)
    observed_token_max = int(per_head_trigger_count.max().item())
    if token_cap > 0 and observed_token_max > int(token_cap):
        raise RuntimeError(
            f"strict token support {observed_token_max} exceeds cap {token_cap}"
        )
    prior_token_cap = int(getattr(metrics, "strict_phase_token_cap", token_cap))
    if prior_token_cap != int(token_cap):
        raise RuntimeError(
            f"mixed strict token caps in one controller: {prior_token_cap} vs "
            f"{token_cap}"
        )
    metrics.strict_phase_token_cap = int(token_cap)
    metrics.strict_phase_remote_eligible_count = int(
        getattr(metrics, "strict_phase_remote_eligible_count", 0)
    ) + eligible_count
    metrics.strict_phase_raw_trigger_count = int(
        getattr(metrics, "strict_phase_raw_trigger_count", 0)
    ) + raw_calls
    metrics.strict_phase_capped_trigger_count = int(
        getattr(metrics, "strict_phase_capped_trigger_count", 0)
    ) + calls
    metrics.strict_phase_max_triggers_per_head = max(
        int(getattr(metrics, "strict_phase_max_triggers_per_head", 0)),
        observed_token_max,
    )
    feasible = stats["solver_feasible"] & trigger
    metrics.strict_phase_solver_calls = int(
        getattr(metrics, "strict_phase_solver_calls", 0)
    ) + calls
    metrics.strict_phase_trigger_count = int(
        getattr(metrics, "strict_phase_trigger_count", 0)
    ) + calls
    metrics.strict_phase_feasible_count = int(
        getattr(metrics, "strict_phase_feasible_count", 0)
    ) + int(feasible.sum().item())
    metrics.strict_phase_target_lift_sum = float(
        getattr(metrics, "strict_phase_target_lift_sum", 0.0)
    ) + float(stats["desired_lift"].masked_select(trigger).sum().item())
    metrics.strict_phase_solver_lift_sum = float(
        getattr(metrics, "strict_phase_solver_lift_sum", 0.0)
    ) + float(stats["solver_lift"].masked_select(trigger).sum().item())
    metrics.strict_phase_applied_lift_sum = float(
        getattr(metrics, "strict_phase_applied_lift_sum", 0.0)
    ) + float(applied_lift.masked_select(trigger).sum().item())
    support = stats["active_plane_count"].masked_select(trigger)
    metrics.strict_phase_support_sum = int(
        getattr(metrics, "strict_phase_support_sum", 0)
    ) + int(support.sum().item())
    if int(support.numel()) > 0:
        observed_support_max = int(support.max().item())
        if observed_support_max > int(frequency_budget):
            raise RuntimeError(
                f"strict phase support {observed_support_max} exceeds "
                f"budget {frequency_budget}"
            )
        metrics.strict_phase_support_max = max(
            int(getattr(metrics, "strict_phase_support_max", 0)),
            observed_support_max,
        )
    metrics.strict_phase_budget_sum = int(
        getattr(metrics, "strict_phase_budget_sum", 0)
    ) + calls * int(frequency_budget)
    metrics.strict_phase_cap_sum = float(
        getattr(metrics, "strict_phase_cap_sum", 0.0)
    ) + calls * float(phase_cap)
    shift_abs_max = float(stats["shift_abs_max"].max().item())
    if shift_abs_max > float(phase_cap) + 1e-6:
        raise RuntimeError(
            f"strict phase shift {shift_abs_max} exceeds cap {phase_cap}"
        )
    metrics.strict_phase_shift_abs_max = max(
        float(getattr(metrics, "strict_phase_shift_abs_max", 0.0)),
        shift_abs_max,
    )
    metrics.strict_phase_shift_l2_sum = float(
        getattr(metrics, "strict_phase_shift_l2_sum", 0.0)
    ) + float(stats["shift_norm"].masked_select(trigger).sum().item())
    metrics.strict_phase_random_reference_l2_sum = float(
        getattr(metrics, "strict_phase_random_reference_l2_sum", 0.0)
    ) + float(stats["matched_reference_norm"].masked_select(trigger).sum().item())
    metrics.strict_phase_random_reference_lift_sum = float(
        getattr(metrics, "strict_phase_random_reference_lift_sum", 0.0)
    ) + float(stats["matched_reference_lift"].masked_select(trigger).sum().item())
    metrics.strict_phase_random_norm_match_max = max(
        float(getattr(metrics, "strict_phase_random_norm_match_max", 0.0)),
        float(stats["norm_match_error"].masked_select(trigger).max().item())
        if calls
        else 0.0,
    )
    metrics.strict_phase_random_support_delta_max = max(
        int(getattr(metrics, "strict_phase_random_support_delta_max", 0)),
        int(stats["support_match_error"].masked_select(trigger).max().item())
        if calls
        else 0,
    )
    metrics.strict_phase_nontrigger_noop_max = max(
        float(getattr(metrics, "strict_phase_nontrigger_noop_max", 0.0)),
        float(stats["nontrigger_noop_max"].item()),
    )
    capped_out = raw_trigger & ~trigger
    capped_out_count = int(capped_out.sum().item())
    token_cap_noop_max = (
        float(applied_lift.masked_select(capped_out).abs().max().item())
        if capped_out_count
        else 0.0
    )
    metrics.strict_phase_token_cap_noop_count = int(
        getattr(metrics, "strict_phase_token_cap_noop_count", 0)
    ) + capped_out_count
    metrics.strict_phase_token_cap_noop_max = max(
        float(getattr(metrics, "strict_phase_token_cap_noop_max", 0.0)),
        token_cap_noop_max,
    )
    frozen_token_cap_mismatch = int(
        stats.get(
            "frozen_token_cap_mismatch_count",
            torch.zeros((), dtype=torch.long, device=trigger.device),
        ).item()
    )
    metrics.strict_phase_frozen_token_cap_mismatch_max = max(
        int(
            getattr(
                metrics,
                "strict_phase_frozen_token_cap_mismatch_max",
                0,
            )
        ),
        frozen_token_cap_mismatch,
    )
    metrics.strict_phase_random_support = int(random_frequency_support)
    metrics.strict_phase_partition_preserve = int(partition_preserve)
    if partition_preserve:
        active_partition_heads = trigger.any(dim=-1)
        partition_error = stats["partition_preservation_error"].float().masked_select(
            active_partition_heads
        )
        metrics.strict_phase_partition_error_sum = float(
            getattr(metrics, "strict_phase_partition_error_sum", 0.0)
        ) + float(partition_error.sum().item())
        metrics.strict_phase_partition_head_count = int(
            getattr(metrics, "strict_phase_partition_head_count", 0)
        ) + int(partition_error.numel())
        metrics.strict_phase_partition_error_max = max(
            float(getattr(metrics, "strict_phase_partition_error_max", 0.0)),
            float(partition_error.max().item()) if partition_error.numel() else 0.0,
        )
    # Every strict variant replays the baseline snapshot and never calls the
    # selector on its drifted current Query.
    metrics.strict_phase_exact_pre_selector = 1
    metrics.strict_phase_frozen_reference = 1
    metrics.strict_phase_frozen_support_mismatch_max = max(
        int(getattr(metrics, "strict_phase_frozen_support_mismatch_max", 0)),
        0,
    )


def _all_positions(heads: int, key_count: int, device: torch.device) -> torch.Tensor:
    return torch.arange(key_count, device=device).view(1, -1).expand(heads, -1)


def _make_key_capture_hook(attention: torch.nn.Module) -> Callable[..., None]:
    def hook(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if not _CAPTURE_PREFIX_KEYS:
            return
        # Qwen3 k_norm emits [batch, sequence, kv_heads, head_dim].
        attention._phase_pre_key_chunks.append(
            output.detach()
            .transpose(1, 2)
            .contiguous()
            .to(_PREFIX_KEY_STORAGE)
        )

    return hook


def phase_patch_model(model: Any) -> None:
    """Install the attention patch and capture exact pre-RoPE prefix keys."""

    _BASE_PATCH_MODEL(model)
    rotary = model.model.rotary_emb
    exact_attention_scale = float(getattr(rotary, "attention_scaling", 1.0))
    for attention in model.modules():
        if attention.__class__.__name__ != "Qwen3Attention":
            continue
        attention._phase_pre_key_chunks = []
        attention._phase_pre_key_cache = None
        attention._phase_attention_scale = exact_attention_scale
        attention._strict_mpr_reference_plans = {}
        attention._strict_mpr_intervention_references = {}
        attention._strict_mpr_reference_epoch = 0
        if not hasattr(attention, "_phase_key_capture_handle"):
            attention._phase_key_capture_handle = attention.k_norm.register_forward_hook(
                _make_key_capture_hook(attention)
            )


def capture_prefill_sequence(
    model: Any,
    prompt_prefix: torch.Tensor,
    chunk_size: int,
) -> tuple[Any, float]:
    """Run the normal prefill while retaining exact, unrotated K once per layer."""

    global _CAPTURE_PREFIX_KEYS, _STRICT_REFERENCE_EPOCH
    attentions = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "Qwen3Attention"
    ]
    _STRICT_REFERENCE_EPOCH += 1
    for attention in attentions:
        attention._phase_pre_key_chunks = []
        attention._phase_pre_key_cache = None
        attention._strict_mpr_reference_plans = {}
        attention._strict_mpr_intervention_references = {}
        attention._strict_mpr_reference_epoch = _STRICT_REFERENCE_EPOCH
    _CAPTURE_PREFIX_KEYS = True
    try:
        result = _BASE_PREFILL_SEQUENCE(model, prompt_prefix, chunk_size)
    finally:
        _CAPTURE_PREFIX_KEYS = False
    expected = int(prompt_prefix.shape[1])
    for attention in attentions:
        chunks = attention._phase_pre_key_chunks
        if not chunks:
            raise RuntimeError(
                f"layer {attention.layer_idx} captured no exact pre-RoPE keys"
            )
        cache = torch.cat(chunks, dim=2)
        if int(cache.shape[2]) != expected:
            raise RuntimeError(
                f"layer {attention.layer_idx} captured {cache.shape[2]} keys; "
                f"expected {expected}"
            )
        attention._phase_pre_key_cache = cache
        attention._phase_pre_key_chunks = []
    return result


def phase_kernel_attention_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_value: Any | None = None,
    cache_position: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    controller = runner._ACTIVE_CONTROLLER
    if controller is None or controller.variant not in NEW_VARIANTS:
        return _BASE_FORWARD(
            self,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
    if hidden_states.shape[-2] != 1:
        return self._local_global_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    modeling_qwen3 = self._local_global_modeling_qwen3
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_pre = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    current_key_pre = self.k_norm(
        self.k_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    current_value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_post, current_key_post = modeling_qwen3.apply_rotary_pos_emb(
        query_pre,
        current_key_pre,
        cos.to(query_pre.device),
        sin.to(query_pre.device),
    )
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_post, value = past_key_value.update(
            current_key_post,
            current_value,
            self.layer_idx,
            cache_kwargs,
        )
    else:
        key_post, value = current_key_post, current_value

    groups = query_post.shape[1] // key_post.shape[1]
    expanded_key_post = runner.repeat_kv(key_post, groups)
    expanded_value = runner.repeat_kv(value, groups)
    key_count = int(expanded_key_post.shape[-2])
    score_scale = float(
        getattr(self, "scaling", 1.0 / math.sqrt(query_post.shape[-1]))
    )
    post_scores = torch.matmul(
        query_post,
        expanded_key_post.transpose(2, 3),
    ) * score_scale
    post_scores = runner.add_attention_mask(post_scores, attention_mask)

    positions = torch.arange(key_count, device=key_post.device)
    attention_scale = float(
        getattr(
            self,
            "_phase_attention_scale",
            rope_repair._attention_scaling((cos, sin)),
        )
    )
    cached_key_pre = getattr(self, "_phase_pre_key_cache", None)
    if cached_key_pre is None or int(cached_key_pre.shape[2]) != key_count - 1:
        captured = -1 if cached_key_pre is None else int(cached_key_pre.shape[2])
        raise RuntimeError(
            f"layer {self.layer_idx} exact pre-RoPE cache mismatch: "
            f"captured={captured}, expected={key_count - 1}"
        )
    key_pre = torch.cat(
        (cached_key_pre.to(current_key_pre.device), current_key_pre),
        dim=2,
    )
    expanded_key_pre = runner.repeat_kv(key_pre, groups)
    pre_scores = torch.matmul(
        query_pre,
        expanded_key_pre.transpose(2, 3),
    ) * score_scale
    pre_scores = runner.add_attention_mask(pre_scores, attention_mask)

    current = key_count - 1
    delta = (current - positions).clamp_min(0)
    exact_window = float(controller.local_window)
    if "_w4k_" in controller.variant:
        exact_window = max(exact_window, 4096.0)
    elif "_w8k_" in controller.variant:
        exact_window = max(exact_window, 8192.0)
    preserve = (delta <= exact_window) | (
        positions < int(controller.sink_tokens)
    )
    preserve = preserve.view(1, 1, 1, -1)
    remote_end = max(0, current - int(controller.local_window))
    calibration_remote_end = max(0, current - int(exact_window))

    if controller.variant in MINIMAL_RESCUE_VARIANTS:
        keep_count = max(1, int(math.ceil(controller.ratio * key_count)))
        if controller.minimum_keep_tokens > 0:
            keep_count = max(keep_count, controller.minimum_keep_tokens)
        if controller.maximum_keep_tokens > 0:
            keep_count = min(keep_count, controller.maximum_keep_tokens)
        keep_count = min(key_count, keep_count)
        boundary = _rescue_boundary(controller.variant)
        clipped_scores = clipped_relative_scores(
            query_pre,
            expanded_key_pre,
            delta,
            self._local_global_inv_freq,
            modeling_qwen3.rotate_half,
            attention_scale,
            score_scale,
            boundary,
        )
        clipped_scores = runner.add_attention_mask(clipped_scores, attention_mask)
        calibrated = runner.calibrated_pre_scores(
            pre_scores,
            post_scores,
            remote_end,
            controller.sink_tokens,
        )
        if controller.variant in COUNTERFACTUAL_SELECTION_VARIANTS:
            suppression = clipped_scores.float() - post_scores.float()
            suppression = torch.relu(
                suppression - _rescue_gap_threshold(controller.variant)
            )
            selection_full = post_scores.float() + _rescue_fraction(
                controller.variant
            ) * suppression
            if controller.variant.startswith("cfs_dual_"):
                selection_full = torch.maximum(selection_full, calibrated.float())
            selection_scores = selection_full[0, :, 0, :]
        elif controller.variant == "exact_dual_top2_postscore":
            selection_scores = torch.maximum(
                calibrated[0, :, 0, :].float(),
                post_scores[0, :, 0, :].float(),
            )
        elif controller.variant.startswith("clipped128_top2_"):
            selection_scores = clipped_scores[0, :, 0, :]
        elif controller.variant.startswith("mpr_dual_"):
            selection_scores = torch.maximum(
                calibrated[0, :, 0, :].float(),
                post_scores[0, :, 0, :].float(),
            )
        elif _uses_exact_pre_selector(controller.variant):
            # The exact baseline selects from this tensor.  Strict variants do
            # not re-run Top-K below; they replay its frozen per-layer result.
            selection_scores = pre_scores[0, :, 0, :]
        else:
            selection_scores = pre_scores[0, :, 0, :]
        strict_signature = (
            _strict_reference_signature(controller.variant)
            if controller.variant in STRICT_MPR_VARIANTS
            else None
        )
        strict_plans = getattr(self, "_strict_mpr_reference_plans", {})
        strict_plan = (
            None if strict_signature is None else strict_plans.get(strict_signature)
        )
        selected, selected_remote, frozen_replay = select_or_replay_strict_support(
            variant=controller.variant,
            selection_scores=selection_scores,
            keep_count=keep_count,
            local_window=controller.local_window,
            sink_tokens=controller.sink_tokens,
            plan=strict_plan,
            epoch=int(getattr(self, "_strict_mpr_reference_epoch", -1)),
            layer_idx=int(self.layer_idx),
        )
        post_selected = runner.gather_scores(post_scores, selected)
        clipped_selected = runner.gather_scores(clipped_scores, selected)
        selected_value = _gather_vectors(expanded_value, selected)
        if controller.variant == "exact_pre_top2_postscore":
            self._strict_mpr_intervention_references = {}
            signatures = {
                _strict_reference_signature(variant)
                for variant in STRICT_MPR_VARIANTS
            }
            selected_key_pre = _gather_vectors(expanded_key_pre, selected)
            selected_delta = delta.view(1, -1).expand(
                selected.shape[0], -1
            ).gather(1, selected)
            plans: dict[
                tuple[float, float, float, int, int, float],
                FrozenStrictReferencePlan,
            ] = {}
            for signature in sorted(signatures):
                rescue_eligible = selected_remote & (
                    selected_delta > float(signature[0])
                )
                quantities = strict_phase_reference_quantities(
                    query_pre,
                    selected_key_pre,
                    selected_delta,
                    selected,
                    rescue_eligible,
                    self._local_global_inv_freq,
                    attention_scale,
                    score_scale,
                    signature[0],
                    signature[1],
                    signature[2],
                    signature[3],
                )
                plans[signature] = make_frozen_strict_reference_plan(
                    epoch=int(getattr(self, "_strict_mpr_reference_epoch", -1)),
                    layer_idx=int(self.layer_idx),
                    key_count=key_count,
                    signature=signature,
                    selected=selected,
                    selected_remote=selected_remote,
                    selected_delta=selected_delta,
                    rescue_eligible=rescue_eligible,
                    quantities=quantities,
                )
            self._strict_mpr_reference_plans = plans
        if controller.variant in (
            "clipped128_top2_clippedscore",
            "pre_top2_clipped128score",
        ):
            final_selected = torch.where(
                selected_remote,
                clipped_selected,
                post_selected,
            )
        elif controller.variant == "exact_pre_top2_blend25":
            calibrated_selected = runner.gather_scores(calibrated, selected)
            blended = (
                0.75 * post_selected.float()
                + 0.25 * calibrated_selected.float()
            ).to(post_selected.dtype)
            final_selected = torch.where(
                selected_remote,
                blended,
                post_selected,
            )
        elif controller.variant in STRICT_MPR_VARIANTS:
            assert frozen_replay is not None
            assert strict_signature is not None
            selected_key_pre = _gather_vectors(expanded_key_pre, selected)
            selected_delta = frozen_replay["selected_delta"]
            rescue_eligible = frozen_replay["rescue_eligible"]
            frequency_budget = _rescue_frequency_budget(
                controller.variant,
                int(query_pre.shape[-1] // 2),
            )
            phase_cap = _strict_phase_cap(controller.variant)
            token_cap = _strict_token_cap(controller.variant)
            random_support = _strict_random_support(controller.variant)
            random_reference: dict[str, torch.Tensor] | None = None
            if random_support:
                intervention_plans = getattr(
                    self, "_strict_mpr_intervention_references", {}
                )
                intervention_plan = intervention_plans.get(strict_signature)
                if intervention_plan is None:
                    raise RuntimeError(
                        "random strict control requires the non-random strict "
                        f"arm to run first for layer {self.layer_idx}"
                    )
                random_reference = replay_frozen_strict_intervention_reference(
                    intervention_plan,
                    epoch=int(getattr(self, "_strict_mpr_reference_epoch", -1)),
                    layer_idx=int(self.layer_idx),
                    key_count=key_count,
                    keep_count=int(post_selected.shape[1]),
                    head_count=int(post_selected.shape[0]),
                    signature=strict_signature,
                    device=post_selected.device,
                )
            final_selected, rescue_stats = strict_phase_rescue_scores(
                query_pre,
                selected_key_pre,
                selected_delta,
                post_selected,
                rescue_eligible,
                self._local_global_inv_freq,
                attention_scale,
                score_scale,
                boundary,
                _rescue_fraction(controller.variant),
                _rescue_gap_threshold(controller.variant),
                frequency_budget,
                phase_cap,
                random_frequency_support=random_support,
                random_seed_base=20260801 + int(self.layer_idx) * 1_000_003,
                frozen_raw_trigger=frozen_replay["raw_trigger"],
                frozen_trigger=frozen_replay["trigger"],
                frozen_desired_lift=frozen_replay["desired_lift"],
                frozen_counterfactual_gap=frozen_replay["counterfactual_gap"],
                matched_random_norm=(
                    None if random_reference is None else random_reference["shift_norm"]
                ),
                matched_random_support=(
                    None
                    if random_reference is None
                    else random_reference["support_count"]
                ),
                matched_random_lift=(
                    None
                    if random_reference is None
                    else random_reference["achieved_lift"]
                ),
            )
            rescue_stats["frozen_token_cap_mismatch_count"] = (
                torch.logical_xor(
                    rescue_stats["raw_trigger"],
                    frozen_replay["raw_trigger"],
                )
                .sum()
                + torch.logical_xor(
                    rescue_stats["trigger"],
                    frozen_replay["trigger"],
                ).sum()
            )
            if _strict_is_reference_arm(controller.variant):
                intervention_plans = getattr(
                    self, "_strict_mpr_intervention_references", {}
                )
                intervention_plans[strict_signature] = (
                    make_frozen_strict_intervention_reference(
                        epoch=int(
                            getattr(self, "_strict_mpr_reference_epoch", -1)
                        ),
                        layer_idx=int(self.layer_idx),
                        key_count=key_count,
                        signature=strict_signature,
                        stats=rescue_stats,
                    )
                )
                self._strict_mpr_intervention_references = intervention_plans
            partition_preserve = controller.variant.endswith("_masspreserve")
            if partition_preserve:
                # Preserve the triggered subset's partition.  Since every
                # non-trigger logit remains untouched, this also preserves the
                # complete selected partition.  The measured error below is
                # evaluated after the BF16 cast used by sparse softmax.
                final_selected = preserve_remote_partition(
                    final_selected,
                    post_selected,
                    rescue_stats["trigger"],
                )
                rescue_stats["partition_preservation_error"] = (
                    sparse_log_partition_error(final_selected, post_selected)
                )
            applied_lift = final_selected.float() - post_selected.float()
            nontrigger = ~rescue_stats["trigger"]
            if bool(nontrigger.any()):
                rescue_stats["nontrigger_noop_max"] = applied_lift.masked_select(
                    nontrigger
                ).abs().amax()
            rescue_stats["exact_lift"] = applied_lift
            record_strict_phase_metrics(
                controller,
                rescue_stats,
                applied_lift,
                frequency_budget,
                phase_cap,
                token_cap,
                random_support,
                partition_preserve,
            )
            record_phase_rescue_metrics(
                controller,
                selected,
                rescue_stats,
                key_count,
            )
        elif (
            controller.variant == "clipped128_top2_postscore"
            or controller.variant in COUNTERFACTUAL_SELECTION_VARIANTS
            or controller.variant in (
                "exact_pre_top2_postscore",
                "exact_dual_top2_postscore",
            )
        ):
            final_selected = post_selected
        else:
            selected_key_pre = _gather_vectors(expanded_key_pre, selected)
            selected_delta = delta.view(1, -1).expand(
                selected.shape[0], -1
            ).gather(1, selected)
            rescue_eligible = selected_remote & (
                selected_delta > float(boundary)
            )
            final_selected, rescue_stats = minimal_phase_rescue_scores(
                query_pre,
                selected_key_pre,
                selected_delta,
                post_selected,
                rescue_eligible,
                self._local_global_inv_freq,
                attention_scale,
                score_scale,
                boundary,
                _rescue_fraction(controller.variant),
                _rescue_gap_threshold(controller.variant),
                _rescue_frequency_budget(
                    controller.variant,
                    int(query_pre.shape[-1] // 2),
                ),
            )
            if controller.variant.endswith("_masspreserve"):
                final_selected = preserve_remote_partition(
                    final_selected,
                    post_selected,
                    selected_remote,
                )
                rescue_stats["exact_lift"] = (
                    final_selected.float() - post_selected.float()
                )
            record_phase_rescue_metrics(
                controller,
                selected,
                rescue_stats,
                key_count,
            )
        sparse_scores = final_selected.unsqueeze(0).unsqueeze(2)
        weights = F.softmax(sparse_scores.float(), dim=-1).to(query_post.dtype)
        controller.record(selected, weights, key_count, selected_remote)
        attention_output = torch.matmul(weights, selected_value)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attention_output), weights

    if controller.variant == "relative_rope_reconstructed_full":
        candidate_scores = relative_rotary_scores(
            query_pre,
            expanded_key_pre,
            delta,
            self._local_global_inv_freq,
            modeling_qwen3.rotate_half,
            attention_scale,
            score_scale,
            lambda phase: phase,
        )
        candidate_scores = runner.add_attention_mask(candidate_scores, attention_mask)
    elif controller.variant == "remote_nope_raw_full":
        candidate_scores = (
            pre_scores.float() * float(attention_scale) ** 2
        ).to(pre_scores.dtype)
    elif controller.variant == "remote_nope_cal_full":
        candidate_scores = runner.calibrated_pre_scores(
            pre_scores,
            post_scores,
            remote_end,
            controller.sink_tokens,
        )
    elif controller.variant.startswith("distance_fade_"):
        calibrated = runner.calibrated_pre_scores(
            pre_scores,
            post_scores,
            remote_end,
            controller.sink_tokens,
        )
        excess = (delta.float() - float(controller.local_window)).clamp_min(0.0)
        kappa = torch.exp(
            -torch.square(excess / _distance_scale(controller.variant))
        ).view(1, 1, 1, -1)
        candidate_scores = (
            kappa * post_scores.float() + (1.0 - kappa) * calibrated.float()
        ).to(post_scores.dtype)
    elif controller.variant.startswith("phase_coherent_"):
        candidate_scores = phase_coherent_scores(
            query_pre,
            expanded_key_pre,
            delta,
            self._local_global_inv_freq,
            modeling_qwen3.rotate_half,
            attention_scale,
            score_scale,
            _phase_cutoff(controller.variant),
            onset=_onset(controller.variant),
            normalize_multiplier=controller.variant.startswith(
                "phase_coherent_norm_"
            ),
        )
        candidate_scores = runner.add_attention_mask(candidate_scores, attention_mask)
        if controller.variant.endswith("_cal_full"):
            candidate_scores = runner.calibrated_pre_scores(
                candidate_scores,
                post_scores,
                calibration_remote_end,
                controller.sink_tokens,
            )
    elif controller.variant.startswith("phase_return_"):
        cutoff = _phase_cutoff(controller.variant)
        candidate_scores = relative_rotary_scores(
            query_pre,
            expanded_key_pre,
            delta,
            self._local_global_inv_freq,
            modeling_qwen3.rotate_half,
            attention_scale,
            score_scale,
            lambda phase: phase / (1.0 + torch.square(torch.abs(phase) / cutoff)),
        )
        candidate_scores = runner.add_attention_mask(candidate_scores, attention_mask)
    elif controller.variant == "phase_clamp_c2_full":
        cutoff = 2.0
        candidate_scores = relative_rotary_scores(
            query_pre,
            expanded_key_pre,
            delta,
            self._local_global_inv_freq,
            modeling_qwen3.rotate_half,
            attention_scale,
            score_scale,
            lambda phase: cutoff * torch.tanh(phase / cutoff),
        )
        candidate_scores = runner.add_attention_mask(candidate_scores, attention_mask)
    elif controller.variant.startswith("distance_saturate_"):
        onset = _onset(controller.variant)
        tau = _distance_scale(controller.variant)
        distance = delta.float()
        excess = (distance - onset).clamp_min(0.0)
        effective_distance = torch.where(
            distance <= onset,
            distance,
            onset + tau * torch.tanh(excess / tau),
        )
        ratio = effective_distance / distance.clamp_min(1.0)
        candidate_scores = relative_rotary_scores(
            query_pre,
            expanded_key_pre,
            delta,
            self._local_global_inv_freq,
            modeling_qwen3.rotate_half,
            attention_scale,
            score_scale,
            lambda phase: phase * ratio.unsqueeze(-1),
        )
        candidate_scores = runner.add_attention_mask(candidate_scores, attention_mask)
    elif controller.variant == "distance_log_w4k_t4k_full":
        onset = 4096.0
        tau = 4096.0
        distance = delta.float()
        excess = (distance - onset).clamp_min(0.0)
        effective_distance = torch.where(
            distance <= onset,
            distance,
            onset + tau * torch.log1p(excess / tau),
        )
        ratio = effective_distance / distance.clamp_min(1.0)
        candidate_scores = relative_rotary_scores(
            query_pre,
            expanded_key_pre,
            delta,
            self._local_global_inv_freq,
            modeling_qwen3.rotate_half,
            attention_scale,
            score_scale,
            lambda phase: phase * ratio.unsqueeze(-1),
        )
        candidate_scores = runner.add_attention_mask(candidate_scores, attention_mask)
    else:
        raise ValueError(f"unsupported phase variant: {controller.variant}")

    merged_scores = torch.where(preserve, post_scores, candidate_scores)
    weights = F.softmax(merged_scores.float(), dim=-1).to(query_post.dtype)
    selected = _all_positions(query_post.shape[1], key_count, key_post.device)
    remote_mask = (~preserve[0, 0, 0]).view(1, -1).expand(query_post.shape[1], -1)
    controller.record(selected, weights, key_count, remote_mask)

    attention_output = torch.matmul(weights, expanded_value)
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attention_output), weights


runner.local_global_attention_forward = phase_kernel_attention_forward
runner.patch_model = phase_patch_model
runner.base.prefill_sequence = capture_prefill_sequence


if __name__ == "__main__":
    runner.main()
