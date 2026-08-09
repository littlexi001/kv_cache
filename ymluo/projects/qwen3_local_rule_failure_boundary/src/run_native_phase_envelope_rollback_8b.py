from __future__ import annotations

"""Frozen Qwen3-8B screen for Native Phase Envelope + Coherent Rollback.

The experiment changes only the final-query attention calculation.  Prefix
states are produced by the pretrained model with native RoPE.  Online variants
share the exact-pre Top-2% selector policy, while ``npe_frozen_*`` variants
replay the literal per-layer support and treatment plan recorded on the
untreated ``npe_native_pre_top2`` trajectory.  The latter are the strict causal
controls: downstream query drift cannot reselect or reassign treatment.

Exact pre-RoPE K capture is imported from
``run_phase_coherent_rope_probe_8b``; that module is not modified here.
"""

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

import run_phase_coherent_rope_probe_8b as phase_runner


runner = phase_runner.runner

NPE_VARIANTS = (
    "npe_native_pre_top2",
    "npe_distance_clip_pre_top2",
    "npe_rollback_pre_top2",
    "npe_rollback_masspreserve_pre_top2",
    "npe_random_matched_pre_top2",
    "npe_frozen_rollback_pre_top2",
    "npe_frozen_rollback_masspreserve_pre_top2",
    "npe_frozen_random_matched_pre_top2",
)
FROZEN_REFERENCE_VARIANTS = (
    "npe_frozen_rollback_pre_top2",
    "npe_frozen_rollback_masspreserve_pre_top2",
    "npe_frozen_random_matched_pre_top2",
)
runner.VARIANTS = tuple(dict.fromkeys((*runner.VARIANTS, *NPE_VARIANTS)))

_PHASE_FORWARD = runner.local_global_attention_forward
_BASE_PARSE_ARGS = runner.parse_args
_BASE_METRIC_SUMMARY = runner.MetricAccumulator.summary
_BASE_SUMMARIZE = runner.summarize
_BASE_PREFILL_SEQUENCE = runner.base.prefill_sequence


@dataclass(frozen=True)
class EnvelopeConfig:
    anchor_distances: tuple[float, ...] = (
        0.0,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
    )
    mad_lambda: float = 2.5
    dense_rollback_tokens: int = 64
    coarse_search_points: int = 48
    refinement_steps: int = 2
    refinement_bins: int = 8
    reconstruction_guard_multiplier: float = 2.0
    reconstruction_guard_floor: float = 1e-3


_CONFIG = EnvelopeConfig()
_REFERENCE_EPOCH = 0


@dataclass(frozen=True)
class FrozenReferencePlan:
    """Untreated, per-layer treatment assignment stored on CPU.

    The baseline fixes support, eligibility, certificate assignment, target,
    and effective distance.  A frozen intervention may recompute only the QK
    coefficients and score under its evolving query state.  In particular, it
    may not reselect tokens or condition treatment on a post-intervention Q.
    """

    epoch: int
    key_count: int
    positions: torch.Tensor
    selected_remote: torch.Tensor
    raw_trigger: torch.Tensor
    certificate_trigger: torch.Tensor
    applied: torch.Tensor
    target_lower: torch.Tensor
    rollback: torch.Tensor
    effective_distance: torch.Tensor
    random_applied: torch.Tensor
    random_rollback: torch.Tensor
    random_effective_distance: torch.Tensor


def _cpu_snapshot(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to("cpu").contiguous().clone()


def _plan_tensor(plan: FrozenReferencePlan, name: str, device: torch.device) -> torch.Tensor:
    return getattr(plan, name).to(device=device)


def reference_aware_prefill_sequence(
    model: Any,
    prompt_prefix: torch.Tensor,
    chunk_size: int,
) -> tuple[Any, float]:
    """Invalidate treatment plans whenever a new prompt prefix is prefetched."""

    global _REFERENCE_EPOCH
    _REFERENCE_EPOCH += 1
    for module in model.modules():
        if module.__class__.__name__ == "Qwen3Attention":
            module._npe_frozen_reference_plan = None
    return _BASE_PREFILL_SEQUENCE(model, prompt_prefix, chunk_size)


runner.base.prefill_sequence = reference_aware_prefill_sequence


def _parse_anchor_distances(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(item) for item in raw.split(",") if item.strip()}))
    if len(values) < 8:
        raise ValueError("Native Phase Envelope needs at least eight anchors")
    if values[0] < 0.0:
        raise ValueError("anchor distances must be non-negative")
    return values


def parse_args() -> argparse.Namespace:
    """Add NPE options without copying the upstream experiment parser."""

    global _CONFIG
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument(
        "--npe-anchor-distances",
        default="0,1,2,4,8,16,32,64,128",
    )
    extra.add_argument("--npe-mad-lambda", type=float, default=2.5)
    extra.add_argument("--npe-dense-rollback-tokens", type=int, default=64)
    extra.add_argument("--npe-coarse-search-points", type=int, default=48)
    extra.add_argument("--npe-refinement-steps", type=int, default=2)
    extra.add_argument("--npe-refinement-bins", type=int, default=8)
    extra.add_argument(
        "--npe-reconstruction-guard-multiplier", type=float, default=2.0
    )
    extra.add_argument(
        "--npe-reconstruction-guard-floor", type=float, default=1e-3
    )
    npe, remaining = extra.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        args = _BASE_PARSE_ARGS()
    finally:
        sys.argv = original_argv

    anchors = _parse_anchor_distances(npe.npe_anchor_distances)
    if npe.npe_mad_lambda < 0.0:
        raise ValueError("--npe-mad-lambda must be non-negative")
    if npe.npe_dense_rollback_tokens < 0:
        raise ValueError("--npe-dense-rollback-tokens must be non-negative")
    if npe.npe_coarse_search_points < 1:
        raise ValueError("--npe-coarse-search-points must be positive")
    if npe.npe_refinement_steps < 0 or npe.npe_refinement_bins < 2:
        raise ValueError("invalid rollback refinement configuration")
    if (
        npe.npe_reconstruction_guard_multiplier < 0.0
        or npe.npe_reconstruction_guard_floor < 0.0
    ):
        raise ValueError("reconstruction guard values must be non-negative")
    _CONFIG = EnvelopeConfig(
        anchor_distances=anchors,
        mad_lambda=float(npe.npe_mad_lambda),
        dense_rollback_tokens=int(npe.npe_dense_rollback_tokens),
        coarse_search_points=int(npe.npe_coarse_search_points),
        refinement_steps=int(npe.npe_refinement_steps),
        refinement_bins=int(npe.npe_refinement_bins),
        reconstruction_guard_multiplier=float(
            npe.npe_reconstruction_guard_multiplier
        ),
        reconstruction_guard_floor=float(npe.npe_reconstruction_guard_floor),
    )
    requested_variants = [
        item.strip() for item in str(args.variants).split(",") if item.strip()
    ]
    frozen_indices = [
        index
        for index, variant in enumerate(requested_variants)
        if variant in FROZEN_REFERENCE_VARIANTS
    ]
    if frozen_indices:
        if "npe_native_pre_top2" not in requested_variants:
            raise ValueError(
                "frozen NPE variants require npe_native_pre_top2 in --variants"
            )
        if requested_variants.index("npe_native_pre_top2") > min(frozen_indices):
            raise ValueError(
                "npe_native_pre_top2 must precede every frozen NPE variant"
            )
    # Persist the method configuration in upstream config.json.
    for name, value in vars(npe).items():
        setattr(args, name, value)
    return args


runner.parse_args = parse_args


def pair_coefficients(
    query_pre: torch.Tensor,
    selected_key_pre: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the 64 RoPE-plane coefficients A and B for every head/key."""

    if query_pre.shape[0] != 1 or query_pre.shape[-2] != 1:
        raise ValueError("query_pre must be [1, heads, 1, head_dim]")
    if selected_key_pre.dim() != 4 or selected_key_pre.shape[0] != 1:
        raise ValueError("selected_key_pre must be [1, heads, keys, head_dim]")
    head_dim = int(query_pre.shape[-1])
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    half = head_dim // 2
    qx = query_pre[0, :, 0, :half].unsqueeze(1)
    qy = query_pre[0, :, 0, half:].unsqueeze(1)
    kx = selected_key_pre[0, :, :, :half]
    ky = selected_key_pre[0, :, :, half:]
    # Qwen split-half RoPE convention.  For delta=t-p>=0, the relative score is
    # A cos(delta*omega) + B sin(delta*omega).
    a = qx.float() * kx.float() + qy.float() * ky.float()
    b = qx.float() * ky.float() - qy.float() * kx.float()
    return a, b


def scores_at_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    distance: torch.Tensor,
    inv_freq: torch.Tensor,
    total_scale: float,
) -> torch.Tensor:
    """Evaluate one coherent distance shared by every frequency of a pair."""

    if a.shape != b.shape:
        raise ValueError("A and B must have identical shapes")
    if a.shape[-1] != inv_freq.numel():
        raise ValueError("frequency count does not match the RoPE planes")
    phase = distance.float().unsqueeze(-1) * inv_freq.float()
    return (
        (a * torch.cos(phase) + b * torch.sin(phase)).sum(dim=-1)
        * float(total_scale)
    )


def native_phase_envelope(
    a: torch.Tensor,
    b: torch.Tensor,
    native_score: torch.Tensor,
    distance: torch.Tensor,
    inv_freq: torch.Tensor,
    total_scale: float,
    anchor_distances: Sequence[float],
    mad_lambda: float,
    reconstruction_guard_multiplier: float = 0.0,
    reconstruction_guard_floor: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Build a robust local native-RoPE envelope and certify suppression.

    For local anchors D, m=median_d s(d), MAD=median_d |s(d)-m|, and

        lower = m - lambda * (1.4826 MAD + eps).

    A remote pair is certified only when its exact native post-RoPE score lies
    below this lower envelope.  The exact native score, rather than a float32
    reconstruction, is used in the decision.
    """

    anchors = torch.as_tensor(
        tuple(anchor_distances), device=a.device, dtype=torch.float32
    )
    if anchors.numel() < 8:
        raise ValueError("at least eight local-anchor distances are required")
    # [heads, keys, anchors, frequencies]
    phase = anchors.view(1, 1, -1, 1) * inv_freq.float().view(1, 1, 1, -1)
    anchor_scores = (
        a.unsqueeze(-2) * torch.cos(phase) + b.unsqueeze(-2) * torch.sin(phase)
    ).sum(dim=-1) * float(total_scale)
    median = anchor_scores.median(dim=-1).values
    mad = (anchor_scores - median.unsqueeze(-1)).abs().median(dim=-1).values
    robust_sigma = 1.4826 * mad
    lower = median - float(mad_lambda) * (robust_sigma + 1e-6)
    remote = distance.float() > float(anchors.max().item())
    suppression_gap = lower - native_score.float()
    trigger = remote & (suppression_gap > 0.0)
    reconstructed_native = scores_at_distance(
        a, b, distance, inv_freq, total_scale
    )
    reconstruction_error = (reconstructed_native - native_score.float()).abs()
    guard_margin = torch.maximum(
        reconstruction_error * float(reconstruction_guard_multiplier),
        torch.full_like(reconstruction_error, float(reconstruction_guard_floor)),
    )
    guarded_trigger = trigger & (suppression_gap > guard_margin)
    return {
        "anchor_scores": anchor_scores,
        "median": median,
        "mad": mad,
        "robust_sigma": robust_sigma,
        "lower": lower,
        "remote": remote,
        "trigger": trigger,
        "raw_trigger": trigger,
        "guarded_trigger": guarded_trigger,
        "suppression_gap": suppression_gap,
        "reconstructed_native": reconstructed_native,
        "reconstruction_error": reconstruction_error,
        "guard_margin": guard_margin,
    }


def _candidate_rollbacks(
    distance: torch.Tensor,
    anchor_distances: Sequence[float],
    dense_rollback_tokens: int,
    coarse_search_points: int,
) -> torch.Tensor:
    """Ascending per-pair grid, including dense, coarse, and exact anchors."""

    count = int(distance.numel())
    dense = torch.arange(
        0,
        int(dense_rollback_tokens) + 1,
        device=distance.device,
        dtype=torch.float32,
    ).view(1, -1).expand(count, -1)
    dense = torch.minimum(dense, distance.float().unsqueeze(-1))

    start = torch.minimum(
        distance.float(),
        torch.full_like(distance.float(), float(dense_rollback_tokens)),
    )
    fractions = torch.linspace(
        1.0 / float(coarse_search_points),
        1.0,
        int(coarse_search_points),
        device=distance.device,
    )
    coarse = start.unsqueeze(-1) + (
        distance.float() - start
    ).unsqueeze(-1) * fractions.view(1, -1)

    anchors = torch.as_tensor(
        tuple(anchor_distances), device=distance.device, dtype=torch.float32
    )
    anchor_rollback = (
        distance.float().unsqueeze(-1) - anchors.view(1, -1)
    ).clamp_min(0.0)
    anchor_rollback = torch.minimum(
        anchor_rollback, distance.float().unsqueeze(-1)
    )
    candidates = torch.cat((dense, coarse, anchor_rollback), dim=-1)
    return candidates.sort(dim=-1).values


def _first_passing_candidate(
    a: torch.Tensor,
    b: torch.Tensor,
    distance: torch.Tensor,
    target: torch.Tensor,
    candidates: torch.Tensor,
    inv_freq: torch.Tensor,
    total_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    effective = (distance.unsqueeze(-1) - candidates).clamp_min(0.0)
    phase = effective.unsqueeze(-1) * inv_freq.float().view(1, 1, -1)
    values = (
        a.unsqueeze(1) * torch.cos(phase) + b.unsqueeze(1) * torch.sin(phase)
    ).sum(dim=-1) * float(total_scale)
    passing = values >= target.unsqueeze(-1)
    has_solution = passing.any(dim=-1)
    first = passing.to(torch.int64).argmax(dim=-1)
    row = torch.arange(distance.numel(), device=distance.device)
    rollback = candidates[row, first]
    score = values[row, first]
    return rollback, score, has_solution


def coherent_rollback_search(
    a: torch.Tensor,
    b: torch.Tensor,
    native_score: torch.Tensor,
    distance: torch.Tensor,
    trigger: torch.Tensor,
    target: torch.Tensor,
    inv_freq: torch.Tensor,
    total_scale: float,
    anchor_distances: Sequence[float],
    dense_rollback_tokens: int = 64,
    coarse_search_points: int = 48,
    refinement_steps: int = 2,
    refinement_bins: int = 8,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Find the smallest sampled rollback that re-enters the envelope.

    Crucially, a triggered head/token pair receives one scalar effective
    distance; all 64 RoPE frequencies use that same distance.  The first pass
    searches an ascending rollback grid.  Optional refinement repeatedly
    subdivides the bracket immediately before the first passing point.
    Untriggered logits are returned bit-for-bit from ``native_score``.
    """

    corrected = native_score.clone()
    rollback_full = torch.zeros_like(native_score, dtype=torch.float32)
    effective_full = distance.float().clone()
    success_full = torch.zeros_like(trigger)
    if not bool(trigger.any()):
        return corrected, {
            "rollback": rollback_full,
            "effective_distance": effective_full,
            "success": success_full,
            "applied": trigger.clone(),
            "score_lift": corrected.float() - native_score.float(),
        }

    flat_trigger = trigger.reshape(-1)
    indices = flat_trigger.nonzero(as_tuple=False).flatten()
    flat_a = a.reshape(-1, a.shape[-1])
    flat_b = b.reshape(-1, b.shape[-1])
    flat_native = native_score.reshape(-1).float()
    flat_distance = distance.reshape(-1).float()
    flat_target = target.reshape(-1).float()
    flat_corrected = corrected.reshape(-1)
    flat_rollback = rollback_full.reshape(-1)
    flat_effective = effective_full.reshape(-1)
    flat_success = success_full.reshape(-1)

    for start in range(0, int(indices.numel()), int(chunk_size)):
        idx = indices[start : start + int(chunk_size)]
        ca = flat_a[idx]
        cb = flat_b[idx]
        cd = flat_distance[idx]
        ct = flat_target[idx]
        candidates = _candidate_rollbacks(
            cd,
            anchor_distances,
            dense_rollback_tokens,
            coarse_search_points,
        )
        rollback, score, solved = _first_passing_candidate(
            ca, cb, cd, ct, candidates, inv_freq, total_scale
        )

        # Refine only the bracket immediately preceding the first passing
        # sampled point.  This preserves the "minimum on the evaluated grid"
        # interpretation without assuming global monotonicity of RoPE scores.
        for _ in range(int(refinement_steps)):
            previous = torch.where(
                candidates < rollback.unsqueeze(-1),
                candidates,
                torch.full_like(candidates, -torch.inf),
            ).amax(dim=-1)
            previous = torch.where(
                torch.isfinite(previous), previous, torch.zeros_like(previous)
            )
            fractions = torch.linspace(
                1.0 / float(refinement_bins),
                1.0,
                int(refinement_bins),
                device=cd.device,
            )
            refined = previous.unsqueeze(-1) + (
                rollback - previous
            ).unsqueeze(-1) * fractions.view(1, -1)
            new_rollback, new_score, new_solved = _first_passing_candidate(
                ca, cb, cd, ct, refined, inv_freq, total_scale
            )
            use = solved & new_solved
            rollback = torch.where(use, new_rollback, rollback)
            score = torch.where(use, new_score, score)
            # Keep the adaptive grid so the next pass refines the new bracket.
            candidates = torch.cat((candidates, refined), dim=-1).sort(dim=-1).values

        # Exact local anchors guarantee a solution unless numerical input is
        # pathological.  Leave an unsolved pair native rather than inventing a
        # correction.
        use = solved
        flat_corrected[idx] = torch.where(
            use, score.to(flat_corrected.dtype), flat_native[idx].to(flat_corrected.dtype)
        )
        flat_rollback[idx] = torch.where(use, rollback, torch.zeros_like(rollback))
        flat_effective[idx] = torch.where(use, cd - rollback, cd)
        flat_success[idx] = use

    return corrected, {
        "rollback": rollback_full,
        "effective_distance": effective_full,
        "success": success_full,
        "applied": trigger & success_full,
        "score_lift": corrected.float() - native_score.float(),
    }


def preserve_trigger_partition(
    corrected: torch.Tensor,
    native: torch.Tensor,
    trigger: torch.Tensor,
) -> torch.Tensor:
    """Preserve trigger-set exp mass while leaving every other logit exact."""

    negative_infinity = torch.full_like(corrected.float(), -torch.inf)
    repaired = torch.where(trigger, corrected.float(), negative_infinity)
    original = torch.where(trigger, native.float(), negative_infinity)
    shift = torch.logsumexp(repaired, dim=-1, keepdim=True) - torch.logsumexp(
        original, dim=-1, keepdim=True
    )
    shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
    return torch.where(trigger, corrected.float() - shift, native.float()).to(
        native.dtype
    )


def deterministic_matched_mask(
    trigger: torch.Tensor,
    eligible: torch.Tensor,
    positions: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Deterministic random control with the trigger count matched per head."""

    output = torch.zeros_like(trigger)
    heads = int(trigger.shape[0])
    for head in range(heads):
        count = int(trigger[head].sum().item())
        if count == 0:
            continue
        pool = eligible[head] & ~trigger[head]
        if int(pool.sum().item()) < count:
            pool = eligible[head]
        candidates = pool.nonzero(as_tuple=False).flatten()
        count = min(count, int(candidates.numel()))
        if count == 0:
            continue
        x = (
            (positions[head, candidates].float() + 1.0) * 12.9898
            + float(head + 1) * 78.233
            + float(layer_idx + 1) * 37.719
        )
        pseudo = torch.remainder(torch.sin(x) * 43758.5453, 1.0)
        chosen = candidates[torch.topk(pseudo, k=count, largest=True).indices]
        output[head, chosen] = True
    return output


def matched_random_rollback(
    native_score: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    distance: torch.Tensor,
    true_trigger: torch.Tensor,
    true_rollback: torch.Tensor,
    eligible: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    total_scale: float,
    layer_idx: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply matched rollback fractions to random eligible pairs."""

    random_mask = deterministic_matched_mask(
        true_trigger, eligible, positions, layer_idx
    )
    random_rollback = torch.zeros_like(distance, dtype=torch.float32)
    for head in range(int(true_trigger.shape[0])):
        source = true_rollback[head][true_trigger[head] & (true_rollback[head] > 0)]
        target_idx = random_mask[head].nonzero(as_tuple=False).flatten()
        count = min(int(source.numel()), int(target_idx.numel()))
        if count == 0:
            continue
        # Match relative rollback, not raw tokens, so assignment remains valid
        # when random pairs have different absolute distances.
        source_distance = distance[head][
            true_trigger[head] & (true_rollback[head] > 0)
        ].float().clamp_min(1.0)
        fraction = (source / source_distance).clamp(0.0, 1.0)
        order = torch.argsort(fraction)
        fraction = fraction[order][:count]
        target_idx = target_idx[:count]
        random_rollback[head, target_idx] = (
            fraction * distance[head, target_idx].float()
        )
    effective = distance.float() - random_rollback
    candidate = scores_at_distance(a, b, effective, inv_freq, total_scale)
    corrected = torch.where(random_mask, candidate.to(native_score.dtype), native_score)
    return corrected, {
        "rollback": random_rollback,
        "effective_distance": effective,
        "success": random_mask,
        "applied": random_mask,
        "score_lift": corrected.float() - native_score.float(),
        "comparison_trigger": true_trigger,
    }


def make_frozen_reference_plan(
    *,
    epoch: int,
    key_count: int,
    positions: torch.Tensor,
    selected_remote: torch.Tensor,
    raw_trigger: torch.Tensor,
    certificate_trigger: torch.Tensor,
    target_lower: torch.Tensor,
    repair: dict[str, torch.Tensor],
    random_repair: dict[str, torch.Tensor],
) -> FrozenReferencePlan:
    """Snapshot the untreated assignment; no post-treatment tensor is stored."""

    applied = certificate_trigger & repair["success"]
    return FrozenReferencePlan(
        epoch=int(epoch),
        key_count=int(key_count),
        positions=_cpu_snapshot(positions),
        selected_remote=_cpu_snapshot(selected_remote),
        raw_trigger=_cpu_snapshot(raw_trigger),
        certificate_trigger=_cpu_snapshot(certificate_trigger),
        applied=_cpu_snapshot(applied),
        target_lower=_cpu_snapshot(target_lower.float()),
        rollback=_cpu_snapshot(repair["rollback"].float()),
        effective_distance=_cpu_snapshot(repair["effective_distance"].float()),
        random_applied=_cpu_snapshot(random_repair["applied"]),
        random_rollback=_cpu_snapshot(random_repair["rollback"].float()),
        random_effective_distance=_cpu_snapshot(
            random_repair["effective_distance"].float()
        ),
    )


def load_frozen_reference_plan(
    attention: torch.nn.Module,
    *,
    key_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    plan = getattr(attention, "_npe_frozen_reference_plan", None)
    if not isinstance(plan, FrozenReferencePlan):
        raise RuntimeError(
            "frozen NPE variant requires npe_native_pre_top2 earlier in the "
            "same prompt's variant order"
        )
    if plan.epoch != _REFERENCE_EPOCH:
        raise RuntimeError(
            f"stale frozen NPE plan: plan_epoch={plan.epoch}, "
            f"prompt_epoch={_REFERENCE_EPOCH}"
        )
    if int(plan.key_count) != int(key_count):
        raise RuntimeError(
            f"frozen NPE key-count mismatch: plan={plan.key_count}, now={key_count}"
        )
    return {
        name: _plan_tensor(plan, name, device)
        for name in (
            "positions",
            "selected_remote",
            "raw_trigger",
            "certificate_trigger",
            "applied",
            "target_lower",
            "rollback",
            "effective_distance",
            "random_applied",
            "random_rollback",
            "random_effective_distance",
        )
    }


def apply_frozen_distance_plan(
    native_score: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    applied: torch.Tensor,
    effective_distance: torch.Tensor,
    rollback: torch.Tensor,
    inv_freq: torch.Tensor,
    total_scale: float,
    *,
    comparison_trigger: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply a baseline-assigned distance using the current intervention Q.

    Support, treatment mask, target and effective distance are frozen.  The
    current query may change A/B and therefore the realized score; this is a
    downstream treatment effect, not a reason to reassign the treatment.
    """

    candidate = scores_at_distance(
        a, b, effective_distance, inv_freq, total_scale
    )
    corrected = torch.where(applied, candidate.to(native_score.dtype), native_score)
    result = {
        "rollback": rollback.float(),
        "effective_distance": effective_distance.float(),
        "success": applied,
        "applied": applied,
        "score_lift": corrected.float() - native_score.float(),
        "certificate_success": applied,
    }
    if comparison_trigger is not None:
        result["comparison_trigger"] = comparison_trigger
    return corrected, result


def _gather_vectors(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    index = positions.view(1, positions.shape[0], -1, 1).expand(
        1, positions.shape[0], positions.shape[1], values.shape[-1]
    )
    return values.gather(2, index)


def _record_npe_metrics(
    controller: runner.Controller,
    selected: torch.Tensor,
    certificate: dict[str, torch.Tensor],
    repair: dict[str, torch.Tensor],
    native_score: torch.Tensor,
    final_score: torch.Tensor,
    key_count: int,
) -> None:
    metrics = controller.metrics
    remote = certificate["remote"]
    trigger = certificate["trigger"]
    raw_trigger = certificate.get("raw_trigger", trigger)
    guarded_trigger = certificate.get("guarded_trigger", trigger)
    applied = repair["applied"]
    success = repair.get("certificate_success", repair["success"] & trigger)
    gap = certificate["suppression_gap"]
    metrics.npe_support_count = int(getattr(metrics, "npe_support_count", 0)) + int(
        selected.numel()
    )
    metrics.npe_remote_count = int(getattr(metrics, "npe_remote_count", 0)) + int(
        remote.sum().item()
    )
    metrics.npe_trigger_count = int(getattr(metrics, "npe_trigger_count", 0)) + int(
        trigger.sum().item()
    )
    metrics.npe_applied_count = int(getattr(metrics, "npe_applied_count", 0)) + int(
        applied.sum().item()
    )
    metrics.npe_success_count = int(getattr(metrics, "npe_success_count", 0)) + int(
        success.sum().item()
    )
    metrics.npe_suppression_gap_sum = float(
        getattr(metrics, "npe_suppression_gap_sum", 0.0)
    ) + float(gap.masked_select(trigger).sum().item())
    metrics.npe_anchor_median_sum = float(
        getattr(metrics, "npe_anchor_median_sum", 0.0)
    ) + float(certificate["median"].masked_select(remote).sum().item())
    metrics.npe_anchor_mad_sum = float(
        getattr(metrics, "npe_anchor_mad_sum", 0.0)
    ) + float(certificate["mad"].masked_select(remote).sum().item())
    metrics.npe_score_lift_sum = float(
        getattr(metrics, "npe_score_lift_sum", 0.0)
    ) + float(repair["score_lift"].masked_select(applied).sum().item())

    unmodified = ~applied
    no_op_error = (final_score.float() - native_score.float()).abs().masked_select(
        unmodified
    )
    current_error = float(no_op_error.max().item()) if no_op_error.numel() else 0.0
    metrics.npe_unmodified_max_error = max(
        float(getattr(metrics, "npe_unmodified_max_error", 0.0)), current_error
    )

    rollback = repair["rollback"].masked_select(applied).detach().float().cpu()
    effective = repair["effective_distance"].masked_select(applied).detach().float().cpu()
    rollback_values = getattr(metrics, "npe_rollback_values", [])
    rollback_values.extend(rollback.tolist())
    metrics.npe_rollback_values = rollback_values
    metrics.npe_effective_distance_sum = float(
        getattr(metrics, "npe_effective_distance_sum", 0.0)
    ) + float(effective.sum().item())

    gold = controller.evidence_mask(key_count, selected.device)[selected]
    metrics.npe_selected_gold_count = int(
        getattr(metrics, "npe_selected_gold_count", 0)
    ) + int(gold.sum().item())
    metrics.npe_gold_trigger_count = int(
        getattr(metrics, "npe_gold_trigger_count", 0)
    ) + int((gold & trigger).sum().item())

    reconstruction_error = certificate["reconstruction_error"].masked_select(remote)
    metrics.npe_reconstruction_error_sum = float(
        getattr(metrics, "npe_reconstruction_error_sum", 0.0)
    ) + float(reconstruction_error.sum().item())
    metrics.npe_reconstruction_error_count = int(
        getattr(metrics, "npe_reconstruction_error_count", 0)
    ) + int(reconstruction_error.numel())
    if reconstruction_error.numel():
        metrics.npe_reconstruction_error_max = max(
            float(getattr(metrics, "npe_reconstruction_error_max", 0.0)),
            float(reconstruction_error.max().item()),
        )
    raw_count = int(raw_trigger.sum().item())
    guarded_count = int(guarded_trigger.sum().item())
    metrics.npe_raw_trigger_count = int(
        getattr(metrics, "npe_raw_trigger_count", 0)
    ) + raw_count
    metrics.npe_guarded_trigger_count = int(
        getattr(metrics, "npe_guarded_trigger_count", 0)
    ) + guarded_count

    changed = applied & (final_score != native_score)
    metrics.npe_actual_changed_count = int(
        getattr(metrics, "npe_actual_changed_count", 0)
    ) + int(changed.sum().item())
    satisfied = applied & (final_score.float() >= certificate["lower"].float())
    metrics.npe_final_satisfied_count = int(
        getattr(metrics, "npe_final_satisfied_count", 0)
    ) + int(satisfied.sum().item())

    comparison = repair.get("comparison_trigger")
    if comparison is not None:
        intersection = applied & comparison
        union = applied | comparison
        metrics.npe_random_overlap_count = int(
            getattr(metrics, "npe_random_overlap_count", 0)
        ) + int(intersection.sum().item())
        metrics.npe_random_union_count = int(
            getattr(metrics, "npe_random_union_count", 0)
        ) + int(union.sum().item())
        metrics.npe_random_reference_count = int(
            getattr(metrics, "npe_random_reference_count", 0)
        ) + int(comparison.sum().item())

    if bool(repair.get("mass_preserve", False)):
        negative_infinity = torch.full_like(final_score.float(), -torch.inf)
        final_partition = torch.logsumexp(
            torch.where(applied, final_score.float(), negative_infinity), dim=-1
        )
        native_partition = torch.logsumexp(
            torch.where(applied, native_score.float(), negative_infinity), dim=-1
        )
        valid = torch.isfinite(final_partition) & torch.isfinite(native_partition)
        partition_error = (final_partition - native_partition).abs()[valid]
        metrics.npe_mass_partition_error_sum = float(
            getattr(metrics, "npe_mass_partition_error_sum", 0.0)
        ) + float(partition_error.sum().item())
        metrics.npe_mass_partition_error_count = int(
            getattr(metrics, "npe_mass_partition_error_count", 0)
        ) + int(partition_error.numel())
        if partition_error.numel():
            metrics.npe_mass_partition_error_max = max(
                float(getattr(metrics, "npe_mass_partition_error_max", 0.0)),
                float(partition_error.max().item()),
            )


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = q * (len(ordered) - 1)
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return ordered[low]
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _metric_summary(self: runner.MetricAccumulator) -> dict[str, float]:
    summary = _BASE_METRIC_SUMMARY(self)
    support = int(getattr(self, "npe_support_count", 0))
    remote = int(getattr(self, "npe_remote_count", 0))
    triggers = int(getattr(self, "npe_trigger_count", 0))
    applied = int(getattr(self, "npe_applied_count", 0))
    success = int(getattr(self, "npe_success_count", 0))
    gold = int(getattr(self, "npe_selected_gold_count", 0))
    raw_triggers = int(getattr(self, "npe_raw_trigger_count", 0))
    guarded_triggers = int(getattr(self, "npe_guarded_trigger_count", 0))
    changed = int(getattr(self, "npe_actual_changed_count", 0))
    reconstruction_count = int(
        getattr(self, "npe_reconstruction_error_count", 0)
    )
    random_reference = int(getattr(self, "npe_random_reference_count", 0))
    random_union = int(getattr(self, "npe_random_union_count", 0))
    mass_partition_count = int(
        getattr(self, "npe_mass_partition_error_count", 0)
    )
    rollback = list(getattr(self, "npe_rollback_values", []))
    summary.update(
        {
            "npe_support_count": float(support),
            "npe_remote_support_count": float(remote),
            "npe_certificate_trigger_count": float(triggers),
            "npe_certificate_trigger_fraction": triggers / max(1, remote),
            "npe_applied_count": float(applied),
            "npe_applied_fraction": applied / max(1, remote),
            "npe_search_success_fraction": success / max(1, triggers),
            "npe_suppression_gap_mean": float(
                getattr(self, "npe_suppression_gap_sum", 0.0)
            )
            / max(1, triggers),
            "npe_local_anchor_median_mean": float(
                getattr(self, "npe_anchor_median_sum", 0.0)
            )
            / max(1, remote),
            "npe_local_anchor_mad_mean": float(
                getattr(self, "npe_anchor_mad_sum", 0.0)
            )
            / max(1, remote),
            "npe_score_lift_mean": float(
                getattr(self, "npe_score_lift_sum", 0.0)
            )
            / max(1, applied),
            "npe_rollback_tokens_mean": statistics.fmean(rollback)
            if rollback
            else 0.0,
            "npe_rollback_tokens_median": _quantile(rollback, 0.50),
            "npe_rollback_tokens_p90": _quantile(rollback, 0.90),
            "npe_rollback_tokens_p95": _quantile(rollback, 0.95),
            "npe_rollback_tokens_max": max(rollback) if rollback else 0.0,
            "npe_effective_distance_mean": float(
                getattr(self, "npe_effective_distance_sum", 0.0)
            )
            / max(1, applied),
            "npe_gold_certificate_trigger_fraction": int(
                getattr(self, "npe_gold_trigger_count", 0)
            )
            / max(1, gold),
            "npe_unmodified_native_max_error": float(
                getattr(self, "npe_unmodified_max_error", 0.0)
            ),
            "npe_native_reconstruction_error_mean": float(
                getattr(self, "npe_reconstruction_error_sum", 0.0)
            )
            / max(1, reconstruction_count),
            "npe_native_reconstruction_error_max": float(
                getattr(self, "npe_reconstruction_error_max", 0.0)
            ),
            "npe_margin_guard_rejected_fraction": (
                raw_triggers - guarded_triggers
            )
            / max(1, raw_triggers),
            "npe_raw_trigger_fraction": raw_triggers / max(1, remote),
            "npe_guarded_trigger_fraction": guarded_triggers / max(1, remote),
            "npe_reference_search_success_fraction": int(
                getattr(self, "npe_reference_success_count", 0)
            )
            / max(1, int(getattr(self, "npe_reference_trigger_count", 0))),
            "npe_actual_changed_count": float(changed),
            "npe_actual_changed_fraction": changed / max(1, applied),
            "npe_final_envelope_satisfaction_fraction": int(
                getattr(self, "npe_final_satisfied_count", 0)
            )
            / max(1, applied),
            "npe_random_overlap_fraction": int(
                getattr(self, "npe_random_overlap_count", 0)
            )
            / max(1, random_reference),
            "npe_random_jaccard": int(
                getattr(self, "npe_random_overlap_count", 0)
            )
            / max(1, random_union),
            "npe_mass_log_partition_error_mean": float(
                getattr(self, "npe_mass_partition_error_sum", 0.0)
            )
            / max(1, mass_partition_count),
            "npe_mass_log_partition_error_max": float(
                getattr(self, "npe_mass_partition_error_max", 0.0)
            ),
        }
    )
    return summary


runner.MetricAccumulator.summary = _metric_summary


def _summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = _BASE_SUMMARIZE(rows)
    extras = (
        "npe_certificate_trigger_fraction",
        "npe_applied_fraction",
        "npe_search_success_fraction",
        "npe_suppression_gap_mean",
        "npe_score_lift_mean",
        "npe_rollback_tokens_mean",
        "npe_rollback_tokens_median",
        "npe_rollback_tokens_p90",
        "npe_rollback_tokens_p95",
        "npe_effective_distance_mean",
        "npe_gold_certificate_trigger_fraction",
        "npe_unmodified_native_max_error",
        "npe_native_reconstruction_error_mean",
        "npe_native_reconstruction_error_max",
        "npe_margin_guard_rejected_fraction",
        "npe_raw_trigger_fraction",
        "npe_guarded_trigger_fraction",
        "npe_reference_search_success_fraction",
        "npe_actual_changed_count",
        "npe_actual_changed_fraction",
        "npe_final_envelope_satisfaction_fraction",
        "npe_random_overlap_fraction",
        "npe_random_jaccard",
        "npe_mass_log_partition_error_mean",
        "npe_mass_log_partition_error_max",
    )
    by_key = {
        (int(row["target_context_tokens"]), str(row["variant"])): row
        for row in output
    }
    for key, aggregate in by_key.items():
        selected = [
            row
            for row in rows
            if (int(row["target_context_tokens"]), str(row["variant"])) == key
        ]
        for name in extras:
            if selected and name in selected[0]:
                aggregate[name] = statistics.fmean(
                    float(row[name]) for row in selected
                )
    return output


runner.summarize = _summarize


def native_phase_envelope_attention_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_value: Any | None = None,
    cache_position: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    controller = runner._ACTIVE_CONTROLLER
    if controller is None or controller.variant not in NPE_VARIANTS:
        return _PHASE_FORWARD(
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
            current_key_post, current_value, self.layer_idx, cache_kwargs
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
        query_post, expanded_key_post.transpose(2, 3)
    ) * score_scale
    post_scores = runner.add_attention_mask(post_scores, attention_mask)

    cached_key_pre = getattr(self, "_phase_pre_key_cache", None)
    if cached_key_pre is None or int(cached_key_pre.shape[2]) != key_count - 1:
        captured = -1 if cached_key_pre is None else int(cached_key_pre.shape[2])
        raise RuntimeError(
            f"layer {self.layer_idx} exact pre-RoPE cache mismatch: "
            f"captured={captured}, expected={key_count - 1}"
        )
    key_pre = torch.cat(
        (cached_key_pre.to(current_key_pre.device), current_key_pre), dim=2
    )
    expanded_key_pre = runner.repeat_kv(key_pre, groups)
    pre_scores = torch.matmul(
        query_pre, expanded_key_pre.transpose(2, 3)
    ) * score_scale
    pre_scores = runner.add_attention_mask(pre_scores, attention_mask)

    keep_count = max(1, int(math.ceil(controller.ratio * key_count)))
    if controller.minimum_keep_tokens > 0:
        keep_count = max(keep_count, controller.minimum_keep_tokens)
    if controller.maximum_keep_tokens > 0:
        keep_count = min(keep_count, controller.maximum_keep_tokens)
    keep_count = min(key_count, keep_count)

    frozen_plan: dict[str, torch.Tensor] | None = None
    if controller.variant in FROZEN_REFERENCE_VARIANTS:
        frozen_plan = load_frozen_reference_plan(
            self, key_count=key_count, device=pre_scores.device
        )
        selected = frozen_plan["positions"].long()
        selected_remote = frozen_plan["selected_remote"].bool()
        if int(selected.shape[-1]) != keep_count:
            raise RuntimeError(
                f"frozen NPE support width {selected.shape[-1]} != budget {keep_count}"
            )
    else:
        # Reuse the *literal* selector used by exact_pre_top2_postscore.  In
        # particular, the 2% budget reserves the same sink/local/current slots
        # and spends only the remainder on semantic remote keys.
        selected, selected_remote = runner.local_global_selection(
            pre_scores[0, :, 0, :],
            keep_count,
            controller.local_window,
            controller.sink_tokens,
        )
    post_selected = runner.gather_scores(post_scores, selected)
    selected_key_pre = _gather_vectors(expanded_key_pre, selected)
    selected_value = _gather_vectors(expanded_value, selected)

    positions = torch.arange(key_count, device=selected.device)
    delta_all = (key_count - 1 - positions).clamp_min(0)
    selected_delta = delta_all.view(1, -1).expand(
        selected.shape[0], -1
    ).gather(1, selected)
    a, b = pair_coefficients(query_pre, selected_key_pre)
    attention_scale = float(getattr(self, "_phase_attention_scale", 1.0))
    total_scale = score_scale * attention_scale**2
    certificate = native_phase_envelope(
        a,
        b,
        post_selected,
        selected_delta,
        self._local_global_inv_freq,
        total_scale,
        _CONFIG.anchor_distances,
        _CONFIG.mad_lambda,
        _CONFIG.reconstruction_guard_multiplier,
        _CONFIG.reconstruction_guard_floor,
    )
    # Sink tokens can be far away in absolute distance, but they are a fixed
    # part of the exact-pre baseline rather than semantic remote candidates.
    # Restrict every intervention and diagnostic denominator to the selector's
    # own remote mask, so the native NPE baseline remains exactly matched.
    certificate["remote"] = selected_remote
    certificate["raw_trigger"] = certificate["raw_trigger"] & selected_remote
    certificate["guarded_trigger"] = (
        certificate["guarded_trigger"] & selected_remote
    )
    certificate["trigger"] = certificate["raw_trigger"]

    if frozen_plan is not None:
        # Treatment assignment and target come exclusively from the untreated
        # reference.  Current-Q envelope values remain diagnostics only.
        certificate["raw_trigger"] = frozen_plan["raw_trigger"].bool()
        certificate["guarded_trigger"] = frozen_plan[
            "certificate_trigger"
        ].bool()
        certificate["trigger"] = frozen_plan["certificate_trigger"].bool()
        certificate["lower"] = frozen_plan["target_lower"].float()
        certificate["suppression_gap"] = (
            certificate["lower"] - post_selected.float()
        )

    if controller.variant == "npe_native_pre_top2":
        # Plan once on the untreated trajectory.  The reconstruction guard is
        # used only by the new frozen-reference controls; existing online
        # variants retain their historical raw-trigger behavior.
        reference_trigger = certificate["guarded_trigger"]
        _, reference_repair = coherent_rollback_search(
            a,
            b,
            post_selected,
            selected_delta,
            reference_trigger,
            certificate["lower"],
            self._local_global_inv_freq,
            total_scale,
            _CONFIG.anchor_distances,
            _CONFIG.dense_rollback_tokens,
            _CONFIG.coarse_search_points,
            _CONFIG.refinement_steps,
            _CONFIG.refinement_bins,
        )
        _, reference_random_repair = matched_random_rollback(
            post_selected,
            a,
            b,
            selected_delta,
            reference_trigger & reference_repair["success"],
            reference_repair["rollback"],
            selected_remote,
            selected,
            self._local_global_inv_freq,
            total_scale,
            int(self.layer_idx),
        )
        self._npe_frozen_reference_plan = make_frozen_reference_plan(
            epoch=_REFERENCE_EPOCH,
            key_count=key_count,
            positions=selected,
            selected_remote=selected_remote,
            raw_trigger=certificate["raw_trigger"],
            certificate_trigger=reference_trigger,
            target_lower=certificate["lower"],
            repair=reference_repair,
            random_repair=reference_random_repair,
        )
        controller.metrics.npe_reference_trigger_count = int(
            getattr(controller.metrics, "npe_reference_trigger_count", 0)
        ) + int(reference_trigger.sum().item())
        controller.metrics.npe_reference_success_count = int(
            getattr(controller.metrics, "npe_reference_success_count", 0)
        ) + int(
            (reference_trigger & reference_repair["success"]).sum().item()
        )

    empty_repair = {
        "rollback": torch.zeros_like(post_selected, dtype=torch.float32),
        "effective_distance": selected_delta.float(),
        "success": torch.zeros_like(certificate["trigger"]),
        "applied": torch.zeros_like(certificate["trigger"]),
        "score_lift": torch.zeros_like(post_selected, dtype=torch.float32),
    }
    final_selected = post_selected
    repair = empty_repair
    if controller.variant in FROZEN_REFERENCE_VARIANTS:
        if frozen_plan is None:
            raise AssertionError("frozen variant lost its reference plan")
        if controller.variant == "npe_frozen_random_matched_pre_top2":
            final_selected, repair = apply_frozen_distance_plan(
                post_selected,
                a,
                b,
                frozen_plan["random_applied"].bool(),
                frozen_plan["random_effective_distance"].float(),
                frozen_plan["random_rollback"].float(),
                self._local_global_inv_freq,
                total_scale,
                comparison_trigger=frozen_plan["applied"].bool(),
            )
            # Search success describes the frozen certificate, not the random
            # treatment mask.
            repair["certificate_success"] = frozen_plan["applied"].bool()
        else:
            final_selected, repair = apply_frozen_distance_plan(
                post_selected,
                a,
                b,
                frozen_plan["applied"].bool(),
                frozen_plan["effective_distance"].float(),
                frozen_plan["rollback"].float(),
                self._local_global_inv_freq,
                total_scale,
            )
            if controller.variant == "npe_frozen_rollback_masspreserve_pre_top2":
                final_selected = preserve_trigger_partition(
                    final_selected,
                    post_selected,
                    repair["applied"],
                )
                repair["score_lift"] = (
                    final_selected.float() - post_selected.float()
                )
                repair["mass_preserve"] = True
    elif controller.variant == "npe_distance_clip_pre_top2":
        clip_distance = float(max(_CONFIG.anchor_distances))
        effective = selected_delta.float().clamp_max(clip_distance)
        clipped = scores_at_distance(
            a, b, effective, self._local_global_inv_freq, total_scale
        )
        applied = certificate["remote"]
        final_selected = torch.where(
            applied, clipped.to(post_selected.dtype), post_selected
        )
        repair = {
            "rollback": torch.where(
                applied, selected_delta.float() - effective, torch.zeros_like(effective)
            ),
            "effective_distance": effective,
            "success": applied,
            "applied": applied,
            "score_lift": final_selected.float() - post_selected.float(),
        }
    elif controller.variant in (
        "npe_rollback_pre_top2",
        "npe_rollback_masspreserve_pre_top2",
        "npe_random_matched_pre_top2",
    ):
        rollback_selected, true_repair = coherent_rollback_search(
            a,
            b,
            post_selected,
            selected_delta,
            certificate["trigger"],
            certificate["lower"],
            self._local_global_inv_freq,
            total_scale,
            _CONFIG.anchor_distances,
            _CONFIG.dense_rollback_tokens,
            _CONFIG.coarse_search_points,
            _CONFIG.refinement_steps,
            _CONFIG.refinement_bins,
        )
        if controller.variant == "npe_random_matched_pre_top2":
            final_selected, repair = matched_random_rollback(
                post_selected,
                a,
                b,
                selected_delta,
                certificate["trigger"] & true_repair["success"],
                true_repair["rollback"],
                certificate["remote"],
                selected,
                self._local_global_inv_freq,
                total_scale,
                int(self.layer_idx),
            )
            # The random control applies corrections elsewhere, but the search
            # success statistic must still describe the true certificate set.
            repair["certificate_success"] = true_repair["success"]
        elif controller.variant == "npe_rollback_masspreserve_pre_top2":
            final_selected = preserve_trigger_partition(
                rollback_selected,
                post_selected,
                true_repair["applied"],
            )
            repair = dict(true_repair)
            repair["score_lift"] = final_selected.float() - post_selected.float()
            repair["mass_preserve"] = True
        else:
            final_selected = rollback_selected
            repair = true_repair

    _record_npe_metrics(
        controller,
        selected,
        certificate,
        repair,
        post_selected,
        final_selected,
        key_count,
    )
    sparse_scores = final_selected.unsqueeze(0).unsqueeze(2)
    weights = F.softmax(sparse_scores.float(), dim=-1).to(query_post.dtype)
    controller.record(selected, weights, key_count, certificate["remote"])
    attention_output = torch.matmul(weights, selected_value)
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attention_output), weights


runner.local_global_attention_forward = native_phase_envelope_attention_forward


if __name__ == "__main__":
    runner.main()
