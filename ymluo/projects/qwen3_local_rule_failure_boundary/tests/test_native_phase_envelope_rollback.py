from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_native_phase_envelope_rollback_8b as npe  # noqa: E402


def _rotate_split_half(
    value: torch.Tensor,
    position: float,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    phase = float(position) * inv_freq
    cos = torch.cat((torch.cos(phase), torch.cos(phase)))
    sin = torch.cat((torch.sin(phase), torch.sin(phase)))
    half = value.shape[-1] // 2
    rotated_half = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
    return value * cos + rotated_half * sin


def test_pair_formula_reconstructs_native_relative_rope_score() -> None:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(1, 2, 1, 128, generator=generator)
    keys = torch.randn(1, 2, 3, 128, generator=generator)
    inv_freq = torch.exp(torch.linspace(math.log(1.0), math.log(1e-4), 64))
    query_position = 257.0
    key_positions = torch.tensor([3.0, 101.0, 256.0])
    distance = query_position - key_positions

    a, b = npe.pair_coefficients(query, keys)
    reconstructed = npe.scores_at_distance(a, b, distance, inv_freq, 0.125)

    query_post = _rotate_split_half(query, query_position, inv_freq)
    direct = []
    for index, position in enumerate(key_positions):
        key_post = _rotate_split_half(keys[:, :, index : index + 1], position, inv_freq)
        direct.append(
            (query_post * key_post).sum(dim=-1).squeeze(0).squeeze(-1) * 0.125
        )
    direct_score = torch.stack(direct, dim=-1)
    torch.testing.assert_close(reconstructed, direct_score, atol=2e-5, rtol=2e-5)


def test_native_phase_envelope_uses_median_mad_certificate() -> None:
    a = torch.zeros(1, 2, 64)
    b = torch.zeros_like(a)
    a[..., 0] = 1.0
    inv_freq = torch.zeros(64)
    native = torch.tensor([[0.25, 1.25]])
    distance = torch.tensor([[200.0, 200.0]])
    anchors = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 128.0)

    result = npe.native_phase_envelope(
        a, b, native, distance, inv_freq, 1.0, anchors, 2.5
    )
    torch.testing.assert_close(result["median"], torch.ones_like(native))
    torch.testing.assert_close(result["mad"], torch.zeros_like(native))
    assert result["trigger"].tolist() == [[True, False]]
    assert result["remote"].all()


def test_coherent_rollback_finds_minimum_refined_distance_and_is_exact_noop() -> None:
    # Only plane 0 is active: s(d)=cos(pi*d/4).  At d=4, s=-1.  The first
    # rollback satisfying s>=0.5 is r=8/3.  All 64 planes receive d-r.
    a = torch.zeros(1, 2, 64)
    b = torch.zeros_like(a)
    a[..., 0] = 1.0
    inv_freq = torch.zeros(64)
    inv_freq[0] = math.pi / 4.0
    distance = torch.tensor([[4.0, 4.0]])
    native = torch.tensor([[-1.0, 123.456]], dtype=torch.float32)
    trigger = torch.tensor([[True, False]])
    target = torch.tensor([[0.5, 0.5]])

    corrected, stats = npe.coherent_rollback_search(
        a,
        b,
        native,
        distance,
        trigger,
        target,
        inv_freq,
        1.0,
        (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
        dense_rollback_tokens=4,
        coarse_search_points=4,
        refinement_steps=3,
        refinement_bins=8,
    )
    assert 2.66 <= float(stats["rollback"][0, 0]) <= 2.68
    assert float(corrected[0, 0]) >= 0.5
    assert bool(stats["success"][0, 0])
    # The untriggered path must return the exact supplied native logit, not a
    # float32 RoPE reconstruction.
    assert corrected[0, 1].item() == native[0, 1].item()
    assert stats["rollback"][0, 1].item() == 0.0


def test_mass_preserve_changes_only_triggered_logits() -> None:
    native = torch.tensor([[0.0, 1.0, 2.0, -1.0]])
    corrected = torch.tensor([[2.0, 9.0, 4.0, 8.0]])
    trigger = torch.tensor([[True, False, True, False]])
    output = npe.preserve_trigger_partition(corrected, native, trigger)

    assert output[0, 1].item() == native[0, 1].item()
    assert output[0, 3].item() == native[0, 3].item()
    torch.testing.assert_close(
        torch.exp(output[trigger]).sum(),
        torch.exp(native[trigger]).sum(),
        atol=1e-6,
        rtol=1e-6,
    )


def test_random_control_matches_trigger_rate_per_head_deterministically() -> None:
    trigger = torch.zeros(2, 20, dtype=torch.bool)
    trigger[0, [1, 3, 5]] = True
    trigger[1, [2, 4, 6, 8, 10]] = True
    eligible = torch.ones_like(trigger)
    positions = torch.arange(20).view(1, -1).expand(2, -1)

    first = npe.deterministic_matched_mask(trigger, eligible, positions, 17)
    second = npe.deterministic_matched_mask(trigger, eligible, positions, 17)
    assert torch.equal(first, second)
    assert first.sum(dim=-1).tolist() == trigger.sum(dim=-1).tolist()
    assert not bool((first & trigger).any())


def test_envelope_rejects_fewer_than_eight_anchors() -> None:
    a = torch.zeros(1, 1, 64)
    b = torch.zeros_like(a)
    try:
        npe.native_phase_envelope(
            a,
            b,
            torch.zeros(1, 1),
            torch.ones(1, 1),
            torch.ones(64),
            1.0,
            (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            2.5,
        )
    except ValueError as error:
        assert "eight" in str(error)
    else:
        raise AssertionError("expected anchor validation failure")


def test_npe_support_contract_uses_exact_pre_local_global_selector() -> None:
    """Guard against replacing the matched baseline by an unrestricted Top-K.

    The exact-pre baseline reserves sink/local/current tokens inside the same
    2% budget.  The unrestricted selector differs on this constructed input,
    which makes the regression observable without loading a model.
    """

    scores = torch.arange(200, dtype=torch.float32).view(1, -1)
    scores[0, :40] += 10_000.0
    keep_count = 20
    selected, selected_remote = npe.runner.local_global_selection(
        scores,
        keep_count,
        local_window=8,
        sink_tokens=4,
    )
    baseline_selected, baseline_remote = npe.runner.local_global_selection(
        scores,
        keep_count,
        local_window=8,
        sink_tokens=4,
    )
    unrestricted = npe.runner.force_current_topk(scores, keep_count)

    assert torch.equal(selected, baseline_selected)
    assert torch.equal(selected_remote, baseline_remote)
    assert not torch.equal(selected, unrestricted)
    assert selected.shape[-1] == keep_count
    assert selected_remote.sum().item() == keep_count - 1 - 8 - 4


def test_reconstruction_margin_guard_rejects_numerical_scale_ambiguity() -> None:
    a = torch.zeros(1, 1, 64)
    b = torch.zeros_like(a)
    a[..., 0] = 1.0
    result = npe.native_phase_envelope(
        a,
        b,
        native_score=torch.tensor([[0.9]]),
        distance=torch.tensor([[200.0]]),
        inv_freq=torch.zeros(64),
        total_scale=1.0,
        anchor_distances=(0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 128.0),
        mad_lambda=2.5,
        reconstruction_guard_multiplier=2.0,
        reconstruction_guard_floor=1e-3,
    )
    assert bool(result["raw_trigger"][0, 0])
    assert not bool(result["guarded_trigger"][0, 0])
    torch.testing.assert_close(result["reconstruction_error"], torch.tensor([[0.1]]))


def test_frozen_plan_snapshots_assignment_and_replays_current_query_score() -> None:
    positions = torch.tensor([[2, 5, 9]])
    remote = torch.tensor([[True, True, False]])
    trigger = torch.tensor([[True, False, False]])
    repair = {
        "success": trigger.clone(),
        "rollback": torch.tensor([[3.0, 0.0, 0.0]]),
        "effective_distance": torch.tensor([[4.0, 2.0, 0.0]]),
    }
    random_repair = {
        "applied": torch.tensor([[False, True, False]]),
        "rollback": torch.tensor([[0.0, 1.0, 0.0]]),
        "effective_distance": torch.tensor([[7.0, 1.0, 0.0]]),
    }
    plan = npe.make_frozen_reference_plan(
        epoch=npe._REFERENCE_EPOCH,
        key_count=10,
        positions=positions,
        selected_remote=remote,
        raw_trigger=trigger,
        certificate_trigger=trigger,
        target_lower=torch.tensor([[0.5, 0.5, 0.5]]),
        repair=repair,
        random_repair=random_repair,
    )
    positions[0, 0] = 99
    module = torch.nn.Module()
    module._npe_frozen_reference_plan = plan
    loaded = npe.load_frozen_reference_plan(module, key_count=10, device=torch.device("cpu"))
    assert loaded["positions"].tolist() == [[2, 5, 9]]

    # Replaying the fixed d_eff uses the current A/B, while the unassigned
    # score remains the exact native value.
    a = torch.zeros(1, 3, 64)
    b = torch.zeros_like(a)
    a[..., 0] = torch.tensor([[2.0, 4.0, 6.0]])
    corrected, stats = npe.apply_frozen_distance_plan(
        native_score=torch.tensor([[-10.0, 123.0, 456.0]]),
        a=a,
        b=b,
        applied=loaded["applied"],
        effective_distance=loaded["effective_distance"],
        rollback=loaded["rollback"],
        inv_freq=torch.zeros(64),
        total_scale=1.0,
    )
    assert corrected.tolist() == [[2.0, 123.0, 456.0]]
    assert stats["rollback"].tolist() == [[3.0, 0.0, 0.0]]


def test_extended_metrics_report_random_overlap_change_and_partition_error() -> None:
    controller = npe.runner.Controller(
        variant="npe_frozen_random_matched_pre_top2",
        ratio=0.02,
        minimum_keep_tokens=0,
        maximum_keep_tokens=0,
        local_window=1,
        sink_tokens=0,
        evidence_spans=(),
    )
    selected = torch.tensor([[0, 1, 2]])
    remote = torch.ones_like(selected, dtype=torch.bool)
    trigger = torch.tensor([[True, True, False]])
    applied = torch.tensor([[False, True, True]])
    native = torch.tensor([[0.0, 1.0, 2.0]])
    final = torch.tensor([[0.0, 3.0, 4.0]])
    certificate = {
        "remote": remote,
        "trigger": trigger,
        "raw_trigger": trigger,
        "guarded_trigger": trigger,
        "suppression_gap": torch.ones_like(native),
        "median": torch.zeros_like(native),
        "mad": torch.zeros_like(native),
        "lower": torch.tensor([[-1.0, 2.0, 3.0]]),
        "reconstruction_error": torch.tensor([[0.01, 0.02, 0.03]]),
    }
    repair = {
        "applied": applied,
        "success": applied,
        "rollback": torch.ones_like(native),
        "effective_distance": torch.ones_like(native),
        "score_lift": final - native,
        "comparison_trigger": trigger,
        "mass_preserve": True,
    }
    npe._record_npe_metrics(
        controller, selected, certificate, repair, native, final, key_count=3
    )
    summary = controller.metrics.summary()
    assert summary["npe_actual_changed_count"] == 2.0
    assert summary["npe_actual_changed_fraction"] == 1.0
    assert summary["npe_final_envelope_satisfaction_fraction"] == 1.0
    assert summary["npe_random_overlap_fraction"] == 0.5
    assert abs(summary["npe_random_jaccard"] - 1.0 / 3.0) < 1e-8
    assert abs(summary["npe_native_reconstruction_error_max"] - 0.03) < 1e-7
    assert summary["npe_mass_log_partition_error_max"] > 0.0
