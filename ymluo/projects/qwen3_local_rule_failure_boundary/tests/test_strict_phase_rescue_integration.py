from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import run_phase_coherent_rope_probe_8b as target  # noqa: E402


def _frozen_plan() -> target.FrozenStrictReferencePlan:
    selected = torch.tensor([[0, 100, 299], [1, 120, 299]], dtype=torch.long)
    selected_remote = torch.tensor(
        [[False, True, False], [False, True, False]], dtype=torch.bool
    )
    selected_delta = 299 - selected
    rescue_eligible = selected_remote.clone()
    quantities = {
        "raw_trigger": torch.tensor(
            [[False, True, False], [False, True, False]], dtype=torch.bool
        ),
        "trigger": torch.tensor(
            [[False, True, False], [False, True, False]], dtype=torch.bool
        ),
        "desired_lift": torch.tensor(
            [[0.0, 0.5, 0.0], [0.0, 0.75, 0.0]], dtype=torch.float32
        ),
        "counterfactual_gap": torch.tensor(
            [[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float32
        ),
    }
    return target.make_frozen_strict_reference_plan(
        epoch=7,
        layer_idx=3,
        key_count=300,
        signature=target._strict_reference_signature(
            "strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25"
        ),
        selected=selected,
        selected_remote=selected_remote,
        selected_delta=selected_delta,
        rescue_eligible=rescue_eligible,
        quantities=quantities,
    )


def test_t1_t4_variant_matrix_is_complete_and_explicit() -> None:
    expected = {
        f"strict_mpr_pre_w128_lift25_gap1_t{cap}_f8_cap0p25{suffix}"
        for cap in (1, 4)
        for suffix in (
            "",
            "_random",
            "_masspreserve",
            "_random_masspreserve",
        )
    }
    assert set(target.TOKEN_SPARSE_STRICT_MPR_VARIANTS) == expected
    assert {target._strict_token_cap(item) for item in expected} == {1, 4}
    reference_arms = {
        item for item in expected if target._strict_is_reference_arm(item)
    }
    assert reference_arms == {
        "strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25",
        "strict_mpr_pre_w128_lift25_gap1_t4_f8_cap0p25",
    }


def test_token_cap_uses_gap_then_position_for_deterministic_ties() -> None:
    raw = torch.ones(2, 4, dtype=torch.bool)
    gap = torch.tensor([[5.0, 5.0, 5.0, 4.0], [1.0, 7.0, 3.0, 7.0]])
    selected = torch.tensor([[30, 10, 20, 40], [8, 90, 2, 70]])

    t1 = target.cap_strict_token_triggers(raw, gap, selected, token_cap=1)
    t2 = target.cap_strict_token_triggers(raw, gap, selected, token_cap=2)

    # Head zero has a three-way gap tie: token position 10 wins, then 20.
    assert torch.equal(t1[0], torch.tensor([False, True, False, False]))
    assert torch.equal(t2[0], torch.tensor([False, True, True, False]))
    # Head one has a gap tie at slots 1/3: smaller token position 70 wins.
    assert torch.equal(t1[1], torch.tensor([False, False, False, True]))
    assert torch.equal(t2[1], torch.tensor([False, True, False, True]))


def test_strict_variants_share_exact_pre_selector_policy() -> None:
    torch.manual_seed(17)
    pre_scores = torch.randn(3, 97)
    exact_selected, exact_remote = target.runner.local_global_selection(
        pre_scores,
        keep_count=19,
        local_window=8,
        sink_tokens=3,
    )

    assert target._uses_exact_pre_selector("exact_pre_top2_postscore")
    for variant in target.STRICT_MPR_VARIANTS:
        assert target._uses_exact_pre_selector(variant)
        # This checks the baseline selector policy only.  Real strict forwards
        # replay the captured result and are covered by the adversarial test.
        strict_selected, strict_remote = target.runner.local_global_selection(
            pre_scores,
            keep_count=19,
            local_window=8,
            sink_tokens=3,
        )
        torch.testing.assert_close(strict_selected, exact_selected)
        torch.testing.assert_close(strict_remote, exact_remote)

    assert not target._uses_exact_pre_selector("exact_dual_top2_postscore")


def test_frozen_plan_snapshots_cpu_tensors_and_replay_returns_copies() -> None:
    source = torch.tensor([[0, 100, 299], [1, 120, 299]], dtype=torch.long)
    remote = torch.tensor(
        [[False, True, False], [False, True, False]], dtype=torch.bool
    )
    delta = 299 - source
    quantities = {
        "raw_trigger": remote.clone(),
        "trigger": remote.clone(),
        "desired_lift": torch.ones(2, 3),
        "counterfactual_gap": torch.full((2, 3), 4.0),
    }
    plan = target.make_frozen_strict_reference_plan(
        epoch=7,
        layer_idx=3,
        key_count=300,
        signature=(128.0, 0.25, 1.0, 1, 8, 0.25),
        selected=source,
        selected_remote=remote,
        selected_delta=delta,
        rescue_eligible=remote,
        quantities=quantities,
    )
    source.fill_(2)
    remote.zero_()
    quantities["trigger"].zero_()

    assert plan.selected.device.type == "cpu"
    torch.testing.assert_close(
        plan.selected, torch.tensor([[0, 100, 299], [1, 120, 299]])
    )
    replay = target.replay_frozen_strict_reference_plan(
        plan,
        epoch=7,
        layer_idx=3,
        key_count=300,
        keep_count=3,
        head_count=2,
        signature=(128.0, 0.25, 1.0, 1, 8, 0.25),
        device=torch.device("cpu"),
    )
    replay["selected"].fill_(0)
    torch.testing.assert_close(
        plan.selected, torch.tensor([[0, 100, 299], [1, 120, 299]])
    )


def test_strict_replay_ignores_adversarial_current_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _frozen_plan()

    def fail_if_reselected(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("strict replay consulted the live selector")

    monkeypatch.setattr(target.runner, "local_global_selection", fail_if_reselected)
    selected, selected_remote, replay = target.select_or_replay_strict_support(
        variant="strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25",
        selection_scores=torch.arange(600, dtype=torch.float32).reshape(2, 300),
        keep_count=3,
        local_window=1,
        sink_tokens=1,
        plan=plan,
        epoch=7,
        layer_idx=3,
    )

    torch.testing.assert_close(selected, plan.selected)
    torch.testing.assert_close(selected_remote, plan.selected_remote)
    assert replay is not None
    torch.testing.assert_close(replay["trigger"], plan.trigger)
    torch.testing.assert_close(replay["desired_lift"], plan.desired_lift)


def test_frozen_plan_fails_closed_when_missing_or_stale() -> None:
    scores = torch.randn(2, 300)
    with pytest.raises(RuntimeError, match="requires exact_pre"):
        target.select_or_replay_strict_support(
            variant="strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25",
            selection_scores=scores,
            keep_count=3,
            local_window=1,
            sink_tokens=1,
            plan=None,
            epoch=7,
            layer_idx=3,
        )
    with pytest.raises(RuntimeError, match="stale/incompatible"):
        target.select_or_replay_strict_support(
            variant="strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25",
            selection_scores=scores,
            keep_count=3,
            local_window=1,
            sink_tokens=1,
            plan=_frozen_plan(),
            epoch=8,
            layer_idx=3,
        )


def test_frozen_plan_fails_closed_on_cap_or_capped_mask_mismatch() -> None:
    plan = _frozen_plan()
    with pytest.raises(RuntimeError, match="stale/incompatible"):
        target.replay_frozen_strict_reference_plan(
            replace(plan, token_cap=4),
            epoch=7,
            layer_idx=3,
            key_count=300,
            keep_count=3,
            head_count=2,
            signature=target._strict_reference_signature(
                "strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25"
            ),
            device=torch.device("cpu"),
        )
    broken_trigger = plan.trigger.clone()
    broken_trigger[0, 0] = True
    with pytest.raises(RuntimeError, match="token cap"):
        target.replay_frozen_strict_reference_plan(
            replace(plan, trigger=broken_trigger),
            epoch=7,
            layer_idx=3,
            key_count=300,
            keep_count=3,
            head_count=2,
            signature=plan.signature,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    "override",
    [
        {"epoch": 8},
        {"layer_idx": 4},
        {"key_count": 301},
        {"keep_count": 2},
        {"head_count": 1},
        {"signature": (128.0, 0.25, 2.0, 1, 8, 0.25)},
        {"signature": (128.0, 0.25, 1.0, 4, 8, 0.25)},
        {"signature": (128.0, 0.25, 1.0, 1, 16, 0.25)},
        {"signature": (128.0, 0.25, 1.0, 1, 8, 0.5)},
    ],
)
def test_reference_replay_validates_every_identity_field(
    override: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "epoch": 7,
        "layer_idx": 3,
        "key_count": 300,
        "keep_count": 3,
        "head_count": 2,
        "signature": (128.0, 0.25, 1.0, 1, 8, 0.25),
        "device": torch.device("cpu"),
    }
    arguments.update(override)
    with pytest.raises(RuntimeError, match="stale/incompatible"):
        target.replay_frozen_strict_reference_plan(_frozen_plan(), **arguments)


def test_cross_arm_intervention_reference_is_snapshotted_and_validated() -> None:
    stats = {
        "shift_norm": torch.tensor([[0.2, 0.4]]),
        "active_plane_count": torch.tensor([[2, 4]]),
        "solver_lift": torch.tensor([[0.1, 0.3]]),
    }
    plan = target.make_frozen_strict_intervention_reference(
        epoch=5,
        layer_idx=2,
        key_count=99,
        signature=(128.0, 0.25, 1.0, 1, 8, 0.25),
        stats=stats,
    )
    stats["shift_norm"].zero_()
    replay = target.replay_frozen_strict_intervention_reference(
        plan,
        epoch=5,
        layer_idx=2,
        key_count=99,
        keep_count=2,
        head_count=1,
        signature=(128.0, 0.25, 1.0, 1, 8, 0.25),
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(replay["shift_norm"], torch.tensor([[0.2, 0.4]]))
    with pytest.raises(RuntimeError, match="stale/incompatible"):
        target.replay_frozen_strict_intervention_reference(
            plan,
            epoch=6,
            layer_idx=2,
            key_count=99,
            keep_count=2,
            head_count=1,
            signature=(128.0, 0.25, 1.0, 1, 8, 0.25),
            device=torch.device("cpu"),
        )


def test_strict_rescue_hits_target_with_budget_and_cap() -> None:
    # One selected remote token is suppressed by phase 1.5; the second token is
    # local and must remain an exact no-op.
    query = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    keys = torch.tensor([[[[1.0, 0.0], [0.3, 0.7]]]], dtype=torch.float32)
    delta = torch.tensor([[1.5, 0.0]], dtype=torch.float32)
    post = torch.tensor(
        [[math.cos(1.5), 0.3]],
        dtype=torch.float32,
    )
    remote = torch.tensor([[True, False]])

    corrected, stats = target.strict_phase_rescue_scores(
        query,
        keys,
        delta,
        post,
        remote,
        inv_freq=torch.tensor([1.0]),
        attention_scale=1.0,
        score_scale=1.0,
        boundary=0.0,
        lift_fraction=0.25,
        minimum_counterfactual_gap=0.0,
        frequency_budget=1,
        max_phase_shift=0.25,
    )

    assert bool(stats["trigger"][0, 0])
    assert bool(stats["solver_feasible"][0, 0])
    assert int(stats["active_plane_count"][0, 0]) <= 1
    assert float(stats["shift_abs_max"][0, 0]) <= 0.25 + 1e-7
    assert float(stats["exact_lift"][0, 0]) >= float(
        stats["desired_lift"][0, 0]
    ) - 2e-7
    assert corrected[0, 1].view(torch.int32).item() == post[0, 1].view(
        torch.int32
    ).item()
    assert float(stats["nontrigger_noop_max"]) == 0.0


def test_untriggered_strict_pairs_are_bitwise_noops() -> None:
    torch.manual_seed(23)
    query = torch.randn(1, 2, 1, 8)
    keys = torch.randn(1, 2, 4, 8)
    post = torch.randn(2, 4, dtype=torch.bfloat16)
    remote = torch.zeros(2, 4, dtype=torch.bool)

    corrected, stats = target.strict_phase_rescue_scores(
        query,
        keys,
        torch.zeros(2, 4, dtype=torch.long),
        post,
        remote,
        inv_freq=torch.tensor([1.0, 0.2, 0.04, 0.008]),
        attention_scale=1.0,
        score_scale=1.0,
        boundary=128.0,
        lift_fraction=0.25,
        minimum_counterfactual_gap=1.0,
        frequency_budget=2,
        max_phase_shift=0.25,
    )

    assert torch.equal(corrected, post)
    assert not bool(stats["trigger"].any())
    assert float(stats["nontrigger_noop_max"]) == 0.0


def test_random_frequency_ablation_is_deterministic_and_strict() -> None:
    torch.manual_seed(29)
    query = torch.randn(1, 1, 1, 8)
    keys = torch.randn(1, 1, 2, 8)
    delta = torch.tensor([[41, 43]])
    post = torch.full((1, 2), -5.0)
    remote = torch.ones(1, 2, dtype=torch.bool)
    base_arguments = dict(
        query_pre=query,
        selected_key_pre=keys,
        selected_delta=delta,
        post_selected=post,
        remote_mask=remote,
        inv_freq=torch.tensor([1.0, 0.4, 0.15, 0.03]),
        attention_scale=1.0,
        score_scale=1.0,
        boundary=0.0,
        lift_fraction=0.10,
        minimum_counterfactual_gap=-1e9,
        frequency_budget=2,
        max_phase_shift=0.20,
    )
    _, normal_stats = target.strict_phase_rescue_scores(**base_arguments)
    arguments = {
        **base_arguments,
        "query_pre": query * 2.5,
        "random_frequency_support": True,
        "random_seed_base": 1234,
        "frozen_raw_trigger": normal_stats["raw_trigger"],
        "frozen_trigger": normal_stats["trigger"],
        "frozen_desired_lift": normal_stats["desired_lift"],
        "frozen_counterfactual_gap": normal_stats["counterfactual_gap"],
        "matched_random_norm": normal_stats["shift_norm"],
        "matched_random_support": normal_stats["active_plane_count"],
        "matched_random_lift": normal_stats["solver_lift"],
    }

    first, first_stats = target.strict_phase_rescue_scores(**arguments)
    second, second_stats = target.strict_phase_rescue_scores(**arguments)

    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        first_stats["active_plane_count"],
        second_stats["active_plane_count"],
        atol=0.0,
        rtol=0.0,
    )
    assert int(first_stats["active_plane_count"].max()) <= 2
    assert float(first_stats["shift_abs_max"].max()) <= 0.20 + 1e-7
    trigger = first_stats["trigger"]
    torch.testing.assert_close(
        first_stats["shift_norm"].masked_select(trigger),
        first_stats["matched_reference_norm"].masked_select(trigger),
        atol=1e-7,
        rtol=1e-7,
    )
    torch.testing.assert_close(
        first_stats["active_plane_count"].masked_select(trigger),
        first_stats["matched_reference_support"].masked_select(trigger),
    )
    assert float(first_stats["norm_match_error"].max()) <= 1e-7
    assert int(first_stats["support_match_error"].max()) == 0
    torch.testing.assert_close(
        first_stats["matched_reference_lift"], normal_stats["solver_lift"]
    )


def test_frozen_trigger_and_target_override_current_query_decisions() -> None:
    query = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    keys = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)
    delta = torch.tensor([[1.5, 1.5]], dtype=torch.float32)
    post = torch.tensor([[math.cos(1.5), math.cos(1.5)]], dtype=torch.float32)
    remote = torch.ones(1, 2, dtype=torch.bool)
    frozen_trigger = torch.tensor([[False, True]])
    frozen_target = torch.tensor([[0.0, 0.05]])
    frozen_gap = torch.tensor([[0.0, 0.20]])

    corrected, stats = target.strict_phase_rescue_scores(
        query,
        keys,
        delta,
        post,
        remote,
        inv_freq=torch.tensor([1.0]),
        attention_scale=1.0,
        score_scale=1.0,
        boundary=0.0,
        lift_fraction=1.0,
        minimum_counterfactual_gap=-1e9,
        frequency_budget=1,
        max_phase_shift=0.25,
        frozen_raw_trigger=torch.ones_like(frozen_trigger),
        frozen_trigger=frozen_trigger,
        frozen_desired_lift=frozen_target,
        frozen_counterfactual_gap=frozen_gap,
    )

    torch.testing.assert_close(stats["trigger"], frozen_trigger)
    torch.testing.assert_close(stats["desired_lift"], frozen_target)
    torch.testing.assert_close(stats["counterfactual_gap"], frozen_gap)
    assert corrected[0, 0].view(torch.int32).item() == post[0, 0].view(
        torch.int32
    ).item()
    assert corrected[0, 1] > post[0, 1]


def test_raw_but_token_capped_out_pairs_are_bitwise_noops() -> None:
    query = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    keys = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32
    )
    delta = torch.full((1, 3), 1.5)
    post = torch.full((1, 3), math.cos(1.5), dtype=torch.bfloat16)
    raw = torch.ones(1, 3, dtype=torch.bool)
    capped = torch.tensor([[False, True, False]])

    corrected, stats = target.strict_phase_rescue_scores(
        query,
        keys,
        delta,
        post,
        raw,
        inv_freq=torch.tensor([1.0]),
        attention_scale=1.0,
        score_scale=1.0,
        boundary=0.0,
        lift_fraction=0.25,
        minimum_counterfactual_gap=1.0,
        frequency_budget=1,
        max_phase_shift=0.25,
        frozen_raw_trigger=raw,
        frozen_trigger=capped,
        frozen_desired_lift=torch.full((1, 3), 0.1),
        frozen_counterfactual_gap=torch.full((1, 3), 2.0),
    )

    assert torch.equal(corrected[raw & ~capped], post[raw & ~capped])
    assert float(stats["nontrigger_noop_max"]) == 0.0


def test_token_cap_metrics_record_raw_capped_and_exact_noop() -> None:
    controller = target.runner.Controller(
        variant="strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25",
        ratio=0.02,
        minimum_keep_tokens=0,
        maximum_keep_tokens=0,
        local_window=128,
        sink_tokens=16,
        evidence_spans=(),
    )
    raw = torch.ones(2, 4, dtype=torch.bool)
    trigger = torch.tensor(
        [[False, True, False, False], [False, False, False, True]]
    )
    applied = torch.zeros(2, 4)
    applied[trigger] = torch.tensor([0.2, 0.3])
    stats = {
        "raw_trigger": raw,
        "trigger": trigger,
        "remote_mask": raw,
        "solver_feasible": trigger.clone(),
        "desired_lift": torch.full((2, 4), 0.25),
        "solver_lift": applied.clone(),
        "active_plane_count": trigger.long() * 2,
        "shift_abs_max": trigger.float() * 0.1,
        "shift_norm": trigger.float() * 0.14,
        "matched_reference_norm": trigger.float() * 0.14,
        "matched_reference_lift": applied.clone(),
        "norm_match_error": torch.zeros(2, 4),
        "support_match_error": torch.zeros(2, 4, dtype=torch.long),
        "nontrigger_noop_max": torch.tensor(0.0),
        "frozen_token_cap_mismatch_count": torch.tensor(0),
    }

    target.record_strict_phase_metrics(
        controller,
        stats,
        applied,
        frequency_budget=8,
        phase_cap=0.25,
        token_cap=1,
        random_frequency_support=False,
        partition_preserve=False,
    )
    summary = controller.metrics.summary()

    assert summary["strict_phase_remote_eligible_count"] == 8
    assert summary["strict_phase_raw_trigger_count"] == 8
    assert summary["strict_phase_capped_trigger_count"] == 2
    assert summary["strict_phase_raw_trigger_fraction"] == 1.0
    assert summary["strict_phase_capped_trigger_fraction"] == 0.25
    assert summary["strict_phase_max_triggers_per_head"] == 1
    assert summary["strict_phase_token_cap"] == 1
    assert summary["strict_phase_token_cap_noop_count"] == 6
    assert summary["strict_phase_token_cap_noop_max"] == 0.0
    assert summary["strict_phase_frozen_token_cap_mismatch_max"] == 0


def test_mass_preserve_can_leave_every_nontrigger_pair_unchanged() -> None:
    native = torch.tensor([[0.2, -0.1, 0.7, -0.4]])
    corrected = native.clone()
    corrected[0, 0] += 0.8
    corrected[0, 2] += 0.3
    trigger = torch.tensor([[True, False, True, False]])

    preserved = target.preserve_remote_partition(corrected, native, trigger)

    assert torch.equal(preserved[~trigger], native[~trigger])
    native_partition = torch.logsumexp(native.masked_fill(~trigger, -torch.inf), -1)
    new_partition = torch.logsumexp(
        preserved.masked_fill(~trigger, -torch.inf), -1
    )
    torch.testing.assert_close(new_partition, native_partition, atol=1e-6, rtol=1e-6)
    assert float(target.sparse_log_partition_error(preserved, native).max()) <= 1e-6


@pytest.mark.parametrize("trigger_count", [0, 1, 3])
def test_mass_preserve_partition_edges(trigger_count: int) -> None:
    native = torch.tensor([[0.2, -0.1, 0.7, -0.4]], dtype=torch.bfloat16)
    trigger = torch.zeros_like(native, dtype=torch.bool)
    trigger[:, :trigger_count] = True
    corrected = native.clone()
    corrected[trigger] += torch.linspace(
        0.2, 0.8, steps=max(1, trigger_count), dtype=torch.bfloat16
    )[:trigger_count]

    preserved = target.preserve_remote_partition(corrected, native, trigger)

    assert torch.equal(preserved[~trigger], native[~trigger])
    assert torch.isfinite(preserved).all()
    if trigger_count == 1:
        assert torch.equal(preserved, native)
    assert float(target.sparse_log_partition_error(preserved, native).max()) < 0.01


def test_strict_variant_config_is_explicit_and_auditable() -> None:
    variant = "strict_mpr_pre_w128_lift25_gap1_t4_f8_cap0p25_random_masspreserve"
    assert target._rescue_boundary(variant) == 128.0
    assert target._rescue_fraction(variant) == 0.25
    assert target._rescue_gap_threshold(variant) == 1.0
    assert target._rescue_frequency_budget(variant, 64) == 8
    assert target._strict_phase_cap(variant) == 0.25
    assert target._strict_token_cap(variant) == 4
    assert target._strict_random_support(variant)


def test_strict_diagnostics_are_retained_in_aggregate_summary() -> None:
    variant = "strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25"

    def row(seed: int, calls: int, feasible: float, partition_count: int) -> dict:
        metrics = target.runner.MetricAccumulator().summary()
        metrics.update(
            {
                "strict_phase_solver_calls": calls,
                "strict_phase_remote_eligible_count": 20,
                "strict_phase_raw_trigger_count": 10,
                "strict_phase_capped_trigger_count": calls,
                "strict_phase_raw_trigger_fraction": 0.5,
                "strict_phase_capped_trigger_fraction": calls / 20.0,
                "strict_phase_max_triggers_per_head": 1,
                "strict_phase_token_cap": 1,
                "strict_phase_token_cap_noop_count": 10 - calls,
                "strict_phase_token_cap_noop_max": 0.0,
                "strict_phase_frozen_token_cap_mismatch_max": 0,
                "strict_phase_feasible_fraction": feasible,
                "strict_phase_target_lift_mean": 0.2 + seed,
                "strict_phase_solver_lift_mean": 0.1 + seed,
                "strict_phase_applied_lift_mean": 0.05 + seed,
                "strict_phase_support_mean": 2.0 + seed,
                "strict_phase_budget_mean": 8.0,
                "strict_phase_cap_mean": 0.25,
                "strict_phase_shift_l2_mean": 0.3 + seed,
                "strict_phase_random_reference_l2_mean": 0.3 + seed,
                "strict_phase_random_reference_lift_mean": 0.1 + seed,
                "strict_phase_support_max": 3 + seed,
                "strict_phase_shift_abs_max": 0.2,
                "strict_phase_random_norm_match_max": 1e-9 * seed,
                "strict_phase_random_support_delta_max": 2,
                "strict_phase_nontrigger_noop_max": 0.0,
                "strict_phase_partition_error_max": 0.002 * seed,
                "strict_phase_frozen_support_mismatch_max": 0,
                "strict_phase_partition_preserve": 1.0,
                "strict_phase_random_support": 0.0,
                "strict_phase_exact_pre_selector": 1.0,
                "strict_phase_frozen_reference": 1.0,
                "strict_phase_partition_head_count": partition_count,
                "strict_phase_partition_error_mean": 0.001 * seed,
            }
        )
        return {
            "target_context_tokens": 8192,
            "variant": variant,
            "seed": seed,
            "gold_nll": 1.0,
            "gold_evidence_token_recall": 0.5,
            "gold_evidence_line_hit_rate": 0.5,
            "gold_chain_complete_rate": 0.5,
            "gold_evidence_attention_mass": 0.1,
            "next_token_correct": 1,
            "query_seconds": 0.2,
            **metrics,
        }

    summary = target._phase_summarize(
        [row(seed=1, calls=2, feasible=0.5, partition_count=4),
         row(seed=2, calls=6, feasible=1.0, partition_count=12)]
    )[0]

    assert summary["strict_phase_solver_calls"] == 8
    assert summary["strict_phase_remote_eligible_count"] == 40
    assert summary["strict_phase_raw_trigger_count"] == 20
    assert summary["strict_phase_capped_trigger_count"] == 8
    assert summary["strict_phase_raw_trigger_fraction"] == pytest.approx(0.5)
    assert summary["strict_phase_capped_trigger_fraction"] == pytest.approx(0.2)
    assert summary["strict_phase_token_cap_noop_count"] == 12
    assert summary["strict_phase_max_triggers_per_head"] == 1
    assert summary["strict_phase_token_cap"] == 1
    assert summary["strict_phase_feasible_fraction"] == pytest.approx(0.875)
    assert summary["strict_phase_support_max"] == 5
    assert summary["strict_phase_partition_head_count"] == 16
    assert summary["strict_phase_partition_error_mean"] == pytest.approx(0.00175)
    assert summary["strict_phase_frozen_reference"] == 1.0
