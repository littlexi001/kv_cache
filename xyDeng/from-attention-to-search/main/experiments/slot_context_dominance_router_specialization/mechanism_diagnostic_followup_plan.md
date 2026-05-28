# Mechanism Diagnostic Follow-Up Plan: slot_context_dominance_router_specialization

Anchor:

```text
../../problem_anchors/gated_main_causes/slot_context_dominance_router_specialization_anchor.md
```

## 1. Run Goal

结论先行：已有 full run 已能支持当前 anchor 更新，但报告还缺机制过程图。这个 follow-up 不改变实验问题；它补齐 routing 从初始化到最终状态如何演化的证据链。

核心问题：

```text
slot 信息在 h_B 中一直可见吗？
router weight 是否持续追踪 slot-centroid direction？
route-slot NMI 是从 step-0 就低，还是训练中从高变低？
expert utility 对齐是在什么时候出现或断开？
```

## 2. Two 4-GPU Jobs

提交两个 4 卡 full 诊断任务，每个任务保留 4 seeds 和 4 conditions：

| Job | Mode | Run name | Purpose |
|---|---|---|---|
| J1 | `single_b_sanity` | `slot_context_single_b_trajectory_full_20260527` | fixed-B 下追踪 slot center、router weight、route NMI 是否保持对角 |
| J2 | `multi_b_primary` | `slot_context_multi_b_trajectory_full_20260527` | multi-B 下追踪 slot signal 可见但 routing/utility 脱钩的过程 |

Conditions:

```text
C0_short_slot_init
A_long_repeated_slot_init
B_long_distributed_slot_init
Oracle_slot_router
```

Seeds:

```text
20260521,20260522,20260523,20260524
```

Training:

```text
full-sequence causal NTP CE
steps = 1600
seq_len = 32
checkpoint/eval interval = every 100 steps plus step 0
```

## 3. Five Gaps To Fill

### Gap 1: Initial-vs-final routing NMI

Required table:

```text
trajectory_route_metrics_by_step.csv
```

Required columns:

```text
mode, condition, seed, step,
nmi_route_slot, best_perm_diag_purity,
nmi_route_b_id, assignment_utility_agreement
```

Required figure:

```text
figures/<run_name>/route_nmi_initial_final_bar.png
```

Interpretation target:

```text
If step-0 route-slot NMI is high but final NMI is low, the issue is training drift.
If step-0 route-slot NMI is already low, the issue is initialization or hidden-state geometry.
```

### Gap 2: Routing NMI trajectory

Required figure:

```text
figures/<run_name>/route_slot_nmi_trajectory.png
figures/<run_name>/diag_purity_trajectory.png
figures/<run_name>/assignment_utility_trajectory.png
```

The report must show these near the claim:

```text
learned routing does or does not keep slot assignment over training.
```

### Gap 3: Per-condition route heatmaps and utility heatmaps

Required figures:

```text
figures/<run_name>/route_slot_heatmap_step0_final.png
figures/<run_name>/forced_expert_loss_heatmap_final.png
figures/<run_name>/ablation_delta_heatmap_final.png
```

Report use:

```text
route_slot_heatmap shows assignment;
forced_expert_loss and ablation_delta show causal utility;
the two must be interpreted together.
```

### Gap 4: Oracle stricter control

Add two Oracle variants if implementation cost is small:

| Variant | Meaning |
|---|---|
| `Oracle_all_positions` | current forced `slot -> expert` for all positions |
| `Oracle_b_position_only` | force route only at B position; all other positions use learned router |

Decision role:

```text
If Oracle_all_positions works but Oracle_b_position_only weakens, expert utility depends on non-B positions too.
If both work, B-position routing is sufficient for the utility upper bound.
```

### Gap 5: Random-init control

Add random-init controls only after trajectory instrumentation is verified:

```text
A_long_random_init
B_long_random_init
```

Decision role:

```text
Compare slot-centroid init vs random init on step-0 NMI, final NMI, and assignment-utility.
This quantifies whether slot-centroid init helps but drifts, or never helps in multi-B.
```

## 4. Slot-Center And Router-Weight Movement

Yes, we can directly track routing/slot differentiation by comparing slot centers and router weights over training.

At every checkpoint step, collect B-position router input:

$$
h_{s,i}^{(t)} = h^{(t)}(B_i \mid slot=s)
$$

Compute slot centers:

$$
\mu_s^{(t)} = \mathbb{E}_i[h_{s,i}^{(t)}], \qquad \bar{\mu}^{(t)} = \mathbb{E}_s[\mu_s^{(t)}]
$$

Compute slot direction:

$$
d_s^{(t)}=\operatorname{Normalize}(\mu_s^{(t)}-\bar{\mu}^{(t)})
$$

Track router row:

$$
w_e^{(t)}
$$

Key alignment matrix:

$$
M_{s,e}^{(t)}=\cos(d_s^{(t)}, w_e^{(t)})
$$

Required outputs:

```text
slot_center_norm_by_step.csv
slot_center_cosine_by_step.csv
router_weight_cosine_to_slot_center_by_step.csv
router_weight_delta_from_init_by_step.csv
router_weight_pairwise_cosine_by_step.csv
```

Required figures:

```text
slot_center_separation_trajectory.png
router_weight_to_slot_center_cosine_trajectory.png
router_weight_delta_from_init_trajectory.png
slot_center_pca_step0_final.png
router_weight_pca_step0_final.png
```

Interpretation:

```text
If slot centers separate but router rows stop aligning with them, the failure is router tracking.
If router rows align with slot centers but route NMI is low, another hidden component or bias dominates logits.
If centers collapse, the representation stopped carrying slot at B position.
If centers stay separated and router rows stay aligned but assignment-utility is low, the failure is expert utility binding.
```

## 5. Implementation Plan

Extend:

```text
scripts/run_slot_context_dominance_router_specialization.py
```

Add mode:

```text
--trajectory
```

Add config fields:

```json
{
  "trajectory": {
    "eval_every": 100,
    "include_step0": true,
    "save_checkpoint_every": 400,
    "track_slot_centers": true,
    "track_router_weight_alignment": true,
    "oracle_b_position_only": true,
    "random_init_controls": true
  }
}
```

ACP submission shape:

```bash
SLOT_CONTEXT_ALLOW_REAL_SUBMIT=1 \
RUN_NAME=slot_context_single_b_trajectory_full_20260527 \
JOB_NAME=ats-slot-context-single-b-trajectory-0527 \
MODES=single_b_sanity \
RUN_STAGE=full \
EXTRA_ARGS=--trajectory \
bash scripts/submit_slot_context_dominance_4gpu_acp.sh
```

```bash
SLOT_CONTEXT_ALLOW_REAL_SUBMIT=1 \
RUN_NAME=slot_context_multi_b_trajectory_full_20260527 \
JOB_NAME=ats-slot-context-multi-b-trajectory-0527 \
MODES=multi_b_primary \
RUN_STAGE=full \
EXTRA_ARGS=--trajectory \
bash scripts/submit_slot_context_dominance_4gpu_acp.sh
```

## 6. Required Report Update

After both jobs finish, update these files:

```text
Projects/from-attention-to-search/main/experiments/slot_context_dominance_router_specialization/summary.md
Projects/from-attention-to-search/main/experiments/slot_context_dominance_router_specialization/detailed.md
Projects/from-attention-to-search/XingyuD/sync/0526_slot_context_dominance/report.md
Projects/from-attention-to-search/main/problem_anchors/gated_main_causes/slot_context_dominance_router_specialization_anchor.md
sync/S000_current_specialization/anchors/expert_specialization/slot_context_dominance_router_specialization_anchor.md
```

Figures that must be embedded, not merely linked:

```text
route_nmi_initial_final_bar.png
route_slot_nmi_trajectory.png
assignment_utility_trajectory.png
route_slot_heatmap_step0_final.png
forced_expert_loss_heatmap_final.png
ablation_delta_heatmap_final.png
router_weight_to_slot_center_cosine_trajectory.png
slot_center_pca_step0_final.png
```

Anchor update rule:

```text
Only update anchor interpretation if trajectory evidence changes the mechanism diagnosis.
Otherwise add it as support for the existing claim boundary.
```

## 7. Pass / Fail Criteria

The follow-up is complete only if:

```text
both 4-GPU jobs finish successfully;
trajectory CSVs exist for both single-B and multi-B;
all required figures are generated;
summary/detailed/sync include the figures near the claims they support;
anchor and root sync copy are updated and identical.
```

