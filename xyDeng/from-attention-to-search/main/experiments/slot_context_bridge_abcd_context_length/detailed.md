# Detailed Result: slot_context_bridge_abcd_context_length

Anchor:

```text
../../problem_anchors/gated_main_causes/slot_context_dominance_router_specialization_anchor.md
```

Story:

```text
This bridge experiment aligns the old r-B / AB / CB / DB setting with the current full-NTP slot-context protocol, to decide whether short versus long role context changes B-position route-role alignment.
```

## 0. Quick Recap

目的：补一个和 0524 可对比的 bridge，判断旧 AB/CB semantic-prior decay 是否和当前 fixed-B NMI 上升矛盾。

假设：如果旧 decay 主要来自 context signal 或 target-position pressure 不够对齐，那么在当前 full-NTP + B-position metrics 协议下，long role context 应该比 short role cue 更能保持 route-role NMI。

实验思路：构造 `r-B / AB / CB / DB -> Y_{role,i}`，分别跑 fixed-B 和 multi-B，比较 random init 与 B-position semantic centroid init，在所有条件上使用 normal full-sequence causal NTP。

结论：bridge 支持 context strength helps routing，但不支持 robust multi-B natural specialization。fixed-B long semantic 是稳定正例；multi-B long semantic 只部分改善。

证据：fixed-B `long_role_semantic_init` final NMI `1.000` across four seeds；multi-B semantic final NMI 高于对应 random：short `0.338 > 0.047`，long `0.593 > 0.373`，distributed `0.297 > 0.010`。

## 1. Anchor Link And Decision Point

Anchor:

```text
Projects/from-attention-to-search/main/problem_anchors/gated_main_causes/slot_context_dominance_router_specialization_anchor.md
```

This experiment updates:

```text
Current Evidence
Claim Boundary And Next Decision
```

It is not a new anchor. It is a bridge/control experiment under the same slot-context dominance question.

## 2. Setup

Data motif:

```text
positions 10-14: role/slot prefix
position 15: B_i
position 16: Y_{role,i}
seq_len = 32
```

Conditions:

| Condition | Prefix meaning |
|---|---|
| `short_role_*` | one role cue plus background tokens |
| `long_role_*` | repeated role cue, length 5 |
| `distributed_role_*` | distributed 5-token role code |

Modes:

| Mode | Meaning |
|---|---|
| `single_b_sanity` | one shared B identity, pure role/context control |
| `multi_b_primary` | 256 B identities, identity variation present |

Initialization:

| Init | Meaning |
|---|---|
| `random_init` | normal router initialization |
| `semantic_init` | router rows initialized from B-position role centroids |

Model/training:

```text
model: top1_selected_gate
experts: 4
loss: normal full-sequence causal NTP, mean CE over positions
primary metrics: B-position route-role NMI and seed-level route heatmaps
seeds: 20260521, 20260522, 20260523, 20260524
```

Implementation check: `Top1SelectedGateFFN` computes `router_probs = softmax(router(h))`, selects `argmax`, and writes `selected_prob * expert(h)` on the selected expert path. This keeps the selected gate probability in the forward path, so the router weight matrix receives gradient through the selected probability.

## 3. Jobs

| Run | Mode | Job id | Status |
|---|---|---|---|
| `bridge_abcd_single_b_full_20260527` | `single_b_sanity` | `pt-qeorv2p9` | completed |
| `bridge_abcd_multi_b_full_20260527` | `multi_b_primary` | `pt-gr1vtgfn` | completed |

Submission shape:

```bash
bash scripts/submit_slot_context_dominance_4gpu_acp.sh
```

with conditions:

```text
short_role_random_init,short_role_semantic_init,
long_role_random_init,long_role_semantic_init,
distributed_role_random_init,distributed_role_semantic_init
```

## 4. Main Result

| Mode | Condition | Init | Final NMI | NMI range | Final Assign-Utility | Target Acc |
|---|---|---|---:|---:|---:|---:|
| fixed-B | short role | random | 0.000 | 0.000-0.000 | 1.000 | 1.000 |
| fixed-B | short role | semantic | 0.467 | 0.000-1.000 | 1.000 | 1.000 |
| fixed-B | long role | random | 0.159 | 0.000-0.637 | 1.000 | 1.000 |
| fixed-B | long role | semantic | 1.000 | 1.000-1.000 | 1.000 | 1.000 |
| fixed-B | distributed role | random | 0.000 | 0.000-0.000 | 1.000 | 1.000 |
| fixed-B | distributed role | semantic | 0.893 | 0.707-1.000 | 1.000 | 1.000 |
| multi-B | short role | random | 0.047 | 0.000-0.187 | 0.934 | 1.000 |
| multi-B | short role | semantic | 0.338 | 0.199-0.419 | 0.759 | 1.000 |
| multi-B | long role | random | 0.373 | 0.016-0.648 | 0.912 | 1.000 |
| multi-B | long role | semantic | 0.593 | 0.221-0.825 | 0.921 | 1.000 |
| multi-B | distributed role | random | 0.010 | 0.000-0.021 | 0.998 | 1.000 |
| multi-B | distributed role | semantic | 0.297 | 0.221-0.452 | 0.733 | 1.000 |

## 5. Key Figures

![bridge final NMI by seed](figures/bridge_abcd_final_nmi_by_seed_heatmap.png)

Supports: no seed averaging is needed for the main comparison. fixed-B long semantic is stable; multi-B long semantic improves but remains seed-variable.

![bridge short vs long init-final route heatmaps](figures/bridge_abcd_short_long_init_final_route_heatmaps.png)

Supports: long context plus semantic initialization keeps the most interpretable route-role partition. Random init does not reliably discover that partition by itself.

## 6. Evidence Walkthrough

Observation 1: all target accuracies are `1.000`. The model learns the prediction task in all conditions, so routing differences are not explained by failed training.

Observation 2: fixed-B is decisive as a positive control. `long_role_semantic_init` is perfect across seeds, while `short_role_semantic_init` decays in two seeds and `distributed_role_semantic_init` remains high but not perfect.

Observation 3: multi-B preserves the same direction but weakens the claim. Semantic init improves NMI for short, long, and distributed contexts, yet final routing is not uniformly diagonal.

Observation 4: Assign-Utility can disagree with NMI. The clearest case is multi-B `distributed_role_random_init`, where Assign-Utility is `0.998` but route NMI is `0.010`. This is a utility-collapse warning, not evidence for slot-specialized routing.

## 7. Comparison To 0524

The 0524 line asked whether semantic/context initialization is sufficient for stable functional expert specialization in AB/CB synthetic data. It found early semantic NMI improvement followed by decay, and showed that final hidden states can still contain recoverable semantic boundaries.

The bridge shows why this is not a contradiction:

```text
0524 negative result: semantic separability or semantic init alone does not imply stable functional specialization.
0527 bridge positive result: when the target requires role disambiguation at B position, stronger context can improve route-role alignment.
```

These are compatible because the bridge tests a weaker, more local statement. It does not overturn the 0524 conclusion about route-function mismatch.

## 8. Artifact Map

Code workspace:

```text
Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding
```

Runner:

```text
scripts/run_slot_context_dominance_router_specialization.py
```

Config:

```text
configs/slot_context_dominance_router_specialization.json
```

Result dirs:

```text
results/slot_context_dominance_router_specialization/bridge_abcd_single_b_full_20260527
results/slot_context_dominance_router_specialization/bridge_abcd_multi_b_full_20260527
```

Curated tables:

```text
Projects/from-attention-to-search/main/experiments/slot_context_bridge_abcd_context_length/tables/bridge_abcd_initial_final_by_seed.csv
Projects/from-attention-to-search/main/experiments/slot_context_bridge_abcd_context_length/tables/bridge_abcd_single_b_summary_by_condition.csv
Projects/from-attention-to-search/main/experiments/slot_context_bridge_abcd_context_length/tables/bridge_abcd_multi_b_summary_by_condition.csv
```

Curated figures:

```text
Projects/from-attention-to-search/main/experiments/slot_context_bridge_abcd_context_length/figures/bridge_abcd_final_nmi_by_seed_heatmap.png
Projects/from-attention-to-search/main/experiments/slot_context_bridge_abcd_context_length/figures/bridge_abcd_short_long_init_final_route_heatmaps.png
```

Job ids:

```text
single_b_sanity: pt-qeorv2p9
multi_b_primary: pt-gr1vtgfn
```

## 9. Claim Boundary

Safest claim:

```text
Stronger role/slot context helps B-position routing align with role/slot, and fixed-B long semantic is a stable positive control.
```

What cannot be claimed:

```text
Ordinary top-1 NTP naturally learns stable slot-functional experts under multi-B identity variation.
```

Next smallest decision:

```text
Move from more context-length sweeps to explicit route-function binding in multi-B.
```
