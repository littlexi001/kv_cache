---
anchor: slot_context_dominance_router_specialization
parent_anchor: gating_induced_expert_specialization
status: pre-run experiment plan
---

# Slot-Dominant Context Routing Diagnostic - Experiment Plan

## 1. Decision Question

When the same `B_i` token appears under different slot contexts and requires different next-token targets, does strengthening the slot context signal plus slot-centroid router initialization make a standard top-1 MoE route `B_i` positions by slot and form slot-specific expert utility?

This experiment tests whether the previous failure was mainly caused by weak context signal in the `B` hidden state, rather than by an intrinsic inability of top-1 gating to use slot features.

## 2. Hypothesis

Primary hypothesis:

```text
If slot information is strong in the B-position hidden state and slot determines the NTP target,
then top-1 gating with slot-centroid initialization can maintain slot-aligned routing
and produce slot-specific expert utility.
```

Rival explanation:

```text
Even when slot is visible and predictive, NTP-trained top-1 gating may still not turn slot into a stable expert specialization axis.
```

## 3. Data Construction

Use full slots:

```text
number of slots S = 4
number of experts E = 4
B token pool = {B_1, ..., B_N}
```

For every shared token `B_i`, all 4 slots appear:

```text
slot 0 + B_i -> Y_{0,i}
slot 1 + B_i -> Y_{1,i}
slot 2 + B_i -> Y_{2,i}
slot 3 + B_i -> Y_{3,i}
```

Target rule:

$$
Y_{s,i}=\pi_s(B_i)
$$

Require:

$$
Y_{0,i},Y_{1,i},Y_{2,i},Y_{3,i}\ \text{are distinct for the same}\ B_i.
$$

So `B_i` alone cannot solve the task. The slot must disambiguate the target.

Training should use standard full-sequence causal NTP CE:

```text
loss = mean CE(x_t -> x_{t+1}) over normal next-token positions
```

The anchor decision, however, is judged at the target position:

```text
[prefix slot context, B_i] -> predict Y_{s,i}
```

Therefore report both:

```text
full-sequence NTP CE: optimization sanity check
target-position CE / accuracy: primary task-validity metric for this anchor
```

Route assignment, hidden-state slot probes, forced expert loss, and ablation utility should all be measured at the `B_i` position, because that is where slot context and B-token identity compete.

## 4. Two Long-Slot Context Schemes

### Scheme A: Repeated Cue Slot Prefix

Purpose: strongest and simplest test of whether longer context can amplify slot signal.

```text
slot 0: A A A A A  B_i -> Y_{0,i}
slot 1: C C C C C  B_i -> Y_{1,i}
slot 2: G G G G G  B_i -> Y_{2,i}
slot 3: K K K K K  B_i -> Y_{3,i}
```

This is the easiest positive-control version. If this fails, the failure is unlikely to be only "slot prefix too short".

### Scheme B: Distributed Compositional Slot Code

Purpose: avoid the trivial claim that the router only learned one repeated prefix token.

Use 5 context positions. Each individual cue token appears in two slots; only the full combination identifies the slot.

```text
slot 0: p0 q0 r0 t0 u0  B_i -> Y_{0,i}
slot 1: p0 q1 r1 t1 u1  B_i -> Y_{1,i}
slot 2: p1 q0 r1 t0 u1  B_i -> Y_{2,i}
slot 3: p1 q1 r0 t1 u0  B_i -> Y_{3,i}
```

This forces the model to aggregate multi-token context. A single prefix token is not enough to identify the slot.

## 5. Model and Router Initialization

Model:

```text
top-1 selected-gate sparse MoE
experts = 4
slots = 4
load balance = off for first diagnostic
same model budget as previous AB/CB run if possible
```

Router:

$$
p=\operatorname{softmax}(Wh), \qquad e=\arg\max_j p_j,
$$

$$
\operatorname{MoE}(h)=p_e E_e(h).
$$

Slot-centroid router initialization:

1. Run a calibration pass before training.
2. Collect `B_i` hidden states grouped by slot:

$$
h_{s,i}=h(B_i\mid \text{slot }s)
$$

3. Compute slot centers:

$$
\mu_s=\mathbb{E}_i[h_{s,i}], \qquad \bar\mu=\mathbb{E}_s[\mu_s]
$$

4. Initialize router row for expert `s`:

$$
w_s \leftarrow \alpha\cdot \operatorname{Normalize}\left((\Sigma+\lambda I)^{-1}(\mu_s-\bar\mu)\right)
$$

This gives the router an initial slot prior but does not freeze routing.

Optional diagnostic upper bound:

```text
oracle-slot router: force slot s -> expert s
```

Use it only to test whether slot-specific expert utility can exist in this task.

## 6. Conditions

Minimal required conditions:

| Condition | Data | Router init | Purpose |
|---|---|---|---|
| C0_short | 1-token slot cue | slot-centroid init | test whether short context is too weak |
| A_long | Scheme A, 5-token repeated cue | slot-centroid init | strongest positive-control test |
| B_long | Scheme B, 5-token distributed code | slot-centroid init | test non-trivial multi-token slot context |
| Oracle | Scheme A or B | forced slot router | utility upper bound |

Optional if time:

| Condition | Purpose |
|---|---|
| A_long_random_init | whether long context alone induces slot routing |
| B_long_random_init | whether compositional context alone induces slot routing |

Use at least 4 seeds if runtime allows. If time is limited before meeting, run 1 seed smoke + 4 seed full job submitted.

## 7. Metrics

### Primary Metric: Assignment-Utility Agreement

For each slot `s`, compute the utility-best expert:

$$
e_s^*=\arg\max_e \Delta CE(s,e)
$$

where:

$$
\Delta CE(s,e)=CE_s(\operatorname{ablate}(e))-CE_s(\operatorname{full}).
$$

Then measure:

$$
A_{assign-utility}=P(e_{route}=e_s^*\mid slot=s).
$$

This is primary because specialization requires routing assignment and expert causal utility to agree.

### Supporting Metrics

Routing-slot alignment:

$$
R_{s,e}=P(route=e\mid slot=s, \text{position}=B)
$$

Report best-permutation diagonal purity and NMI:

```text
NMI(route, slot)
NMI(route, B token id)
```

Slot information in `B` hidden state:

```text
linear probe: h_B -> slot
slot centroid separation ||mu_s - mean(mu)||
```

Forced expert loss matrix:

$$
L^{forced}_{s,e}=CE_s(\text{force route to expert }e)
$$

Supported pattern:

$$
L^{forced}_{s,s}<L^{forced}_{s,e}, \quad e\neq s
$$

after best expert-slot permutation.

Task validity:

```text
target-position CE
target-position accuracy
dense baseline accuracy
```

## 8. Decision Rules

Supported if:

```text
long-slot conditions produce high h_B slot probe accuracy;
route-slot matrix is close to diagonal after permutation;
assignment-utility agreement is high;
forced loss and ablation utility show diagonal slot-specific expert advantage;
results are stable across seeds;
short-prefix condition is weaker than long-prefix conditions.
```

Weakened if:

```text
slot is visible in h_B but routing does not align with slot;
routing aligns with slot but expert utility is not slot-specific;
long repeated cue works but distributed code fails;
only one seed shows the pattern.
```

Redirect if:

```text
slot is not decodable from h_B even in long-prefix data -> representation problem;
oracle-slot router has no slot-specific utility -> expert utility gap problem;
oracle works but learned router fails -> top-1 routing dynamics / objective mismatch problem.
```

## 9. Minimal Report Table

| Condition | Target Acc | Probe slot from h_B | NMI(route, slot) | NMI(route, B id) | Assign-Utility | Forced Loss Diagonal Gap | Conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| C0_short |  |  |  |  |  |  |  |
| A_long |  |  |  |  |  |  |  |
| B_long |  |  |  |  |  |  |  |
| Oracle |  |  |  |  |  |  |  |

## 10. Concrete Run Sheet For Review

本节是执行前检查版。目标不是扩大实验矩阵，而是让首轮运行能直接回答：

```text
stronger slot context 是否足以让 learned top-1 router 的 assignment 与 expert utility 对齐？
```

### 10.1 Implementation Contract

Code workspace:

```text
Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding
```

Suggested new runner:

```text
scripts/run_slot_context_dominance_router_specialization.py
```

Suggested new config:

```text
configs/slot_context_dominance_router_specialization.json
```

Suggested output roots:

```text
results/slot_context_dominance_router_specialization/<run_name>/
figures/slot_context_dominance_router_specialization/<run_name>/
logs/slot_context_dominance_router_specialization/<run_name>.log
```

The runner should reuse the corrected `top1_selected_gate_sparse` path from the H0526-H0528 line. Before any result claim, record a reliability audit showing:

```text
dispatch_mode = top1_selected_gate_sparse
selected expert count per token = 1
selected gate probability is nonzero
router grad norm > 0 during learned-router conditions
router weight delta > 0 after training
```

### 10.2 Config Defaults

Use these defaults unless the execution environment requires a smaller smoke:

| Field | Value | Reason |
|---|---:|---|
| `num_slots` | 4 | matches anchor and experts |
| `num_experts` | 4 | one possible specialist per slot |
| `prefix_len_short` | 1 | weak-context baseline |
| `prefix_len_long` | 5 | direct test of stronger slot signal |
| `b_identity_mode` | `multi_b_primary` | primary conclusion should use many shared B identities, not a single token |
| `b_token_count` | 256 | enough shared `B_i` identities to test whether routing uses slot rather than one fixed token identity |
| `train_samples_per_slot` | 5000 | match prior H0526 scale where possible |
| `eval_samples_per_slot` | 1600 | stable route and utility matrices |
| `seq_len` | 32 | match H0526-H0528 sparse-router baseline and remove sequence-length as a rival explanation |
| `motif_positions` | prefix at 10-14, `B_i` at 15, target at 16 | stable target-position metrics and B-position routing diagnostics |
| `training_loss` | full-sequence causal NTP CE | normal NTP training; target position is primary evaluation, not the only training loss |
| `d_model` | 128 | match previous sparse-router baseline |
| `n_heads` | 4 | match previous sparse-router baseline |
| `ffn_dim` | 256 | match previous sparse-router baseline |
| `dropout` | 0.0 | remove stochastic regularization as rival explanation |
| `load_balance` | off | first diagnostic tests natural no-LB top-1 dynamics |
| `steps_full` | 1600 | match H0526 full budget |
| `steps_smoke` | 80 | check data, gradients, outputs |
| `batch_size` | 384 | match H0526 if memory allows |
| `lr` | 0.0008 | match H0526 unless unstable |
| `weight_decay` | 0.01 | match H0526 |
| `seeds_full` | `[20260521, 20260522, 20260523, 20260524]` | cross-seed stability check |
| `seed_smoke` | `20260521` | first implementation check |

If runtime is tight, do not add new conditions. Reduce only:

```text
steps_smoke, train_samples_per_slot, eval_samples_per_slot
```

Do not reduce the number of slots or experts in the formal run, because that would change the anchor question.

### 10.2.1 B-Identity Design Decision

The primary experiment should use many `B_i` tokens, not a single `B_i`.

Reason:

```text
With one B token, slot is the only varying semantic factor near the target.
That is a useful sanity check, but it does not test whether the router can resist B-identity dominance across many identities.
```

Use:

| Mode | `b_token_count` | Role | Claim boundary |
|---|---:|---|---|
| `single_b_sanity` | 1 | implementation and upper-bound diagnostic | can show slot routing is possible when identity variation is removed |
| `multi_b_primary` | 256 | main experiment | tests whether slot context can dominate routing despite many B identities |

If `single_b_sanity` is positive but `multi_b_primary` is negative, the interpretation is:

```text
the model can route by slot in the degenerate no-identity-variation case,
but stronger slot context is not sufficient to overcome B-identity variation.
```

If both are negative, first suspect representation visibility, initialization, or objective mismatch before proposing new architecture.

Execution order:

```text
stage 1 smoke: single_b_sanity for A_long / B_long / Oracle to check implementation and slot separability
stage 1 full: single_b_sanity for all required conditions across seeds
stage 2 smoke: multi_b_primary for all required conditions
stage 2 full: multi_b_primary for all required conditions across seeds
```

Stage 1 is not just an implementation smoke. It is the cleanest same-`B_i`
test: for one fixed B identity, only the slot context changes, so the
route-slot heatmap directly tests whether the B-position hidden state is split
by slot.

Stage 2 adds many `B_i` identities to test whether the same slot-aligned
routing survives B-identity variation.

### 10.3 Required Conditions

Run these four conditions in both `single_b_sanity` and `multi_b_primary`:

| Condition | Stage | Data | Router | Seeds | Decision role |
|---|---|---|---|---|---|
| `C0_short_slot_init` | full | 1-token slot cue | slot-centroid learned router | 4 | weak-context baseline |
| `A_long_repeated_slot_init` | full | 5-token repeated slot cue | slot-centroid learned router | 4 | strongest positive test for context-strength explanation |
| `B_long_distributed_slot_init` | full | 5-token distributed slot code | slot-centroid learned router | 4 | checks whether result requires trivial repeated cue |
| `Oracle_slot_router` | diagnostic | Scheme A and/or B | forced `slot -> expert` | 1-4 | tests whether the task admits slot-specific expert utility |

Run optional random-init controls only after the four required conditions finish:

| Condition | Add only if | Decision role |
|---|---|---|
| `A_long_random_init` | learned slot-init condition is positive | separates long-context effect from slot-centroid init |
| `B_long_random_init` | distributed condition is positive or ambiguous | separates compositional context visibility from init prior |

### 10.4 Data Validity Checks

Before training, write a small data audit JSON:

```text
results/slot_context_dominance_router_specialization/<run_name>/data_audit.json
```

It must verify:

```text
each B_i appears in all four slots
for each B_i, Y_{0,i}, Y_{1,i}, Y_{2,i}, Y_{3,i} are distinct
slot labels are balanced
B token frequencies are balanced across slots
Scheme B has no single prefix position that uniquely identifies the slot
full-sequence NTP labels are present for ordinary next-token training
target position index after B_i is recorded for primary metrics
```

If any check fails, stop before training. A failed data audit would invalidate the anchor test.

### 10.5 Smoke Run

Smoke command shape:

```bash
cd Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding
python scripts/run_slot_context_dominance_router_specialization.py \
  --config configs/slot_context_dominance_router_specialization.json \
  --run-name slot_context_dominance_smoke_20260526 \
  --run-stage smoke \
  --conditions C0_short_slot_init,A_long_repeated_slot_init,B_long_distributed_slot_init,Oracle_slot_router \
  --seeds 20260521
```

Smoke pass criteria:

```text
data_audit.json exists and passes
one batch trains without NaN
full-sequence NTP CE is finite and non-increasing enough to rule out implementation failure
target-position CE decreases over 80 steps in at least A_long or Oracle
router grad norm > 0 for learned-router conditions
router weight delta > 0 for learned-router conditions
route-slot matrix and forced-loss matrix files are produced
no metric script silently reports empty B-position samples
```

If smoke fails because CE does not decrease but the implementation audit passes, run one short debugging condition on `Oracle_slot_router` first. If Oracle also cannot learn, the task construction or loss mask is wrong.

### 10.6 Full Run

Full command shape:

```bash
cd Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding
python scripts/run_slot_context_dominance_router_specialization.py \
  --config configs/slot_context_dominance_router_specialization.json \
  --run-name slot_context_dominance_full_20260526 \
  --run-stage full \
  --conditions C0_short_slot_init,A_long_repeated_slot_init,B_long_distributed_slot_init,Oracle_slot_router \
  --seeds 20260521,20260522,20260523,20260524 \
  --parallel
```

Do not submit optional random-init controls in the same first full job. The first full job should close the weak-slot-signal explanation or identify the exact ambiguity.

### 10.7 Required Artifacts

Each condition and seed should write:

```text
metrics.json
training_curve.csv
routing_slot_matrix.csv
routing_b_id_matrix_or_summary.csv
hidden_slot_probe.json
forced_expert_loss_matrix.csv
ablation_delta_matrix.csv
assignment_utility_agreement.json
reliability_audit.json
model checkpoint path or checkpoint omission note
```

Aggregated run outputs:

```text
summary_by_condition.csv
summary_by_seed.csv
condition_seed_manifest.csv
```

Required report-ready figures:

```text
route_slot_heatmap_by_condition.png
forced_expert_loss_heatmap_by_condition.png
ablation_delta_heatmap_by_condition.png
assignment_utility_agreement_bar.png
hidden_slot_probe_vs_route_alignment.png
```

### 10.8 Metric Computation Order

Compute metrics in this order to avoid overclaiming from routing heatmaps:

1. Task validity:

```text
full-sequence NTP CE and accuracy
target-position CE and accuracy
```

2. Slot visibility:

```text
linear probe accuracy for h_B -> slot
slot centroid separation
```

3. Routing assignment:

```text
route-slot NMI
best-permutation route-slot diagonal purity
route-B-id NMI
```

4. Expert utility:

```text
forced expert loss matrix
ablation delta matrix
```

5. Primary judgment:

```text
assignment-utility agreement
```

The main result table should put `assignment_utility_agreement` before route NMI. Route NMI is supporting evidence only.

### 10.9 Stop / Continue Logic

Stop with a representation-problem diagnosis if:

```text
A_long and B_long both have low h_B -> slot probe accuracy
```

Interpretation:

```text
long prefix did not actually make slot visible at B; the experiment did not reach the intended physical prior.
```

Stop with a routing-objective mismatch diagnosis if:

```text
h_B -> slot probe is high
Oracle has slot-specific utility
learned conditions have low assignment-utility agreement
```

Interpretation:

```text
slot is visible and useful, but learned top-1 NTP routing does not bind assignment to expert utility.
```

Treat the context-strength explanation as supported only if:

```text
A_long and/or B_long improve over C0_short on h_B slot probe,
route-slot alignment,
forced/ablation utility diagonal gap,
and assignment-utility agreement across most seeds.
```

Treat a repeated-cue-only result as partial and bounded:

```text
A_long positive but B_long negative
```

Interpretation:

```text
strong repeated local cue can dominate routing, but distributed slot composition is not yet shown.
```

### 10.10 Post-Run Reporting Contract

After full results exist, write:

```text
Projects/from-attention-to-search/main/experiments/slot_context_dominance_router_specialization/summary.md
Projects/from-attention-to-search/main/experiments/slot_context_dominance_router_specialization/detailed.md
```

Then update the anchor only if the result changes one of:

```text
current evidence
claim boundary
next decision
```

If the anchor is updated, copy it to:

```text
sync/S000_current_specialization/anchors/expert_specialization/slot_context_dominance_router_specialization_anchor.md
```

and verify source and sync copy are identical.
