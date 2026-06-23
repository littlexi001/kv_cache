# Experiment Design

## Shared implementation contract

### Input

- Synthetic token-transition tables.
- One tied embedding matrix `E`.
- One trainable linear context map `M` taking two concatenated embeddings.
- Cross-entropy next-token loss.

### Fixed parameters

- Hidden dimension: `8` in Stages 1--3 and `16` in Stage 4; large enough for
  residual directions, small enough for exact full SVD.
- Branch count: `8` in Stages 1--3. Stage 4 uses `32` independent groups to make
  the uniform-disjoint null less dominated by finite-sample noise.
- Seeds: `0..4` for the first pass.
- Optimizer: explicit full-batch gradient descent, with no momentum or weight
  decay, so the measured gradients equal the applied updates.
- Record interval: every `10` steps plus step `0`.

### Outputs

Every stage writes:

- `history.csv`: checkpoint-level metrics for every run.
- `summary.json`: aggregate pass/fail/insufficient-evidence decisions.
- `metrics.png`: trajectories with axis definitions and condition labels.
- `config.json`: exact parameters.

## Stage 1: direction nucleation

### Question

Do shared-target and shared-input roles create distinct aligned gradient modes,
and does the gradient mode appear before the parameter singular mode?

### Conditions

All conditions contain 16 equally weighted patterns before optional reweighting.

1. `distributed_high_no_k_input`: eight diverse contexts predict eight distinct
   targets; K is not shared.
2. `shared_high_no_k_input`: the same eight contexts predict K; K is not used as
   a continuation input.
3. `shared_high_k_input`: the same eight contexts predict K, and K is also used
   as input for eight distinct continuation targets.
4. `shared_high_k_input_reweight`: condition 3 with inverse-square-root target
   frequency weighting.
5. `shared_lowdiv_no_k_input`: K is shared, but only two unique contexts predict
   it; this separates target frequency from context diversity.

### Exact procedure

1. Initialize every condition from the same seed-specific `E` and `M`.
2. Before the first update, compute exact output-role, input-role, and `M`
   gradients.
3. Train with full-batch gradient descent.
4. At each checkpoint, record gradient spectra, parameter spectra, hidden spectra,
   K/context alignment, and per-pattern-family loss/accuracy/margin.
5. Aggregate trajectories across seeds.

### Pass conditions

Stage 1 passes only if all are observed:

1. Shared-target conditions have a stronger output-gradient top mode than the
   distributed-target control at step 0.
2. The negative output gradient for K aligns with the mean K-target hidden state.
3. The shared-input condition has a stronger continuation-hidden-state top mode
   than the no-K-input condition. Shared input guarantees a common forward
   component; it does not guarantee aligned input gradients when continuation
   targets differ.
4. Gradient concentration precedes growth of parameter concentration.
5. Reweighting reduces parameter concentration without preventing K learning.

If only some conditions pass, the result is `partial`. If differences are smaller
than seed variation, the result is `insufficient evidence`.

### Named failure reasons

- `no_output_nucleation`: shared K does not increase output-gradient concentration.
- `no_shared_hidden_component`: K input does not increase continuation-hidden
  concentration.
- `no_temporal_precedence`: parameter concentration does not follow gradient
  concentration.
- `reweight_no_spectral_effect`: reweighting does not reduce concentration.
- `reweight_underfits_k`: reweighting prevents K prediction.

## Stage 2: nested attraction

### Question

Does a pre-existing singular gain amplify aligned hidden gradients, and does a
nested transition topology subsequently place more branch-specific residual
information into that direction than a frequency-matched rewired topology?

### Frequency-matched topology

Each of eight branches has four transitions. The nested condition is:

```text
(G0, G1) -> K
(G1, K)  -> G2
(K, G2)  -> G0
(G2, G0) -> G1
```

The rewired condition cyclically permutes branch targets for the last three
transition families. Every token keeps the same total input and target count;
only within-branch inheritance is broken.

### Gain conditions

- `flat`: original shared initialization.
- `aligned_gain`: multiply the embedding right-space gain by `4` along the top
  hidden-gradient direction measured in the flat nested model at step 0.
- `misaligned_gain`: apply the same gain along an orthogonal direction.
- `aligned_gain_clip`: start aligned, then clip the first embedding singular value
  to at most `1.2 * sigma2` after step `100`.

### Metrics

- hidden-gradient energy in the seeded direction;
- hidden-state energy in the seeded direction;
- centered branch-extra energy in the seeded direction;
- extra-feature effective rank;
- embedding top singular value and alignment to the seeded direction;
- family accuracy, loss, and margin.

### Pass conditions

1. At step 0, aligned gain increases hidden-gradient projection into the seeded
   direction relative to flat and misaligned gain.
2. With identical token frequencies, nested aligned gain produces greater
   centered extra-feature occupation of the seeded direction than rewired aligned
   gain.
3. Mid-training clipping reduces later occupation and embedding concentration
   without preventing full training accuracy.

Failure of condition 2 falsifies this operationalization of nested attraction;
it does not falsify every possible definition of linguistic nesting.

## Stage 3: feedback, saturation, and tail harm

### Question

After a common mode forms naturally, does hidden occupation predict later
singular-value growth, does that growth slow when K-target error saturates, and
does the dominant mode causally slow tail learning?

### Conditions

- `uniform`: eight branches with equal group weights.
- `zipf`: branch 0 has weight `0.55`; seven tail branches share `0.45`.
- `zipf_reweight`: Zipf plus inverse-square-root weighted target-frequency loss.
- `zipf_stop_k`: stop applying K-target loss after its mean CE falls below `0.1`.
- `zipf_clip`: clip embedding `sigma1` to at most `1.2 * sigma2` from step `100`.

All conditions share the same nested transition graph and seed-specific
initialization.

### Metrics

- K-target CE, accuracy, and gradient norm;
- embedding `sigma1`, top-1 energy, and next-interval `delta_sigma1`;
- hidden top-1 energy and its lagged relation to `delta_sigma1`;
- common and tail accuracy, loss, and margin;
- tail residual effective rank after removing the current embedding top direction;
- group-conditioned gradient SIR and common-tail gradient cosine.

### Pass conditions

1. K-target gradient norm and `delta_sigma1` decrease after K loss crosses `0.1`.
2. Hidden top-1 occupation positively predicts next-interval `delta_sigma1` in
   the natural Zipf baseline.
3. Zipf delays stable full tail accuracy relative to uniform.
4. At least one causal intervention improves tail speed or residual rank while
   preserving K accuracy.

Failure of condition 2 is evidence against the simple positive-feedback model.

## Stage 4: uniform-disjoint null versus shared statistical features

### Question

Is a first common direction produced by nestedness alone, or does it require a
shared statistical feature such as a high-frequency target, high-frequency input,
or shared prefix?

### Conditions

All conditions use independent two-token contexts and no cross-boundary loss.

1. `uniform_disjoint`: examples of the form `(U_Ai, U_Bi) -> U_Ci` plus matched
   background examples. Every token appears in one role for one example, and
   every complete pattern has equal weight.
2. `uniform_disjoint_relabel`: same token-role counts as condition 1, but probe
   targets are permuted. This is the matched null for deterministic labels.
3. `shared_target`: diverse disjoint contexts all predict the same target `K`,
   plus matched background examples.
4. `shared_target_reweight`: condition 3 with inverse-square-root target
   frequency weighting.
5. `shared_input_prefix`: every probe context shares the first input token `P`,
   but has a distinct second input and distinct target.
6. `shared_two_token_prefix`: every probe context shares both input tokens
   `(P, Q)` and has distinct targets. This is intentionally impossible to solve
   with a deterministic next-token model; it tests whether an identical short
   prefix can create hidden commonness without learnable target structure.

### Metrics

- centered output-gradient top-1 energy;
- centered total-embedding-gradient top-1 energy;
- raw and centered probe-hidden top-1 energy;
- centered embedding top-1 energy;
- cosine between the repeated target's negative output gradient and the mean
  probe hidden state;
- probe and background loss/accuracy.

### Pass conditions

Stage 4 passes if:

1. `uniform_disjoint` and `uniform_disjoint_relabel` match at step 0.
2. `shared_target` has much larger centered output-gradient top-1 energy than
   `uniform_disjoint` at step 0.
3. The repeated target's negative output gradient aligns with the mean hidden
   state of contexts predicting it.
4. The gradient gap appears before a comparable embedding-spectrum gap, while
   final centered embedding concentration becomes larger for `shared_target`.
5. `shared_input_prefix` creates a raw hidden common component but is not
   equivalent to the shared-target output-gradient mode.
6. `shared_target_reweight` lowers final centered embedding concentration
   without preventing probe learning.

This stage directly tests whether the initial common mode needs a shared
frequency/statistical source rather than nestedness by itself.

## Stage 5: reweighting and optimizer-path mechanism

### Question

If target-frequency reweighting works for the reason proposed here, then it
should not merely lower the final spectrum. It should change the early optimizer
path: the update should be less concentrated in the current common direction,
should use more residual directions, and should produce larger immediate tail
loss improvement.

### Conditions

Stage 5 reuses the Stage 3 nested Zipf task:

1. `uniform`: equal branch weights.
2. `zipf`: branch 0 has weight `0.55`; tail branches share the rest.
3. `zipf_reweight`: Zipf plus inverse-square-root target-frequency reweighting.
4. `zipf_clip`: Zipf plus singular clipping from step `100`. This is not treated
   as a refutation of reweighting if it fails, because it changes the spectrum
   after the optimization path has already been formed.

### Metrics

At each checkpoint, before applying the next update, the script computes the
exact full-batch gradient and decomposes the one-step update.

- `total_update_common_share`: fraction of current gradient/update energy
  projected onto the centered embedding top direction and the corresponding
  `M` row direction.
- `e_update_effective_rank`: effective rank of the embedding update.
- `e_residual_update_effective_rank`: effective rank of the embedding update
  after removing the current common direction.
- `tail_loss_delta_next_step`: exact tail CE decrease after one hypothetical
  gradient step from the current checkpoint.
- `tail_margin_delta_next_step`: exact tail margin increase after one
  hypothetical gradient step.
- final centered embedding top-1 energy, final tail residual effective rank,
  and stable full tail-accuracy step.

### Pass conditions

The primary window is the nucleation window `0..50`, because this is where
reweighting should change the path if it acts by preventing early frequency
domination. Stage 5 passes if:

1. `zipf_reweight` has lower nucleation-window common-update share than `zipf`.
2. `zipf_reweight` has higher nucleation-window update effective rank than
   `zipf`.
3. `zipf_reweight` has higher nucleation-window residual-update effective rank
   than `zipf`.
4. `zipf_reweight` has larger nucleation-window next-step tail loss decrease
   than `zipf`.
5. `zipf_reweight` reaches stable full tail accuracy earlier than `zipf`.
6. `zipf_reweight` lowers final centered embedding top-1 energy.
7. `zipf_clip` lowers the spectrum but does not improve the tail path, showing
   that forcing the final spectrum flatter is not equivalent to learning under
   a scale-balanced path from the start.
