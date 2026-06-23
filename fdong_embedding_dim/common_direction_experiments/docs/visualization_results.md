# Visualization and Results

## Status

Stage 1 passed after one conjecture correction. Stage 2 is partial. Stage 3
passed with a causal caveat. Stage 4 passed and directly separates a
uniform-disjoint null from shared statistical features. Stage 5 passed for the
early optimizer-path mechanism of reweighting.

## Reading contract

Each stage plot must state:

- exact conditions being compared;
- x-axis checkpoint and y-axis metric definition;
- observed trajectory and seed variability;
- pass, fail, or insufficient-evidence outcome;
- what the result proves and does not prove.

## Stage 1: direction nucleation

### What was tested

Five conditions separated distributed targets, shared K targets, shared K input,
inverse-square-root target-frequency reweighting, and low context diversity. All
conditions used the same seed-specific initialization, 16 full-batch patterns,
five seeds, 400 gradient-descent steps, and an 8-dimensional tied embedding plus
linear two-token context map.

### Evidence

Artifacts:

- `outputs/common_direction_causal/stage1_nucleation/history.csv`
- `outputs/common_direction_causal/stage1_nucleation/aggregate.csv`
- `outputs/common_direction_causal/stage1_nucleation/metrics.png`
- `outputs/common_direction_causal/stage1_nucleation/summary.json`

At step 0, shared-target output-gradient top-1 energy was `0.542`, compared with
`0.354` for distributed targets, while embedding top-1 energy was identical
(`0.223`) because initialization was shared. By step 50, embedding top-1 energy
had separated (`0.258` shared versus `0.220` distributed). This is the required
gradient-before-parameter temporal order.

The negative output gradient for K aligned with the mean K-target hidden state.
K/context cosine then rose from a seed-dependent initial mean of `0.205` to above
`0.98`. The shared-K-input condition also had greater continuation-hidden top-1
energy at initialization than its no-K-input control.

At step 400, reweighting reduced embedding top-1 energy from `0.284` to `0.270`;
both conditions reached `1.0` K-target accuracy.

### Conjecture correction

The original pass condition incorrectly predicted that shared K input must make
the input-role gradient more low-rank. It did not. Shared input guarantees a
common forward hidden component, but diverse continuation targets can produce
different or cancelling backward errors. The stage contract was corrected to
measure continuation-hidden concentration instead of input-gradient
concentration.

### Allowed conclusion

In this tied linear-context toy, shared K targets create an aligned output-gradient
mode before the embedding spectrum separates; shared K input independently
creates a common hidden component; soft reweighting weakens final concentration
without preventing K learning.

This stage does not establish nested attraction or positive feedback.

## Stage 2: frequency-matched nested attraction

### What was tested

Nested and cyclically rewired transition graphs had identical token input counts,
target counts, and sample weights. The model was initialized with a flat spectrum,
a gain of `4` aligned to the initial top hidden-gradient direction, the same gain
in an orthogonal direction, or aligned gain followed by singular clipping from
step `100`. Five seeds were trained for 600 steps.

Artifacts:

- `outputs/common_direction_causal/stage2_nested_attraction/history.csv`
- `outputs/common_direction_causal/stage2_nested_attraction/aggregate.csv`
- `outputs/common_direction_causal/stage2_nested_attraction/metrics.png`
- `outputs/common_direction_causal/stage2_nested_attraction/summary.json`

### What passed

Aligned gain increased the step-0 hidden-gradient energy in the seeded direction
from `0.361` in the flat model to `0.958`. The equally strong orthogonal gain gave
`0.769`. This supports the local gain-amplification statement: gain matters most
when its direction is aligned with the current error-induced hidden gradient.

Clipping reduced final embedding top-1 energy from `0.599` to `0.257` and retained
`1.0` training accuracy.

### What failed

The nested aligned-gain condition ended with centered extra-feature occupation
`0.097` in the seeded direction, compared with `0.090` for the frequency-matched
rewired condition. This did not meet the preregistered `10%` separation.

Clipping increased rather than reduced occupation of the original seeded
direction (`0.177` versus `0.097`), even though it reduced global embedding
concentration. Therefore global spectral concentration and projection onto one
historical reference direction are not interchangeable metrics.

### Conjecture update

This experiment supports instantaneous singular-gain amplification but does not
support the stronger claim that nested branch-specific residuals must continue
accumulating in the same direction. In this model, residual features can rotate
or redistribute while the task is learned. Stage 3 therefore tests natural
nucleation and temporal feedback directly rather than assuming attraction.

## Stage 3: natural feedback, saturation, and tail harm

### What was tested

Eight nested branches were trained under uniform group weights, Zipf group
weights, Zipf plus target-frequency reweighting, Zipf with K-target loss removed
after mean K CE fell below `0.1`, and Zipf with embedding singular clipping from
step `100`. Five seeds were trained for 1200 steps.

Artifacts:

- `outputs/common_direction_causal/stage3_feedback_saturation/history.csv`
- `outputs/common_direction_causal/stage3_feedback_saturation/aggregate.csv`
- `outputs/common_direction_causal/stage3_feedback_saturation/metrics.png`
- `outputs/common_direction_causal/stage3_feedback_saturation/summary.json`

### Saturation evidence

Before mean K-target CE crossed `0.1`, mean K-target gradient norm was `0.265`;
afterward it was `0.017`. Mean embedding `sigma1` growth per 10-step interval
fell from `0.027` to `0.012`. This supports the saturation statement: a large
singular value cannot maintain rapid growth after the relevant error signal has
mostly vanished.

### Temporal feedback evidence

In the natural Zipf baseline, hidden top-1 energy correlated `0.70` with the next
10-step `sigma1` increment. This is temporal evidence consistent with feedback,
but it is not sufficient causal proof: both variables also change with training
time and total gradient norm. The Stage 2 clipping intervention did not reduce
occupation of the historical reference direction, so the strong causal feedback
claim remains open.

### Long-tail evidence

Stable full tail accuracy occurred at:

| condition | stable step |
|---|---:|
| uniform | 520 |
| Zipf | 970 |
| Zipf + reweighting | 770 |
| Zipf + hard K stopping | not reached by 1200 |
| Zipf + spectral clipping | 970 |

Reweighting was the only intervention that improved stable tail speed while
preserving final K accuracy. It also raised final tail residual effective rank
from `3.177` to `3.285` and lowered final tail loss from `0.118` to `0.037`.

Hard removal of K-target loss was harmful: the removal occurred at mean step
`469`, but final K accuracy fell to `0.90` and stable full tail accuracy was not
reached. The tied system continues to update K through its input role, so simply
removing its target anchor after a threshold is not safe.

Spectral clipping reduced concentration but did not improve tail learning speed.
This separates two claims that should not be conflated: a flatter embedding
spectrum is not by itself sufficient to improve tail optimization.

## Stage 4: uniform-disjoint null versus shared statistical features

### What was tested

Six conditions compared a fully uniform-disjoint token table with matched
relabeling, high-frequency shared target, shared target plus reweighting, a
shared one-token input prefix, and an identical two-token prefix with many
distinct targets. The model was a tied embedding plus linear two-token context
map with dimension `16`, `32` probe groups, `32` matched background groups, five
seeds, and 600 full-batch gradient-descent steps.

Artifacts:

- `outputs/common_direction_causal/stage4_uniform_vs_shared/history.csv`
- `outputs/common_direction_causal/stage4_uniform_vs_shared/aggregate.csv`
- `outputs/common_direction_causal/stage4_uniform_vs_shared/metrics.png`
- `outputs/common_direction_causal/stage4_uniform_vs_shared/summary.json`

### What passed

The uniform-disjoint condition matched the relabeled null at step 0. Both had
the same centered total-embedding-gradient top-1 energy (`0.102`), because the
relabeling preserved token-role counts and introduced no shared target or shared
prefix source.

The shared-target condition produced a much larger centered output-gradient top
mode at step 0: `0.500` versus `0.125` for uniform-disjoint. The repeated
target's negative output gradient aligned with the mean hidden state of the
contexts predicting it (`0.99997` cosine). This is direct evidence for the
target-frequency nucleation mechanism.

At step 0, the embedding spectrum was almost unchanged between uniform-disjoint
and shared-target (`0.095` versus `0.096` centered top-1 energy), while the
gradient spectrum was already very different. By step 600, the centered
embedding top-1 energy was larger for shared-target (`0.143`) than
uniform-disjoint (`0.112`). This supports the temporal order:

```text
shared target statistics -> gradient concentration -> later parameter concentration
```

Reweighting reduced final centered embedding top-1 energy from `0.143` to
`0.128` while preserving final probe accuracy (`1.0`).

The shared-input-prefix condition created a large raw hidden common component at
step 0: raw probe-hidden top-1 energy was `0.562`, compared with `0.150` for
uniform-disjoint. However its centered probe-hidden energy was only `0.159`,
close to the uniform value `0.150`. This means the one-token prefix mostly adds
a shared mean component; it does not by itself create the same output-gradient
nucleation as a repeated target.

The identical two-token prefix condition had raw probe-hidden top-1 energy
`1.0`, but final probe accuracy stayed near chance (`0.031`) because the same
context maps to many distinct targets. This is a useful negative control:
short-prefix commonness can exist without a learnable deterministic prediction
direction.

### Allowed conclusion

In this toy model, a fully uniform-disjoint data table does not create an extra
common mode beyond the matched relabeled null. A repeated target creates a strong
early output-gradient common direction, and target-frequency reweighting weakens
the later embedding concentration. Shared input/prefix structure creates hidden
commonness, but it is not equivalent to shared-target gradient nucleation.

This supports the sharper claim:

> The first common direction needs a shared statistical source. Nested or short
> prefix structure matters when it creates a high-frequency subfeature or reuses
> a direction that already exists; nestedness alone is not a sufficient
> explanation for the first direction in this operationalization.

### Remaining boundary

This is still a tied-linear-context toy. It does not prove that real LLM top
singular directions are dominated by high-frequency targets. It does show that
the agenda-style statement "some common information is learned first" is
incomplete unless it specifies why that information has larger early gradient
mass.

## Stage 5: does reweighting change the optimizer path?

### What was tested

Stage 5 reused the Stage 3 nested Zipf task and compared `uniform`, `zipf`,
`zipf_reweight`, and `zipf_clip`. At every checkpoint, the script computed the
exact full-batch update before applying it. It then measured how much update
energy lay in the current centered embedding top direction, how many effective
directions the update used, and how much the next exact gradient step would
reduce tail loss.

Artifacts:

- `outputs/common_direction_causal/stage5_reweight_optimizer_path/history.csv`
- `outputs/common_direction_causal/stage5_reweight_optimizer_path/aggregate.csv`
- `outputs/common_direction_causal/stage5_reweight_optimizer_path/metrics.png`
- `outputs/common_direction_causal/stage5_reweight_optimizer_path/summary.json`

### Nucleation-window evidence

The primary window is steps `0..50`, because the proposed mechanism says
reweighting should change the path before the common channel has already shaped
the parameter space.

| metric, steps 0..50 | uniform | Zipf | Zipf + reweight | Zipf + clip |
|---|---:|---:|---:|---:|
| common-update share | 0.235 | 0.262 | 0.200 | 0.262 |
| embedding-update effective rank | 5.097 | 1.824 | 3.333 | 1.824 |
| residual-update effective rank | 4.752 | 1.982 | 3.020 | 1.982 |
| next-step tail loss decrease | 0.00516 | 0.00076 | 0.00391 | 0.00160 |

This supports the intended mechanism. Zipf makes the early update much more
low-rank and much less useful for immediate tail loss reduction. Reweighting
partly restores update rank and tail loss decrease in the same early window.

### Final and speed evidence

Stable full tail accuracy occurred at:

| condition | stable step |
|---|---:|
| uniform | 520 |
| Zipf | 970 |
| Zipf + reweighting | 770 |
| Zipf + clipping | 970 |

Final centered embedding top-1 energy was:

| condition | final centered top-1 energy |
|---|---:|
| uniform | 0.217 |
| Zipf | 0.219 |
| Zipf + reweighting | 0.215 |
| Zipf + clipping | 0.218 |

Final tail residual effective rank was:

| condition | final tail residual rank |
|---|---:|
| uniform | 3.199 |
| Zipf | 3.164 |
| Zipf + reweighting | 3.292 |
| Zipf + clipping | 3.172 |

### Important boundary

The result does not support the overly strong statement that reweighting keeps
the current top-vector update share lower throughout all training. In the wider
`0..200` window, common-update share is not monotonically lower for reweighting.
The supported statement is narrower and more useful:

> Reweighting changes the nucleation path. It prevents the earliest Zipf update
> from becoming as low-rank, increases residual-update rank, and gives the tail
> a larger immediate loss decrease before the common channel has fully shaped
> the parameter space.

`zipf_clip` is not a counterexample to this mechanism. It lowers the spectrum
after step `100`, but it does not improve stable tail speed. This means forcing
a flatter spectrum after the fact is not equivalent to training under a
scale-balanced path from the start.

## Overall claim audit after all five stages

### Supported in this toy model

1. Shared high-frequency targets create an aligned gradient mode before parameter
   spectral separation.
2. Shared K input creates a common hidden component, but not necessarily an
   aligned input-gradient mode.
3. Singular gain amplifies hidden gradients that are aligned with it.
4. K-related gradient and singular-value growth both decay after K loss
   saturates.
5. Zipf weighting delays tail learning, and soft target-frequency reweighting
   partially recovers it.
6. A fully uniform-disjoint table matches a relabeled null and does not produce
   the shared-target gradient mode.
7. Shared input/prefix structure creates hidden commonness, but this is not the
   same mechanism as shared-target output-gradient nucleation.
8. Target-frequency reweighting changes the early optimizer path: it reduces
   nucleation-window common-update share, increases update effective rank, and
   improves immediate tail loss decrease.
9. Forcing a flatter spectrum after the path has already formed does not recover
   the same tail-learning speed, so "flat final spectrum" is not the same as
   "scale-balanced optimization path."

### Not supported or still open

1. Frequency-matched nested topology did not substantially increase final
   branch-extra occupation of the seeded singular direction relative to rewiring.
2. Lower global spectral concentration did not necessarily lower occupation of a
   historical direction or improve tail speed.
3. The representation-to-parameter positive-feedback loop has temporal
   correlation evidence but lacks a successful causal intervention.
4. No result here establishes that the same mechanism dominates a real LLM.
5. The optimizer-path evidence is strongest in the nucleation window. It does
   not prove that the update projection onto the current top vector is lower at
   every later point in training.

### Updated working theory

The evidence supports a narrower story than the original agenda:

> Frequency and shared targets can nucleate a common gradient/parameter mode, and
> existing gain amplifies aligned errors. The mode's growth saturates as those
> errors vanish. However, nested residual features are not forced to remain in
> that mode; they can rotate or redistribute. Long-tail slowdown is real in the
> Zipf toy, but spectral concentration alone is not yet established as its unique
> causal mediator.

After Stage 4, the stronger nucleation statement is:

> A first common direction should be traced to a shared statistical source, such
> as repeated targets, repeated inputs, or high-frequency subpatterns. Nested
> language structure is a plausible amplifier and reuse mechanism, but it is not
> yet a complete explanation for how the first dominant direction appears.

After Stage 5, the reweighting mechanism should be stated as an early-path
effect:

> Reweighting works by reducing frequency domination during nucleation, which
> keeps early updates higher-rank and more useful for tail loss reduction. It is
> not merely a post-hoc spectrum-flattening method.
