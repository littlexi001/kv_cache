# Section 30: R2H-KV Hybrid Smoke Experiments

Date: 2026-06-29

## 0. Experiment Scope

This section records the first server smoke experiments for the R2H-KV idea from Section 29.

Important limitation:

```text
The GPU-friendly page/q-sketch kernels are not implemented yet.
These experiments use existing layerbudgetattn backends:
  recent
  qabs8cand3reuse
  headmix_qabs_reuse
  landmark
```

Therefore these results test the policy idea:

```text
Different layers/backends behave differently,
and a refined hybrid policy can be safer than uniform compression.
```

They do not yet test the final GPU-friendly runtime.

## 1. Server Setup

Server:

```text
host: df
path: /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
model: /home/fdong/hrj/prove/Qwen3-0.6B
text: /home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt
```

Common config:

```text
prefill_tokens = 2048
eval_tokens = 64
chunk_size = 64
eval_chunk_size = 1
dtype = bfloat16
attn_implementation = eager
qabs_dims = 8
candidate_fraction = 0.03
top_fraction = 0.03
protect_sink_tokens = 10
protect_recent_tokens = 10
local_recent = 512
```

Baseline:

```text
loss = 1.65158
PPL  = 5.21522
```

## 2. Experiment A: Existing Hybrid Budget Search

Command output directory:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/r2h_hybrid_smoke_codex_0629_1
```

Script:

```text
src/run_influence_gated_hybrid_budget.py
```

Top results by PPL ratio:

| Mode | Layers | Backend | PPL Ratio | Time Ratio |
| --- | --- | --- | ---: | ---: |
| pcic_0_13_qabs3set | 0,13 | qabs8cand3reuse | 0.9886 | 1.072 |
| pcic_0_13_landmark | 0,13 | landmark | 0.9918 | 1.006 |
| synthetic_safe_4_5_landmark | 4,5 | landmark | 0.9950 | 1.003 |
| auto_mse_1_2_5_qabs3set | 1,2,5 | qabs8cand3reuse | 0.9960 | 1.092 |
| mid_7_14_headmix8 | 7-14 | headmix_qabs_reuse | 0.9972 | 1.281 |
| synthetic_safe_4_5_headmix8 | 4,5 | headmix_qabs_reuse | 0.9981 | 1.083 |
| synthetic_safe_4_5_qabs3set | 4,5 | qabs8cand3reuse | 1.0008 | 1.073 |
| mid_7_14_qabs3set | 7-14 | qabs8cand3reuse | 1.0029 | 1.221 |
| pcic_0_13_headmix8 | 0,13 | headmix_qabs_reuse | 1.0033 | 1.088 |

Worst notable cases:

| Mode | Layers | Backend | PPL Ratio |
| --- | --- | --- | ---: |
| pcic_0_6_headmix8 | 0,6 | headmix_qabs_reuse | 1.0262 |
| pcic_0_6_qabs3set | 0,6 | qabs8cand3reuse | 1.0152 |
| mid_7_14_landmark | 7-14 | landmark | 1.0167 |
| auto_mse_1_2_5_landmark | 1,2,5 | landmark | 1.0116 |

Interpretation:

```text
Layer/backend interaction is strong.
The same backend can be safe on one layer set and risky on another.
For example:
  layers 0,13 + qabs improves PPL;
  layers 0,6 + qabs degrades PPL;
  mid 7-14 prefers headmix over landmark.
```

This supports the R2H-KV premise:

```text
Hybrid policy should be selected from measured layer/head behavior,
not from a single global rule.
```

Timing note:

```text
The first landmark run had a 49x time ratio due to one-time overhead/warmup.
Do not use this smoke run for final speed claims.
```

## 3. Experiment B: Naive Depth Prior vs Refined Hybrid

Command output directory:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/r2h_prior_smoke_codex_0629_2
```

Script added:

```text
src/run_r2h_prior_smoke.py
```

Policies tested:

```text
r2h_local_shallow_tail:
  layers 0-5 and 23-27 -> recent512

r2h_mid_qabs:
  layers 6-13 -> qabs8cand3reuse

r2h_prior_local_midqabs:
  layers 0-5 and 23-27 -> recent512
  layers 6-13 -> qabs8cand3reuse
  layers 14-22 -> full

r2h_refined_safe:
  layers 0,13 -> qabs8cand3reuse

r2h_refined_mid_headmix:
  layers 7-14 -> headmix_qabs_reuse

r2h_refined_hybrid:
  layer 0 -> qabs8cand3reuse
  layers 4,5 -> headmix_qabs_reuse
  layers 7-14 -> headmix_qabs_reuse

uniform_local_all:
  all layers -> recent512

uniform_qabs_all:
  all layers -> qabs8cand3reuse
```

Results:

| Mode | Compressed Layers | Backend Summary | PPL Ratio | Time Ratio |
| --- | --- | --- | ---: | ---: |
| r2h_refined_safe | 0,13 | qabs8cand3reuse | 0.9886 | 1.067 |
| r2h_refined_mid_headmix | 7-14 | headmix_qabs_reuse | 0.9972 | 1.292 |
| r2h_refined_hybrid | 0,4,5,7-14 | qabs + headmix | 0.9991 | 1.360 |
| r2h_mid_qabs | 6-13 | qabs8cand3reuse | 1.0328 | 1.498 |
| uniform_qabs_all | 0-27 | qabs8cand3reuse | 1.1382 | 1.685 |
| r2h_local_shallow_tail | 0-5,23-27 | recent512 | 1.4212 | 1.020 |
| r2h_prior_local_midqabs | 0-13,23-27 | recent + qabs | 1.6249 | 1.249 |
| uniform_local_all | 0-27 | recent512 | 12.6279 | 1.050 |

## 4. Main Findings

### Finding 1: Naive shallow-aggressive recent compression fails

The policy:

```text
layers 0-5 and 23-27 -> recent512
```

got:

```text
PPL ratio = 1.4212
```

This directly rejects the overly simple rule:

```text
shallow layers only contain token-frequency/local information,
so they can be aggressively recent-window compressed.
```

The better interpretation is:

```text
Some shallow layers are more compressible than semantic layers,
but shallow layers still contain important sink/position/format/routing structure.
They cannot all be collapsed to recent-only.
```

### Finding 2: Uniform compression is unsafe

Uniform policies were bad:

```text
all layers recent512:       PPL ratio = 12.6279
all layers qabs8cand3reuse: PPL ratio = 1.1382
```

This supports the hybrid direction.

### Finding 3: Refined hybrid is quality-safe in this smoke setting

The refined hybrid:

```text
layer 0 -> qabs
layers 4,5 -> headmix_qabs_reuse
layers 7-14 -> headmix_qabs_reuse
other layers -> full
```

got:

```text
PPL ratio = 0.9991
```

It compressed 11/28 layers with essentially no PPL loss in this 64-token hard-topic smoke.

This is the strongest result from this run:

```text
profile/refinement matters more than a monotonic depth prior.
```

### Finding 4: The current prototype is not yet faster

Even when PPL is safe:

```text
r2h_refined_hybrid time ratio = 1.36
```

This is expected because current qabs/headmix backends use eager/Python and dynamic candidate handling.

It reinforces Section 29's GPU-friendly design:

```text
replace dynamic token qabs with fixed q-sketch page routing;
avoid dense masks and ragged gather;
compile a small number of fixed backend templates.
```

## 5. Updated Direction

The simple prior should be revised from:

```text
shallow aggressive, deep conservative
```

to:

```text
use depth as a weak prior,
but choose backend by layer/head profiling and risk calibration.
```

Specifically:

```text
Do not use recent-only for all shallow layers.
Prefer qabs/headmix for selected shallow/mid layers that show safe behavior.
Keep semantic or high-risk layers full unless profiling says otherwise.
```

## 6. Next Experiment

The next necessary experiment is not another manual layer list. It should build the actual profile:

```text
profile_by_layer_head.csv
  remote_mass
  qabs_energy
  reuse_energy
  margin_risk
  delta_loss
```

Then compile:

```text
recommended_hybrid_policy.json
```

and compare:

```text
manual refined hybrid
vs profile-compiled hybrid
vs uniform qabs
vs uniform local
```

on longer eval windows and at least three datasets.

