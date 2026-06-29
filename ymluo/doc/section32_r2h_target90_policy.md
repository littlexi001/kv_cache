# Section 32: R2H-KV Target 90% Compression Policy

Date: 2026-06-29

## 0. Motivation

The previous profile-compiled conservative policy achieved only about 19% effective KV-access compression.

New target:

```text
KV compression around 90%
keep fraction around 10%
```

This requires a different compiler objective. Threshold-based conservative compilation will not naturally reach this compression level.

## 1. New Script

Added:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/compile_r2h_target_policy.py
```

Input:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/r2h_profile_codex_0629_local/profile_by_layer_head.csv
```

Output:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/r2h_profile_codex_0629_local/target90/target_profile_by_layer_head.csv
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/r2h_profile_codex_0629_local/target90/recommended_hybrid_policy_target90.json
```

Command:

```bash
python ymluo/projects/qwen3_top2_head_limit3_ppl/src/compile_r2h_target_policy.py \
  --profile_csv ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/r2h_profile_codex_0629_local/profile_by_layer_head.csv \
  --output_dir ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/r2h_profile_codex_0629_local/target90 \
  --target_keep_fraction 0.10 \
  --full_reserve_fraction 0.06 \
  --hybrid_reserve_fraction 0.10 \
  --head_group_size 4
```

## 2. Result

Target policy summary:

| Metric | Value |
| --- | ---: |
| Heads | 448 |
| Target keep fraction | 10.00% |
| Estimated keep fraction | 9.99% |
| Estimated compression fraction | 90.01% |

Template counts:

| Template | Heads |
| --- | ---: |
| FULL | 27 |
| HYBRID_SAFE_B4 | 21 |
| SYNTH_M8 | 400 |

Interpretation:

```text
To reach 90% compression, the policy must reserve only a small number of high-risk heads as FULL/HYBRID_SAFE,
and replace most heads with aggressive synthetic/page-compressed representations.
```

## 3. Why Current Runtime Cannot Validate This Yet

The current runtime supports:

```text
recent
landmark
qabs
headmix_qabs
synthetic at layer-budget granularity
```

But the target90 policy is:

```text
head-group template policy
mostly SYNTH_M8
```

That requires:

```text
per-head or head-group physical KV replacement
page/synthetic KV storage
fixed-shape GPU kernels
```

Current `layerbudgetattn` fallback cannot execute:

```text
layer 13 head group A -> FULL
layer 13 head group B -> SYNTH_M8
layer 13 head group C -> HYBRID_SAFE_B4
```

without collapsing the head-level policy into a much coarser layer-level approximation.

## 4. Important Consequence

90% compression is not a tuning of the current qabs/headmix prototype.

It requires changing the product design from:

```text
select fewer real tokens at decode time
```

to:

```text
physically store a much smaller KV representation for most heads
and keep only a small high-risk subset full.
```

The practical 90% design is:

```text
6% heads FULL
10% heads HYBRID_SAFE
84% heads SYNTH/PAGE compressed
```

The exact numbers can change, but the structure is necessary.

## 5. Risk

This policy is aggressive. Based on previous synthetic/static KV negative results, it is not safe to assume PPL will hold.

The next required experiment is not wall-time benchmarking. It is:

```text
Can SYNTH_M8 approximate enough heads well enough at this compression level?
```

Minimum validation:

```text
1. Per-head synthetic reconstruction MSE.
2. Held-out query MSE.
3. Counterfactual delta-loss for SYNTH_M8 heads.
4. PPL with only the lowest-risk 25%, 50%, 75%, 84% heads using SYNTH_M8.
```

## 6. Next Implementation Step

Add a head-group synthetic runtime:

```text
layer -> head_groups -> template
```

Required backend:

```text
SYNTH_M8:
  keep sink + recent
  replace remote KV with 8 synthetic prototypes per head or page-group
```

Then run a staged target curve:

| Target Compression | Expected Purpose |
| ---: | --- |
| 50% | sanity check |
| 70% | moderate aggressive |
| 80% | hard but plausible |
| 90% | final target |

The 90% policy should only be trusted if the staged curve shows smooth degradation.

