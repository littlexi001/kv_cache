# Causal Per-Head Distortion Teacher and Conformal Router

Date: 2026-07-11

## 1. Why this experiment exists

The natural holdout showed substantial operator complementarity but no reliable query-level action signal. Retrieval recall, task labels, generic query features, question likelihood, answer-prefix entropy, and specialist QK margins all failed to predict downstream tail regret.

This experiment replaces the sparse answer-NLL target with a dense causal teacher for every `(query, layer, query_head, operator)`:

```text
d_lh(a, q) = ||o_full_lh(q) - o_a_lh(q)|| / ||o_full_lh(q)||
```

It also records exact omitted attention mass. The label is computed from post-RoPE Q/K/V and the actual value vectors, so it measures the head output directly instead of treating block recall as a proxy.

## 2. Setup

| Item | Setting |
| --- | --- |
| model | Qwen3-0.6B |
| queries | 64 real LongBench queries |
| layers / query heads | 28 / 16 = 448 heads |
| context | up to 4,096 tokens |
| token block | 256 tokens |
| full context | mean 15.36 blocks |
| sparse remote budget | 8 blocks |
| streaming budget | 2 blocks: sink + recent |
| GQA | 2 query heads / physical KV head |
| query positions | final question-content token |
| labels | 172,032 exact head/action rows |

Candidate actions:

1. `full` numerical control;
2. `streaming` sink + recent;
3. `uniform` block sampling;
4. `lexical_blocks` question-token overlap;
5. `qk_top_blocks` post-RoPE QK block score;
6. `mass_oracle_blocks` exact attention-mass oracle, used only for diagnosis.

The teacher runs one ordinary SDPA forward. Each attention layer is wrapped only long enough to recompute the selected query positions from its Q/K/V projections; activations are released layer by layer.

## 3. Numerical validation

| Action | Mean blocks | Mean omitted mass | Mean relative output L2 | p95 relative L2 | Mean cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 15.36 | 0.000 | `4.7e-8` | `2.0e-7` | 1.0000 |
| mass oracle | 8 | 0.0272 | 0.0381 | 0.1382 | 0.9984 |
| QK top-blocks | 8 | 0.0305 | 0.0417 | 0.1502 | 0.9981 |
| lexical | 8 | 0.0612 | 0.0946 | 0.3284 | 0.9898 |
| uniform | 8 | 0.0701 | 0.1056 | 0.3551 | 0.9881 |
| streaming | 2 | 0.1206 | 0.1983 | 0.6629 | 0.9571 |

The full control is numerically zero. The omitted-mass output bound is satisfied for all sparse actions. Exact QK retrieval closely approaches the mass oracle on average, but its tail remains too large for a single-policy solution.

## 4. Exact risk-constrained oracle compilation

For each head/query, choose the cheapest deployable action whose exact relative output L2 does not exceed the threshold; otherwise choose full.

| Error threshold | Mean blocks | p95 error | Full | QK | Streaming | Lexical | Uniform |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 10.74 | 0.0172 | 14,324 | 8,794 | 4,968 | 564 | 22 |
| 0.05 | **8.28** | **0.0447** | 8,630 | 9,657 | 9,726 | 590 | 69 |
| 0.10 | 6.15 | 0.0889 | 3,649 | 10,685 | 13,591 | 621 | 126 |

At threshold 0.05, exact routing reduces logical blocks by about 46% relative to full while satisfying the threshold for every compiled sample.

Cross-input action stability is neither zero nor perfect:

- mean per-head majority-action agreement: `0.762`;
- heads with at least 80% agreement: `193/448`.

This directly supports the hierarchical claim: heads have useful stable priors, but many require query-conditioned activation.

## 5. Query-disjoint learned router

Queries are separated before fitting:

- fit: 17 queries;
- conformal calibration: 16 queries;
- test: 31 queries, 13,888 head samples.

The router sees no full attention and no teacher label at test time. Its 462 inputs contain:

- one-hot `(layer, query_head)` prior;
- QK block-score top value, margins, spread, and entropy;
- lexical block-score top value, margins, spread, entropy, and nonzero fraction;
- QK/lexical Top-k disagreement.

Separate ridge regressors predict distortion for streaming, lexical, uniform, and QK. A head-local one-sided conformal correction produces an upper error bound. The cheapest operator whose upper bound is below 0.05 is selected; otherwise the router falls back to full.

### Risk/compute curve

| Router | Logical blocks | Physical GQA blocks | Physical saving | p95 error | Violation rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| global conformal 95% | 14.35 | not measured | — | 0.0020 | 0.00% |
| head-local conformal 95% | 11.53 | not measured | — | 0.0208 | 0.44% |
| **head-local conformal 90%** | **10.95** | **12.77** | **16.46%** | **0.0275** | **0.86%** |
| head-local conformal 80% | 10.15 | not measured | — | 0.0375 | 2.08% |
| static head prior | 11.07 | 12.94 | 15.37% | 0.0294 | 1.18% |
| test oracle | 8.24 | 10.38 | 32.12% | 0.0445 | 0.00% |
| full | 15.29 | 15.29 | 0.0% | `2e-7` | 0.0% |

The 90% head-local router is the first positive query-disjoint routing result: it is slightly cheaper than the static head prior and has lower mean error, p95 error, and violation rate. GQA union reduces the nominal logical saving, but the learned policy still preserves 16.46% physical KV-block savings and remains better than the static prior's 15.37%.

### Fixed-operator controls on the same test heads

| Fixed operator | Logical blocks | Physical GQA blocks | p95 error | Violation rate at 0.05 |
| --- | ---: | ---: | ---: | ---: |
| full | 15.29 | 15.29 | `2.0e-7` | 0.0% |
| QK top-blocks | 8 | 9.72 | 0.1490 | 29.5% |
| lexical | 8 | 8.00 | 0.3124 | 48.8% |
| uniform | 8 | 8.00 | 0.3437 | 51.3% |
| streaming | 2 | 2.00 | 0.6579 | 65.9% |

No fixed sparse operator satisfies the risk target. This closes the central operator-identity ablation at the head-output level.

### Adjacent-position stability audit

To test whether the result is an artifact of the final question token, we generated a second exact teacher on 16 query-disjoint natural examples using each example's final four question-content positions. This gives another 172,032 head/action labels.

| Error threshold | Mean blocks | p95 error | Within-query majority agreement | Adjacent-position agreement |
| ---: | ---: | ---: | ---: | ---: |
| 0.02 | 11.33 | 0.0170 | 82.17% | 71.95% |
| **0.05** | **8.97** | **0.0448** | **78.69%** | **66.90%** |
| 0.10 | 6.72 | 0.0899 | 79.84% | 68.53% |

Thus a head's operator preference is structured but not invariant. At the main 0.05 threshold, a dominant action explains about 79% of the four positions within a query, while about one third of adjacent position transitions change the exact cheapest safe action. This is the empirical reason to combine a stable head prior with query-position-conditioned routing rather than assigning one permanent operator to each head.

### End-to-end causal NLL reference

We next intervene in the actual model forward pass. At every layer, only the final prompt token's concatenated per-head attention output is replaced by the sparse output before the ordinary output projection, residual connection, MLP, and later layers. All earlier token states remain unchanged. The gold continuation follows the established protocol: the prompt ends in `Answer:` and the target begins with one space. This gives a causal first-answer-token NLL measurement, although it is still a correctness reference rather than a speed kernel.

On the router's strict 31-query test split at 4K context:

| Policy | Mean blocks | Mean head error | Head violation rate | Mean delta NLL vs full | 95% paired CI | p95 absolute delta NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **learned conformal 5%** | **10.81** | **0.0080** | **3.20%** | **+0.0642** | **[0.0394, 0.0900]** | **0.187** |
| exact 5% risk oracle | 8.05 | 0.0153 | 0.0% | +0.0679 | [0.0358, 0.0994] | 0.210 |
| fixed QK-8 | 8.00 | 0.0428 | 30.8% | -0.0044 | [-0.0636, 0.0582] | 0.369 |
| fixed lexical-8 | 8.00 | 0.0938 | 47.9% | +0.338 | [-0.0069, 0.860] | 1.256 |
| fixed uniform-8 | 8.00 | 0.1006 | 51.1% | +0.302 | [-0.0767, 0.834] | 1.565 |

The all-64 exact-oracle audit gives the same qualitative result: 8.10/15.36 blocks, zero head-level threshold violations, but a small `+0.0746` mean NLL change with 95% CI `[0.0458, 0.1041]` and p95 absolute change `0.242`.

Therefore 5% per-layer head-output error is not lossless. It does, however, produce a much smaller NLL tail than unsafe fixed policies: on test31 the learned route halves QK-8's p95 absolute NLL change while exposing only about 3% rather than 31% of head decisions above the threshold.

### Learned risk/quality Pareto

Changing only the deployment threshold of the frozen conformal bundle gives a monotone test31 curve:

| Error threshold | Logical blocks | Physical GQA saving | Mean delta NLL | 95% paired CI | p95 absolute delta NLL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 13.67 | 4.18% | +0.0259 | [0.0159, 0.0360] | 0.068 |
| 0.02 | 12.96 | 6.92% | +0.0381 | [0.0228, 0.0534] | 0.110 |
| 0.03 | 12.13 | 10.68% | +0.0494 | [0.0315, 0.0680] | 0.136 |
| 0.05 | 10.81 | 17.45% | +0.0639 | [0.0393, 0.0896] | 0.183 |

The curve is well behaved, but no tested nonzero-compression point is statistically lossless. Even the exact 1% local oracle has a positive mean NLL shift, showing that independently bounded head errors accumulate across 28 layers. This motivates a stronger cross-layer risk-propagation objective instead of treating every `(layer, head)` constraint independently.

### Cross-layer amplification audit

Using the frozen epsilon=0.02 route in only one seven-layer group at a time reveals highly nonuniform downstream sensitivity:

| Sparse layers | Logical saving | Physical GQA saving | Mean delta NLL | 95% paired CI | p95 absolute delta | Delta NLL per 1% physical saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **0-6** | **4.78%** | **2.34%** | **-0.0050** | **[-0.0104, -0.0008]** | 0.021 | **-0.0021** |
| 21-27 | 3.29% | 1.35% | +0.0047 | [0.0003, 0.0091] | 0.027 | 0.0035 |
| 7-13 | 3.69% | 1.95% | +0.0067 | [0.0033, 0.0103] | 0.021 | 0.0035 |
| **14-20** | **3.47%** | **1.28%** | **+0.0319** | **[0.0215, 0.0430]** | **0.083** | **0.0249** |

Layers 14-20 are roughly seven times more damaging per percentage point of physical saving than the other positive-cost groups, while layers 0-6 are benign in this audit. This is direct evidence for allocating risk by downstream amplification rather than by a uniform layer/head threshold. The ranking is empirical and must be learned on calibration queries, then frozen before the test set.

An exploratory cumulative allocation gives the first dominating point:

| Epsilon=0.02 sparse layers | Physical GQA saving | Mean delta NLL | 95% paired CI | p95 absolute delta |
| --- | ---: | ---: | ---: | ---: |
| 0-6 | 2.34% | -0.0050 | [-0.0104, -0.0008] | 0.021 |
| 0-6 and 21-27 | 3.70% | +0.0013 | [-0.0052, 0.0068] | 0.031 |
| **0-13 and 21-27** | **5.63%** | **+0.0084** | **[-0.0002, 0.0157]** | **0.041** |
| uniform epsilon=0.01, all layers | 4.18% | +0.0259 | [0.0159, 0.0360] | 0.068 |

The amplification-aware point saves more physical KV than the stricter uniform route while reducing both mean and tail NLL drift. However, the layer ranking and cumulative subsets were inspected on the same test31 queries, so this is post-hoc mechanism evidence, not a valid headline result. It must be selected on calibration data and confirmed on a new zero-overlap holdout.

### Frozen external holdout validation

We then built a second 64-query natural holdout that excludes all 128 `record_uid`s from the original clean64 and holdout1 sets. Pairwise record overlap is zero. Its corpus hash differs from the earlier build because the corpus construction/filtering code evolved, so this is treated as an external distribution test rather than an index-reuse view. The router bundle and the decision to keep layers 14-20 full were frozen before this run.

| Frozen policy | Physical GQA saving | Mean delta NLL | 95% paired CI | p95 absolute delta |
| --- | ---: | ---: | ---: | ---: |
| uniform epsilon=0.01, all layers | 4.55% | +0.0368 | [0.0249, 0.0496] | 0.144 |
| uniform epsilon=0.02, all layers | 7.56% | +0.0500 | [0.0348, 0.0668] | 0.184 |
| **allocated epsilon=0.02, layers 0-13 and 21-27** | **6.19%** | **+0.0106** | **[0.0031, 0.0187]** | **0.056** |
| allocated epsilon=0.03, layers 0-13 and 21-27 | 8.97% | +0.0183 | [0.0084, 0.0288] | 0.119 |
| **allocated epsilon=0.05, layers 0-13 and 21-27** | **14.07%** | **+0.0363** | **[0.0198, 0.0541]** | **0.128** |

The allocated policy saves more physical KV than uniform epsilon=0.01 while reducing mean NLL drift by about 71% and p95 drift by about 61%. It also recovers most of uniform epsilon=0.02's memory saving while reducing mean drift by about 79%. The residual mean shift is still statistically positive, so the valid claim is a dominating risk/quality trade-off, not lossless compression.

The higher-compression allocated epsilon=0.05 point is also notable at the first token: it triples uniform epsilon=0.01's physical saving while matching its mean NLL drift and lowering p95 drift, and it dominates uniform epsilon=0.02 in both memory and quality. Complete-answer evaluation below shows that this aggressive setting does not preserve the same tail-quality dominance.

Scoring the first four gold answer tokens on the same external holdout gives `6.27%` physical saving, mean delta NLL `+0.00563`, 95% CI `[-0.00113, 0.01252]`, and p95 absolute delta `0.0657`. The four-token mean is not detectably different from full attention.

Across the complete first reference answer, the allocated policy saves `6.32%` physical KV with mean delta NLL `+0.01334`, 95% CI `[0.00103, 0.02890]`, median `+0.00193`, and p95 absolute delta `0.0892`. Thus full-answer loss is small but statistically measurable and driven by a minority tail; the correct claim remains risk/quality dominance, not losslessness.

The complete-answer uniform epsilon=0.01 control saves only `4.97%` physical KV and has mean delta NLL `+0.02279` (95% CI `[0.01023, 0.03828]`), median `+0.01168`, and p95 `0.1120`. At higher saving, cross-layer allocation reduces mean degradation by about 41%, median degradation by about 83%, and p95 degradation by about 20%. The dominance therefore persists beyond the first token.

The intermediate allocated epsilon=0.03 point was rerun after enforcing the mathematically required non-negative floor on ridge-plus-conformal error bounds. It saves `8.89%` and has mean complete-answer delta NLL `+0.02598` (95% CI `[0.01102, 0.04397]`), median `+0.00812`, and p95 `0.1526`. The floor leaves the query-disjoint head-level router summary unchanged but alters a small number of equal-budget action tie breaks in the end-to-end route; these corrected numbers supersede the earlier `+0.02308` intermediate result. The aggressive allocated epsilon=0.05 point does not preserve full-answer dominance: it saves `13.85%` but has mean delta NLL `+0.0436`, median `+0.0097`, and p95 `0.269`. Both remain capacity points; allocated epsilon=0.02 is the default complete-answer quality point.

### Deployable query-risk gate audit

The corrected epsilon=0.03 complete-answer run also records the selected action's mean, p95, and maximum conformal upper bound, plus the fraction of decisions near the deployment threshold. None is a reliable query-level tail predictor on the 64-query external set: absolute-delta Pearson correlations are `-0.043`, `-0.120`, `0.064`, and `-0.132`; absolute-delta Spearman correlations are `0.166`, `0.004`, `-0.100`, and `-0.017`. Weighting every layer's upper bound by calibration-derived downstream amplification remains weak (`-0.048` Pearson, `0.172` Spearman with absolute answer-NLL change). Sending the highest indicated-risk queries back to full attention can improve the observed curve post hoc, but the rank correlations are too weak to claim a deployable gate without a new holdout. This is a useful negative result: per-head conformal calibration, even with static layer amplification, is not a sequence-level answer-loss certificate.

The physical-layout audit is also stricter than the per-KV-head byte count: at this 4K point, the union across all physical KV heads in a layer still retains essentially all `15.47/15.47` blocks (mean `3959.7` tokens). Thus the reported `8.89%` saving requires a backend that supports different block sets per KV head. A single shared layer-level token layout gets no meaningful compression here.

### Physical GQA execution timing

A real reduced-compute reference takes the block union for each physical KV head, gathers only those K/V tokens, and runs one PyTorch SDPA call for the two query heads sharing that KV head. Timing includes gather and eight group launches:

| Full KV length | Active tokens | Dense SDPA | 8-call grouped | 1-call padded-ragged | Batched speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 512-3,072 | 0.128 ms | 0.701-0.703 ms | 0.431-0.437 ms | **0.29-0.30x** |
| 16,384 | 512-3,072 | 0.511 ms | 0.704-0.754 ms | 0.431-0.436 ms | **1.17-1.18x** |
| 32,768 | 512-3,072 | 1.011 ms | 0.699-0.754 ms | 0.431-0.676 ms | **1.50-2.34x** |

Both paths are numerically equivalent when all blocks are retained and truly avoid unselected K/V computation. Packing physical groups into one padded-ragged gather and one SDPA call moves the crossover from 32K to 16K, but 4K remains about 3.4x slower than dense. These are fixed 512-3,072-active-token attention-subsystem measurements; they do not imply a speedup for the current external policy's modest 6% physical saving. A fused page kernel remains necessary for a 4K claim.

At a policy-matched 93.75% KV retention, there is no useful batched-ragged speedup: even with a packed union reused for 64 steps, speed is `0.26x` at 4K, `0.99x` at 16K, and `0.49x` at 32K. The 32K group-loop reference reaches `1.23x`, but its layout is not competitive at shorter lengths. Consequently the current 6% memory point is a capacity result only; a latency claim requires substantially stronger safe compression or a fused kernel that avoids padded K/V replication.

### Qwen3-8B all-layer transfer pilot

The teacher now supports Accelerate `device_map=balanced`. Qwen3-8B (36 layers, 32 query heads, 8 KV heads, GQA group size 4) loads across two 24GB GPUs and completes exact post-RoPE capture. On one 2K query at layer 0 with a 4/8-block budget:

| Action | Mean relative head-output L2 |
| --- | ---: |
| full | `2.0e-8` |
| mass oracle-4 | 0.0622 |
| QK-4 | 0.0644 |
| lexical-4 | 0.1087 |
| uniform-4 | 0.1100 |
| streaming-2 | 0.1925 |

The numerical control and operator ordering transfer. The subsequent eight-query all-layer pilot covers all 36 layers and 1,152 query heads (55,296 action rows). Across 9,216 head/query decisions, mean relative output L2 is `0.0857` for the fixed-budget mass oracle, `0.0931` for QK, `0.1350` for lexical, `0.1415` for uniform, and `0.2227` for streaming. At exact epsilon=0.05, the cheapest feasible operator uses `5.839/7.875` logical blocks, has p95 relative error `0.0449`, and has mean cross-query head-action agreement `78.98%`; `606/1,152` heads keep the same action on at least 80% of queries. The action mix includes full, QK, streaming, lexical, and uniform, so operator complementarity and context-dependent activation both transfer beyond the 0.6B model.

Qwen3-8B has GQA group size four. Exact query-head actions at epsilon=0.05 become `7.324/7.875` physical blocks after union, only `6.99%` physical saving; epsilon=0.02 and 0.10 yield `1.21%` and `19.11%`. This is an oracle capacity statistic, not a learned-router or downstream-quality result. It exposes a central scaling constraint: larger GQA groups erase much of the logical gain unless routing is trained and optimized directly at the physical KV-head union.

A 32-query all-layer teacher run is in progress to enable a nondegenerate query-disjoint fit/conformal/test router split. No 8B downstream-NLL or speed claim is made yet.

## 6. What is proven and what is not

Proven in this pilot:

1. different heads/queries require different KV retrieval operators under an exact causal metric;
2. a static head prior is useful but incomplete;
3. query-conditioned score signatures improve the risk/compute trade-off over the static prior;
4. head-local uncertainty calibration is much less conservative than a global gate;
5. fixed streaming, lexical, uniform, and QK policies have unacceptable tail distortion at the same sparse budget.
6. query-head decisions retain a measurable 16.46% physical saving after GQA union.
7. operator identity is mostly, but not perfectly, stable across four adjacent query positions.
8. per-layer head-output risk control predicts and reduces the downstream NLL tail, but the 5% operating point has a small measurable mean loss.

Not yet proven:

1. a stricter deployable distortion route preserves multi-token/full-answer NLL and generation quality tightly enough at scale;
2. the result transfers beyond four adjacent token positions, 4K context, Qwen3-0.6B, and the current query sets;
3. a real paged-KV sparse kernel produces a speedup after index/router overhead;
4. the method beats strong current baselines on 8B models.

## 7. Revised paper contribution

The strongest current formulation is no longer generic “mixture of retrievers.” It is:

> **Causal, risk-constrained operator routing for function-specialized KV heads:** use exact per-head attention-output distortion as a dense teacher, predict operator-specific distortion from low-cost score signatures, calibrate uncertainty per head, and project the actions through a physical GQA block union.

This formulation is more distinct from budget-only head methods and layer-wise method selection because the routed variable is operator identity, the training target is causal output distortion, and the safety objective explicitly controls tail error.

The end-to-end audit adds an important correction: the final method should allocate a *global propagated risk budget* across layers, because a uniform local threshold does not control cumulative logit drift. A paper claim based only on independent per-head conformal bounds would now be too weak.
