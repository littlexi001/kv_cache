# MoR-KV Natural Holdout and Routing Audit

Date: 2026-07-11

## Executive finding

The natural-data evidence supports **operator complementarity**, but it does not yet support a deployable learned router.

On the first 64-query LongBench sample, adding a cross-dataset specialist-head operator increased per-query oracle headroom from `0.187` to `0.402` answer NLL. A fixed `27 deep-QK + 12 specialist` portfolio had an exploratory mean NLL of `3.258`, slightly below the strongest single deep-QK operator at `3.278`.

That small portfolio gain did **not** replicate on a query-disjoint holdout. On 64 new queries, frozen before evaluation, deep-QK reached `3.147`, while the frozen portfolio reached `3.239`. The specialist remains genuinely complementary—it beats deep-QK on 45.3% of holdout queries and the four-action oracle reaches `2.799`—but its tail regressions dominate the mean. Every tested answer-free risk signal either regresses or safely falls back to deep-QK on every query.

This is a No-Go for the current hard-routing/static-quota formulation, not a No-Go for head-conditioned KV retrieval. The next method must learn a causal per-head distortion target rather than infer downstream regret from task identity, generic query text, or retrieval-score confidence alone.

## 1. First natural sample: expanded operator library

All methods use exactly 39 blocks and 9,984 context tokens.

### Candidate means

| Operator | Mean answer NLL |
| --- | ---: |
| deep question-likelihood + QK | **3.278** |
| risk BM25 + QK | 3.344 |
| semantic/RRF record routes | 3.383–3.390 |
| hybrid record39 + QK | 3.412 |
| BM25 record39 | 3.550 |
| LODO specialist hybrid | 3.513 |

The specialist operator is compiled with nested leave-one-dataset-out calibration. Query heads sharing a KV head are deduplicated before ranking.

### Complementarity

With BM25, hybrid, risk, deep, and three semantic record routes, the per-query oracle mean is `3.092`, giving `0.187` NLL headroom over deep-QK. Adding the specialist operator lowers oracle NLL to `2.877`, increasing headroom to `0.402`.

Oracle action counts among the eight candidates are:

| Action | Queries |
| --- | ---: |
| specialist | 22 |
| BM25 | 22 |
| hybrid | 12 |
| deep | 4 |
| semantic route | 3 |
| risk | 1 |

The specialist is therefore not a duplicate of deep-QK or BM25, even though its global mean is weaker.

### Router failures

- A 121-feature query-text and retrieval-disagreement ridge router obtains `3.498`, versus deep-QK `3.278`.
- Full-model answer-free question likelihood/entropy routing obtains approximately `3.281–3.285`; it matches but does not beat deep-QK and costs about `0.88 s` per candidate-query.
- Two-fold fixed-portfolio selection overfits: out-of-fold mean is `3.426`, versus deep-QK `3.278`.

## 2. Exploratory static portfolios

Fixed portfolios reserve a quota for each operator and deduplicate to the same 39 physical blocks.

| Portfolio | Mean NLL on first sample |
| --- | ---: |
| deep35 + specialist4 | 3.389 |
| deep31 + specialist8 | 3.299 |
| **deep27 + specialist12** | **3.258** |
| deep23 + specialist16 | 3.406 |
| deep19 + specialist20 | 3.510 |
| deep31 + BM25-8 control | 3.358 |

The `deep27 + specialist12` result is exploratory because the quota grid was inspected on the same 64-query sample. It is not used as confirmation evidence.

## 3. Query-disjoint frozen holdout

### Construction and integrity

A second 64-query corpus view was generated from the same 535 eligible real LongBench records while excluding every calibration `record_uid`.

- calibration queries: 64;
- holdout queries: 64;
- `record_uid` overlap: **0**;
- `blocks.npy`, `blocks.jsonl`, and `records.jsonl`: identical SHA-256 between calibration and holdout views;
- K-SVD bases and the 134GB all-head K index: frozen and reused;
- only query profiles and per-head Top-16 rankings were recomputed;
- specialist heads, operator action, and `deep27 + specialist12` quota were frozen before answer NLL evaluation.

The holdout contains 16 `2wikimqa`, 16 `hotpotqa`, 16 `musique`, 15 `qasper`, and 1 `multifieldqa_en` query.

### Frozen policies

The specialist compiler selects four GQA-deduplicated query heads from calibration only:

```text
L11/H3, L14/H15, L21/H8, L16/H15
```

Its nested-calibration action is:

```text
head_count=4, depth=2, aggregation=minority_max, BM25 quota=32
```

### Held-out answer NLL

| Frozen action | Mean NLL | Delta vs deep | Query-bootstrap 95% CI |
| --- | ---: | ---: | ---: |
| deep-QK | **3.147** | 0.000 | `[0, 0]` |
| BM25 | 3.239 | +0.092 | `[-0.181, +0.382]` |
| specialist | 3.313 | +0.167 | `[-0.105, +0.444]` |
| deep27 + specialist12 | 3.239 | +0.092 | `[-0.055, +0.249]` |

The fixed portfolio does not replicate the first-sample gain. It loses on average and therefore cannot be a headline result.

### Complementarity remains

The negative mean is caused by tail loss rather than universal inferiority:

- specialist beats deep-QK on `29/64 = 45.3%` of queries;
- fixed portfolio beats deep-QK on `25/64 = 39.1%`;
- four-action per-query oracle mean is `2.799`;
- oracle headroom over deep-QK is `0.347`.

This is the central natural-data mechanism result: useful specialist actions exist frequently, but errors are asymmetric and expensive.

## 4. Frozen risk-gate audit

All gates are trained on the first 64 queries and applied without target labels to the zero-overlap holdout.

### Generic answer-free proxy

Inputs are per-action question NLL, question last-token NLL, answer-prefix entropy, and next-token logit margin.

- calibration selects an entropy gate with a conservative fallback;
- target switches: 4/64, all from deep to specialist;
- target mean: `3.166` versus deep `3.147`;
- mean delta: `+0.0195`, query CI `[-0.009, +0.062]`;
- action-wise regret model selects deep on all 64 queries.

### Head-confidence gate

The 61 features include specialist-head QK scores and margins, score entropy, cross-head agreement, query-text statistics, specialist/deep block overlap, and policy size. Nested calibration selects a conservative policy with zero switches. The frozen target result exactly equals deep-QK (`3.147`). This is safe but obtains none of the `0.347` oracle headroom.

## 5. What has been falsified

The evidence rejects the following versions of the paper claim:

1. a dataset-level operator selector is sufficient;
2. generic query text and retrieval disagreement predict natural operator regret;
3. question likelihood or answer-prefix entropy is a reliable self-verification target;
4. a fixed specialist quota transfers to new queries;
5. retrieval recall or specialist-head score margin can stand in for downstream loss;
6. a small first-sample NLL improvement is enough evidence for a paper contribution.

## 6. Revised method direction

The next router should be trained on a **causal per-head distortion teacher**:

```text
d_lh(a, q) = ||o_full_lh(q) - o_a_lh(q)||
```

or on exact omitted attention mass for the candidate operator. This target is local to the head, directly linked to the attention-output error bound, and much denser than one answer NLL label per query.

The revised runtime is:

```text
static head-function prior
  + query/head score signature
  -> predict per-head operator distortion and uncertainty
  -> choose the cheapest operator satisfying a distortion/CVaR constraint
  -> GQA physical block union
  -> sparse attention
```

The primary learning objective should be constrained risk rather than mean action classification:

```text
minimize physical KV bytes
subject to E[distortion] <= epsilon_mean
        and CVaR_0.95(distortion) <= epsilon_tail
```

## 7. Evidence gate after this audit

The project remains active, but the current implementation is not ICLR-ready. The next Go decision requires:

1. exact per-head attention-output distortion labels on at least 1,000 natural calibration queries;
2. a query-disjoint router that switches away from the strongest global operator while controlling p95/p99 regret;
3. replication on at least two 8B GQA models;
4. physical GQA block union and a real sparse-attention kernel;
5. quality/throughput gains over PolyKV, CompilerKV, DuoAttention, and strong single-policy retrieval at equal total bytes.

