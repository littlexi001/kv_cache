# Compute-Quality Search Log

Goal: find an attention/KV-cache selection method that reduces exact QK work while keeping PPL close to baseline.

Current local benchmark:

- Model: `ymluo/models/Qwen3-0.6B`
- Text: `external/needle-in-a-haystack/needlehaystack/PaulGrahamEssays/worked.txt`
- Token split: `prefill_tokens=1536`, `eval_tokens=512`, total `2048`
- Metric: next-token PPL over the evaluation tokens
- Protection for diagnostic runs unless stated otherwise: `sink=0`, `recent=0`, `always_keep_self=true`

Success definition for the fast local benchmark:

- Quality: PPL should be close to baseline. A useful local target is within about 5% of baseline PPL.
- Compute: the method must have a clear path to reducing exact QK dot products. A quality-only mask that requires full QK over all history is not enough.

Known results before this log:

| Mode | Exact QK reduction path | PPL |
|---|---:|---:|
| baseline | none | 25.5983 |
| true top2 | none in oracle form; useful as an upper-bound quality target | 25.1006 |
| true top2union | none in oracle form; useful as an upper-bound quality target | 24.8161 |
| signxnor2 | replaces exact QK ranking with sign popcount | 5913.4193 |
| signxnor5 | replaces exact QK ranking with sign popcount | 1291.8312 |
| signxnor10 | replaces exact QK ranking with sign popcount | 601.3069 |
| signxnor20 | replaces exact QK ranking with sign popcount | 193.3937 |
| signxnorknorm2 | sign popcount multiplied by cached `||K||_2` | 11282399.5884 |
| signxnorknorm5 | sign popcount multiplied by cached `||K||_2` | 9932591.1115 |
| signxnorknorm10 | sign popcount multiplied by cached `||K||_2` | 29964.5058 |
| signxnorknorm20 | sign popcount multiplied by cached `||K||_2` | 3538.3756 |

Interpretation:

- Pure sign-XNOR candidate selection is cheap but misses too many high-value tokens.
- Multiplying popcount by whole-key norm makes the ranking worse. It rewards large-norm keys even when their signed direction is wrong for the current query.
- The next test should not directly replace true QK with sign popcount. It should use sign popcount only as a cheap candidate generator, then compute exact QK on a smaller candidate set.

## Iteration 1: Sign-XNOR Candidate Then Exact QK Rerank

Conjecture: sign-XNOR may be good enough for coarse recall if the candidate fraction is larger than 20%, and exact QK reranking inside the candidate set may recover most of the true top2 quality while reducing exact QK dot products.

Physical prior:

- Neighboring key vectors whose signs agree with the query on many dimensions are more likely to have large positive QK scores.
- Popcount alone is too coarse, but it may still remove clearly bad keys.

Mathematical model:

- For query `q_h` and historical key `k_{h,t}`, compute `s_{h,t} = count_i(sign(q_{h,i}) == sign(k_{h,t,i}))`.
- Keep all historical keys whose `s_{h,t}` is in the top `candidate_fraction` bucket for that head. Boundary ties are kept.
- Within those candidates, compute exact QK score and keep the top `top_fraction=0.02` historical keys per head.

Implementation contract:

- Modes will be named `signxnor{C}rerank`, where `C` is the candidate percentage.
- The quality evaluator may still compute full QK internally to simulate the mask, but the method's intended exact-QK cost is approximately `candidate_fraction` of the full history plus the cheap sign-popcount pass.
- Pass condition: PPL is near baseline at a candidate fraction materially below 100%.
- Fail condition: PPL remains far above baseline even at large candidate fractions.

Planned fast parameters:

- Candidate fractions: `20%`, `40%`, `60%`, `80%`
- Compare to: `baseline`, `true top2`, `true top2union`

## Iteration 2: Fixed Recent/Sink Window

Conjecture: for next-token PPL on this local text, many important attention links may be local or sink-like. A fixed recent window can reduce exact QK work without needing a learned or query-dependent retriever.

Physical prior:

- Recent context often contains syntactic and semantic information needed for the next token.
- Early sink tokens may stabilize attention patterns.

Mathematical model:

- Keep the last `W` historical tokens for every head.
- Optionally keep the first `S` sink tokens for every head.
- Mask all other historical tokens.

Implementation contract:

- Modes will be named `recent{W}`.
- Global `--protect_sink_tokens S` can add sink tokens.
- Intended exact-QK cost is approximately `(W + S) / history_count`.
- Pass condition: a small `W` produces PPL close to baseline.
- Fail condition: PPL degrades sharply unless `W` is so large that compute reduction is weak.

## Results: Iteration 1 And 2 Local 2048

Artifact:

- `outputs/compute_quality_search_local_2048_iter1/ppl_by_mode.csv`
- `outputs/compute_quality_search_local_2048_iter1/limit_load_by_head.csv`
- `outputs/compute_quality_search_local_2048_iter2_recent128/ppl_by_mode.csv`
- `outputs/compute_quality_search_local_2048_iter2_recent256/ppl_by_mode.csv`

Setup:

- `prefill_tokens=1536`
- `eval_tokens=512`
- `chunk_size=64`
- `top_fraction=0.02`
- `always_keep_self=true`

Observed PPL:

| Mode | Protection | PPL | Interpretation |
|---|---:|---:|---|
| baseline | none | 25.5983 | Reference quality. |
| true top2 | none | 25.1006 | Oracle-quality target, but not a reduced-compute method because it needs full QK to find the top2 set. |
| true top2union | none | 24.8161 | Oracle-quality target with more kept tokens. |
| signxnor20rerank | none | 225.1836 | Candidate recall is too low. |
| signxnor40rerank | none | 88.5633 | Still too many important tokens missing. |
| signxnor60rerank | none | 45.7300 | Better but still far from baseline. |
| signxnor80rerank | none | 30.8503 | Close enough to be informative, but candidate fraction is too high for strong compute savings and still worse than baseline. |
| recent128 | none | 1784.4238 | Local window alone fails badly on this example. |
| recent256 | none | 1017.6229 | Local window alone fails badly. |
| recent512 | none | 905.9035 | Local window alone still fails badly. |
| recent1024 | none | 888.2027 | Even a large local window fails, so important context is not only recent. |
| signxnor40rerank | recent128 | 65.2262 | Recent protection helps but not enough. |
| signxnor60rerank | recent128 | 38.7179 | Recent protection helps but not enough. |
| signxnor80rerank | recent128 | 30.3750 | Small improvement over no recent protection. |
| signxnor40rerank | recent256 | 61.9843 | Small improvement over recent128. |
| signxnor60rerank | recent256 | 38.7179 | No meaningful improvement over recent128. |
| signxnor80rerank | recent256 | 29.5367 | Best sign-XNOR result so far, still above baseline. |

Failure interpretation:

- Sign-XNOR is useful only as a very coarse filter. It must keep about 80% of history before quality becomes moderately close, which leaves little room for compute savings.
- Recent-window protection does not repair the main failure. The missed important tokens are not only in the nearest 128 or 256 positions.
- Fixed recent-only attention is not a viable method for this benchmark.

Conjecture update:

- A reduced-compute method needs a candidate score closer to the true dot product than sign agreement. It should still avoid full-dimensional QK over the whole history.

## Iteration 3: Query-Selected Partial QK Candidate Then Exact Rerank

Conjecture: the largest-magnitude dimensions of the current query carry most of the useful variation in QK ranking. Computing partial QK on only those dimensions can create a much better candidate set than sign-XNOR while reducing multiplications.

Physical prior:

- In a dot product, dimensions with larger `|q_i|` can change the score more.
- Ignoring small-`|q_i|` dimensions should preserve much of the ranking signal while reducing the cost of the coarse pass.

Mathematical model:

- For each query/head, choose `D` dimensions with largest `|q_i|`.
- For every historical key, compute `partial_score_t = sum_{i in topD(|q|)} q_i k_{t,i}`.
- Keep the top `candidate_fraction` historical tokens by `partial_score_t`.
- Compute exact full-dimensional QK only inside those candidates and keep final true top2 within that candidate set.

Implementation contract:

- Modes are named `qabs{D}cand{C}rerank`.
- Example: `qabs32cand20rerank` means use 32 query-selected dimensions for the coarse pass, keep 20% candidates, then exact-rerank inside those candidates.
- Intended conservative multiplication proxy, if the exact rerank recomputes full QK on candidates: `D / head_dim + candidate_fraction`.
- If the partial score is reused during exact rerank, the proxy is slightly lower: `D / head_dim + candidate_fraction * (1 - D / head_dim)`.

## Results: Iteration 3 Local 2048

Artifact:

- `outputs/compute_quality_search_local_2048_iter3_qabs/ppl_by_mode.csv`

Observed PPL and conservative work proxy, using local Qwen3 head dimension `128`:

| Mode | PPL | Conservative QK work proxy |
|---|---:|---:|
| baseline | 25.5983 | 1.000 |
| true top2 | 25.1006 | 1.000 oracle |
| qabs16cand20rerank | 26.0668 | 0.325 |
| qabs16cand40rerank | 25.7902 | 0.525 |
| qabs16cand60rerank | 25.2984 | 0.725 |
| qabs32cand20rerank | 25.4608 | 0.450 |
| qabs32cand40rerank | 25.2481 | 0.650 |
| qabs32cand60rerank | 25.0913 | 0.850 |
| qabs64cand20rerank | 25.1246 | 0.700 |
| qabs64cand40rerank | 25.1788 | 0.900 |
| qabs64cand60rerank | 25.1615 | 1.100 |

Interpretation:

- This is the first tested family that satisfies the local fast benchmark.
- `qabs32cand20rerank` has PPL `25.4608`, slightly better than baseline `25.5983`, with a conservative QK work proxy of `0.45`.
- `qabs16cand20rerank` is more aggressive: PPL `26.0668`, about `1.8%` worse than baseline, with proxy `0.325`.
- `qabs16cand40rerank` is a balanced candidate: PPL `25.7902`, about `0.75%` worse than baseline, with proxy `0.525`.
- `qabs32cand40rerank` is safer: PPL `25.2481`, better than baseline and close to oracle top2, with proxy `0.65`.

Current best candidates for server-scale tests:

1. `qabs32cand20rerank`: best quality/compute tradeoff in local 2048.
2. `qabs16cand40rerank`: lower compute than `qabs32cand20rerank` if a small PPL increase is acceptable.
3. `qabs32cand40rerank`: safer quality candidate for long-context validation.

Claim boundary:

- These are local 2048-token results on one text prefix and Qwen3-0.6B.
- The evaluator still computes full QK internally to simulate masks, so wall-clock time is not the real optimized runtime.
- The compute proxy is a multiplication-count estimate for an optimized implementation, not measured kernel speed.
- The next required test is long-context validation at larger prefill lengths and a speed prototype that avoids full QK outside candidates.

## Results: Iteration 3 Top2 Overlap And Attention Mass

Artifact:

- `outputs/qabs_top2_overlap_local_2048/overlap_by_mode.csv`
- `outputs/qabs_top2_overlap_local_2048/overlap_by_layer.csv`
- `outputs/qabs_top2_overlap_local_2048/overlap_by_layer_head.csv`
- `outputs/qabs_top2_overlap_local_2048/per_query_overlap.csv`

Setup:

- Same local 2048 token split as the PPL run.
- `max_query_samples=32` sampled eval positions.
- Cases are sampled `query_token x layer x head`.
- `true_top_overlap` means: final qabs-rerank selected top2 tokens divided by true full-QK top2 tokens.
- `selected_attention_mass_mean` means: full-softmax attention mass on the final qabs-rerank selected tokens.
- `candidate_attention_mass_mean` means: full-softmax attention mass covered by all coarse candidates before exact rerank.
- `true_top_attention_mass_mean` means: full-softmax attention mass on the true full-QK top2 tokens.

Observed summary:

| Mode | True top2 overlap | Selected attention mass | Candidate attention mass | True top2 attention mass | Conservative work proxy |
|---|---:|---:|---:|---:|---:|
| qabs16cand20rerank | 0.9501 | 0.8453 | 0.9341 | 0.8517 | 0.325 |
| qabs16cand40rerank | 0.9901 | 0.8494 | 0.9721 | 0.8517 | 0.525 |
| qabs32cand20rerank | 0.9939 | 0.8511 | 0.9535 | 0.8517 | 0.450 |
| qabs32cand40rerank | 0.9993 | 0.8515 | 0.9815 | 0.8517 | 0.650 |

Interpretation:

- The PPL behavior matches the overlap/mass evidence.
- `qabs16cand20rerank` misses about `5%` of the true top2 tokens, but the selected tokens still capture `0.8453 / 0.8517` of the true-top2 attention mass. This explains why PPL only rises modestly.
- `qabs32cand20rerank` recovers about `99.39%` of true top2 tokens and almost the same attention mass as true top2. This explains why its PPL is close to or slightly better than baseline in the local run.
- `qabs32cand40rerank` is almost an oracle for true top2 on this sample, but costs more.

Conjecture update:

- Partial-QK candidate generation is currently the strongest direction.
- The next uncertainty is whether the same overlap and mass behavior holds at 10k, 20k, 40k, and longer context lengths.

## Results: Query Sparsity And Qabs Mechanism

Question:

- Does `qabs` work because the query vector is truly sparse, or because the query vector has concentrated magnitude in a small number of dimensions?

Artifact:

- `outputs/q_sparsity_qabs_local_2048/q_sparsity_summary.csv`
- `outputs/q_sparsity_qabs_local_2048/q_sparsity_by_layer.csv`
- `outputs/q_sparsity_qabs_local_2048/q_sparsity_by_layer_head.csv`
- `outputs/q_sparsity_qabs_local_2048/candidate_rule_comparison.csv`

Setup:

- Same local 2048 token split.
- `max_query_samples=32`.
- Cases are sampled `query_token x layer x head`, total `14336` cases.
- Query vectors are the attention `query_states` seen by Qwen3 eager attention, after the model's projection/RoPE path used in attention.

Query vector distribution:

| Metric | Value |
|---|---:|
| Fraction `abs(q_i) < 1e-8` | 0.000385 |
| Fraction `abs(q_i) < 1e-6` | 0.000385 |
| Fraction `abs(q_i) < 1e-4` | 0.000425 |
| Fraction `abs(q_i) < 1% of row max` | 0.0861 |
| Mean effective L2 dimension `(sum q_i^2)^2 / sum q_i^4` | 15.70 |
| Mean `max(abs(q)) / mean(abs(q))` | 8.39 |
| Top 8 dims L2 energy | 0.5886 |
| Top 16 dims L2 energy | 0.7346 |
| Top 32 dims L2 energy | 0.8723 |
| Top 64 dims L2 energy | 0.9722 |

Interpretation:

- Query vectors are not exactly sparse. Almost no dimensions are exactly zero or even below `1e-4`.
- Query vectors are strongly magnitude-concentrated. In a 128-dimensional head, the effective L2 dimension is about `15.7`.
- The largest 16 dimensions contain about `73.5%` of query L2 energy; the largest 32 contain about `87.2%`.

Dimension-selection control test:

| Rule | Dims | Candidate fraction | True top2 overlap | Candidate attention mass | Final selected attention mass |
|---|---:|---:|---:|---:|---:|
| qabs | 8 | 0.20 | 0.8399 | 0.8806 | 0.8093 |
| random | 8 | 0.20 | 0.4225 | 0.3988 | 0.3715 |
| qabs | 16 | 0.20 | 0.9501 | 0.9341 | 0.8453 |
| random | 16 | 0.20 | 0.5480 | 0.5290 | 0.4920 |
| qabs | 32 | 0.20 | 0.9939 | 0.9535 | 0.8511 |
| random | 32 | 0.20 | 0.7112 | 0.6816 | 0.6277 |
| qabs | 64 | 0.20 | 0.9999 | 0.9617 | 0.8517 |
| random | 64 | 0.20 | 0.9002 | 0.8398 | 0.7598 |

Interpretation:

- The improvement does not come merely from using fewer dimensions. Random dimensions with the same count are much worse.
- The improvement comes from choosing the dimensions with largest `|q_i|` for each query/head.
- The qabs dimensions work because they preserve most of the dot-product variation that matters for ranking keys.

Conjecture update:

- The correct statement is not "Q is sparse." The measured statement is: Q is dense but highly magnitude-concentrated.
- `qabs` works because the largest `|q_i|` dimensions dominate the QK score ranking, and because exact rerank fixes the remaining candidate-order errors.
- Next tests should check whether this magnitude concentration and qabs overlap remain stable at longer prefill lengths and on other texts.

## Results: Adjacent Decode-Step Stability

Question:

- Are the largest-`|q|` channels stable between adjacent decode steps?
- Are the true full-QK top2 tokens stable between adjacent decode steps?

Artifact:

- `outputs/adjacent_stability_local_2048/q_channel_adjacent_by_dim.csv`
- `outputs/adjacent_stability_local_2048/top2_token_adjacent_overall.csv`
- `outputs/adjacent_stability_local_2048/q_channel_adjacent_by_layer.csv`
- `outputs/adjacent_stability_local_2048/top2_token_adjacent_by_layer.csv`

Setup:

- Same local 2048 token split.
- `max_pair_samples=128` adjacent eval pairs.
- Cases are sampled `adjacent_pair x layer x head`, total `57344` cases.
- Adjacent pair `(t-1, t)` compares consecutive eval query positions.

Metrics:

- `overlap_fraction_current = |set_t intersect set_{t-1}| / |set_t|`.
- `jaccard = |set_t intersect set_{t-1}| / |set_t union set_{t-1}|`.
- For top2 token sets, `previous_set_current_attention_mass_mean` measures the current query's true attention mass on the previous query's top2 token set.

Q top-`|q|` channel stability:

| Top-D channels | Mean intersection | Overlap fraction | Jaccard |
|---:|---:|---:|---:|
| 8 | 4.89 / 8 | 0.6106 | 0.4395 |
| 16 | 9.58 / 16 | 0.5987 | 0.4272 |
| 32 | 19.56 / 32 | 0.6111 | 0.4400 |
| 64 | 43.97 / 64 | 0.6871 | 0.5233 |

True full-QK top2 token stability:

| Metric | Value |
|---|---:|
| Mean current top2 set size | 36.875 |
| Mean intersection with previous step | 20.952 |
| Overlap fraction | 0.5682 |
| Jaccard | 0.3968 |
| Current true top2 attention mass | 0.8514 |
| Current mass on previous step's top2 tokens | 0.7158 |
| Current mass on intersection tokens | 0.7047 |

Interpretation:

- The largest-`|q|` channels are moderately stable across adjacent decode steps. About `60%` of the top 8, 16, and 32 channels repeat. Top 64 repeats more because the set is larger.
- The true top2 token set is also moderately stable. About `56.8%` of the current top2 tokens were also in the previous step's top2 set.
- The previous step's top2 token set still captures substantial current attention mass: `0.7158` versus current true top2 mass `0.8514`.
- This means adjacent reuse is plausible, but direct reuse alone will lose important tokens. It should be used as a candidate source, not as the final selected set.

Conjecture update:

- A useful next method is a multi-source candidate set:
  1. current-step qabs partial-QK candidates;
  2. previous-step qabs dimensions or previous-step top token candidates;
  3. optional sink/recent tokens.
- The stability numbers suggest previous-step reuse can reduce recomputation or improve recall, but it cannot replace current-step scoring.

## Results: Qabs Adjacent Candidate Reuse

Question:

- Can adjacent-step reuse improve quality while keeping the current-step approximate QK candidate fraction smaller?
- Candidate formula:
  - current qabs partial-QK candidates;
  - previous step raw qabs partial-QK candidates;
  - previous step final top2-selected tokens;
  - optional sink/recent protected tokens;
  - exact full-QK rerank inside the union to final top2.

Implementation:

- Added mode pattern `qabs{D}cand{C}reuse`, for example `qabs32cand10reuse`.
- Added `--eval_chunk_size`. Use `--eval_chunk_size 1` for token-by-token decode-style evaluation while keeping prefill chunked.
- Reuse state is per `(layer, head)` and only applies when the previous query token is exactly `current_token - 1`.
- Important implementation fix: store the previous step's raw qabs candidate, not the previous step's full union. Storing the full union recursively snowballed the candidate set to roughly `0.78` to `0.85` of history after 512 eval steps, which defeated the compute goal.
- Protection semantics were changed to hard protect:
  - dynamic candidate union is `current_qabs ∪ previous_raw_qabs ∪ previous_final_top2`;
  - exact full-QK rerank selects top2 within that dynamic candidate union;
  - sink/recent tokens are added after rerank and are therefore guaranteed to be kept.
- Do not use `sink=1000,recent=1000` for local 2048-token experiments. It covers nearly the whole available history and invalidates the sparse-compute interpretation. Use a small local value such as `10/10`; reserve large values only for long-context runs where they do not dominate the full history.

Compute proxy:

- For `qabsD candC rerank`: conservative proxy is `D / 128 + C`.
- For `qabsD candC reuse`: conservative proxy is `D / 128 + measured_union_fraction`.
- A less conservative incremental proxy, if the partial dimensions are reused for rerank, is `D / 128 + measured_union_fraction * (1 - D / 128)`.
- Wall-clock time in this Python prototype is not the real kernel-time estimate; reuse modes are slower locally because they build masks and update per-head CPU state in Python.

Artifact:

- `outputs/compute_quality_search_local_2048_reuse_fixed_decode_iter1/ppl_by_mode.csv`
- `outputs/compute_quality_search_local_2048_reuse_fixed_decode_iter1/*_candidate_load_by_head.csv`

Setup:

- Local 2048-token run on `worked.txt`.
- `prefill_tokens=1536`, `eval_tokens=512`.
- `chunk_size=64`, `eval_chunk_size=1`.
- No sink/recent protection: `protect_sink_tokens=0`, `protect_recent_tokens=0`.
- `top_fraction=0.02`.

Results:

| Mode | PPL | Seconds | Candidate union fraction | Conservative work proxy | Incremental work proxy |
|---|---:|---:|---:|---:|---:|
| baseline | 25.5495 | 29.21 | - | 1.0000 | 1.0000 |
| qabs16cand20rerank | 26.2118 | 81.94 | 0.2000 fixed | 0.3250 | - |
| qabs32cand20rerank | 25.3408 | 82.19 | 0.2000 fixed | 0.4500 | - |
| qabs16cand10reuse | 25.1058 | 198.62 | 0.1503 | 0.2753 | 0.2565 |
| qabs16cand15reuse | 25.0413 | 172.26 | 0.2173 | 0.3423 | 0.3151 |
| qabs32cand10reuse | 24.9006 | 173.27 | 0.1462 | 0.3962 | 0.3597 |
| qabs32cand15reuse | 24.9324 | 171.99 | 0.2125 | 0.4625 | 0.4094 |

Interpretation:

- The fixed reuse method is promising. It gets PPL slightly better than baseline in this local 2048-token run while using a measured candidate union around `14.6%` to `21.7%`.
- `qabs16cand10reuse` is the best compute/quality point in this run: PPL `25.1058` with conservative proxy `0.2753`, better than `qabs16cand20rerank` in both quality and proxy cost.
- `qabs32cand10reuse` gives the best PPL among these modes: `24.9006`, with conservative proxy `0.3962`, still below `qabs32cand20rerank` proxy `0.4500`.
- The current local seconds are not favorable for reuse because this is an unoptimized prototype. A real implementation should avoid per-head Python loops and keep candidate masks/indices on device.

Next tests:

- Run the same fixed reuse modes at server lengths `10k/20k/40k/60k+` prefill with `eval_chunk_size=1`.
- Include sink/recent protection for the target long-context setting only when the protected span is small relative to history. Local default is now `sink=10`, `recent=10`.
- Add overlap/mass analysis for reuse modes: final selected tokens vs true full-QK top2, and candidate-union attention mass.

### Follow-up: Hard Protect With Local 10/10

Artifact:

- `outputs/compute_quality_search_local_2048_reuse_hardprotect10/ppl_by_mode.csv`
- `outputs/compute_quality_search_local_2048_reuse_hardprotect10/*_candidate_load_by_head.csv`

Setup:

- Same 2048-token local split.
- `prefill_tokens=1536`, `eval_tokens=512`.
- `eval_chunk_size=1`.
- `protect_sink_tokens=10`, `protect_recent_tokens=10`.
- Candidate statistics report the dynamic candidate union only; the hard-protected sink/recent tokens are added after rerank.

Results:

| Mode | PPL | Dynamic candidate union fraction | Conservative proxy |
|---|---:|---:|---:|
| baseline | 25.5495 | - | 1.0000 |
| qabs16cand10reuse | 25.0853 | 0.1563 | 0.2813 |
| qabs32cand10reuse | 24.8911 | 0.1521 | 0.4021 |

Interpretation:

- Hard-protecting 10 sink and 10 recent tokens does not wash out the sparse behavior on the local 2048-token test.
- `qabs16cand10reuse` remains the better compute/quality tradeoff; `qabs32cand10reuse` remains the better quality point.

### Follow-up: More Aggressive Candidate Fractions

Question:

- Can the dynamic candidate fraction be pushed below `10%` while keeping PPL near or better than baseline?

Artifact:

- `outputs/compute_quality_search_local_2048_reuse_aggressive_candidates/ppl_by_mode.csv`
- `outputs/compute_quality_search_local_2048_reuse_aggressive_candidates/*_candidate_load_by_head.csv`

Setup:

- Same 2048-token local split.
- `prefill_tokens=1536`, `eval_tokens=512`.
- `eval_chunk_size=1`.
- `protect_sink_tokens=10`, `protect_recent_tokens=10`.
- Swept `qabs{8,16,32}cand{3,5,7,10}reuse`.
- Candidate statistics report dynamic candidate union only; hard-protected sink/recent tokens are added after rerank.

Results:

| Mode | PPL | Dynamic candidate union fraction | Conservative proxy |
|---|---:|---:|---:|
| baseline | 25.5495 | - | 1.0000 |
| qabs8cand3reuse | 25.2084 | 0.0620 | 0.1245 |
| qabs8cand5reuse | 25.0223 | 0.0904 | 0.1529 |
| qabs8cand7reuse | 24.6731 | 0.1185 | 0.1810 |
| qabs16cand3reuse | 25.1123 | 0.0587 | 0.1837 |
| qabs16cand5reuse | 25.3559 | 0.0870 | 0.2120 |
| qabs16cand7reuse | 24.9001 | 0.1151 | 0.2401 |
| qabs16cand10reuse | 25.0853 | 0.1563 | 0.2813 |
| qabs32cand3reuse | 25.1780 | 0.0552 | 0.3052 |
| qabs32cand5reuse | 25.0690 | 0.0833 | 0.3333 |
| qabs32cand7reuse | 24.9230 | 0.1112 | 0.3612 |
| qabs32cand10reuse | 24.8911 | 0.1521 | 0.4021 |

Interpretation:

- The aggressive sweep is surprisingly strong on this local sample. Even `qabs8cand3reuse` keeps PPL better than baseline with a conservative proxy around `0.125`.
- `qabs8cand7reuse` is the best observed point in this single run: PPL `24.6731` with conservative proxy `0.1810`.
- Because this is one text segment and only 512 eval tokens, treat the exact ranking as noisy. The robust conclusion is that `cand3/cand5/cand7` are worth server-scale retesting.
- The next server sweep should prioritize `qabs8cand3reuse`, `qabs8cand5reuse`, `qabs8cand7reuse`, `qabs16cand3reuse`, `qabs16cand7reuse`, and keep `qabs16cand10reuse` / `qabs32cand10reuse` as anchors.

## Results: Qabs Reuse Fast Decode Prototype

Question:

- Can we remove the prototype's initial full-history QK matmul and measure a more realistic speed trend?

Implementation:

- Added `--qabs_fast_path`.
- Added `--disable_sparse_stats` to avoid per-token `.cpu()` synchronizations during timing.
- Added `--log_every` to reduce progress-print overhead when using `eval_chunk_size=1`.
- Added `scripts/run_qabs_fast_speed_server.sh` for server timing runs.

Fast-path algorithm:

1. Select top-`|q|` dimensions for each query/head.
2. Compute partial-QK over full history on only those dimensions.
3. Build dynamic candidate union from current raw qabs candidate, previous raw qabs candidate, and previous final top2.
4. Gather only candidate K vectors and compute exact full-QK for rerank.
5. Hard-protect sink/recent after rerank.
6. Gather final selected K/V vectors and compute attention only on those indices.

Important limitation:

- This is a PyTorch/GPU gather prototype, not a fused Triton/CUDA kernel.
- It avoids the initial full QK matmul, but still pays heavy overhead for `topk`, boolean mask compaction, padded gathers, and many small kernels.
- Triton is not installed in the current local environment, so no Triton kernel was compiled locally.

Artifact:

- `outputs/speed_qabs_slow_no_stats/ppl_by_mode.csv`
- `outputs/speed_qabs_fast_no_stats/ppl_by_mode.csv`

Setup:

- Local 2048-token split.
- `prefill_tokens=1536`, `eval_tokens=512`.
- `eval_chunk_size=1`.
- `protect_sink_tokens=10`, `protect_recent_tokens=10`.
- Sparse stats disabled for timing.
- Progress logging every 128 eval tokens.

Results:

| Mode | Slow prototype PPL | Slow seconds | Slow tok/s | Fast prototype PPL | Fast seconds | Fast tok/s |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 25.5495 | 45.00 | 11.38 | 25.5495 | 27.96 | 18.31 |
| qabs8cand3reuse | 25.2084 | 117.53 | 4.36 | 25.4245 | 56.97 | 8.99 |
| qabs8cand7reuse | 24.6731 | 117.06 | 4.37 | 24.7360 | 53.50 | 9.57 |

Interpretation:

- Removing the initial full QK matmul roughly halves qabs wall time in this local run.
- The fast PyTorch prototype is still slower than baseline because candidate compaction and many small gather/topk kernels dominate at this length.
- `qabs8cand7reuse` remains better than baseline PPL in both slow and fast prototypes.
- Fast and slow PPL are close but not bit-identical. The fast path computes only selected/gathered scores and changes some low-level matmul/softmax ordering, so small numeric drift is expected.

Kernel design needed for real speed:

- A fused candidate kernel should output compact candidate indices directly, not dense bool masks.
- Candidate union should be a packed bitset or compact sorted index union on GPU:
  - `current_candidate_bits | previous_candidate_bits | previous_top2_bits`;
  - or a small custom merge for sorted candidate-index lists.
- Rerank and final attention should be fused or at least use persistent on-device buffers to avoid repeated allocation.
- The current PyTorch fast path is the correctness and speed-trend prototype; real acceleration requires replacing mask compaction/topk/gather chains with Triton/CUDA kernels.

## Implementation: CUDA Final Sparse Attention Kernel

Question:

- Can we replace at least one expensive small-kernel chain in the qabs fast path with a real CUDA kernel?

Implementation:

- Added `src/qabs_cuda_kernels.py`.
- Added `--qabs_cuda_final_kernel`.
- Updated `scripts/run_qabs_fast_speed_server.sh` to enable the kernel by default for server timing.
- The kernel fuses the final selected-token attention step:
  1. full-QK only over the final selected indices;
  2. stable softmax over selected tokens;
  3. weighted V reduction.

Important limitation:

- This is a real CUDA extension kernel, but it only covers the final sparse attention stage.
- Candidate generation is still PyTorch:
  - qabs top-dim selection;
  - partial-QK candidate scoring;
  - topk thresholding;
  - candidate reuse union;
  - final top2 rerank index selection.
- Therefore this is not yet the fully fused kernel design needed to beat baseline decisively. It should reduce part of the overhead and gives us a safe correctness bridge before fusing candidate generation.

How to run:

```bash
QABS_CUDA_FINAL_KERNEL=true bash scripts/run_qabs_fast_speed_server.sh
```

Notes:

- The CUDA extension is compiled lazily on first use through `torch.utils.cpp_extension`.
- `torch.utils.cpp_extension` requires `ninja`; install it on the server before trusting kernel timing.
- If compilation or launch fails, the eval script prints one warning and falls back to the existing PyTorch qabs fast path.
- Server timing should check `ppl_by_mode.csv` columns `qabs_fast_path` and `qabs_cuda_final_kernel` before comparing seconds.

Crash fix:

- The first server run hit an illegal memory access in the CUDA final attention kernel at the first qabs decode token.
- CUDA illegal memory access poisons the CUDA context, so Python fallback after the failed launch is not reliable for this class of error.
- Updated the extension to `qabs_final_attention_ext_v2` so server runs do not reuse the old cached build.
- Changed the valid-mask ABI from `bool*` to `uint8_t*` and convert `valid` to `torch.uint8` before entering the extension.

## Implementation: Shared Prefill For Mode Sweeps

Question:

- Can mode sweeps avoid re-running the same prefill for every sparse method?

Implementation:

- Added `--reuse_prefill_cache`, default `true`.
- Added `--baseline_last`, default `true`.
- Prefill is now run once before the mode loop when `--reuse_prefill_cache true`.
- Each mode clones the shared prefill KV cache and starts eval/decode from the same prefill logits.
- `seconds` in `ppl_by_mode.csv` now measures per-mode cache clone + eval/decode time when shared prefill is enabled.
- Added `shared_prefill_seconds` to `ppl_by_mode.csv` so total wall time can be reconstructed.

Notes:

- This is valid for the current experiments because sparse attention is only installed around eval/decode, while prefill is always dense/baseline.
- Shared prefill keeps one full prefill KV cache resident and clones it for each mode. If memory becomes tight at very long contexts, set `--reuse_prefill_cache false`.
- The server speed script now defaults to shared prefill and moves `baseline` to the end of the mode order.
