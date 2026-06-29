# Section 26: Counterfactual Risk-Gated KV Cache Compression

Date: 2026-06-29

## Working Hypothesis

The strongest ICML direction is not another static KV/token retrieval rule, but an online risk-gated compression policy:

1. Use a short calibration window at each block.
2. Run full-cache and candidate compressed-cache forwards on the same calibration tokens.
3. Estimate counterfactual risk from the token-level loss gap:
   `risk(token, combo) = loss_compressed(token, combo) - loss_full(token)`.
4. Select the fastest candidate that satisfies both mean-loss and tail-risk constraints.
5. Optionally rescue risky decode tokens back to full attention using a cheap gate derived from calibration statistics.

This reframes KV compression as a decision problem with observable counterfactual risk, rather than as static retrieval similarity.

## Why This Direction Is Promising

Existing notes show three useful facts:

- Full context is not always optimal for PPL; oracle/gold-only context can beat full context.
- Attention-token pruning can match or beat baseline PPL with a very small keep ratio in oracle settings.
- Static synthetic KV or static layer replacement is unstable across longer eval windows.

So the method should not claim that compression is always safe. The better claim is:

> Compression is safe only in locally verifiable regions; short counterfactual calibration can identify those regions online.

## Prototype Definitions

Current server prototype:

- Server path: `/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl`
- Script: `src/run_pcic_rescue_blockwise_local.py`
- Model: `/home/fdong/hrj/prove/Qwen3-0.6B`
- Compression backend: landmark attention on selected layer pairs.
- Candidate combos used in the main multi-combo runs:
  `1,2;4,5;7,8;22,23;0,6;0,7;0,13;2,0`

Policies tested:

- `fastest_safe + none`: choose fastest candidate whose calibration mean loss is within slack.
- `fastest_safe + calib_margin`: add token-level rescue using calibration-derived low-margin gate.
- `risk_pareto + none`: choose candidate by calibration tail risk, without token rescue.

## Current Results

All results below compare compressed method against the direct full-cache baseline on the same eval tokens.

| Dataset | Policy | Eval Tokens | PPL Ratio | Delta Loss | Rescue | Compressed | Speed Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| topic-stress | fastest_safe + none | 512 | 1.0253 | +0.0249 | 0 | 512 | 1.0389 |
| topic-stress | fastest_safe + calib_margin q95 | 512 | 0.9995 | -0.0005 | 258 | 254 | 1.0166 |
| topic-stress | risk_pareto + none | 512 | 1.0044 | +0.0044 | 0 | 512 | 1.0392 |
| War and Peace | fastest_safe + none | 256 | 1.0011 | +0.0011 | 0 | 256 | 1.0364 |
| War and Peace | fastest_safe + calib_margin q95 | 256 | 1.0077 | +0.0076 | 85 | 171 | 1.0266 |
| War and Peace | risk_pareto + none | 256 | 0.9948 | -0.0052 | 0 | 256 | 1.0358 |
| Monte Cristo | fastest_safe + none | 256 | 1.0034 | +0.0034 | 0 | 256 | 1.0451 |
| Monte Cristo | fastest_safe + calib_margin q95 | 256 | 1.0302 | +0.0298 | 62 | 194 | 1.0303 |
| Monte Cristo | risk_pareto + none | 256 | 1.0039 | +0.0039 | 0 | 256 | 1.0369 |

`Speed Ratio` is method eval seconds divided by baseline eval seconds. Values above 1.0 mean the current eager/Python prototype is slower. This table does not include calibration/candidate-search overhead; true end-to-end runtime is therefore worse unless calibration is amortized or optimized.

## Long-Context Multi-Layer Results

These runs use 512 eval tokens per dataset, 4 blocks, 16 calibration tokens per block, and long prefill contexts. Network use was kept low by launching jobs with `nohup` and writing logs on the server under `outputs/crg_long_logs/`.

Candidate sets:

- `4plus`: only 4/6/8/10-layer compressed candidates.
- `ladder`: 2/4/6/8/10-layer candidates, so the risk gate can fall back to 2 layers.
- `4plus + calib_margin`: 4plus candidates plus token-level margin rescue.

8k prefill results:

| Dataset | Policy | PPL Ratio | Delta Loss | Speed Ratio | Compressed Coverage | Effective Avg Layers | Effective KV Saved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| War and Peace | 4plus | 1.0450 | +0.0440 | 1.0579 | 100.0% | 4.00/28 | 13.2% |
| War and Peace | ladder | 1.0214 | +0.0212 | 0.9674 | 100.0% | 2.00/28 | 6.6% |
| War and Peace | 4plus + calib_margin | 1.0251 | +0.0248 | 1.0470 | 76.0% | 3.04/28 | 10.0% |
| Monte Cristo | 4plus | 1.0146 | +0.0145 | 1.0408 | 100.0% | 4.00/28 | 13.2% |
| Monte Cristo | ladder | 0.9963 | -0.0037 | 1.0352 | 100.0% | 2.00/28 | 6.6% |
| Monte Cristo | 4plus + calib_margin | 1.0070 | +0.0070 | 1.0539 | 94.1% | 3.77/28 | 12.4% |
| topic-stress | 4plus | 1.0370 | +0.0363 | 1.0667 | 100.0% | 4.00/28 | 13.2% |
| topic-stress | ladder | 1.0042 | +0.0042 | 1.0397 | 100.0% | 2.00/28 | 6.6% |
| topic-stress | 4plus + calib_margin | 1.0085 | +0.0085 | 1.0442 | 81.2% | 3.25/28 | 10.7% |

16k prefill results:

| Dataset | Policy | PPL Ratio | Delta Loss | Speed Ratio | Compressed Coverage | Avg Layers | Effective KV Saved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| War and Peace | 4plus | 1.0552 | +0.0537 | 1.0632 | 100.0% | 4.00/28 | 13.6% |
| War and Peace | ladder | 0.9957 | -0.0043 | 1.0399 | 100.0% | 2.00/28 | 6.8% |
| Monte Cristo | 4plus | 1.0217 | +0.0215 | 1.0617 | 100.0% | 4.00/28 | 13.6% |
| Monte Cristo | ladder | 1.0013 | +0.0013 | 1.0390 | 100.0% | 2.00/28 | 6.8% |
| topic-stress | 4plus | 1.0760 | +0.0732 | 1.0810 | 100.0% | 5.50/28 | 18.7% |
| topic-stress | ladder | 1.0219 | +0.0217 | 1.0440 | 100.0% | 2.50/28 | 8.5% |

Interpretation:

- Forcing 4+ compressed layers gives meaningful KV reduction, but PPL degradation is too large.
- The ladder gate preserves quality better by falling back to 2-layer candidates; War 16k and Monte 8k are slightly better than full-cache baseline.
- The current eager prototype is still usually slower even at 8k/16k. The only faster eval segment observed here is War 8k ladder, with speed ratio `0.9674`; this is not yet robust enough to claim speedup.
- Margin rescue improves 4plus quality but does not fully recover baseline and adds runtime overhead.

## Head-Granular Risk Budget

Motivation:

Layer-level compression is too coarse. A safer version is to compress only selected heads inside selected layers:

- unchanged layers: full attention.
- selected layers: `head_recent`, where `full_heads = K` heads use full attention and the remaining heads use recent-window attention.
- risk-budget selection: among candidates passing calibration mean-loss and tail-risk constraints, choose the candidate with the largest layer/head compression budget.

Implementation changes:

- Added `head_recent` as a `layerbudgetattn` budget type in `evaluate_qwen3_top2_head_limit3_ppl.py`.
- Added `--budget_type head_recent`, `--full_heads`, and `--combo_select_policy risk_budget` to `run_pcic_rescue_blockwise_local.py`.

Negative head-only result:

- Compressing heads across all 28 layers is too aggressive.
- With 8k prefill, `fullhKrecent` over all layers either falls back to no compression or causes large PPL loss.
- Example: topic-stress with `recent=1024`, 16-token calibration selected compressed heads and got PPL ratio `2.2308`.
- With stricter 64-token calibration and `recent=2048`, the gate selected K16 everywhere, giving zero compression.

Positive `(layer, head)` result:

8k prefill, 512 eval tokens, 32 calibration tokens per block, selected layers use `head_recent`, `recent=512`.

| Dataset | Full Heads Kept | Compressed Heads | PPL Ratio | Delta Loss | Speed Ratio | Avg Selected Layers | Effective KV Saved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| War and Peace | 12/16 | 4/16 | 1.0117 | +0.0117 | 1.0780 | 3.50/28 | 2.94% |
| War and Peace | 8/16 | 8/16 | 1.0081 | +0.0081 | 1.0532 | 2.00/28 | 3.36% |
| War and Peace | 4/16 | 12/16 | 1.0133 | +0.0132 | 1.0629 | 2.00/28 | 5.03% |
| Monte Cristo | 12/16 | 4/16 | 1.0026 | +0.0026 | 1.0610 | 2.50/28 | 2.10% |
| Monte Cristo | 8/16 | 8/16 | 1.0074 | +0.0074 | 1.0536 | 2.00/28 | 3.36% |
| Monte Cristo | 4/16 | 12/16 | 1.0149 | +0.0148 | 1.0364 | 2.00/28 | 5.03% |
| topic-stress | 12/16 | 4/16 | 1.0010 | +0.0010 | 1.0987 | 4.50/28 | 3.78% |
| topic-stress | 8/16 | 8/16 | 1.0063 | +0.0063 | 1.0612 | 4.00/28 | 6.71% |
| topic-stress | 4/16 | 12/16 | 1.0058 | +0.0058 | 1.1655 | 3.50/28 | 8.81% |

Interpretation:

- `(layer, head)` granularity is much more stable than compressing whole layers or heads across all layers.
- The risk budget naturally expands selected layers on topic-stress while keeping War/Monte closer to 2-layer compression.
- Current speed is still slower because `head_recent` computes full-head and recent-head branches separately in Python/eager mode.
- The best paper direction is now: risk-budgeted expansion over `(layer, head)` units, not just layer sets.

## Runtime Accounting

The prototype records per-block runtime in `pcic_r_blockwise_results.csv`:

- `seconds`: method eval segment time.
- `baseline_seconds`: full-cache baseline eval segment time.
- `rescue_tokens`: eval tokens decoded with full attention.
- `compressed_tokens`: eval tokens decoded with the compressed attention path.

Current speed signal:

- Best quality-safe topic-stress run: `fastest_safe + calib_margin q95`, speed ratio `1.0166`.
- Best War and Peace run: `risk_pareto + none`, speed ratio `1.0358`.
- Best Monte Cristo stable run: `risk_pareto + none`, speed ratio `1.0369`.

Interpretation: quality can already match or beat baseline in some settings, but the current implementation is not yet faster. The likely causes are per-token Python control flow, eager attention patch overhead, and the conservative setting that compresses only two layers at a time.

## Compression Accounting

There are three different compression notions:

1. **Eval-token compressed coverage**: fraction of eval tokens that use the compressed path instead of rescue/full.
2. **Compressed-layer KV keep ratio**: within compressed layers, landmark attention keeps recent tokens plus sparse landmarks.
3. **Whole-model effective KV reduction**: compressed-layer reduction multiplied by how many layers are compressed and how often rescue is used.

Current landmark setting:

- `recent_tokens = 512`
- `landmark_stride = 64`
- approximate compressed-layer keep ratio:
  `keep(H) = (512 + floor((H - 512) / 64)) / H`

Approximate keep ratios:

- topic-stress, history about 1k-1.6k: compressed-layer keep ratio averages about `39.5%`.
- War/Monte, history about 4.1k-4.4k: compressed-layer keep ratio averages about `13.4%`.

Because the current candidate combos compress only two layers, if Qwen3-0.6B is treated as a 28-layer model, the whole-model KV reduction is much smaller:

- topic-stress, no rescue: about `4.4%` whole-model KV reduction.
- War/Monte, no rescue: about `6.2%` whole-model KV reduction.
- topic-stress, `calib_margin q95`: `254/512 = 49.6%` compressed eval-token coverage, so effective whole-model KV reduction is about `2.2%`.
- War, `calib_margin q95`: `171/256 = 66.8%` compressed coverage, effective whole-model KV reduction about `4.1%`.
- Monte, `calib_margin q95`: `194/256 = 75.8%` compressed coverage, effective whole-model KV reduction about `4.7%`.

Thus the current experiments are conservative quality probes, not high-compression speed demonstrations. To show large speedups, the next version must safely compress more layers or use longer contexts where each compressed layer has a much lower keep ratio.

Important block-level observations:

- On topic-stress, no-rescue compression fails mainly on blocks selecting `22,23` or `7,8`; margin rescue recovers the PPL loss and slightly beats baseline.
- On War and Peace, risk-pareto selection is strongest so far: PPL improves over baseline while compressing all eval tokens.
- On Monte Cristo, margin rescue failed because one high-risk block selected `22,23`; calibration already showed large max loss gap, but the token margin feature was not aligned. Risk-pareto avoids this failure and selects safer combos.

## Negative Result

Synthetic/static KV replacement is not stable enough as the main paper direction.

On topic-stress with 512 eval tokens:

- baseline PPL: 3.6702
- synthetic layers `4,5`: PPL 3.7027, ratio 1.0089
- synthetic layers `7-14`: PPL 4.0712, ratio 1.1093
- synthetic layers `22-27`: PPL 4.4718, ratio 1.2184

This is useful as motivation: static compression can work locally, but online counterfactual risk is needed for robustness.

## Interpretation

The cleanest story is a two-level gate:

1. **Block-level risk gate** chooses the compressed layer budget using calibration mean loss plus tail-risk metrics such as max positive loss gap and positive-gap ratio.
2. **Token-level rescue gate** is optional and only used when calibration shows that cheap features, such as margin or entropy, align with high loss-gap tokens.

Do not rely on margin rescue alone. It can over-rescue and can still miss risk when margin is not aligned with counterfactual loss.

## Current Bottleneck

The current Python/eager prototype is slower than baseline even when compressing all eval tokens:

- typical speed ratio is about 1.03-1.05x slower.

This is likely implementation overhead from the custom landmark attention path and per-token Python loop, not evidence that the final method cannot be fast. For a paper, speed claims need either:

- a fused/batched implementation, or
- a fairer long-context benchmark where reduced attention work dominates overhead.

## Next Experiments

Priority 1:

- Run `risk_pareto + calib_margin q95` on topic-stress, War and Peace, Monte Cristo.
- The goal is to combine the stability of block-level risk-pareto with the PPL recovery of token-level rescue.

Priority 2:

- Increase context and eval length:
  - `prefill_tokens`: 8192 or 16384
  - `eval_tokens_per_block`: 128 or 256
  - `num_blocks`: 8 or 16
- This checks whether compression speed becomes visible when context is longer.

Priority 3:

- Add an explicit rule:
  if the selected combo has `risk_max_loss_gap > threshold` or `risk_positive_ratio > threshold`, either choose the next safest combo or fall back to full for that block.

Priority 4:

- Compare against more credible baselines:
  - fixed landmark compression without calibration
  - fixed low-layer or high-layer compression
  - token-level margin rescue without counterfactual calibration
  - random layer-pair compression with same budget

## Paper Angle

Potential title:

**Counterfactual Risk-Gated KV Cache Compression**

Core contribution:

- A calibration-based counterfactual risk estimator for KV cache compression.
- A tail-risk constrained online budget selector.
- A conditional rescue policy that uses cheap uncertainty features only when calibration proves alignment.

Main claim to prove:

> Short calibration windows can identify when and where KV cache compression is safe, producing lower or matched PPL than full-cache decoding while enabling aggressive compression in low-risk regions.

The strongest experimental target is not yet raw speed in this prototype. The first target is robust PPL improvement or parity under aggressive compression; speed should be demonstrated after the backend is optimized or under long-context settings.
