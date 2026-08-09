# MoR-KV: Query-Conditioned Mixtures of Retrievers for Function-Specialized KV Access

## Status and scope

This is a working paper skeleton, not a claim that the project is submission-ready. Statements marked **target claim** still require the natural-task, multi-model, and kernel evidence defined in `iclr_execution_plan.md`. The controlled and LongBench probe results below are frozen observations, including negative results.

## Draft abstract

Long-context inference methods typically apply one token-importance rule to all attention heads, or adapt only the number of cached tokens assigned to each head. Our analysis instead finds that heads exhibit stable functional priors but context-dependent attention targets, suggesting that the *identity* of the KV retrieval operator should also be routed. We introduce MoR-KV, a query-conditioned mixture of KV retrievers that assigns GQA head groups to a small library of streaming, lexical/structural, semantic-QK, and dense fallback operators. MoR-KV compiles head-specialist portfolios on calibration data, preserves minority nominations that would be suppressed by all-head consensus, projects query-head decisions onto physical GQA KV blocks, and uses a risk gate to control routing failures. On a controlled 100K-token benchmark, a development-NLL-compiled policy improves mean answer NLL from 3.765 to 3.520 over BM25 at the same four-block budget; its advantage over the strongest single hybrid is promising but borderline. A held-out LongBench audit reveals a key failure mode: the current natural operator library has insufficient complementarity, so dataset-level routing underperforms the strongest global operator. These results motivate a systems implementation and a natural-task evaluation of operator-identity routing rather than budget-only adaptation. **Target claim:** with complementary low-cost operators and a calibrated risk gate, MoR-KV improves the quality-throughput Pareto frontier over strong head-, layer-, and query-adaptive KV baselines.

**Updated natural audit.** Expanding the library with a cross-dataset specialist-head operator creates substantial complementarity, but does not solve routing. On a zero-overlap 64-query holdout, frozen deep-QK reaches 3.147 mean NLL; a frozen deep/specialist portfolio reaches 3.239 and does not replicate its exploratory first-sample gain. Specialist retrieval still wins on 45.3% of target queries and the four-action oracle reaches 2.799, leaving 0.347 NLL headroom. Generic answer-free proxies and QK head-confidence gates cannot safely identify these wins. The paper claim must therefore depend on a causal per-head distortion router, not on the current score-signature router.

**Causal-router update.** We now compute exact post-RoPE per-head attention-output distortion for every candidate operator and train operator-specific regressors from low-cost QK/lexical score signatures. On a query-disjoint 17-fit/16-calibration/31-test split, a head-local conformal router reduces mean logical blocks from 15.29 to 10.95 while keeping p95 relative head-output error at 0.0275 and the 0.05-threshold violation rate at 0.86%. It is slightly cheaper and safer than a static head prior; fixed QK-8 violates the same threshold on 29.5% of heads.

After deduplicating the two query heads that share each physical KV head, the learned route retains a 16.46% physical block saving, compared with 15.37% for the static prior and 32.12% for the test oracle. Downstream quality, 8B transfer, and kernel speed remain required before this can become the paper abstract.

An adjacent-position audit on 16 additional query-disjoint examples (four final question positions per example) finds 78.69% within-query majority-action agreement and 66.90% adjacent-position agreement at the 0.05 error threshold. Operator preference therefore has a stable head component but is not a permanent head label; this motivates the paper's hierarchical `head prior + query-position activation` formulation.

In an end-to-end causal reference on the strict 31-query test split, the deployable 0.05 conformal route uses 10.81 of 15.29 logical blocks and changes first-answer-token NLL by +0.0642 (paired 95% CI [0.0394, 0.0900], p95 absolute change 0.187). Fixed QK-8 has near-zero mean change but a 0.369 p95 tail and violates the head-error target on 30.8% of decisions, versus 3.2% for the learned route. The teacher therefore controls a downstream-relevant tail, but 5% head-output error is not a lossless operating point; the paper needs a stricter risk/quality sweep and multi-token NLL.

The frozen-router sweep is monotone: thresholds 0.01/0.02/0.03/0.05 yield physical GQA savings 4.18%/6.92%/10.68%/17.45%, mean first-token delta NLL +0.0259/+0.0381/+0.0494/+0.0639, and p95 absolute deltas 0.068/0.110/0.136/0.183. Even the exact 1% local oracle has a positive mean shift. The stronger paper method must therefore propagate and allocate a global error budget across layers rather than independently threshold every head.

A frozen epsilon=0.02 layer-group intervention identifies the missing structure: layers 0-6 save 2.34% physical KV with delta NLL -0.0050, layers 7-13 and 21-27 cost only +0.0067 and +0.0047, whereas layers 14-20 save just 1.28% but cost +0.0319. The latter group has about seven times higher NLL amplification per physical percentage point. This supports an amplification-weighted global risk allocator as the main method extension.

Post-hoc cumulative allocation that sparsifies layers 0-13 and 21-27 at epsilon=0.02, while keeping 14-20 full, saves 5.63% physical KV with delta NLL +0.0084 (95% CI [-0.0002, 0.0157], p95 0.041). It dominates the all-layer epsilon=0.01 point, which saves 4.18% but costs +0.0259 NLL (p95 0.068). Because the layer ranking was inspected on test31, this is mechanism evidence only; the result is publishable only if calibration-frozen selection replicates on a new zero-overlap holdout.

That external validation now exists. A second natural holdout64 has zero `record_uid` overlap with both earlier 64-query sets and a separately rebuilt corpus. The frozen allocated epsilon=0.02 policy saves 6.19% physical GQA KV with delta NLL +0.0106 (95% CI [0.0031, 0.0187], p95 0.056). Uniform epsilon=0.01 saves less (4.55%) but costs +0.0368 NLL (p95 0.144), while uniform epsilon=0.02 saves 7.56% but costs +0.0500 (p95 0.184). Cross-layer allocation therefore yields a valid dominating risk/quality point, though it is not lossless.

At epsilon=0.05 the same frozen layer allocation saves 14.07% physical KV with delta NLL +0.0363 (p95 0.128): roughly triple the saving of uniform epsilon=0.01 at matched mean drift and lower tail, and nearly double the saving of uniform epsilon=0.02 at lower mean and tail. This is the strongest first-token capacity point; complete-answer validation below limits the scope of that claim.

Full-answer validation narrows that claim: allocated epsilon=0.05 saves 13.85% but costs +0.0436 mean answer NLL with p95 0.269, so it is an aggressive capacity point rather than the default quality setting. The robust full-answer point remains allocated epsilon=0.02 at 6.32% saving and +0.0133 mean delta.

On the first four gold answer tokens of the same external holdout, the allocated route saves 6.27% physical KV with mean delta NLL +0.00563 (95% CI [-0.00113, 0.01252], p95 0.0657). Across the complete first reference answer it saves 6.32% with mean delta +0.01334 (95% CI [0.00103, 0.02890], median +0.00193, p95 0.0892). Full-answer degradation is therefore small but measurable and tail-driven.

The full-answer uniform epsilon=0.01 control saves only 4.97% and costs +0.02279 mean NLL (median +0.01168, p95 0.1120). Allocation thus achieves higher compression while reducing mean degradation by about 41%, median degradation by about 83%, and tail degradation by about 20%.

After flooring ridge-plus-conformal bounds at zero, the corrected allocated epsilon=0.03 complete-answer point saves 8.89% with mean delta NLL +0.02598 (95% CI [0.01102, 0.04397]), median +0.00812, and p95 0.1526. The head-level router summary is unchanged, but a few equal-budget tie breaks change. A post-hoc deployable-gate audit finds that mean/p95/max conformal bounds and near-threshold fraction have only -0.13 to +0.06 Pearson correlation with absolute answer-NLL change; calibration-derived layer-amplification weighting is also weak (-0.048 Pearson, 0.172 Spearman). Per-head conformal bounds therefore cannot yet serve as sequence-level risk certificates; the paper needs a propagated logit/answer-risk estimator rather than a cosmetic query fallback. Moreover, the union across all KV heads within each layer still retains essentially every 4K block, so the byte-saving claim assumes a per-KV-head sparse page layout.

The initial eight-call physical-GQA execution reference is slower than dense SDPA at 4K (0.18x) and 16K (about 0.72x). Packing all physical groups into one padded-ragged gather and one SDPA call improves this to 0.29-0.30x at 4K, 1.17-1.18x at 16K, and 1.50-2.34x at 32K for 512-3,072 active tokens. This closes the 16K attention-subsystem Gate but not 4K, and the fixed-budget microbenchmark must not be conflated with the current 6% physical-saving end-to-end point.

At policy-matched 93.75% retention, even 64-step packed-KV reuse reaches only 0.26x/0.99x/0.49x at 4K/16K/32K in the padded-ragged path. The present quality point is therefore a capacity result, not a latency result; safe compression must increase materially or the backend must avoid K/V replication.

An eight-query Qwen3-8B all-layer pilot covers 36 layers, 1,152 query heads, and 55,296 exact teacher rows. Fixed QK remains close to the same-budget mass oracle (mean relative head-output L2 0.0931 versus 0.0857), while lexical, uniform, and streaming reach 0.1350, 0.1415, and 0.2227. At epsilon=0.05, the exact cheapest-feasible action averages 5.839/7.875 logical blocks and 78.98% head-action agreement across queries, confirming stable priors with meaningful dynamic activation. However, GQA group size four raises the physical union to 7.324 blocks, only 6.99% saving. This is an oracle mechanism result, not yet a learned 8B router or downstream-quality result; physical-GQA-aware training is mandatory at scale.

## 1. Introduction

### Motivation

Head-aware KV compression usually answers one of two questions:

1. Which heads deserve more cache budget?
2. Which tokens or blocks are globally important?

These questions leave a third decision implicit: *which retrieval rule should a head use for the current query?* A recent/sink head, a delimiter-sensitive head, a lexical-copy head, and a semantic-evidence head should not necessarily score remote KV blocks with the same operator.

### Central hypothesis

Attention heads have stable functional priors, but their realized targets depend on the query and context. Therefore the right action is not a permanent head label. It is a query-conditioned action

```text
a_lh(q) = (operator template, specialist portfolio, physical budget, fallback level).
```

### Claimed contributions, pending full validation

1. **Operator-identity routing.** Route the KV retrieval rule per query and GQA head group, not only the cache budget.
2. **Specialist-minority preservation.** Retain blocks nominated by a small number of high-utility heads rather than requiring all-head agreement.
3. **GQA-aware realization.** Map query-head preferences onto deduplicated physical `(layer, kv_head)` blocks and a finite set of runtime templates.
4. **Risk-controlled routing.** Learn policy regret from downstream loss and expand the budget or fall back when routing confidence is low.
5. **Mechanism and systems evidence.** Report both positive controlled results and a natural-task No-Go result, then test whether a real sparse-attention implementation turns the mechanism into a quality-throughput Pareto improvement.
6. **Causal risk teacher.** Train operator routing on exact per-head attention-output distortion and calibrate a one-sided error bound per head, rather than treating retrieval recall or global answer NLL as the routing label.

## 2. Diagnostic study: stable priors, conditional targets

The companion head-function experiment profiles all 448 query heads of Qwen3-0.6B over 32 controlled inputs. Position-type head rankings are highly stable across inputs, while 63 heads are context-sensitive and semantic heads cluster in deeper layers. This supports a hierarchical model:

```text
static prior p(role | layer, head)
    + query evidence p(operator advantage | q, score signature)
    -> routed runtime action
```

The diagnostic is observational. The paper must include causal interventions: remove or replace only the blocks selected by each operator and measure attention-output and answer-loss changes.

## 3. Method

### 3.1 Operator library

- `streaming`: sink plus recent window;
- `lexical/structural`: token hashes, delimiters, record boundaries, and compact n-gram sketches;
- `semantic-QK`: low-rank pre-RoPE Q/K block scoring;
- `dense/risk`: expanded budget or full-attention fallback.

The final implementation must expose the memory and latency cost of every side index. BM25 is only a research proxy for the lexical/structural operator.

### 3.2 Query-conditioned router

The prototype forms a low-cost score signature from each head's top score, top-score margin, and candidate-score spread. A paper version should train a cost-sensitive router on per-query downstream regret, with an explicit reject action. Dataset or task labels must not be available at test time.

### 3.3 Specialist portfolios and GQA projection

Head utility is compiled only on calibration data. Query heads sharing a KV head are deduplicated before physical cache accounting. The runtime decision operates over a small number of GQA-compatible templates to preserve CUDA Graph and paged-attention compatibility.

### 3.4 Specialist-preserving selection

The current prototype includes weighted reciprocal-rank fusion, minority-max nomination, and a group-saturating submodular greedy rule. The submodular rule has a standard cardinality-constrained approximation guarantee, but the current empirical ablation is mixed; it remains a hypothesis rather than a confirmed contribution.

### 3.5 Risk gate

Let `L(a, q)` be downstream loss for action `a`. The router predicts action regret and rejects when the upper confidence bound exceeds a threshold. Rejection selects a larger-budget template. Evaluation must report mean loss, p95/p99 loss, fallback rate, and realized compute.

The first natural holdout falsifies generic query-level regret proxies: conservative gates make zero useful switches. The revised teacher is exact per-head attention-output distortion or omitted mass, with a CVaR constraint on tail distortion. This is a required method change, not an optional ablation.

## 4. Evidence already obtained

### 4.1 Controlled 100K-token benchmark

At a four-block budget, raw MoR-KV improves the controlled retrieval utility from 0.530 for BM25 and 0.553 for a single hybrid to 0.563. A deliberately wrong router falls to 0.413, providing evidence that operator matching matters.

Using only development answer NLL to choose among frozen policies yields a test mean NLL of 3.520, compared with 3.765 for BM25 and 3.706 for the single global hybrid. The paired 95% interval versus BM25 excludes zero; the interval versus the global hybrid reaches +0.002 and is therefore not yet a robust win.

### 4.2 Natural LongBench No-Go audit

On a 64-query held-out probe, the strongest global deep-QK operator reaches mean NLL 3.304, while dataset-routed selection among three operators reaches 3.510. Even the per-query oracle among those candidates has only 0.057 NLL headroom. This falsifies the idea that a better router alone is sufficient: the operator library itself must become more complementary.

### 4.3 Submodular ablation

The group-saturating selector helps slightly at budgets 1 and 39 but hurts at budgets 4, 8, and 16. It cannot be presented as an established contribution without a better utility definition and natural-task validation.

## 5. Experiments required for the main paper

### Main table

Qwen3-8B, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct on LongBench, RULER, InfiniteBench/LooGLE, long code, summarization, and reasoning-decode KV. Compare at identical physical KV bytes and include index memory.

### Baselines

FullKV, StreamingLLM, H2O, SnapKV/PyramidKV/Ada-KV, Quest-class query-aware retrieval, RazorAttention, DuoAttention, HeadKV, Task-KV, CompilerKV, PolyKV, HARD-KV, and the strongest local single-policy hybrid.

### Essential ablations

- one operator for all heads versus budget-only routing versus operator-identity routing;
- static head labels versus query-conditioned routes;
- query-head decisions versus GQA physical union;
- majority consensus versus specialist preservation;
- each operator removed in turn;
- risk gate removed, confidence threshold swept, and wrong-router intervention;
- calibration size and cross-dataset/cross-model transfer;
- block size, low-rank dimension, and metadata budget.

### Systems table

Report physical KV bytes, side-index bytes, TTFT, decode throughput at batch 1/4/8/16, router/scoring/gather/kernel time, GPU/CPU/PCIe traffic, fallback rate, and p95/p99 latency. At least 30 measured repetitions after warm-up are required.

## 6. Planned figures

1. Head functional-prior stability versus query-conditioned operator advantage.
2. MoR-KV runtime diagram from score signatures to GQA block union and sparse attention.
3. Quality versus physical KV bytes across models and tasks.
4. Throughput versus task quality, including side-index overhead.
5. Router regret calibration and the risk/compute trade-off.
6. Specialist blocks lost by majority consensus and recovered by MoR-KV.

## 7. Decision gates

The work becomes a full systems paper only if operator-identity routing beats the strongest single-policy and budget-only baselines on at least two 8B models, on several natural task families, at equal physical memory, while producing a real decode-speed gain after routing overhead. If natural gains remain absent or GQA union removes the nominal savings, the honest paper should be reframed as a diagnostic study of specialist-head retrieval failure rather than a general KV-cache method.
