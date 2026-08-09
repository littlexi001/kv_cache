# Query-span pre-RoPE block retrieval protocol

## Status and claim boundary

This is a **deployable control and diagnostic**, not yet a paper-method claim.
It tests one narrow question:

> Can several pre-RoPE Query vectors already present in the question recover a
> better remote support than the final Query alone, while the pretrained model
> still consumes that support with completely native post-RoPE scores and
> Values?

The selector is training-free and answer-free, but its ingredients are close to
well-established late-interaction and query-conditioned sparse retrieval.  It
must not be described as the core novelty unless held-out experiments show a
distinct, reproducible benefit attributable to the combination of
**RoPE-free query-span late interaction + native-RoPE consumption**.

## Motivation: why several question tokens may help

The controlled input contains two plausible statements:

- gold: `The school register lists Xiaoming's age as nine years.`
- conflict: `A family note lists Xiaoming's age as four/six/eight/two years.`

The question asks for Xiaoming's age **according to the school register**.  A
single final Query vector has to compress at least two facets:

1. entity: `Xiaoming`;
2. source constraint: `school register`.

An entity-only retriever can therefore prefer both the gold and conflict
sentences.  The proposed block score allows different question-token Queries to
match different tokens inside the same block:

$$
S_h(B)
=
\frac{1}{|\mathcal A|}
\sum_{a\in\mathcal A}
\max_{j\in B}
\frac{q^{\mathrm{pre}}_{h,a}\cdot k^{\mathrm{pre}}_{h,j}}
{\lVert q^{\mathrm{pre}}_{h,a}\rVert_2\lVert k^{\mathrm{pre}}_{h,j}\rVert_2}.
$$

Here, $\mathcal A$ is a deterministic, evenly spaced subset of token positions
inside the visible question span.  Thus one anchor can match `Xiaoming`, while
another matches `school register`; a block containing both facets can outrank a
block containing only the entity.  This is a hypothesis, not a guaranteed
property of pretrained Q/K geometry.

## Arms

All sparse arms use exactly the same per-head token budget:

$$
K=\lceil 0.02L\rceil
=K_{\mathrm{sink}}+K_{\mathrm{remote}}+K_{\mathrm{local}}+1_{\mathrm{current}}.
$$

1. `native_noop`: untouched model forward; no selector is active.
2. `exact_final_pre_top2_postscore`: the final token's exact pre-RoPE Query
   ranks individual remote pre-RoPE Keys.
3. `queryspan_block_top2_postscore`: question-span Queries rank fixed remote
   blocks using $S_h(B)$.  Whole blocks are retained in rank order; if the last
   block exceeds the remaining budget, its tokens are ranked by their maximum
   anchor similarity and exactly the remaining number is kept.
4. `queryspan_tokenmax_top2_postscore`: minimal ablation that ranks every remote
   token by its maximum similarity to any question anchor, without block-level
   conjunction.

The experiment caches all prefix pre-RoPE Keys once, but caches **only the
selected question anchors**, never a full-sequence Query matrix.

## Native consumer invariant

Retrieval scores are used only to choose token indices $\mathcal S_h$.  Every
sparse arm then discards them and computes:

$$
a_{h,j}
=
\operatorname{softmax}_{j\in\mathcal S_h}
\left(
\frac{(R_tq_{h,t})^\top(R_jk_{h,j})}{\sqrt d}
\right),
\qquad
o_h=\sum_{j\in\mathcal S_h}a_{h,j}v_{h,j}.
$$

Therefore, selected Keys keep their original position, native RoPE phase,
native post-RoPE score, and original Value.  There is no position repair, score
blend, answer-gradient term, or gold-dependent intervention.

## Information firewall

Selector inputs:

- deterministic question-span positions from prompt construction;
- pre-RoPE Query vectors at those positions;
- pre-RoPE historical Keys;
- fixed block boundaries and fixed sink/local budgets.

Forbidden selector inputs:

- gold/conflict spans or labels;
- gold answer token or answer probability;
- gradients, Value attribution, or oracle attention;
- evidence-aware tuning per seed.

Gold/conflict positions enter only `QuerySpanController.record(...)`, after the
support and sparse probabilities already exist.  The core selection functions
have no evidence-label argument.

## Measurements and hard audits

Primary quality metrics:

- gold evidence token recall and sentence-hit rate;
- conflict token recall and sentence-hit rate;
- gold/conflict attention mass;
- gold answer PPL, next-token accuracy, and gold–conflict output margin;
- paired deltas from untouched `native_noop`.

Required validity metrics (must all be zero violations):

- selected count differs from $K$;
- duplicate selected positions;
- missing mandatory sink, local, or current positions;
- nonzero `cached_full_prefix_query_tokens`;
- nonzero `selector_used_evidence_labels`.

The report also records support Jaccard overlap with exact final-query pre-RoPE
Top-2%, selected block count, boundary-block token count, and query latency.
Seeds where gold evidence falls inside the fixed sink region must be reported
separately or excluded from the retrieval-difference claim, because those seeds
do not test semantic selection.

## Falsification criteria

The idea is rejected as a useful replacement if any of the following holds on
held-out seeds:

1. block retrieval does not improve gold-versus-conflict separation over both
   final-query exact-pre and token-max controls;
2. recall rises but native attention mass, PPL, or accuracy degrades, indicating
   coarse blocks merely add weak competitors;
3. gains occur only when gold evidence overlaps sink tokens or a favorable block
   boundary;
4. latency/memory overhead of multiple anchors erases the sparse-attention
   benefit;
5. improvements disappear under changed block sizes, evidence order, or
   paraphrases.

A credible positive result requires paired held-out gains at 32K/64K, no budget
violations, block-size robustness, and a larger gold-minus-conflict retrieval
gap than the token-max ablation.

## Novelty audit and closest work

- **Multi-Token Attention (MTA)** is the most direct novelty warning
  ([COLM 2025 / arXiv:2504.00927](https://arxiv.org/abs/2504.00927)).  It
  explicitly frames single-token Q/K matching as a bottleneck, motivates the
  problem with conjunction-style retrieval such as `Alice` + `rabbit`, and lets
  multiple nearby queries, keys, and heads jointly determine attention through
  convolutions.  We therefore make **no** claim to be the first to solve a
  single-query bottleneck.  The narrower purpose here is to test whether the
  already-frozen internal vectors of Qwen3 support an inference-time,
  RoPE-free, sparse selector without retraining its attention mechanism.
- **ColBERT-style late interaction** already establishes multi-vector
  query-to-document matching.  The max-over-token/aggregate-over-query form is
  intentionally borrowed as a control.
- **BSER** ([OpenReview: ItvdPuHAMq](https://openreview.net/forum?id=ItvdPuHAMq))
  studies query-conditioned block relevance.  This strongly overlaps the block
  selector abstraction.
- **BlockRank**
  ([OpenReview: zj45hoQhjD](https://openreview.net/forum?id=zj45hoQhjD)) studies
  query-token-to-document-block relevance.  This is another direct neighbor.
- **SALS** performs RoPE-free sparse token selection in a compressed latent
  space before sparse attention
  ([NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/00a0ebcad584c59dbc439c2af8793638-Abstract-Conference.html)).
- **FASA** uses RoPE frequency structure and query-aware token eviction
  ([ICLR 2026](https://openreview.net/forum?id=FnSgecCEwg)).
- **SpotAttention** learns a query-conditioned sparse selector by distilling the
  dense attention distribution
  ([arXiv:2606.22874](https://arxiv.org/abs/2606.22874)).

Consequently, “multi-query block selection” or “pre-RoPE retrieval” alone is
not a defensible novelty claim.  At most, this probe can establish a useful
empirical component: whether preserving separate question facets before RoPE,
then handing the selected support back to an unchanged native-RoPE consumer,
specifically reduces source/entity conflict in very long contexts.

## Artifacts

- runner: `src/run_queryspan_prerope_retrieval_probe_8b.py`
- CPU tests: `tests/test_queryspan_prerope_retrieval_probe.py`
- GPU6/7 launcher (file only):
  `scripts/run_queryspan_prerope_retrieval_gpu67_20260801.sh`

Minimal one-GPU smoke (documented only; not launched during implementation):

```bash
ROOT=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
CUDA_VISIBLE_DEVICES=6 \
QUERYSPAN_PREKEY_STORAGE=cuda \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/fdong/miniconda3/envs/moe/bin/python \
  "$ROOT/src/run_queryspan_prerope_retrieval_probe_8b.py" \
  --model-name-or-path /home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --output-dir "$ROOT/outputs/20260801_queryspan_prerope_smoke_gpu6" \
  --lengths 8192 --seed-start 0 --num-seeds 1 \
  --ratio 0.02 \
  --variants native_noop,exact_final_pre_top2_postscore,queryspan_block_top2_postscore,queryspan_tokenmax_top2_postscore \
  --local-window 128 --sink-tokens 16 \
  --block-size 64 --query-anchor-count 16 --score-chunk-blocks 32 \
  --class-sample-count 8 --packet-gap-tokens 16 \
  --prefill-chunk-size 64 --dtype bfloat16 --load-in-4bit \
  --attn-implementation sdpa \
  --original-max-position-embeddings 40960 --global-max-position 70000
```

Analytical 8K memory estimate for Qwen3-8B NF4/BF16 on an RTX 3090:

- model and quantization workspace: roughly 6--8 GiB;
- native GQA KV cache: about 1.13 GiB;
- extra exact pre-RoPE K cache: about 0.56 GiB;
- 16 cached question Queries: below 5 MiB over all layers;
- largest selector temporaries at 8K: tens of MiB.

Thus the expected peak is approximately **9--11 GiB**.  Based on neighboring
Qwen3-8B 8K probes on the same server, model loading is about 20 seconds and
prefill about 7--15 seconds.  The two 16-anchor selector passes should add a few
seconds; a conservative end-to-end estimate for one seed and all four arms is
**35--70 seconds**.  These are planning estimates, not measured results from
this runner.
