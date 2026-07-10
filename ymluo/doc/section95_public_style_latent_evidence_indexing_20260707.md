# Section 95: Public-style latent evidence indexing and innovation positioning (2026-07-07)

## Goal

This section moves the learned summary-memory direction away from "KV reconstruction with low MSE" and toward a more defensible paper idea:

**Latent Evidence KV Indexing (LEKI): learn a compact query-conditioned index over KV/pages, retrieve a small set of evidence pages, then compose boundary-safe evidence text or KV spans for downstream answering.**

The main claim should not be "we compress K/V vectors with an autoencoder." That is too close to existing KV-compression and learned-eviction work.
The stronger claim is:

> MSE-preserving KV compression is not evidence-preserving. Long-context compression needs a learned evidence index plus a risk-aware evidence recovery/composition stage.

## New public-style case families

Added three benchmark-like synthetic families to `run_real_qwen_seq_ae_search_trace.py`:

| family | purpose |
|---|---|
| `needle_fact` | single-hop hidden fact / needle retrieval |
| `multi_hop_bridge` | two-hop locator -> answer retrieval |
| `current_conflict` | stale/current conflicting records |

These are intentionally closer to public long-context tasks than the previous internal names `old_single`, `two_old`, and `decoy_exact`.

The supervised ranker now also treats `current_conflict` stale pages as hard negatives and treats `multi_hop_bridge` as a two-page adaptive-budget family.

## Rank-only public-style suite

Output:

`ymluo/projects/learned_hierarchical_summary_memory/outputs/supervised_latent_page_ranker_public_style_page16_rankonly_20260707`

Setup:

- Qwen3-0.6B hidden/KV trace
- context = 256 tokens
- page = 16 tokens
- train variants = `a,b,c,d,e,f`
- eval variants = `g,h,i,j,k,l`
- families = `needle_fact,multi_hop_bridge,current_conflict`
- train cases = 18
- eval cases = 18
- trace samples = 864
- latent storage ratio vs KV = 0.78125%

Results:

| ranker | top pages | center page recall | span page recall |
|---|---:|---:|---:|
| attention | 1 | 0.028 | 0.222 |
| attention | 2 | 0.083 | 0.389 |
| supervised latent ranker | 1 | 0.667 | 0.806 |
| supervised latent ranker | 2 | 0.833 | 0.944 |

Interpretation:

The learned latent evidence index generalizes across held-out keys/answers/positions and is much stronger than direct attention ranking. This supports keeping the direction, but it also clarifies that the core novelty is retrieval/indexing, not plain vector reconstruction.

## Composer coverage analysis

Because local Windows PyTorch is CPU-only, full prompt NLL evaluation is too slow for rapid iteration. Instead, I added a fast non-forward analysis script:

`ymluo/projects/learned_hierarchical_summary_memory/src/analyze_public_style_composer_coverage.py`

It consumes saved `page_rankings.json` and `summary.json`, rebuilds the synthetic cases with the tokenizer, and measures:

- whether selected raw evidence contains the exact answer,
- whether composed evidence contains the exact answer,
- composed prompt token count,
- center/span page recall.

### Extra halo sweep

Supervised top2 results:

| composer extra halo | span recall | raw answer coverage | composed answer coverage | composed tokens |
|---:|---:|---:|---:|---:|
| 1 | 0.944 | 0.722 | 0.778 | 80.0 |
| 2 | 0.944 | 0.722 | 0.889 | 81.1 |
| 3 | 0.944 | 0.722 | 1.000 | 80.6 |

Attention comparison at extra halo = 3:

| ranker | policy | span recall | raw answer coverage | composed answer coverage | composed tokens |
|---|---|---:|---:|---:|---:|
| attention | top2 | 0.389 | 0.111 | 0.444 | 68.6 |
| supervised latent ranker | top2 | 0.944 | 0.722 | 1.000 | 80.6 |

Key observation:

Page recall alone is insufficient. In `multi_hop_bridge`, the answer record can cross a 16-token page boundary, so the center evidence page can be recalled while the actual answer string is partially outside the raw selected span.
The composer extra-halo stage fixes this boundary failure while keeping the prompt around 31% of the original context.

This is an important innovation point:

**The method is not just "retrieve pages"; it retrieves latent evidence anchors and then performs boundary-safe evidence reconstruction.**

## GPU prompt/composer NLL validation

After the server recovered, I synced the updated LEKI/public-style scripts to the GPU machine and ran real prompt NLL/generation evaluation with Qwen3-0.6B.

Main 18-case output:

`/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/supervised_latent_page_ranker_public_style_page16_composer_gl_gpu_20260707`

Local copy:

`ymluo/projects/learned_hierarchical_summary_memory/outputs/supervised_latent_page_ranker_public_style_page16_composer_gl_gpu_20260707`

Setup:

- train variants = `a,b,c,d,e,f`
- eval variants = `g,h,i,j,k,l`
- eval cases = 18
- context = 256 tokens
- page = 16 tokens
- rankers = attention vs supervised latent ranker
- policy = top2 pages
- composer extra halo = 3
- decode steps = 16

Rank recall:

| ranker | top pages | center page recall | span page recall |
|---|---:|---:|---:|
| attention | 2 | 0.000 | 0.194 |
| supervised latent ranker | 2 | 0.806 | 1.000 |

Prompt/composer quality:

| method | active tokens | span recall | answer NLL | exact |
|---|---:|---:|---:|---:|
| full KV cache | 256.0 | 1.000 | 0.6815 | 0.667 |
| attention top2 + raw text | 144.2 | 0.194 | 4.9479 | 0.000 |
| attention top2 + composer | 55.3 | 0.194 | 2.9015 | 0.278 |
| oracle pages + raw text | 138.1 | 1.000 | 0.7908 | 0.611 |
| oracle pages + composer | 80.6 | 1.000 | 0.8567 | 0.778 |
| supervised top2 + raw text | 152.8 | 1.000 | 0.9370 | 0.611 |
| supervised top2 + composer | 80.6 | 1.000 | 0.8567 | 0.778 |

Important result:

**Supervised top2 + composer exactly matches oracle pages + composer on the 18-case public-style suite, while using only 31.5% of the original context tokens.**

This is the strongest evidence so far for the LEKI framing:

- the latent ranker finds the same effective evidence as the oracle composer needs;
- the composer roughly halves the selected raw text budget, from 152.8 to 80.6 tokens;
- attention ranking fails badly on the same suite, especially on current-conflict and multi-hop cases.

Family-level summary:

| family | method | active tokens | NLL | exact | span recall |
|---|---|---:|---:|---:|---:|
| `needle_fact` | full KV | 256.0 | 0.6257 | 1.000 | 1.000 |
| `needle_fact` | supervised top2 + composer | 70.2 | 0.8394 | 1.000 | 1.000 |
| `multi_hop_bridge` | full KV | 256.0 | 1.0745 | 0.000 | 1.000 |
| `multi_hop_bridge` | supervised top2 + composer | 80.0 | 1.2310 | 0.333 | 1.000 |
| `current_conflict` | full KV | 256.0 | 0.3443 | 1.000 | 1.000 |
| `current_conflict` | supervised top2 + composer | 91.5 | 0.4997 | 1.000 | 1.000 |

The remaining weakness is not evidence retrieval on this suite: supervised top2 + composer has perfect span recall.
The weak family is `multi_hop_bridge`, where even full KV has exact = 0. This suggests the next quality gain should target multi-hop answer formatting/reasoning rather than page retrieval alone.

## Stronger paper framing

Recommended main method name:

**LEKI: Latent Evidence KV Indexing**

Possible subtitle:

**Evidence-preserving KV compression via learned latent page indexing and boundary-safe evidence composition.**

Main contributions should be framed as:

1. **Problem definition: evidence-preserving KV compression.**
   Show that low MSE K/V reconstruction does not guarantee rare evidence preservation.

2. **Latent evidence index.**
   Learn a compact query-conditioned page/block index from real model K/V traces. The index is trained to recover future evidence pages, not to reconstruct average K/V geometry.

3. **Boundary-safe evidence composition.**
   Retrieved page centers are expanded and semantically composed so that evidence sentences crossing page boundaries are preserved under a small token budget.

4. **Risk/budget control.**
   The existing variable-budget planner can become the runtime policy layer: choose top-k pages, halo, or fallback full KV based on uncertainty.

## Current weakness

The current evidence is promising but not yet ICML-complete:

- The public-style suite is still synthetic and short-context.
- Prompt NLL/generation quality for the public-style suite needs a GPU run.
- The composer is rule-based; this is acceptable as a first system component, but the paper will be stronger if halo/composition is learned or at least calibrated.
- Multi-hop remains the hardest family. Top2 span recall is high, but answer coverage only becomes perfect after extra halo = 3.

## Next step

Do not spend the next iteration on plain AE MSE.

The most useful next experiment is:

1. Add an **evidence-boundary coverage metric** to training/eval, not just page recall.
2. Add a small **halo/budget planner** that predicts extra halo = 1/2/3 from ranker confidence and family-like features.
3. Run GPU prompt NLL for:
   - full context,
   - attention top2 + composer,
   - supervised top2 + composer,
   - oracle pages + composer.
4. Add an ablation table:
   - attention ranking,
   - AE latent scorer,
   - supervised latent evidence ranker,
   - supervised ranker + boundary-safe composer,
   - supervised ranker + budget planner.

Current conclusion:

**The direction should continue, but the thesis must be upgraded from "compress KV with summaries" to "learn an evidence-preserving latent index and recover only the evidence needed for the query."**
