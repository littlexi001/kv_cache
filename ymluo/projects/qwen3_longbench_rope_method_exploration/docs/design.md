# Design: LongBench position-preserving RoPE retrieval test

## Research question

On the same frozen 18-example LongBench HotpotQA cohort, does the paper's
mechanism-guided selector improve answer quality over exact post-RoPE Top-2%
under the same 2% per-layer, per-head support?

The primary hypothesis is

$$
\mathbb E[\mathrm{NLL}_{\mathrm{LS}}-\mathrm{NLL}_{\mathrm{post2}}] < 0,
$$

where LS is local-RoPE/global-semantic selection. QA-F1, EM, evidence recall,
attention mass, and runtime are secondary outcomes.

## Frozen method

For layer $l$, Query head $h$, current position $t$, and history position $j$:

$$
s^R_{l,h}(t,j)=\frac{(R_tq_{l,h,t})^\top(R_jk_{l,g(h),j})}{\sqrt d},
\qquad
s^C_{l,h}(t,j)=\frac{q_{l,h,t}^\top k_{l,g(h),j}}{\sqrt d}.
$$

Let $B=\lceil0.02N\rceil$. The exact post-RoPE baseline keeps the $B-1$
highest $s^R$ history positions plus the current token. LS reserves 16 sink
tokens and the most recent 128 history tokens, fills the remaining support
with the highest remote $s^C$, then consumes every selected original K/V with
native $s^R$ and a sparse softmax. Retrieval scores never enter the consumer.

Inputs are the prompt, current pre/post-RoPE Q/K, native Values, and fixed
budgets. Gold answers and evidence labels are used only after selection for
measurement. No position is repacked and no cached K/V is rewritten.

## Arms

- `native_full`: untouched dense attention.
- `full_rope_replay`: mathematically dense custom path; no-op audit.
- `rope_top2`: exact post-RoPE Top-2% matched-budget baseline.
- `semantic_top2_postscore`: pre-RoPE Top-2% without local/sink structure.
- `local_global_postscore`: primary LS method.
- `local_global_blend25`: exploratory SAGE consumer that mixes 75% native
  post-RoPE score with 25% calibrated semantic score on remote slots.

Only `local_global_postscore` is the confirmatory paper method. The blend arm is
reported as exploratory because it changes the consumer score.

## Exploratory iteration: preserve the question as multiple semantic Queries

The first LongBench run can falsify the assumption that one final pre-RoPE
Query is an adequate semantic retriever. If LS fails to raise evidence recall,
we test a separate screening hypothesis: evenly sampled pre-RoPE Queries from
the visible question span score each remote token by max late interaction,
while the selected K/V are still consumed with native post-RoPE scores. This
screen is compared with the single-final-Query pre-RoPE selector and is not
promoted to the main method unless recall and answer NLL move coherently.

## Causal interpretation and boundaries

A positive LS result establishes semantic recoverability under native
positions: some useful evidence omitted by post-RoPE ranking remains selectable
from pre-RoPE content. It does not prove that RoPE caused every Full-attention
error, does not isolate a full mediated effect, and is not an acceleration
result because exact pre-RoPE scoring scans the full history.

Direct virtual-position or frequency-pair repair is not promoted: the existing
strict phase-repair smoke was slow, unstable, and equivalent to sparse logit
editing when Q, support, and V were fixed.

## Falsifiers

- Reject the quality claim if the paired 95% bootstrap CI for LS minus exact
  post-RoPE Top-2% mean gold NLL includes zero.
- Reject the retrieval explanation if answer quality improves without higher
  gold-evidence recall or attention mass on rescued cases.
- Reject implementation fidelity if dense replay differs materially from the
  untouched dense logits, any arm exceeds the 2% support, prompt hashes differ
  from the frozen Full baseline, or evidence spans cannot be aligned exactly.
- Treat QA-F1/EM changes as inconclusive if they rest on fewer than three
  paired rescue cases or if the result reverses across the two GPU shards.
