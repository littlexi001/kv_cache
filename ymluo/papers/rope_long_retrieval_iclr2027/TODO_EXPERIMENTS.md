# Experiments required for a submission-ready mechanism paper

The completed evidence supports a local, no-op-matched mechanism claim in
Qwen3-8B. It does not yet establish a universal or fully mediated explanation.

## P0 — required before submission

1. **Second-model replication.** Repeat first-layer reconstruction and BF16
   singleton closure on another RoPE family, ideally one native long-context
   checkpoint. Preserve the same-path `epsilon=0` audit.
2. **Complete mediation interventions.** Under a frozen plan, separately change
   effective phase, softmax cardinality/denominator, Value writes, residual
   states, and later pre-RoPE Query states. Report mediated effects rather than
   only correlations between adjacent links.
3. **Counterbalanced natural/public tasks.** Randomize evidence position,
   conflict value, record order, filler source, block boundary, and query
   template. Include RULER retrieval, a natural multi-hop task, and a local-order
   control.
4. **Population activation patching.** Register success→failure transitions,
   patch multiple layers, and include same-position and random-vector negative
   controls.
5. **Native-window failure boundaries.** Reproduce dense non-monotone crossings
   entirely inside a model's native evaluated window.

## P1 — strengthens identification

1. **Grouped-GQA 32K control.** Implement an exactly validated native-equivalent
   kernel that avoids materializing repeated KV heads, then rerun the unquantized
   BF16 audit. The present 32K attempt is OOM and has no result.
2. **Intervention-size curve.** Sweep small positive and negative score changes
   to quantify the Taylor-linear region and second-order error.
3. **Non-oracle probes.** Evaluate whether answer-independent centered-Value or
   gradient-proxy signals preserve the causal ranking. Do not conflate this with
   a deployable method unless selection quality and task quality both improve.
4. **Conflict taxonomy.** Separate plausible contradictory facts, same-format
   decoys, lexical competitors, and filler at matched length and token count.
5. **Cache theorem checks.** Numerically construct separable repairs that reuse
   one GQA cache and non-separable four-cycle violations that require recompute
   or multiple bases.

## P2 — optional diagnostic-system work

1. Compare exact pre-RoPE proposal with SALS, LongHeads/InfLLM-style retrieval,
   exact post-RoPE Top-2%, and Full at a matched token budget.
2. If sparse retrieval remains, replace the full pre-RoPE scan with a real
   index and report latency, throughput, memory, gather bytes, build cost, and
   update cost.
3. Treat any semantic-phase adapter as future work unless it satisfies the
   single-cache four-cycle constraint and outperforms direct prior art.

## Claims that must remain scoped

- Do not say RoPE is the unique cause of long-context failure.
- Do not say attention mass, evidence recall, or suppression certifies utility.
- Do not call the 15/16 intervention sign agreement answer accuracy.
- Do not claim PPL or free-generation gains from the singleton experiment; all
  target-versus-random NLL intervals cross zero.
- Do not call query-specific pairwise phase edits a reusable positional encoding.
- Do not claim a 32K BF16 closure result; the registered run is OOM.
- Do not call pre-RoPE proposal novel or accelerated while it scans full history.
