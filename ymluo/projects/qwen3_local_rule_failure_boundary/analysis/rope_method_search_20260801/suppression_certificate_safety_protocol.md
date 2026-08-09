# Suppression-certificate safety probe protocol

## Question

The proposed phase-repair mechanism uses a counterfactual certificate to find a remote key whose native post-RoPE score is lower than a local or position-free reference. This probe asks a stricter safety question:

> Does that certificate identify the **correct evidence**, or does it merely identify any semantically compatible token that RoPE happened to suppress?

The distinction matters because a phase repair that boosts both gold and plausible conflicting evidence can improve retrieval recall while making generation less reliable.

## Data

The existing fixed-300 age set has gold, different-name distractors, and filler but no same-entity plausible conflict. The existing rule-chain conflict sets explicitly expose `VERIFIED`/`UNVERIFIED`-style status cues. Neither cleanly answers this safety question, so the probe uses a new minimal age construction while retaining the project's one-hop age vocabulary and period filler.

The input is a simple English one-hop age task. It contains four token classes:

1. `gold_evidence`: `The school register lists Xiaoming's age as nine years.`
2. `conflict_evidence`: `A family note lists Xiaoming's age as four/six/eight/two years.`
3. `lexical_format_distractor`: school-register age facts about Alice, Bob, Carol, and David.
4. `filler`: repeated period tokens.

The query is:

> `According to the school register, what is Xiaoming's age?`

The rendered prompt ends in `Answer: ` (including the trailing space) and asks
for exactly one digit.  Evidence values remain English number words, but the
scored gold answer is the single no-leading-space token `9` (with the analogous
digit for the conflict answer).  This matches Qwen3-8B's natural short-context
output and explicitly owns the otherwise standalone Qwen space token in the
prompt, so PPL is measured at the answer token rather than one token too early.
The record order is shuffled by seed, so correctness is not equivalent to
appearing first. The conflict is plausible and answer-like but comes from a
different source than the source requested by the query. The model input
contains neither `VERIFIED` nor `UNVERIFIED` (in any capitalization). The source
and entity wording is task semantics, not an externally supplied truth label.

Evidence records and a small set of filler tokens are placed in the same early packet. This keeps the four diagnostic samples at nearly matched long-range distances. The remaining filler extends the total prompt to exactly 8K, 32K, or 64K tokens. Each class contributes eight deterministic diagnostic positions; decisive age tokens are always retained.

## Frozen-final-query design

The entire prefix is computed with ordinary Qwen3-8B attention and ordinary RoPE. No prefix hidden state or KV entry is rewritten. Only the final query token is instrumented. Therefore this experiment is a causal screen of final-query phase suppression, not an end-to-end retraining claim.

For a query at position $t$, a key at position $p$, and head scale $1/\sqrt d$, define:

$$
s_{\mathrm{post}}(t,p)
=
\frac{(R_t q)^\top(R_p k)}{\sqrt d}.
$$

We record four counterfactual references on the same score scale:

$$
s_{\mathrm{pre}}
=
\frac{q^\top k}{\sqrt d},
$$

$$
s_{128}
=
\frac{q^\top R_{-128}k}{\sqrt d},
$$

$$
s_{\mathrm{grid}}
=
\max_{\delta\in\{1,2,4,8,16,32,64,128\}}
\frac{q^\top R_{-\delta}k}{\sqrt d},
$$

and an independent-frequency upper envelope

$$
s_{\mathrm{upper}}
=
\frac{1}{\sqrt d}
\sum_i\sqrt{A_i^2+B_i^2}.
$$

The corresponding suppression certificates are

$$
C_x=s_x-s_{\mathrm{post}},
\qquad
x\in\{\mathrm{pre},128,\mathrm{grid},\mathrm{upper}\}.
$$

A certificate triggers when $C_x>0$ by default. `s_upper` is only a diagnostic upper bound: its independently optimal frequency phases do not generally correspond to one realizable token position. Consequently, `phase_upper_suppression` should be non-negative up to numerical error for nearly every token; its trigger rate is a reconstruction sanity check, not a useful gate. Its magnitude and AUROC remain informative.

## Metrics

For every layer, attention head, sampled token, length, and seed, the runner saves:

- native post-RoPE QK score;
- pre-RoPE, fixed-local-anchor, local-grid-envelope, and phase-upper scores;
- all four suppression certificates;
- native attention probability;
- best local anchor distance;
- trigger indicator and RoPE reconstruction error.

It reports certificate distributions and trigger rates for all four classes.
Tie-aware AUROC is computed for both the four suppression certificates and the
five raw scores.  The raw `grid_envelope_score` is especially important: it
tests whether a realizable local-phase envelope could serve as a remote
retrieval score, rather than merely as a repair trigger.  The comparisons are:

- gold versus conflict;
- gold versus lexical/format distractors;
- gold versus filler;
- gold versus all non-gold tokens;
- semantic evidence (`gold + conflict`) versus non-semantic tokens (`lexical + filler`).

Each comparison has two explicit scopes: `all_sampled` and `decisive_only`.
The latter compares age-answer tokens such as `nine` and the conflicting age
word.  If a class has no decisive token (period filler), that decisive-only
comparison is reported as `NA`, never silently replaced by the all-token
estimate.

The critical interpretation is asymmetric. High semantic-evidence AUROC together with gold-versus-conflict AUROC near 0.5 means that the certificate recognizes suppressed semantic compatibility but is **not** a truth certificate.

### Seed-level uncertainty

Pooled head/token rows are correlated and are not treated as independent observations for uncertainty. For each AUROC task, the runner first computes one AUROC for every `seed × length`; the reported `auroc` is the mean of these seed-level values, while `pooled_auroc` is retained only as a descriptive diagnostic. The report then uses a cluster bootstrap whose sampling unit is the complete seed:

- for a fixed context length, resample seeds with replacement and retain every layer/head/token row belonging to each sampled seed;
- for the `all`-length report, draw each seed once per replicate and carry that seed's complete 8K/32K/64K trajectory together, preserving the paired length design;
- use 2,000 deterministic replicates and percentile 95% intervals by default;
- report intervals for mean seed-level AUROC and, for every intervention class, mean `delta_gold_nll` and mean `delta_gold_conflict_margin`;
- require at least four distinct seeds in every included length stratum.

If a length has fewer than four seeds, or if the all-length seed-by-length grid is unbalanced, bootstrapping is disabled for that statistic. Both interval endpoints are written explicitly as `NA`, together with a machine-readable reason such as `NA:insufficient_seeds[32768:3]<minimum_4` or `NA:unbalanced_seed_length_grid`. A pooled point estimate may still be shown, but it must not be presented as having measured seed-level uncertainty.

## Matched causal intervention

Every case first runs an untouched native-attention final query.  A second,
explicit-QK instrumented baseline constructs the certificates and freezes, for
every layer and head, the top local-grid position and its best local anchor for
each class.  The instrumented-minus-native PPL and logit-margin drift is a
mandatory validity measurement: intervention effects are not interpreted when
the measurement path itself materially changes the decision. Four fresh
instrumented final-query passes then repair one token per head per layer for
exactly one class at a time:

$$
K_p^{\mathrm{post}}
\longrightarrow
R_{(t-\delta^*)-p}K_p^{\mathrm{post}},
$$

where $\delta^*$ is the frozen best local anchor. Position, anchor, and token
count are frozen from the instrumented reference trajectory. Later-layer scores
are recomputed under the intervened hidden state, but the intervention plan
itself is not reselected. This avoids giving any class extra slots or adaptive
search.

For each class intervention, report:

- change in gold NLL and PPL (plus PPL ratio);
- change in gold-versus-strongest-token margin;
- change in gold-versus-conflict-answer margin;
- next-token accuracy;
- explicit-QK instrumentation drift from the native baseline;
- intervention score lift and the fraction of selected positions with a positive baseline certificate.

Expected safety signature:

- repairing gold evidence should lower gold NLL and increase margins;
- repairing conflict evidence should reduce the gold-versus-conflict margin;
- lexical/format and filler repairs should have smaller effects;
- if conflict repair is as strong as gold repair, an unconditional suppression-triggered consumer is unsafe and needs query/source consistency gating.

## Default run (not launched by this change)

The launcher is:

`scripts/run_suppression_certificate_safety_gpu67_20260801.sh`

It is hard-limited to physical GPUs 6 and 7:

- GPU 6: seeds 0--3;
- GPU 7: seeds 4--7;
- lengths: 8,192, 32,768, and 65,536 tokens;
- Qwen3-8B, NF4 weights, BF16 compute, eager attention for the causal audit;
- one shared 64K-capable YaRN configuration (factor 2) for all three lengths,
  so length comparisons do not silently reload different RoPE geometries;
- eight diagnostic tokens per class;
- one matched intervention token per layer/head;
- local anchors: 1, 2, 4, 8, 16, 32, 64, 128;
- prefix chunk size: 64 (required for 64K eager-attention peak memory on 24 GiB GPUs).
- seed-cluster bootstrap: 2,000 replicates, fixed RNG seed, minimum four seeds per length.

Each seed/length performs one standard prefix prefill, one native final query,
one instrumented reference query, and four intervention final queries. Prefill
dominates runtime. An existing Qwen3-8B 64K run on the same server recorded
roughly 107 seconds of prefill and about 0.2--0.3 seconds per final-query
variant. This probe evaluates only small balanced certificate samples, so the
expected wall time is approximately 20--45 minutes per four-seed shard, with
both shards running concurrently. This remains an estimate; the first
completed 8K and 64K cases should be used to update it.

Outputs are written under:

`outputs/20260801_suppression_certificate_safety_gpu67_v2/`

Each shard has resumable per-case raw files. After both GPU processes finish, the launcher performs a CPU-only merge and writes certificate distributions, AUROCs, intervention summaries, and a combined JSON summary.

## Validity limits

- A synthetic one-hop result does not establish open-domain factual truth detection.
- Heads and sampled tokens within a seed are correlated. `pooled_auroc` remains descriptive; the primary `auroc` and its confidence interval are computed from seed-level AUROCs with the cluster bootstrap above.
- The local grid searches only eight realizable distances; the phase upper envelope is not itself a valid intervention.
- This probe diagnoses the consumer's safety. It does not yet train a gate that separates gold from conflict.
- Because only the final query is changed, conclusions do not automatically transfer to a model trained end to end with a modified positional encoding.
