# Method audit: phase repair, logit bias, and identifiability

## 1. Final-query phase repair is exactly a score bias

For one attention head and one query, native RoPE attention is

\[
s_j=\frac{q^\top R(\Delta_j)k_j}{\sqrt d},
\qquad
a_j=\frac{e^{s_j}}{\sum_r e^{s_r}},
\qquad
o=\sum_j a_jv_j.
\]

Suppose a query-specific repair changes only the phase used by interaction \(j\):

\[
s'_j=\frac{q^\top R(\Delta_j+\delta_j)k_j}{\sqrt d}.
\]

Define

\[
b_j=s'_j-s_j.
\]

Then

\[
o'
=\sum_j\operatorname{softmax}(s')_jv_j
=\sum_j\operatorname{softmax}(s+b)_jv_j.
\]

This is an exact identity, not an approximation. If the intervention is computed separately for the current query, changes no Value, and is not reused by later queries, then “phase repair” and “adding the same per-interaction attention-logit bias” are functionally indistinguishable.

Consequences:

- a final-query MPR gain can establish that a particular score change is useful;
- it cannot by itself establish a new positional encoding;
- a matched additive-score control must reproduce the output elementwise;
- to support a new-PE claim, the correction must define a reusable geometry: for example a repaired Key/position state shared by future queries, or a train-time rule that is fixed before seeing the current answer query.

Qwen3-8B adds a stronger realizability constraint. It has 32 Query heads but only 8 KV heads, so four Query heads share each cached Key. The current MPR solver is allowed to choose a different frequency-plane shift for every `(query head, token)` interaction. One physical cached Key cannot simultaneously realize four different rotations. Thus the current intervention is not merely *similar* to a score edit: under GQA it generally cannot be represented as one reusable repaired KV cache at all. A realizable positional method must tie the phase rule across all Query heads sharing a KV head, freeze it before future queries, and then pass a multi-token reuse test.

## 2. Suppression alone cannot identify truth

The suppression certificate

\[
c_j=s_j^{\mathrm{pre}}-s_j^{\mathrm{post}}
\]

measures how much native RoPE lowers one Query--Key interaction. It contains no explicit statement of whether the token is true, false, useful, or harmful. Therefore a rule of the form

\[
c_j>\tau \Longrightarrow \text{repair }j
\]

can amplify gold evidence, conflicting evidence, formatting tokens, and filler whenever they happen to have the same phase geometry.

The 64K seed-0 smoke supports this limitation: gold-vs-conflict and gold-vs-all-nongold AUROC are both approximately 0.50, and repairing different classes does not preferentially help gold. The formal multi-seed run is required before treating this as established.

## 3. What remains scientifically meaningful

Three claims remain separable:

1. **Mechanism:** native RoPE can suppress semantically relevant remote pairs.
2. **Selection:** pre-RoPE or another semantic channel can recover candidates missed by post-RoPE ranking.
3. **Safe consumption:** the model needs a deployable signal that determines which recovered Values should influence the residual stream.

The first two already have positive evidence in the current experiments. The third is the unresolved paper-level problem. A credible method must use a label-free content/relation signal, not only distance or phase suppression, and must beat matched random and direct-score-bias controls.

## 4. A scoped realizability theorem for reusable RoPE repair

The previous score-bias identity concerns one fixed Query. A stronger constraint follows when a proposed PE must produce one prefix-frozen cache that can be reused by many future Queries.

Assume that, for every prefix token (i), KV group (g), and rotary plane (r), the cache stores one norm-preserving transform

\[
k^*_{g,i,r}=T_{g,i,r}k_{g,i,r},
\]

before the future Query is known. A future Query head (h\in\mathcal H_g) may apply a transform (U_{h,t,r}) that is independent of candidate (i). If the method preserves the standard two-dimensional rotary planes, every orientation-preserving norm-preserving transform is a rotation, so write

\[
T_{g,i,r}=R(\beta_{g,i,r}),
\qquad
U_{h,t,r}=R(\alpha_{h,t,r}).
\]

The resulting interaction is

\[
(U_{h,t,r}q_{h,t,r})^\top(T_{g,i,r}k_{g,i,r})
=
q_{h,t,r}^\top
R(\beta_{g,i,r}-\alpha_{h,t,r})
k_{g,i,r}.
\]

Therefore a target pairwise phase tensor (\Delta_{q,i,r}) can be represented by one reusable cache only if it is additively separable:

\[
\boxed{
\Delta_{q,i,r}
=
\alpha_{q,r}-\beta_{g,i,r}
\pmod{2\pi}
}
\]

up to the chosen sign convention. The Key-side factor (\beta_{g,i,r}) must be shared by all Query heads in the same GQA group.

An equivalent observable condition is cycle consistency. For every two Queries (q_1,q_2) and Keys (i_1,i_2), a reusable repair must satisfy

\[
\boxed{
\Delta_{q_1,i_1,r}
+\Delta_{q_2,i_2,r}
-\Delta_{q_1,i_2,r}
-\Delta_{q_2,i_1,r}
=0
\pmod{2\pi}.
}
\]

Necessity follows by substituting the separable form and cancelling all Query- and Key-owned terms. Sufficiency follows by fixing a reference Query and Key and constructing (\alpha) and (\beta) from the corresponding row and column. Consequently, an independently optimized phase for every `(Query, head, token)` generally cannot be materialized as one cached Key; it is a pairwise router or score edit.

This is deliberately a scoped theorem. It applies to a frozen Qwen-style model with one GQA cache, prefix-frozen multi-Query reuse, RoPE-compatible two-dimensional norm-preserving planes, and no pairwise cache rewrite. It does not claim that every nonlinear neural geometry is impossible. Allowing a general linear (T=OP) separates a rotational part (O) from a positive content metric (P); the latter is no longer merely a positional encoding and, without joint training, has no reason to preserve the pretrained model's semantics.

Under the stated contract, the remaining separable designs fall into already populated families: static/chunk or semantic position remapping, KV-group/head/frequency scaling, remote NoPE/partial-RoPE, multiple cached phase bases, or a prefix Writer that compiles semantic Keys. This is why the current experiment should treat reusable new PE as a NO-GO rather than continue tuning per-interaction phase patches.
